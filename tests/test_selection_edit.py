import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QPoint, Qt
from PySide6.QtTest import QTest

from wm import pipeline, masks
from wm.app import MainWindow, make_app
from wm.batches import Batch
from wm.contract import PageImage, PageResult, MaskLayer, VerdictReport, EngineKind, RepairName, Thresholds
from wm.selection_edit import SelectionEdit, apply_selection
from wm.viewer import ImageViewer


def example():
    image = np.full((120, 160, 3), 230, np.uint8)
    image[30:70, 30:80] = [180, 190, 200]
    page = PageImage(image, 100, '画笔测试')
    alpha = np.zeros(image.shape[:2], np.float32)
    alpha[30:70, 30:80] = .4
    alpha[42:58, 45:65] = 0  # actual glyph hole
    selected = alpha >= .08
    layer = MaskLayer(alpha, confidence=.8, footprint=selected, repair_mode='inpaint')
    report = VerdictReport(page.key, EngineKind.RASTER, [], {}, {}, [], 1)
    out = image.copy(); out[selected] = 100
    return PageResult(page, replace(page, image=out), layer, report,
                      processing={'thresholds': {'mask_alpha_min': .08}})


def stroke(edit, start, end=None, diameter=5, include=True):
    edit.begin_stroke()
    edit.paint(start, end or start, diameter, include)
    edit.end_stroke()


def deterministic_repair(image, layer, thresholds):
    out = image.copy()
    out[layer.footprint] = np.maximum(0, image[layer.footprint].astype(int)-20).astype(np.uint8)
    return out, [RepairName.LAMA]


class SelectionRepairTests(unittest.TestCase):
    def test_last_stroke_wins_holes_and_per_stroke_undo_redo(self):
        edit = SelectionEdit(example())
        original = edit.selected.copy()
        stroke(edit, (50.5, 50.5), diameter=7)
        self.assertTrue(edit.selected[50, 50])
        stroke(edit, (50.5, 50.5), diameter=7, include=False)
        self.assertFalse(edit.selected[50, 50])
        stroke(edit, (50.5, 50.5), diameter=3)
        self.assertTrue(edit.selected[50, 50])
        edit.undo(); self.assertFalse(edit.selected[50, 50])
        edit.redo(); self.assertTrue(edit.selected[50, 50])
        edit.undo(); edit.undo(); edit.undo()
        np.testing.assert_array_equal(edit.selected, original)
        self.assertFalse(edit.dirty)
        edit.redo()
        stroke(edit, (100.5, 80.5))
        self.assertFalse(edit.redo_stack)

    def test_long_continuous_stroke_and_original_pixel_diameter(self):
        edit = SelectionEdit(example())
        stroke(edit, (100.5, 10.5), (100.5, 100.5), diameter=5)
        self.assertTrue(edit.selected[10:101, 98:103].all())
        self.assertFalse(edit.selected[10:101, 97].any())
        self.assertFalse(edit.selected[10:101, 103].any())

    def test_repair_uses_source_not_last_output_and_never_detects_again(self):
        result = example(); edit = SelectionEdit(result)
        stroke(edit, (32.5, 40.5), diameter=5, include=False)
        stroke(edit, (100.5, 80.5), diameter=7)
        request = edit.request(Thresholds(lama_mask_dilate=9))
        calls = []
        def repair(image, layer, settings):
            np.testing.assert_array_equal(image, result.source.image)
            self.assertEqual(settings.lama_mask_dilate, 0)
            calls.append(layer)
            return deterministic_repair(image, layer, settings)
        with patch('wm.selection_edit.repair.repair', side_effect=repair), patch.object(
                masks, 'estimate_single_page', side_effect=AssertionError('must not detect')):
            applied = apply_selection(result, request)
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(applied.output.image[~edit.selected], result.source.image[~edit.selected])
        self.assertTrue(applied.mask.binary()[80, 100])
        self.assertEqual(applied.mask.alpha[80, 100], 0)
        self.assertFalse(applied.mask.binary()[40, 32])
        self.assertIs(applied.automatic_result, result)
        self.assertTrue(applied.verdict.review_needed)
        self.assertNotIn('residual', applied.verdict.conditions)
        self.assertTrue(np.isnan(applied.verdict.metrics['residual_coverage']))
        again = SelectionEdit(applied)
        stroke(again, (100.5, 80.5), diameter=7, include=False)
        stroke(again, (100.5, 80.5), diameter=7)
        with patch('wm.selection_edit.repair.repair', side_effect=deterministic_repair):
            repeated = apply_selection(applied, again.request(Thresholds()))
        np.testing.assert_array_equal(repeated.output.image, applied.output.image)

    def test_retained_automatic_route_keeps_measured_alpha_and_excludes_erased_pixels(self):
        result = example()
        result.mask.repair_mode = 'document_overlay_full'
        result.mask.veil_strength = np.array([.7, .7, .7], np.float32)
        result.mask.veil_offset = np.zeros(3, np.float32)
        result.mask.paper = np.full(result.source.image.shape, 255, np.float32)
        edit = SelectionEdit(result)
        stroke(edit, (32.5, 40.5), diameter=5, include=False)
        def check(image, layer, settings):
            self.assertEqual(layer.repair_mode, 'document_overlay_full')
            np.testing.assert_array_equal(layer.veil_strength, result.mask.veil_strength)
            np.testing.assert_array_equal(layer.alpha[edit.selected], result.mask.alpha[edit.selected])
            self.assertFalse(layer.alpha[~edit.selected].any())
            return deterministic_repair(image, layer, settings)
        with patch('wm.selection_edit.repair.repair', side_effect=check):
            apply_selection(result, edit.request(Thresholds()))

    def test_empty_selection_restores_original_and_does_not_call_model(self):
        result = example(); edit = SelectionEdit(result)
        stroke(edit, (80, 60), diameter=300, include=False)
        with patch('wm.selection_edit.repair.repair', side_effect=AssertionError('empty')):
            applied = apply_selection(result, edit.request(Thresholds()))
        np.testing.assert_array_equal(applied.output.image, result.source.image)
        self.assertFalse(applied.mask.binary().any())

    def test_manual_selection_on_pdf_has_explicit_raster_route(self):
        result = example(); result.mask.repair_mode = 'structure'
        edit = SelectionEdit(result); stroke(edit, (100, 80))
        def check(image, layer, settings):
            self.assertEqual(layer.repair_mode, 'inpaint')
            np.testing.assert_array_equal(image, result.source.image)
            return deterministic_repair(image, layer, settings)
        with patch('wm.selection_edit.repair.repair', side_effect=check):
            applied = apply_selection(result, edit.request(Thresholds()))
        self.assertEqual(applied.processing['human_assistance'][0]['retained_automatic_route'], 'structure')

    def test_stale_or_mutated_request_and_outside_writes_are_rejected(self):
        result = example(); edit = SelectionEdit(result); stroke(edit, (100, 80))
        request = edit.request(Thresholds())
        request.selected[0, 0] = True
        with self.assertRaisesRegex(ValueError, '选区已变化'):
            apply_selection(result, request)
        request = edit.request(Thresholds())
        result.source.image[0, 0] = 0
        with self.assertRaisesRegex(ValueError, '原图或处理结果已变化'):
            apply_selection(result, request)
        result = example(); edit = SelectionEdit(result); stroke(edit, (100, 80))
        with patch('wm.selection_edit.repair.repair', return_value=(np.zeros_like(result.source.image), [])):
            with self.assertRaisesRegex(ValueError, '超出确认选区'):
                apply_selection(result, edit.request(Thresholds()))

    def test_export_contains_complete_selection_and_not_invented_alpha(self):
        result = example(); edit = SelectionEdit(result)
        stroke(edit, (100.5, 80.5), diameter=7)
        stroke(edit, (32.5, 40.5), include=False)
        with patch('wm.selection_edit.repair.repair', side_effect=deterministic_repair):
            applied = apply_selection(result, edit.request(Thresholds()))
        with tempfile.TemporaryDirectory() as folder:
            root = pipeline.write_results([applied], folder)
            record = json.loads((root/'manifest.json').read_text())['pages'][0]
            binary = np.array(Image.open(root/record['files']['mask'])) > 0
            np.testing.assert_array_equal(binary, edit.selected)
            with np.load(root/record['files']['mask_data'], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays['selection'], edit.selected)
                np.testing.assert_array_equal(arrays['selection_overrides'], edit.overrides)
                self.assertEqual(arrays['alpha'][80, 100], 0)
            self.assertIn('actual_changes', record['files'])
            np.testing.assert_array_equal(np.array(Image.open(root/record['automatic_baseline']['files']['clean'])),
                                          result.output.image)
            self.assertIsNone(record['verdict']['metrics']['residual_coverage'])


class SelectionDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = make_app()

    def setUp(self):
        self.win = MainWindow(); self.result = example()
        batch = Batch('画笔交互', [self.result.source, self.result.source])
        batch.results[0] = self.result
        batch.results[1] = self.result
        self.win.batches = [batch]
        self.win.refresh_batches(); self.win.batch_list.setCurrentRow(0)
        self.win.show(); self.app.processEvents()

    def tearDown(self):
        self.win.batch.dirty.clear(); self.win.close()

    def paint(self, include=True, point=(100.5, 80.5)):
        w = self.win
        w.set_brush_tool('add' if include else 'erase')
        rect = w.canvas.image_rect()
        x = rect.left()+point[0]*rect.width()/160
        y = rect.top()+point[1]*rect.height()/120
        pos = QPoint(round(x), round(y))
        QTest.mousePress(w.canvas, Qt.LeftButton, pos=pos)
        QTest.mouseRelease(w.canvas, Qt.LeftButton, pos=pos)

    def wait_job(self):
        until = time.monotonic()+5
        while self.win._assist_busy and time.monotonic() < until:
            self.app.processEvents(); QTest.qWait(10)
        self.assertFalse(self.win._assist_busy)

    def test_views_share_full_selection_but_switching_page_discards_unprocessed_strokes(self):
        w = self.win; w.assist_select_btn.click()
        self.assertEqual(w._view_mode, '原始图片')
        self.paint()
        edit = w._selection_edit
        self.assertTrue(edit.selected[80, 100])
        self.assertIs(w.batch.results[0], self.result)
        self.assertFalse(w.export_btn.isEnabled())
        for mode in ['处理结果', '拖拽对比', '水印选区', '原始图片']:
            w.set_view_mode(mode)
            self.assertIs(w._selection_edit, edit)
            self.assertTrue(edit.selected[80, 100])
        w.set_view_mode('水印选区')
        overlay = w.canvas._overlay.toImage()
        self.assertGreater(overlay.pixelColor(35, 35).alpha(), 0)
        self.assertGreater(overlay.pixelColor(100, 80).alpha(), 0)
        self.assertEqual(overlay.pixelColor(55, 50).alpha(), 0)
        w.page_list.setCurrentRow(1)
        self.assertIsNone(w._selection_edit)
        w.page_list.setCurrentRow(0); w.assist_select_btn.click()
        self.assertFalse(w._selection_edit.selected[80, 100])

    def test_brush_undo_redo_and_zoom_do_not_change_pixel_diameter(self):
        w = self.win; w.assist_select_btn.click()
        w.brush_size.setValue(5)
        self.paint()
        selected = w._selection_edit.selected.copy()
        w.stroke_undo_btn.click(); self.assertFalse(w._selection_edit.selected[80, 100])
        w.stroke_redo_btn.click(); self.assertTrue(w._selection_edit.selected[80, 100])
        w.stroke_undo_btn.click()
        rect = w.canvas.image_rect(); anchor = QPointF(rect.left()+100.5*rect.width()/160,
                                                      rect.top()+80.5*rect.height()/120)
        before = w.canvas._image_point(anchor)
        w.canvas.zoom_at(anchor, 2)
        np.testing.assert_allclose(w.canvas._image_point(anchor), before, atol=1e-6)
        self.assertEqual(w.canvas.brush_diameter, 5)
        self.paint()
        np.testing.assert_array_equal(w._selection_edit.selected, selected)

    def test_space_pan_does_not_add_a_stroke_and_mask_hide_is_temporary(self):
        w = self.win; w.assist_select_btn.click()
        before = w._selection_edit.selected.copy()
        center = w.canvas.image_rect().center().toPoint()
        QTest.keyPress(w.canvas, Qt.Key_Space)
        QTest.mousePress(w.canvas, Qt.LeftButton, pos=center)
        QTest.mouseMove(w.canvas, center+QPoint(20, 10))
        QTest.mouseRelease(w.canvas, Qt.LeftButton, pos=center+QPoint(20, 10))
        QTest.keyRelease(w.canvas, Qt.Key_Space)
        np.testing.assert_array_equal(before, w._selection_edit.selected)
        QTest.keyPress(w.canvas, Qt.Key_H); self.assertTrue(w.canvas._hide_overlay)
        QTest.keyRelease(w.canvas, Qt.Key_H); self.assertFalse(w.canvas._hide_overlay)

    def test_processing_commits_full_selection_and_undo_restores_prior_result(self):
        w = self.win; w.assist_select_btn.click(); self.paint()
        wanted = w._selection_edit.selected.copy()
        with patch('wm.selection_edit.repair.repair', side_effect=deterministic_repair):
            w.assist_confirm_btn.click(); self.wait_job()
        self.assertIsNone(w._selection_edit)
        np.testing.assert_array_equal(w.batch.results[0].mask.binary() > 0, wanted)
        w.page_list.setCurrentRow(1); w.page_list.setCurrentRow(0)
        w.set_view_mode('水印选区')
        self.assertGreater(w.canvas._overlay.toImage().pixelColor(100, 80).alpha(), 0)
        w.assist_undo_btn.click()
        self.assertIs(w.batch.results[0], self.result)

    def test_failed_processing_preserves_both_edit_and_previous_result(self):
        w = self.win; w.assist_select_btn.click(); self.paint()
        edit = w._selection_edit
        with patch('wm.selection_edit.repair.repair', side_effect=ValueError('测试失败')):
            w.assist_confirm_btn.click(); self.wait_job()
        self.assertIs(w._selection_edit, edit)
        self.assertIs(w.batch.results[0], self.result)
        self.assertTrue(w.assist_confirm_btn.isEnabled())
        self.assertIn('测试失败', w.status.text())

    def test_reprocessing_committed_selection_never_runs_automatic_detection(self):
        w = self.win; w.assist_select_btn.click(); self.paint()
        with patch('wm.selection_edit.repair.repair', side_effect=deterministic_repair):
            w.assist_confirm_btn.click(); self.wait_job()
        committed = w.batch.results[0]
        self.assertIn('已确认选区', w.reprocess_btn.text())
        self.assertTrue(w.auto_redetect_btn.isVisible())
        with patch('wm.selection_edit.repair.repair', side_effect=deterministic_repair), patch.object(
                masks, 'estimate_single_page', side_effect=AssertionError('must preserve manual selection')):
            w.reprocess_btn.click(); self.wait_job()
        np.testing.assert_array_equal(w.batch.results[0].output.image, committed.output.image)
        np.testing.assert_array_equal(w.batch.results[0].mask.selection, committed.mask.selection)

    def test_brush_motion_far_outside_image_does_not_paint_a_border(self):
        w = self.win; w.assist_select_btn.click()
        w.brush_size.setValue(3)
        rect = w.canvas.image_rect()
        def position(x, y):
            return QPointF(rect.left()+x*rect.width()/160, rect.top()+y*rect.height()/120)
        w.canvas._last_brush = None
        w._begin_brush_stroke()
        for x,y in [(10,10),(-100,-100),(150,-100)]:
            w.canvas._paint_segment(position(x,y))
        w._end_brush_stroke()
        self.assertFalse(w._selection_edit.selected[0, 80])

    def test_close_has_no_draft_prompt_or_saved_result(self):
        w = self.win; w.assist_select_btn.click(); self.paint()
        with patch('wm.app.QMessageBox.question', side_effect=AssertionError('must not ask about a draft')):
            w.close()
        self.assertIs(w.batch.results[0], self.result)
        self.assertFalse(w.batch.dirty)


if __name__ == '__main__':
    unittest.main()
