import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from wm import assisted, pipeline
from wm.app import MainWindow, make_app
from wm.batches import Batch
from wm.contract import (PageImage, PageResult, MaskLayer, VerdictReport,
                         EngineKind, RepairName, Thresholds)
from wm.viewer import ImageViewer


def baseline():
    image = np.full((120, 160, 3), 230, np.uint8)
    image[35:65, 45:85] = [70, 180, 120]
    page = PageImage(image, 72, '辅助测试')
    report = VerdictReport(page.key, EngineKind.RASTER, [],
                           {'recognition': True, 'residual': True}, {}, [], 1,
                           has_ground_truth=True)
    return PageResult(page, replace(page, image=image.copy()),
                      MaskLayer(np.zeros(image.shape[:2], np.float32)), report,
                      processing={'thresholds': {'mask_alpha_min': .08}})


def proposal(result):
    selected = np.zeros(result.output.image.shape[:2], bool)
    selected[40:60, 50:80] = True
    selected[44:56, 55:75] = False  # preserve a glyph hole
    layer = MaskLayer(selected.astype(np.float32), footprint=selected,
                      repair_mode='inpaint', confidence=.8)
    return assisted.StrokeProposal((30, 20, 100, 80), assisted.image_hash(result.output.image),
                                    layer, Thresholds(), 1, 'test_observed_strokes',
                                    sha256(selected.tobytes()).hexdigest())


def fake_repair(image, layer, thr):
    out = image.copy()
    out[layer.footprint] = 220
    return out, [RepairName.LAMA]


class AssistedBoundaryTests(unittest.TestCase):
    def test_uniform_region_cannot_become_a_rectangle_from_ocr_or_detector(self):
        result = baseline()
        result.output.image[:] = 240
        def falsely_detect(crop, *args, **kwargs):
            return MaskLayer(np.ones(crop.shape[:2], np.float32), repair_mode='inpaint')
        with patch.object(assisted.masks, 'estimate_single_page', side_effect=falsely_detect):
            p = assisted.find_strokes(result, (20, 20, 120, 100), Thresholds())
        self.assertFalse(p.layer.footprint.any())
        with self.assertRaisesRegex(ValueError, '未找到'):
            assisted.apply_proposal(result, p)

    def test_actual_pixel_contrast_keeps_hole_and_search_boundary(self):
        result = baseline()
        image = result.output.image
        image[:] = 230
        cv2.circle(image, (70, 55), 18, (70, 180, 120), 4)
        seed = np.zeros(image.shape[:2], np.float32)
        cv2.circle(seed, (70, 55), 18, 1., 4)
        with patch.object(assisted.masks, 'estimate_single_page', return_value=MaskLayer(seed, repair_mode='inpaint')):
            p = assisted.find_strokes(result, (20, 20, 110, 100), Thresholds(lama_mask_dilate=8))
        self.assertGreater(p.layer.footprint.sum(), 20)
        self.assertFalse(p.layer.footprint[55, 70])
        self.assertFalse(p.layer.footprint[:, 110:].any())
        self.assertEqual(p.thresholds.lama_mask_dilate, 0)

    def test_confirm_changes_only_preview_strokes_and_keeps_automatic_baseline(self):
        result = baseline(); p = proposal(result); before = result.output.image.copy()
        with patch.object(assisted.repair, 'repair', side_effect=fake_repair):
            out = assisted.apply_proposal(result, p)
        np.testing.assert_array_equal(result.output.image, before)
        np.testing.assert_array_equal(out.output.image[~p.layer.footprint], before[~p.layer.footprint])
        self.assertIs(out.automatic_result, result)
        self.assertTrue(out.verdict.review_needed)
        self.assertEqual(out.processing['mode'], 'human_assisted')
        self.assertNotIn('human_assistance', result.processing)

    def test_outside_change_is_rejected_without_repairing_or_mutating_base(self):
        result = baseline(); p = proposal(result); before = result.output.image.copy()
        with patch.object(assisted.repair, 'repair', return_value=(np.zeros_like(before), [])):
            with self.assertRaisesRegex(ValueError, '超出预览'):
                assisted.apply_proposal(result, p)
        np.testing.assert_array_equal(result.output.image, before)

    def test_stale_image_and_changed_preview_are_rejected(self):
        result = baseline(); p = proposal(result)
        result.output.image[0, 0] = 0
        with self.assertRaisesRegex(ValueError, '图片已变化'):
            assisted.apply_proposal(result, p)
        result = baseline(); p = proposal(result); p.layer.alpha[50, 60] = 1
        with self.assertRaisesRegex(ValueError, '候选笔画已变化'):
            assisted.apply_proposal(result, p)

    def test_export_keeps_automatic_and_assisted_metrics_and_every_confirmed_mask(self):
        result = baseline()
        with patch.object(assisted.repair, 'repair', side_effect=fake_repair):
            out = assisted.apply_proposal(result, proposal(result))
            out = assisted.apply_proposal(out, proposal(out))
        self.assertIs(out.automatic_result, result)
        with tempfile.TemporaryDirectory() as directory:
            root = pipeline.write_results([out], directory)
            data = json.loads((root/'manifest.json').read_text())
            self.assertEqual(data['summary']['automatic']['success_rate'], 1.)
            self.assertIsNone(data['summary']['human_assisted']['success_rate'])
            page = data['pages'][0]
            self.assertEqual(page['processing_mode'], 'human_assisted')
            for path in page['automatic_baseline']['files'].values():self.assertTrue((root/path).is_file())
            np.testing.assert_array_equal(np.array(Image.open(root/page['automatic_baseline']['files']['clean'])), result.output.image)
            self.assertEqual(len(page['assistance_steps']), 2)
            for step in page['assistance_steps']:
                self.assertTrue((root/step['mask']).is_file())
            self.assertIn('不计为全自动处理效果', (root/'水印选区与处理结果复核.html').read_text())


class AssistedDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app = make_app()

    def setUp(self):
        self.win = MainWindow(); self.base = baseline()
        batch = Batch('测试', [self.base.source, self.base.source])
        batch.results[0] = self.base; batch.states[0] = '通过'
        self.win.batches = [batch]; self.win.refresh_batches(); self.win.batch_list.setCurrentRow(0)

    def tearDown(self):
        self.win.batch.dirty.clear(); self.win.close()

    def wait_job(self):
        end = time.monotonic()+5
        while self.win._assist_busy and time.monotonic() < end:
            self.app.processEvents(); QTest.qWait(10)
        self.assertFalse(self.win._assist_busy)

    def test_preview_cancel_confirm_and_undo_use_separate_transitions(self):
        w = self.win; candidate = proposal(self.base)
        with patch.object(assisted, 'find_strokes', return_value=candidate):
            w.assist_select_btn.click(); w.find_assisted_strokes(candidate.rectangle); self.wait_job()
        self.assertIs(w.batch.results[0], self.base)
        self.assertFalse(w.batch.dirty)
        self.assertTrue(w.assist_confirm_btn.isEnabled())
        self.assertIsNotNone(w._assist_pending)
        w.assist_cancel_btn.click()
        self.assertIsNone(w._assist_pending)
        self.assertIs(w.batch.results[0], self.base)
        with patch.object(assisted, 'find_strokes', return_value=candidate):
            w.assist_select_btn.click(); w.find_assisted_strokes(candidate.rectangle); self.wait_job()
        with patch.object(assisted.repair, 'repair', side_effect=fake_repair):
            w.assist_confirm_btn.click(); self.wait_job()
        self.assertEqual(w.batch.states[0], '辅助修复·待复核')
        self.assertTrue(w.assist_undo_btn.isEnabled())
        w.assist_undo_btn.click()
        self.assertIs(w.batch.results[0], self.base)
        self.assertEqual(w.batch.states[0], '通过')
        self.assertFalse(w.batch.dirty)

    def test_switch_page_invalidates_pending_proposal(self):
        self.win._assist_pending = proposal(self.base)
        self.win.page_list.setCurrentRow(1)
        self.assertIsNone(self.win._assist_pending)
        self.assertFalse(self.win.assist_confirm_btn.isEnabled())

    def test_parameter_change_clears_both_candidate_and_red_preview(self):
        self.win._assist_pending = proposal(self.base)
        self.win.render_canvas()
        slider = self.win._sliders['mask_alpha_min']
        slider.setValue(slider.value()+1)
        self.assertIsNone(self.win._assist_pending)
        self.assertFalse(self.win.assist_confirm_btn.isEnabled())
        self.assertEqual(self.win.canvas.before.toImage().pixelColor(50, 40).getRgb()[:3],
                         tuple(self.base.output.image[40, 50]))

    def test_reverse_drag_and_letterbox_mapping_survive_resize(self):
        viewer = ImageViewer(); viewer.resize(800, 500)
        viewer.set_images(self.base.output.image, page_key='test'); viewer.show()
        self.app.processEvents(); viewer.set_selection_mode(True)
        captured = []; viewer.rectangleFinished.connect(captured.append)
        rect = viewer.image_rect()
        start = QPoint(round(rect.left()+rect.width()*.8), round(rect.top()+rect.height()*.8))
        end = QPoint(round(rect.left()+rect.width()*.2), round(rect.top()+rect.height()*.2))
        QTest.mousePress(viewer, Qt.LeftButton, pos=start)
        QTest.mouseMove(viewer, end); QTest.mouseRelease(viewer, Qt.LeftButton, pos=end)
        box = captured[0]
        for actual, expected in zip(box, (32, 24, 128, 96)):self.assertAlmostEqual(actual, expected, delta=1)
        viewer.resize(400, 800); self.app.processEvents()
        self.assertEqual(viewer.search_rectangle, box)
        viewer.set_images(self.base.output.image, page_key='other')
        self.assertIsNone(viewer.search_rectangle)
        viewer.close()
