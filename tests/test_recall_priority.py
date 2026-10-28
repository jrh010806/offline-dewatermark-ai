import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import masks, pipeline, recall, verdict, observed_text, lama
from wm.app import MainWindow, make_app
from wm.batches import Batch
from wm.contract import EngineKind, MaskLayer, PageImage, PageResult, Thresholds


class RecallGeometryTests(unittest.TestCase):
    def test_search_box_never_authorizes_blank_background_repair(self):
        image = np.full((260, 500, 3), 180, np.uint8)
        box = np.array([[40, 70], [420, 70], [420, 190], [40, 190]], np.float32)
        seed = np.zeros((260, 500), bool)
        seed[90:105, 60:75] = True
        base = MaskLayer(seed.astype(np.float32), footprint=seed, confidence=.8,
                         method='photo_confirmed_colour_line')
        with patch('wm.recall._lines', return_value=[(box, 'READ 17', .95)]):
            selected = recall.expand(image, base, Thresholds()).binary() > 0
        self.assertFalse(selected.any())
        np.testing.assert_array_equal(image, np.full_like(image, 180))

    def test_missing_glyphs_are_recovered_but_letter_holes_and_spaces_remain_open(self):
        strokes = np.zeros((260, 500), np.uint8)
        cv2.putText(strokes, 'O 8 B', (50, 175), cv2.FONT_HERSHEY_SIMPLEX,
                    3.2, 255, 6, cv2.LINE_AA)
        alpha = strokes.astype(np.float32)/255
        source = np.uint8(220*(1-alpha[..., None])+np.array([65, 110, 75])*alpha[..., None])
        original = source.copy()
        box = np.array([[40, 85], [430, 85], [430, 190], [40, 190]], np.float32)
        seed = (strokes > 100) & (np.indices(strokes.shape)[1] < 140)
        base = MaskLayer(seed.astype(np.float32), footprint=seed, confidence=.8,
                         method='photo_confirmed_colour_line', text_keepout=np.ones_like(seed))
        with patch('wm.recall._lines', return_value=[(box, 'UNSEEN GLYPHS', .95)]):
            result = recall.expand(source, base, Thresholds(lama_mask_dilate=0))
        selected = result.binary() > 0
        self.assertGreater(selected[strokes == 255].mean(), .97)
        background = (strokes == 0).astype(np.uint8)
        flooded = background.copy()
        cv2.floodFill(flooded, None, (0, 0), 2)
        holes = cv2.erode((flooded == 1).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        self.assertTrue(holes.any())
        self.assertFalse(selected[holes].any())
        self.assertFalse(selected[20:60].any())
        self.assertLess(selected.mean(), .12)
        self.assertIsNone(result.text_keepout)
        np.testing.assert_array_equal(original, source)

    def test_repeat_template_preserves_holes_and_rejects_inverse_background_view(self):
        glyphs = np.zeros((100, 280), np.uint8)
        cv2.putText(glyphs, 'O8 B', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 5)
        field = glyphs.astype(np.float32)/255*10
        box = np.array([[0, 0], [279, 0], [279, 99], [0, 99]], np.float32)
        observations = [(box, 100., .99)]
        self.assertTrue(observed_text._positive_ink_view(field, observations))
        self.assertFalse(observed_text._positive_ink_view(10-field, observations))
        selected = observed_text._stroke_template(field)
        self.assertTrue(selected[glyphs == 255].all())
        self.assertFalse(selected[glyphs == 0].any())
        self.assertGreater(selected[glyphs > 0].mean(), .90)
        self.assertFalse(observed_text._stroke_template(np.zeros_like(field)).any())

    def test_measured_texture_and_inverse_repairs_keep_their_physical_models(self):
        image = np.full((40, 60, 3), 180, np.uint8)
        alpha = np.zeros((40, 60), np.float32)
        alpha[15:25, 20:30] = .8
        for mode in ('texture_overlay', 'photo_repeat_inverse', 'coloured_document_overlay'):
            with self.subTest(mode=mode):
                mask = MaskLayer(alpha, footprint=np.ones(alpha.shape, bool),
                                 repair_mode=mode, paper=image.astype(np.float32),
                                 veil_strength=np.full(3, .7), veil_offset=np.full(3, 20.))
                with patch('wm.recall._lines', side_effect=AssertionError('measured model must survive')):
                    result = recall.expand(image, mask, Thresholds())
                self.assertEqual(result.repair_mode, mode)
                self.assertIs(result.alpha, mask.alpha)
                self.assertIs(result.paper, mask.paper)
                np.testing.assert_array_equal(result.footprint, alpha >= .08)

    def test_zero_requested_dilation_does_not_write_into_letter_holes(self):
        image = np.full((32, 40, 3), 180, np.uint8)
        region = np.zeros((32, 40), bool)
        region[5:25, 8:28] = True
        region[9:21, 12:24] = False
        with patch('wm.lama._load_session', return_value=object()), \
             patch('wm.lama._inpaint_region', side_effect=lambda session, src, mask, thr: np.full_like(src, 90)):
            result = lama.inpaint(image, region, Thresholds(lama_mask_dilate=0))
        np.testing.assert_array_equal(result[~region], image[~region])
        self.assertTrue(np.all(result[region] == 90))

    def test_document_inverse_respects_selection_even_with_faint_alpha_outside(self):
        image = np.full((40, 60, 3), 180, np.uint8)
        alpha = np.full((40, 60), .02, np.float32)
        alpha[15:25, 20:30] = .8
        selected = alpha >= .08
        mask = MaskLayer(alpha, footprint=selected, confidence=.9,
                         repair_mode='coloured_document_overlay',
                         paper=np.full_like(image, 220, dtype=np.float32),
                         veil_strength=np.full(3, .7), veil_offset=np.full(3, 20.))
        output, _ = lama.repair_with_lama(image, mask, Thresholds(lama_mask_dilate=0))
        np.testing.assert_array_equal(output[~selected], image[~selected])
        self.assertTrue(np.any(output[selected] != image[selected]))
        self.assertTrue(np.all(mask.alpha[~selected] == .02))

    def test_priority_is_explicit_and_empty_detections_never_authorize_page_erasure(self):
        image = np.full((80, 120, 3), 180, np.uint8)
        base = MaskLayer(np.zeros((80, 120), np.float32))
        with patch('wm.masks._estimate_single_page', return_value=base), \
             patch('wm.recall.expand', return_value=base) as expand:
            masks.estimate_single_page(image)
            expand.assert_not_called()
            masks.estimate_single_page(image, Thresholds(watermark_recall_priority=True))
            expand.assert_called_once()
        with patch('wm.recall._lines', side_effect=AssertionError('no seed')):
            self.assertIs(recall.expand(image, base, Thresholds()), base)
        with patch('wm.recall._lines', side_effect=AssertionError('OCR disabled')):
            self.assertIs(recall.expand(image, base, Thresholds(), use_ocr=False), base)

    def test_priority_geometry_is_not_replaced_by_a_crosspage_prior(self):
        image = np.ones((20, 30), np.float32)
        base = MaskLayer(image, method='recall_text_lines')
        prior = MaskLayer(image, confidence=1.)
        self.assertIs(masks.with_prior(base, prior), base)


class UnknownResidualTests(unittest.TestCase):
    def test_missing_photo_measurement_stays_unknown_in_verdict_and_export(self):
        source = PageImage(np.full((20, 30, 3), 180, np.uint8), 96, 'photo')
        mask = MaskLayer(np.ones((20, 30), np.float32), confidence=.8)
        report = verdict.build_verdict(source.image, source.image, mask,
            EngineKind.RASTER, [], 1, {}, Thresholds())
        self.assertTrue(np.isnan(report.metrics['residual_coverage']))
        self.assertNotIn('residual', report.conditions)
        self.assertTrue(report.review_needed)
        with tempfile.TemporaryDirectory() as folder:
            root = pipeline.write_results([PageResult(source, source, mask, report)], folder)
            data = json.loads((root/'manifest.json').read_text())
            self.assertIsNone(data['pages'][0]['verdict']['metrics']['residual_coverage'])
            self.assertIsNone(data['summary']['success_rate'])


class RecallWorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def test_default_priority_is_saved_per_batch_and_can_be_disabled(self):
        window = MainWindow()
        page = PageImage(np.full((20, 30, 3), 180, np.uint8), 96)
        window.batches = [Batch('one', [page]), Batch('two', [page])]
        window.refresh_batches()
        window.batch_list.setCurrentRow(0)
        self.assertTrue(window.current_thr().watermark_recall_priority)
        window.recall_box.setChecked(False)
        self.assertFalse(window.batch.recall_priority)
        window.batch_list.setCurrentRow(1)
        self.assertTrue(window.current_thr().watermark_recall_priority)
        window.batch_list.setCurrentRow(0)
        self.assertFalse(window.current_thr().watermark_recall_priority)
        window.close()


if __name__ == '__main__':
    unittest.main()
