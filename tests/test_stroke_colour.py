import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import cli, recall, stroke_colour
from wm.contract import MaskLayer, Thresholds


class StrokeColourTests(unittest.TestCase):
    def test_native_evidence_preserves_holes_and_rejects_smoothed_halo(self):
        ink = np.zeros((260, 500), np.uint8)
        cv2.putText(ink, 'O 8 B', (40, 180), cv2.FONT_HERSHEY_SIMPLEX,
                    3, 255, 5, cv2.LINE_AA)
        for colour in ([100, 50, 160], [60, 140, 80]):
            with self.subTest(colour=colour):
                alpha = ink.astype(np.float32)/255
                source = np.uint8(245*(1-alpha[..., None])+np.array(colour)*alpha[..., None])
                original = source.copy()
                domain = np.zeros_like(ink, bool)
                domain[80:200, 20:460] = True
                seed = (ink > 150) & (np.indices(ink.shape)[1] < 150)
                region = stroke_colour.refine(source, domain, seed, 90.)
                self.assertIsNotNone(region)
                self.assertGreater(region[ink == 255].mean(), .98)
                self.assertFalse(region[ink == 0].any())
                self.assertFalse(region[~domain].any())
                np.testing.assert_array_equal(source, original)

    def test_uncoloured_false_seed_cannot_authorize_repair(self):
        image = np.full((100, 160, 3), 245, np.uint8)
        seed = np.zeros((100, 160), bool)
        seed[30:60, 40:80] = True
        self.assertIsNone(stroke_colour.refine(image, np.ones_like(seed), seed, 30))
        photo = np.full_like(image, (30, 70, 100))
        with patch('wm.stroke_colour.cv2.medianBlur', side_effect=AssertionError('photo must decline')):
            self.assertIsNone(stroke_colour.refine(photo, np.ones_like(seed), seed, 30))

    def test_default_and_measured_inverse_do_not_enter_experiment(self):
        image = np.full((100, 160, 3), 245, np.uint8)
        region = np.zeros((100, 160), bool)
        region[30:60, 40:80] = True
        box = np.array([[20, 20], [130, 20], [130, 80], [20, 80]], np.float32)
        mask = MaskLayer(region.astype(np.float32), footprint=region)
        with patch('wm.recall._lines', return_value=[(box, 'UNREAD 89', .9)]), \
             patch('wm.recall._pixel_strokes', return_value=region), \
             patch('wm.stroke_colour.refine', side_effect=AssertionError('default changed')):
            output = recall.expand(image, mask, Thresholds())
            np.testing.assert_array_equal(output.binary() > 0, region)
        mask.repair_mode = 'texture_overlay'
        with patch('wm.stroke_colour.refine', side_effect=AssertionError('inverse replaced')):
            output = recall.expand(image, mask, Thresholds(experimental_colour_refinement=True))
            self.assertEqual(output.repair_mode, 'texture_overlay')
            self.assertIs(output.alpha, mask.alpha)

    def test_cli_experiment_is_explicit_and_enables_supported_stroke_route(self):
        for arguments, priority, experimental in [([], False, False),
                (['--recall-priority'], True, False),
                (['--experimental-colour-refinement'], True, True)]:
            with self.subTest(arguments=arguments), \
                 patch('wm.cli.pipeline.process_file', return_value=[]) as process, \
                 patch('wm.cli.pipeline.write_results', return_value=Path('/tmp/mock-export')), \
                 contextlib.redirect_stdout(io.StringIO()):
                cli.main(['run', 'not-read.png', *arguments])
                thr = process.call_args.kwargs['thr']
                self.assertEqual(thr.watermark_recall_priority, priority)
                self.assertEqual(thr.experimental_colour_refinement, experimental)

    def test_explicit_candidate_is_used_only_when_the_source_model_accepts(self):
        image = np.full((100, 160, 3), 245, np.uint8)
        region = np.zeros((100, 160), bool)
        region[30:60, 40:80] = True
        refined = region.copy()
        refined[35:40, 85:90] = True
        box = np.array([[20, 20], [130, 20], [130, 80], [20, 80]], np.float32)
        mask = MaskLayer(region.astype(np.float32), footprint=region)
        for candidate in (refined, None):
            with self.subTest(accepted=candidate is not None), \
                 patch('wm.recall._lines', return_value=[(box, 'SOURCE OBSERVATION', .9)]), \
                 patch('wm.recall._pixel_strokes', return_value=region), \
                 patch('wm.stroke_colour.refine', return_value=candidate) as refine:
                output = recall.expand(image, mask, Thresholds(experimental_colour_refinement=True))
                refine.assert_called_once()
                np.testing.assert_array_equal(output.binary() > 0, refined if candidate is not None else region)
                self.assertEqual(output.method, 'recall_colour_refinement' if candidate is not None else 'recall_pixel_strokes')


if __name__ == '__main__':
    unittest.main()
