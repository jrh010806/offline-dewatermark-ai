import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import chromatic, masks, recall
from wm.contract import MaskLayer, Thresholds


class BalancedChromaticTests(unittest.TestCase):
    def test_each_polarity_gets_distinct_directions_despite_unequal_strength(self):
        signal = np.full((80, 160), 12, np.float32)
        signal[20:40, 30:110] = -8
        directions = chromatic._balanced_directions(signal)
        for sign in (-1, 1):
            angles = [a for s, a in directions if s == sign]
            self.assertEqual(len(angles), 3)
            self.assertTrue(all(abs(a-b) >= 10 for i, a in enumerate(angles)
                                for b in angles[i+1:]))

    def test_ocr_box_cannot_grant_pixels_on_uniform_source(self):
        image = np.full((240, 400, 3), (20, 50, 80), np.uint8)
        box = np.array([[20, 60], [380, 60], [380, 180], [20, 180]], np.float32)
        with patch('wm.photo_text._fit', side_effect=AssertionError('no pixels to fit')):
            self.assertIsNone(chromatic._native_strokes(image, np.zeros((240, 400), np.float32), box, -1))
            self.assertIsNone(chromatic._native_strokes(image, np.full((240, 400), -12, np.float32), box, -1))

    def test_native_selection_preserves_glyph_holes_and_whitespace(self):
        image = np.full((180, 440, 3), (20, 50, 80), np.uint8)
        strokes = np.zeros(image.shape[:2], np.uint8)
        cv2.putText(strokes, 'O8BO', (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.8, 1, 6)
        field = -strokes.astype(np.float32)*14
        image[strokes > 0] = (80, 80, 90)
        box = np.array([[10, 40], [425, 40], [425, 145], [10, 145]], np.float32)
        # Test pixel permission separately from the existing physical-fit tests.
        with patch('wm.photo_text._fit', return_value=(np.ones(3)*.5, np.ones(3)*40)):
            selected = chromatic._native_strokes(image, field, box, -1)
        self.assertIsNotNone(selected)
        self.assertGreater(selected.sum(), 100)
        self.assertFalse(selected[strokes == 0].any())

    def test_no_ocr_no_permission_and_neutral_input_skips_fallback(self):
        image = np.full((180, 300, 3), 60, np.uint8)
        with patch('wm.ocr.read_lines', side_effect=AssertionError('neutral source')):
            self.assertIsNone(chromatic.estimate_balanced(image))
        image[:] = (20, 50, 80)
        with patch('wm.ocr.read_lines', return_value=[]):
            self.assertIsNone(chromatic.estimate_balanced(image))

    def test_verified_native_strokes_survive_recall_without_generic_reselection(self):
        image = np.zeros((30, 40, 3), np.uint8)
        alpha = np.zeros((30, 40), np.float32);alpha[5:15, 7:10] = 1
        layer = MaskLayer(alpha, method='experimental_balanced_chromatic_strokes', footprint=alpha > 0)
        with patch('wm.recall._lines', side_effect=AssertionError('already verified')):
            self.assertIs(recall.expand(image, layer, Thresholds(watermark_recall_priority=True)), layer)
        self.assertFalse(Thresholds().experimental_oblique_detection)


if __name__ == '__main__':
    unittest.main()
