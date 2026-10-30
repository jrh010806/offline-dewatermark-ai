import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools.experimental_mask_contract import enabled
from wm import recall
from wm.contract import MaskLayer, Thresholds


class ExperimentalMaskContractTests(unittest.TestCase):
    def test_dark_stroke_retention_requires_native_evidence_and_keeps_holes(self):
        source = np.full((220, 440, 3), 220, np.uint8)
        cv2.putText(source, 'O 8 B', (25, 155), cv2.FONT_HERSHEY_SIMPLEX,
                    2.8, (64, 60, 60), 5, cv2.LINE_AA)
        original = source.copy()
        seed = cv2.dilate((source.mean(2) < 105).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        base = MaskLayer(seed.astype(np.float32), footprint=seed, repair_mode='inpaint',
                         method='contrast_dark_oblique')
        box = np.array([[15, 60], [420, 60], [420, 180], [15, 180]], np.float32)
        observations = []
        thr = Thresholds(watermark_recall_priority=True)
        with patch.object(recall, '_lines', return_value=[(box, 'OBSERVED', .95)]):
            old = recall.expand(source, base, thr).binary() > 0
            with enabled(observations):
                new = recall.expand(source, base, thr).binary() > 0
        added = new & ~old
        self.assertTrue(added.any())
        self.assertFalse((old & ~new).any())
        self.assertFalse((added & ~seed).any())
        self.assertTrue((source.mean(2)[added] < 105).all())
        self.assertFalse(added[source.mean(2) == 220].any())
        self.assertTrue(observations)
        np.testing.assert_array_equal(source, original)

    def test_nonempty_box_and_seed_cannot_retain_pixels_on_uniform_input(self):
        source = np.full((220, 440, 3), 60, np.uint8)
        seed = np.zeros(source.shape[:2], bool)
        seed[90:120, 65:100] = True
        base = MaskLayer(seed.astype(np.float32), footprint=seed, repair_mode='inpaint',
                         method='contrast_dark_oblique')
        box = np.array([[15, 60], [420, 60], [420, 180], [15, 180]], np.float32)
        with patch.object(recall, '_lines', return_value=[(box, 'OBSERVED', .95)]), enabled([]):
            result = recall.expand(source, base, Thresholds(watermark_recall_priority=True))
        self.assertFalse(result.binary().any())

    def test_edge_variant_requires_observed_contrast_instead_of_filling_padding(self):
        source = np.full((220, 440, 3), 220, np.uint8)
        cv2.putText(source, 'O 8 B', (25, 155), cv2.FONT_HERSHEY_SIMPLEX,
                    2.8, (64, 60, 60), 5, cv2.LINE_AA)
        seed = cv2.dilate((source.mean(2) < 105).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        base = MaskLayer(seed.astype(np.float32), footprint=seed,
                         repair_mode='inpaint', method='contrast_dark_oblique')
        box = np.array([[15, 60], [420, 60], [420, 180], [15, 180]], np.float32)
        thr = Thresholds(watermark_recall_priority=True)
        with patch.object(recall, '_lines', return_value=[(box, 'OBSERVED', .95)]):
            old = recall.expand(source, base, thr).binary() > 0
            with enabled([], edge_support=True):
                result = recall.expand(source, base, thr).binary() > 0
        added = result & ~old
        self.assertTrue((added & (source.mean(2) >= 105)).any())
        self.assertFalse((added & ~seed).any())
        self.assertFalse(added[source.mean(2) == 220].any())

    def test_candidate_does_not_replace_physical_routes_or_other_style_masks(self):
        source = np.full((40, 60, 3), 180, np.uint8)
        for mode, method in [('document_overlay_full', 'single_document'),
                             ('inpaint', 'single_periodic_overlay')]:
            mask = MaskLayer(np.ones((40, 60), np.float32), repair_mode=mode, method=method)
            with enabled([]), patch.object(recall, '_pixel_strokes', side_effect=AssertionError('must bypass')):
                result = recall.expand(source, mask, Thresholds(watermark_recall_priority=True))
            self.assertIs(result.alpha, mask.alpha)
            self.assertEqual(result.repair_mode, mode)


if __name__ == '__main__':
    unittest.main()
