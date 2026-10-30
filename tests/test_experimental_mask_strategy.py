import unittest
from unittest.mock import patch

import numpy as np

from tools.experimental_mask_strategy import enabled
from wm import repair
from wm.contract import MaskLayer, RepairName, Thresholds


class ExperimentalMaskStrategyTests(unittest.TestCase):
    def test_abstention_preserves_complete_detection_and_source_without_calling_model(self):
        source = np.full((100, 140, 3), 180, np.uint8)
        selected = np.zeros((100, 140), bool)
        selected[40:70, 25:110] = True
        mask = MaskLayer(selected.astype(np.float32), footprint=selected.copy(),
                         repair_mode='inpaint', method='recall_pixel_strokes')
        before = mask.binary().copy()
        lines = [(np.array([[5, y], [50, y], [50, y+10], [5, y+10]], np.float32),
                  'visible', .9) for y in (10, 30, 60)]
        observations = []
        with patch('wm.ocr.read_lines', return_value=lines), \
                patch('wm.lama.inpaint', side_effect=AssertionError('must abstain')), enabled(observations):
            output, used = repair.repair(source, mask, Thresholds())
        np.testing.assert_array_equal(source, output)
        np.testing.assert_array_equal(mask.binary(), before)
        np.testing.assert_array_equal(mask.footprint, selected)
        self.assertEqual(used, [RepairName.NONE])
        self.assertTrue(observations[0]['abstained'])
        self.assertIsNone(observations[0]['unrecoverable_pixels'])

    def test_unavailable_ocr_is_unknown_risk_instead_of_a_safe_background_claim(self):
        source = np.full((20, 30, 3), 180, np.uint8)
        mask = MaskLayer(np.ones((20, 30), np.float32), repair_mode='inpaint',
                         method='recall_pixel_strokes')
        observations=[]
        with patch('wm.ocr.read_lines', return_value=None), enabled(observations):
            output, _ = repair.repair(source, mask, Thresholds())
        np.testing.assert_array_equal(output, source)
        self.assertFalse(observations[0]['source_ocr_available'])
        self.assertTrue(observations[0]['abstained'])

    def test_measured_physical_repairs_bypass_this_research_policy(self):
        source = np.full((20, 30, 3), 180, np.uint8)
        expected = source.copy()
        mask = MaskLayer(np.ones((20, 30), np.float32), repair_mode='document_overlay_full')
        with patch.object(repair, 'repair', return_value=(expected, [RepairName.ALPHA_UNMIX])), \
                patch('wm.ocr.read_lines', side_effect=AssertionError('physical route must bypass')), enabled([]):
            output, used = repair.repair(source, mask, Thresholds())
        self.assertIs(output, expected)
        self.assertEqual(used, [RepairName.ALPHA_UNMIX])
