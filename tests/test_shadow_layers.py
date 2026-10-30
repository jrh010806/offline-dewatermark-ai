from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wm import export_review, observed_text, pipeline, recall, repair, shadow_layers
from wm.contract import MaskLayer, PageImage, RepairName, Thresholds
from wm.selection_edit import SelectionEdit, apply_selection


class ShadowLayerIntegrationTests(unittest.TestCase):
    def example(self):
        image = np.full((60, 90, 3), 230, np.uint8)
        search = np.zeros(image.shape[:2], bool)
        search[10:50, 15:75] = True
        image[search] = (120, 140, 130)
        image[12:18, 20:45] = (40, 60, 50)
        selected = search.copy()
        selected[20:40, 30:60] = False
        legacy = observed_text._layer(search, 'single_observed_shadow',
                                      'shadow_paper_cleanup', np.full(image.shape, 255, np.float32))
        reconstruction = image.copy()
        reconstruction[selected] = 255
        return image, legacy, selected, reconstruction

    def prepared(self):
        image, legacy, selected, reconstruction = self.example()
        with patch.object(shadow_layers, 'supported_inverse',
                          return_value=(reconstruction, selected, {'source_only': True})):
            mask = shadow_layers.prepare(image, legacy)
        return image, mask, selected, reconstruction

    def test_default_detection_uses_layered_repair_only_after_shadow_search(self):
        image, legacy, selected, reconstruction = self.example()
        with patch.object(observed_text, '_shadow_paper_search', return_value=legacy), \
             patch.object(shadow_layers, 'supported_inverse',
                          return_value=(reconstruction, selected, {})) as fit:
            mask = observed_text.shadow_paper(image, [])
        self.assertEqual(mask.repair_mode, 'shadow_layer_inverse')
        self.assertEqual(fit.call_args.args[2], .5)
        np.testing.assert_array_equal(mask.footprint, selected)
        self.assertFalse(mask.shadow_evidence['quality_verified'])
        self.assertFalse(mask.footprint[25, 40])
        # A non-shadow result cannot invoke the new fitter.
        with patch.object(observed_text, '_shadow_paper_search', return_value=None), \
             patch.object(shadow_layers, 'supported_inverse', side_effect=AssertionError('unrelated path')):
            self.assertIsNone(observed_text.shadow_paper(image, []))

    def test_rejected_fit_preserves_original_path_and_records_reason(self):
        image, legacy, _, _ = self.example()
        with patch.object(shadow_layers, 'supported_inverse', side_effect=ValueError('配准未通过')):
            mask = shadow_layers.prepare(image, legacy)
        self.assertEqual(mask.repair_mode, legacy.repair_mode)
        self.assertIs(mask.alpha, legacy.alpha)
        self.assertIs(mask.footprint, legacy.footprint)
        self.assertEqual(mask.shadow_evidence['status'], 'fallback')
        self.assertIn('配准', mask.shadow_evidence['reason'])
        self.assertIsNone(mask.shadow_reconstruction)

    def test_unbounded_candidate_cannot_replace_the_original(self):
        image, legacy, selected, reconstruction = self.example()
        reconstruction[0, 0] = 0
        with patch.object(shadow_layers, 'supported_inverse', return_value=(reconstruction, selected, {})):
            mask = shadow_layers.prepare(image, legacy)
        self.assertEqual(mask.repair_mode, 'shadow_paper_cleanup')
        self.assertEqual(mask.shadow_evidence['status'], 'fallback')

    def test_completion_does_not_replace_fit_and_repair_keeps_holes(self):
        image, mask, selected, expected = self.prepared()
        with patch('wm.recall._lines', side_effect=AssertionError('do not redetect')), \
             patch('wm.lama._load_session', side_effect=AssertionError('no neural repair')):
            retained = recall.expand(image, mask, Thresholds(watermark_recall_priority=True))
            output, used = repair.repair(image, retained)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(output[~selected], image[~selected])
        self.assertIs(retained.shadow_reconstruction, mask.shadow_reconstruction)
        self.assertEqual(used, [RepairName.ALPHA_UNMIX])

    def test_brush_exclusions_do_not_reapply_the_cached_reconstruction(self):
        image, mask, _, _ = self.prepared()
        result = pipeline._process_one(PageImage(image, 100, 'unit-shadow'), mask, Thresholds())
        edit = SelectionEdit(result)
        edit.begin_stroke()
        edit.paint((20.5, 15.5), (20.5, 15.5), 7, False)
        edit.end_stroke()
        with patch('wm.masks.estimate_single_page', side_effect=AssertionError('no redetection')):
            output = apply_selection(result, edit.request(Thresholds()))
        np.testing.assert_array_equal(output.output.image[~edit.selected], image[~edit.selected])
        self.assertTrue(np.all(output.output.image[edit.selected] == 255))
        self.assertEqual(output.processing['human_assistance'][-1]['retained_automatic_route'],
                         'shadow_layer_inverse')

    def test_export_retains_fit_and_distinguishes_permission_from_opacity(self):
        image, mask, _, _ = self.prepared()
        result = pipeline._process_one(PageImage(image, 100, 'unit-shadow'), mask, Thresholds())
        with tempfile.TemporaryDirectory() as folder:
            evidence = export_review.write_selection(result, Path(folder), 'shadow')
            with np.load(Path(folder) / evidence['files']['mask_data']) as data:
                np.testing.assert_array_equal(data['shadow_reconstruction'], mask.shadow_reconstruction)
            self.assertIn('不是测得', evidence['selection']['alpha_note'])
            self.assertFalse(evidence['selection']['shadow_layers']['quality_verified'])
        self.assertTrue(result.verdict.review_needed)


if __name__ == '__main__':
    unittest.main()
