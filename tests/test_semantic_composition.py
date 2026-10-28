import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "tools/compose_semantic_repair.py"
spec = importlib.util.spec_from_file_location("semantic_composition_tested", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
blend = module.blend_generated_background


class SemanticCompositionTests(unittest.TestCase):
    def test_generated_structure_is_recovered_without_editing_surroundings(self):
        clean = np.full((20, 24, 3), 90, np.uint8)
        mask = np.zeros((20, 24), bool)
        mask[4:16, 5:19] = True
        generated = clean.copy()
        generated[7:12, 8:14] = [150, 120, 60]
        marked = clean.copy()
        marked[mask] = [180, 160, 5]
        result = blend(marked, generated, mask)
        np.testing.assert_array_equal(result, generated)
        np.testing.assert_array_equal(marked[mask], np.tile([180, 160, 5], (mask.sum(), 1)))

    def test_global_color_drift_does_not_leave_a_mask_shaped_patch(self):
        original = np.full((18, 22, 3), 70, np.uint8)
        mask = np.zeros((18, 22), bool)
        mask[5:13, 6:16] = True
        original[mask] = [170, 140, 10]
        generated = np.full_like(original, 110)
        result = blend(original, generated, mask)
        np.testing.assert_array_equal(result[mask], 70)
        np.testing.assert_array_equal(result[~mask], original[~mask])

    def test_disconnected_selection_touching_image_edge_is_valid(self):
        original = np.full((12, 14, 3), 80, np.uint8)
        mask = np.zeros((12, 14), bool)
        mask[:3, :4] = True
        mask[7:10, 8:12] = True
        original[mask] = [200, 180, 15]
        generated = np.full_like(original, 95)
        result = blend(original, generated, mask)
        np.testing.assert_array_equal(result[mask], 80)
        np.testing.assert_array_equal(result[~mask], original[~mask])

    def test_empty_and_full_selection(self):
        original = np.full((5, 7, 3), 20, np.uint8)
        generated = np.full_like(original, 100)
        np.testing.assert_array_equal(blend(original, generated, np.zeros((5, 7), bool)), original)
        np.testing.assert_array_equal(blend(original, generated, np.ones((5, 7), bool)), generated)

    def test_bad_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            blend(np.zeros((4, 5, 3), np.uint8), np.zeros((4, 6, 3), np.uint8), np.zeros((4, 5)))


if __name__ == "__main__":
    unittest.main()
