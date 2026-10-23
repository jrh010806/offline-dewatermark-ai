import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm.photo_overlay import fit, refine
from wm.contract import MaskLayer, Thresholds
from wm.lama import repair_with_lama


class PhotoOverlayTests(unittest.TestCase):
    def test_calibration_recovers_a_different_transmission_from_source_only(self):
        yy, xx = np.indices((180, 260))
        gray = 130+60*np.sin(yy/31)+25*np.cos(xx/53)
        clean = np.uint8(np.stack([gray-8, gray, gray+12], 2))
        region = np.zeros(gray.shape, bool)
        for x in (35, 80, 135, 195):
            region[15:165, x:x+8] = True
        image = clean.copy()
        image[region] = np.uint8(clean[region]*.6+np.array([40, 45, 55]))
        fitted = fit(image, region)
        self.assertIsNotNone(fitted)
        transmission, offset = fitted
        np.testing.assert_allclose(transmission, .6, atol=.04)
        np.testing.assert_allclose(offset, [40, 45, 55], atol=7)

    def test_source_texture_is_recovered_without_inserting_mask_edges(self):
        yy, xx = np.indices((100, 120))
        clean = np.uint8(np.repeat((110+25*np.sin(xx*1.2)+20*np.cos(yy*.9))[:, :, None], 3, 2))
        region = np.zeros((100, 120), bool)
        region[15:85, 15:105] = True
        s, t = np.full(3, .5, np.float32), np.full(3, 50, np.float32)
        source = clean.copy()
        source[region] = np.uint8(clean[region]*s+t)
        model = cv2.GaussianBlur(clean, (0, 0), 2)
        model[~region] = 0  # model context is not permission to change output
        output = refine(source, model, region, s, t)
        interior = cv2.erode(region.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        before = np.mean(np.abs(model[interior].astype(float)-clean[interior]))
        after = np.mean(np.abs(output[interior].astype(float)-clean[interior]))
        self.assertLess(after, before*.6)
        distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
        rim = region & (distance <= 3)
        np.testing.assert_array_equal(output[rim], model[rim])
        np.testing.assert_array_equal(output[~region], source[~region])

    def test_disagreeing_inverse_is_not_written_as_bright_halos(self):
        image = np.full((80, 90, 3), 240, np.uint8)
        region = np.zeros((80, 90), bool)
        region[10:70, 10:80] = True
        model = np.full_like(image, 100)
        output = refine(image, model, region, np.full(3, .4, np.float32), np.full(3, 10, np.float32))
        np.testing.assert_array_equal(output[region], model[region])
        np.testing.assert_array_equal(output[~region], image[~region])

    def test_flat_or_tiny_region_cannot_calibrate_transmission(self):
        image = np.full((100, 100, 3), 180, np.uint8)
        region = np.zeros((100, 100), bool)
        region[20:80, 20:80] = True
        self.assertIsNone(fit(image, region))
        region[:] = False
        region[30:33, 30:33] = True
        self.assertIsNone(fit(image, region))

    def test_repair_keeps_explicit_mask_expansion_while_refining_original_core(self):
        source = np.full((100, 120, 3), 150, np.uint8)
        region = np.zeros(source.shape[:2], bool)
        region[30:70, 40:80] = True
        expanded = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        prediction = source.copy()
        prediction[expanded] = 100
        mask = MaskLayer(region.astype(np.float32), footprint=region,
                         veil_strength=np.full(3, .5, np.float32),
                         veil_offset=np.full(3, 50, np.float32), repair_mode='photo_overlay')
        with patch('wm.lama.inpaint', return_value=prediction):
            output, _ = repair_with_lama(source, mask, Thresholds(lama_mask_dilate=2))
        np.testing.assert_array_equal(output[expanded & ~region], prediction[expanded & ~region])
        np.testing.assert_array_equal(output[~expanded], source[~expanded])


if __name__ == '__main__':
    unittest.main()
