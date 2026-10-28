import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import photo_repeat_inverse, lama, pipeline
from wm.contract import MaskLayer, Thresholds


def sample(strength=.16):
    h, w = 600, 720
    yy, xx = np.indices((h, w))
    gray = np.uint8(65+120*xx/w+35*yy/h)
    clean = np.repeat(gray[..., None], 3, axis=2)
    cv2.putText(clean, 'KEEP ARROWS >', (120, 295), cv2.FONT_HERSHEY_SIMPLEX,
                .9, (20, 20, 20), 2, cv2.LINE_AA)
    strokes = np.zeros((h, w), np.uint8)
    for y in range(60, 600, 110):
        for x in range(10, 700, 175):
            cv2.putText(strokes, 'FX42', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, 255, 4, cv2.LINE_AA)
    opacity = strokes.astype(np.float32)/255*strength
    source = np.uint8(clean*(1-opacity[..., None])+np.array([40, 75, 140])*opacity[..., None])
    coarse = cv2.dilate(strokes, np.ones((5, 5), np.uint8)) > 0
    return clean, source, coarse, opacity


class PhotoRepeatInverseTests(unittest.TestCase):
    def test_measured_inverse_reduces_faint_repeats_without_repainting_scene(self):
        clean, source, coarse, truth = sample()
        original = source.copy()
        mask = photo_repeat_inverse.estimate(source, coarse)
        self.assertIsNotNone(mask)
        with patch('wm.lama.inpaint', side_effect=AssertionError('must not repaint')):
            output, used = lama.repair_with_lama(source, mask, Thresholds())
        marked = truth > 0
        before = np.abs(source.astype(float)-clean).mean(2)
        after = np.abs(output.astype(float)-clean).mean(2)
        self.assertLess(after[marked].mean(), before[marked].mean()*.65)
        self.assertLess(after[~marked].mean(), .3)
        np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
        np.testing.assert_array_equal(original, source)

    def test_normal_scene_and_strong_watermarks_decline(self):
        clean, _, coarse, _ = sample()
        self.assertIsNone(photo_repeat_inverse.estimate(clean, coarse))
        _, heavy, coarse, _ = sample(.65)
        self.assertIsNone(photo_repeat_inverse.estimate(heavy, coarse))
        self.assertIsNone(photo_repeat_inverse.estimate(clean, np.zeros(coarse.shape, bool)))

    def test_inverse_requires_measured_opacity_and_honours_selection(self):
        image = np.full((20, 30, 3), 100, np.uint8)
        region = np.zeros((20, 30), bool)
        region[5:10, 5:10] = True
        mask = MaskLayer(region.astype(np.float32), footprint=region,
                         repair_mode='photo_repeat_inverse', veil_strength=np.zeros(3),
                         veil_offset=np.array([40, 75, 140]))
        with self.assertRaises(ValueError):
            lama.repair_with_lama(image, mask, Thresholds())
        mask.overlay_opacity = np.full(region.shape, .15, np.float32)
        output, _ = lama.repair_with_lama(image, mask, Thresholds())
        np.testing.assert_array_equal(output[~region], image[~region])
        mask.overlay_opacity[0, 0] = np.nan
        with self.assertRaises(ValueError):
            lama.repair_with_lama(image, mask, Thresholds())

    def test_export_uses_physical_opacity_not_binary_selection(self):
        opacity = np.full((4, 5), .15, np.float32)
        mask = MaskLayer(np.ones((4, 5), np.float32), overlay_opacity=opacity,
                         veil_strength=np.zeros(3), veil_offset=np.array([40, 75, 140]))
        exported = pipeline.extract_watermark(mask)
        np.testing.assert_array_equal(exported[..., 3], np.uint8(opacity*255))


if __name__ == '__main__':
    unittest.main()
