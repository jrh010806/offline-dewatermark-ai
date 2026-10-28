import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import masks, photo_observations, texture_overlay
from wm.contract import MaskLayer, Thresholds
from wm.lama import repair_with_lama


def fixture(angle=25, cell=24, opacity=.48, colour=(155, 125, 145), periodic=True):
    h, w = 480, 640
    yy, xx = np.indices((h, w))
    checker = ((xx//cell + yy//cell) % 2).astype(float)
    level = 65+110*checker if periodic else 110+35*np.sin(xx/63)+20*np.cos(yy/49)
    clean = np.stack([level, level+12, level+24], axis=2)
    clean += np.random.default_rng(31).normal(0, 2, clean.shape)
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    stroke = np.zeros((h, w), np.uint8)
    cv2.putText(stroke, 'CHECK 42', (160, 260), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, 255, 4, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    stroke = cv2.warpAffine(stroke, matrix, (w, h))
    box = np.array([[154, 218], [414, 218], [414, 270], [154, 270]], np.float32)
    box = cv2.transform(box[None], matrix)[0]
    alpha = stroke.astype(float)/255*opacity
    source = np.uint8(clean*(1-alpha[..., None])+np.array(colour)*alpha[..., None])
    return clean, source, alpha, [(box, 'CHECK 42', .96)]


class TextureOverlayTests(unittest.TestCase):
    def test_angles_colours_and_periods_preserve_checker_edges_without_generation(self):
        for angle, cell, opacity, colour in [
                (0, 24, .48, (155, 125, 145)),
                (25, 24, .48, (155, 125, 145)),
                (-35, 18, .35, (110, 175, 140)),
                (60, 30, .60, (180, 95, 125)),
                (85, 20, .48, (155, 125, 145))]:
            with self.subTest(angle=angle, cell=cell):
                clean, source, alpha, lines = fixture(angle, cell, opacity, colour)
                mask = photo_observations.local_lines(source, lines)
                self.assertIsNotNone(mask)
                self.assertEqual(mask.method, 'single_periodic_texture_text')
                with patch('wm.lama._load_session', side_effect=AssertionError('no generative repair')):
                    output, _ = repair_with_lama(source, mask, Thresholds())
                region = alpha > .01
                before = np.abs(source.astype(float)-clean)[region].mean()
                after = np.abs(output.astype(float)-clean)[region].mean()
                self.assertLess(after, before*.35)
                np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
                yy, xx = np.indices(alpha.shape)
                edges = region & ((xx % cell == 0) | (yy % cell == 0))
                self.assertLess(np.abs(output.astype(float)-clean)[edges].mean(), 8)

    def test_clean_texture_and_opaque_print_are_not_authorized_by_ocr(self):
        for angle in (0, 25, -35, 85):
            for opacity in (0, 1):
                with self.subTest(angle=angle, opacity=opacity):
                    _, source, _, lines = fixture(angle=angle, opacity=opacity)
                    self.assertIsNone(texture_overlay.estimate(source, lines))
                    self.assertIsNone(photo_observations.local_lines(source, lines))

    def test_missing_geometry_or_nonperiodic_background_declines(self):
        _, source, _, lines = fixture(periodic=False)
        self.assertIsNone(texture_overlay.estimate(source, lines))
        self.assertIsNone(texture_overlay.estimate(source, []))

    def test_center_style_routes_to_periodic_repair_and_respects_scope(self):
        _, source, _, lines = fixture()
        with patch('wm.ocr.read_lines', return_value=[]), \
                patch('wm.photo_observations.observations', return_value=lines):
            mask = masks.estimate_single_page(source, style='big_center')
        self.assertEqual(mask.method, 'single_periodic_texture_text')
        # A batch prior must not replace measured transmission with a generic
        # mask, even when its support happens to match exactly.
        prior = MaskLayer(mask.alpha.copy(), confidence=1.)
        self.assertIs(masks.with_prior(mask, prior), mask)
        self.assertIsNone(photo_observations.local_lines(source, lines, allow_center=False))


if __name__ == '__main__':
    unittest.main()
