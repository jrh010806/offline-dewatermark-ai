import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import chromatic, masks, periodic
from wm.contract import MaskLayer, Thresholds
from wm.lama import repair_with_lama


class PeriodicPhotoTests(unittest.TestCase):
    @staticmethod
    def background():
        yy, xx = np.indices((280, 370))
        luminance = 100+45*np.sin(xx/37)+35*np.cos(yy/31)+15*np.sin((xx+yy)/15)
        noise = np.random.default_rng(17).normal(0, 15, yy.shape).astype(np.float32)
        texture = cv2.GaussianBlur(noise, (0, 0), .8)
        return np.uint8(np.clip(np.stack([luminance+texture+10,
                                         luminance+texture, luminance+texture-15], 2), 0, 255))

    def test_unseen_pitches_and_offsets_recover_without_model_or_references(self):
        clean = self.background()
        yy, xx = np.indices(clean.shape[:2])
        for period in (13, 23):
            with self.subTest(period=period):
                dx = (xx-4+period/2) % period-period/2
                dy = (yy-7+period/2) % period-period/2
                alpha = np.clip(2.5-np.sqrt(dx*dx+dy*dy), 0, 1)*.48
                source = np.uint8(clean*(1-alpha[:, :, None])+
                                  alpha[:, :, None]*np.array([140, 140, 150]))
                mask = periodic.estimate(source)
                self.assertIsNotNone(mask)
                self.assertEqual(mask.lattice, (period, period))
                with patch('wm.lama._load_session', side_effect=AssertionError('no neural inference')):
                    output, _ = repair_with_lama(source, mask, Thresholds())
                before = np.mean(np.abs(source.astype(float)-clean))
                after = np.mean(np.abs(output.astype(float)-clean))
                self.assertLess(after, .6*before)
                np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
                self.assertFalse(mask.footprint[alpha == 0].any())

    def test_unmarked_texture_and_real_black_grid_are_not_translucent_dots(self):
        image = self.background()
        self.assertIsNone(periodic.estimate(image))
        grid = image.copy()
        grid[::17] = 30
        grid[:, ::17] = 30
        self.assertIsNone(periodic.estimate(grid))

    def test_prior_cannot_replace_measured_photo_overlay(self):
        alpha = np.zeros((50, 60), np.float32)
        alpha[::10, ::10] = .5
        own = MaskLayer(alpha, method='single_periodic_overlay', confidence=.9)
        prior = MaskLayer(alpha.copy(), method='crosspage', confidence=1)
        self.assertIs(masks.with_prior(own, prior), own)


class ChromaticPhotoTests(unittest.TestCase):
    def test_no_ocr_cannot_authorize_painting_a_colour_outlier(self):
        image = PeriodicPhotoTests.background()
        cv2.putText(image, 'BUILD 42', (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (100, 90, 150), 4)
        with patch('wm.ocr.read_lines', return_value=None):
            self.assertIsNone(chromatic.estimate(image))

    def test_monochrome_document_does_not_enter_photo_colour_path(self):
        image = np.full((300, 200, 3), 255, np.uint8)
        cv2.putText(image, 'BODY', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        with patch('wm.ocr.read_lines', side_effect=AssertionError('document path')):
            self.assertIsNone(chromatic.estimate(image))


if __name__ == '__main__':
    unittest.main()
