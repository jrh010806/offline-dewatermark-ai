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


class PaperBackgroundGuardTests(unittest.TestCase):
    def test_normal_chart_bars_and_footer_are_left_unchanged(self):
        from wm import pipeline
        from wm.contract import PageImage
        image = np.full((350, 450, 3), 255, np.uint8)
        for i in range(5):
            cv2.rectangle(image, (35+i*75, 100+i*12), (65+i*75, 260),
                          (50+i*15, 100, 170-i*10), -1)
        cv2.line(image, (20, 320), (430, 320), (185, 190, 195), 1)
        output = pipeline.process_pages([PageImage(image, 96)], force_single=True, use_ocr=False)[0]
        np.testing.assert_array_equal(output.output.image, image)

    def test_light_footer_rule_is_protected_but_thick_overlay_is_not(self):
        from wm.masks import document_rules
        image = np.full((300, 400, 3), 255, np.uint8)
        cv2.line(image, (20, 270), (380, 270), (190, 190, 190), 1)
        cv2.rectangle(image, (80, 90), (300, 125), (150, 150, 150), -1)
        protected = document_rules(image)
        self.assertTrue(protected[270, 200])
        self.assertFalse(protected[105, 200])

    def test_document_fallback_requires_a_real_neutral_paper_population(self):
        from wm.masks import paper_background_fraction
        document = np.full((240, 320, 3), 250, np.uint8)
        cv2.putText(document, 'BODY TEXT', (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (30, 30, 30), 2)
        self.assertGreater(paper_background_fraction(document), .25)

        yy, xx = np.indices((240, 320))
        photo = np.stack((70 + xx % 100, 45 + yy % 130,
                          90 + ((xx + yy) % 120)), axis=2).astype(np.uint8)
        self.assertLess(paper_background_fraction(photo), .25)

    def test_photo_without_specialist_evidence_gets_no_destructive_fallback_mask(self):
        from wm import masks
        yy, xx = np.indices((240, 320))
        photo = np.stack((70 + xx % 100, 45 + yy % 130,
                          90 + ((xx + yy) % 120)), axis=2).astype(np.uint8)
        with patch('wm.paper_grid.estimate', return_value=None), \
             patch('wm.periodic.estimate', return_value=None), \
             patch('wm.corner_id.estimate', return_value=None):
            mask = masks.estimate_single_page(photo, use_ocr=False)
        self.assertEqual(mask.method, 'nonpaper_background')
        self.assertEqual(mask.coverage(), 0)


if __name__ == '__main__':
    unittest.main()
