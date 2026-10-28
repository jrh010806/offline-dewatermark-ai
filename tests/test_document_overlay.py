import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import document_overlay
from wm.contract import Thresholds
from wm.lama import repair_with_lama


class ColouredDocumentTests(unittest.TestCase):
    @staticmethod
    def fixture(text='SAMPLE 42', angle=-32, colour=(95, 95, 120), opacity=.52):
        h, w = 650, 550
        clean = np.full((h, w, 3), 255, np.uint8)
        keepout = np.zeros((h, w), np.uint8)
        for row in range(16):
            y = 90+row*30
            ink = 0 if row % 2 == 0 else 150
            cv2.putText(clean, 'Invoice item 2035 paid', (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (ink, ink, ink), 1, cv2.LINE_AA)
            keepout[y-15:y+3, 20:330] = 1
        cv2.line(clean, (20, 60), (520, 60), (0, 0, 0), 1)
        layer = np.zeros((h, w), np.uint8)
        cv2.putText(layer, text, (20, 355), cv2.FONT_HERSHEY_SIMPLEX,
                    2.5, 255, 5, cv2.LINE_AA)
        turn = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        layer = cv2.warpAffine(layer, turn, (w, h))
        alpha = layer.astype(float)/255*opacity
        image = np.uint8(clean*(1-alpha[:, :, None])+alpha[:, :, None]*np.array(colour))
        encoded = cv2.imencode('.jpg', image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
        return clean, image, alpha, keepout.astype(bool)

    def test_unseen_words_angles_and_colours_preserve_black_and_gray_body_text(self):
        for phrase, angle, colour in [('SAMPLE 42', -32, (95, 95, 120)),
                                      ('CHECK 78', 28, (110, 135, 155))]:
            with self.subTest(phrase=phrase):
                clean, source, alpha, keepout = self.fixture(phrase, angle, colour)
                mask = document_overlay.estimate(source, keepout)
                self.assertIsNotNone(mask)
                with patch('wm.lama._load_session', side_effect=AssertionError('no generative text repair')):
                    output, _ = repair_with_lama(source, mask, Thresholds())
                region = alpha > .05
                before = np.abs(source.astype(float)-clean)[region].mean()
                after = np.abs(output.astype(float)-clean)[region].mean()
                self.assertLess(after, before*.2)
                np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
                # Gray printed strokes must not be flattened into paper.
                gray_ink = (clean.mean(2) >= 140) & (clean.mean(2) <= 165) & region
                self.assertLess(np.mean(output[gray_ink].mean(1) > 220), .02)
                black = (clean.mean(2) < 40) & region
                self.assertLess(np.mean(output[black].mean(1) > 120), .02)

    def test_horizontal_coloured_heading_is_not_a_diagonal_stamp(self):
        image = np.full((600, 600, 3), 255, np.uint8)
        cv2.putText(image, 'INVOICE 2035', (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    2.5, (145, 145, 170), 5, cv2.LINE_AA)
        with patch('wm.ocr.text_zones', side_effect=AssertionError('reject before OCR')):
            self.assertIsNone(document_overlay.estimate(image))

    def test_faint_coloured_stamps_preserve_body_text_without_inpainting(self):
        for colour, opacity, angle in [((150, 40, 55), .15, -25),
                                       ((70, 145, 160), .20, 32),
                                       ((135, 70, 150), .20, -38)]:
            with self.subTest(colour=colour, opacity=opacity, angle=angle):
                clean, source, alpha, keepout = self.fixture(
                    colour=colour, opacity=opacity, angle=angle)
                mask = document_overlay.estimate(source, keepout)
                self.assertIsNotNone(mask)
                with patch('wm.lama._load_session', side_effect=AssertionError('no text generation')):
                    output, _ = repair_with_lama(source, mask, Thresholds())
                region = alpha > .05
                before = np.abs(source.astype(float)-clean)[region].mean()
                after = np.abs(output.astype(float)-clean)[region].mean()
                self.assertLess(after, before*.25)
                np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
                for lo, hi, limit in [(0, 40, 120), (140, 165, 220)]:
                    ink = (clean.mean(2) >= lo) & (clean.mean(2) <= hi) & region
                    self.assertLess(np.mean(output[ink].mean(1) > limit), .02)

    def test_faint_horizontal_heading_and_near_white_noise_are_not_stamps(self):
        heading = np.full((600, 600, 3), 255, np.uint8)
        cv2.putText(heading, 'INVOICE 2035', (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    2.5, (239, 222, 224), 5, cv2.LINE_AA)
        noise = np.random.default_rng(42).integers(248, 256, (600, 600, 3), dtype=np.uint8)
        with patch('wm.ocr.text_zones', side_effect=AssertionError('reject before OCR')):
            self.assertIsNone(document_overlay.estimate(heading))
            self.assertIsNone(document_overlay.estimate(noise))

    def test_faint_stamp_without_covered_ink_declines_fit(self):
        _, source, _, keepout = self.fixture(colour=(150, 40, 55), opacity=.15)
        self.assertIsNone(document_overlay.estimate(source, np.zeros_like(keepout)))

    def test_no_covered_ink_evidence_declines_a_fit(self):
        _, image, _, keepout = self.fixture()
        self.assertIsNone(document_overlay.estimate(image, np.zeros_like(keepout)))


if __name__ == '__main__':
    unittest.main()
