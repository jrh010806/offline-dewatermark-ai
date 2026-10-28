import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import masks, photo_observations


class PhotoObservationsTests(unittest.TestCase):
    @staticmethod
    def line_image(opacity):
        yy, xx = np.indices((320, 500))
        base = 100+35*np.sin(xx/65)+30*np.cos(yy/80)
        clean = np.uint8(np.stack([base+25, base, base-20], 2))
        strokes = np.zeros((320, 500), np.uint8)
        cv2.putText(strokes, 'REVIEW 38', (130, 175), cv2.FONT_HERSHEY_SIMPLEX,
                    1, 255, 2, cv2.LINE_AA)
        alpha = strokes.astype(float)/255*opacity
        source = np.uint8(clean*(1-alpha[..., None])+np.array([220, 230, 240])*alpha[..., None])
        box = np.array([[126, 147], [320, 147], [320, 181], [126, 181]], np.float32)
        return source, [(box, 'REVIEW 38', .98)]

    def test_central_translucent_line_can_be_located_without_a_corner(self):
        source, lines = self.line_image(.45)
        mask = photo_observations.local_lines(source, lines)
        self.assertIsNotNone(mask)
        self.assertTrue(mask.footprint[165, 190])
        self.assertFalse(mask.footprint[10, 10])
        self.assertIsNone(photo_observations.local_lines(source, lines, allow_center=False))

    def test_opaque_print_is_not_authorized_by_readable_text_alone(self):
        source, lines = self.line_image(1.)
        self.assertIsNone(photo_observations.local_lines(source, lines))
        self.assertIsNone(photo_observations.local_lines(source, []))

    def test_missing_repeated_row_and_clipped_ends_are_completed(self):
        image = np.full((700, 900, 3), (100, 140, 180), np.uint8)
        theta = np.radians(-30)
        tangent = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-np.sin(theta), np.cos(theta)])
        lines = []
        # Row 440 is deliberately absent from OCR. Text fragments differ, as
        # they do when OCR splits repeated lettering at different characters.
        for row in (200, 320, 560, 680):
            for along in (150, 400):
                centre = along*tangent+row*normal
                box = np.array([centre-100*tangent-19*normal, centre+100*tangent-19*normal,
                                centre+100*tangent+19*normal, centre-100*tangent+19*normal], np.float32)
                lines.append((box, f'PART {row} {along}', .96))
        found = photo_observations._repeated_domain(image, lines)
        self.assertIsNotNone(found)
        domain, _ = found
        self.assertTrue(domain[508, 0])  # missing row, clipped left edge
        self.assertFalse(domain[450, 0])  # between rows
        # Geometric OCR boxes by themselves must not erase a flat photograph.
        self.assertIsNone(photo_observations.repeated_lines(image, lines))

    def test_one_oblique_sign_and_horizontal_captions_do_not_form_a_repeat(self):
        image = np.full((300, 400, 3), 120, np.uint8)
        box = np.array([[30, 180], [200, 80], [216, 107], [46, 207]], np.float32)
        self.assertIsNone(photo_observations.repeated_lines(image, [(box, 'SCENE SIGN', .99)]*6))
        box = np.array([[30, 80], [300, 80], [300, 110], [30, 110]], np.float32)
        self.assertIsNone(photo_observations.repeated_lines(image, [(box, 'SCENE CAPTION', .99)]*6))

    def test_dot_style_and_ocr_disabled_do_not_enter_photo_lettering(self):
        image = np.full((150, 240, 3), (100, 140, 180), np.uint8)
        with patch('wm.photo_observations.observations', side_effect=AssertionError('wrong style')):
            masks.estimate_single_page(image, use_ocr=False)
            masks.estimate_single_page(image, style='dot_grid')


if __name__ == '__main__':
    unittest.main()
