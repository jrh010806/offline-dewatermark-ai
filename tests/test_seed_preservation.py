import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import recall, repeated
from wm.contract import MaskLayer, Thresholds


class SeedPreservationTests(unittest.TestCase):
    def test_faint_hue_does_not_drop_supported_neutral_antialias_edges(self):
        ink = np.zeros((220, 440), np.uint8)
        cv2.putText(ink, 'O 8 B', (25, 155), cv2.FONT_HERSHEY_SIMPLEX,
                    2.8, 255, 5, cv2.LINE_AA)
        alpha = ink.astype(np.float32) / 255
        source = np.uint8(220 * (1-alpha[..., None])
                          + np.array([64, 60, 60]) * alpha[..., None])
        domain = np.zeros(ink.shape, bool)
        domain[65:175, 15:420] = True
        seed = ink > 0
        old = recall._pixel_strokes(source, domain, seed, 85)
        repeated_core = (ink > 0) & (np.indices(ink.shape)[1] < 330)
        candidate = recall._pixel_strokes(source, domain, seed, 85, preserve_seeds=True,
                                          repeat_support=repeated_core)
        self.assertTrue(candidate[old].all())
        self.assertGreater(int((candidate & seed).sum()), int((old & seed).sum()))
        # Newly retained pixels must be old source-supported strokes; no new
        # whitespace, letter holes or search-box background is authorized.
        self.assertFalse((candidate & ~old & ~seed).any())
        self.assertFalse((candidate & ~old & ~repeated_core).any())
        self.assertFalse(candidate[~domain].any())

    def test_missing_repeat_evidence_cannot_retain_a_luminance_seed(self):
        source = np.full((120, 200, 3), 220, np.uint8)
        cv2.putText(source, 'O8', (25, 85), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (64, 60, 60), 4, cv2.LINE_AA)
        seed = source.mean(2) < 210
        domain = np.ones(seed.shape, bool)
        old = recall._pixel_strokes(source, domain, seed, 60)
        for evidence in (None, np.zeros_like(seed)):
            result = recall._pixel_strokes(source, domain, seed, 60,
                                             preserve_seeds=True, repeat_support=evidence)
            np.testing.assert_array_equal(old, result)

    def test_uniform_source_rejects_even_nonempty_seed_and_search_box(self):
        source = np.full((120, 200, 3), 180, np.uint8)
        seed = np.zeros((120, 200), bool)
        seed[40:60, 55:85] = True
        domain = np.zeros_like(seed)
        domain[20:100, 20:180] = True
        candidate = recall._pixel_strokes(source, domain, seed, 40, preserve_seeds=True,
                                          repeat_support=np.ones_like(seed))
        self.assertFalse(candidate.any())

    def test_registered_support_uses_native_strokes_not_padded_footprint(self):
        h, w = 750, 600
        image = np.full((h, w, 3), 255, np.uint8)
        boxes = []
        text = 'PRIVATE 2035'
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        for y in (220, 470):
            cv2.putText(image, text, (150, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (100, 100, 100), 2, cv2.LINE_AA)
            boxes.append(np.array([[148, y-size[1]-3], [153+size[0], y-size[1]-3],
                                   [153+size[0], y+5], [148, y+5]], np.float32))
        turn = cv2.getRotationMatrix2D((w/2, h/2), -28, 1)
        image = cv2.warpAffine(image, turn, (w, h), borderValue=(255, 255, 255))
        source_boxes = [cv2.transform(b[None], turn)[0] for b in boxes]
        undo = cv2.getRotationMatrix2D((w/2, h/2), 28, 1)
        rw = int(abs(undo[0, 0])*w + abs(undo[0, 1])*h)
        rh = int(abs(undo[0, 0])*h + abs(undo[0, 1])*w)
        undo[:, 2] += [(rw-w)/2, (rh-h)/2]
        upright = [cv2.transform(b[None], undo)[0] for b in source_boxes]
        observations = [[(b, text, .95) for b in source_boxes],
                        [(b, text, .95) for b in upright]]
        with patch('wm.repeated.ocr.read_lines', side_effect=observations), \
             patch('wm.repeated.ocr.text_zones', return_value=np.zeros((h, w), bool)):
            result = repeated.estimate(image)
        support = result.repeat_stroke_support
        self.assertIsNotNone(support)
        self.assertTrue(support.any())
        self.assertFalse((support & ~result.footprint).any())
        self.assertFalse(support[image.mean(2) > 235].any())
        self.assertLess(int(support.sum()), int(result.footprint.sum()))

    def test_candidate_does_not_replace_measured_physical_models(self):
        source = np.full((40, 60, 3), 180, np.uint8)
        alpha = np.zeros((40, 60), np.float32)
        alpha[12:25, 20:35] = .7
        for mode in ('texture_overlay', 'photo_repeat_inverse',
                     'coloured_document_overlay', 'document_overlay_full',
                     'periodic_paper_overlay'):
            with self.subTest(mode=mode):
                mask = MaskLayer(alpha, footprint=alpha > 0, repair_mode=mode,
                                 veil_strength=np.full(3, .7), veil_offset=np.full(3, 20.))
                with patch('wm.recall._lines', side_effect=AssertionError('must bypass search')):
                    result = recall.expand(source, mask, Thresholds(experimental_seed_preservation=True))
                self.assertIs(result.alpha, mask.alpha)
                self.assertIs(result.veil_strength, mask.veil_strength)
                self.assertEqual(result.repair_mode, mode)

    def test_candidate_is_opt_in_and_empty_seed_does_not_start_detection(self):
        self.assertFalse(Thresholds().experimental_seed_preservation)
        source = np.full((40, 60, 3), 180, np.uint8)
        mask = MaskLayer(np.zeros((40, 60), np.float32), repair_mode='inpaint')
        with patch('wm.recall._lines', side_effect=AssertionError('no original evidence')):
            self.assertIs(recall.expand(source, mask, Thresholds(experimental_seed_preservation=True)), mask)
