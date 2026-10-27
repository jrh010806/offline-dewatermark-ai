import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import photo_text, pipeline, masks
from wm.contract import Thresholds, PageImage, MaskLayer
from wm.lama import repair_with_lama


def example(opacity):
    yy, xx = np.indices((320, 500))
    base = 100+35*np.sin(xx/65)+30*np.cos(yy/80)
    clean = np.uint8(np.stack([base+25, base, base-20], 2))
    strokes = np.zeros((320, 500), np.uint8)
    cv2.putText(strokes, "APP NOTE 42", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    alpha = strokes.astype(float)/255*opacity
    source = np.uint8(clean*(1-alpha[..., None])+np.array([220, 230, 240])*alpha[..., None])
    box = np.array([[12, 17], [230, 17], [230, 50], [12, 50]], np.float32)
    return clean, source, [(box, "APP NOTE 42", .95)]


class PhotoTextTests(unittest.TestCase):
    def test_no_ocr_evidence_never_authorises_photo_erasure(self):
        clean, source, lines = example(.45)
        with patch("wm.photo_text.ocr.read_lines", return_value=None):
            self.assertIsNone(photo_text.estimate(source))

    def test_opaque_legitimate_text_is_not_a_translucent_overlay(self):
        clean, source, lines = example(1.)
        with patch("wm.photo_text.ocr.read_lines", return_value=lines):
            self.assertIsNone(photo_text.estimate(source))

    def test_translucent_strokes_are_bounded_and_repair_improves_them(self):
        clean, source, lines = example(.45)
        with patch("wm.photo_text.ocr.read_lines", return_value=lines):
            mask = photo_text.estimate(source)
        self.assertIsNotNone(mask)
        output, used = repair_with_lama(source, mask, Thresholds())
        np.testing.assert_array_equal(output[~mask.footprint], source[~mask.footprint])
        self.assertLess(np.abs(output.astype(float)-clean).mean(),
                        np.abs(source.astype(float)-clean).mean())
        result = pipeline._process_one(PageImage(source, 96), mask, Thresholds())
        self.assertEqual(result.output.image.shape, source.shape)
        self.assertEqual(result.extracted_watermark.shape, (*source.shape[:2], 4))

    def test_style_restriction_and_batch_prior_cannot_override_evidence(self):
        clean, source, lines = example(.45)
        with patch("wm.photo_text.ocr.read_lines", return_value=lines):
            self.assertIsNone(photo_text.estimate(source, allow_corner=False))
            self.assertIsNone(photo_text.estimate(source, require_repeat=True))
            own = photo_text.estimate(source)
        prior = MaskLayer(np.ones(source.shape[:2], np.float32), confidence=1.)
        self.assertIs(masks.with_prior(own, prior), own)
