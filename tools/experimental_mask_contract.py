"""Isolated candidate: retain source-supported neutral dark stroke seeds.

No application default or model changes. This narrowly rechecks seeds from the
existing dark-oblique detector; it does not union an old padded footprint, fill
OCR boxes, invent repeats, or change a measured physical inverse.
"""
from contextlib import contextmanager
from unittest.mock import patch

import cv2
import numpy as np


@contextmanager
def enabled(observations, *, edge_support=False):
    from wm import recall
    original_expand = recall.expand
    original_pixels = recall._pixel_strokes

    def expand(image, mask, thr, *, use_ocr=True, style=None):
        if (not use_ocr or not thr.watermark_recall_priority
                or mask.method != 'contrast_dark_oblique' or mask.repair_mode != 'inpaint'):
            return original_expand(image, mask, thr, use_ocr=use_ocr, style=style)

        def pixels(source, domain, seed_region, height, **kwargs):
            region = original_pixels(source, domain, seed_region, height, **kwargs)
            smooth = cv2.GaussianBlur(source.astype(np.float32), (0, 0), .8)
            size = min(31, max(9, int(.35*height) | 1))
            feature = cv2.morphologyEx(smooth.mean(2), cv2.MORPH_BLACKHAT,
                                     np.ones((size, size), np.uint8))
            luminance, score = recall._grow_strokes(feature, domain, seed_region, height)
            # These native-pixel conditions are the existing detector's dark
            # and neutral observations, not new thresholds fitted to labels.
            native = ((source.astype(np.float32).mean(2) < 105)
                      & (source.max(2).astype(float)-source.min(2) < 45))
            if edge_support:
                # The native dark-core cut omits antialias edges. Recheck the
                # current input in every channel, admitting an edge only when
                # it is darker than its local closing envelope in all three.
                # Morphology measures contrast here; it never supplies pixels.
                native_contrast = cv2.morphologyEx(source.astype(np.float32),
                    cv2.MORPH_BLACKHAT, np.ones((size, size), np.uint8))
                native = native_contrast.min(2) > 1
            retained = seed_region & luminance & native if score >= 3 else np.zeros_like(region)
            result = region | retained
            observations.append(dict(method=mask.method, added_pixels=int((retained & ~region).sum()),
                luminance_score=float(score), seed_pixels=int(seed_region.sum()),
                evidence=('原检测旧种子、原尺寸三通道局部变暗、连通亮度暗笔画共同支持'
                          if edge_support else '原检测旧种子、原图深色中性像素、连通亮度暗笔画共同支持'),
                edge_support=edge_support,
                reference_read=False, experimental=True))
            return result

        with patch.object(recall, '_pixel_strokes', pixels):
            return original_expand(image, mask, thr, use_ocr=use_ocr, style=style)

    with patch.object(recall, 'expand', expand):
        yield
