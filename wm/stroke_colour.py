"""Experimental chromatic stroke refinement for bright document backgrounds.

Only source pixels enter this module. A median background estimates the ink's
colour direction, not a clean reference. Smoothed contrast may propose strokes;
native-resolution contrast must independently support every selected pixel.
"""
from __future__ import annotations

import cv2
import numpy as np


def refine(image: np.ndarray, domain: np.ndarray, seed: np.ndarray,
           height: float) -> np.ndarray | None:
    from .masks import paper_background_fraction
    from .recall import _grow_strokes

    # The local colour model assumes bright, mostly flat page stock. It must
    # not replace photo, grain, or measured inverse/texture restoration paths.
    if not seed.any() or paper_background_fraction(image) < .65:
        return None
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .8)
    size = min(31, max(9, int(.45*height) | 1))
    background = cv2.medianBlur(image, size).astype(np.float32)
    delta = smooth-background
    delta -= delta.mean(2, keepdims=True)
    magnitude = np.linalg.norm(delta, axis=2)
    pool = seed & (magnitude > 3)
    if pool.sum() < 30:
        return None
    directions = delta[pool]/magnitude[pool, None]
    direction = np.median(directions, axis=0)
    # Componentwise medians need not sum to zero. Remove luminance explicitly
    # so ordinary dark text is not mistaken for chromatic watermark evidence.
    direction -= direction.mean()
    length = float(np.linalg.norm(direction))
    if length < .5:
        return None
    direction /= length
    if np.mean(directions@direction > .8) < .6:
        return None
    kernel = np.ones((min(31, max(9, int(.35*height) | 1)),)*2, np.uint8)
    projected = smooth@direction
    choices = []
    for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        feature = cv2.morphologyEx(projected, operation, kernel)
        region, score = _grow_strokes(feature, domain, seed, height)
        choices.append((score, region, operation))
    score, region, operation = max(choices, key=lambda item: item[0])
    if score < 3 or not region.any():
        return None
    raw = cv2.morphologyEx(image.astype(np.float32)@direction, operation, kernel)
    control = domain & ~cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if control.sum() < 100:
        return None
    noise = raw[control]
    centre = float(np.median(noise))
    sigma = max(.5, float(np.median(np.abs(noise-centre)))*1.4826)
    # No dilation/closing/filling of this output. A smoothed halo with no
    # native pixel evidence cannot become repair permission.
    return region & (raw > centre+sigma)
