"""Recover a small translucent lattice from repetitions within one photograph.

The lattice, its coverage and its colour are fitted from the input. No sample
names, reference masks, clean images or fixed dot coordinates are used.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource


def _regress(background, observed, period, iterations=0):
    """Robust affine fit for each lattice phase, with balanced spatial samples."""
    yy, xx = np.indices(background.shape)
    phase = (yy % period * period + xx % period).ravel()
    x, y = background.ravel().astype(float), observed.ravel().astype(float)
    if x.size > 200_000:
        chosen = np.random.default_rng(0).choice(x.size, 200_000, replace=False)
        phase, x, y = phase[chosen], x[chosen], y[chosen]
    weight = np.ones_like(x)
    for _ in range(iterations + 1):
        n = np.maximum(np.bincount(phase, weight, minlength=period**2), 1)
        sx = np.bincount(phase, x*weight, minlength=period**2)
        sy = np.bincount(phase, y*weight, minlength=period**2)
        sxx = np.bincount(phase, x*x*weight, minlength=period**2)
        sxy = np.bincount(phase, x*y*weight, minlength=period**2)
        slope = (sxy-sx*sy/n) / np.maximum(sxx-sx*sx/n, 1)
        offset = (sy-slope*sx)/n
        error = y-(slope[phase]*x+offset[phase])
        weight = np.minimum(1, 5/np.maximum(np.abs(error), 1e-3))
    return slope.reshape(period, period), offset.reshape(period, period)


def estimate(img: np.ndarray) -> MaskLayer | None:
    h, w = img.shape[:2]
    if min(h, w) < 90:
        return None
    # Cropping preserves an unknown lattice pitch; resizing would change it.
    crop = img[:640, :640].astype(np.float32).mean(2)
    background = cv2.GaussianBlur(crop, (0, 0), 3)
    if float(background.std()) < 18:
        return None
    candidates = []
    for period in range(7, min(65, min(crop.shape)//5)):
        slope, _ = _regress(background, crop-background, period)
        candidates.append((-float(np.percentile(slope, 1)), period))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    if best < .18:
        return None
    # Multiples of a real period have the same response. Prefer its fundamental.
    period = min(p for score, p in candidates if score >= .9*best)
    slope, _ = _regress(background, crop-background, period, 3)
    support = slope < -.18
    if not .005 < support.mean() < .15:
        return None
    # The same attenuation pattern must recur in separate parts of the image.
    half = (crop.shape[0]//2//period)*period
    first, _ = _regress(background[:half], crop[:half]-background[:half], period)
    second, _ = _regress(background[half:], crop[half:]-background[half:], period)
    if (np.median(first[support]) > -.15 or np.median(second[support]) > -.15
            or np.corrcoef(first.ravel(), second.ravel())[0, 1] < .85):
        return None
    # A dot is compact, not an entire horizontal/vertical image structure.
    tiled = np.tile(support.astype(np.uint8), (3, 3))
    count, _, stats, _ = cv2.connectedComponentsWithStats(tiled, 8)
    if count < 2 or np.max(stats[1:, cv2.CC_STAT_WIDTH]) >= .65*period \
            or np.max(stats[1:, cv2.CC_STAT_HEIGHT]) >= .65*period:
        return None
    yy, xx = np.indices((h, w))
    footprint = support[yy % period, xx % period]
    # The interpolated background is used only to fit the overlay. The restored
    # luminance comes from the original pixels through its inverse, not a blur.
    predicted = cv2.inpaint(img, footprint.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
    trans, _ = _regress(predicted.astype(np.float32).mean(2),
                       img.astype(np.float32).mean(2), period, 5)
    phase_alpha = np.clip(1-trans, 0, .85)*support
    if not .15 < np.median(phase_alpha[support]) < .8:
        return None
    colour = []
    for channel in range(3):
        s, t = _regress(predicted[:, :, channel], img[:, :, channel], period, 5)
        values = t[support]/np.maximum(1-s[support], .1)
        if np.std(values) > 22:
            return None
        colour.append(float(np.median(values)))
    if min(colour) < 0 or max(colour) > 255:
        return None
    return MaskLayer(alpha=phase_alpha[yy % period, xx % period].astype(np.float32),
                     source=MaskSource.PRIOR, confidence=.9,
                     veil_strength=np.zeros(3, np.float32),
                     veil_offset=np.array(colour, np.float32),
                     paper=np.full(img.shape, 255, np.float32),
                     footprint=footprint, lattice=(period, period),
                     repair_mode="periodic_overlay", method="single_periodic_overlay")
