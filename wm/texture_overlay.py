"""Measured translucent lettering on a verified repeating background.

OCR bounds the search; it does not authorize erasure. Background periods and
colours come only from uncovered source pixels. Held-out repeats validate the
prediction before it can be used to fit a veil. Repair inverts source pixels,
never paints an OCR rectangle or rotates the output image.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource
from .photo_text import _fit


def _period(gray: np.ndarray, excluded: np.ndarray, axis: int) -> int | None:
    data = gray if axis == 1 else gray.T
    valid = (~excluded if axis == 1 else ~excluded.T)[::4]
    data = data[::4]
    scores = []
    for shift in range(8, min(256, data.shape[1]//4)+1):
        pairs = valid[:, shift:] & valid[:, :-shift]
        if pairs.sum() < 1000:
            continue
        error = np.minimum(np.abs(data[:, shift:]-data[:, :-shift]), 40)
        scores.append((float(error[pairs].mean()), shift))
    if not scores:
        return None
    best = min(score for score, _ in scores)
    if best > 10 or best > .45*np.median([s for s, _ in scores]):
        return None
    # Prefer the fundamental period over equally good multiples.
    return min(shift for score, shift in scores if score <= best+.15)


def _background(image: np.ndarray, excluded: np.ndarray) -> np.ndarray | None:
    h, w = image.shape[:2]
    gray = image.astype(np.float32).mean(2)
    contrast = float(np.percentile(gray[~excluded], 95)-np.percentile(gray[~excluded], 5))
    if contrast < 30:
        return None
    py, px = _period(gray, excluded, 0), _period(gray, excluded, 1)
    if py is None or px is None:
        return None
    ny, nx = (h+py-1)//py, (w+px-1)//px
    yy, xx = np.indices((h, w))
    holdout = ((yy//py + xx//px) % 3 == 0) & ~excluded
    samples = np.full((ny*py, nx*px, 3), np.nan, np.float32)
    samples[:h, :w] = image
    samples[:h, :w][excluded | holdout] = np.nan
    samples = samples.reshape(ny, py, nx, px, 3).transpose(1, 3, 0, 2, 4)
    samples = samples.reshape(py, px, -1, 3)
    if np.min(np.isfinite(samples[..., 0]).sum(2)) < 4:
        return None
    tile = np.nanmedian(samples, axis=2)
    background = np.tile(tile, (ny, nx, 1))[:h, :w]
    # Validation repeats were never used to estimate their predicted colours.
    error = np.abs(image.astype(np.float32)-background).mean(2)[holdout]
    if error.size < 1000 or np.percentile(error, 90) > max(8, .12*contrast):
        return None
    return background


def estimate(image: np.ndarray, lines: list) -> MaskLayer | None:
    if not lines:
        return None
    from .ocr import _geometry
    h, w = image.shape[:2]
    domain = np.zeros((h, w), np.uint8)
    heights = []
    for box, _, _ in lines:
        cv2.fillConvexPoly(domain, np.rint(box).astype(np.int32), 1)
        heights.append(_geometry(box)[0])
    pad = max(3, min(12, int(np.median(heights)*.06)))
    domain = cv2.dilate(domain, np.ones((2*pad+1, 2*pad+1), np.uint8)) > 0
    if not .005 < domain.mean() < .25:
        return None
    background = _background(image, domain)
    if background is None:
        return None
    delta = image.astype(np.float32)-background
    signal = np.linalg.norm(cv2.GaussianBlur(delta, (0, 0), 1.2), axis=2)
    noise = float(np.percentile(signal[~domain], 99))
    seed = (signal > max(6, 1.2*noise)) & domain
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(6, .15*np.median(heights))
    seed = keep[labels] > 0
    if keep.sum() < 3 or not .02 < seed.sum()/domain.sum() < .5:
        return None
    veil = _fit(image, background, seed, min_strength=.15)
    if veil is None:
        return None
    transmission, offset = veil
    direction = offset-(1-transmission)*background
    raw = np.sum(delta*direction, axis=2)/np.maximum(np.sum(direction**2, axis=2), 1)
    # Deperiodize first: smoothing the original image would blur checker edges.
    alpha = np.clip(cv2.GaussianBlur(raw, (0, 0), .7), 0, 1)
    alpha[alpha > .8] = 1
    region = (cv2.dilate(seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & domain
    alpha[~region] = 0
    return MaskLayer(alpha.astype(np.float32), source=MaskSource.PRIOR,
                     confidence=.85, veil_strength=transmission, veil_offset=offset,
                     paper=background, footprint=region, repair_mode='texture_overlay',
                     method='single_periodic_texture_text')
