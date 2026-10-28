"""Invert faint repeated coloured overlays using only the input photograph.

The existing detector must confirm repetition first. Smooth source neighbours
fit a colour/transmission model; independent held-out pixels check that fit.
Repeated patches supply stroke geometry over scene edges. Neither clean images,
OCR text strings nor stored watermark templates are used here.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource


def _fit_model(rgb, background, core):
    x, y = background[core], rgb[core]
    if len(x) < 500:
        return None
    holdout = np.arange(len(x)) % 5 == 0
    keep = ~holdout
    for _ in range(5):
        if keep.sum() < 200:
            return None
        xc, yc = x[keep]-x[keep].mean(0), y[keep]-y[keep].mean(0)
        if np.mean(xc**2) < 16:
            return None
        transmission = float(np.sum(xc*yc)/np.sum(xc**2))
        offset = np.median(y[keep]-transmission*x[keep], axis=0)
        errors = np.abs(y-transmission*x-offset).mean(1)
        keep = ~holdout & (errors < max(1., np.percentile(errors[~holdout], 70)))
    # Dark/opaque overlays remain on the established repair path. Their inverse
    # amplifies texture and model-fit errors; a larger mask cannot fix that.
    if not .70 < transmission < .96:
        return None
    colour = offset/(1-transmission)
    if not np.isfinite(colour).all() or np.any(colour < 0) or np.any(colour > 255):
        return None
    validation = errors[holdout]
    if np.median(validation) > 3 or np.mean(validation < 5) < .8:
        return None
    return transmission, offset.astype(np.float32)


def _peaks(scores, size, threshold, limit):
    found = []
    for _ in range(limit):
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < threshold:
            break
        found.append((x, y))
        scores[max(0, y-size//2):y+size//2,
               max(0, x-size//2):x+size//2] = -1
    return found


def estimate(image, verified_region):
    """Refine an independently verified repeated mask, or decline safely."""
    h, w = image.shape[:2]
    if min(h, w) < 120 or not .01 < verified_region.mean() < .4:
        return None
    rgb = image.astype(np.float32)
    background = cv2.medianBlur(image, 15).astype(np.float32)
    delta = rgb-background
    opponent = delta-(delta.sum(2, keepdims=True)-delta)/2
    channel = int(np.argmax([np.percentile(opponent[..., c][verified_region], 90)
                             for c in range(3)]))
    signal = np.maximum(opponent[..., channel], 0)
    seed = verified_region & (signal > max(3, np.percentile(signal[verified_region], 60)))
    gradient = np.max(np.abs(background-cv2.GaussianBlur(background, (0, 0), 2)), 2)
    core = (cv2.erode(seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (gradient < 3)
    fit = _fit_model(rgb, background, core)
    if fit is None:
        return None
    transmission, offset = fit
    direction = offset-(1-transmission)*background
    contrast = direction[..., channel]-(direction.sum(2)-direction[..., channel])/2
    beta = np.clip(signal/np.maximum(contrast, 2), 0, 2.5)
    feature = np.clip(signal/10, 0, 1).astype(np.float32)
    size = min(80, min(h, w)//3)
    step = size//2
    candidates = []
    for y in range(0, h-size, step):
        for x in range(0, w-size, step):
            region = np.s_[y:y+size, x:x+size]
            positive = feature[region] > .3
            if .07 < positive.mean() < .4 and verified_region[region][positive].mean() > .8:
                candidates.append((float(gradient[region][positive].mean()),
                                   -float(positive.mean()), x, y))
    accumulated = np.zeros((h, w), np.float32)
    weight = np.zeros_like(accumulated)
    padded = cv2.copyMakeBorder(feature, size, size, size, size, cv2.BORDER_CONSTANT)
    accepted = 0
    for _, _, x, y in sorted(candidates)[:160]:
        patch = feature[y:y+size, x:x+size]
        scores = cv2.matchTemplate(feature, patch, cv2.TM_CCOEFF_NORMED)
        matches = _peaks(scores, size, .55, 60)
        if len(matches) < 4:
            continue
        centres = np.array(matches)
        if np.ptp(centres[:, 0]) < size or np.ptp(centres[:, 1]) < size:
            continue
        stack = np.stack([beta[py:py+size, px:px+size] for px, py in matches])
        template = np.median(stack, axis=0)
        template[np.mean(stack > .15, axis=0) < .6] = 0
        template = np.clip(template, 0, 2)
        if not .03 < np.mean(template > .05) < .5:
            continue
        accepted += 1
        # Clipped border copies can extend an already verified full template;
        # they never contribute zeros or guessed pixels to the learned shape.
        edge_scores = cv2.matchTemplate(padded, patch, cv2.TM_CCOEFF_NORMED)
        edge_scores[size:h+1, size:w+1] = -1
        for ex, ey in _peaks(edge_scores, size, .60, 20):
            px, py = ex-size, ey-size
            if (min(w, px+size)-max(0, px) >= size*.4 and
                    min(h, py+size)-max(0, py) >= size*.4):
                matches.append((px, py))
        for px, py in matches:
            left, top = max(0, px), max(0, py)
            right, bottom = min(w, px+size), min(h, py+size)
            local = template[top-py:bottom-py, left-px:right-px]
            region = np.s_[top:bottom, left:right]
            reliable = (gradient[region] < 3) & (local > .3)
            scale = (float(np.median(beta[region][reliable]/local[reliable]))
                     if reliable.sum() > 50 else 1.)
            accumulated[region] += local*np.clip(scale, .5, 2.)
            weight[region] += local > .05
    if accepted < 4:
        return None
    coverage = accumulated/np.maximum(weight, 1)
    reliable = ((gradient < 3) & (contrast > 2) &
                (np.linalg.norm(delta-beta[..., None]*direction, axis=2) < 3))
    local = reliable & ((coverage > .05) | verified_region)
    coverage[local] = beta[local]
    coverage[coverage < .05] = 0
    # Keep numerical opacity separate from the selectable binary repair region:
    # faint antialiasing below the UI mask threshold must not disappear there.
    opacity = np.clip(coverage*(1-transmission), 0, .6).astype(np.float32)
    footprint = opacity > 0
    if not .01 < footprint.mean() < .4:
        return None
    return MaskLayer(alpha=footprint.astype(np.float32), footprint=footprint,
                     overlay_opacity=opacity, confidence=.8, source=MaskSource.PRIOR,
                     veil_strength=np.zeros(3, np.float32),
                     veil_offset=(offset/(1-transmission)).astype(np.float32),
                     repair_mode='photo_repeat_inverse', method='photo_repeated_colour_inverse')
