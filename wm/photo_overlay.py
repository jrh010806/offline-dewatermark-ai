"""Conservative source-detail recovery inside a located translucent photo stamp."""
from __future__ import annotations

import cv2
import numpy as np


def fit(img: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a shared transmission using nearby source pixels as a predictor.

    Interpolation is a calibration observation, not the final replacement image.
    A fit is declined without texture range, a physical veil, and broad agreement.
    """
    core = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if core.sum() < 150:
        return None
    background = cv2.inpaint(img, region.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)
    x = background[core].astype(np.float32).mean(1)
    y = img[core].astype(np.float32).mean(1)
    if x.std() < 20:
        return None
    s, t = np.polyfit(x, y, 1)
    for _ in range(5):
        error = y-(s*x+t)
        keep = np.abs(error) < np.percentile(np.abs(error), 65)
        if keep.sum() < 50 or x[keep].std() < 15:
            return None
        s, t = np.polyfit(x[keep], y[keep], 1)
    error = y-(s*x+t)
    if not .25 < s < .85 or np.mean(np.abs(error) < 20) < .65:
        return None
    offset = np.median(img[core]-s*background[core], axis=0)
    if np.any(offset < 0) or np.any(offset > 255*(1-s)):
        return None
    return np.full(3, s, np.float32), offset.astype(np.float32)


def refine(img: np.ndarray, model: np.ndarray, region: np.ndarray,
           transmission: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Blend an agreeing source inverse into stroke interiors only.

    Uncertain edges and disagreeing estimates keep the model result. This avoids
    inserting the watermark's own boundary or bright inverse halos as texture.
    The 50-level gate is a conservative consistency bound, not a quality score.
    """
    inverse = (img.astype(np.float32)-offset)/np.maximum(transmission, .25)
    disagreement = np.abs(inverse-model.astype(np.float32)).mean(2)
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(distance-3, 0, 1)*np.clip(1-disagreement/50, 0, 1)*.8
    restored = model.astype(np.float32)+weight[:, :, None]*(inverse-model)
    out = img.copy()
    out[region] = np.clip(restored[region], 0, 255).astype(np.uint8)
    return out
