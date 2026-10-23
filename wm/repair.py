"""Restoration backends. The veil inverse is the primary path because it is
deterministic and cannot hallucinate glyphs; deep-learning inpainting (LaMa)
handles the residual opaque core where the watermark occludes text.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, RepairName, Thresholds
from .lama import repair_with_lama

OPAQUE = 0.97


def alpha_unmix(img: np.ndarray, mask: MaskLayer, coverage: np.ndarray | None = None) -> np.ndarray:
    beta = (mask.alpha if coverage is None else coverage)[..., None].astype(np.float32)
    s = mask.veil_strength.reshape(1, 1, 3)
    t = mask.veil_offset.reshape(1, 1, 3)
    denom = np.maximum(1.0 - beta * (1.0 - s), 0.06)
    return np.clip((img.astype(np.float32) - beta * t) / denom, 0, 255).astype(np.uint8)


def inpaint(img: np.ndarray, binary_mask: np.ndarray, radius: int = 4) -> np.ndarray:
    if not binary_mask.any():
        return img.copy()
    return cv2.inpaint(img, (binary_mask.astype(np.uint8)) * 255, radius, cv2.INPAINT_TELEA)


def repair(img: np.ndarray, mask: MaskLayer,
           thr: Thresholds = Thresholds()) -> tuple[np.ndarray, list[RepairName]]:
    """Halo suppression is applied here rather than only in the mask so that the
    shared cross-page coverage field, which is measured on a content-free median
    and therefore knows nothing about this page's glyph ramps, is still protected
    per page. Leaving a little watermark where it crosses text is the deliberate
    bias: erasing a stroke breaches the tender's third clause outright.

    The opaque core (coverage >= 0.97) is handed to LaMa for deep-learning
    reconstruction — this is where the watermark sits on top of text and
    algebraic un-mixing alone cannot recover the original glyphs.
    """
    return repair_with_lama(img, mask, thr)
