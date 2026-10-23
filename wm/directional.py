"""Directional watermark detection using Gabor filters.

Text is horizontal; diagonal stamp watermarks like "青岛红创" are at 30-45°.
Gabor filters tuned to diagonal orientations respond strongly to the watermark
but weakly to horizontal text, solving the tonal ambiguity problem.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource, Thresholds


def _gabor_bank(lambd: float = 12.0, sigma: float = 8.0, gamma: float = 0.5) -> list[np.ndarray]:
    """Gabor kernels at diagonal orientations (30°, 45°, 60°, 120°, 135°, 150°)."""
    kernels = []
    for theta_deg in [30, 45, 60, 120, 135, 150]:
        theta = np.deg2rad(theta_deg)
        ksize = int(6 * sigma) | 1
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    return kernels


def directional_response(gray: np.ndarray, kernels: list[np.ndarray]) -> np.ndarray:
    """Max response across all diagonal Gabor kernels."""
    responses = []
    for k in kernels:
        resp = cv2.filter2D(gray, cv2.CV_32F, k)
        responses.append(np.abs(resp))
    return np.max(responses, axis=0)


def estimate_diagonal_watermark(img: np.ndarray, thr: Thresholds = Thresholds()) -> MaskLayer:
    """Detect diagonal stamp watermarks using Gabor direction filtering.

    Returns a MaskLayer with alpha coverage map and footprint.
    """
    gray = img.astype(np.float32).mean(2)
    h, w = gray.shape

    # Compute directional response
    kernels = _gabor_bank()
    resp = directional_response(gray, kernels)

    # Threshold: keep strong diagonal structures
    resp_mean = resp.mean()
    resp_std = resp.std()
    threshold = resp_mean + 2.0 * resp_std
    binary = (resp > threshold).astype(np.uint8)

    # Morphological cleanup: remove small noise, fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Dilate to cover full watermark area
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    footprint = cv2.dilate(binary, kernel_dilate, iterations=2)

    # Compute coverage as normalized response within footprint
    alpha = np.zeros((h, w), dtype=np.float32)
    mask_fp = footprint > 0
    if mask_fp.any():
        resp_max = resp[mask_fp].max()
        if resp_max > 0:
            alpha[mask_fp] = resp[mask_fp] / resp_max

    # Build MaskLayer
    paper = np.full((h, w, 3), 255.0, dtype=np.float32)
    mask = MaskLayer(
        alpha=alpha,
        source=MaskSource.PRIOR,
        confidence=float(mask_fp.mean()),
        veil_strength=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        veil_offset=np.array([100.0, 100.0, 100.0], dtype=np.float32),
        paper=paper,
    )
    mask.footprint = footprint > 0
    return mask
