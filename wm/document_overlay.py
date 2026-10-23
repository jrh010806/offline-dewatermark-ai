"""Single-page coloured stamp matting on predominantly neutral white paper.

Covered paper identifies the tint and stroke geometry. Chroma supplies coverage
where body ink hides the paper; a separately observed covered-ink pole completes
the compositing fit. Reference images and watermark text are never consulted.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource


def estimate(img: np.ndarray, keepout: np.ndarray | None = None) -> MaskLayer | None:
    pixels = img.astype(np.float32)
    gray = pixels.mean(2)
    paper_rgb = np.percentile(pixels.reshape(-1, 3), 97, axis=0).astype(np.float32)
    paper = float(paper_rgb.mean())
    if paper < 220 or np.mean(gray > .92*paper) < .65:
        return None
    # Subtract the paper's own tint rather than assuming all white scans are RGB
    # neutral. A neutral watermark cannot be identified by this chromatic path.
    chroma = pixels-gray[:, :, None]-(paper_rgb-paper)
    energy = np.linalg.norm(chroma, axis=2)
    eligible = (energy > 3) & (gray > .35*paper) & (gray < .88*paper)
    if eligible.sum() < 100:
        return None
    bins = np.arange(0, 258, 2)
    histogram, _ = np.histogram(gray[eligible], bins=bins)
    if histogram.max() < 50 or histogram.max() < .035*eligible.sum():
        return None
    covered = float(bins[histogram.argmax()]+1)
    plateau = eligible & (np.abs(gray-covered) < 3)
    covered_rgb = np.median(pixels[plateau], axis=0)
    tint = covered_rgb-covered_rgb.mean()-(paper_rgb-paper)
    tint_energy = float(tint@tint)
    if tint_energy < 16:
        return None
    band = max(8., .13*(paper-covered))
    seed = ((np.abs(gray-covered) < band) &
            (np.linalg.norm(chroma-tint, axis=2) < 5)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    keep = np.zeros(count, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 25
    seed = keep[labels]
    yy, xx = np.nonzero(seed)
    if keep.sum() < 3 or xx.size < 150:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(np.c_[xx, yy].astype(np.float32)))
    edges = np.roll(box, -1, axis=0)-box
    lengths = np.linalg.norm(edges, axis=1)
    vector = edges[lengths.argmax()]
    angle = abs(float(np.degrees(np.arctan2(vector[1], vector[0])))) % 180
    angle = min(angle, 180-angle)
    long, short = float(lengths.max()), float(lengths.min())
    if not 15 < angle < 65 or long < .3*min(gray.shape) or long < 2.5*short:
        return None
    # Only a thin, separated collection of strokes can be a word-like stamp.
    if xx.size/(long*max(short, 1)) > .6:
        return None
    shape = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    region = cv2.dilate(shape, np.ones((7, 7), np.uint8)) > 0
    if region.mean() > .2:
        return None
    beta_chroma = np.clip(np.sum(chroma*tint, axis=2)/tint_energy, 0, 1)
    if keepout is None:
        from .ocr import text_zones
        keepout = text_zones(img)
    if keepout is None or not keepout.any():
        return None
    ink = gray[region & keepout & (beta_chroma > .7) &
               (gray > .04*paper) & (gray < .7*covered)]
    if ink.size < 20:
        return None
    covered_ink = float(np.percentile(ink, 10))
    transmittance = (covered-covered_ink)/paper
    if not .15 < transmittance < .85:
        return None
    offset = covered_rgb-transmittance*paper_rgb
    if np.any(offset < 0) or np.any(offset > 255*(1-transmittance)):
        return None
    measured = np.clip((paper-gray)/max(paper-covered, 1), 0, 1)
    alpha = np.where(gray < .9*covered, beta_chroma, measured)*region
    return MaskLayer(alpha=alpha.astype(np.float32), source=MaskSource.PRIOR,
                     confidence=.85, veil_strength=np.full(3, transmittance, np.float32),
                     veil_offset=offset.astype(np.float32),
                     paper=np.broadcast_to(paper_rgb, img.shape).copy(),
                     footprint=region, repair_mode='coloured_document_overlay',
                     method='single_coloured_document_overlay', text_keepout=keepout)
