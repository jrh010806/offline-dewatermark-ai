"""Recover translucent dot grids from phase statistics within one document.

Bright paper supplies the repeated dot profile. Ink/paper quantiles inside and
outside dot cores identify transmission without erasing printed text or seals.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource


def _phase_mean(values, period):
    yy, xx = np.indices(values.shape)
    phase = (yy % period*period+xx % period).ravel()
    count = np.bincount(phase, minlength=period**2)
    return (np.bincount(phase, weights=values.ravel(), minlength=period**2)/
            np.maximum(count, 1)).reshape(period, period)


def estimate(img: np.ndarray) -> MaskLayer | None:
    gray = img.astype(np.float32).mean(2)
    if min(gray.shape) < 120 or np.mean(gray > 200) < .6:
        return None
    crop = gray[:640, :640]
    paper = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    signal = np.clip(paper-crop, 0, 60)
    candidates = []
    for period in range(7, min(65, min(crop.shape)//5)):
        profile = _phase_mean(signal, period)
        score = float(np.percentile(profile, 99)-np.median(profile))
        candidates.append((score, period))
    if not candidates or max(s for s, _ in candidates) < 18:
        return None
    best = max(s for s, _ in candidates)
    period = min(p for s, p in candidates if s >= .9*best)
    half = crop.shape[0]//2//period*period
    upper = _phase_mean(signal[:half], period)
    lower = _phase_mean(signal[half:], period)
    correlation = float(np.corrcoef(upper.ravel(), lower.ravel())[0, 1])
    if not np.isfinite(correlation) or correlation < .9:
        return None
    rows, cols = crop.shape[0]//period, crop.shape[1]//period
    values = np.where(crop > .6*paper, paper-crop, np.nan)
    blocks = values[:rows*period, :cols*period].reshape(rows, period, cols, period)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, period, period)
    # Sparse body letters cannot define a phase; each phase needs bright-paper
    # observations in a majority of cells, not just one repeated table entry.
    if np.min(np.isfinite(blocks).mean(0)) < .5:
        return None
    profile = np.maximum(np.nanpercentile(blocks, 60, axis=0), 0)
    peak = float(np.percentile(profile, 99))
    support = profile > max(4, peak*.15)
    if peak < 15 or not .005 < support.mean() < .15:
        return None
    tiled = np.tile(support.astype(np.uint8), (3, 3))
    count, _, stats, _ = cv2.connectedComponentsWithStats(tiled, 8)
    if count < 2 or np.max(stats[1:, cv2.CC_STAT_WIDTH]) >= .65*period \
            or np.max(stats[1:, cv2.CC_STAT_HEIGHT]) >= .65*period:
        return None
    yy, xx = np.indices(gray.shape)
    footprint = support[yy % period, xx % period]
    core = (profile > .9*peak)[yy % period, xx % period]
    neutral = img.max(2).astype(np.float32)-img.min(2) < 20
    outside, inside = gray[~footprint & neutral], gray[core & neutral]
    if min(outside.size, inside.size) < 500:
        return None
    ink, white = np.percentile(outside, [1, 85])
    covered_ink, covered_white = np.percentile(inside, [1, 85])
    if white-ink < 70:
        return None
    transmission = float((covered_white-covered_ink)/(white-ink))
    if not .15 < transmission < .85:
        return None
    white_rgb = np.percentile(img[~footprint & neutral], 85, axis=0)
    covered_rgb = np.percentile(img[core & neutral], 85, axis=0)
    offset = covered_rgb-transmission*white_rgb
    if np.any(offset < 0) or np.any(offset > 255*(1-transmission)):
        return None
    phase_alpha = np.clip(profile/peak, 0, 1)*support
    return MaskLayer(alpha=phase_alpha[yy % period, xx % period].astype(np.float32),
                     source=MaskSource.PRIOR, confidence=.9,
                     veil_strength=np.full(3, transmission, np.float32),
                     veil_offset=offset.astype(np.float32),
                     paper=np.broadcast_to(white_rgb.astype(np.float32), img.shape).copy(),
                     footprint=footprint, lattice=(period, period),
                     repair_mode='periodic_paper_overlay', method='single_periodic_document')
