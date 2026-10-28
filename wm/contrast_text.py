"""Source-only contrast views for lettering missed by the conservative cascade.

OCR provides geometry, never a fixed phrase. Masks contain observed strokes,
not whole text rectangles. This cleanup prioritizes removal; small overlaps
with legitimate content can be lost and the output still requires review.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ocr
from .contract import MaskLayer, MaskSource


def _domain(shape, box, pad=0):
    region = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(region, np.rint(box).astype(np.int32), 1)
    if pad:
        region = cv2.dilate(region, np.ones((2*pad+1, 2*pad+1), np.uint8))
    return region > 0


def _layer(region, method, mode, confidence, paper=None, covered=None):
    return MaskLayer(region.astype(np.float32), source=MaskSource.PRIOR,
                     footprint=region, confidence=confidence, method=method,
                     repair_mode=mode, paper=paper,
                     veil_strength=np.ones(3, np.float32),
                     veil_offset=np.zeros(3, np.float32) if covered is None else covered)


def dark_oblique(image, lines):
    """Large dark oblique type, bounded by recognized source geometry."""
    h, w = image.shape[:2]
    gray = image.astype(np.float32).mean(2)
    neutral = image.max(2).astype(float)-image.min(2) < 45
    region = np.zeros((h, w), bool)
    scores = []
    for box, text, score in lines:
        height, length, tilt = ocr._geometry(box)
        if not (score >= .70 and len(text.strip()) >= 4 and
                height >= .04*min(h, w) and length >= 3*height and 12 <= tilt <= 40):
            continue
        domain = _domain((h, w), box, 2)
        strokes = domain & neutral & (gray < 105)
        if not .035 < strokes.sum()/max(1, domain.sum()) < .65:
            continue
        # Include antialiasing around detected dark strokes, not the entire box.
        strokes = cv2.dilate(strokes.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        region |= strokes & domain
        scores.append(score)
    if scores and .0005 < region.mean() < .12:
        return _layer(region, 'contrast_dark_oblique', 'inpaint', min(scores))
    return None


def bright_repeated(image):
    """Reveal bright lettering on photographed gray paper, with repeat evidence."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Fill thin dark printed strokes for detection only. Remove larger bright
    # strokes to estimate the shaded paper; neither operation edits the source.
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    paper_gray = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((41, 41), np.uint8))
    residual = closed.astype(float)-paper_gray
    feature = np.uint8(255-np.clip(residual*6, 0, 255))
    observations = ocr.read_lines(np.repeat(feature[..., None], 3, axis=2))
    candidates = []
    for box, text, score in observations or []:
        height, length, tilt = ocr._geometry(box)
        if (score >= .65 and len(text.strip()) >= 3 and
                height >= .035*min(gray.shape) and length >= 2.5*height and
                12 <= tilt <= 40):
            candidates.append((box, score, tilt, height))
    if len(candidates) < 2:
        return None
    # Two spatially distinct lines with consistent direction/size are required.
    median_tilt = float(np.median([v[2] for v in candidates]))
    median_height = float(np.median([v[3] for v in candidates]))
    candidates = [v for v in candidates if abs(v[2]-median_tilt) <= 6 and
                  .6*median_height <= v[3] <= 1.5*median_height]
    distinct = []
    for item in candidates:
        if all(np.linalg.norm(item[0].mean(0)-other[0].mean(0)) > median_height
               for other in distinct):
            distinct.append(item)
    if len(distinct) < 2:
        return None
    domains = []
    for box, _, _, height in distinct:
        # OCR may read only the middle of a repeated word or a clipped edge
        # fragment. Extend along the observed reading direction by one glyph;
        # only measured bright strokes inside this search area are adopted.
        direction = box[1]-box[0]
        direction /= max(float(np.linalg.norm(direction)), 1.)
        extended = box.copy()
        extended[[0,3]] -= direction*height
        extended[[1,2]] += direction*height
        domains.append(_domain(gray.shape, extended, 4))
    domain = np.logical_or.reduce(domains)
    strokes = domain & (residual > 5)
    strokes = cv2.dilate(strokes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    strokes &= domain
    if not .005 < strokes.mean() < .18:
        return None
    # Estimate shaded paper from actual neighboring paper after excluding the
    # located white lettering and thin dark print. A median alone still carries
    # large, faint copies of dense watermark glyphs into the output.
    size = max(31, int(1.5*median_height) | 1)
    size = min(size, (min(gray.shape)//2)*2-1)
    coarse = cv2.medianBlur(gray, size)
    print_strokes = (closed.astype(float)-gray > 6) & (gray < coarse-5)
    # Exclude the whole OCR search area from the *background estimate* so a
    # missed white fringe cannot contaminate the paper used to fill its neighbor.
    # Actual output permission remains the narrower observed-stroke mask.
    missing_paper = cv2.dilate((domain|print_strokes).astype(np.uint8),np.ones((3,3),np.uint8))
    background = cv2.inpaint(gray,missing_paper*255,5,cv2.INPAINT_TELEA)
    background = cv2.GaussianBlur(background, (0,0), 3)
    paper = np.repeat(background[..., None], 3, axis=2).astype(np.float32)
    covered = np.full(3, float(np.median(gray[strokes])), np.float32)
    return _layer(strokes, 'contrast_bright_repeated', 'bright_paper_cleanup',
                  min(v[1] for v in distinct), paper, covered)


def bright_repair_region(image, mask):
    """Repair bright paper strokes, retaining independently darker printed ink."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = mask.paper.mean(2)
    ink = gray < background-8
    ink = cv2.dilate(ink.astype(np.uint8),np.ones((3,3),np.uint8)) > 0
    ink &= gray < background-3
    colored = image.max(2).astype(float)-image.min(2) > 18
    return mask.footprint & ~ink & ~colored


def restore_bright(image, mask, thresholds=None):
    """Use the existing inpainter only on bright strokes, not body glyphs.

    Estimated paper is protection evidence, never pasted into output: direct
    paper fills left a visible watermark-shaped texture/brightness boundary.
    Source printed glyphs retain their observed contrast in this first cleanup.
    """
    from .lama import inpaint
    region = bright_repair_region(image, mask)
    return inpaint(image, region, thresholds)


def faint_chromatic(image, source_lines):
    """Expose faint colored type using opponent-color contrast, not its wording."""
    h, w = image.shape[:2]
    rgb = image.astype(float)
    region = np.zeros((h, w), bool)
    scores = []
    for excluded in range(3):
        others = [i for i in range(3) if i != excluded]
        opponent = np.uint8(np.clip((rgb[..., others[0]]+rgb[..., others[1]])/2-rgb[..., excluded], 0, 255))
        feature = cv2.morphologyEx(opponent, cv2.MORPH_TOPHAT, np.ones((17, 17), np.uint8))
        view = np.uint8(255-np.clip(feature.astype(float)*10, 0, 255))
        for box, text, score in ocr.read_lines(np.repeat(view[..., None], 3, axis=2)) or []:
            height, length, tilt = ocr._geometry(box)
            if not (score >= .80 and len(text.strip()) >= 4 and
                    10 <= height <= .05*min(h, w) and length >= 4*height and tilt < 10):
                continue
            # Ordinary readable signs/captions do not become watermarks merely
            # because a contrast view also detects them.
            if any(s >= .65 and np.linalg.norm(b.mean(0)-box.mean(0)) < height
                   for b, _, s in source_lines):
                continue
            domain = _domain((h, w), box)
            strokes = domain & (feature > 3)
            if not .04 < strokes.sum()/max(1, domain.sum()) < .55:
                continue
            strokes = cv2.dilate(strokes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            # Low-contrast is measured on the original image, never on the
            # amplified OCR view. Strong legitimate colored lettering is excluded.
            proxy = cv2.inpaint(image, strokes.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
            delta = np.abs(image.astype(float)-proxy).mean(2)
            if strokes.sum() < 100 or np.median(delta[strokes]) > 15:
                continue
            region |= strokes
            scores.append(score)
    if scores and region.mean() < .06:
        return _layer(region, 'contrast_faint_chromatic', 'inpaint', min(scores))
    return None


def estimate(image, *, allow_dark=True, allow_bright=True, allow_photo=True):
    if min(image.shape[:2]) < 256:
        return None
    source_lines = ocr.read_lines(image)
    if source_lines is None:
        return None
    if allow_dark:
        dark = dark_oblique(image, source_lines)
        if dark is not None:
            return dark
    neutral = float(np.mean(image.max(2).astype(float)-image.min(2) < 18))
    if allow_bright and neutral > .75 and np.mean(image.mean(2) > 120) > .65:
        bright = bright_repeated(image)
        if bright is not None:
            return bright
    if allow_photo and neutral < .65:
        return faint_chromatic(image, source_lines)
    return None
