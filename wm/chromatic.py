"""Locate large oblique lettering through an image's own colour residual.

This is a bounded fallback for photographs with a narrow background colour
locus. OCR validates large lettering on an enhanced residual; it never grants
permission to repaint the entire word rectangle.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contract import MaskLayer, MaskSource


def _rotate(shape, angle):
    h, w = shape
    matrix = cv2.getRotationMatrix2D((w/2, h/2), float(angle), 1)
    width = int(w*abs(matrix[0, 0])+h*abs(matrix[0, 1]))
    height = int(h*abs(matrix[0, 0])+w*abs(matrix[0, 1]))
    matrix[:, 2] += [(width-w)/2, (height-h)/2]
    return matrix, (width, height)


def _residual(img):
    red, green, blue = np.moveaxis(img.astype(np.float32), -1, 0)
    design = np.c_[red.ravel(), blue.ravel(), np.ones(red.size)]
    target = green.ravel()
    step = max(1, target.size//100_000)
    x, y = design[::step], target[::step]
    fit = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(8):
        error = y-x@fit
        keep = np.abs(error) < max(1.5, np.percentile(np.abs(error), 70))
        fit = np.linalg.lstsq(x[keep], y[keep], rcond=None)[0]
    return (target-design@fit).reshape(red.shape).astype(np.float32)


def _balanced_directions(smooth, limit=3):
    """Give each residual polarity its own bounded, distinct-angle budget."""
    choices = {}
    for sign in (-1, 1):
        signal = np.clip(sign*smooth-3, 0, 10)
        if np.mean(signal > 2) < .01:
            continue
        ranked = []
        for angle in range(-60, 61, 5):
            if abs(angle) < 15:
                continue
            matrix, size = _rotate(signal.shape, angle)
            projection = cv2.warpAffine(signal, matrix, size).sum(1)
            score = float(np.sum(projection**2)/max(projection.sum(), 1))
            ranked.append((score, angle))
        picked = []
        for _, angle in sorted(ranked, reverse=True):
            if all(abs(angle-other) >= 10 for other in picked):
                picked.append(angle)
            if len(picked) >= limit:
                break
        choices[sign] = picked
    return [(sign, angles[index]) for index in range(limit)
            for sign, angles in choices.items() if index < len(angles)]


def _native_strokes(image, residual, box, sign):
    """A word box bounds the search; raw colour and local contrast grant pixels."""
    from .ocr import _geometry
    from .photo_text import _fit
    height, _, _ = _geometry(box)
    h, w = image.shape[:2]
    pad = max(5, int(.3*height))
    x0, y0 = np.maximum(np.floor(box.min(0)).astype(int)-pad, 0)
    x1, y1 = np.minimum(np.ceil(box.max(0)).astype(int)+pad+1, [w, h])
    crop = image[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 9:
        return None
    domain = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(domain, np.rint(box-[x0, y0]).astype(np.int32), 1)
    field = sign*residual[y0:y1, x0:x1]
    # Median is used only to measure source contrast, never to fill a mask.
    encoded = np.uint8(np.clip(128+field, 0, 255))
    size = max(7, int(.25*height) | 1)
    contrast = encoded.astype(np.float32)-cv2.medianBlur(encoded, size)
    noise = float(np.median(np.abs(contrast[domain == 0]))) if (domain == 0).any() else 0.
    region = (field > 4) & (contrast > max(4., 3*noise)) & (domain > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = ((stats[1:, cv2.CC_STAT_AREA] >= max(12, .3*height)) &
                (np.maximum(stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_WIDTH]) >= .2*height))
    if keep.sum() < 3:
        return None
    region = keep[labels]
    fraction = region.sum()/max(1, domain.sum())
    if not .025 < fraction < .50:
        return None
    background = cv2.inpaint(crop, region.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
    if _fit(crop, background, region, min_strength=.08) is None:
        return None
    full = np.zeros((h, w), bool)
    full[y0:y1, x0:x1] = region
    return full


def estimate_balanced(img: np.ndarray) -> MaskLayer | None:
    """Opt-in missed-direction fallback, without weakening the large-type gate."""
    gray = img.astype(np.float32).mean(2)
    if min(gray.shape) < 120 or np.mean(gray > 235) > .65:
        return None
    if float(np.mean(img.max(2).astype(float)-img.min(2))) < 3:
        return None
    scale = min(1., 900/max(gray.shape))
    work = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    residual = _residual(work)
    smooth = cv2.GaussianBlur(residual, (0, 0), .8)
    if np.mean(np.abs(smooth) < 4) < .65:
        return None
    from .ocr import read_lines, _geometry
    native_residual = None
    for sign, angle in _balanced_directions(smooth):
        enhanced = np.uint8(255-np.clip(sign*residual*20, 0, 255))
        matrix, size = _rotate(enhanced.shape, angle)
        rotated = cv2.warpAffine(enhanced, matrix, size, borderValue=255)
        regions = []
        for box, text, confidence in read_lines(np.repeat(rotated[..., None], 3, axis=2)) or []:
            height, length, tilt = _geometry(box)
            if (float(confidence) < .5 or len(text.strip()) < 3 or tilt > 15
                    or height < .14*min(work.shape[:2]) or length < 2.2*height):
                continue
            native_box = cv2.transform(box[None].astype(np.float32),
                                      cv2.invertAffineTransform(matrix))[0]/scale
            _, _, native_tilt = _geometry(native_box)
            if not 20 <= native_tilt <= 65:
                continue
            if native_residual is None:
                native_residual = _residual(img)
            strokes = _native_strokes(img, native_residual, native_box, sign)
            if strokes is not None:
                regions.append(strokes)
        if regions:
            region = np.logical_or.reduce(regions)
            if .001 < region.mean() < .12:
                return MaskLayer(alpha=region.astype(np.float32), footprint=region,
                    source=MaskSource.PRIOR, confidence=.75, repair_mode='inpaint',
                    method='experimental_balanced_chromatic_strokes',
                    veil_strength=np.ones(3, np.float32), veil_offset=np.zeros(3, np.float32))
    return None


def estimate(img: np.ndarray) -> MaskLayer | None:
    gray = img.astype(np.float32).mean(2)
    if min(gray.shape) < 120 or np.mean(gray > 235) > .65:
        return None
    # Neutral monochrome input contains no chromatic evidence for this method.
    if float(np.mean(img.max(2).astype(float)-img.min(2))) < 3:
        return None
    scale = min(1., 900/max(gray.shape))
    work = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    residual = _residual(work)
    smooth = cv2.GaussianBlur(residual, (0, 0), .8)
    # Reject broadly multicoloured scenes where a plane does not describe most
    # pixels. Such photos need a different detector, not arbitrary colour cuts.
    if np.mean(np.abs(smooth) < 4) < .65:
        return None
    candidates = []
    for sign in (-1, 1):
        signal = np.clip(sign*smooth-3, 0, 10)
        if np.mean(signal > 2) < .01:
            continue
        for angle in range(-60, 61, 5):
            if abs(angle) < 15:
                continue
            matrix, size = _rotate(signal.shape, angle)
            projection = cv2.warpAffine(signal, matrix, size).sum(1)
            score = float(np.sum(projection**2)/max(projection.sum(), 1))
            candidates.append((score, sign, angle))
    if not candidates:
        return None
    from .ocr import read_lines, _geometry
    domain = np.zeros(work.shape[:2], np.uint8)
    chosen_sign = None
    # Angle is measured from the input, not a hard-coded watermark orientation.
    tried = []
    for _, sign, angle in sorted(candidates, reverse=True):
        if any(sign == s and abs(angle-a) < 10 for s, a in tried):
            continue
        tried.append((sign, angle))
        enhanced = np.uint8(255-np.clip(sign*residual*20, 0, 255))
        matrix, size = _rotate(enhanced.shape, angle)
        rotated = cv2.warpAffine(enhanced, matrix, size, borderValue=255)
        lines = read_lines(np.repeat(rotated[:, :, None], 3, axis=2)) or []
        for box, text, confidence in lines:
            height, length, tilt = _geometry(box)
            if (float(confidence) < .5 or len(text.strip()) < 3 or tilt > 15
                    or height < .14*min(work.shape[:2]) or length < 2.2*height):
                continue
            box = cv2.transform(box[None].astype(np.float32),
                                cv2.invertAffineTransform(matrix))[0]
            cv2.fillConvexPoly(domain, np.round(box).astype(np.int32), 1)
        if domain.any():
            chosen_sign = sign
            break
        if len(tried) >= 2:
            return None
    if chosen_sign is None:
        return None
    cutoff = max(4., 2*np.median(np.abs(smooth)))
    strokes = ((chosen_sign*smooth > cutoff) & (domain > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strokes, 8)
    keep = np.zeros(count, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > 25
    strokes = keep[labels]
    strokes = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    strokes = cv2.dilate(strokes, np.ones((3, 3), np.uint8))
    if not .005 < strokes.mean() < .3:
        return None
    footprint = cv2.resize(strokes, (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
    from .photo_overlay import fit
    veil = fit(img, footprint)
    return MaskLayer(alpha=footprint.astype(np.float32), source=MaskSource.PRIOR,
                     confidence=.75,
                     veil_strength=veil[0] if veil is not None else np.ones(3, np.float32),
                     veil_offset=veil[1] if veil is not None else np.zeros(3, np.float32),
                     paper=np.full(img.shape, 255, np.float32), footprint=footprint,
                     repair_mode="photo_overlay" if veil is not None else "inpaint",
                     method="single_chromatic_oblique")
