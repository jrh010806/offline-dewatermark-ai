"""Bounded photo-text detection using OCR geometry and measured translucency.

No filenames, template phrases, clean references or stored masks are used.
Recognised text alone is never permission to erase it: a fitted translucent
overlay must agree across multiple strokes and background colours.
"""
import cv2
import numpy as np

from . import ocr
from .contract import MaskLayer, MaskSource


def _repeat_locations(image, candidates):
    """Extend a measured stroke template only after two matching OCR seeds."""
    if len(candidates) < 2:
        return None
    groups = {}
    for region, veil, text in candidates:
        key = "".join(text.split()).casefold()
        groups.setdefault(key, []).append(region)
    regions = []
    for group in groups.values():
        distinct = []
        for region in group:
            yy, xx = np.nonzero(region)
            centre = np.array([xx.mean(), yy.mean()])
            length = max(np.ptp(xx), np.ptp(yy))
            if all(np.linalg.norm(centre-c) > .6*max(length, size) for _, c, size in distinct):
                distinct.append((region, centre, length))
        if len(distinct) >= 2:
            regions.extend(r for r, _, _ in distinct)
    if len(regions) < 2:
        return None
    template_region = max(regions, key=np.count_nonzero)
    yy, xx = np.nonzero(template_region)
    x0, y0, x1, y1 = xx.min(), yy.min(), xx.max()+1, yy.max()+1
    template = template_region[y0:y1, x0:x1].astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Whiten local contrast so changing photo colours do not grant a match just
    # by sharing the template's mean intensity.
    median = cv2.medianBlur(gray, 15)
    feature = np.clip(np.abs(gray.astype(float)-median)/12, 0, 1).astype(np.float32)
    scores = cv2.matchTemplate(feature, template, cv2.TM_CCOEFF_NORMED)
    result = np.logical_or.reduce(regions)
    th, tw = template.shape
    for _ in range(80):
        _, confidence, _, (x, y) = cv2.minMaxLoc(scores)
        if not np.isfinite(confidence) or confidence < .60:
            break
        result[y:y+th, x:x+tw] |= template > 0
        scores[max(0, y-th//2):y+th//2+1, max(0, x-tw//2):x+tw//2+1] = -1
    return result


def _fit(image, background, region, *, min_strength=.22):
    core = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if core.sum() < 25:
        return None
    x, y = background[core].astype(float), image[core].astype(float)
    keep = np.ones(len(x), bool)
    for _ in range(4):
        xc, yc = x[keep]-x[keep].mean(0), y[keep]-y[keep].mean(0)
        variance = np.mean(xc**2)
        if variance < 4:
            return None
        s = float(np.sum(xc*yc) / max(np.sum(xc**2), 1))
        offset = np.median(y-s*x, axis=0)
        errors = np.abs(y-(s*x+offset)).mean(1)
        keep = errors <= max(2., np.percentile(errors, 75))
    if not min_strength < s < .90 or np.any(offset < -3) or np.any(offset > 255*(1-s)+3):
        return None
    signal = np.abs(y-x).mean(1)
    if np.median(errors) > 3.5 or np.mean(errors < .35*np.maximum(signal, 4)) < .7:
        return None
    return np.full(3, s, np.float32), offset.astype(np.float32)


def _strokes(image, box):
    h, w = image.shape[:2]
    height, length, tilt = ocr._geometry(box)
    pad = max(4, int(height*.5))
    x0, y0 = np.maximum(np.floor(box.min(0)).astype(int)-pad, 0)
    x1, y1 = np.minimum(np.ceil(box.max(0)).astype(int)+pad+1, [w, h])
    crop = image[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 9:
        return None
    domain = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(domain, np.round(box-[x0, y0]).astype(np.int32), 1)
    k = min(31, max(7, int(height*.35) | 1))
    smooth = cv2.medianBlur(crop, k).astype(float)
    delta = crop.astype(float)-smooth
    gray = delta.mean(2)
    outside = gray[domain == 0]
    noise = np.median(np.abs(outside)) if outside.size else 2
    threshold = max(4., 3*noise)
    choices = []
    for sign in (-1, 1):
        region = (sign*gray > threshold) & (domain > 0)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
        good = np.zeros(n, np.uint8)
        good[1:] = ((stats[1:, cv2.CC_STAT_AREA] >= max(3, height*.08))
                    & (stats[1:, cv2.CC_STAT_HEIGHT] >= height*.15)
                    & (stats[1:, cv2.CC_STAT_HEIGHT] <= height*1.15))
        if good.sum() < 4:
            continue
        region = good[labels] > 0
        if not .025 < region.sum()/max(domain.sum(), 1) < .50:
            continue
        background = cv2.inpaint(crop, region.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
        veil = _fit(crop, background, region)
        if veil is None:
            continue
        # Repair actual strokes and a one-pixel antialias fringe, not the OCR
        # rectangle. The fringe remains bounded by the observed text geometry.
        region = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (domain > 0)
        full = np.zeros((h, w), bool); full[y0:y1, x0:x1] = region
        choices.append((full, veil))
    return choices[0] if len(choices) == 1 else None


def estimate(image, *, allow_corner=True, allow_oblique=True, require_repeat=False):
    h, w = image.shape[:2]
    lines = list(ocr.read_lines(image) or [])
    # Large oblique text can disappear when the OCR detector downsizes a full
    # photograph. Rotate a contrast-enhanced view, then map observations back;
    # pixel fitting always uses the untouched native-resolution source.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8)).apply(gray)
    for angle in (-30, 30):
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.)
        view = cv2.warpAffine(enhanced, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        for box, text, confidence in ocr.read_lines(np.repeat(view[..., None], 3, axis=2)) or []:
            mapped = cv2.transform(box[None], cv2.invertAffineTransform(matrix))[0]
            if np.all(mapped >= 0) and np.all(mapped[:, 0] < w) and np.all(mapped[:, 1] < h):
                if not any(np.linalg.norm(mapped.mean(0)-b.mean(0)) < 10 for b, _, _ in lines):
                    lines.append((mapped, text, confidence))
    candidates = []
    local_domains = []
    for box, text, confidence in lines:
        height, length, tilt = ocr._geometry(box)
        if confidence < .65 or len(text.strip()) < 4 or length < height*2.5:
            continue
        centre = box.mean(0)
        corner = ((centre[0] < .35*w or centre[0] > .65*w)
                  and (centre[1] < .20*h or centre[1] > .80*h))
        if not ((allow_corner and corner)
                or (allow_oblique and height > .03*min(h, w) and tilt > 10)):
            continue
        located = _strokes(image, box)
        if located is not None:
            candidates.append((*located, text))
            if corner and tilt < 8 and height < .065*min(h,w):
                # The fitted translucent corner line is already confirmed.
                # At a scene edge its stroke contrast can change sign, leaving
                # holes in the narrow mask. Give the inpainter the bounded OCR
                # line plus a small fringe so retained source letters cannot
                # leak back into that repair.
                domain=np.zeros((h,w),np.uint8)
                cv2.fillConvexPoly(domain,np.rint(box).astype(np.int32),1)
                domain=cv2.dilate(domain,np.ones((7,7),np.uint8))>0
                local_domains.append(domain)
    if not candidates:
        return None
    region = np.logical_or.reduce([x[0] for x in candidates])
    repeated = _repeat_locations(image, candidates)
    if require_repeat and repeated is None:
        return None
    if repeated is not None:
        region |= repeated
    if region.mean() > .12:
        return None
    if local_domains and not require_repeat:
        region |= np.logical_or.reduce(local_domains)
        if region.mean() < .035:
            return MaskLayer(region.astype(np.float32),source=MaskSource.PRIOR,
                footprint=region,confidence=.75,repair_mode='inpaint',
                veil_strength=np.ones(3,np.float32),veil_offset=np.zeros(3,np.float32),
                method='photo_confirmed_corner_line')
    # Separate lines can have different colours/opacity. Use only the measured
    # location for LaMa; do not impose one guessed inverse on all of them.
    return MaskLayer(region.astype(np.float32), source=MaskSource.PRIOR,
                     footprint=region, confidence=.75, repair_mode="thin_photo_text",
                     veil_strength=np.ones(3, np.float32), veil_offset=np.zeros(3, np.float32),
                     method="photo_translucent_text")
