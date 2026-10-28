"""Source-only colour-enhanced observations for missed photo lettering.

OCR supplies bounded geometry. Single lines additionally require a measured
translucent colour layer; repeated lines require a coherent spatial family.
No clean images, dataset identifiers, stored phrases or labelled masks enter
this module.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import ocr
from .contract import MaskLayer, MaskSource
from .photo_text import _fit, _strokes


def _layer(region, method):
    return MaskLayer(region.astype(np.float32), footprint=region,
                     confidence=.8, source=MaskSource.PRIOR,
                     repair_mode='inpaint', method=method,
                     veil_strength=np.ones(3, np.float32),
                     veil_offset=np.zeros(3, np.float32))


def observations(image, source_lines):
    """Enhance narrow chromatic strokes without assuming their colour."""
    result = list(source_lines)
    rgb = image.astype(np.float32)
    for channel in range(3):
        opponent = np.uint8(np.clip(128 + rgb[..., channel] -
            (rgb[..., (channel+1) % 3] + rgb[..., (channel+2) % 3]) / 2, 0, 255))
        feature = cv2.morphologyEx(opponent, cv2.MORPH_TOPHAT,
                                   np.ones((17, 17), np.uint8))
        view = np.uint8(255 - np.clip(feature.astype(np.float32)*12, 0, 255))
        result.extend(ocr.read_lines(np.repeat(view[..., None], 3, axis=2)) or [])
    return result


def _colour_fit(image, box):
    """Find actual coloured strokes, then test their relation to local context.

    Both signs are needed: lettering may brighten a dark bar while darkening
    the light wall next to it. A fitted slope above zero rejects opaque print.
    """
    height, _, _ = ocr._geometry(box)
    h, w = image.shape[:2]
    pad = max(5, int(.35*height))
    x0, y0 = np.maximum(np.floor(box.min(0)).astype(int)-pad, 0)
    x1, y1 = np.minimum(np.ceil(box.max(0)).astype(int)+pad+1, [w, h])
    crop = image[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 9:
        return False
    domain = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(domain, np.rint(box-[x0, y0]).astype(np.int32), 1)
    rgb = crop.astype(np.float32)
    for channel in range(3):
        opponent = np.uint8(np.clip(128+rgb[..., channel] -
            (rgb[..., (channel+1) % 3]+rgb[..., (channel+2) % 3])/2, 0, 255))
        for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            feature = cv2.morphologyEx(opponent, operation, np.ones((31, 31), np.uint8))
            for threshold in (8, 20, 40):
                region = (feature > threshold) & (domain > 0)
                fraction = region.sum()/max(1, domain.sum())
                if not .04 < fraction < .65:
                    continue
                count, _, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
                parts = stats[1:]
                glyphs = (parts[:, cv2.CC_STAT_HEIGHT] > .3*height) & \
                         (parts[:, cv2.CC_STAT_AREA] > height*.25)
                if glyphs.sum() < 4:
                    continue
                expanded = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8))
                background = cv2.inpaint(crop, expanded*255, 3, cv2.INPAINT_TELEA)
                if _fit(crop, background, region, min_strength=.08) is not None:
                    return True
    return False


def local_lines(image, lines, *, allow_corner=True, allow_center=True):
    h, w = image.shape[:2]
    region = np.zeros((h, w), np.uint8)
    accepted = []
    candidates = []
    for box, text, score in sorted(lines, key=lambda x: x[2], reverse=True):
        height, length, tilt = ocr._geometry(box)
        if score < .70 or len(text.strip()) < 4 or \
                not .012*min(h, w) < height < .16*min(h, w) or length < 3*height:
            continue
        centre = box.mean(0)
        corner = ((centre[0] < .35*w or centre[0] > .65*w) and
                  (centre[1] < .20*h or centre[1] > .80*h))
        if not (allow_corner if corner else allow_center):
            continue
        candidates.append((box, text, score))
    # Orientation selects the verification path, not whether text is allowed
    # to be a watermark. A repeating background can validate any reading angle.
    from . import texture_overlay
    measured = texture_overlay.estimate(image, candidates)
    if measured is not None:
        return measured
    for box, text, score in candidates:
        height, length, tilt = ocr._geometry(box)
        centre = box.mean(0)
        if any(np.linalg.norm(centre-c) < height for c in accepted):
            continue
        if tilt > 8:
            # If periodic prediction is unavailable, require independently
            # fitted strokes. Never repaint an oblique OCR bounding rectangle.
            found = _strokes(image, box)
            if found is not None:
                region |= found[0].astype(np.uint8)
                accepted.append(centre)
            continue
        if not _colour_fit(image, box):
            continue
        domain = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(domain, np.rint(box).astype(np.int32), 1)
        # Full line bounds include opposite-sign strokes and a drop shadow.
        pad = max(4, int(height*.22))
        region |= cv2.dilate(domain, np.ones((2*pad+1, 2*pad+1), np.uint8))
        accepted.append(centre)
    if not accepted or region.mean() > .18:
        return None
    return _layer(region > 0, 'photo_confirmed_colour_line')


def _repeated_domain(image, lines):
    """Complete coherent oblique rows, including clipped ends at image edges.

    Reading a few fragments is not a coverage check. Project all observations
    onto the common row normal and use the measured row spacing to find missing
    rows. Only a consistent family of at least three distinct rows can do this.
    """
    h, w = image.shape[:2]
    candidates = []
    for box, text, score in lines:
        height, length, tilt = ocr._geometry(box)
        if score >= .65 and len(text.strip()) >= 4 and \
                .02*min(h, w) < height < .11*min(h, w) and length > 2.5*height and 12 < tilt < 50:
            signed = float(np.degrees(np.arctan2(*(box[1]-box[0])[::-1])))
            candidates.append((box, float(height), signed))
    if len(candidates) < 4:
        return None
    # Choose the largest orientation family rather than letting unrelated
    # scene text pull its median away from the repeated overlay.
    seed = max(candidates, key=lambda a: sum(abs(a[2]-b[2]) < 5 for b in candidates))
    family = [a for a in candidates if abs(a[2]-seed[2]) < 5 and .65*seed[1] < a[1] < 1.5*seed[1]]
    if len(family) < 4:
        return None
    height = float(np.median([a[1] for a in family]))
    # Long reading edges give a more reliable direction than short fragments;
    # a one-pixel OCR error on a short word otherwise drifts across the frame.
    weights = [np.linalg.norm(a[0][1]-a[0][0])**2 for a in family]
    angle = float(np.average([a[2] for a in family], weights=weights))
    theta = np.radians(angle)
    normal = np.array([-np.sin(theta), np.cos(theta)])
    centres = np.array([box.mean(0) for box, _, _ in family])
    if np.ptp(centres[:, 0]) < 3*height or np.ptp(centres[:, 1]) < 2*height:
        return None
    positions = sorted(float(c @ normal) for c in centres)
    groups = []
    for position in positions:
        if not groups or position-np.median(groups[-1]) > .65*height:
            groups.append([position])
        else:
            groups[-1].append(position)
    rows = np.array([np.median(g) for g in groups])
    if len(rows) < 3:
        return None
    gaps = np.diff(rows)
    # A missing row creates a gap of two or more pitches. Reject inconsistent
    # spacing, but do not reject the very missing rows this path must complete.
    pitch = float(np.median(gaps[gaps <= np.percentile(gaps, 65)]))
    steps = np.maximum(1, np.rint(gaps/pitch)).astype(int)
    if not 1.5*height < pitch < 7*height or \
            np.max(np.abs(gaps-steps*pitch)) > .25*pitch:
        return None
    yy, xx = np.indices((h, w), dtype=np.float32)
    position = xx*normal[0]+yy*normal[1]
    # Project observed polygons, not just the nominal font size: oblique OCR
    # boxes can clip the top of a tall glyph or merge two adjacent words.
    half = max(height*.68, float(np.percentile([
        np.max(np.abs(box @ normal - box.mean(0) @ normal)) for box, _, _ in family], 90)) + 5)
    all_rows = [(row, half) for row in rows]
    for left, right, count in zip(rows[:-1], rows[1:], steps):
        # Unread rows have less certain centres on irregular tiling. Their
        # measured spacing constrains a wider search band than observed rows.
        all_rows.extend((row, half+.3*height)
                        for row in np.linspace(left, right, count+1)[1:-1])
    for direction, start in [(-1, rows[0]), (1, rows[-1])]:
        value = start + direction*pitch
        for _ in range(32):
            if value < position.min()-half-.3*height or value > position.max()+half+.3*height:
                break
            all_rows.append((value, half+.3*height))
            value += direction*pitch
    region = np.zeros((h, w), bool)
    for row, radius in all_rows:
        region |= np.abs(position-row) <= radius
    return region, family


def repeated_lines(image, lines):
    found_domain = _repeated_domain(image, lines)
    if found_domain is None:
        return None
    region, family = found_domain
    h, w = image.shape[:2]
    # Row geometry is a search domain, not permission to erase entire stripes.
    # Learn the chromatic response from independently fitted translucent
    # strokes, then search that response throughout the confirmed row family.
    seeds = np.zeros((h, w), bool)
    domains = np.zeros((h, w), np.uint8)
    for box, _, _ in family:
        cv2.fillConvexPoly(domains, np.rint(box).astype(np.int32), 1)
        found = _strokes(image, box)
        if found is not None:
            seeds |= found[0]
    if seeds.sum() < 100:
        return None
    rgb = image.astype(np.float32)
    surrounding = (domains > 0) & ~cv2.dilate(seeds.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if surrounding.sum() < 100:
        return None
    features = []
    for channel in range(3):
        opponent = np.uint8(np.clip(128+rgb[..., channel] -
            (rgb[..., (channel+1) % 3]+rgb[..., (channel+2) % 3])/2, 0, 255))
        for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            feature = cv2.morphologyEx(opponent, operation, np.ones((31, 31), np.uint8))
            score = float(np.median(feature[seeds])-np.median(feature[surrounding]))
            features.append((score, feature))
    score, feature = max(features, key=lambda item: item[0])
    if score < 2:
        return None
    strokes = (feature > 2) & region
    strokes |= seeds
    strokes = cv2.morphologyEx(strokes.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    region = cv2.dilate(strokes, np.ones((7, 7), np.uint8)) > 0
    if not .01 < region.mean() < .50:
        return None
    return _layer(region, 'photo_complete_repeated_rows')
