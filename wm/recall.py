"""Refine watermark strokes using input pixels; OCR defines search areas only.

No rectangle, inferred row or padding is copied into the repair mask. Measured
inverse/periodic repairs keep their original alpha and physical model.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from . import ocr, photo_observations
from .contract import MaskLayer


def _lines(image, base):
    observations = photo_observations.observations(image, ocr.read_lines(image) or [])
    h, w = image.shape[:2]
    result = []
    for box, text, score in observations:
        height, length, tilt = ocr._geometry(box)
        if (score >= .45 and len(text.strip()) >= 3 and
                .012*min(h, w) < height < .22*min(h, w) and
                length > 2*height and 12 < tilt < 65):
            result.append((box, text, score))
    return result


def _search_domain(shape, lines):
    domain = np.zeros(shape, np.uint8)
    for box, _, _ in lines:
        height, _, _ = ocr._geometry(box)
        direction = box[1]-box[0]
        direction /= max(float(np.linalg.norm(direction)), 1.)
        extended = box.copy()
        extended[[0, 3]] -= direction*height
        extended[[1, 2]] += direction*height
        cv2.fillConvexPoly(domain, np.rint(extended).astype(np.int32), 1)
    return domain > 0


def _grow_strokes(feature, domain, seed_region, height):
    control = domain & ~cv2.dilate(seed_region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if control.sum() < 100:
        control = ~domain
    if control.sum() < 100 or not seed_region.any():
        return np.zeros_like(domain), 0.
    noise = feature[control]
    centre = float(np.median(noise))
    sigma = max(1., float(np.median(np.abs(noise-centre)))*1.4826)
    score = (float(np.percentile(feature[seed_region], 80))-float(np.percentile(noise, 80)))/sigma
    strong = domain & (feature > centre+max(4., 3*sigma))
    weak = domain & (feature > centre+max(1., sigma))
    distance = cv2.distanceTransform((~strong).astype(np.uint8), cv2.DIST_L2, 5)
    weak &= distance <= 3
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    good = np.zeros(count, bool)
    votes = np.bincount(labels[strong], minlength=count)
    good[:] = votes >= max(3, int(.03*height))
    good[0] = False
    good &= stats[:, cv2.CC_STAT_AREA] >= max(8, int(.12*height))
    # Connected low-contrast support grows a stroke; no closing, dilation or
    # hole-filling is allowed to supply pixels that lack this support.
    return good[labels], score


def _pixel_strokes(image, domain, seed_region, height, *, preserve_seeds=False,
                   repeat_support=None):
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .8)
    gray = smooth.mean(2)
    channels = [gray]
    channels.extend(smooth[..., c]-(smooth.sum(2)-smooth[..., c])/2 for c in range(3))
    kernel = np.ones((min(31, max(9, int(.35*height) | 1)),)*2, np.uint8)
    choices = []
    for channel_index, channel in enumerate(channels):
        for operation in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
            feature = cv2.morphologyEx(channel, operation, kernel)
            region, score = _grow_strokes(feature, domain, seed_region, height)
            choices.append((score, region, channel_index, operation))
    # A positive opponent-colour peak describes the ink's added hue. Its
    # complementary negative channel can also respond strongly to dark scene
    # edges (window frames, bars) and must not outrank that independent hue.
    coloured = [item for item in choices if item[2] and item[3] == cv2.MORPH_TOPHAT and item[0] >= 3]
    best = max(coloured or choices, key=lambda item: item[0])
    region = best[1]
    if not coloured and best[2] and region.any():
        gray = max((item for item in choices if item[2] == 0), key=lambda item: item[0])
        # Weakly tinted grain can hide an antialias edge in one colour channel.
        # Admit independently measured luminance strokes only near an existing
        # stroke, never all pixels in a dilated neighbourhood.
        if gray[0] >= .5:
            nearby = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5) <= 3
            region = region | (gray[1] & nearby)
    if preserve_seeds and coloured and repeat_support is not None:
        gray = max((item for item in choices if item[2] == 0), key=lambda item: item[0])
        # Luminance alone also selects the halo around a real stroke. Require
        # the independently registered repeat core and native pixel contrast,
        # in addition to the old seed and this connected luminance observation.
        if gray[0] >= max(3., 2*best[0]):
            region = region | (seed_region & gray[1] & repeat_support)
    return region


def expand(image, mask, thr, *, use_ocr=True, style=None):
    if not use_ocr or style in ('dot_grid', 'corner_id'):
        return mask
    # Keep validated numerical alpha and the corresponding inverse. A search
    # pass must never replace a measured background model with a box inpainter.
    if mask.repair_mode in ('texture_overlay', 'photo_repeat_inverse',
                            'coloured_document_overlay', 'document_overlay_full',
                            'periodic_paper_overlay'):
        return replace(mask, footprint=mask.binary(thr.mask_alpha_min) > 0)
    if mask.method in ('single_periodic_overlay', 'single_periodic_document'):
        return mask
    base = mask.binary(thr.mask_alpha_min) > 0
    if not base.any():
        return mask
    if mask.method == 'experimental_balanced_chromatic_strokes':
        # This opt-in fallback already validates native-resolution strokes.
        # Another generic channel choice would discard its source evidence.
        return mask
    if mask.method in ('photo_observed_repeats', 'photo_complete_repeated_rows', 'repeated_oblique_text'):
        from . import observed_text
        located = observed_text.repeated_photo(image, ocr.read_lines(image) or [],
                                               stroke_only=True, search_mask=base)
        if located is not None:
            return located
    lines = _lines(image, base)
    if not lines:
        return mask
    height = float(np.median([ocr._geometry(box)[0] for box, _, _ in lines]))
    domain = _search_domain(base.shape, lines)
    if mask.method in ('photo_observed_repeats', 'photo_complete_repeated_rows', 'repeated_oblique_text'):
        found = photo_observations._repeated_domain(image, lines)
        if found is not None:
            domain |= found[0]
    seed = base & domain
    preserve = (thr.experimental_seed_preservation and mask.repair_mode == 'inpaint'
                and not thr.experimental_colour_refinement)
    region = _pixel_strokes(image, domain, seed, height, preserve_seeds=preserve,
                            repeat_support=mask.repeat_stroke_support)
    method = 'recall_pixel_strokes'
    if thr.experimental_colour_refinement:
        from .stroke_colour import refine
        refined = refine(image, domain, seed, height)
        if refined is not None:
            region = refined
            method = 'recall_colour_refinement'
    return replace(mask, alpha=region.astype(np.float32), footprint=region,
                   text_keepout=None, method=method, repair_mode='inpaint',
                   veil_strength=np.ones(3, np.float32), veil_offset=np.zeros(3, np.float32),
                   overlay_opacity=None)
