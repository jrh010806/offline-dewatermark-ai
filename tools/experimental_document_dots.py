"""Source-only staggered-dot experiment; never installed by the application.

The explicit context manager is for frozen BatchWorker experiments only.  It
keeps every successful existing periodic detector ahead of this candidate.
Geometry supplies search locations; repeated measured shapes and the current
local contrast supply repair permission.  No reference or sample identity is
accepted by this module.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager

import cv2
import numpy as np

from wm.contract import MaskLayer, RepairName

METHOD = 'experimental_staggered_document_dots'


def _mode(values):
    counts = Counter(int(round(v)) for v in values)
    return counts.most_common(1)[0] if counts else (0, 0)


def _geometry(points, shape):
    """Measure independent row/column pitches and alternating row phases."""
    h, w = shape
    rows = {}
    for x, y in points:
        rows.setdefault(int(round(y)), []).append(float(x))
    rows = {y: sorted(xs) for y, xs in rows.items() if len(xs) >= 4}
    if len(rows) < 6:
        return None
    px, votes = _mode(d for xs in rows.values() for d in np.diff(xs) if d >= 10)
    if votes < 20 or px < 10:
        return None
    ys = sorted(y for y, xs in rows.items()
                if sum(abs(d-px) <= .5 for d in np.diff(xs)) >= 2)
    py, votes = _mode(d for d in np.diff(ys) if d >= 10)
    if votes < 4 or py < 10:
        return None
    y0, votes = _mode(y % py for y in ys)
    aligned = [y for y in ys if abs(y % py-y0) <= .25]
    if votes < 6 or np.ptp(aligned) < .55*h:
        return None
    phases = []
    for parity in (0, 1):
        xs = [x for y in aligned if (round((y-y0)/py) % 2) == parity
              for x in rows[y]]
        phase, votes = _mode(x % px for x in xs)
        if votes < 15:
            return None
        phases.append(phase)
    shift = (phases[1]-phases[0]) % px
    if not .2*px <= shift <= .8*px:
        return None  # existing square-grid behaviour is deliberately retained
    on_grid = []
    for x, y in points:
        row = round((y-y0)/py)
        phase = phases[row % 2]
        if abs(y-(y0+row*py)) < .25 and abs((x-phase+px/2) % px-px/2) < .25:
            on_grid.append((int(round(x)), int(round(y))))
    if len(on_grid) < 40 or np.ptp([x for x, _ in on_grid]) < .6*w:
        return None
    quadrants = Counter((int(x >= w/2), int(y >= h/2)) for x, y in on_grid)
    if min(quadrants.get(q, 0) for q in ((0, 0), (0, 1), (1, 0), (1, 1))) < 4:
        return None
    return px, py, phases, y0, on_grid


def estimate(image, trace=None):
    """Return a measured candidate or None; uncertain points stay untouched."""
    info = {'accepted': False, 'reason': 'paper_evidence'}
    if trace is not None:
        trace.append(info)
    gray = image.astype(np.float32).mean(2)
    h, w = gray.shape
    if min(h, w) < 180 or np.mean(gray > 200) < .65:
        return None
    radius_limit = min(12, max(3, int(min(h, w)*.008)))
    kernel = np.ones((2*radius_limit+3,)*2, np.uint8)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    contrast = paper-gray
    count, labels, stats, centres = cv2.connectedComponentsWithStats(
        (contrast > 8).astype(np.uint8), 8)
    points = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if not (2 <= bw <= 2*radius_limit+1 and 2 <= bh <= 2*radius_limit+1
                and .75 <= bw/bh <= 1.34 and .5 <= area/(bw*bh) <= .95):
            continue
        cx, cy = centres[i]
        # Intact small round components on paper establish shape and geometry.
        if abs(cx-round(cx)) > .15 or abs(cy-round(cy)) > .15:
            continue
        region = labels[y:y+bh, x:x+bw] == i
        if not (paper[y:y+bh, x:x+bw][region].mean() > 210
                and 8 < np.median(contrast[y:y+bh, x:x+bw][region]) < 80):
            continue
        points.append((cx, cy))
    info.update(reason='staggered_geometry', compact_components=len(points))
    geometry = _geometry(points, gray.shape)
    if geometry is None:
        return None
    px, py, phases, y0, matched = geometry
    radius = min(radius_limit, max(3, int(min(px, py)*.14)))
    yy, xx = np.indices((2*radius+1,)*2)
    ring = np.maximum(abs(yy-radius), abs(xx-radius)) >= radius-1

    def patch_at(x, y):
        if x-radius < 0 or y-radius < 0 or x+radius >= w or y+radius >= h:
            return None
        patch = image[y-radius:y+radius+1, x-radius:x+radius+1].astype(float)
        values = patch[ring]
        if np.max(np.percentile(values, 95, axis=0)-np.percentile(values, 5, axis=0)) > 3:
            return None
        return patch, np.median(values, axis=0)

    templates = []
    for x, y in matched:
        found = patch_at(x, y)
        if found is not None and found[1].mean() > 210:
            patch, background = found
            templates.append(background-patch)
    info.update(reason='repeated_shape', white_shape_observations=len(templates))
    if len(templates) < 30:
        return None
    template = np.median(templates, axis=0)
    magnitude = template.mean(2)
    support = magnitude > max(2., float(magnitude.max())*.05)
    core = magnitude > .8*magnitude.max()
    if (not core.any() or support.sum() < 5 or support[ring].any()
            or support.sum() > .15*px*py):
        return None
    shape_matches = sum(np.mean(abs((t-template)[support])) < 2.5 for t in templates)
    if shape_matches < max(25, .85*len(templates)):
        return None

    sites = []
    for row, y in enumerate(range(y0, h, py)):
        for x in range(phases[row % 2], w, px):
            found = patch_at(x, y)
            if found is not None:
                patch, background = found
                sites.append((x, y, patch, background))
    # Fit one physical transmission from separately observed flat backgrounds.
    # Paper alone cannot identify opacity and colour; in that case only the
    # directly observed paper dots can be locally interpolated.
    transmission = offset = None
    if len(sites) >= 30:
        backgrounds = np.array([b for _, _, _, b in sites])
        covered = np.array([p[core].mean(0) for _, _, p, _ in sites])
        centred = backgrounds-backgrounds.mean(0)
        energy = float(np.sum(centred**2))
        if backgrounds.mean(1).std() > 15 and np.sum(backgrounds.mean(1) < 200) >= 12:
            s = float(np.sum(centred*(covered-covered.mean(0)))/max(energy, 1))
            t = covered.mean(0)-s*backgrounds.mean(0)
            error = np.abs(covered-(s*backgrounds+t)).mean(1)
            if (.4 < s < .98 and np.all(t >= 0) and np.all(t <= 255*(1-s))
                    and np.percentile(error, 90) < 3.5):
                transmission, offset = s, t

    selected = np.zeros((h, w), bool)
    restored = image.copy()
    paper_sites = inverse_sites = 0
    for x, y, patch, background in sites:
        diff = background-patch
        is_paper = background.mean() > 210
        if is_paper:
            # Both shape agreement and a local source contrast are required.
            error = np.abs(diff-template).mean(2)
            actual = support & (error < 2.5) & (diff.mean(2) > 2)
            if actual.sum() < .95*support.sum():
                continue
            fixed = np.broadcast_to(background, patch.shape)
            paper_sites += 1
        elif transmission is not None:
            coverage = np.clip(magnitude/max(float(magnitude[core].mean()), 1), 0, 1)
            prediction = ((1-coverage[..., None]*(1-transmission))*background
                          + coverage[..., None]*offset)
            error = np.abs(patch-prediction).mean(2)
            local_signal = np.abs(patch-background).mean(2)
            actual = support & (error < 3.5) & (local_signal > 2)
            if actual.sum() < .95*support.sum():
                continue
            fixed = ((patch-coverage[..., None]*offset)
                     / np.maximum(1-coverage[..., None]*(1-transmission), .4))
            inverse_sites += 1
        else:
            continue
        ys, xs = slice(y-radius, y+radius+1), slice(x-radius, x+radius+1)
        selected[ys, xs] |= actual
        view = restored[ys, xs]
        view[actual] = np.rint(np.clip(fixed[actual], 0, 255)).astype(np.uint8)
    info.update(reason='per_dot_evidence', pitch_x=px, row_pitch=py,
                alternating_phases=phases, row_origin=y0,
                verified_shape_pixels=int(support.sum()), paper_sites=paper_sites,
                inverse_sites=inverse_sites, physical_transmission=transmission,
                searched_sites=sum(len(range(phases[i % 2], w, px))
                                   for i, _ in enumerate(range(y0, h, py))),
                selected_pixels=int(selected.sum()), uncertain_intersections_retained=True)
    if paper_sites < 30 or not selected.any():
        return None
    info.update(accepted=True, reason='accepted_source_supported_dots')
    mask = MaskLayer(alpha=selected.astype(np.float32), footprint=selected,
                     paper=restored.astype(np.float32), method=METHOD,
                     repair_mode=METHOD, confidence=.9,
                     veil_strength=np.ones(3, np.float32),
                     veil_offset=np.zeros(3, np.float32))
    mask.experiment_evidence = info
    return mask


@contextmanager
def enabled(trace=None):
    """Explicit research adapter; restores every hook even after an exception."""
    from wm import paper_grid, masks, recall, repair
    old_grid, old_prior = paper_grid.estimate, masks.with_prior
    old_expand, old_repair = recall.expand, repair.repair

    def detect(image):
        original = old_grid(image)
        return original if original is not None else estimate(image, trace)

    def prior(single, shared, thr):
        return single if single.method == METHOD else old_prior(single, shared, thr)

    def expand(image, mask, thr, **kwargs):
        return mask if mask.method == METHOD else old_expand(image, mask, thr, **kwargs)

    def restore(image, mask, thr):
        if mask.method != METHOD:
            return old_repair(image, mask, thr)
        permission = mask.binary(thr.mask_alpha_min) > 0
        out = image.copy()
        out[permission] = np.clip(mask.paper[permission], 0, 255).astype(np.uint8)
        used = [RepairName.INPAINT]
        if mask.experiment_evidence['inverse_sites']:
            used.insert(0, RepairName.ALPHA_UNMIX)
        return out, used

    paper_grid.estimate, masks.with_prior = detect, prior
    recall.expand, repair.repair = expand, restore
    try:
        yield
    finally:
        paper_grid.estimate, masks.with_prior = old_grid, old_prior
        recall.expand, repair.repair = old_expand, old_repair
