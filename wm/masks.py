"""Watermark coverage estimation.

Document pages are bimodal: a paper pole P and an ink pole near 0. A
semi-transparent overlay maps them to two *observable* poles,

    covered paper = s*P + t        covered ink = t

so the veil (s, t) is recoverable and coverage beta is measured where the
overlay sits on paper. Two rules keep body text alive, which is where naive
magnitude thresholding fails (measured precision 0.04-0.43 before this):

* the overlay pole is located as a histogram *spike*, because antialiased
  strokes fill the deviation range as a smooth continuum while the overlay
  piles up at one value;
* a pixel darker than the covered-ink pole cannot be produced by the overlay
  at any coverage, so it is forced to beta = 0.
"""
from __future__ import annotations

import numpy as np
import cv2

from .contract import MaskLayer, MaskSource, Thresholds, WatermarkInstance

_MIN_FRACTION = 1e-4


def paper_level(img: np.ndarray, block: int = 48) -> np.ndarray:
    """Page paper level (97th percentile). Global rather than per-block: the
    documents in scope are uniformly white and a flat field keeps the veil fit
    stable. Per-block illumination modelling is a later concern."""
    small = cv2.resize(img.astype(np.float32),
                       (max(1, img.shape[1] // block), max(1, img.shape[0] // block)),
                       interpolation=cv2.INTER_AREA)
    hi = np.percentile(small.reshape(-1, 3), 97, axis=0).astype(np.float32)
    return np.broadcast_to(hi, (img.shape[0], img.shape[1], 3)).copy()


def paper_anchor(gray: np.ndarray, P: float, radius: int = 9) -> np.ndarray:
    """Neighbourhoods that contain uncovered paper.

    A partial overlay is only *verifiable* where uncovered paper sits beside it:
    the pole pair (P, P-m) has to be observable in the same window. Solid
    low-opacity content - a photo gradient, a tint block, a logo fill - is
    layout-invariant the same way a watermark is and lands in the same tonal
    band, but it has no paper anchor inside its own interior, so its deviation
    is unverifiable and coverage is left at zero with a manual-review flag
    instead of being inverted. Measured: one photo block accounted for 40%+
    occupancy of the cross-page candidate band in four of eight 8x8 blocks."""
    near = (gray >= 0.97 * P).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(near, kernel) > 0


def flat_background(img: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Unbroken paper, for reporting broad-area residue only."""
    gray = img.astype(np.float32).mean(2)
    level = paper.astype(np.float32).mean(2)
    bright = (gray > 0.94 * level).astype(np.uint8)
    return (cv2.erode(bright, np.ones((5, 5), np.uint8)) > 0) & (gray > 0.90 * level)


def _spike(vals: np.ndarray, lo: float, hi: float, bin_w: float = 2.0) -> float:
    if vals.size < 400 or hi <= lo:
        return 0.0
    bins = np.arange(lo, hi + bin_w, bin_w)
    hist, edges = np.histogram(vals, bins=bins)
    in_range = int(hist.sum())
    if in_range < 400 or hist.max() < 0.02 * in_range:
        return 0.0
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def _veil_magnitude(dev: np.ndarray, cand: np.ndarray, P: float) -> tuple[np.ndarray, float]:
    m = _spike(dev[cand], 0.04 * P, 0.86 * P)
    if m <= 0:
        return np.zeros(dev.shape, np.float32), 0.0
    seed = cand & (np.abs(dev - m) <= max(0.14 * m, 4.0))
    return seed.astype(np.float32), m


def _extend_shape(seed: np.ndarray, kernel_px: int) -> np.ndarray:
    """Bridge the glyph where it disappears behind ink."""
    k = max(3, int(kernel_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)
    return cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def _fit_poles(img: np.ndarray, gray: np.ndarray, shape: np.ndarray, P: float,
               m: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Recover the veil from the covered-paper pole plus a one-parameter search.

    alpha0 * (P - W) = m pins strength to colour, so only W is free. Each
    candidate is scored by how well the inverse veil puts the page's own pixels
    back onto the paper and ink poles, which works even when the covered-ink
    cluster is too weak to find by itself.
    """
    fimg = img.astype(np.float32)
    core = shape > 0.5
    c1_gray = P - m
    if core.mean() < _MIN_FRACTION or c1_gray <= 0:
        return np.ones(3, np.float32), np.zeros(3, np.float32), 0.0
    paper_sel = core & (np.abs(gray - c1_gray) <= max(0.10 * m, 5.0))
    if paper_sel.mean() < _MIN_FRACTION:
        return np.ones(3, np.float32), np.zeros(3, np.float32), 0.0
    covered_paper = np.median(fimg[paper_sel], axis=0).astype(np.float32)
    ink_pole = _spike(gray[gray < 0.5 * P], 0.0, 0.5 * P)

    ys, xs = np.nonzero(core)
    if ys.size == 0:
        return np.ones(3, np.float32), np.zeros(3, np.float32), 0.0
    step = max(1, int(np.sqrt(ys.size / 250_000)))
    ys, xs = ys[::step], xs[::step]
    sample_gray = gray[ys, xs]
    sample_dev = P - sample_gray
    beta = np.clip(sample_dev / m, 0.0, 1.0)
    tol = 22.0
    # (s, t) is only identifiable from the covered-ink pole: on paper the inverse
    # is a tautology, so a direct colour estimate from high-coverage paper pixels
    # just returns the covered-paper level. Hence the scored one-parameter search.
    best, best_score = None, -1.0
    for W in np.linspace(ink_pole + 4.0, max(ink_pole + 5.0, P - m / 0.98), 26):
        alpha0 = m / max(P - W, 1e-3)
        if not 0.02 <= alpha0 <= 0.98:
            continue
        t = alpha0 * W
        restored = (sample_gray - beta * t) / np.maximum(1.0 - beta * (1.0 - alpha0), 0.04)
        near = np.minimum(np.abs(restored - P), np.abs(restored - ink_pole))
        score = float((near < tol).mean())
        if score > best_score:
            best, best_score = (alpha0, W, t, score), score
    if best is None:
        return np.ones(3, np.float32), np.zeros(3, np.float32), 0.0
    alpha0, W, t_scalar, score = best
    s = np.full(3, 1.0 - alpha0, np.float32)
    t = np.clip(covered_paper - s * np.float32(P), 0.0, 255.0).astype(np.float32)
    return s, t, t_scalar


def ink_halo(gray: np.ndarray, P: float, radius: int = 3) -> np.ndarray:
    """The ramp beside a glyph, which must never be touched.

    Measured on the synthetic set: 83-91% of the pixels the tonal channel claims
    as watermark but the ground truth does not lie within 3-5 px of real ink,
    while only 11-30% of the true ones do. A mid-tone pixel next to a dark stroke
    is ambiguous between "overlay on paper" and "antialiased edge", and guessing
    wrong erases part of a glyph - the exact breach of the tender's third effect
    clause, which costs more than leaving a little watermark behind."""
    core = (gray < 0.40 * P).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(core, kernel) > 0


def _beta_field(dev: np.ndarray, m: float, shape: np.ndarray,
                gray: np.ndarray, t_level: float, anchor: np.ndarray,
                P: float, halo_px: int = 3) -> np.ndarray:
    """Coverage is only asserted inside the overlay footprint. Leaving the raw
    ratio outside it would admit the whole antialiasing continuum, which is what
    collapsed precision before."""
    beta = np.where(shape > 0.5, np.clip(dev / max(m, 1e-3), 0.0, 1.0), 0.0)
    beta = np.where(anchor, beta, 0.0)
    beta = np.where(ink_halo(gray, P, halo_px), 0.0, beta)
    return np.where(gray < 0.92 * t_level, 0.0, beta).astype(np.float32)


def _complete_occluded_coverage(beta: np.ndarray, footprint: np.ndarray, gray: np.ndarray,
                                t_level: float, radius: int = 5) -> np.ndarray:
    """Carry each stroke's measured coverage onto the pixels it hides.

    Where the overlay sits on top of ink the tone mixes coverage with the unknown
    ink, so coverage is unmeasurable and is forced to zero - but the ground truth
    counts those pixels as watermark, which capped recall at ~0.87 and left grey
    strokes behind. Coverage is smooth along a stroke, so the value measured on
    the paper side of the same stroke is a legitimate estimate on the ink side,
    and the inverse stays exact there. Local propagation keeps antialiasing,
    which flattening to solid ink would not."""
    hidden = footprint & (beta < Thresholds.mask_alpha_min) & (gray < 0.92 * t_level)
    if not hidden.any():
        return beta
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    spread = cv2.dilate(beta, kernel)
    out = beta.copy()
    out[hidden] = spread[hidden]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _empty(paper: np.ndarray) -> MaskLayer:
    return MaskLayer(alpha=np.zeros(paper.shape[:2], np.float32), source=MaskSource.PRIOR,
                     confidence=0.0, veil_strength=np.ones(3, np.float32),
                     veil_offset=np.zeros(3, np.float32), paper=paper,
                     footprint=np.zeros(paper.shape[:2], bool))


def blocking_ratio(img: np.ndarray) -> float:
    """How much of the page's quiet-area gradient sits on 8-pixel boundaries.

    JPEG quantises whole 8x8 blocks, so its error is periodic; lossless pages
    have no such structure and the ratio sits near 1. Restricted to quiet
    transitions because glyph strokes would otherwise drown the signal."""
    g = img.astype(np.float32).mean(2)
    ratios = []
    for axis in (1, 0):
        d = np.abs(np.diff(g, axis=axis))
        quiet = d < 3.0
        n = d.shape[1] if axis == 1 else d.shape[0]
        grid = np.zeros(n, bool)
        grid[((np.arange(n) + 1) % 8) == 0] = True
        sel = (np.broadcast_to(grid, d.shape) if axis == 1
               else np.broadcast_to(grid[:, None], d.shape))
        b, i = d[sel & quiet], d[~sel & quiet]
        if b.size and i.size:
            ratios.append(float(b.mean() / max(i.mean(), 1e-3)))
    return float(np.mean(ratios)) if ratios else 1.0


def deblock(img: np.ndarray) -> np.ndarray:
    """A copy of the page used for *locating*, median-filtered when it carries
    JPEG blocking.

    A JPEG round trip moves every pixel by a few levels, which is the same
    magnitude as a light veil's deviation, so the covered-paper spike smears
    into a continuum and whole blocks read as watermark: at quality 75 mask
    precision falls 0.90 -> 0.31, and the large-centred style collapses from
    0.55 to 0.02. A 3x3 median pulls the poles back (0.02 -> 0.45) but costs
    precision on clean pages (0.55 -> 0.47), so it is gated on the blocking
    ratio, which separates the two regimes on the synthetic set (lossless
    0.97-1.07 against 1.20-2.16 at quality 75). Locating never reaches the
    output: un-mixing runs on the pixels the client gave us, otherwise the
    fidelity clause would be measuring our own filtering."""
    return img if blocking_ratio(img) < 1.35 else cv2.medianBlur(img, 3)


def estimate_single_page(img: np.ndarray, thr: Thresholds = Thresholds(),
                         close_px: int = 11) -> MaskLayer:
    img = deblock(img)
    paper = paper_level(img)
    P = float(np.median(paper))
    gray = img.astype(np.float32).mean(2)
    dev = P - gray
    # Candidate tones are restricted to the bright half of the range on purpose.
    # Widening it to find *dark* heavy veils (covered-paper pole at gray 83 of 255)
    # was measured and lost: the antialias mass beside the body-ink cluster then
    # out-votes the watermark spike, the fit collapses to s=1,t=0 and the page is
    # returned untouched. A veil that dark is tonally indistinguishable from ink,
    # so it belongs to the structural/semantic channel, not this one.
    side = gray > 0.55 * P
    seed, m = _veil_magnitude(dev, side, P)
    if m <= 0:
        return _empty(paper)
    shape = _extend_shape(seed, close_px)
    s, t, t_level = _fit_poles(img, gray, shape, P, m)
    anchor = paper_anchor(gray, P)
    beta = _beta_field(dev, m, shape, gray, t_level, anchor, P)
    mask = MaskLayer(alpha=beta, source=MaskSource.PRIOR,
                     confidence=float(np.clip(seed.mean() / 0.002, 0.0, 1.0)),
                     veil_strength=s, veil_offset=t, paper=paper)
    mask.footprint = (shape > 0.5) & anchor
    mask.alpha = _complete_occluded_coverage(beta, mask.footprint, gray, t_level)
    mask.instances = _instances(mask.alpha, thr)
    return mask


def estimate_crosspage(images: list[np.ndarray], thr: Thresholds = Thresholds(),
                       close_px: int = 5) -> MaskLayer:
    """Content-free coverage: the overlay recurs per pixel while the text does
    not, so the per-pixel median stamps the watermark with the content removed.

    The median removes *body* text only. Template text (headers, page numbers,
    table rules) recurs on every page and survives it as dark ink, and raw
    dev/m reads that as full coverage, which un-mixes pixels the watermark never
    touched (measured: ink IoU 0.31-0.44, worse than single-page). So coverage
    is still gated by the overlay footprint, bridged across ink by morphology.
    """
    if len(images) < 2:
        raise ValueError("cross-page estimation needs at least two aligned pages")
    median = np.median(np.stack(images, axis=0).astype(np.float32), axis=0)
    med_u8 = median.astype(np.uint8)
    paper = paper_level(med_u8)
    P = float(np.median(paper))
    gray = median.mean(2)
    dev = P - gray
    seed, m = _veil_magnitude(dev, dev > 0.04 * P, P)
    if m <= 0:
        return estimate_single_page(images[0], thr)
    band = (dev > 0.45 * m) & (dev < 1.6 * m)
    if band.mean() < _MIN_FRACTION:
        band = dev > 0.45 * m
    shape = _extend_shape(band.astype(np.float32), close_px)
    page0 = deblock(images[0])
    gray0 = page0.astype(np.float32).mean(2)
    s, t, t_level = _fit_poles(page0, gray0, shape, P, m)
    anchor = paper_anchor(gray, P)
    beta = _beta_field(dev, m, shape, gray, t_level, anchor, P)
    mask = MaskLayer(
        alpha=beta, source=MaskSource.CROSSPAGE,
        confidence=float(np.clip((beta > 0.5).mean() / 0.002, 0.0, 1.0)),
        veil_strength=s, veil_offset=t, paper=paper,
    )
    mask.footprint = (shape > 0.5) & anchor
    mask.alpha = _complete_occluded_coverage(beta, mask.footprint, gray, t_level)
    mask.instances = _instances(mask.alpha, thr)
    return mask


def residual_visible_fraction(source: np.ndarray, output: np.ndarray,
                              clean: np.ndarray, min_signal: float = 8.0,
                              keep_ratio: float = 0.5) -> float:
    """Fraction of the pixels that *did* carry a watermark and still carry at
    least ``keep_ratio`` of it. Anchored on the original-vs-clean difference, so
    a clean page scores exactly zero and the threshold needs no null-floor
    calibration. Ground-truth only."""
    signal = np.abs(source.astype(np.int16) - clean.astype(np.int16)).mean(2)
    carried = signal >= min_signal
    if not carried.any():
        return 0.0
    leftover = np.abs(output.astype(np.int16) - clean.astype(np.int16)).mean(2)
    return float((leftover[carried] / signal[carried] >= keep_ratio).mean())


def residual_background_fraction(img: np.ndarray, mask: MaskLayer,
                                 tol: float = 14.0) -> float:
    """Blind fallback: fraction of unbroken paper still sitting on the
    covered-paper pole.

    The pole comes from the original analysis, never re-derived: if the whole
    background is still veiled, re-estimating the paper level would move the
    pole onto the residue and report a clean page. The tolerance is clamped to a
    fraction of the pole's distance from clean paper, otherwise a faint veil
    makes its own pole indistinguishable from paper and the metric counts clean
    background as residue (measured null floor 0.018, larger than the value it
    was meant to catch)."""
    if mask.paper is None or mask.veil_offset is None or mask.veil_strength is None:
        return 0.0
    pole = mask.veil_strength * mask.paper + mask.veil_offset
    gray = img.astype(np.float32).mean(2)
    P = float(np.median(mask.paper))
    gap = P - float(np.mean(pole))
    if gap < 4.0:
        return 0.0
    flat = flat_background(img, mask.paper)
    if not flat.any():
        return 0.0
    dev = np.abs(img.astype(np.float32) - pole).mean(2)
    return float((dev[flat] < min(tol, 0.4 * gap)).mean())


def _instances(beta: np.ndarray, thr: Thresholds) -> list[WatermarkInstance]:
    binary = (beta >= thr.mask_alpha_min).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    order = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:40]
    out = []
    for i in order:
        x, y, w, h, area = stats[i]
        sel = labels == i
        out.append(WatermarkInstance(
            instance_id=int(i), bbox=(int(x), int(y), int(x + w), int(y + h)),
            pixel_count=int(area), mean_alpha=float(beta[sel].mean()),
        ))
    return out


def mask_iou(mask: np.ndarray, gt_beta: np.ndarray, threshold: float) -> float:
    a = mask >= threshold
    b = gt_beta >= threshold
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if not a.any() and not b.any() else 0.0
    return float(np.logical_and(a, b).sum() / union)
