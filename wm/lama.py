"""LaMa (Large Mask Inpainting) deep-learning repair backend.

Reconstructs pixels where the watermark sits on top of text — the opaque core
(coverage >= 0.97) that algebraic un-mixing cannot recover because the original
ink colour is lost under the veil.

Uses the ONNX export of LaMa from ``opencv/inpainting_lama`` on Hugging Face,
run through onnxruntime. No torch dependency. The CPU execution provider is
the always-available baseline; set ``WM_LAMA_CUDA=1`` on a machine with an
NVIDIA GPU (the tender's RTX A6000) to prefer the CUDA provider.

The ONNX file (~93 MB) is resolved from ``WM_LAMA_ONNX``, then the project
``models/`` directory, then the Hugging Face cache. A missing or unloadable
model is a hard error, never a silent downgrade: on the organisers' offline
workstation a quiet fallback to ``cv2.inpaint`` would silently degrade the
result and the difference would be invisible from a local run.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .contract import RepairName, Thresholds

log = logging.getLogger(__name__)

_MODEL_SIZE = 512
_HF_REPO = "opencv/inpainting_lama"
_HF_FILE = "inpainting_lama_2025jan.onnx"


class LamaModelMissing(RuntimeError):
    """The LaMa ONNX file cannot be found or loaded — repair must not proceed
    on a silent fallback."""


def _resolve_model_path() -> str:
    """Find the ONNX model file.

    Search order:
    1. Environment variable ``WM_LAMA_ONNX`` (explicit path)
    2. ``models/`` directory in project root (offline deployment bundle)
    3. Hugging Face cache / download (dev machines with network)
    Raises LamaModelMissing when none of them yields a file.
    """
    env = os.environ.get("WM_LAMA_ONNX")
    if env and Path(env).is_file():
        return env

    local = Path(__file__).resolve().parent.parent / "models" / _HF_FILE
    if local.is_file():
        return str(local)

    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(_HF_REPO, _HF_FILE)
    except Exception as exc:
        raise LamaModelMissing(
            f"LaMa ONNX model {_HF_FILE} not found (WM_LAMA_ONNX unset, "
            f"models/ empty, HF cache unusable: {exc}). For offline "
            f"deployment place the file in the project models/ directory."
        ) from exc


def _providers() -> list[str]:
    if os.environ.get("WM_LAMA_CUDA") == "1":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        log.warning("WM_LAMA_CUDA=1 but CUDAExecutionProvider unavailable; "
                    "running on CPU")
    return ["CPUExecutionProvider"]


_sess: ort.InferenceSession | None = None


def _load_session() -> ort.InferenceSession:
    global _sess
    if _sess is not None:
        return _sess
    path = _resolve_model_path()
    opts = ort.SessionOptions()
    # The exported graph carries unused initializers; ORT drops them with a
    # warning per node — harmless noise, the repair path logs its own summary.
    opts.log_severity_level = 3
    # Large host-wide default thread pools slow these 512px FFT tiles and can
    # starve the UI/OCR. Keep a small, configurable CPU pool.
    opts.intra_op_num_threads = max(1, int(os.environ.get("WM_ORT_THREADS", "4")))
    opts.inter_op_num_threads = 1
    try:
        _sess = ort.InferenceSession(path, sess_options=opts,
                                     providers=_providers())
    except Exception as exc:
        raise LamaModelMissing(f"Failed to load LaMa ONNX model from {path}") from exc
    log.info("LaMa ONNX model loaded from %s, providers=%s",
             path, _sess.get_providers())
    return _sess


def is_available() -> bool:
    """Health check for install scripts and the self-check page: True when the
    ONNX model file can be found and loaded. Loading never happens silently on
    the repair path — see ``inpaint``."""
    try:
        _load_session()
        return True
    except LamaModelMissing:
        return False


def _feather_weight(size: int, overlap: int) -> np.ndarray:
    """1-D weight ramp: 1.0 in the centre, linearly fading to 0 at the edges
    of the overlap zone. Used for seamless tile blending."""
    w = np.ones(size, dtype=np.float32)
    if overlap <= 0:
        return w
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    w[:overlap] = ramp
    w[-overlap:] = ramp[::-1]
    return w


def _run_tile(sess: ort.InferenceSession, img_tile: np.ndarray,
              mask_tile: np.ndarray) -> np.ndarray:
    """Run the ONNX model on a single 512x512 tile, preserving channel order."""
    img_blob = np.ascontiguousarray(
        img_tile.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
    mask_blob = np.ascontiguousarray(
        (mask_tile > 0).astype(np.float32)[None, None])
    output = sess.run(None, {"image": img_blob, "mask": mask_blob})[0]
    return np.transpose(output[0], (1, 2, 0)).clip(0, 255).astype(np.uint8)


def inpaint(img: np.ndarray, binary_mask: np.ndarray,
            thr: Thresholds | None = None) -> np.ndarray:
    """LaMa inpainting via onnxruntime.

    The work is confined to the mask's bounding box plus a margin: a page-sized
    mask typically covers a couple of percent of the page, and tiling the whole
    page costs an order of magnitude more time for pixels the model cannot
    influence. The margin is what the model uses as context, and the box is
    snapped to a multiple of the tile size so the interior stays on the native
    grid.

    Small crops (both dims <= 512) are padded to the model's native 512x512.
    Larger ones are split into overlapping 512x512 tiles, each tile is inferred
    independently, and the results are blended with feathered weights so there
    are no visible seams.

    A model that cannot be found or loaded raises ``LamaModelMissing`` — the
    repair never silently degrades to a cheaper inpainter.
    """
    thr = thr or Thresholds()
    if thr.lama_mask_dilate > 0:
        k = 2 * int(thr.lama_mask_dilate) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel) > 0
    if not binary_mask.any():
        return img.copy()

    sess = _load_session()

    h, w = img.shape[:2]
    S = _MODEL_SIZE
    margin = int(thr.lama_context_margin)
    ys, xs = np.nonzero(binary_mask)
    y0 = max(0, int(ys.min()) - margin)
    x0 = max(0, int(xs.min()) - margin)
    y1 = min(h, int(ys.max()) + 1 + margin)
    x1 = min(w, int(xs.max()) + 1 + margin)
    if (y0, x0) != (0, 0):
        y0 = max(0, y0 - (y1 - y0) % S)
        x0 = max(0, x0 - (x1 - x0) % S)
    crop_img = img[y0:y1, x0:x1]
    crop_mask = binary_mask[y0:y1, x0:x1]
    out = img.copy()
    restored = _inpaint_region(sess, crop_img, crop_mask, thr)
    # Model output and interpolation are allowed to affect only the requested
    # repair pixels. Context is input, not permission to rewrite the crop.
    target = out[y0:y1, x0:x1]
    target[crop_mask] = restored[crop_mask]
    return out


def _inpaint_region(sess: ort.InferenceSession, img: np.ndarray,
                    binary_mask: np.ndarray,
                    thr: Thresholds | None = None) -> np.ndarray:
    thr = thr or Thresholds()
    h, w = img.shape[:2]
    mask_u8 = (binary_mask > 0.5).astype(np.uint8)
    S = _MODEL_SIZE

    if h <= S and w <= S:
        # Padding preserves glyph geometry and avoids resampling small scans.
        img_r = cv2.copyMakeBorder(img, 0, S-h, 0, S-w, cv2.BORDER_REFLECT_101)
        mask_r = cv2.copyMakeBorder(mask_u8, 0, S-h, 0, S-w, cv2.BORDER_CONSTANT)
        return _run_tile(sess, img_r, mask_r)[:h, :w]

    overlap = int(np.clip(thr.lama_tile_overlap, 0, S // 2))
    stride = S - overlap
    acc = np.zeros((h, w, 3), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    wy = _feather_weight(S, overlap)
    wx = _feather_weight(S, overlap)
    ys = list(range(0, max(1, h - S + 1), stride))
    if ys[-1] + S < h:
        ys.append(h - S)
    xs = list(range(0, max(1, w - S + 1), stride))
    if xs[-1] + S < w:
        xs.append(w - S)

    for y0 in ys:
        for x0 in xs:
            tile_mask = mask_u8[y0:y0 + S, x0:x0 + S]
            th, tw = tile_mask.shape[:2]
            # Feather only where another tile can contribute. At the outer
            # image/crop boundary there is no neighbour; a zero ramp there
            # used to copy the original watermark back into the output.
            local_y, local_x = wy[:th].copy(), wx[:tw].copy()
            if y0 == 0:
                local_y[:overlap] = 1
            if y0 + th == h and overlap:
                local_y[-overlap:] = 1
            if x0 == 0:
                local_x[:overlap] = 1
            if x0 + tw == w and overlap:
                local_x[-overlap:] = 1
            tile_w_crop = np.outer(local_y, local_x).astype(np.float64)

            if not tile_mask.any():
                acc[y0:y0 + th, x0:x0 + tw] += img[y0:y0 + th, x0:x0 + tw].astype(np.float64) * tile_w_crop[..., None]
                weight[y0:y0 + th, x0:x0 + tw] += tile_w_crop
                continue

            tile_img = img[y0:y0 + S, x0:x0 + S]

            if tile_img.shape[0] != S or tile_img.shape[1] != S:
                tile_img = cv2.copyMakeBorder(tile_img, 0, S-th, 0, S-tw,
                                             cv2.BORDER_REFLECT_101)
                tile_mask = cv2.copyMakeBorder(tile_mask, 0, S-th, 0, S-tw,
                                              cv2.BORDER_CONSTANT)

            result_tile = _run_tile(sess, tile_img, tile_mask)

            if result_tile.shape[0] != th or result_tile.shape[1] != tw:
                result_tile = result_tile[:th, :tw]

            acc[y0:y0 + th, x0:x0 + tw] += result_tile.astype(np.float64) * tile_w_crop[..., None]
            weight[y0:y0 + th, x0:x0 + tw] += tile_w_crop

    # A tile's feather ramps to zero at its outer edge, and only an adjacent tile
    # restores the weight there. At the border of the region - the page edge, or
    # the edge of a cropped mask bounding box - nothing does, so dividing there
    # amplifies a few accumulated pixels into a visible black seam. Those pixels
    # were never covered by any tile with real weight, so the input is kept.
    weight = np.maximum(weight, 1e-8)
    blended = acc / weight[..., None]
    return np.where((weight > 1e-3)[..., None], blended,
                    img.astype(np.float64)).clip(0, 255).astype(np.uint8)


OPAQUE = 0.97
DENOM_FLOOR = 0.06


def _degenerate_limit(mask) -> float:
    """Coverage at which the inverse stops being solvable.

    Below this the un-mix is exact; at or above it the denominator hits the floor
    the un-mix uses for conditioning and the result is a function of the floor
    rather than of the page. A fixed 0.97 is wrong for a flat-colour overlay
    (s = 0), where the denominator is 1 - beta and the floor bites at 0.94.
    """
    s = float(np.min(mask.veil_strength))
    return min(OPAQUE, (1.0 - DENOM_FLOOR) / max(1.0 - s, 1e-3))


def repair_with_lama(img: np.ndarray, mask, thr) -> tuple[np.ndarray, list[RepairName]]:
    """Full repair pipeline: alpha un-mix → halo suppression → LaMa over the footprint.

    Three stages, each handling a different coverage regime:

    1. **Alpha un-mix**: deterministic algebraic inversion. Handles low-to-medium
       coverage (faint watermark on paper). Cannot hallucinate. Uses the
       conservative anchored beta so text edges are never algebraically disturbed.
    2. **Ink halo suppression**: zeros coverage in a 3-5 px band around real ink
       glyphs so we never erase stroke edges during un-mixing.
    3. **LaMa inpainting**: deep-learning reconstruction over the located
       footprint, which carries the paper texture the algebra cannot invent.
    """
    from .masks import ink_halo

    used: list[RepairName] = []
    if mask.coverage(thr.mask_alpha_min) <= 0:
        return img.copy(), used

    if mask.repair_mode == 'corner_cleanup':
        out = img.copy()
        out[mask.footprint] = np.clip(mask.paper[mask.footprint], 0, 255).astype(np.uint8)
        return out, [RepairName.PAPER_CLEANUP]

    if mask.repair_mode == 'paper_repeat_cleanup':
        from .paper_repeat import restore
        return restore(img, mask), [RepairName.PAPER_CLEANUP]

    if mask.repair_mode == 'bright_paper_cleanup':
        from .contrast_text import restore_bright
        return restore_bright(img, mask, thr), [RepairName.LAMA]

    if mask.repair_mode == 'shadow_paper_cleanup':
        from .observed_text import restore_shadow
        return restore_shadow(img, mask), [RepairName.PAPER_CLEANUP]

    if mask.repair_mode == 'shadow_layer_inverse':
        from .shadow_layers import restore
        return restore(img, mask, thr.mask_alpha_min), [RepairName.ALPHA_UNMIX]

    if mask.repair_mode == 'thin_photo_text':
        region = mask.footprint
        distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
        if np.percentile(distance[region], 95) <= 4:
            # Thin confirmed strokes have nearby source observations on both
            # sides. Local interpolation avoids inventing texture in a broad
            # model tile, while larger glyphs retain the neural repair path.
            target = region
            if thr.lama_mask_dilate > 0:
                k = 2*int(thr.lama_mask_dilate)+1
                target = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k, k))) > 0
            out = cv2.inpaint(img, target.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
            out[~target] = img[~target]
            return out, [RepairName.INPAINT]
        return inpaint(img, region, thr), [RepairName.LAMA]

    if mask.repair_mode == 'photo_repeat_inverse':
        if (mask.overlay_opacity is None or mask.overlay_opacity.shape != img.shape[:2]
                or not np.isfinite(mask.overlay_opacity).all()
                or np.any(mask.overlay_opacity < 0) or np.any(mask.overlay_opacity > .6)):
            raise ValueError('Repeated-photo inverse requires bounded measured opacity')
        out = _alpha_unmix(img, mask, coverage=mask.overlay_opacity)
        out[~mask.footprint] = img[~mask.footprint]
        return out, [RepairName.ALPHA_UNMIX]

    if mask.repair_mode == 'texture_overlay':
        out = _alpha_unmix(img, mask)
        out[~mask.footprint] = img[~mask.footprint]
        return out, [RepairName.ALPHA_UNMIX]

    if mask.repair_mode == 'periodic_paper_overlay':
        out = _alpha_unmix(img, mask)
        # Residual JPEG colour/antialiasing on otherwise blank paper is not
        # printed content. Fill only the verified dots away from dark strokes
        # and coloured seals; their intersections retain the algebraic inverse.
        gray = out.astype(np.float32).mean(2)
        paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        colour = out.max(2).astype(np.float32)-out.min(2)
        protected = cv2.dilate(((gray < .8*paper) | (colour > 25)).astype(np.uint8),
                               np.ones((5, 5), np.uint8)) > 0
        fill = mask.footprint & ~protected
        if fill.any():
            fixed = cv2.inpaint(out, fill.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
            out[fill] = fixed[fill]
        out[~mask.footprint] = img[~mask.footprint]
        return out, [RepairName.ALPHA_UNMIX, RepairName.INPAINT]

    if mask.repair_mode == "periodic_overlay":
        out = _alpha_unmix(img, mask)
        region = mask.footprint
        # JPEG chroma subsampling leaves colour specks after a luminance inverse.
        # Interpolate only chroma inside the tiny verified dots; keep the recovered
        # source luminance and every pixel outside the footprint exactly intact.
        background = cv2.inpaint(img, region.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
        channels = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        predicted = cv2.cvtColor(background, cv2.COLOR_RGB2YCrCb)
        channels[region, 1:] = predicted[region, 1:]
        restored = cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB)
        out = img.copy()
        out[region] = restored[region]
        return out, [RepairName.ALPHA_UNMIX, RepairName.INPAINT]

    if mask.repair_mode in ("document_overlay", "coloured_document_overlay", "document_overlay_full"):
        out = _alpha_unmix(img, mask)
        if (mask.repair_mode == "document_overlay_full" and mask.text_keepout is not None
                and float(np.min(mask.veil_strength)) < 0.4):
            # Heavy veils only: a pixel that brightens implausibly under a weak
            # alpha is a misfit on clean antialiasing, not a veil.
            over = (out.astype(np.int16) - img.astype(np.int16)).mean(2) > 40
            revert = mask.text_keepout & over & (mask.alpha > 0) & (mask.alpha < 0.9)
            out[revert] = img[revert]
        paper = float(np.median(mask.paper))
        covered = float(np.mean(mask.veil_strength)*paper + np.mean(mask.veil_offset))
        # Full un-mixing at text intersections is only safe for light veils:
        # below s≈0.4 the inverse amplifies fit noise into real glyph damage
        # (measured: heavy_dark text damage jumps from ~10% to ~40%). Heavy
        # veils keep the conservative dark guard instead of trusting the fit.
        apply_dark_guard = (mask.repair_mode != "document_overlay_full"
                            or float(np.min(mask.veil_strength)) < 0.4)
        if apply_dark_guard:
            dark = (img.mean(2) < covered*.8) & (mask.alpha > 0)
            out[dark] = np.minimum(out[dark], img[dark])
        # A fitted white-paper overlay is invertible. Do not ask an inpainter
        # to redraw text or table rules we can preserve algebraically.
        # Keep dark stroke cores and their one-pixel antialiasing rim even where
        # the watermark crosses an OCR line; the remaining paper can be cleaned
        # without generating new letter shapes.
        core = out.mean(2) < .4*paper
        if mask.text_keepout is not None:
            if mask.repair_mode == "document_overlay_full" and float(np.min(mask.veil_strength)) >= 0.4:
                # Light veils: the inverse should leave a bimodal page (paper
                # vs body ink). Estimate the body-ink level on uncovered text
                # and only protect strokes near it; faint leftover veil above
                # that level is flattened instead of being protected as text.
                body_pool = img.mean(2)[mask.text_keepout & ~mask.footprint
                                        & (img.mean(2) < .5*paper)]
                body_ink = float(np.percentile(body_pool, 20)) if body_pool.size >= 50 else 0.0
                cut = min(.85*paper, body_ink*1.5 + 25)
                core |= mask.text_keepout & (out.mean(2) < cut)
            else:
                # Light body strokes and their antialiasing rims must not be
                # whitened by the paper fill either; only confident paper inside
                # the OCR zones may be flattened after the inverse.
                core |= mask.text_keepout & (out.mean(2) < (.85 if mask.repair_mode == "coloured_document_overlay" else .92)*paper)
        cores = core.astype(np.uint8)
        protected = cv2.dilate(cores, np.ones((3,3),np.uint8)) > 0
        from .paper_repeat import table_rules
        rules = table_rules(img) & mask.footprint
        protected |= rules
        out[rules] = np.minimum(out[rules], img[rules])
        paper_pixels = mask.footprint & ~protected
        out[paper_pixels] = np.clip(mask.paper[paper_pixels], 0, 255).astype(np.uint8)
        # Numerical alpha can retain a faint tail outside the selected strokes.
        # The measured inverse still needs the explicit repair permission mask.
        out[~mask.footprint] = img[~mask.footprint]
        return out, [RepairName.ALPHA_UNMIX]

    if mask.repair_mode in ("inpaint", "photo_overlay"):
        region = mask.footprint if mask.footprint is not None else mask.binary(thr.mask_alpha_min) > 0
        if region.any():
            out = inpaint(img, region, thr)
            if mask.repair_mode == "photo_overlay":
                from .photo_overlay import refine
                detail = refine(img, out, region, mask.veil_strength, mask.veil_offset)
                # Keep any extra repair pixels explicitly requested through the
                # mask-dilation control; source detail uses the original core.
                out[region] = detail[region]
                return out, [RepairName.LAMA, RepairName.ALPHA_UNMIX]
            return out, [RepairName.LAMA]
        return img.copy(), used

    P = float(np.median(mask.paper)) if mask.paper is not None else 255.0

    # Where the overlay covers a pixel completely the inverse is degenerate: the
    # observed value no longer depends on what was underneath, and dividing by the
    # floored denominator returns a function of the floor rather than of the page.
    # Those pixels leave the algebra and are reconstructed instead.
    opaque = mask.alpha >= _degenerate_limit(mask)
    if mask.footprint is not None:
        opaque &= mask.footprint

    # Alpha un-mix uses conservative anchored coverage (protects text edges)
    coverage = mask.alpha.copy()
    coverage[ink_halo(img.astype(np.float32).mean(2), P)] = 0.0
    coverage[opaque] = 0.0
    out = _alpha_unmix(img, mask, coverage)
    used.append(RepairName.ALPHA_UNMIX)

    # The inpainter gets the located footprint, not just the opaque core: the
    # skirt around a stamp stroke is where the un-mix is exact but the paper
    # texture is not, and repainting the whole region is what makes the removal
    # read as clean paper. Safety comes from the footprint itself being bounded by
    # the OCR keep-out, not from shrinking it here.
    region = mask.footprint if mask.footprint is not None else opaque
    if region.mean() > 0.25:
        # Never gamble more than a quarter of a page on generative repainting:
        # keep the algebraic inverse and flag the page for manual review.
        mask.confidence = min(mask.confidence, 0.05)
        return out, used
    if region.mean() > 1e-5:
        out = inpaint(out, region, thr)
        used.append(RepairName.LAMA)
    return out, used


def _alpha_unmix(img: np.ndarray, mask, coverage: np.ndarray | None = None) -> np.ndarray:
    """Deterministic inverse veil: B = (I - beta*t) / (1 - beta*(1-s))."""
    beta = (mask.alpha if coverage is None else coverage)[..., None].astype(np.float32)
    s = mask.veil_strength.reshape(1, 1, 3)
    t = mask.veil_offset.reshape(1, 1, 3)
    denom = np.maximum(1.0 - beta * (1.0 - s), DENOM_FLOOR)
    return np.clip((img.astype(np.float32) - beta * t) / denom, 0, 255).astype(np.uint8)
