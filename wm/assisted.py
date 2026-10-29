"""Explicit human search hints; rectangles never grant repaint permission.

This entry is used only after an automatic result exists. Detection observes
the current output, references are never accepted, and confirmation is a
separate operation. Existing automatic detection and routing are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np

from . import masks, recall, repair, verdict
from .contract import EngineKind, MaskLayer, MaskSource, PageResult, Thresholds


def image_hash(image: np.ndarray) -> str:
    return sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


@dataclass
class StrokeProposal:
    rectangle: tuple[int, int, int, int]
    base_hash: str
    layer: MaskLayer
    thresholds: Thresholds
    detection_ms: int
    detector_method: str
    selection_hash: str


def find_strokes(result: PageResult, rectangle, thr: Thresholds,
                 *, close_px: int = 11, use_ocr: bool = True,
                 style: str | None = None) -> StrokeProposal:
    image = result.output.image
    h, w = image.shape[:2]
    if len(rectangle) != 4 or not all(np.isfinite(v) for v in rectangle):
        raise ValueError("圈选坐标无效，请重新圈选。")
    x0, y0, x1, y1 = map(int, rectangle)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("圈选必须位于图片内。")
    if min(x1-x0, y1-y0) < 8:
        raise ValueError("圈选太小，请包含完整的残留笔画。")
    # The preview has an exact permission mask. Widening it after confirmation
    # would authorize pixels that the user never previewed, so this mode uses 0.
    settings = replace(thr, lama_mask_dilate=0)
    start = perf_counter()
    pad = 64
    left, top = max(0, x0-pad), max(0, y0-pad)
    right, bottom = min(w, x1+pad), min(h, y1+pad)
    crop = image[top:bottom, left:right].copy()
    local = masks.estimate_single_page(crop, settings, close_px=close_px,
                                       use_ocr=use_ocr, style=style)
    domain = np.zeros(crop.shape[:2], bool)
    domain[y0-top:y1-top, x0-left:x1-left] = True
    seed = local.binary(settings.mask_alpha_min) > 0
    # Tonal candidates and padded footprints may contain background. Require
    # the existing positive colour/local contrast stroke evidence as well.
    height = max(12., min(x1-x0, y1-y0) / 3.)
    supported = recall._pixel_strokes(crop, domain, seed & domain, height)
    chosen = seed & supported & domain
    alpha = np.zeros((h, w), np.float32)
    alpha[top:bottom, left:right] = np.where(chosen, local.alpha, 0.)
    selected = alpha >= settings.mask_alpha_min
    # Preserve invertible routes where their measured fields are available.
    def expand(value, *, background=0):
        if value is None or value.shape[:2] != crop.shape[:2]:
            return deepcopy(value)
        full = np.full((h, w) + value.shape[2:], background, dtype=value.dtype)
        full[top:bottom, left:right] = value
        return full
    paper_fill = float(np.median(local.paper)) if local.paper is not None else 0
    layer = replace(local, alpha=alpha, footprint=selected,
                    paper=expand(local.paper, background=paper_fill),
                    text_keepout=expand(local.text_keepout),
                    overlay_opacity=expand(local.overlay_opacity), instances=[])
    # A non-invertible generic layer has no validated numerical veil model.
    if layer.repair_mode == 'hybrid':
        layer = replace(layer, repair_mode='inpaint',
                        veil_strength=np.ones(3, np.float32),
                        veil_offset=np.zeros(3, np.float32))
    return StrokeProposal((x0, y0, x1, y1), image_hash(image), layer,
                          settings, round((perf_counter()-start)*1000), local.method,
                          sha256(selected.tobytes()).hexdigest())


def apply_proposal(result: PageResult, proposal: StrokeProposal) -> PageResult:
    before = result.output.image
    if image_hash(before) != proposal.base_hash:
        raise ValueError("图片已变化，请重新圈选并识别。")
    selected = proposal.layer.binary(proposal.thresholds.mask_alpha_min) > 0
    if sha256(selected.tobytes()).hexdigest() != proposal.selection_hash:
        raise ValueError("候选笔画已变化，请重新预览并确认。")
    x0, y0, x1, y1 = proposal.rectangle
    permitted = np.zeros(selected.shape, bool)
    permitted[y0:y1, x0:x1] = True
    if np.any(selected & ~permitted) or not np.array_equal(selected, proposal.layer.footprint):
        raise ValueError("候选选区与预览不一致，请重新识别。")
    if not selected.any():
        raise ValueError("圈内未找到有足够证据的水印笔画，图片保持不变。")
    start = perf_counter()
    output, repairs = repair.repair(before.copy(), deepcopy(proposal.layer), proposal.thresholds)
    if output.shape != before.shape or output.dtype != before.dtype:
        raise ValueError("修复结果尺寸或格式异常，原结果已保留。")
    changed = np.any(output != before, axis=2)
    if np.any(changed & ~selected):
        # Reject the result, never hide a boundary breach by copying pixels back.
        raise ValueError("修复改动超出预览笔画，已拒绝本次结果并保留原图。")
    repair_ms = round((perf_counter()-start)*1000)
    automatic = result.automatic_result or result
    original_threshold = result.processing.get('thresholds', {}).get('mask_alpha_min', .08)
    previous = result.mask.binary(original_threshold) > 0
    combined = previous | selected
    footprint = combined.copy()
    if result.mask.footprint is not None:
        footprint |= result.mask.footprint
    combined_layer = MaskLayer(combined.astype(np.float32), source=MaskSource.HUMAN,
                               confidence=.8, footprint=footprint,
                               method='human_search_strokes', repair_mode='assisted_composite')
    report = verdict.build_verdict(result.source.image, output, combined_layer,
                                  EngineKind.RASTER, repairs,
                                  result.verdict.elapsed_ms + proposal.detection_ms + repair_ms,
                                  dict(assist_mask_ms=proposal.detection_ms, assist_repair_ms=repair_ms),
                                  proposal.thresholds)
    report.page_key = result.source.key
    processing = deepcopy(result.processing)
    history = processing.setdefault('human_assistance', [])
    history.append(dict(rectangle=list(proposal.rectangle),
                        confirmed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                        base_output_sha256=proposal.base_hash,
                        candidate_sha256=sha256(selected.tobytes()).hexdigest(),
                        selected_pixels=int(selected.sum()), changed_pixels=int(changed.sum()),
                        outside_selected_changed_pixels=0, detector_method=proposal.detector_method,
                        repair_mode=proposal.layer.repair_mode,
                        thresholds=asdict(proposal.thresholds),
                        evidence='current_output_pixel_contrast_and_existing_detector',
                        quality_verified=False))
    processing.update(mode='human_assisted', thresholds=asdict(proposal.thresholds))
    return PageResult(result.source, replace(result.output, image=output), combined_layer,
                      report, processing=processing, automatic_result=automatic,
                      assistance_masks=[*result.assistance_masks, selected.copy()])
