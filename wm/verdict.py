"""Per-page verdict. Mirrors the four-condition protocol proposed in the
roadmap document, so "success rate" is a computed number instead of an
impression. Recognition is only asserted when ground truth exists; without it
the condition is reported as unknown rather than silently passed.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .contract import (EngineKind, FailureCode, MaskLayer, RepairName,
                       Thresholds, VerdictReport)
from . import masks as mask_mod

INK_DELTA = 24


def _ink(img: np.ndarray) -> np.ndarray:
    """OCR-like stroke presence. Deliberately not a global Otsu threshold: a
    global cut moves with the average veil level, so a page with a faint leftover
    pattern and a page with damaged glyphs scored the same for the wrong reason.
    Local adaptive thresholding is what a recogniser actually sees - a surviving
    watermark reads as extra strokes and is charged against the score."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 10)
    return binary.astype(bool)


def build_verdict(
    source: np.ndarray,
    output: np.ndarray,
    mask: MaskLayer,
    engine: EngineKind,
    repairs: list[RepairName],
    elapsed_ms: int,
    stage_ms: dict[str, int],
    thr: Thresholds = Thresholds(),
    clean_reference: np.ndarray | None = None,
    alpha_reference: np.ndarray | None = None,
    veil_reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> VerdictReport:
    metrics: dict[str, float] = {}
    conditions: dict[str, bool] = {}
    failures: list[FailureCode] = []

    coverage = mask.coverage(thr.mask_alpha_min)
    metrics["coverage"] = coverage
    if coverage < 1e-5:
        failures.append(FailureCode.NO_WATERMARK_DETECTED)

    dilated = cv2.dilate(mask.binary(thr.mask_alpha_min), np.ones((7, 7), np.uint8))
    outside = ~dilated.astype(bool)
    delta = np.abs(output.astype(np.int16) - source.astype(np.int16)).mean(2)
    metrics["fidelity_mean_delta"] = float(delta[outside].mean()) if outside.any() else 0.0
    metrics["fidelity_p99_delta"] = float(np.percentile(delta[outside], 99)) if outside.any() else 0.0
    conditions["fidelity"] = bool(
        metrics["fidelity_mean_delta"] <= thr.fidelity_mean_delta_max
        and metrics["fidelity_p99_delta"] <= thr.fidelity_p99_delta_max
    )
    if not conditions["fidelity"]:
        failures.append(FailureCode.FIDELITY_BREACH)

    if alpha_reference is not None and clean_reference is not None:
        residual_cov = mask_mod.residual_visible_fraction(source, output, clean_reference,
                                                          alpha_reference=alpha_reference)
        metrics["residual_basis"] = 1.0
    else:
        residual_cov = mask_mod.residual_background_fraction(output, mask)
        metrics["residual_basis"] = 0.0
    metrics["residual_coverage"] = residual_cov
    conditions["residual"] = bool(residual_cov <= thr.residual_coverage_max)
    if not conditions["residual"]:
        failures.append(FailureCode.RESIDUAL_WATERMARK)

    if alpha_reference is not None:
        metrics["mask_iou"] = mask_mod.mask_iou(mask.alpha, alpha_reference, thr.mask_alpha_min)
        metrics["mask_precision"] = _precision(mask.alpha, alpha_reference, thr)
        metrics["mask_recall"] = _recall(mask.alpha, alpha_reference, thr)
    if veil_reference is not None:
        s_gt, t_gt = veil_reference
        metrics["veil_s_error"] = float(np.abs(mask.veil_strength - s_gt).mean())
        metrics["veil_t_error"] = float(np.abs(mask.veil_offset - t_gt).mean())

    if clean_reference is not None:
        # The region is taken from the ground truth, never from our own prediction:
        # a prediction-dependent region makes two strategies incomparable, because
        # tightening the mask shrinks the denominator as well as the errors.
        base = (alpha_reference > 0.5) if alpha_reference is not None else dilated.astype(bool)
        gt_mask = cv2.dilate(base.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        metrics["psnr"] = float(peak_signal_noise_ratio(clean_reference, output, data_range=255))
        metrics["ssim"] = float(structural_similarity(clean_reference, output, channel_axis=2))
        metrics["psnr_in_mask"] = float(peak_signal_noise_ratio(
            clean_reference[gt_mask], output[gt_mask], data_range=255))
        gt_ink, out_ink = _ink(clean_reference), _ink(output)
        if not gt_ink[gt_mask].any():
            # No body text to preserve in this region: recognition is vacuously
            # satisfied and any leftover veil is judged by the residual condition,
            # not by a meaningless 0/0 IoU.
            metrics["ink_iou"] = 1.0
        else:
            agree = np.logical_and(gt_ink, out_ink)[gt_mask].sum()
            metrics["ink_iou"] = float(agree / max(np.logical_or(gt_ink, out_ink)[gt_mask].sum(), 1))
        # Split the same disagreement by who is at fault: strokes we damaged where
        # no watermark sits (clause 3) versus watermark we left behind (clause 2).
        if alpha_reference is not None:
            beside = gt_mask & (alpha_reference == 0)
            g2, o2 = gt_ink[beside], out_ink[beside]
            metrics["ink_damage"] = float(np.logical_and(g2, ~o2).sum() / max(g2.sum(), 1))
            on = gt_mask & (alpha_reference > 0)
            g3, o3 = gt_ink[on], out_ink[on]
            metrics["ink_residue"] = float(np.logical_and(o3, ~g3).sum() / max(o3.sum(), 1))
        conditions["recognition"] = bool(metrics["ink_iou"] >= thr.ink_iou_min)
        if not conditions["recognition"]:
            failures.append(FailureCode.RECOGNITION_REGRESSION)
    else:
        metrics["ink_iou"] = float("nan")

    conditions["time_budget"] = bool(elapsed_ms <= thr.time_budget_ms)
    if not conditions["time_budget"]:
        failures.append(FailureCode.OUT_OF_TIME_BUDGET)

    if mask.confidence < 0.12:
        failures.append(FailureCode.NEEDS_MANUAL_MASK)

    return VerdictReport(
        page_key="", engine=engine, repairs=repairs, conditions=conditions,
        metrics=metrics, failures=failures, elapsed_ms=elapsed_ms,
        stage_ms=stage_ms, has_ground_truth=clean_reference is not None,
    )


def _precision(beta: np.ndarray, gt: np.ndarray, thr: Thresholds) -> float:
    a = beta >= thr.mask_alpha_min
    b = gt >= thr.mask_alpha_min
    return float(np.logical_and(a, b).sum() / a.sum()) if a.any() else 1.0


def _recall(beta: np.ndarray, gt: np.ndarray, thr: Thresholds) -> float:
    b = gt >= thr.mask_alpha_min
    a = beta >= thr.mask_alpha_min
    return float(np.logical_and(a, b).sum() / b.sum()) if b.any() else 1.0
