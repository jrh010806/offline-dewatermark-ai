"""Shared data contract between the algorithm pipeline and the UI.

Both sides code against these structures; changing a field here is the only
coordination cost the two tracks pay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class MaskSource(StrEnum):
    STRUCTURE = "structure"
    CROSSPAGE = "crosspage"
    PRIOR = "prior"
    MODEL = "model"
    HUMAN = "human"


class EngineKind(StrEnum):
    STRUCTURE = "engine_a_structure"
    RASTER = "engine_b_raster"


class RepairName(StrEnum):
    ALPHA_UNMIX = "alpha_unmix"
    INPAINT = "inpaint"
    PAPER_CLEANUP = "paper_cleanup"
    LAMA = "lama"
    DIFFUSION = "diffusion"
    NONE = "none"


class RegionKind(StrEnum):
    BACKGROUND = "background"
    PHOTO_TEXTURE = "photo_texture"
    TEXT = "text"
    TABLE_LINE = "table_line"
    SEAL = "seal"
    PROTECTED = "protected"


REGION_CODES: dict[RegionKind, int] = {kind: i for i, kind in enumerate(RegionKind)}


class FailureCode(StrEnum):
    NO_WATERMARK_DETECTED = "no_watermark_detected"
    RESIDUAL_WATERMARK = "residual_watermark"
    FIDELITY_BREACH = "fidelity_breach"
    RECOGNITION_REGRESSION = "recognition_regression"
    REPAIR_FAILED = "repair_failed"
    STRUCTURE_NOT_APPLICABLE = "structure_not_applicable"
    OUT_OF_TIME_BUDGET = "out_of_time_budget"
    NEEDS_MANUAL_MASK = "needs_manual_mask"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class Thresholds:
    """The tender never defines the success-rate formula, so these are the
    prototype's own four-condition protocol. residual_coverage_max is the
    fraction of watermark-carrying pixels still holding at least half of their
    veil; it is set from the synthetic set and must be re-fitted against the
    organiser's real samples before it means anything commercially."""

    mask_alpha_min: float = 0.08
    fidelity_mean_delta_max: float = 1.0
    fidelity_p99_delta_max: float = 4.0
    residual_coverage_max: float = 0.15
    ink_iou_min: float = 0.90
    time_budget_ms: int = 120_000
    # LaMa inference knobs. The ONNX weights are fixed; these control how the
    # model is fed, which is what actually changes the result on real samples.
    lama_context_margin: int = 256   # px of clean page the model gets as context
    lama_tile_overlap: int = 64      # px of feathered blend between 512 tiles
    lama_mask_dilate: int = 0        # px to widen the repaint region before inpaint
    watermark_recall_priority: bool = False  # source-supported strokes, including text intersections
    experimental_colour_refinement: bool = False  # opt-in; no independent holdout claim
    experimental_seed_preservation: bool = False  # source-supported old strokes; opt-in candidate
    experimental_oblique_detection: bool = False  # balanced chromatic fallback; opt-in candidate


@dataclass
class PageImage:
    image: np.ndarray
    dpi: int
    doc_id: str = ""
    page_index: int = 0
    source_kind: str = "pdf_page"
    # Original PDF identity survives load_many() renumbering; never infer a path from doc_id.
    source_path: str | None = None
    source_page_index: int | None = None
    source_revision: tuple[int, int] | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def key(self) -> str:
        return f"{self.doc_id or 'single'}#{self.page_index}"


@dataclass
class WatermarkInstance:
    instance_id: int
    bbox: tuple[int, int, int, int]
    pixel_count: int
    mean_alpha: float
    style_signature: str = "default"
    keep: bool = True


@dataclass
class MaskLayer:
    """alpha is the per-pixel watermark coverage estimate in [0, 1]."""

    alpha: np.ndarray
    source: MaskSource = MaskSource.PRIOR
    confidence: float = 0.0
    veil_strength: np.ndarray | None = None
    veil_offset: np.ndarray | None = None
    paper: np.ndarray | None = None
    instances: list[WatermarkInstance] = field(default_factory=list)
    lattice: tuple[int, int] | None = None
    footprint: np.ndarray | None = None
    # OCR/repetition supplies geometry, not a measured alpha-compositing model.
    # Such masks must not run the algebraic inverse with invented veil poles.
    repair_mode: str = "hybrid"
    method: str = "tonal"
    text_keepout: np.ndarray | None = None
    # Measured physical opacity, separate from selectable stroke coverage.
    overlay_opacity: np.ndarray | None = None
    # An explicit human selection grants repair permission, not physical alpha.
    # Automatic layers leave this unset and retain their existing behaviour.
    selection: np.ndarray | None = None
    # Source-only glyph core registered across verified repeats, before any
    # footprint padding. Evidence for the opt-in seed-preservation candidate.
    repeat_stroke_support: np.ndarray | None = None

    def binary(self, threshold: float = Thresholds.mask_alpha_min) -> np.ndarray:
        if self.selection is not None:
            return self.selection.astype(np.uint8) * 255
        return (self.alpha >= threshold).astype(np.uint8) * 255

    def coverage(self, threshold: float = Thresholds.mask_alpha_min) -> float:
        if self.selection is not None:
            return float(self.selection.mean())
        return float((self.alpha >= threshold).mean())

    @property
    def instance_ids(self) -> tuple[int, ...]:
        return tuple(i.instance_id for i in self.instances)


@dataclass
class RegionMap:
    label: np.ndarray

    def mask_of(self, kind: RegionKind) -> np.ndarray:
        return self.label == REGION_CODES[kind]


@dataclass
class VerdictReport:
    page_key: str
    engine: EngineKind
    repairs: list[RepairName]
    conditions: dict[str, bool]
    metrics: dict[str, float]
    failures: list[FailureCode]
    elapsed_ms: int
    stage_ms: dict[str, int] = field(default_factory=dict)
    has_ground_truth: bool = False

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def review_needed(self) -> bool:
        """Unmeasured recognition or residual quality cannot count as success."""
        return "recognition" not in self.conditions or "residual" not in self.conditions

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_key": self.page_key,
            "engine": str(self.engine),
            "repairs": [str(r) for r in self.repairs],
            "conditions": self.conditions,
            "metrics": {k: round(v, 5) if np.isfinite(v) else None for k, v in self.metrics.items()},
            "failures": [str(f) for f in self.failures],
            "passed": self.passed,
            "review_needed": self.review_needed,
            "elapsed_ms": self.elapsed_ms,
            "stage_ms": self.stage_ms,
            "has_ground_truth": self.has_ground_truth,
        }


@dataclass
class PageResult:
    source: PageImage
    output: PageImage
    mask: MaskLayer
    verdict: VerdictReport
    extracted_watermark: np.ndarray | None = None
    clean_reference: np.ndarray | None = None
    regions: RegionMap | None = None
    processing: dict[str, Any] = field(default_factory=dict)
    # Assisted results retain the immutable automatic baseline for separate export.
    automatic_result: PageResult | None = None
    assistance_masks: list[np.ndarray] = field(default_factory=list)
    # -1 = automatic decision; 0 = manually excluded; 1 = manually included.
    # Only committed edits belong in a PageResult; pending edits stay in the UI.
    selection_overrides: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.source.key,
            "mask_source": str(self.mask.source),
            "mask_confidence": round(self.mask.confidence, 4),
            "mask_method": self.mask.method,
            "coverage": round(self.mask.coverage(), 5),
            "instances": len(self.mask.instances),
            "verdict": self.verdict.to_dict(),
            "processing": self.processing,
        }


@dataclass
class StructureProbe:
    applicable: bool
    object_count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobEvent:
    job_id: str
    page_key: str
    state: JobState
    index: int
    total: int
    elapsed_ms: int = 0
    message: str = ""
    passed_count: int = 0
    review_count: int = 0
    failure_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "page_key": self.page_key,
            "state": str(self.state),
            "index": self.index,
            "total": self.total,
            "elapsed_ms": self.elapsed_ms,
            "message": self.message,
            "passed_count": self.passed_count,
            "running_success_rate": round(
                self.passed_count / max(self.index, 1), 4
            ),
            "review_count": self.review_count,
            "failure_codes": self.failure_codes,
        }
