"""Pipeline: structure probe -> coverage estimate -> veil inverse -> verdict."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import documents, masks, ocr, repair, structure, verdict as verdict_mod
from .contract import (EngineKind, MaskLayer, PageImage, PageResult, Thresholds)

GroundTruth = dict[str, tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]


def _keepout(page: PageImage, use_ocr: bool) -> np.ndarray | None:
    """OCR body-text zones, or None when the channel is off or the model missing.

    Detection is unchanged in the None case, so a machine without the OCR
    dependency still runs the classic channel end to end.
    """
    return ocr.text_zones(page.image) if use_ocr else None


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def extract_watermark(mask: MaskLayer) -> np.ndarray:
    s = mask.veil_strength.astype(np.float32)
    color = np.clip(mask.veil_offset / np.maximum(1.0 - s, 1e-3), 0, 255)
    rgba = np.empty((*mask.alpha.shape, 4), np.uint8)
    rgba[..., :3] = color.reshape(1, 1, 3).astype(np.uint8)
    rgba[..., 3] = (mask.alpha * 255).astype(np.uint8)
    return rgba


def _process_one(page: PageImage, mask: MaskLayer, thr: Thresholds,
                 gt: GroundTruth | None = None, mask_ms: int = 0) -> PageResult:
    """Run repair + verdict on a single page. Shared by batch and streaming paths."""
    local_t0 = time.perf_counter()
    output, used = repair.repair(page.image, mask, thr)
    entry = gt.get(page.key) if gt else None
    clean_ref, alpha_ref, veil_ref = entry if entry else (None, None, None)
    report = verdict_mod.build_verdict(
        source=page.image, output=output, mask=mask,
        engine=EngineKind.RASTER, repairs=used,
        elapsed_ms=mask_ms + _ms(local_t0),
        stage_ms={"mask_ms": mask_ms, "repair_ms": _ms(local_t0)},
        thr=thr, clean_reference=clean_ref, alpha_reference=alpha_ref,
        veil_reference=veil_ref,
    )
    report.page_key = page.key
    return PageResult(
        source=page,
        output=PageImage(image=output, dpi=page.dpi, doc_id=page.doc_id,
                         page_index=page.page_index, source_kind=page.source_kind),
        mask=mask, verdict=report,
        extracted_watermark=extract_watermark(mask),
        clean_reference=clean_ref,
        processing={"thresholds": asdict(thr)},
    )


def _run_raster(pages: list[PageImage], thr: Thresholds, gt: GroundTruth | None,
                force_single: bool = False, use_ocr: bool = True,
                use_structure: bool = True, style: str | None = None) -> list[PageResult]:
    structural = {id(p): structure.process_text_page(p, thr) if use_structure else None for p in pages}
    raster_pages = [p for p in pages if structural[id(p)] is None]
    images = [p.image for p in raster_pages]
    t0 = time.perf_counter()
    use_cross = (not force_single) and len(images) >= 2 and \
        all(i.shape == images[0].shape for i in images)
    shared = None
    if use_cross:
        try:
            shared = masks.estimate_crosspage(images, thr)
        except Exception as exc:
            logging.getLogger(__name__).warning("Cross-page prior unavailable; using single-image detection: %s", exc)
    shared_ms = _ms(t0)

    out: list[PageResult] = []
    for page in pages:
        if structural[id(page)] is not None:
            out.append(structural[id(page)])
            continue
        t0 = time.perf_counter()
        single = masks.estimate_single_page(page.image, thr,
                                           use_ocr=use_ocr, style=style)
        mask = masks.with_prior(single, shared, thr)
        mask_ms = _ms(t0) + shared_ms // max(len(pages), 1)
        result = _process_one(page, mask, thr, gt, mask_ms=mask_ms)
        out.append(result)
    return out


def iter_raster(pages_iter: Iterator[PageImage], thr: Thresholds,
                total: int, use_ocr: bool = True) -> Iterator[PageResult]:
    """Streaming single-page path: yields one result per page, memory-flat.

    Cross-page estimation is *not* available here — each page gets its own
    single-page mask. Use ``_run_raster`` (batch) when the cross-page prior
    matters (same watermark across many aligned pages).
    """
    for page in pages_iter:
        structural = structure.process_text_page(page, thr)
        if structural is not None:
            yield structural
            continue
        t0 = time.perf_counter()
        mask = masks.estimate_single_page(page.image, thr, use_ocr=use_ocr)
        yield _process_one(page, mask, thr, mask_ms=_ms(t0))


def process_pages(pages: list[PageImage], thr: Thresholds = Thresholds(),
                 gt: GroundTruth | None = None, force_single: bool = False,
                 use_ocr: bool = True, style: str | None = None) -> list[PageResult]:
    return _run_raster(pages, thr, gt, force_single, use_ocr, style=style)


def remask_page(page: PageImage, thr: Thresholds = Thresholds(),
                use_ocr: bool = True) -> MaskLayer:
    """Re-run detection on one page; used by the workbench when a mask-affecting
    parameter changed and no shared cross-page mask can be reused."""
    return masks.estimate_single_page(page.image, thr,
                                      use_ocr=use_ocr)


def process_file(path: str | Path, thr: Thresholds = Thresholds(), dpi: int = 200,
                 skip_structure: bool = False, use_ocr: bool = True,
                 style: str | None = None) -> list[PageResult]:
    p = Path(path)
    # Per-page structural preflight is shared with streaming and the desktop
    # worker. Do not globally delete an image xref or render at a fixed 200 DPI:
    # a shared image can be legitimate elsewhere and the caller controls DPI.
    pages = documents.load_pages(p, dpi=dpi)
    return _run_raster(pages, thr, gt=None, force_single=len(pages) < 2, use_ocr=use_ocr,
                       use_structure=not skip_structure, style=style)


def process_file_streaming(path: str | Path, thr: Thresholds = Thresholds(),
                           dpi: int = 200, use_ocr: bool = True) -> Iterator[PageResult]:
    """Yield results one page at a time — constant memory regardless of doc size.

    Tries transparent-text operator removal before raster detection. Cross-page
    statistics are unavailable here; each remaining page uses single-image detection.
    """
    yield from iter_raster(documents.iter_pages(path, dpi=dpi), thr, total=0,
                           use_ocr=use_ocr)


def _run_dir(base: Path, source_name: str | None) -> Path:
    """Timestamped subdir so re-runs never overwrite previous output."""
    base.mkdir(parents=True, exist_ok=True)
    if not source_name:
        return base
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = Path(source_name).stem.replace(" ", "_")
    return base / f"{stamp}_{stem}"


def write_results(results: Iterable[PageResult], base_dir: str | Path,
                  source_path: str | Path | None = None) -> Path:
    """Save clean/watermark PNGs plus a manifest.json.

    When *source_path* is given, files go under ``<base_dir>/<timestamp>_<stem>/``
    so each run lands in its own folder; otherwise they drop directly into
    *base_dir*. Returns the directory actually used.
    """
    src = Path(source_path) if source_path else None
    out = _run_dir(Path(base_dir), src.name if src else None)
    out.mkdir(parents=True, exist_ok=True)

    pages: list[dict] = []
    passed = 0
    checks_passed = 0
    review = 0
    for r in results:
        name = f"{r.source.doc_id}_p{r.source.page_index:02d}"
        documents.save_png(out / f"{name}_clean.png", r.output.image)
        if r.extracted_watermark is not None:
            documents.save_png(out / f"{name}_watermark.png", r.extracted_watermark)
        pages.append(r.to_dict())
        if r.verdict.passed:
            checks_passed += 1
            if not r.verdict.review_needed:
                passed += 1
        if r.verdict.review_needed:
            review += 1

    total = len(pages)
    manifest = {
        "source": str(src) if src else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total": total,
            "passed": passed,
            "checks_passed": checks_passed,
            "needs_review": review,
            "quality_evaluated": total-review,
            "success_rate": round(passed/total, 4) if total and review == 0 else None,
            "success_rate_basis": "reference_evaluated_pages_only; unknown when review is pending",
        },
        "pages": pages,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    return out
