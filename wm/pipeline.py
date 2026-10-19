"""Pipeline: structure probe -> coverage estimate -> veil inverse -> verdict."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import documents, masks, repair, structure, verdict as verdict_mod
from .contract import (EngineKind, MaskLayer, PageImage, PageResult, Thresholds)

GroundTruth = dict[str, tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def extract_watermark(mask: MaskLayer) -> np.ndarray:
    s = mask.veil_strength.astype(np.float32)
    color = np.clip(mask.veil_offset / np.maximum(1.0 - s, 1e-3), 0, 255)
    rgba = np.empty((*mask.alpha.shape, 4), np.uint8)
    rgba[..., :3] = color.reshape(1, 1, 3).astype(np.uint8)
    rgba[..., 3] = (mask.alpha * 255).astype(np.uint8)
    return rgba


def _run_raster(pages: list[PageImage], thr: Thresholds, gt: GroundTruth | None,
                force_single: bool = False) -> list[PageResult]:
    images = [p.image for p in pages]
    stage: dict[str, int] = {}
    t0 = time.perf_counter()
    use_cross = (not force_single) and len(images) >= 2 and \
        all(i.shape == images[0].shape for i in images)
    shared = masks.estimate_crosspage(images, thr) if use_cross else None
    stage["mask_ms"] = _ms(t0)

    out: list[PageResult] = []
    for page in pages:
        local_t0 = time.perf_counter()
        mask = shared if use_cross else masks.estimate_single_page(page.image, thr)
        mask_ms = stage["mask_ms"] if not use_cross else stage["mask_ms"] // max(len(pages), 1)
        output, used = repair.repair(page.image, mask, thr)
        rep_ms = _ms(local_t0)
        entry = gt.get(page.key) if gt else None
        clean_ref, alpha_ref, veil_ref = entry if entry else (None, None, None)
        report = verdict_mod.build_verdict(
            source=page.image, output=output, mask=mask,
            engine=EngineKind.RASTER, repairs=used,
            elapsed_ms=_ms(local_t0), stage_ms={"mask_ms": int(mask_ms), "repair_ms": int(rep_ms)},
            thr=thr, clean_reference=clean_ref, alpha_reference=alpha_ref,
            veil_reference=veil_ref,
        )
        report.page_key = page.key
        out.append(PageResult(
            source=page, output=PageImage(image=output, dpi=page.dpi, doc_id=page.doc_id,
                                          page_index=page.page_index, source_kind=page.source_kind),
            mask=mask, verdict=report, extracted_watermark=extract_watermark(mask),
            clean_reference=clean_ref,
        ))
    return out


def _run_structure(path: Path, thr: Thresholds) -> list[PageResult] | None:
    probes = structure.probe_pdf(path)
    if not probes or not all(p.applicable for p in probes):
        return None
    stripped = path.with_name(path.stem + "_stripped.pdf")
    removed = structure.strip_pdf(path, stripped)
    before = documents.load_pages(path, dpi=200)
    after = documents.load_pages(stripped, dpi=200)
    results = []
    for src, dst in zip(before, after):
        t0 = time.perf_counter()
        mask = masks.estimate_single_page(src.image, thr)
        report = verdict_mod.build_verdict(
            source=src.image, output=dst.image, mask=mask,
            engine=EngineKind.STRUCTURE,
            repairs=[], elapsed_ms=_ms(t0),
            stage_ms={"probe_ms": 0, "strip_ms": 0}, thr=thr,
        )
        report.metrics["objects_removed"] = float(removed)
        report.page_key = src.key
        results.append(PageResult(source=src, output=dst, mask=mask, verdict=report))
    return results


def process_pages(pages: list[PageImage], thr: Thresholds = Thresholds(),
                 gt: GroundTruth | None = None, force_single: bool = False) -> list[PageResult]:
    return _run_raster(pages, thr, gt, force_single)


def process_file(path: str | Path, thr: Thresholds = Thresholds(), dpi: int = 200,
                 skip_structure: bool = False) -> list[PageResult]:
    p = Path(path)
    if p.suffix.lower() == ".pdf" and not skip_structure:
        structural = _run_structure(p, thr)
        if structural is not None:
            return structural
    pages = documents.load_pages(p, dpi=dpi)
    return _run_raster(pages, thr, gt=None, force_single=len(pages) < 2)


def write_results(results: list[PageResult], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for r in results:
        name = f"{r.source.doc_id}_p{r.source.page_index:02d}"
        documents.save_png(out / f"{name}_clean.png", r.output.image)
        if r.extracted_watermark is not None:
            documents.save_png(out / f"{name}_watermark.png", r.extracted_watermark)
