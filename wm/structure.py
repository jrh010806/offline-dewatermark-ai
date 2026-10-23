"""Engine A: structural removal of PDF watermark objects.

A watermark drawn as its own transparent image XObject is a parsing problem,
not an AI problem: removing it is lossless and instant, so the pipeline must
check this before spending any compute.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import logging
from time import perf_counter

import numpy as np
import pymupdf as fitz

from .contract import StructureProbe

MIN_AREA_FRACTION = 0.4
MAX_OPAQUE_FRACTION = 0.35


def _alpha_stats(doc: fitz.Document, smask_xref: int) -> tuple[float, float]:
    pix = fitz.Pixmap(doc, smask_xref)
    data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    alpha = data[..., 0].astype(np.float32) / 255.0
    return float((alpha > 0.5).mean()), float(alpha.mean())


def probe_pdf(path: str | Path) -> list[StructureProbe]:
    doc = fitz.open(str(path))
    results: list[StructureProbe] = []
    try:
        for page in doc:
            page_area = page.rect.get_area() or 1.0
            found = []
            for item in page.get_images(full=True):
                xref, smask = item[0], item[1]
                if not smask:
                    continue
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                area_frac = max(r.get_area() for r in rects) / page_area
                if area_frac < MIN_AREA_FRACTION:
                    continue
                opaque_frac, mean_alpha = _alpha_stats(doc, smask)
                if opaque_frac <= MAX_OPAQUE_FRACTION:
                    found.append({"xref": xref, "area_fraction": area_frac,
                                  "opaque_fraction": opaque_frac,
                                  "mean_alpha": mean_alpha})
            results.append(StructureProbe(
                applicable=bool(found), object_count=len(found),
                detail={"objects": found, "page": page.number},
            ))
    finally:
        doc.close()
    return results


def strip_pdf(path: str | Path, out_path: str | Path) -> int:
    probes = probe_pdf(path)
    doc = fitz.open(str(path))
    removed = 0
    try:
        for probe in probes:
            if not probe.applicable:
                continue
            page = doc[probe.detail["page"]]
            for obj in probe.detail["objects"]:
                page.delete_image(obj["xref"])
                removed += 1
        doc.save(str(out_path), garbage=4, deflate=True)
    finally:
        doc.close()
    return removed


@lru_cache(maxsize=2)
def _text_copy(path: str, revision: tuple[int, int]):
    from .pdf_text import strip_translucent_text
    source = Path(path)
    data = source.read_bytes()
    stat = source.stat()
    if (stat.st_mtime_ns, stat.st_size) != revision:
        raise ValueError("PDF changed since upload")
    return strip_translucent_text(data)


def process_text_page(page, thr):
    """Optional structure preflight shared by the UI, batch, and CLI.

    Source revision and original page number must match the loaded raster.
    The edited PDF is kept in memory; user input directories are never written.
    """
    from .contract import EngineKind, MaskLayer, MaskSource, PageImage, PageResult
    from . import verdict
    if page.source_path is None or page.source_page_index is None or page.source_revision is None:
        return None
    t0 = perf_counter()
    try:
        source = Path(page.source_path)
        stat = source.stat()
        if (stat.st_mtime_ns, stat.st_size) != page.source_revision:
            return None
        data, counts = _text_copy(page.source_path, page.source_revision)
        index = page.source_page_index
        if index >= len(counts) or not counts[index]:
            return None
        # Do not apply the structural result to a caller-modified input image.
        with fitz.open(page.source_path) as original:
            pix = original[index].get_pixmap(dpi=page.dpi, colorspace=fitz.csRGB)
            original_pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
            if not np.array_equal(original_pixels, page.image):
                return None
        with fitz.open(stream=data, filetype="pdf") as cleaned:
            pix = cleaned[index].get_pixmap(dpi=page.dpi, colorspace=fitz.csRGB)
            output = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
        region = np.any(page.image != output, axis=2)
        mask = MaskLayer(region.astype(np.float32), source=MaskSource.STRUCTURE,
                         confidence=.95, footprint=region, method="transparent_pdf_text",
                         repair_mode="structure",
                         veil_strength=np.zeros(3, np.float32), veil_offset=np.zeros(3, np.float32))
        ms = round((perf_counter()-t0)*1000)
        report = verdict.build_verdict(source=page.image, output=output, mask=mask,
                    engine=EngineKind.STRUCTURE, repairs=[], elapsed_ms=ms,
                    stage_ms={"structure_ms": ms}, thr=thr)
        report.metrics["text_blocks_removed"] = float(counts[index])
        report.page_key = page.key
        dst = PageImage(output, page.dpi, page.doc_id, page.page_index, page.source_kind)
        return PageResult(page, dst, mask, report)
    except Exception as exc:
        logging.getLogger(__name__).warning("PDF text preflight unavailable; using raster: %s", exc)
        return None
