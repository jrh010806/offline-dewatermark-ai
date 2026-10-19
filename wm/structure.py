"""Engine A: structural removal of PDF watermark objects.

A watermark drawn as its own transparent image XObject is a parsing problem,
not an AI problem: removing it is lossless and instant, so the pipeline must
check this before spending any compute.
"""
from __future__ import annotations

from pathlib import Path

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
