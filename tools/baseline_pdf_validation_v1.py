#!/usr/bin/env python3
"""Run native-PDF handling on one held-out v1 document per PDF subtype."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from wm import documents, pipeline, verdict
from wm.contract import Thresholds

DATA = REPO / "data/watermark_training_v1"
OUT = REPO / "out/baseline_20261026"


def read_alpha(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS),
                      dtype=np.float32) / 255.0


def main() -> int:
    rows = [json.loads(x) for x in (DATA / "splits/validation.jsonl").read_text().splitlines()]
    pdfs = [r for r in rows if r["kind"] == "pdf"]
    selected, seen = [], set()
    for row in pdfs:
        if row["subtype"] not in seen:
            selected.append(row)
            seen.add(row["subtype"])

    records = []
    for n, row in enumerate(selected, 1):
        marked_path = DATA / row["paths"]["watermarked"]
        clean_pages = documents.load_pages(DATA / row["paths"]["clean"], dpi=96)
        results = pipeline.process_file(marked_path, dpi=96)
        page_records = []
        for i, result in enumerate(results):
            page = row["pages"][i]
            clean = clean_pages[i].image
            alpha_path = DATA / page["paths"]["alpha"]
            alpha = read_alpha(alpha_path, (result.source.width, result.source.height))
            scored = verdict.build_verdict(
                source=result.source.image,
                output=result.output.image,
                mask=result.mask,
                engine=result.verdict.engine,
                repairs=result.verdict.repairs,
                elapsed_ms=result.verdict.elapsed_ms,
                stage_ms=result.verdict.stage_ms,
                thr=Thresholds(),
                clean_reference=clean,
                alpha_reference=alpha,
            )
            metrics = {k: (None if not np.isfinite(v) else round(float(v), 6))
                       for k, v in scored.metrics.items()}
            page_records.append({
                "page": i + 1, "mask_method": result.mask.method,
                "mask_source": str(result.mask.source),
                "mask_coverage": round(float(result.mask.coverage()), 6),
                "passed": bool(scored.passed), "failures": [str(f) for f in scored.failures],
                "elapsed_ms": int(scored.elapsed_ms), "metrics": metrics,
            })
            Image.fromarray(result.output.image.astype(np.uint8)).save(
                OUT / f"{row['id']}__page_{i+1}__output.png")
        records.append({
            "id": row["id"], "subtype": row["subtype"], "label": row["label"],
            "is_negative": bool(row.get("is_negative")), "source_group": row["source_group"],
            "input_sha256": row["sha256"]["watermarked"], "pages": page_records,
        })
        print(f"[{n}/{len(selected)}] {row['id']}: " + ", ".join(
            f"p{p['page']}={'PASS' if p['passed'] else 'FAIL'}" for p in page_records), flush=True)

    flat = [p for r in records for p in r["pages"]]
    summary = {
        "title": "Native PDF pipeline baseline on frozen v1 validation subset",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "split": "validation", "split_manifest": "splits/validation.jsonl",
        "method": "One first-listed PDF per subtype; process_file at 96 DPI, preserving the native PDF path. Clean PDF and alpha/mask annotations are used only after inference for scoring. Alpha is resampled to rendered page dimensions.",
        "selection": {"documents": len(selected), "subtypes": sorted(seen),
                      "pages": len(flat), "negative_documents": sum(bool(r["is_negative"]) for r in selected)},
        "aggregate": {
            "passed": sum(p["passed"] for p in flat), "total": len(flat),
            "pass_rate": round(sum(p["passed"] for p in flat) / max(1, len(flat)), 6),
            "failure_counts": dict(Counter(f for p in flat for f in p["failures"])),
            "mean_mask_iou": round(float(np.mean([p["metrics"]["mask_iou"] for p in flat
                                                   if p["metrics"].get("mask_iou") is not None])), 6),
            "median_elapsed_ms": int(np.median([p["elapsed_ms"] for p in flat])) if flat else 0,
        },
        "results": records,
    }
    (OUT / "pdf_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"metrics -> {OUT / 'pdf_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
