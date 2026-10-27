#!/usr/bin/env python3
"""Run a small, reproducible raster baseline on the frozen v1 validation split.

Only watermarked inputs are passed to the application pipeline. Clean references
and masks are loaded for scoring after inference returns. This is a smoke-sized
coverage baseline (one independently sourced record per image subtype), not a
full benchmark or a native-PDF evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))

from wm import pipeline, verdict
from wm.contract import PageImage, Thresholds


DEFAULT_DATA = REPO / "data/watermark_training_v1"
DEFAULT_OUT = REPO / "out/baseline_20261026"


def load(path: Path, mode: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert(mode))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-subtype", type=int, default=1)
    ap.add_argument("--negatives-per-subtype", type=int, default=1)
    ap.add_argument("--disable-ocr", action="store_true")
    args = ap.parse_args()
    data = args.data.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in (data / "splits/validation.jsonl").read_text().splitlines()]
    positives = [r for r in rows if r["kind"] == "image" and not r.get("is_negative")]
    negatives = [r for r in rows if r["kind"] == "image" and r.get("is_negative")]

    selected = []
    per_type: Counter[str] = Counter()
    for row in positives:
        if per_type[row["subtype"]] < args.per_subtype:
            selected.append(row)
            per_type[row["subtype"]] += 1
    neg_type_counts: Counter[str] = Counter()
    for row in negatives:
        if neg_type_counts[row["subtype"]] < args.negatives_per_subtype:
            selected.append(row)
            neg_type_counts[row["subtype"]] += 1

    results = []
    for index, row in enumerate(selected, 1):
        paths = row["paths"]
        source = load(data / paths["watermarked"], "RGB")
        page = PageImage(image=source, dpi=96, doc_id=row["id"], page_index=0,
                         source_kind="validation_image")
        # References are intentionally unavailable to the detector and repairer.
        # They are only attached to the pipeline's post-inference verdict step.
        result = pipeline.process_pages(
            [page], Thresholds(),
            force_single=True,
            use_ocr=not args.disable_ocr,
        )[0]
        clean = load(data / paths["clean"], "RGB")
        alpha = load(data / paths["alpha"], "L").astype(np.float32) / 255.0
        report = verdict.build_verdict(
            source=source,
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
        report.page_key = page.key
        result.verdict = report

        metrics = {k: (None if not np.isfinite(v) else round(float(v), 6))
                   for k, v in report.metrics.items()}
        changed = np.any(source != result.output.image, axis=2)
        changed_fraction = float(changed.mean())
        mean_abs_change = float(np.abs(source.astype(np.float32) -
                                       result.output.image.astype(np.float32)).mean())
        item = {
            "id": row["id"], "kind": row["kind"], "subtype": row["subtype"],
            "label": row["label"], "is_negative": bool(row.get("is_negative")),
            "source_group": row["source_group"], "base_id": row["base_id"],
            "input_sha256": row["sha256"]["watermarked"],
            "mask_source": str(result.mask.source), "mask_method": result.mask.method,
            "mask_confidence": round(float(result.mask.confidence), 6),
            "mask_coverage": round(float(result.mask.coverage()), 6),
            "changed_fraction": round(changed_fraction, 6),
            "mean_abs_change": round(mean_abs_change, 6),
            "negative_control_passed": bool(row.get("is_negative") and
                                             result.mask.coverage() == 0 and
                                             changed_fraction == 0),
            "passed": bool(report.passed), "failures": [str(f) for f in report.failures],
            "elapsed_ms": int(report.elapsed_ms), "metrics": metrics,
        }
        results.append(item)
        Image.fromarray(result.output.image.astype(np.uint8)).save(out / f"{row['id']}__output.png")
        Image.fromarray(result.mask.binary()).save(out / f"{row['id']}__predicted_mask.png")
        print(f"[{index}/{len(selected)}] {row['id']}: passed={item['passed']} "
              f"mask={item['mask_coverage']:.4f} failures={','.join(item['failures']) or '-'} "
              f"{item['elapsed_ms']}ms", flush=True)

    summary = {
        "title": "Raster pipeline baseline on frozen v1 validation subset",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_root": str(data), "split": "validation", "split_manifest": "splits/validation.jsonl",
        "method": "One first-listed held-out record per image subtype plus one per negative subtype; force_single=true; no style prompt; references loaded and scored only after process_pages inference.",
        "scope": "Raster image pipeline only. No native-PDF pipeline results are represented.",
        "selection": {"records": len(selected), "positive_types": len(per_type),
                      "positive_records": sum(not r["is_negative"] for r in selected),
                      "negative_records": sum(r["is_negative"] for r in selected),
                      "subtypes": sorted(per_type)},
        "aggregate": {
            "passed": sum(r["passed"] for r in results), "total": len(results),
            "pass_rate": round(sum(r["passed"] for r in results) / max(1, len(results)), 6),
            "failure_counts": dict(Counter(f for r in results for f in r["failures"])),
            "mean_mask_iou": round(float(np.mean([r["metrics"]["mask_iou"] for r in results
                                                   if r["metrics"].get("mask_iou") is not None])), 6),
            "median_elapsed_ms": int(np.median([r["elapsed_ms"] for r in results])) if results else 0,
            "positive_passed": sum(r["passed"] for r in results if not r["is_negative"]),
            "positive_total": sum(not r["is_negative"] for r in results),
            "negative_controls_passed": sum(r["negative_control_passed"] for r in results),
            "negative_controls_total": sum(r["is_negative"] for r in results),
        },
        "results": results,
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"metrics -> {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
