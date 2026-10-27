"""Fixed before/after foundation regression. References are scoring-only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import warnings

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/watermark_training_v1"
OUT = ROOT / "out/foundation_round_20261026"


def selection():
    rows = [json.loads(x) for x in (DATA/"manifest.jsonl").read_text().splitlines()]
    old = json.loads((ROOT/"out/baseline_20261026/metrics.json").read_text())
    old_pdf = json.loads((ROOT/"out/baseline_20261026/pdf_metrics.json").read_text())
    primary = {r["id"] for r in old["results"] + old_pdf["results"]}
    basic = {"local_text", "center_text", "center_white", "tiled_diagonal"}
    extra_bases = {"photo_008", "photo_016", "photo_024", "document_026", "document_027"}
    selected = []
    for r in rows:
        extra = (r["kind"] == "image" and
                 ((r["base_id"] in extra_bases and r["subtype"] in basic)
                  or (r["is_negative"] and (r["split"] == "validation" or r["base_id"] in extra_bases))))
        extra |= (r["kind"] == "pdf" and r["base_id"] == "document_026"
                  and r["subtype"] in {"unicode_text", "independent_image", "no_watermark"})
        if r["id"] in primary or extra:
            selected.append({**r, "suite": "original" if r["id"] in primary else "extra"})
    return selected


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def finite(value):
    return {k: round(float(v), 7) if np.isfinite(v) else None for k, v in value.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("phase", choices=["before", "after"])
    ap.add_argument("--kind", choices=["image", "pdf", "all"], default="all")
    ap.add_argument("--restart", action="store_true", help="Archive this phase's previous output and rerun it")
    args = ap.parse_args()
    source_root = OUT/"before_source" if args.phase == "before" else ROOT
    sys.path.insert(0, str(source_root))
    from wm import pipeline, verdict, documents, lama
    from wm.contract import PageImage, Thresholds
    code_sha256 = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (source_root/"wm").glob("*.py")}
    # The same CPU pool in both runs avoids host oversubscription affecting the
    # time-budget comparison. No model weights or scoring thresholds are changed.
    import onnxruntime as ort
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    opts.intra_op_num_threads = 4; opts.inter_op_num_threads = 1
    lama._sess = ort.InferenceSession(lama._resolve_model_path(), sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    rows = selection()
    (OUT/"selection.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    folder = OUT/args.phase
    if args.restart and folder.exists():
        archive = OUT/(args.phase+"_"+time.strftime("%Y%m%d_%H%M%S"))
        if archive.exists():
            raise SystemExit(f"Archive already exists: {archive}")
        folder.rename(archive)
    folder.mkdir(parents=True, exist_ok=True)
    metrics_path = folder/f"{args.kind}_metrics.json"
    previous = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if previous.get("code_sha256") and previous["code_sha256"] != code_sha256:
        raise SystemExit("Source changed; use --restart to preserve the old run and recompute.")
    results = previous.get("results", [])
    completed = {r["id"] for r in results}
    for r in rows:
        if args.kind != "all" and r["kind"] != args.kind or r["id"] in completed:
            continue
        start = time.perf_counter()
        source_path = DATA/r["paths"]["watermarked"]
        if r["kind"] == "image":
            src = load(source_path)
            page = PageImage(src, 96, r["id"], 0, "validation_image")
            outputs = pipeline.process_pages([page], force_single=True)
            clean_pages = [load(DATA/r["paths"]["clean"])]
            alphas = [np.asarray(Image.open(DATA/r["paths"]["alpha"]).convert("L"), float)/255]
        else:
            outputs = pipeline.process_file(source_path, dpi=96)
            clean_pages = [p.image for p in documents.load_pages(DATA/r["paths"]["clean"], dpi=96)]
            alphas = [np.asarray(Image.open(DATA/p["paths"]["alpha"]).convert("L").resize(
                (o.source.width, o.source.height), Image.Resampling.LANCZOS), float)/255
                for p, o in zip(r["pages"], outputs)]
        for i, (o, clean, alpha) in enumerate(zip(outputs, clean_pages, alphas)):
            assert clean.shape == o.output.image.shape == o.source.image.shape
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                score = verdict.build_verdict(o.source.image, o.output.image, o.mask,
                    o.verdict.engine, o.verdict.repairs, o.verdict.elapsed_ms,
                    o.verdict.stage_ms, clean_reference=clean, alpha_reference=alpha)
            name = r["id"]+f"__p{i+1}"
            Image.fromarray(o.output.image).save(folder/f"{name}__output.png")
            Image.fromarray(o.mask.binary()).save(folder/f"{name}__mask.png")
            changed = np.any(o.source.image != o.output.image, axis=2)
            support = alpha > 0
            before = float(np.abs(o.source.image.astype(float)-clean)[support].mean()) if support.any() else 0.
            after = float(np.abs(o.output.image.astype(float)-clean)[support].mean()) if support.any() else 0.
            row = dict(id=r["id"], page=i+1, name=name, kind=r["kind"], label=r["label"],
                       subtype=r["subtype"], suite=r["suite"], split=r["split"],
                       source_group=r["source_group"], is_negative=r["is_negative"],
                       input_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                       method=o.mask.method, passed=bool(score.passed),
                       negative_unchanged=bool(r["is_negative"] and not changed.any()),
                       changed_pixels=int(changed.sum()), elapsed_ms=o.verdict.elapsed_ms,
                       before_error=round(before, 5), after_error=round(after, 5),
                       failures=[str(x) for x in score.failures], metrics=finite(score.metrics))
            results.append(row)
            print(args.phase, r["id"], i+1, row["method"], row["passed"],
                  row["failures"], "negative_changed", row["changed_pixels"] if r["is_negative"] else "-", flush=True)
        payload = dict(phase=args.phase, source_tree=str(source_root),
                       created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       protocol="Original thresholds; no clean references/masks supplied to inference; 96 DPI; 4 CPU threads",
                       code_sha256=code_sha256,
                       thresholds=vars(Thresholds()), results=results)
        metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        print("seconds", round(time.perf_counter()-start, 2), flush=True)


if __name__ == "__main__":
    main()
