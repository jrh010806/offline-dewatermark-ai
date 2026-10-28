"""Save current detector selections and current repair outputs for a semantic-edit trial."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wm.contract import Thresholds
from wm.masks import estimate_single_page
from wm.repair import repair


def prepare(input_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    inputs = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    thresholds = Thresholds()
    index = {"input_directory": str(input_dir.resolve()), "thresholds": asdict(thresholds),
             "detector": "wm.masks.estimate_single_page", "cases": []}
    for number, path in enumerate(inputs, 1):
        case = out / path.stem
        case.mkdir()
        record = {"id": path.stem, "input": str(path.resolve()),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "status": "preparing", "quality_verified": False}
        index["cases"].append(record)
        try:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            source = np.array(image)
            image.save(case / "source.png")
            t0 = time.monotonic()
            mask = estimate_single_page(source, thresholds, use_ocr=True)
            record["detection_seconds"] = round(time.monotonic() - t0, 3)
            selection = Image.fromarray(mask.binary())
            selection.save(case / "selection.png")
            overlay = Image.composite(Image.new("RGB", image.size, "red"), image,
                                      selection.point(lambda p: 105 if p else 0))
            overlay.save(case / "selection_overlay.png")
            record.update(size=list(image.size), selected_pixels=int(np.count_nonzero(np.asarray(selection))),
                          selection_bbox=selection.getbbox(), method=mask.method, repair_mode=mask.repair_mode,
                          status="selection_ready")
            (case / "case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
            (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
            print(f"[{number}/{len(inputs)}] selection {path.name}: {record['selected_pixels']} pixels", flush=True)
            t0 = time.monotonic()
            baseline, used = repair(source, mask, thresholds)
            Image.fromarray(baseline).save(case / "baseline.png")
            selected = np.asarray(selection) >= 128
            changed = np.any(baseline != source, axis=2)
            record.update(status="prepared", baseline_seconds=round(time.monotonic() - t0, 3),
                          baseline_repairs=[v.value for v in used],
                          baseline_changed_outside_selection=int(np.count_nonzero(changed & ~selected)))
            print(f"[{number}/{len(inputs)}] baseline {path.name}: {record['baseline_seconds']}s", flush=True)
        except Exception as exc:
            record.update(status="error", error=str(exc))
            (case / "preparation_error.log").write_text(traceback.format_exc())
            print(f"[{number}/{len(inputs)}] ERROR {path.name}: {exc}", flush=True)
        (case / "case.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input_dir, args.out)
