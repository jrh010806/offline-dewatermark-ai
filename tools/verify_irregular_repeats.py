"""Compare default detection/repair on the reported photos and clean controls.

References are loaded only after inference for scoring; they never select a
detector or a repair region. These development samples are not a general
success-rate benchmark. Optional before snapshots support local comparisons.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from wm import masks, observed_text, pipeline
from wm.contract import PageImage

DATA = ROOT / "data/watermark_training_increment_v1_1"
CASES = ("indoor_signage", "food_soup", "food_dish", "cat_lowlight")


def load(path):
    return np.array(Image.open(path).convert("RGB"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location("wm." + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(stem, out, variant):
    folder = DATA / "images" / f"inc11_{stem}__tiled_irregular"
    source = load(folder / "watermarked.png")
    started = time.perf_counter()
    result = pipeline.process_pages([PageImage(source, 96)], force_single=True)[0]
    elapsed = time.perf_counter() - started
    output = result.output.image
    selected = result.mask.binary() > 0
    # Only now read ground truth, after normal application inference finishes.
    clean = load(DATA / "bases" / (stem + ".png"))
    truth = np.array(Image.open(folder / "alpha.png")) > 0
    error = np.abs(output.astype(float) - clean).mean(2)
    original_error = np.abs(source.astype(float) - clean).mean(2)
    record = dict(case=stem, variant=variant, method=result.mask.method,
                  mode=result.mask.repair_mode, seconds=elapsed,
                  coverage=float(selected.mean()),
                  mask_recall=float(selected[truth].mean()),
                  mask_precision=float(truth[selected].mean()) if selected.any() else 0.,
                  watermark_mae_before=float(original_error[truth].mean()),
                  watermark_mae_after=float(error[truth].mean()),
                  clean_area_mae=float(error[~truth].mean()),
                  outside_selection_unchanged=bool(np.array_equal(output[~selected], source[~selected])))
    for name, image in (("source", source), (variant, output),
                        (variant + "_mask", np.uint8(selected) * 255)):
        Image.fromarray(image).save(out / f"{stem}_{name}.png")
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--cases", nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "results.json"
    if report_path.exists():
        raise SystemExit("Use a new output directory to preserve the previous run")
    records = []
    code_paths = [ROOT / 'wm' / (name + '.py') for name in
                  ('masks', 'observed_text', 'photo_observations', 'photo_text', 'lama', 'repeated')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in code_paths}
    (args.out / 'code_hashes.json').write_text(json.dumps(hashes, indent=2))
    if args.before:
        old_masks = load_module(args.before / "masks_before.py", "_before_masks")
        old_observed = load_module(args.before / "observed_text_before.py", "_before_observed")
        with patch.object(masks, "estimate_single_page", old_masks.estimate_single_page), \
                patch.object(observed_text, "repeated_photo", old_observed.repeated_photo):
            for stem in args.cases:
                records.append(evaluate(stem, args.out, "before"))
                report_path.write_text(json.dumps(records, indent=2))
    for stem in args.cases:
        records.append(evaluate(stem, args.out, "after"))
        report_path.write_text(json.dumps(records, indent=2))
    for stem in (*CASES, "outdoor_building"):
        source = load(DATA / "bases" / (stem + ".png"))
        result = pipeline.process_pages([PageImage(source, 96)], force_single=True)[0]
        record = dict(case=stem, variant="clean_control", method=result.mask.method,
                      coverage=result.mask.coverage(),
                      unchanged=bool(np.array_equal(source, result.output.image)))
        records.append(record)
        print(json.dumps(record), flush=True)
        report_path.write_text(json.dumps(records, indent=2))
    current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in code_paths}
    if current != hashes:
        raise SystemExit('Inference code changed during evaluation; verify the final masks before using these results')


if __name__ == "__main__":
    main()
