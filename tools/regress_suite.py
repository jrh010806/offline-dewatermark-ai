"""Native-size single-image regression; reference files are evaluation-only.

Example: HF_HUB_OFFLINE=1 python tools/regress_suite.py --stage after
Existing baseline outputs are kept unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_single import metrics
from wm import documents, masks, pipeline
from wm.contract import Thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('before', 'after'), default='after')
    parser.add_argument('--cases', nargs='+', help='source stems to run; default: all local samples')
    parser.add_argument('--overwrite', action='store_true', help='explicitly replace saved baseline rows')
    parser.add_argument('--output-root', type=Path, default=ROOT/'out/center_refinement')
    args = parser.parse_args()
    output = args.output_root/args.stage
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'metrics.json'
    rows = json.loads(manifest.read_text()) if manifest.exists() else {}
    paths = sorted((ROOT/'data/real_samples/jpg').glob('*.jpg'))
    if args.cases:
        found = {p.stem for p in paths}
        missing = set(args.cases)-found
        if missing:
            parser.error(f'Unknown sample stems: {sorted(missing)}')
        paths = [p for p in paths if p.stem in args.cases]
    errors = []
    for path in paths:
        stem = path.stem
        if args.stage == 'before' and stem in rows and not args.overwrite:
            print(stem, 'preserved baseline', flush=True)
            continue
        print(args.stage, stem, 'start', flush=True)
        try:
            start = time.perf_counter()
            page = documents.load_pages(path)[0]
            mask = masks.estimate_single_page(page.image)
            detection_ms = round((time.perf_counter()-start)*1000)
            result = pipeline._process_one(page, mask, Thresholds(), mask_ms=detection_ms)
            pixels = result.output.image
            np.savez_compressed(output/f'{stem}_mask.npz', alpha=mask.alpha,
                                footprint=mask.footprint, veil_strength=mask.veil_strength,
                                veil_offset=mask.veil_offset)
            Image.fromarray(pixels).save(output/f'{stem}.png')
            # Inference has finished. Clean images and alpha are never passed to
            # detection/repair, including for the single-image opacity fits.
            clean = np.array(Image.open(ROOT/'data/real_samples/clean'/f'{stem}.png').convert('RGB'))
            alpha = np.load(ROOT/'data/real_samples/mask'/f'{stem}_veil.npz')['alpha']
            ink = clean.mean(2) < 120
            lifted = float(np.mean((pixels.mean(2) > clean.mean(2)+50)[ink])) if ink.any() else None
            document = bool(np.mean(clean.mean(2) > 235) > .65)
            rows[stem] = dict(
                method=mask.method, repair_mode=mask.repair_mode,
                elapsed_ms=result.verdict.elapsed_ms, stage_ms=result.verdict.stage_ms,
                input=metrics(page.image, clean, page.image, alpha),
                output=metrics(pixels, clean, page.image, alpha),
                dark_pixel_lift_gt50=lifted,
                document_ink_erased_gt50=lifted if document else None,
                outside_mask_unchanged=bool(np.array_equal(
                    pixels[~mask.footprint], page.image[~mask.footprint])),
                shape=list(pixels.shape), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                inference_used_ground_truth=False,
                failures=[str(f) for f in result.verdict.failures])
        except Exception as exc:
            rows[stem] = {'error': repr(exc)}
            errors.append(stem)
            traceback.print_exc()
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), 'utf-8')
        temporary.replace(manifest)
        print(args.stage, stem, json.dumps(rows[stem], ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(f'Regression failed for {errors}; errors recorded in {manifest}')


if __name__ == '__main__':
    main()
