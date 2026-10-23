"""Run native single-image inference, then evaluate against separate references."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnose_single import metrics
from wm import documents, masks, pipeline
from wm.contract import Thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out/photo_styles'
CASES = ('dot_grid_02', 'big_center_02', 'big_center_04', 'heavy_dark_02', 'heavy_dark_04')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_dir = OUT/'baseline'
    baseline_dir.mkdir(exist_ok=True)
    old = json.loads((ROOT/'out/edge_cleanup/other_styles.json').read_text())
    rows = {}
    panels = []
    font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 20)
    for stem in CASES:
        page = documents.load_pages(ROOT/'data/real_samples/jpg'/f'{stem}.jpg')[0]
        start = time.perf_counter()
        mask = masks.estimate_single_page(page.image)
        detection_ms = round((time.perf_counter()-start)*1000)
        result = pipeline._process_one(page, mask, Thresholds(), mask_ms=detection_ms)
        output = result.output.image
        # Only now open the clean reference and evaluator's alpha.
        clean = np.array(Image.open(ROOT/'data/real_samples/clean'/f'{stem}.png').convert('RGB'))
        alpha = np.load(ROOT/'data/real_samples/mask'/f'{stem}_veil.npz')['alpha']
        previous_path = (ROOT/'out/edge_cleanup'/f'{stem}_regression.png' if stem in old
                         else ROOT/'out/edge_cleanup'/f'{stem}_after.png')
        if not previous_path.exists():
            previous_path = ROOT/'out/single_improvement'/f'{stem}_after.png'
        baseline_path = baseline_dir/f'{stem}.png'
        if not baseline_path.exists():
            Image.open(previous_path).save(baseline_path)
        previous = np.array(Image.open(baseline_path).convert('RGB'))
        state = 'failed' if not result.verdict.passed else ('needs_review' if result.verdict.review_needed else 'passed')
        row = dict(method=mask.method, state=state,
                   elapsed_ms=result.verdict.elapsed_ms, mask_ms=detection_ms,
                   stage_ms=result.verdict.stage_ms,
                   input=metrics(page.image, clean, page.image, alpha),
                   before=metrics(previous, clean, page.image, alpha),
                   after=metrics(output, clean, page.image, alpha),
                   byte_identical_to_previous=bool(np.array_equal(output, previous)),
                   outside_requested_mask_unchanged=bool(np.array_equal(
                       output[~mask.footprint], page.image[~mask.footprint])))
        rows[stem] = row
        Image.fromarray(output).save(OUT/f'{stem}_after.png')
        Image.fromarray(mask.binary()).save(OUT/f'{stem}_mask.png')
        Image.fromarray(previous).save(OUT/f'{stem}_before.png')
        if stem in ('dot_grid_02', 'big_center_02'):
            panel = Image.new('RGB', (1800, 400), 'white')
            draw = ImageDraw.Draw(panel)
            for i, (label, pixels) in enumerate((('原图', page.image), ('旧版结果', previous), ('新版结果', output))):
                photo = Image.fromarray(pixels)
                photo.thumbnail((596, 340))
                panel.paste(photo, (i*600, 45))
                draw.text((i*600+12, 8), f'{stem} · {label}', fill='black', font=font)
            panels.append(panel)
        (OUT/'verification.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), 'utf-8')
        print(stem, json.dumps(row, ensure_ascii=False), flush=True)
    canvas = Image.new('RGB', (1800, 400*len(panels)), 'white')
    for i, panel in enumerate(panels):
        canvas.paste(panel, (0, i*400))
    canvas.save(OUT/'before_after.png')


if __name__ == '__main__':
    main()
