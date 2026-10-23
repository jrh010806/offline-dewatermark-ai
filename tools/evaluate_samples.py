"""Paired-sample audit and isolated research evaluation.

Clean references are loaded only after positive source-only inference. The SLBR
command is a research runner; it does not change the app or install a model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inventory(root, output):
    rows = json.loads((root/'manifest.json').read_text())
    for row in rows:
        for role in ('clean', 'watermarked'):
            path = root/row[role]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[role+'_sha256'], path
        if row['kind'] == 'pdf':
            with fitz.open(root/row['clean']) as a, fitz.open(root/row['watermarked']) as b:
                assert len(a) == len(b) == row['pages']
        else:
            with Image.open(root/row['clean']) as a, Image.open(root/row['watermarked']) as b:
                assert a.size == b.size == (row['width'], row['height'])
        digest = hashlib.sha256((row['kind']+':'+row['clean_sha256']).encode()).hexdigest()
        row['split'] = 'validation' if int(digest[:8], 16) % 5 == 0 else 'development'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(root=str(root.resolve()),
        split_rule='sha256(kind:clean_sha256) first8 mod5==0 => validation', rows=rows), indent=2))
    print('Verified pairs:', len(rows), Counter((r['kind'], r['split']) for r in rows))


def image(path):
    return np.asarray(Image.open(path).convert('RGB'))


def roi_for(root, row, src, clean):
    path = root/row['group']/'mask'/(row['id']+'.png')
    target = np.asarray(Image.open(path).convert('L')) > 0 if path.exists() else \
        abs(src.astype(float)-clean).max(2) > 3
    return cv2.dilate(target.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def score(src, pred, clean, roi):
    initial = abs(src.astype(float)-clean).mean(2)
    err = abs(pred.astype(float)-clean).mean(2)
    return dict(input_roi=float(initial[roi].mean()), output_roi=float(err[roi].mean()),
                input_mae=float(initial.mean()), output_mae=float(err.mean()),
                outside_mae=float(err[~roi].mean()) if (~roi).any() else None)


def baseline(inv, out, split):
    from wm import masks, repair
    from wm.contract import Thresholds
    root = Path(inv['root']); out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise ValueError('Baseline already exists; choose a fresh output folder')
    rows = {}
    for r in inv['rows']:
        if r['kind'] != 'image' or (split != 'all' and r['split'] != split):
            continue
        src = image(root/r['watermarked']); start = time.perf_counter()
        mask = masks.estimate_single_page(src)
        pred, _ = repair.repair(src, mask, Thresholds())
        ms = round((time.perf_counter()-start)*1000)
        clean = image(root/r['clean']); roi = roi_for(root, r, src, clean)
        row = dict(type=r['watermark_type'], split=r['split'], method=mask.method,
                   ms=ms, coverage=mask.coverage(), **score(src, pred, clean, roi))
        row['outside_changed'] = float((abs(pred.astype(float)-src).max(2)>4)[~roi].mean()) if (~roi).any() else None
        rows[r['id']] = row
        Image.fromarray(pred).save(out/(r['id']+'.png'))
        (out/'metrics.json').write_text(json.dumps(rows, indent=2))
        if len(rows)%10 == 0:
            print('baseline', len(rows), 'complete', flush=True)


def screen_paper(inv, out):
    """Prove the new paper route declines this photo corpus without rerunning unchanged models."""
    from wm.paper_grid import estimate
    root = Path(inv['root']); rows = []
    for r in inv['rows']:
        if r['kind'] != 'image':
            continue
        candidate = estimate(image(root/r['watermarked']))
        # A clean-image control is an independent inference, never a detector input to the positive case.
        negative = estimate(image(root/r['clean']))
        rows.append(dict(id=r['id'], split=r['split'], type=r['watermark_type'],
                         selected=candidate is not None, clean_selected=negative is not None))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print('Paper route: positive selected', sum(r['selected'] for r in rows),
          'clean selected', sum(r['clean_selected'] for r in rows), 'of', len(rows))


def slbr(inv, out, model_path, split, model_sha):
    import torch  # optional research dependency; no requirement for the desktop app
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_sha, 'Unverified model hash'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = torch.jit.load(str(model_path), map_location='cpu').eval().to(device)
    root = Path(inv['root']); out.mkdir(parents=True, exist_ok=True)
    destination = out/'metrics.json'
    if destination.exists():
        raise ValueError('Evaluation already exists; choose a fresh output folder')
    rows = {}; provenance = dict(device=device, model_sha256=model_sha,
        source='https://github.com/bcmi/SLBR-Visible-Watermark-Removal',
        commit='47c665f1855ab6624cd52b28cefa797a9c8b96f7', split=split,
        composition='background*probability+source*(1-probability); no threshold',
        production_enabled=False)
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    def predict(src):
        if src.shape != (256, 256, 3):
            raise ValueError('Research evaluation supports native 256x256 images only')
        x = torch.from_numpy(src.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)/255
        start = time.perf_counter()
        with torch.inference_mode():
            bg, probability = model(x)
            p = probability[0, 0].cpu().numpy()
            b = bg[0].cpu().numpy().transpose(1, 2, 0)*255
        output = np.uint8(np.clip(np.rint(b*p[..., None]+src*(1-p[..., None])), 0, 255))
        return output, p, round((time.perf_counter()-start)*1000)
    for r in inv['rows']:
        if r['kind'] != 'image' or r['split'] != split:
            continue
        src = image(root/r['watermarked'])
        pred, prob, ms = predict(src)
        # Positive inference above has no reference input.
        clean = image(root/r['clean']); negative, negprob, negative_ms = predict(clean)
        roi = roi_for(root, r, src, clean)
        row = dict(id=r['id'], type=r['watermark_type'], split=split, ms=ms,
                   **score(src, pred, clean, roi),
                   negative_mae=float(abs(negative.astype(float)-clean).mean()),
                   negative_changed_gt4=float((abs(negative.astype(float)-clean).max(2)>4).mean()),
                   negative_confident_coverage=float((negprob>.5).mean()), negative_ms=negative_ms)
        Image.fromarray(pred).save(out/(r['id']+'.png'))
        rows[r['id']] = row
        destination.write_text(json.dumps(rows, indent=2))
        if len(rows)%10 == 0:
            print(split, len(rows), 'complete', flush=True)
    print('Completed', len(rows), 'research cases; app unchanged', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'baseline', 'screen-paper', 'slbr'])
    parser.add_argument('--root', type=Path)
    parser.add_argument('--inventory', type=Path, default=Path('out/new_samples/inventory.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-sha256')
    parser.add_argument('--split', choices=['development', 'validation', 'all'], default='validation')
    args = parser.parse_args()
    if args.command == 'inventory':
        if args.root is None:
            parser.error('--root required')
        inventory(args.root, args.out)
    else:
        inv = json.loads(args.inventory.read_text())
        if args.command == 'baseline':
            baseline(inv, args.out, args.split)
        elif args.command == 'screen-paper':
            screen_paper(inv, args.out)
        else:
            if args.model is None or args.model_sha256 is None:
                parser.error('--model and --model-sha256 required')
            slbr(inv, args.out, args.model, args.split, args.model_sha256)


if __name__ == '__main__':
    main()
