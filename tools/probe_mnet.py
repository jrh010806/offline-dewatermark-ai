"""Bounded MNet experiment: the existing eight cases, one fixed LOGO-H checkpoint.

Inference sees RGB input only. Ground truth is loaded afterwards for scoring.
The official model is imported unchanged, without its CUDA-only training runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from PIL import Image

import probe_removal_strategies as base

ROOT = base.ROOT
MODEL_DIR = ROOT / 'out/strategy_probe/models/mnet'
SOURCE = MODEL_DIR / 'source/scripts/mynets/MNetold.py'
WEIGHTS = MODEL_DIR / 'logo_high_model_best.pth.tar'
WEIGHTS_SHA = 'f2ed0ecd54e432d03cfb2313a488854f60bad6997124070bf8cc75ac998afd50'
SOURCE_SHA = 'd610107a9cd4f1201519335e06fab13c5de9304db77575e6272752f9491eafa9'
WEIGHTS_ID = '1bOv8cQ47gPW5air1GNwh_ZpKGQ8L9yBE'
BASELINE = ROOT / 'out/strategy_probe/20261027_small8_mps'
CONFIG = dict(k1=5, k2=2, k3=1, srm=1, inf=1, crf=1, sharing=1)
NAMES = {
    'stored_current': '原有流程（历史保存）',
    'stored_slbr_tiles': 'SLBR 原尺寸分块（历史保存）',
    'mnet_global': 'MNet 整图缩放',
    'mnet_tiles': 'MNet 原尺寸分块',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def prepare():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['scripts/mynets/MNetold.py', 'README.md', 'test_example.sh']:
        dest = MODEL_DIR / 'source' / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = 'https://raw.githubusercontent.com/Aitchson-Hwang/MNet/main/' + name
            with urllib.request.urlopen(url, timeout=40) as response:
                dest.write_bytes(response.read())
    if not WEIGHTS.exists():
        url = f'https://drive.usercontent.google.com/download?id={WEIGHTS_ID}&export=download&confirm=t'
        part = MODEL_DIR / 'checkpoint.part'
        with urllib.request.urlopen(url, timeout=45) as response, part.open('wb') as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if sha(part) != WEIGHTS_SHA:
            raise RuntimeError('Checkpoint hash differs from the audited official-linked file')
        part.rename(WEIGHTS)
    assert sha(WEIGHTS) == WEIGHTS_SHA
    assert sha(SOURCE) == SOURCE_SHA, 'Official model source changed'
    print('MNet checkpoint verified:', WEIGHTS, flush=True)


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def load_model(device):
    assert sha(WEIGHTS) == WEIGHTS_SHA, 'Model changed'
    assert sha(SOURCE) == SOURCE_SHA, 'Official model source changed'
    spec = importlib.util.spec_from_file_location('official_mnetold_probe', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model = module.MNetold(wf=54, scale=42, vscale=42, args=SimpleNamespace(**CONFIG))
    state = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    return model, dict(epoch=checkpoint['epoch'], parameters=sum(p.numel() for p in model.parameters()),
                       state_keys=len(state), strict_state_dict_match=True)


def predict(src, model, device):
    """Original official output composition, adapted to the existing SLBR protocol."""
    h, w = src.shape[:2]

    def infer(rgb):
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(device) / 255
        with torch.inference_mode():
            background, probability = model(tensor)
        assert torch.isfinite(background).all().item()
        assert torch.isfinite(probability).all().item()
        bg = background[0].cpu().numpy().transpose(1, 2, 0) * 255
        prob = probability[0, 0].cpu().numpy()
        return bg, prob

    sync(device)
    start = time.perf_counter()
    small = cv2.resize(src, (256, 256), interpolation=cv2.INTER_AREA)
    bg, prob = infer(small)
    bg = cv2.resize(bg, (w, h))
    prob = cv2.resize(prob, (w, h))
    global_probability = prob.copy()
    global_output = base.quant(bg * prob[..., None] + src * (1 - prob[..., None]))
    sync(device)
    global_seconds = time.perf_counter() - start

    start = time.perf_counter()
    acc = np.zeros((h, w, 3), np.float32)
    pacc = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    ys = sorted(set(list(range(0, max(1, h - 255), 192)) + [max(0, h - 256)]))
    xs = sorted(set(list(range(0, max(1, w - 255), 192)) + [max(0, w - 256)]))
    feather = np.maximum(np.outer(np.hanning(256), np.hanning(256)), .02).astype(np.float32)
    for y in ys:
        for x in xs:
            crop = src[y:y + 256, x:x + 256]
            ch, cw = crop.shape[:2]
            padded = cv2.copyMakeBorder(crop, 0, 256 - ch, 0, 256 - cw, cv2.BORDER_REFLECT_101)
            bg, prob = infer(padded)
            prob, bg, win = prob[:ch, :cw], bg[:ch, :cw], feather[:ch, :cw]
            acc[y:y + ch, x:x + cw] += (bg * prob[..., None] + crop * (1 - prob[..., None])) * win[..., None]
            pacc[y:y + ch, x:x + cw] += prob * win
            weights[y:y + ch, x:x + cw] += win
    assert np.all(weights > 0)
    tiled_output = base.quant(acc / weights[..., None])
    sync(device)
    tiled_seconds = time.perf_counter() - start
    return {
        'mnet_global': (global_output, global_probability, global_seconds),
        'mnet_tiles': (tiled_output, pacc / weights, tiled_seconds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-model', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps')
    args = parser.parse_args()
    if args.prepare_model:
        prepare()
        return
    if args.out is None:
        parser.error('--out is required; existing result directories are never overwritten')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested MPS device is unavailable')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    manifest_path = base.DATA / 'manifest.jsonl'
    manifest = {r['case_id']: r for r in map(json.loads, manifest_path.read_text().splitlines())}
    old_rows = {r['case_id']: r for r in json.loads((BASELINE / 'results.json').read_text())}
    initial_code = {p.name: sha(p) for p in (ROOT / 'wm').glob('*.py')}
    t = time.perf_counter()
    model, model_info = load_model(args.device)
    sync(args.device)
    load_seconds = time.perf_counter() - t
    t = time.perf_counter()
    with torch.inference_mode():
        model(torch.zeros((1, 3, 256, 256), device=args.device))
    sync(args.device)
    warmup_seconds = time.perf_counter() - t
    provenance = dict(
        created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        source='https://github.com/Aitchson-Hwang/MNet',
        model_source_file='scripts/mynets/MNetold.py', source_sha256=sha(SOURCE),
        weights_source=f'https://drive.google.com/file/d/{WEIGHTS_ID}/view',
        checkpoint='LOGO-H/model_best.pth.tar; fixed before seeing outputs',
        weights_sha256=sha(WEIGHTS), official_config=CONFIG, model_info=model_info,
        device=args.device, torch=torch.__version__, python=sys.version, platform=platform.platform(),
        threads=4, model_load_seconds=load_seconds, warmup_seconds=warmup_seconds,
        protocol='RGB/255, no GAN normalization. Official model unchanged, strict checkpoint load. '
                 'Soft predicted mask composition. Global: resize to256 and map back, same as stored SLBR adapter. '
                 'Tiles:256 stride192 native pixels, Hann feather floor .02. Threshold .5 used only for mask scoring. '
                 'No training, no parameter tuning, no ground truth in inference. References loaded only after both predictions. '
                 'All timing includes transfer, inference, composition and sync; excludes warmup, scoring, disk writes. '
                 'One checkpoint, two predefined adaptations; not an original-paper benchmark reproduction.',
        case_ids=base.IDS, manifest_sha256=sha(manifest_path),
        baseline_folder=str(BASELINE), baseline_results_sha256=sha(BASELINE / 'results.json'),
        baseline_note='Saved historical current/SLBR outputs; no rerun of changed production code. '
                      'Input SHA verified and saved outputs rescored with the same current scorer.',
        runner_sha256=sha(Path(__file__)), scoring_source_sha256=sha(Path(base.__file__)),
        production_code_sha256=initial_code,
    )
    dump(out / 'provenance.json', provenance)
    shutil.copy2(__file__, out / 'runner_snapshot.py')
    rows = []
    for cid in base.IDS:
        case = manifest[cid]
        path = base.DATA / case['paths']['input']
        digest = sha(path)
        assert digest == case['source_input_sha256'] == old_rows[cid]['input_sha256']
        src = base.load(path)
        folder = out / cid
        folder.mkdir()
        base.save(folder / 'input.png', src)
        print('CASE', cid, case['label'], src.shape, flush=True)
        predictions = predict(src, model, args.device)
        # Reference arrays are read only after both automatic predictions complete.
        clean = alpha = None
        if case['score_mode'] == 'paired':
            clean = base.load(base.DATA / case['paths']['clean'])
            alpha = np.array(Image.open(base.DATA / case['paths']['alpha']).convert('L'), float) / 255
            base.save(folder / 'clean.png', clean)
            base.save(folder / 'truth_mask.png', (alpha > 0) * 255)
        for name, saved in [('stored_current', 'current'), ('stored_slbr_tiles', 'slbr_tiles')]:
            previous = base.load(BASELINE / cid / (saved + '.png'))
            mask = np.array(Image.open(BASELINE / cid / (saved + '_mask.png')).convert('L')) > 0
            predictions[name] = (previous, mask.astype(np.float32), old_rows[cid]['methods'][saved]['seconds'])
        row = dict(case_id=cid, label=case['label'], input_sha256=digest,
                   negative=case['negative'], score_mode=case['score_mode'], methods={})
        for name in NAMES:
            pred, probability, seconds = predictions[name]
            assert pred.shape == src.shape
            base.save(folder / (name + '.png'), pred)
            mask = probability > .5
            base.save(folder / (name + '_mask.png'), mask * 255)
            base.save(folder / (name + '_probability.png'), base.quant(probability * 255))
            metrics = base.scores(src, pred, clean, alpha, mask, seconds) if clean is not None else dict(seconds=seconds, probe_pass=None)
            metrics['changed_pixels'] = int(np.any(pred != src, axis=2).sum())
            metrics['max_input_change'] = int(np.abs(pred.astype(np.int16) - src).max())
            row['methods'][name] = metrics
        rows.append(row)
        dump(out / 'results.json', rows)
        print('DONE', cid, {k: {q: round(v[q], 3) if isinstance(v.get(q), float) else v.get(q)
                              for q in ['wm_mae', 'probe_pass', 'seconds', 'changed_pixels']}
                              for k, v in row['methods'].items()}, flush=True)
    ending_code = {p.name: sha(p) for p in (ROOT / 'wm').glob('*.py')}
    dump(out / 'verification.json', dict(completed=True, cases=len(rows), model_strict_load=True,
         production_sources_unchanged=initial_code == ending_code,
         concurrent_source_changes=[name for name in set(initial_code) | set(ending_code)
                                    if initial_code.get(name) != ending_code.get(name)],
         output_size='all native resolution', input_hashes_verified=True,
         no_labels_in_predict=True, weights_sha256=sha(WEIGHTS)))
    print('COMPLETED', out, flush=True)


if __name__ == '__main__':
    main()
