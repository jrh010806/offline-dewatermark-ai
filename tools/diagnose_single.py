"""Diagnostic ablation; reference masks/parameters are controls, not production inputs.

Run locally: HF_HUB_OFFLINE=1 .venv/bin/python tools/diagnose_single.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from dataclasses import replace
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm import masks, ocr, repair, lama
from wm.contract import MaskLayer, Thresholds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out' / 'single_diagnosis'
THR = Thresholds()


def save(path, a):
    Image.fromarray(a).save(path)


def overlay(img, region):
    out = img.copy()
    out[region] = (img[region] * .5 + np.array([255, 30, 30]) * .5).astype(np.uint8)
    return out


def metrics(img, clean, src, alpha):
    diff = np.abs(img.astype(np.float32) - clean).mean(2)
    support = cv2.dilate((alpha > .01).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    change = np.abs(img.astype(np.float32) - src).mean(2)
    # Fixed reference region shared by all arms. Ratios are errors, not removal success rates.
    return dict(mae=float(diff.mean()), mae_watermark_roi=float(diff[support].mean()),
                mae_outside=float(diff[~support].mean()),
                outside_changed_gt4=float((change[~support] > 4).mean()))


def panel(path, tiles, crop=None):
    tw, th = 420, 600 if crop is None else 310
    canvas = Image.new('RGB', (tw * 3, (th + 40) * 2), '#e5e7eb')
    draw = ImageDraw.Draw(canvas)
    for i, (name, arr) in enumerate(tiles):
        im = Image.fromarray(arr)
        if crop:
            im = im.crop(crop)
        im = ImageOps.contain(im, (tw-12, th-12))
        x, y = (i % 3)*tw, (i // 3)*(th+40)
        draw.text((x+8,y+9), name, fill='black')
        canvas.paste(im, (x+(tw-im.width)//2,y+40+(th-im.height)//2))
    canvas.save(path)


def edge_control(src, clean, alpha, directory):
    # Known support, expanded only for this diagnostic to cover JPEG edge bleed.
    region = cv2.dilate((alpha > .01).astype(np.uint8), np.ones((5,5),np.uint8)) > 0
    start = time.perf_counter()
    raw = lama.inpaint(src, region, THR)
    elapsed = round((time.perf_counter()-start)*1000)
    strict = src.copy()
    strict[region] = raw[region]
    save(directory/'accurate_expanded_inpaint.png', raw)
    save(directory/'accurate_expanded_strict.png', strict)
    return raw, strict, dict(raw=dict(**metrics(raw,clean,src,alpha), elapsed_ms=elapsed),
                             strict=metrics(strict,clean,src,alpha))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for stem in ('heavy_dark_02', 'heavy_dark_04'):
        directory = OUT / stem
        directory.mkdir(exist_ok=True)
        base = ROOT / 'data' / 'real_samples'
        src = np.array(Image.open(base/'jpg'/f'{stem}.jpg').convert('RGB'))
        clean = np.array(Image.open(base/'clean'/f'{stem}.png').convert('RGB'))
        z = np.load(base/'mask'/f'{stem}_veil.npz')
        alpha, color, strength = z['alpha'], z['color'], float(z['strength'])
        assert alpha.shape == src.shape[:2] == clean.shape[:2]
        # Reconstruct the known synthesis to measure JPEG/reference mismatch.
        forward = np.clip(alpha[...,None]*strength*color +
                          (1-alpha[...,None]*strength)*clean,0,255).astype(np.uint8)
        rows = dict(shape=list(src.shape), reference_forward_mae=float(
            np.abs(src.astype(np.float32)-forward).mean()), arms={})
        t = time.perf_counter()
        keepout = ocr.text_zones(src)
        rows['ocr_ms'] = round((time.perf_counter()-t)*1000)
        rows['ocr_available'] = keepout is not None
        t = time.perf_counter()
        auto = masks.estimate_single_page(src, THR, close_px=11, keepout=keepout)
        rows['mask_ms'] = round((time.perf_counter()-t)*1000)
        no_ocr = masks.estimate_single_page(src, THR, close_px=11, keepout=None, use_ocr=False)
        support = alpha > .08
        pred = auto.alpha >= THR.mask_alpha_min
        pfoot = auto.footprint if auto.footprint is not None else pred
        rows['detection'] = dict(
            precision=float((pred & support).sum()/max(pred.sum(),1)),
            recall=float((pred & support).sum()/max(support.sum(),1)),
            footprint_precision=float((pfoot & support).sum()/max(pfoot.sum(),1)),
            footprint_recall=float((pfoot & support).sum()/max(support.sum(),1)),
            ocr_changes_alpha=bool(not np.array_equal(auto.alpha,no_ocr.alpha)),
            coverage=auto.coverage(), footprint_fraction=float(pfoot.mean()),
            veil_strength=auto.veil_strength.tolist(), veil_offset=auto.veil_offset.tolist(),
            true_strength=strength,true_color=color.tolist())
        accurate = MaskLayer(alpha=alpha.copy(), confidence=1,
                             veil_strength=np.full(3,1-strength,np.float32),
                             veil_offset=(color*strength).astype(np.float32),
                             paper=np.full(src.shape,255,np.float32), footprint=alpha>.01)
        location_only = replace(auto, alpha=alpha.copy(), footprint=alpha>.01)
        save(directory/'mask_auto.png',overlay(src,pred))
        save(directory/'footprint_auto.png',overlay(src,pfoot))
        save(directory/'mask_reference.png',overlay(src,support))
        save(directory/'original.png',src)
        save(directory/'reference.png',clean)
        images = {}
        original_run_tile = lama._run_tile
        for label, mask in [('auto',auto),('accurate_location',location_only),('accurate_location_and_veil',accurate)]:
            tile_ms = []
            def tracked(*args, **kwargs):
                start=time.perf_counter()
                result=original_run_tile(*args,**kwargs)
                tile_ms.append(round((time.perf_counter()-start)*1000))
                return result
            lama._run_tile = tracked
            print(stem,label,'start',flush=True)
            start = time.perf_counter()
            try:
                out, used = repair.repair(src, mask, THR)
            finally:
                lama._run_tile = original_run_tile
            images[label] = out
            rows['arms'][label] = dict(**metrics(out,clean,src,alpha),
                elapsed_ms=round((time.perf_counter()-start)*1000),
                tile_count=len(tile_ms),tile_ms=tile_ms,repairs=[str(u) for u in used])
            save(directory/f'{label}.png',out)
            print(stem,label,rows['arms'][label],flush=True)
        start=time.perf_counter()
        oracle=repair.alpha_unmix(src,accurate)
        images['known_parameter_inverse']=oracle
        rows['arms']['known_parameter_inverse']=dict(**metrics(oracle,clean,src,alpha),
            elapsed_ms=round((time.perf_counter()-start)*1000),
            note='Oracle control: uses known synthesis parameters; not automatic removal.')
        save(directory/'known_parameter_inverse.png',oracle)
        rows['arms']['unchanged_input']=metrics(src,clean,src,alpha)
        tiles=[('Input',src),('Automatic mask + current repair',images['auto']),
               ('Accurate location + current repair',images['accurate_location']),
               ('Accurate location/veil + current repair',images['accurate_location_and_veil']),
               ('Known parameters inverse (oracle)',oracle),('Clean reference',clean)]
        panel(directory/'comparison.png',tiles)
        crop=(75,220,535,550) if stem.endswith('02') else (130,160,450,305)
        panel(directory/'detail.png',tiles,crop)
        panel(directory/'masks.png',[('Input',src),('Automatic alpha',overlay(src,pred)),
            ('True watermark',overlay(src,support)),('Automatic repair footprint',overlay(src,pfoot)),
            ('OCR keep-out',overlay(src,keepout if keepout is not None else np.zeros_like(support))),
            ('Clean reference',clean)])
        raw, strict, edge = edge_control(src,clean,alpha,directory)
        rows['edge_control'] = edge
        panel(directory/'edge_control.png', [('Input',src),('Automatic pipeline',images['auto']),
            ('Accurate expanded mask, inpaint only',strict),('Clean reference',clean),
            ('Current inpaint with accurate mask',images['accurate_location_and_veil']),
            ('Known parameters inverse (oracle)',oracle)], crop)
        all_rows[stem]=rows
        (OUT/'metrics.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2),'utf-8')
    print('Saved',OUT,flush=True)

if __name__=='__main__':
    main()
