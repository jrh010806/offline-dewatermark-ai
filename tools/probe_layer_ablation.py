"""Factorial oracle diagnosis of the frozen core_015 result; not an algorithm score."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from probe_removal_strategies import load, quant, save, scores

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'out/layer_estimation/20261027_fixed'
OUT = PREV / 'alpha_ablation'
DATA = ROOT / 'data/watermark_regression_min_v1'


def inverse(src, alpha, color, support):
    raw = (src.astype(float) - alpha[..., None] * color) / np.maximum(1 - alpha[..., None], 1 / 255)
    result = src.copy()
    result[support] = quant(raw)[support]
    return result


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'results.json').exists():
        raise SystemExit('Refuse to overwrite the completed diagnosis')
    protocol = {
        'case_id': 'core_015',
        'purpose': 'Factorial replacement of alpha and color; isolate error sources, not demonstrate deployable removal.',
        'conditions': ['estimated_alpha_estimated_color', 'true_alpha_estimated_color', 'estimated_alpha_true_color', 'true_alpha_true_color'],
        'no_retuning': True,
        'decision_rule': 'Prioritize spatial alpha estimation only if true-alpha alone cuts ROI error by >=75% and has larger reduction than true-color alone; retain existing absolute fidelity gate. Otherwise report joint estimation or synthesis limitations.',
        'oracle_warning': 'Three conditions consume unavailable true layer parameters. None is a deployable algorithm.',
        'prescreen_amendment': 'Before any scoring: core_015 has 13 fully opaque pixels and 26 alpha>=0.98 pixels. Preserve source at the 13 opaque pixels in ALL four diagnostic conditions; report full ROI and recoverable alpha<0.98 separately. No LaMa fallback. This common oracle exclusion makes the baseline differ slightly from the prior unmodified estimator.',
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    case = next(r for r in map(json.loads, (DATA/'manifest.jsonl').read_text().splitlines()) if r['case_id']=='core_015')
    record = next(r for r in json.loads((PREV/'results.json').read_text()) if r['case_id']=='core_015')
    corpus = next(r for r in map(json.loads, (ROOT/'data/watermark_training_v1/manifest.jsonl').read_text().splitlines()) if r['id']==case['source_id'])
    layer_path = ROOT/'data/watermark_training_v1'/corpus['paths']['overlay_rgba']
    assert hashlib.sha256(layer_path.read_bytes()).hexdigest() == corpus['sha256']['overlay_rgba']
    rgba = np.array(Image.open(layer_path).convert('RGBA'))
    src = load(PREV/'core_015/input.png')
    truth_alpha = rgba[..., 3].astype(float)/255
    assert np.array_equal(rgba[...,3], np.array(Image.open(DATA/case['paths']['alpha']).convert('L')))
    support = truth_alpha > 0
    opaque = truth_alpha >= 1
    recoverable = (truth_alpha > 0) & (truth_alpha < .98)
    estimated_alpha = np.load(PREV/'core_015/estimated_alpha.npy')
    estimated_color = np.array(record['fit']['color_rgb'])
    true_color = rgba[...,:3]
    factors = [(estimated_alpha,estimated_color),(truth_alpha,estimated_color),(estimated_alpha,true_color),(truth_alpha,true_color)]
    # No changes to estimates: all four conditions share exactly one inversion function.
    outputs = {name: inverse(src,a,c,support & ~opaque) for name,(a,c) in zip(protocol['conditions'],factors)}
    assert np.array_equal(outputs['estimated_alpha_estimated_color'][~opaque],load(PREV/'core_015/candidate.png')[~opaque])
    clean = load(DATA/case['paths']['clean'])
    results = {name:scores(src,img,clean,truth_alpha,support,0) for name,img in outputs.items()}
    for name,img in outputs.items():
        results[name]['recoverable_roi_mae'] = float(np.abs(img.astype(float)-clean)[recoverable].mean())
        results[name]['fully_opaque_preserved_pixels'] = int(opaque.sum())
        results[name]['near_opaque_alpha_ge_098_pixels'] = int(((truth_alpha>=.98)&support).sum())
    for name,img in outputs.items():
        assert np.array_equal(src[~support],img[~support])
        save(OUT/(name+'.png'),img)
    baseline = results['estimated_alpha_estimated_color']['wm_mae']
    alpha_gain = 1-results['true_alpha_estimated_color']['wm_mae']/baseline
    color_gain = 1-results['estimated_alpha_true_color']['wm_mae']/baseline
    decision = {'alpha_only_error_reduction':alpha_gain,'color_only_error_reduction':color_gain,
                'prioritize_spatial_alpha':alpha_gain>=.75 and alpha_gain>color_gain,
                'oracle_alpha_estimated_color_absolute_pass':results['true_alpha_estimated_color']['probe_pass'],
                'both_oracle_absolute_pass':results['true_alpha_true_color']['probe_pass'],
                'deployable_success_claim':False}
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    paths=[Path(__file__),OUT/'protocol.json',layer_path,PREV/'core_015/estimated_alpha.npy',PREV/'results.json',DATA/case['paths']['input'],DATA/case['paths']['clean']]
    (OUT/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    outputs={'input':src,**outputs,'clean':clean}
    font=ImageFont.truetype(str(ROOT/'data/watermark_training_increment_v1_1/assets/fonts/NotoSansCJKsc-Regular.otf'),18)
    labels=['输入','当前：估计透明度 + 估计颜色','诊断：真实透明度 + 估计颜色','诊断：估计透明度 + 真实颜色','诊断：真实透明度 + 真实颜色','干净参考']
    crop=(270,260,880,620);sheet=Image.new('RGB',(1220,3*394),'#e6e9ed');draw=ImageDraw.Draw(sheet)
    for i,((name,img),label) in enumerate(zip(outputs.items(),labels)):
        x=i%2*610;y=i//2*394
        draw.text((x+10,y+4),label,font=font,fill='black')
        sheet.paste(Image.fromarray(img).crop(crop),(x,y+34))
    sheet.save(OUT/'comparison.jpg',quality=95)
    print(json.dumps({'results':results,'decision':decision},indent=2))


if __name__=='__main__':
    main()
