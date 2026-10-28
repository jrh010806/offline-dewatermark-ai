"""Frozen default-pipeline probe on three preselected new corpus examples.

Run once. No algorithm changes, manual masks, filename routing or rerolls.
Reference pixels are loaded only after ALL automatic outputs have been saved.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from wm import pipeline, lama
from wm.contract import PageImage

OUT = ROOT / 'out/unseen_three/20261027_fixed'
DATA = ROOT / 'data/watermark_training_v1'
ACCEPTED = ROOT / 'out/three_image_cleanup/20261027/run_03/阶段验收_明显水印清除优先.json'
CASES = [('photo_021__local_text', '01_建筑照片_局部文字'),
         ('charts_015__heavy_dark', '02_图表_深色斜字'),
         ('document_029__tiled_diagonal', '03_中文文档_浅绿色重复斜字')]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load(p):
    return np.array(Image.open(p).convert('RGB'))

def save(p, a):
    Image.fromarray(np.uint8(a)).save(p)

def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'wm').glob('*.py')}

def main():
    resume = '--resume-after-recording-error' in sys.argv
    OUT.mkdir(parents=True, exist_ok=resume)
    if resume:
        assert not (OUT/'verification.json').exists(), 'Do not rerun completed experiment'
    accepted = json.loads(ACCEPTED.read_text())
    frozen = source_hashes()
    assert frozen == accepted['wm_source_sha256'], 'Accepted application code changed'
    manifest = {r['id']: r for r in map(json.loads, (DATA/'manifest.jsonl').read_text().splitlines())}
    regression = [json.loads(l) for l in (ROOT/'data/watermark_regression_min_v1/manifest.jsonl').read_text().splitlines()]
    excluded_groups = {r['source_group'] for r in regression}
    selected = [manifest[k] for k, _ in CASES]
    assert all(r['split'] == 'test' and r['source_group'] not in excluded_groups for r in selected)
    protocol = {
        'selected_before_inference': [r['id'] for r in selected],
        'selection': 'First source ID in test partition matching photo/local_text, chart/heavy_dark, Chinese document/tiled_diagonal; all minimum-regression source groups excluded.',
        'input_review': 'Only input images reviewed before selection. A white-on-white document candidate had little visible watermark except over printed strokes and was not selected; repeated pale-green text selected for an inspectable repetition test. No automatic output was generated for the discarded candidate.',
        'differences_from_development': 'Photo local mark has opacity 0.7 vs previous 0.3; document is project-original flat page with pale-green repeated text, not a photographed white repeated watermark.',
        'limits': 'New to the recent three-image tuning and absent from minimum-regression source groups. Existing corpus with shared synthetic templates; not an independently collected real-world holdout or an overall success-rate estimate.',
        'acceptance': 'Visible watermark removed; localized damage recorded separately and deferred. Obvious residual or extensive content loss means failure.',
        'inference': 'Source RGB only -> accepted default process_pages(force_single=True). No supplied region, style, clean image, alpha, or website output. Generic doc_id=input.',
        'rule': 'No algorithm tuning. Photo/chart one run each. Document first inference hit a runner assertion before saving; rerun once with identical application source/settings to retain its output. See runner_incident.json.',
        'reference_use': 'Clean and alpha loaded only after all three inference outputs are saved, for diagnostics and display.',
        'network': 'HF_HUB_OFFLINE=1; existing local OCR/LaMa assets. No upload.',
        'timing': 'Single run end-to-end detection, repair, application verdict. Excludes reading/saving, reference diagnostics and reporting; first case includes model initialization.'}
    write('protocol.json', protocol)
    write('selection.json', selected)
    provenance = {'wm_source_before': frozen, 'runner_sha256': sha(Path(__file__)),
                  'python': sys.version, 'platform': platform.platform(),
                  'accepted_record_sha256': sha(ACCEPTED), 'inputs': {},
                  'lama_model_sha256': sha(Path(lama._resolve_model_path()))}
    write('provenance.json', provenance)
    records = json.loads((OUT/'results.json').read_text()) if resume else []
    if resume:
        provenance['inputs'] = json.loads((OUT/'provenance.json').read_text())['inputs']
    for cid, label in CASES:
        if any(r['case_id'] == cid for r in records):
            continue
        assert source_hashes() == frozen
        case = manifest[cid]
        p = DATA/case['paths']['watermarked']
        assert sha(p) == case['sha256']['watermarked']
        folder = OUT/cid
        folder.mkdir(exist_ok=resume)
        shutil.copy2(p, folder/'input.png')
        src = load(p)
        print('START', cid, src.shape, flush=True)
        start = time.perf_counter()
        result = pipeline.process_pages([PageImage(src, 96, 'input', 0, 'image')], force_single=True)[0]
        seconds = time.perf_counter()-start
        mask = result.mask.binary() > 0
        output = result.output.image
        save(folder/'output.png', output)
        save(folder/'mask.png', mask*255)
        overlay = src.astype(float)
        overlay[mask] = overlay[mask]*.45 + np.array([255, 45, 45])*.55
        save(folder/'selection_overlay.png', overlay)
        for f, title in [('input.png', '原图'), ('output.png', '自动处理结果'), ('selection_overlay.png', '红色自动选区')]:
            shutil.copy2(folder/f, OUT/(label+'_'+title+'.png'))
        records.append({'case_id': cid, 'label': label, 'method': result.mask.method,
                        'repair_mode': result.mask.repair_mode, 'seconds': seconds,
                        'mask_pixels': int(mask.sum()), 'changed_pixels': int(np.any(src!=output,axis=2).sum()),
                        'outside_candidate_byte_identical': bool(np.array_equal(src[~mask], output[~mask])),
                        'outside_candidate_changed_pixels': int((np.any(src!=output,axis=2)&~mask).sum()),
                        'application_verdict': result.verdict.to_dict()})
        provenance['inputs'][cid] = {'path': str(p), 'sha256': sha(p)}
        write('results.json', records)
        write('provenance.json', provenance)
        print('DONE', cid, result.mask.method, round(seconds,2), flush=True)
    # No reference pixels have been read before this point.
    for rec in records:
        cid = rec['case_id']; case = manifest[cid]; folder = OUT/cid
        src, output = load(folder/'input.png'), load(folder/'output.png')
        ref_path = DATA/case['paths']['clean']
        assert sha(ref_path) == case['sha256']['clean']
        clean = load(ref_path)
        alpha_path = DATA/case['paths']['alpha']
        assert sha(alpha_path) == case['sha256']['alpha']
        alpha = np.array(Image.open(alpha_path).convert('L'))
        truth = alpha>0
        error_before = np.abs(src.astype(float)-clean).mean(2)
        error_after = np.abs(output.astype(float)-clean).mean(2)
        changed = np.any(src!=output,axis=2)
        selected_mask = np.array(Image.open(folder/'mask.png'))>0
        material = error_before>8
        rec['reference_diagnostics'] = {
            'note': 'Reference error mixes watermark residual and repair damage; it is not a removal rate or text accuracy.',
            'known_layer_pixels': int(truth.sum()),
            'known_layer_selected_fraction': float(selected_mask[truth].mean()),
            'source_reference_mean_error_in_layer': float(error_before[truth].mean()),
            'output_reference_mean_error_in_layer': float(error_after[truth].mean()),
            'source_reference_error_gt8_pixels': int(material.sum()),
            'unchanged_fraction_of_source_reference_error_gt8': float((~changed)[material].mean()) if material.any() else None,
            'mean_error_outside_known_layer': float(error_after[~truth].mean())}
        ys, xs = np.where(truth)
        box = [max(0,int(xs.min())-36),max(0,int(ys.min())-36),
               min(src.shape[1],int(xs.max())+37),min(src.shape[0],int(ys.max())+37)]
        rec['diagnostic_crop'] = box
        shutil.copy2(ref_path, folder/'clean_reference.png')
        for name in ['input','output','clean_reference','selection_overlay']:
            Image.open(folder/(name+'.png')).crop(box).save(folder/(name+'_crop.png'))
    provenance['wm_source_after'] = source_hashes()
    assert provenance['wm_source_after'] == frozen
    provenance['reference_loaded_after_all_inference'] = True
    write('results.json', records)
    write('provenance.json', provenance)
    print('COMPLETE', OUT, flush=True)

if __name__ == '__main__':
    main()
