"""One source-only, fixed pixel-alpha projection ablation on the same eight cases."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from probe_layer_estimation import estimate
from probe_removal_strategies import load, quant, save, scores

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT/'out/layer_estimation/20261027_fixed'
OUT = PREV/'spatial_alpha'


def spatial(src, support, background, cfg):
    # Pure inference: no clean image, true alpha/color, paths, or sample identity.
    old_candidate, old_guarded, old_alpha, fit = estimate(src,support,background,cfg)
    if 'color_rgb' not in fit:
        return old_candidate,old_guarded,old_alpha,fit
    color = np.array(fit['color_rgb'])
    direction = color-background.astype(float)
    alpha = np.sum((src.astype(float)-background)*direction,axis=2)/np.maximum(np.sum(direction**2,axis=2),1.)
    alpha = np.where(support,np.clip(alpha,0,.98),0).astype(np.float32)
    raw = (src.astype(float)-alpha[...,None]*color)/np.maximum(1-alpha[...,None],.02)
    candidate = src.copy();candidate[support] = quant(raw)[support]
    guarded = candidate.copy() if fit['accepted'] else src.copy()
    return candidate,guarded,alpha,fit


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'results.json').exists():raise SystemExit('Refuse to overwrite results')
    protocol = {'hypothesis':'Changing only uniform interior alpha to source/proxy RGB projection improves fidelity.',
        'alpha_rule':'clip(dot(I-Bproxy, W-Bproxy)/max(norm(W-Bproxy)^2,1),0,0.98) at every supplied support pixel',
        'fixed':'Previous color estimator, acceptance rules, inputs, support, comparator, and fidelity gates; no per-case tuning.',
        'limitation':'Projected alpha is dependent on the imperfect LaMa background proxy. This is an ablation, not an independent recovery of the true layer.',
        'oracle_inputs':False,'primary_cases':['core_001','core_010','core_015','core_020'],
        'controls':['core_016','core_023','core_024'],'exploratory':'real_003',
        'gate':'Same all-four absolute quality + >=10% MAE gain vs same-mask SLBR + no stroke IoU loss >.01; negatives unchanged. Otherwise do not integrate.',
        'no_retuning':True,'timing':'Estimation only; cached proxy cost excluded. Scores.seconds is not end-to-end timing.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    config=json.loads((PREV/'protocol.json').read_text())['fixed_parameters']
    prev=json.loads((PREV/'results.json').read_text()); prov=json.loads((PREV/'provenance.json').read_text())
    manifest={r['case_id']:r for r in map(json.loads,(ROOT/'data/watermark_regression_min_v1/manifest.jsonl').read_text().splitlines())}
    records=[]
    for r in prev:
        cid=r['case_id'];folder=OUT/cid;folder.mkdir();src=load(PREV/cid/'input.png')
        support=np.array(Image.open(PREV/cid/'support.png'))>0
        background=load(Path(prov['inputs'][cid]['background_origin']))
        assert hashlib.sha256(background.tobytes()).hexdigest()==prov['inputs'][cid]['background_proxy_sha256']
        start=time.perf_counter();candidate,guarded,alpha,fit=spatial(src,support,background,config);elapsed=time.perf_counter()-start
        # References only after all outputs have been finalized.
        rec={'case_id':cid,'fit':fit,'estimation_seconds':elapsed,'methods':{}}
        for name,img in [('candidate',candidate),('guarded',guarded)]:
            assert np.array_equal(src[~support],img[~support]);save(folder/(name+'.png'),img)
            if r['score_mode']=='paired':
                clean=load(PREV/cid/'clean.png');truth=np.array(Image.open(ROOT/'data/watermark_regression_min_v1'/manifest[cid]['paths']['alpha']).convert('L'),float)/255
                rec['methods'][name]=scores(src,img,clean,truth,support,elapsed)
            else:rec['methods'][name]={'manual_review':True}
        save(folder/'estimated_alpha.png',alpha*255);np.save(folder/'estimated_alpha.npy',alpha)
        if cid in protocol['primary_cases']:
            a=rec['methods']['guarded'];b=r['methods']['slbr_same_mask']
            rec['meets_continue_gate']=bool(a['probe_pass'] and a['wm_mae']<=.9*b['wm_mae'] and a['ink_iou']>=b['ink_iou']-.01)
        if r['negative']:rec['negative_unchanged']=bool(np.array_equal(src,guarded))
        records.append(rec);print(cid,rec['methods']['guarded'].get('wm_mae'),rec['methods']['guarded'].get('ink_iou'),flush=True)
    decision={'primary_passed':sum(r.get('meets_continue_gate',False) for r in records),'primary_count':4,
              'negative_safe':all(r.get('negative_unchanged',True) for r in records)}
    decision['proceed_to_detection']=decision['primary_passed']==4 and decision['negative_safe']
    (OUT/'results.json').write_text(json.dumps(records,indent=2))
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2))
    paths=[Path(__file__),ROOT/'tools/probe_layer_estimation.py',OUT/'protocol.json',PREV/'protocol.json',PREV/'provenance.json']
    (OUT/'provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    print(json.dumps(decision))


if __name__=='__main__':main()
