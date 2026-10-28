"""Run the application on exactly three fixed development images and two guards."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wm import pipeline
from wm.contract import PageImage
from probe_removal_strategies import load,save,scores

DATA=ROOT/'data/watermark_regression_min_v1'
OLD=ROOT/'out/strategy_probe/20261027_small8_mps'
IDS=['core_001','core_015','real_003']
GUARDS=['core_023','core_024']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'out/three_image_cleanup/20261027/run_03')
    args=parser.parse_args()
    OUT=args.out.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise SystemExit('Refuse to overwrite results')
    protocol={'primary_cases':IDS,'internal_no_watermark_guards':GUARDS,
      'inference':'Unmodified source RGB -> default application pipeline. No supplied masks, coordinates, true alpha, clean image, filename routing, or website result pixels.',
      'change':'Contrast-assisted OCR localization. Dark/photo reuse LaMa. Bright document extends clipped OCR searches and uses existing LaMa only on white strokes, excluding locally darker print. Crosspage prior cannot replace specific contrast detections.',
      'revision_reason':'Full-page review of run_01/run_02 exposed watermark-shaped paper fills, text loss and negative outlines. Those outputs are retained. Final bright cleanup no longer pastes or inverts the estimated paper field; it is used only to protect print.',
      'priority':'First remove visible watermark. Record localized content damage separately and defer precision refinement; extensive document damage remains unacceptable.',
      'selection_note':'These are development examples repeatedly inspected, not an independent test set or a general success-rate estimate.',
      'tuning_rule':'Freeze initial implementation before this end-to-end output evaluation. Any later repair must be explicitly recorded as a new revision.',
      'network':'HF_HUB_OFFLINE=1; local OCR and cached LaMa weights; no website upload.',
      'timing':'End-to-end processing includes detection, repair and application verdict. Excludes report generation and reference scoring.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    manifest={r['case_id']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())}
    sources={};records=[];provenance={'wm_code':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'wm').glob('*.py')},'runner_sha256':sha(Path(__file__)),'inputs':{}}
    for cid in IDS+GUARDS:
        case=manifest[cid];src=load(DATA/case['paths']['input'])
        assert np.array_equal(src,load(OLD/cid/'input.png'))
        start=time.perf_counter()
        result=pipeline.process_pages([PageImage(src,96,'input',0,'image')],force_single=True)[0]
        elapsed=time.perf_counter()-start
        output=result.output.image;mask=result.mask.binary()>0
        assert np.array_equal(output[~mask],src[~mask])
        rec={'case_id':cid,'label':case['label'],'method':result.mask.method,'repair_mode':result.mask.repair_mode,'seconds':elapsed,'mask_pixels':int(mask.sum()),'changed_pixels':int(np.any(src!=output,axis=2).sum()),'outside_mask_byte_identical':True,'application_verdict':result.verdict.to_dict()}
        provenance['inputs'][cid]={'file':str(DATA/case['paths']['input']),'sha256':sha(DATA/case['paths']['input'])}
        if cid in IDS:
            folder=OUT/cid;folder.mkdir(exist_ok=True)
            for n,img in [('input',src),('baseline',load(OLD/cid/'current.png')),('output',output),('mask',mask*255)]:save(folder/(n+'.png'),img)
            overlay=src.astype(float);overlay[mask]=overlay[mask]*.45+np.array([255,60,60])*.55
            save(folder/'selection_overlay.png',overlay)
            # Reference scoring is after automatic inference. It is diagnostic;
            # the historic fidelity gate is not the user's new visual criterion.
            if case['score_mode']=='paired':
                clean=load(DATA/case['paths']['clean']);alpha=np.array(Image.open(DATA/case['paths']['alpha']).convert('L'),float)/255
                rec['reference_diagnostics']=scores(src,output,clean,alpha,mask,elapsed)
            else:rec['reference_diagnostics']=None
            records.append(rec)
        else:
            sources[cid]={'unchanged':bool(np.array_equal(src,output)),'method':result.mask.method,'changed_pixels':rec['changed_pixels'],'seconds':elapsed}
        (OUT/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        (OUT/'guards.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2))
        print(cid,result.mask.method,'mask',int(mask.sum()),'seconds',round(elapsed,2),'changes',rec['changed_pixels'],flush=True)
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    print('DONE',OUT,flush=True)


if __name__=='__main__':main()
