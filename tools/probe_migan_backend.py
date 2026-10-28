"""Fixed-mask backend comparison; production code and masks are never tuned here."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wm import contrast_text,lama
from wm.contract import Thresholds
from probe_removal_strategies import load,save,scores

OUT=ROOT/'out/repair_backend_compare/20261027_migan_fixed'
OLD=ROOT/'out/three_image_cleanup/20261027/run_03'
DATA=ROOT/'data/watermark_regression_min_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def make_session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1;opts.log_severity_level=3
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])


def migan_tile(session,image,mask):
    keep=(~mask.astype(bool)).astype(np.float32)[None,None]
    norm=image.transpose(2,0,1)[None].astype(np.float32)*2/255-1
    features=np.concatenate([keep-.5,norm*keep],axis=1)
    prediction=session.run(None,{'/Concat_23_output_0':features})[0][0].transpose(1,2,0)
    return np.clip((prediction*.5+.5)*255,0,255).astype(np.uint8)


def residual_diagnostic(image,domain):
    g=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    closed=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    base=cv2.morphologyEx(closed,cv2.MORPH_OPEN,np.ones((41,41),np.uint8))
    residual=closed.astype(float)-base
    return {'positive_contrast_mean':float(residual[domain].mean()),
            'positive_contrast_p90':float(np.percentile(residual[domain],90)),
            'contrast_gt8_fraction':float((residual[domain]>8).mean()),
            'warning':'Source-defined watermark-candidate region, local bright-stroke contrast only. Not a true residual watermark or OCR accuracy score.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['real_003','core_001','core_015'],default='real_003');args=ap.parse_args()
    cid=args.case;folder=OUT/cid;folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():raise SystemExit('Refuse to overwrite completed comparison')
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['wm_code_sha256'].items():assert sha(ROOT/name)==digest
    src=load(OLD/cid/'input.png');original_hash=hashlib.sha256(src.tobytes()).hexdigest()
    start=time.perf_counter();located=contrast_text.estimate(src);detection=time.perf_counter()-start
    assert located is not None
    assert np.array_equal(located.binary(),np.array(Image.open(OLD/cid/'mask.png')))
    domain=located.footprint
    mask=contrast_text.bright_repair_region(src,located) if located.repair_mode=='bright_paper_cleanup' else domain.copy()
    if cid=='real_003':
        assert np.array_equal(mask,np.load(OUT/'fixed_mask.npy'))
        assert hashlib.sha256(mask.tobytes()).hexdigest()==protocol['mask_sha256']
    save(folder/'input.png',src);save(folder/'fixed_mask.png',mask*255);save(folder/'candidate_mask.png',domain*255)
    overlay=src.astype(float);overlay[mask]=overlay[mask]*.45+[255*.55,40*.55,40*.55];save(folder/'fixed_mask_overlay.png',overlay)
    times={};init={};outputs={};checks={};thr=Thresholds()
    start=time.perf_counter();lama_session=lama._load_session();init['lama']=time.perf_counter()-start
    start=time.perf_counter()
    with patch('wm.lama._run_tile',wraps=lama._run_tile) as calls:
        outputs['lama_native']=lama.inpaint(src,mask,thr)
        times['lama_native_tiles']=calls.call_count
    times['lama_native']=time.perf_counter()-start
    checks['lama_reproduces_saved_application_output']=bool(np.array_equal(outputs['lama_native'],load(OLD/cid/'output.png')))
    assert checks['lama_reproduces_saved_application_output']
    print(cid,'lama',round(times['lama_native'],3),flush=True)
    start=time.perf_counter();core=make_session(OUT/'models/migan_core_512.onnx');init['migan_native']=time.perf_counter()-start
    start=time.perf_counter()
    with patch('wm.lama._load_session',return_value=core),patch('wm.lama._run_tile',side_effect=migan_tile) as calls:
        outputs['migan_native']=lama.inpaint(src,mask,thr)
        times['migan_native_tiles']=calls.call_count
    times['migan_native']=time.perf_counter()-start
    checks['same_native_tile_count']=times['lama_native_tiles']==times['migan_native_tiles']
    assert checks['same_native_tile_count']
    print(cid,'migan_native',round(times['migan_native'],3),flush=True)
    start=time.perf_counter();official=make_session(OUT/'models/migan_pipeline_v2.onnx');init['migan_official']=time.perf_counter()-start
    start=time.perf_counter()
    raw=official.run(None,{'image':np.ascontiguousarray(src.transpose(2,0,1)[None]),'mask':np.ascontiguousarray((~mask).astype(np.uint8)[None,None]*255)})[0][0].transpose(1,2,0)
    composed=src.copy();composed[mask]=raw[mask]
    outputs['migan_official']=composed
    times['migan_official']=time.perf_counter()-start
    times['official_raw_changes_outside_mask']=int(np.any(src[~mask]!=raw[~mask],axis=1).sum())
    print(cid,'migan_official',round(times['migan_official'],3),flush=True)
    # Only now load references for paired synthetic cases. Website result is not
    # read by this runner at any time.
    manifest=next(r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines()) if r['case_id']==cid)
    clean=alpha=None
    if manifest['score_mode']=='paired':
        clean=load(DATA/manifest['paths']['clean']);alpha=np.array(Image.open(DATA/manifest['paths']['alpha']).convert('L'),float)/255
    results={}
    for method,img in outputs.items():
        save(folder/(method+'.png'),img)
        checks[method+'_outside_mask_identity']=bool(np.array_equal(img[~mask],src[~mask]))
        checks[method+'_shape_dtype']=img.shape==src.shape and img.dtype==np.uint8
        results[method]={'repair_seconds':times[method],
                         'changed_pixels':int(np.any(src!=img,axis=2).sum()),
                         'outside_mask_byte_identical':checks[method+'_outside_mask_identity'],
                         'reference_diagnostics':scores(src,img,clean,alpha,mask,times[method]) if clean is not None else None,
                         'bright_residual_diagnostic':residual_diagnostic(img,domain) if cid=='real_003' else None}
    # Boundary checks are numerical sanity, not proof of removal quality.
    checks['source_array_not_mutated']=hashlib.sha256(src.tobytes()).hexdigest()==original_hash
    protected=domain&~mask;checks['protected_candidate_pixels_untouched']=all(np.array_equal(img[protected],src[protected]) for img in outputs.values())
    assert all(checks.values()),checks
    record={'case_id':cid,'source_sha256':original_hash,'effective_mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
            'effective_mask_pixels':int(mask.sum()),'candidate_pixels':int(domain.sum()),
            'protected_candidate_pixels':int(protected.sum()),'detection_seconds_excluded':detection,
            'session_initialization_seconds_excluded':init,'tile_context_diagnostics':times,'methods':results,'checks':checks,
            'runtime':{'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider','threads':4},
            'script_sha256':sha(Path(__file__)),'production_code_modified_by_probe':False}
    (folder/'results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print('DONE',cid,flush=True)


if __name__=='__main__':main()
