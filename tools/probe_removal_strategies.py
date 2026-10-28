"""Small, immutable strategy experiment. Automatic methods never receive truth.

Known-mask / known-layer runs are privileged diagnostics, never automatic results.
No application algorithms are changed. Run from any directory with project Python.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/watermark_regression_min_v1'
MODEL=ROOT/'out/new_samples/research/slbr_256_devicefree.pt'
DEVICE='cpu'
IDS=['core_001','core_010','core_015','core_016','core_020','core_023','core_024','real_003']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return np.array(Image.open(p).convert('RGB'))
def save(p,a): Image.fromarray(np.uint8(a)).save(p)
def quant(a): return np.uint8(np.clip(np.rint(a),0,255))
def dilate(m): return cv2.dilate(m.astype('uint8'),np.ones((3,3),np.uint8)).astype(bool)

def slbr_predict(src,model,tiled=False):
    import torch
    h,w=src.shape[:2]
    def predict(a):
        x=torch.from_numpy(a.transpose(2,0,1).copy()).float()[None].to(DEVICE)/255
        with torch.inference_mode(): b,p=model(x)
        return b[0].cpu().numpy().transpose(1,2,0)*255,p[0,0].cpu().numpy()
    if not tiled:
        small=cv2.resize(src,(256,256),interpolation=cv2.INTER_AREA)
        b,p=predict(small)
        # Preserve native image outside predicted overlay; no whole-frame blur.
        p=cv2.resize(p,(w,h)); b=cv2.resize(b,(w,h))
        return quant(b*p[...,None]+src*(1-p[...,None])),p
    acc=np.zeros((h,w,3),np.float32);prob=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32)
    yy=sorted(set(list(range(0,max(1,h-255),192))+[max(0,h-256)]))
    xx=sorted(set(list(range(0,max(1,w-255),192))+[max(0,w-256)]))
    win=np.maximum(np.outer(np.hanning(256),np.hanning(256)),.02).astype(np.float32)
    for y in yy:
        for x in xx:
            crop=src[y:y+256,x:x+256];ch,cw=crop.shape[:2]
            a=cv2.copyMakeBorder(crop,0,256-ch,0,256-cw,cv2.BORDER_REFLECT_101)
            b,p=predict(a);z=win[:ch,:cw];b=b[:ch,:cw];p=p[:ch,:cw]
            acc[y:y+ch,x:x+cw]+=(b*p[...,None]+crop*(1-p[...,None]))*z[...,None]
            prob[y:y+ch,x:x+cw]+=p*z;weight[y:y+ch,x:x+cw]+=z
    return quant(acc/weight[...,None]),prob/weight

def automatic(src):
    """This function has no identifiers, references, masks, or style labels."""
    from wm import pipeline,lama
    from wm.contract import PageImage
    import torch
    outputs={}; timings={}; masks={}; extras={}
    def run(name,fn):
        t=time.perf_counter();out,mask=fn();timings[name]=time.perf_counter()-t
        outputs[name]=out;masks[name]=mask;print(name,round(timings[name],2),flush=True)
    t=time.perf_counter();r=pipeline.process_pages([PageImage(src,96,'probe',0,'image')],force_single=True)[0]
    outputs['current']=r.output.image;timings['current']=time.perf_counter()-t;masks['current']=r.mask.binary()>0
    extras['current_method']=r.mask.method; extras['current_repair_mode']=r.mask.repair_mode
    print('current',r.mask.method,round(timings['current'],2),flush=True)
    bm=dilate(masks['current'])
    run('current_mask_telea',lambda:(cv2.inpaint(src,bm.astype('uint8')*255,3,cv2.INPAINT_TELEA),bm))
    run('current_mask_lama',lambda:(lama.inpaint(src,bm),bm))
    # CPU and thread count fixed for both ONNX and torch experiments.
    t=time.perf_counter(); model=torch.jit.load(str(MODEL),map_location='cpu').eval().to(DEVICE);extras['slbr_load_seconds']=time.perf_counter()-t
    for tiled,name in [(False,'slbr_global'),(True,'slbr_tiles')]:
        t=time.perf_counter();out,p=slbr_predict(src,model,tiled)
        outputs[name]=out;masks[name]=p>.5;timings[name]=time.perf_counter()-t
        print(name,round(timings[name],2),flush=True)
    sm=dilate(masks['slbr_tiles'])
    run('slbr_mask_lama',lambda:(lama.inpaint(src,sm),sm))
    timings['slbr_mask_lama']+=timings['slbr_tiles']
    timings['current_mask_telea']+=timings['current'];timings['current_mask_lama']+=timings['current']
    return outputs,masks,timings,extras

def diagnostics(src,alpha,rgba):
    from wm import lama
    outputs={};masks={};timings={};m=dilate(alpha>0)
    for name,fn in [('known_mask_telea',lambda:cv2.inpaint(src,m.astype('uint8')*255,3,cv2.INPAINT_TELEA)),
                    ('known_mask_lama',lambda:lama.inpaint(src,m))]:
        t=time.perf_counter();outputs[name]=fn();timings[name]=time.perf_counter()-t;masks[name]=m
        print(name,round(timings[name],2),flush=True)
    t=time.perf_counter();a=rgba[...,3:4].astype(np.float32)/255;valid=(a[...,0]>0)&(a[...,0]<.98)
    out=src.copy();inv=(src.astype(np.float32)-a*rgba[...,:3])/np.maximum(1-a,1/255)
    out[valid]=quant(inv)[valid];opaque=(a[...,0]>=.98)
    # Do not claim unknown pixels were recovered: use existing oracle LaMa for them.
    out[opaque]=outputs['known_mask_lama'][opaque]
    outputs['known_layer_unmix']=out;masks['known_layer_unmix']=alpha>0
    timings['known_layer_unmix']=time.perf_counter()-t+(timings['known_mask_lama'] if opaque.any() else 0)
    return outputs,masks,timings,dict(opaque_fraction_of_watermark=float(opaque.sum()/max(1,(alpha>0).sum())),invertible_fraction=float(valid.sum()/max(1,(alpha>0).sum())))

def scores(src,out,clean,alpha,mask,seconds):
    from wm.verdict import _ink
    gt=alpha>0; e=np.abs(out.astype(np.float32)-clean).mean(2); before=np.abs(src.astype(np.float32)-clean).mean(2)
    delta=np.abs(out.astype(np.int16)-src).max(2);outside=~gt
    roi=float(e[gt].mean()) if gt.any() else None;b=float(before[gt].mean()) if gt.any() else None
    g=_ink(clean);o=_ink(out);zone=cv2.dilate(gt.astype('uint8'),np.ones((9,9),np.uint8)).astype(bool)
    iou=float((g&o&zone).sum()/max(1,((g|o)&zone).sum())) if (g&zone).any() else 1.
    result=dict(wm_mae=roi,input_wm_mae=b,wm_error_reduction_pct=100*(1-roi/b) if b else None,
                outside_mae=float(e[outside].mean()),outside_p99=float(np.percentile(e[outside],99)),
                outside_changed_gt4=float((delta[outside]>4).mean()),whole_changed_gt4=float((delta>4).mean()),
                whole_mae=float(e.mean()),ink_iou=iou,seconds=seconds,
                mask_precision=float((mask&gt).sum()/max(1,mask.sum())),mask_recall=float((mask&gt).sum()/max(1,gt.sum())),
                wm_error_gt10_fraction=float((e[gt]>10).mean()) if gt.any() else None)
    # Explicit experimental gate. Not the product's older four-condition score.
    result['probe_pass']=bool(roi<=3 and result['wm_error_gt10_fraction']<=.1 and result['outside_mae']<=1 and result['outside_p99']<=4 and iou>=.9 and seconds<=120) if gt.any() else bool(np.array_equal(out,src))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--slbr-device',choices=['cpu','mps'],default='cpu');args=ap.parse_args()
    global DEVICE
    DEVICE=args.slbr_device
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(4)
    out=args.out or ROOT/'out/strategy_probe'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    cases={r['case_id']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())}
    corpus={r['id']:r for r in map(json.loads,(ROOT/'data/watermark_training_v1/manifest.jsonl').read_text().splitlines())}
    from wm import lama
    provenance=dict(created_at=datetime.now().astimezone().isoformat(),python=sys.version,platform=platform.platform(),device='LaMa/OCR: CPU; SLBR: '+DEVICE,threads=4,
        torch=torch.__version__,model_sha256=sha(MODEL),model_path=str(MODEL),lama_sha256=sha(Path(lama._resolve_model_path())),
        manifest_sha256=sha(DATA/'manifest.jsonl'),script_sha256=sha(Path(__file__)),
        code_sha256={p.name:sha(p) for p in (ROOT/'wm').glob('*.py')},case_ids=IDS,
        protocol='Native resolution. SLBR global uses 256x256 resize, SLBR tiles use 256x256 stride192 feather. Masks dilated1px for inpainting. No truth in automatic inference. Truth only in explicitly named known_* diagnostics. Clean only in scoring.',
        gate='Paired positive: wm_MAE<=3; fraction(wm mean-channel error>10)<=.1; outside_MAE<=1; outside_p99<=4; adaptive-stroke IoU>=.9; seconds<=120. Negative: byte-identical. This diagnostic gate is not a commercial success rate.',
        timing='CPU wall seconds; current-mask variants conservatively include full current pipeline; slbr-mask-lama includes tiled inference. Per-case SLBR load separately recorded. No inference timing includes scoring or saving.')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    records=[]
    for cid in IDS:
        c=cases[cid];folder=out/cid;folder.mkdir();src=load(DATA/c['paths']['input']);save(folder/'input.png',src)
        assert sha(DATA/c['paths']['input'])==c['source_input_sha256']
        print('\nCASE',cid,c['label'],flush=True)
        outputs,masks,timings,extras=automatic(src)
        # Only now load labels/reference pixels; never pass them into automatic().
        clean=alpha=None
        if c['score_mode']=='paired':
            clean=load(DATA/c['paths']['clean']);alpha=np.array(Image.open(DATA/c['paths']['alpha']).convert('L'),float)/255
            save(folder/'clean.png',clean);save(folder/'truth_mask.png',(alpha>0)*255)
            if not c['negative']:
                r=corpus[c['source_id']];p=ROOT/'data/watermark_training_v1'/r['paths']['overlay_rgba'];rgba=np.array(Image.open(p).convert('RGBA'))
                assert sha(p)==r['sha256']['overlay_rgba']
                diag,dm,dt,de=diagnostics(src,alpha,rgba);outputs.update(diag);masks.update(dm);timings.update(dt);extras.update(de)
        row=dict(case_id=cid,label=c['label'],source_id=c['source_id'],source_group=c['source_group'],input_sha256=sha(DATA/c['paths']['input']),negative=c['negative'],score_mode=c['score_mode'],extras=extras,methods={})
        for name,pred in outputs.items():
            assert pred.shape==src.shape
            save(folder/(name+'.png'),pred);save(folder/(name+'_mask.png'),masks[name]*255)
            row['methods'][name]=scores(src,pred,clean,alpha,masks[name],timings[name]) if clean is not None else dict(seconds=timings[name],probe_pass=None,manual_review=True)
        records.append(row);(out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False))
        print('DONE',cid,flush=True)
    print(out,flush=True)
if __name__=='__main__':main()
