"""One frozen known-position layer-estimation experiment.

Estimator is a pure numerical function: no paths, IDs, clean references, or
watermark alpha/colors. The background proxy is the saved source-only LaMa
result from the prior experiment, not a clean image.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from probe_removal_strategies import scores, load, save, quant, sha
OLD=ROOT/'out/strategy_probe/20261027_small8_mps'
DATA=ROOT/'data/watermark_regression_min_v1'

def estimate(src, support, background, cfg):
    """Return proposed inverse, accepted output, predicted alpha, and diagnostics."""
    src=np.asarray(src); support=np.asarray(support,dtype=bool)
    assert src.shape==background.shape and support.shape==src.shape[:2]
    alpha=np.zeros(support.shape,np.float32);info={'accepted':False,'reasons':[]}
    if not support.any():
        info['reasons']=['empty_support'];return src.copy(),src.copy(),alpha,info
    distance=cv2.distanceTransform(support.astype('uint8'),cv2.DIST_L2,5)
    core=distance>=cfg['core_distance_px'];ys,xs=np.nonzero(core)
    info['core_pixels']=len(ys)
    if len(ys)<cfg['minimum_observations']:
        info['reasons']=['insufficient_interior_observations'];return src.copy(),src.copy(),alpha,info
    take=np.linspace(0,len(ys)-1,min(len(ys),cfg['maximum_observations']),dtype=int)
    ys=ys[take];xs=xs[take];x=background[ys,xs].astype(float)/255;y=src[ys,xs].astype(float)/255
    contrast=float(np.sqrt(np.mean((x-x.mean(0))**2))*255)
    info['background_contrast']=contrast
    if contrast<cfg['background_channel_centered_std_min']:
        info['reasons']=['insufficient_background_variation'];return src.copy(),src.copy(),alpha,info
    def fit(x,y):
        # I = s B + (1-s) W, with physical parameter bounds.
        def residual(p):return (p[0]*x+(1-p[0])*p[1:]-y).ravel()
        opt=least_squares(residual,np.r_[.5,np.clip(np.median(y,axis=0),.001,.999)],bounds=(np.r_[.01,np.zeros(3)],np.r_[.995,np.ones(3)]),loss='soft_l1',f_scale=cfg['soft_l1_scale']/255,max_nfev=200)
        return opt.x, bool(opt.success)
    params,success=fit(x,y);s=float(params[0]);color=params[1:]*255
    residual=np.abs(s*x+(1-s)*params[1:]-y).mean(1)*255
    info.update(transmission=s,alpha_core=1-s,color_rgb=color.tolist(),fit_success=success,fit_residual_median=float(np.median(residual)),fit_residual_p90=float(np.percentile(residual,90)))
    split=((xs//32+ys//32)%2)==0
    if min(split.sum(),(~split).sum())<cfg['minimum_observations']//2:
        info['reasons'].append('insufficient_spatial_split')
    else:
        p1,ok1=fit(x[split],y[split]);p2,ok2=fit(x[~split],y[~split])
        da=float(abs(p1[0]-p2[0]));dc=float(abs(p1[1:]-p2[1:]).max()*255)
        info.update(split_alpha_difference=da,split_color_difference=dc,split_parameters=[p1.tolist(),p2.tolist()])
        if not(ok1 and ok2):info['reasons'].append('split_optimizer_failure')
        if da>cfg['split_alpha_difference_max']:info['reasons'].append('unstable_transmission')
        if dc>cfg['split_color_difference_max']:info['reasons'].append('unstable_watermark_color')
    if not success:info['reasons'].append('optimizer_failure')
    if not cfg['transmission_min_accepted']<=s<=cfg['transmission_max_accepted']:info['reasons'].append('unsupported_transmission')
    if info['fit_residual_median']>cfg['fit_residual_median_max']:info['reasons'].append('poor_overlay_fit')
    # Uniform opacity in stroke interiors. Only anti-aliased boundary pixels use
    # a source/proxy projection; the clean reference is never consulted.
    bg=background.astype(np.float32);vec=color-bg
    boundary_alpha=np.sum((src.astype(np.float32)-bg)*vec,axis=2)/np.maximum(np.sum(vec*vec,axis=2),1.)
    alpha[support]=np.clip(boundary_alpha[support],0,1-s)
    alpha[core]=1-s
    inverse=(src.astype(np.float32)-alpha[...,None]*color)/np.maximum(1-alpha[...,None],.01)
    info['inverse_clipped_fraction']=float(np.any((inverse<0)|(inverse>255),axis=2)[support].mean())
    candidate=src.copy();candidate[support]=quant(inverse)[support]
    info['accepted']=not info['reasons']
    output=candidate.copy() if info['accepted'] else src.copy()
    return candidate,output,alpha,info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args();out=args.out.resolve()
    protocol=json.loads((out/'protocol.json').read_text());cfg=protocol['fixed_parameters']
    if (out/'results.json').exists():raise SystemExit('Existing results: choose a fresh directory with frozen protocol.json')
    cases={r['case_id']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())}
    prior={r['case_id']:r for r in json.loads((OLD/'combined_results.json').read_text())}
    prev_pro=json.loads((OLD/'provenance.json').read_text())
    records=[];provenance={'protocol_sha256':sha(out/'protocol.json'),'script_sha256':sha(Path(__file__)),'previous_run':str(OLD),'inputs':{},'wm_code_sha256':{p.name:sha(p) for p in (ROOT/'wm').glob('*.py')}}
    (out/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    ids=protocol['primary_cases']+[protocol['boundary_case']]+protocol['negative_cases']+[protocol['exploratory_case']]
    for cid in ids:
        c=cases[cid];old=OLD/cid;folder=out/cid;folder.mkdir()
        src=load(DATA/c['paths']['input']);assert sha(DATA/c['paths']['input'])==prior[cid]['input_sha256']
        assert np.array_equal(src,load(old/'input.png'))
        negative=c['negative'];paired=c['score_mode']=='paired'
        if negative or not paired:
            support=np.array(Image.open(old/'slbr_tiles_mask.png').convert('L'))>0
            background=load(old/'slbr_mask_lama.png')
            mask_kind='prior_SLBR_false_positive_mask' if negative else 'approximate_prior_SLBR_mask'
        else:
            # Binary support only: this file contains 0/255, never true alpha.
            support=np.array(Image.open(DATA/c['paths']['mask']).convert('L'))>0
            assert np.array_equal(support,np.array(Image.open(old/'truth_mask.png'))>0)
            background=load(old/'known_mask_lama.png');mask_kind='exact_binary_support'
        digest={'source':sha(DATA/c['paths']['input']),'support_binary_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'background_proxy_sha256':hashlib.sha256(background.tobytes()).hexdigest(),'background_origin':str(old/('slbr_mask_lama.png' if negative or not paired else 'known_mask_lama.png'))}
        provenance['inputs'][cid]=digest
        start=time.perf_counter();candidate,pred,alpha,fit=estimate(src,support,background,cfg);seconds=time.perf_counter()-start
        # All estimates are finalized before any reference or true alpha is read.
        outputs={'input':src,'candidate':candidate,'guarded':pred,'background_proxy':src.copy(),'slbr_same_mask':src.copy(),'slbr_original':load(old/'slbr_tiles.png')}
        outputs['background_proxy'][support]=background[support]
        outputs['slbr_same_mask'][support]=outputs['slbr_original'][support]
        if negative:
            _,empty_out,_,empty_info=estimate(src,np.zeros_like(support),src,cfg)
            outputs['empty_mask_control']=empty_out;fit['empty_mask_byte_identical']=bool(np.array_equal(src,empty_out))
        save(folder/'support.png',support*255);save(folder/'estimated_alpha.png',np.clip(alpha*255,0,255));np.save(folder/'estimated_alpha.npy',alpha)
        row={'case_id':cid,'label':c['label'],'negative':negative,'score_mode':c['score_mode'],'mask_kind':mask_kind,'fit':fit,'estimation_seconds':seconds,'methods':{}}
        clean=truth_alpha=None
        if paired:
            clean=load(DATA/c['paths']['clean']);truth_alpha=np.array(Image.open(DATA/c['paths']['alpha']).convert('L'),float)/255;save(folder/'clean.png',clean)
            if not negative:
                gt=truth_alpha>0;core=cv2.distanceTransform(support.astype('uint8'),cv2.DIST_L2,5)>=cfg['core_distance_px']
                row['posthoc_diagnostics']={'alpha_mae_support':float(abs(alpha[gt]-truth_alpha[gt]).mean()),'true_core_alpha_median':float(np.median(truth_alpha[core])) if core.any() else None,'true_core_alpha_p10_p90':np.percentile(truth_alpha[core],[10,90]).tolist() if core.any() else None,'proxy_watermark_mae':float(abs(background.astype(float)-clean)[gt].mean())}
        for name,img in outputs.items():
            save(folder/(name+'.png'),img)
            if paired:row['methods'][name]=scores(src,img,clean,truth_alpha,support,seconds)
            else:row['methods'][name]={'manual_review':True,'probe_pass':None}
        if cid in protocol['primary_cases']:
            a=row['methods']['guarded'];b=row['methods']['slbr_same_mask']
            row['meets_continue_gate']=bool(a['probe_pass'] and a['wm_mae']<=.9*b['wm_mae'] and a['ink_iou']>=b['ink_iou']-.01)
        if negative:row['nonempty_negative_unchanged']=bool(np.array_equal(src,pred))
        records.append(row);(out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2,allow_nan=False));(out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
        print(cid,'accepted',fit['accepted'],fit['reasons'],'candidate_mae',row['methods']['candidate'].get('wm_mae'),'guarded_mae',row['methods']['guarded'].get('wm_mae'),flush=True)
    primary=[r for r in records if r['case_id'] in protocol['primary_cases']];neg=[r for r in records if r['negative']]
    decision={'primary_passed':sum(r['meets_continue_gate'] for r in primary),'primary_count':4,'negative_challenge_safe':sum(r['nonempty_negative_unchanged'] for r in neg),'negative_count':2,'proceed_to_detection':all(r['meets_continue_gate'] for r in primary) and all(r['nonempty_negative_unchanged'] for r in neg),'note':'This decision applies to the frozen estimator, not all possible layer-decomposition models.'}
    (out/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2));print(decision)
if __name__=='__main__':main()
