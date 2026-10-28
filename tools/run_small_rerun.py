"""Rerun the original small-eight plus the newly failed photo, without tuning.

Every inference takes RGB only. Reference scoring starts after all inference.
Errors remain in the fixed denominator. Does not mutate application code.
"""
from pathlib import Path
from datetime import datetime
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
from wm.contract import PageImage, Thresholds
from probe_removal_strategies import scores

OUT = ROOT/'out/small_dataset_rerun/20261027_fixed9'
DATA = ROOT/'data/watermark_regression_min_v1'
EXTRA = ROOT/'data/watermark_training_v1'
OLD = ROOT/'out/strategy_probe/20261027_small8_mps'
ACCEPTED = ROOT/'out/three_image_cleanup/20261027/run_03'
NEW = ROOT/'out/unseen_three/20261027_fixed'
IDS = ['core_001','core_010','core_015','core_016','core_020','real_003',
       'photo_021__local_text','core_023','core_024']
LABELS = ['01_照片_淡色局部文字','02_照片_密集重复文字','03_图表_深色大字',
          '04_图表_大块不透明遮挡','05_文档_带阴影文字','06_拍照文档_白色重复文字',
          '07_建筑照片_前轮失败局部文字','08_无水印照片','09_正常页眉标志与签署文档']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def source_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'wm').glob('*.py')}

def load(p):
    return np.array(Image.open(p).convert('RGB'))

def save(p,a):
    Image.fromarray(np.asarray(a,dtype=np.uint8)).save(p)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    frozen = source_hashes()
    accepted = json.loads((ACCEPTED/'阶段验收_明显水印清除优先.json').read_text())
    assert frozen == accepted['wm_source_sha256']
    core = {r['case_id']:r for r in map(json.loads,(DATA/'manifest.jsonl').read_text().splitlines())}
    corpus = {r['id']:r for r in map(json.loads,(EXTRA/'manifest.jsonl').read_text().splitlines())}
    selected = []
    for cid,label in zip(IDS,LABELS):
        if cid in core:
            r=core[cid]
            selected.append({'case_id':cid,'label':label,'input':str(DATA/r['paths']['input']),
                'input_sha256':r['source_input_sha256'],'source_group':r['source_group'],
                'source_id':r['source_id'],'split':r['original_split'],'negative':r['negative'],
                'reference_kind':r['score_mode'],'clean':str(DATA/r['paths']['clean']) if r['score_mode']=='paired' else None,
                'alpha':str(DATA/r['paths']['alpha']) if r['score_mode']=='paired' else None,
                'baseline':str(OLD/cid/'current.png'),'baseline_label':'最早小合集的自动版本'})
        else:
            r=corpus[cid]
            selected.append({'case_id':cid,'label':label,'input':str(EXTRA/r['paths']['watermarked']),
                'input_sha256':r['sha256']['watermarked'],'source_group':r['source_group'],
                'source_id':r['id'],'split':r['split'],'negative':r['is_negative'],
                'reference_kind':'paired','clean':str(EXTRA/r['paths']['clean']),
                'alpha':str(EXTRA/r['paths']['alpha']),
                'baseline':str(NEW/cid/'output.png'),'baseline_label':'上一轮新三张的同版本结果'})
    assert len(selected)==9 and sum(r['negative'] for r in selected)==2
    write('selection.json',selected)
    write('protocol.json',{
        'created_at':datetime.now().astimezone().isoformat(),
        'selection':'Original fixed strategy small-eight plus photo_021__local_text, a known failure from the preceding new-three test. Fixed before inference.',
        'counts':{'total':9,'watermarked':7,'negative_controls':2},
        'inference':'Source RGB -> accepted default pipeline.process_pages(force_single=True), generic doc_id=input. No masks, style selection, coordinates, clean pixels, filenames or website results passed into inference.',
        'rule':'One automatic call per selected input. No algorithm changes, rerolls or output filtering. Errors remain in the nine-input denominator.',
        'evaluation':'All references loaded only after all automatic inference attempts finish. Visible watermark outcome and localized damage reviewed separately; no automatic passed field is treated as visual success.',
        'limits':'Developer regression, not held-out performance. Includes accepted development samples, existing validation/test corpus with synthetic watermark templates and a previously tuned photographed document from a train-reserved source. No overall success-rate claim.',
        'opaque_case':'Large opaque block included as a difficulty boundary. Missing source data cannot be recovered exactly from a flat image; filling plausibly is not evidence of factual restoration.',
        'timing':'Wall time includes default detection, repair and application verdict, excludes reading/saving/reference evaluation; one pass, first applicable model initialization included.',
        'environment':'HF_HUB_OFFLINE=1; existing local OCR/LaMa assets, no uploads.'})
    provenance={'source_before':frozen,'runner_sha256':sha(Path(__file__)),
        'python':sys.version,'platform':platform.platform(),'thresholds':Thresholds().__dict__,
        'lama_model_sha256':sha(Path(lama._resolve_model_path())),
        'input_hashes':{r['case_id']:sha(Path(r['input'])) for r in selected}}
    write('provenance.json',provenance)
    records=[]
    for i,case in enumerate(selected,1):
        cid=case['case_id'];folder=OUT/cid;folder.mkdir()
        assert source_hashes()==frozen
        assert sha(Path(case['input']))==case['input_sha256']
        shutil.copy2(case['input'],folder/'input.png')
        src=load(folder/'input.png')
        row={k:case[k] for k in ['case_id','label','negative','reference_kind','source_group','split','baseline_label']}
        print(f'START {i}/9 {cid}',flush=True)
        start=time.perf_counter()
        try:
            r=pipeline.process_pages([PageImage(src,96,'input',0,'image')],force_single=True)[0]
            row['seconds']=time.perf_counter()-start
            output=r.output.image
            save(folder/'output.png',output)
            mask=r.mask.binary()>0
            save(folder/'mask.png',mask*255)
            overlay=src.astype(float);overlay[mask]=overlay[mask]*.45+np.array([255,45,45])*.55
            save(folder/'selection_overlay.png',overlay)
            changed=np.any(src!=output,axis=2)
            row.update({'status':'completed','method':r.mask.method,'repair_mode':r.mask.repair_mode,
                'mask_pixels':int(mask.sum()),'changed_pixels':int(changed.sum()),
                'outside_candidate_changed_pixels':int((changed&~mask).sum()),
                'application_verdict':r.verdict.to_dict()})
            shutil.copy2(folder/'output.png',OUT/(case['label']+'_自动处理结果.png'))
        except Exception as exc:
            row.update(status='error',seconds=time.perf_counter()-start,error=f'{type(exc).__name__}: {exc}')
        shutil.copy2(folder/'input.png',OUT/(case['label']+'_原图.png'))
        records.append(row)
        write('results.json',records)
        write('progress.json',{'completed':len(records),'planned':9,'reference_phase':False})
        print('DONE',cid,row['status'],row.get('method'),round(row['seconds'],2),flush=True)
    # Reference and previous-result pixels are not used by any automatic call.
    for case,row in zip(selected,records):
        cid=case['case_id'];folder=OUT/cid
        if row['status']!='completed':continue
        src,output=load(folder/'input.png'),load(folder/'output.png')
        mask=np.array(Image.open(folder/'mask.png'))>0
        baseline=Path(case['baseline'])
        if baseline.exists():
            shutil.copy2(baseline,folder/'baseline.png')
            old=load(baseline)
            row['baseline_output_identical']=bool(np.array_equal(old,output))
            row['baseline_changed_pixels']=int(np.any(old!=src,axis=2).sum())
        if cid in accepted['case_ids']:
            row['accepted_output_byte_identical']=bool(np.array_equal(output,load(ACCEPTED/cid/'output.png')))
        if case['negative']:
            row['negative_safe_byte_identical']=bool(np.array_equal(src,output))
        if case['reference_kind']=='paired':
            clean=load(Path(case['clean']));alpha=np.array(Image.open(case['alpha']).convert('L'),float)/255
            shutil.copy2(case['clean'],folder/'clean_reference.png')
            row['reference_diagnostics']=scores(src,output,clean,alpha,mask,row['seconds'])
            row['reference_diagnostics']['note']='Historical strict diagnostic gate, not phase visual acceptance or watermark removal rate. Reference error combines residual and repair damage.'
            if alpha.any():
                yy,xx=np.where(alpha>0)
                box=[max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(src.shape[1],int(xx.max())+33),min(src.shape[0],int(yy.max())+33)]
            else:box=[0,0,src.shape[1],src.shape[0]]
        else:
            row['reference_diagnostics']=None
            box=[0,0,src.shape[1],src.shape[0]]
        row['diagnostic_crop']=box
        for role in ['input','output','baseline','selection_overlay','clean_reference']:
            if (folder/(role+'.png')).exists():Image.open(folder/(role+'.png')).crop(box).save(folder/(role+'_crop.png'))
    assert source_hashes()==frozen
    provenance['source_after']=source_hashes()
    provenance['all_reference_pixels_loaded_after_all_inference']=True
    write('provenance.json',provenance);write('results.json',records)
    write('progress.json',{'completed':9,'planned':9,'reference_phase':True,'finished':True})
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
