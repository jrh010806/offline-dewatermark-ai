"""Reproducible opt-in stroke experiment through the real application queue.

A plan owns evaluation references. Only RGB PageImages and ordinary processing
settings enter BatchWorker. Each callback loads references after inference.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

PROJECT=Path(__file__).resolve().parents[1]

def rgb(path):return np.array(Image.open(path).convert('RGB'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def score(selected,alpha):
    truth=alpha>0
    count,labels,_,_=cv2.connectedComponentsWithStats((~truth).astype(np.uint8),8)
    exterior=np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
    internal=np.ones(count,bool);internal[0]=False;internal[exterior]=False
    holes=internal[labels]
    tp=int((selected&truth).sum());fp=int((selected&~truth).sum());fn=int((~selected&truth).sum())
    return dict(recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                strong_recall=float(selected[alpha>=.08].mean()) if np.any(alpha>=.08) else None,
                true_positive_pixels=tp,false_positive_pixels=fp,missed_pixels=fn,
                selected_pixels=int(selected.sum()),selected_fraction=float(selected.mean()),
                image_pixels=int(selected.size),truth_pixels=int(truth.sum()),
                hole_pixels=int(holes.sum()),hole_selected_pixels=int((holes&selected).sum()),
                whitespace_metric='无字符级空白标注；字间空白逐图目视核对，多选背景数另报，不冒充精确字间指标')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--code-root',type=Path,default=PROJECT);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['desktop','cli'],default='desktop')
    ap.add_argument('--experimental',action='store_true');ap.add_argument('--baseline-results',type=Path)
    args=ap.parse_args();sys.path.insert(0,str(args.code_root.resolve()))
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import Qt
    from wm import pipeline
    from wm.batches import Batch,BatchWorker
    from wm.contract import Thresholds,PageImage
    from wm.export_review import overlay_mask
    from wm.lama import _resolve_model_path
    args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text());cases=plan['cases'];index={r['id']:r for r in cases}
    assert len(index)==len(cases)
    hashes={str(p.relative_to(args.code_root)):digest(p) for p in (args.code_root/'wm').rglob('*.py')}
    inputs={r['input']:digest(r['input']) for r in cases}
    models={'lama':{'path':_resolve_model_path(),'sha256':digest(_resolve_model_path())}}
    version={'code_root':str(args.code_root),'code_hashes':hashes,'models':models,
             'mode':args.mode,'experimental':args.experimental,'plan_sha256':digest(args.plan),'input_hashes':inputs}
    (args.out/'运行版本与参数.json').write_text(json.dumps(version,ensure_ascii=False,indent=2))
    old={r['id']:r for r in json.loads(args.baseline_results.read_text())} if args.baseline_results else {}
    jobs=[];lookup={};records=[]
    groups={}
    for r in cases:groups.setdefault(r.get('batch',r['id']),[]).append(r)
    for name,group in groups.items():
        pages=[PageImage(rgb(r['input']),r.get('dpi',200),r['id'],r.get('page_index',0),'image_file') for r in group]
        batch=Batch(name,pages);settings=group[0].get('processing',{})
        values=settings.get('thresholds',{}).copy();values['watermark_recall_priority']=args.mode=='desktop' or args.experimental
        if 'experimental_colour_refinement' in Thresholds.__dataclass_fields__:
            values['experimental_colour_refinement']=args.experimental
        elif args.experimental:raise ValueError('Snapshot does not implement the experimental candidate')
        thr=Thresholds(**values)
        jobs.append((batch,list(range(len(pages))),thr,settings.get('mask_close_px',11),settings.get('ocr_enabled',True),settings.get('watermark_style')))
        lookup[batch.uid]=group
    font=ImageFont.truetype('/System/Library/Fonts/PingFang.ttc',20)
    def event(payload):
        if payload['kind']!='page':return
        case=lookup[payload['batch']][payload['index']];number=cases.index(case)+1
        record={'id':case['id'],'label':case['label'],'group':case.get('group'),'number':number}
        result=payload['result']
        if result is None:
            record.update(status='error',error=payload['error']);records.append(record)
        else:
            folder=args.out/f'第{number:02d}张'
            pipeline.write_results([result],folder)
            manifest=json.loads((folder/'manifest.json').read_text());files=manifest['pages'][0]['files']
            record.update(status='completed',files={k:str(folder/v) for k,v in files.items()},
                          method=result.mask.method,repair_mode=result.mask.repair_mode,
                          processing=result.processing,elapsed_ms=result.verdict.elapsed_ms)
            selected=result.mask.binary(result.processing['thresholds']['mask_alpha_min'])>0
            # These files are never passed to the queue, detector or repairer.
            alpha=np.array(Image.open(case['alpha']).convert('L'),dtype=np.float32)/255
            clean=rgb(case['clean']);assert clean.shape==result.source.image.shape and alpha.shape==selected.shape
            record.update(score(selected,alpha));truth=alpha>0
            error=np.abs(result.output.image.astype(float)-clean).mean(2)
            record.update(watermark_mae=float(error[truth].mean()) if truth.any() else None,
                          background_mae=float(error[~truth].mean()) if (~truth).any() else None,
                          outside_selection_changed_pixels=int(np.any(result.output.image!=result.source.image,axis=2)[~selected].sum()),
                          changed_pixels=int(np.any(result.output.image!=result.source.image,axis=2).sum()),negative=bool(case.get('negative')))
            previous=old.get(case['id']);before_files=previous.get('files') if previous else case.get('baseline')
            panels=[('原图',result.source.image)]
            if before_files:
                before_mask=np.array(Image.open(before_files['mask']).convert('L'))>0
                before_output=rgb(before_files['clean'])
                record['baseline']=score(before_mask,alpha)
                e=np.abs(before_output.astype(float)-clean).mean(2)
                record['baseline'].update(watermark_mae=float(e[truth].mean()) if truth.any() else None,
                    background_mae=float(e[~truth].mean()) if (~truth).any() else None)
                panels+=[('基线选区',rgb(before_files['selection_overlay']))]
            panels+=[('候选选区' if args.experimental else '本版选区',overlay_mask(result.source.image,result.mask.alpha))]
            if before_files:panels+=[('基线处理结果',before_output)]
            panels+=[('候选处理结果' if args.experimental else '本版处理结果',result.output.image)]
            sheet=Image.new('RGB',(360*len(panels),650),'#f1f4f2');draw=ImageDraw.Draw(sheet)
            for i,(title,arr) in enumerate(panels):
                im=Image.fromarray(arr);im.thumbnail((348,590));sheet.paste(im,(i*360+(360-im.width)//2,48));draw.text((i*360+12,10),title,font=font,fill='#173d31')
            comparison=f'第{number:02d}张_选区与处理结果对照.jpg';sheet.save(args.out/comparison,quality=94);record['comparison']=comparison
            overlay=result.source.image.astype(float).copy();missed=truth&~selected;extra=~truth&selected
            overlay[missed]=overlay[missed]*.3+np.array([0,130,255])*.7;overlay[extra]=overlay[extra]*.3+np.array([255,150,0])*.7
            diagnostic=f'第{number:02d}张_漏选与多选标记.png';Image.fromarray(np.uint8(overlay)).save(args.out/diagnostic);record['diagnostic']=diagnostic
            records.append(record)
        (args.out/'逐图指标.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in record.items() if k in ('id','status','recall','precision','false_positive_pixels','method','error')},ensure_ascii=False),flush=True)
    worker=BatchWorker(jobs);worker.update.connect(event,Qt.DirectConnection);worker.run()
    assert len(records)==len(cases),'Missing queue outcomes'
    assert all(digest(args.code_root/p)==d for p,d in hashes.items()),'Inference code changed during run'
    assert all(digest(p)==d for p,d in inputs.items()),'Input image changed'
    def pct(value):return '未评估' if value is None else f'{value:.2%}'
    rows=[];sections=[]
    for r in sorted(records,key=lambda r:r['number']):
        title=f'第{r["number"]:02d}张｜{r["label"]}'
        if r['status']=='error':rows.append(f'<tr><td>{escape(title)}</td><td colspan="5">运行失败：{escape(r["error"])}</td></tr>');continue
        rows.append(f'<tr><td><a href="#case{r["number"]}">{escape(title)}</a></td><td>{pct(r["recall"])}</td><td>{pct(r["precision"])}</td><td>{r["missed_pixels"]}</td><td>{r["false_positive_pixels"]}</td><td>{r["hole_selected_pixels"]}</td><td>{pct(r["selected_fraction"])}</td></tr>')
        sections.append(f'<section id="case{r["number"]}"><h2>{escape(title)}</h2><p>原始编号：{escape(r["id"])}；漏选 {r["missed_pixels"]:,} 像素，多选背景 {r["false_positive_pixels"]:,} 像素，字洞误选 {r["hole_selected_pixels"]:,} / {r["hole_pixels"]:,}。选区外改动 {r["outside_selection_changed_pixels"]:,} 像素。字间空白无字符级标注，需目视复核。</p><a href="{r["comparison"]}"><img src="{r["comparison"]}"></a><details><summary>蓝色漏选、橙色多选</summary><a href="{r["diagnostic"]}"><img src="{r["diagnostic"]}"></a></details></section>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>笔画细化逐图验证</title><style>body{font:16px/1.7 system-ui;background:#f1f4f2;color:#253c35;margin:24px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}section{background:white;border-radius:12px;margin:24px 0;padding:20px}img{max-width:100%}a{color:#16734e}</style><h1>笔画细化逐图验证</h1><p>'+escape(plan.get('limitations',''))+'</p><p>实际应用队列；模式 '+escape(args.mode)+'；实验开关 '+str(args.experimental)+'。处理结束后才载入参考评分，所有失败样本保留。像素误差不是残留率。</p><table><tr><th>图片</th><th>召回率</th><th>精确率</th><th>漏选</th><th>多选背景</th><th>字洞误选</th><th>选区占图</th></tr>'+''.join(rows)+'</table>'+''.join(sections)+'</html>'
    (args.out/'笔画细化_逐图验证报告.html').write_text(page)
    print(f'完成 {len(records)} 张：{args.out}',flush=True)
    return int(any(r['status']=='error' for r in records))

if __name__=='__main__':raise SystemExit(main())
