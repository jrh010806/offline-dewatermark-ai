"""Run fixed source-only automatic inference, then score; preserve every run."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DEFAULT=ROOT/'data/watermark_regression_min_v1'


def read_rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate(root):
    cases=read_rows(root/'manifest.jsonl');cov=json.loads((root/'coverage.json').read_text())
    assert sha(root/'manifest.jsonl')==cov['manifest_sha256']
    files=json.loads((root/'file_hashes.json').read_text())
    for path,digest in files.items():
        assert sha(root/path)==digest,path
    from tools.regression_retirements import load_profile
    removal=load_profile() if root.resolve()==DEFAULT.resolve() else None
    retired={r['case_id'] for r in removal['cases']} if removal else set()
    if removal:
        assert sha(root/'manifest.jsonl')==removal['active_manifest_sha256']
        assert not retired.intersection(r['case_id'] for r in cases)
    assert len(cases)==46-len(retired)
    assert all(r['original_split']=='validation' for r in cases if r['suite']=='core')
    assert sum(r['suite']=='core' for r in cases)==36
    assert sha(ROOT/'data/watermark_training_v1/splits/validation.jsonl')==cov['original_validation_split_sha256']
    return cases,dict(status='passed',files_checked=len(files),core_inputs=36,
        real_inputs=sum(r['suite']=='real' for r in cases),total_pages=sum(r['page_count'] for r in cases),
        retired_inputs=len(retired))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--suite',choices=['core','real','all'],default='core');ap.add_argument('--out',type=Path)
    ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    cases,checks=validate(args.root)
    if args.validate_only:print(json.dumps(checks,ensure_ascii=False,indent=2));return 0
    from wm import documents,pipeline,verdict
    from wm.contract import Thresholds
    out=args.out or ROOT/'out/min_regression'/datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    chosen=[r for r in cases if args.suite=='all' or r['suite']==args.suite]
    results=[]
    def save_reports():
        summary=dict(suite=args.suite,created_at=datetime.now().astimezone().isoformat(),
            completed_inputs=len(results),planned_inputs=len(chosen),case_manifest_sha256=sha(args.root/'manifest.jsonl'),results=results,
            note='Source-only inference; references loaded after inference. Real extension train sources are developer probes, not held-out generalization.')
        summary['paired_positive_pages']=sum(1 for r in results if not r['negative'] and r['score_mode']=='paired' for p in r['pages'])
        summary['paired_positive_passed']=sum(p.get('passed') is True for r in results if not r['negative'] and r['score_mode']=='paired' for p in r['pages'])
        summary['negative_pages']=sum(1 for r in results if r['negative'] for p in r['pages'])
        summary['negative_safe']=sum(p.get('passed') is True for r in results if r['negative'] for p in r['pages'])
        summary['manual_review_pages']=sum(p['status']=='manual_review' for r in results for p in r['pages'])
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        cards=[]
        def link(path):return os.path.relpath(path,out).replace(os.sep,'/')
        for r in results:
            for p in r['pages']:
                figs=[]
                for role,title in [('input_preview','输入'),('output','实际输出'),('clean_preview','干净参考')]:
                    if p.get(role):
                        u=link(out/p[role]);figs.append(f'<figure><a href="{html.escape(u)}" target="_blank"><img loading="lazy" src="{html.escape(u)}"></a><figcaption>{title}</figcaption></figure>')
                cards.append(f'<article><h2>{html.escape(r["case_id"]+" · "+r["label"])} · 第 {p["page"]} 页</h2><p>{html.escape(p["status"])} / {html.escape(str(p.get("passed")))}</p><div class="pair">'+''.join(figs)+f'</div><pre>{html.escape(json.dumps(p.get("metrics",{}),ensure_ascii=False,indent=2))}</pre><p>{html.escape(p.get("error", ""))}</p></article>')
        page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最小集回归结果</title><style>body{font:15px/1.8 system-ui;background:#f1f4f8;color:#25394e;padding:25px;max-width:1500px;margin:auto}article{background:white;border:1px solid #dae4ef;border-radius:10px;padding:24px;margin:20px 0}.pair{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;height:350px;object-fit:contain;background:#f3f5f7}pre{white-space:pre-wrap;font-size:12px}</style><h1>固定最小集自动回归结果</h1><p>配对正例 '+str(summary['paired_positive_passed'])+'/'+str(summary['paired_positive_pages'])+' 页通过；负例 '+str(summary['negative_safe'])+'/'+str(summary['negative_pages'])+' 安全保留；待人工复核 '+str(summary['manual_review_pages'])+' 页。</p><p>core 与 real 的来源属性见 summary.json，不能把 train 来源扩展当作留出验证。</p>'+''.join(cards)+'</html>'
        (out/'report.html').write_text(page)
    for index,case in enumerate(chosen,1):
        record={k:case[k] for k in ['case_id','label','suite','negative','score_mode','source_group','original_split']};record['pages']=[]
        try:
            # The application gets exactly one input path; no style or truth is passed.
            processed=pipeline.process_file(args.root/case['paths']['input'],dpi=96)
            assert len(processed)==case['page_count']
            # Load exact clean references only after source-only inference returns.
            clean_pages=documents.load_pages(args.root/case['paths']['clean'],dpi=96) if case['score_mode']=='paired' else None
            for i,result in enumerate(processed):
                folder=out/case['case_id'];folder.mkdir(exist_ok=True)
                def png(name,array):
                    dest=folder/f'p{i+1:02d}_{name}.png';Image.fromarray(np.asarray(array,dtype=np.uint8)).save(dest);return dest.relative_to(out).as_posix()
                changed=float(np.any(result.source.image!=result.output.image,axis=2).mean())
                item=dict(page=i+1,status='manual_review' if case['score_mode']=='manual_review' else 'scored',passed=None,
                    input_preview=png('input',result.source.image),output=png('output',result.output.image),
                    predicted_mask=png('predicted_mask',result.mask.binary()),mask_method=result.mask.method,changed_fraction=changed)
                if clean_pages is not None:
                    truth=case['pages'][i]['paths'] if case['kind']=='pdf' else case['paths']
                    with Image.open(args.root/truth['alpha']) as a:
                        alpha=np.asarray(a.convert('L').resize((result.source.width,result.source.height),Image.Resampling.LANCZOS),np.float32)/255
                    clean=clean_pages[i].image
                    item['clean_preview']=png('clean',clean)
                    score=verdict.build_verdict(source=result.source.image,output=result.output.image,mask=result.mask,
                        engine=result.verdict.engine,repairs=result.verdict.repairs,elapsed_ms=result.verdict.elapsed_ms,
                        stage_ms=result.verdict.stage_ms,thr=Thresholds(),clean_reference=clean,alpha_reference=alpha)
                    item['passed']=bool(changed==0 and result.mask.coverage()==0) if case['negative'] else bool(score.passed)
                    item['metrics']={k:float(v) if np.isfinite(v) else None for k,v in score.metrics.items()}
                    item['failures']=[str(f) for f in score.failures]
                record['pages'].append(item)
        except Exception as exc:
            # Every expected page stays in the denominator, even on failure.
            while len(record['pages'])<case['page_count']:
                record['pages'].append(dict(page=len(record['pages'])+1,status='error',passed=False,error=f'{type(exc).__name__}: {exc}'))
        results.append(record);save_reports()
        print(f'[{index}/{len(chosen)}] {case["case_id"]}: '+', '.join(p['status'] for p in record['pages']),flush=True)
    print(out/'report.html')
    print('这是固定集诊断，不含完整基线对照；项目防退化验收请执行 ./start.sh test。')
    # A completed diagnostic is not a quality pass. Unknown/manual pages also
    # cannot silently report success to shell callers.
    return int(any(p.get('passed') is not True for r in results for p in r['pages']))


if __name__=='__main__':raise SystemExit(main())
