"""Re-run a saved desktop export against current code without altering inputs.

This local regression tool resolves the known sample stems to data/real_samples.
Reference pixels are loaded only after source-only processing, for scoring.
"""
from __future__ import annotations

# Historical data was retired on 2026-10-26; shared imported helpers remain usable.
if __name__ == "__main__":
    raise SystemExit('该脚本依赖的历史样本已退役。当前测试请使用 .venv/bin/python tools/run_min_regression.py --suite all；历史结果仍在 out/，不再覆盖或重跑。')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wm import documents,masks,pipeline
from wm.contract import Thresholds
from tools.diagnose_single import metrics

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for manifest in sorted(args.exports.glob('*/manifest.json')):
        for row in json.loads(manifest.read_text())['pages']:
            stem,index=row['page'].rsplit('#',1)
            if Path(stem).name!=stem or stem in ('.','..') or int(index)<0:
                raise ValueError('Unsafe sample identity in export')
            path=manifest.parent/f'{stem}_p{int(index):02d}_clean.png'
            entries.append((stem,int(index),path,row))
    if args.cases:entries=[e for e in entries if e[0] in args.cases]
    # Process document fixes first, then the more expensive photo regressions.
    entries.sort(key=lambda e:(0 if e[0].startswith('tiled_') or e[0]=='heavy_dark_01' else 2 if e[0]=='big_center_01' else 1,e[0]))
    saved=args.out/"metrics.json"
    rows=json.loads(saved.read_text()) if args.cases and saved.exists() else {}
    thr=Thresholds()
    (args.out/'run.json').write_text(json.dumps(dict(exports=str(args.exports.resolve()),
        thresholds=asdict(thr),close_px=11,use_ocr=True,reference_used_for_inference=False),indent=2))
    for stem,index,export,record in entries:
        print('start',stem,flush=True)
        source_path=ROOT/'data/real_samples/jpg'/f'{stem}.jpg'
        if not source_path.exists():raise FileNotFoundError(f'Original input required: {stem}')
        page=documents.load_pages(source_path)[0];page.page_index=index
        t=perf_counter();mask=masks.estimate_single_page(page.image,thr,close_px=11,use_ocr=True)
        mask_ms=round((perf_counter()-t)*1000)
        result=pipeline._process_one(page,mask,thr,mask_ms=mask_ms)
        result.processing.update(mask_close_px=11,ocr_enabled=True)
        pred=result.output.image
        Image.fromarray(pred).save(args.out/(stem+'.png'))
        np.savez_compressed(args.out/(stem+'_mask.npz'),alpha=mask.alpha,footprint=mask.footprint)
        # Evaluation starts here; neither reference is passed into the pipeline.
        clean=np.array(Image.open(ROOT/'data/real_samples/clean'/f'{stem}.png').convert('RGB'))
        alpha=np.load(ROOT/'data/real_samples/mask'/f'{stem}_veil.npz')['alpha']
        user=np.array(Image.open(export).convert('RGB'))
        assert user.shape==page.image.shape==clean.shape
        printed=clean.mean(2)<120
        paper=bool(np.mean(clean.mean(2)>235)>.65)
        rows[stem]=dict(source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            exported_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
            exported_path=str(export.resolve()),source_path=str(source_path),
            user=metrics(user,clean,page.image,alpha),output=metrics(pred,clean,page.image,alpha),
            user_elapsed_ms=record['verdict']['elapsed_ms'],elapsed_ms=result.verdict.elapsed_ms,
            stage_ms=result.verdict.stage_ms,method=mask.method,repair_mode=mask.repair_mode,
            unchanged_from_export=bool(np.array_equal(user,pred)),
            outside_mask_unchanged=bool(np.array_equal(pred[~mask.footprint],page.image[~mask.footprint])),
            document_ink_lift_gt50=float(np.mean((pred.mean(2)>clean.mean(2)+50)[printed])) if paper and printed.any() else None,
            reference_used_for_inference=False,verdict=result.to_dict())
        (args.out/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
        print('done',stem,'ms',result.verdict.elapsed_ms,'mask',mask.method,
              'user ROI',round(rows[stem]['user']['mae_watermark_roi'],2),
              'result ROI',round(rows[stem]['output']['mae_watermark_roi'],2),flush=True)
    print('Complete:',len(rows),'cases',flush=True)


if __name__=='__main__':main()
