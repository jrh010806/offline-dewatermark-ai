"""Check boundary repair against existing styles; all references are evaluation-only."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wm import documents,masks,ocr,repair
from wm.contract import Thresholds
from tools.diagnose_single import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/edge_cleanup'
CASES=('tiled_text_04','big_center_02','big_center_04','dot_grid_02')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows={}
    for stem in CASES:
        src=documents.load_pages(ROOT/'data/real_samples/jpg'/f'{stem}.jpg')[0].image
        start=time.perf_counter()
        keepout=ocr.text_zones(src)
        enabled=masks.estimate_single_page(src,keepout=keepout)
        with patch('wm.repeated._edge_fragments',return_value=[]):
            disabled=masks.estimate_single_page(src,keepout=keepout)
        unchanged=(np.array_equal(enabled.alpha,disabled.alpha) and
                   np.array_equal(enabled.footprint,disabled.footprint) and
                   enabled.repair_mode==disabled.repair_mode and
                   np.array_equal(enabled.veil_strength,disabled.veil_strength) and
                   np.array_equal(enabled.veil_offset,disabled.veil_offset))
        print(stem,'mask unchanged:',unchanged,'repair start',flush=True)
        output,_=repair.repair(src,enabled,Thresholds())
        elapsed=round((time.perf_counter()-start)*1000)
        # Evaluation only, after automatic inference has finished.
        clean=np.array(Image.open(ROOT/'data/real_samples/clean'/f'{stem}.png').convert('RGB'))
        alpha=np.load(ROOT/'data/real_samples/mask'/f'{stem}_veil.npz')['alpha']
        row=dict(shape=list(src.shape),method=enabled.method,edge_change_does_not_affect_mask=unchanged,
                 elapsed_ms=elapsed,input=metrics(src,clean,src,alpha),after=metrics(output,clean,src,alpha))
        if not unchanged:
            previous,_=repair.repair(src,disabled,Thresholds())
            row['without_edge_change']=metrics(previous,clean,src,alpha)
            Image.fromarray(previous).save(OUT/f'{stem}_without_edges.png')
        rows[stem]=row
        Image.fromarray(output).save(OUT/f'{stem}_regression.png')
        (OUT/'other_styles.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),'utf-8')
        print(stem,json.dumps(row,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
