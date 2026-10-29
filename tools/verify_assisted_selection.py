"""Source-only assisted trial. Human rectangles are explicit, never scored as automatic."""
from pathlib import Path
from dataclasses import fields
from hashlib import sha256
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wm import assisted, pipeline
from wm.contract import PageImage, PageResult, MaskLayer, VerdictReport, EngineKind, Thresholds
from wm.export_review import overlay_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exports', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text()); rows = []
    manifests = list(args.exports.glob('*/manifest.json'))
    allowed = {f.name for f in fields(Thresholds)}
    for cid, rectangle in plan['rectangles'].items():
        folder = args.out/cid; folder.mkdir()
        row = dict(id=cid, human_rectangle=rectangle, status='started', automatic=False)
        try:
            found = [(m.parent, page) for m in manifests for page in json.loads(m.read_text())['pages']
                     if page['page'].split('#')[0] == cid]
            if len(found) != 1:raise ValueError('Expected one saved automatic result')
            parent, page = found[0]
            load = lambda p: np.array(Image.open(p).convert('RGB'))
            source = PageImage(load(parent/page['files']['source']), 200, cid)
            output = PageImage(load(parent/page['files']['clean']), 200, cid)
            with np.load(parent/page['files']['mask_data'], allow_pickle=False) as data:
                mask = MaskLayer(data['alpha'].copy(), footprint=data['footprint'].copy(),
                                 method=page['mask_method'],
                                 repair_mode=page['selection']['repair_mode'])
            v = page['verdict']
            report = VerdictReport(source.key, EngineKind.RASTER, [], v['conditions'],
                                   {k: float('nan') if value is None else value for k, value in v['metrics'].items()},
                                   v['failures'], v['elapsed_ms'], v.get('stage_ms', {}))
            result = PageResult(source, output, mask, report, processing=page['processing'])
            thr = Thresholds(**{k: v for k, v in page['processing']['thresholds'].items() if k in allowed})
            before = output.image.copy()
            begin = time.perf_counter()
            proposal = assisted.find_strokes(result, rectangle, thr)
            selected = proposal.layer.binary(proposal.thresholds.mask_alpha_min) > 0
            Image.fromarray(before).save(folder/'自动处理基线.png')
            Image.fromarray(selected.astype(np.uint8)*255).save(folder/'候选笔画.png')
            Image.fromarray(overlay_mask(before, selected.astype(np.float32))).save(folder/'候选笔画预览.png')
            assert np.array_equal(before, result.output.image)
            row.update(selected_pixels=int(selected.sum()), detection_seconds=time.perf_counter()-begin,
                       preview_changes_pixels=0, detector_method=proposal.detector_method,
                       candidate_sha256=sha256(selected.tobytes()).hexdigest())
            if selected.any():
                final = assisted.apply_proposal(result, proposal)
                pipeline.write_results([final], folder/'应用格式导出')
                Image.fromarray(final.output.image).save(folder/'辅助修复结果.png')
                row.update(status='completed', outside_candidate_changed=int((np.any(final.output.image!=before,2)&~selected).sum()))
            else:
                Image.fromarray(before).save(folder/'辅助修复结果.png')
                row.update(status='no_stroke_evidence', outside_candidate_changed=0)
        except Exception as exc:
            row.update(status='error', error=repr(exc))
        rows.append(row)
        (args.out/'运行记录.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.out/'推理完成.json').write_text(json.dumps(dict(reference_images_read=False, count=len(rows))))


if __name__ == '__main__':main()
