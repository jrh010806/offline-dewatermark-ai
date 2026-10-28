"""Reprocess supplied desktop exports through BatchWorker and score afterwards.

Only source PNGs enter detection/repair. Known labels are loaded after each
result, solely for local evaluation. Original exports remain untouched.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from wm import pipeline
from wm.batches import Batch, BatchWorker
from wm.contract import PageImage, Thresholds
from wm.export_review import overlay_mask


def rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exports', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--comparison-export', type=Path,
                        help='Optional previous exported version, shown separately from the supplied baseline')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs, contexts = [], {}
    comparison = {}
    if args.comparison_export:
        for path in args.comparison_export.glob('*/manifest.json'):
            for page in json.loads(path.read_text())['pages']:
                comparison[page['page'].rsplit('#', 1)[0]] = (path.parent, page)
    for root in args.exports:
        for manifest_path in sorted(root.glob('*/manifest.json')):
            folder = manifest_path.parent
            manifest = json.loads(manifest_path.read_text())
            pages = []
            for p in manifest['pages']:
                key, index = p['page'].rsplit('#', 1)
                pages.append(PageImage(rgb(folder/p['files']['source']), 200, key, int(index), 'image_file'))
            batch = Batch(folder.name, pages)
            settings = manifest['pages'][0]['processing']
            thr = Thresholds(**(settings['thresholds'] | {'watermark_recall_priority': True}))
            jobs.append((batch, list(range(len(pages))), thr,
                         settings.get('mask_close_px', 11), settings.get('ocr_enabled', True),
                         settings.get('watermark_style')))
            contexts[batch.uid] = (batch, folder, manifest)
    records, errors = [], []
    original_paths = []
    for _, folder, manifest in contexts.values():
        original_paths.append(folder/'manifest.json')
        original_paths.extend(folder/p['files'][kind] for p in manifest['pages'] for kind in ('source','mask','clean'))
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original_paths}
    code_paths = [ROOT/'wm'/f'{name}.py' for name in
                  ('recall', 'masks', 'observed_text', 'photo_repeat_inverse', 'photo_observations',
                   'texture_overlay', 'contract', 'lama', 'batches', 'verdict')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    (args.out/'算法版本校验.json').write_text(json.dumps(hashes, indent=2))

    def event(payload):
        if payload['kind'] != 'page':
            return
        batch, folder, manifest = contexts[payload['batch']]
        index = payload['index']
        result = payload['result']
        if result is None:
            errors.append({'batch': batch.name, 'index': index, 'error': payload['error']})
            print(json.dumps(errors[-1]), flush=True)
            return
        batch.results[index] = result
        before_page = manifest['pages'][index]
        key = result.source.doc_id
        # Evaluation only: these images never enter the worker's jobs.
        data = ROOT/'data'/('watermark_training_increment_v1_1' if key.startswith('inc11_') else 'watermark_training_v1')
        entries = {item['id']: item for item in map(json.loads, (data/'manifest.jsonl').read_text().splitlines())}
        entry = entries[key]
        if not np.array_equal(result.source.image, rgb(data/entry['paths']['watermarked'])):
            raise ValueError(f'{key}: export does not match evaluation labels')
        truth_alpha = np.array(Image.open(data/entry['paths']['alpha'])) / 255.
        truth = truth_alpha > 0
        clean = rgb(data/entry['paths']['clean'])
        old_selection = np.array(Image.open(folder/before_page['files']['mask'])) > 0
        old_output = rgb(folder/before_page['files']['clean'])
        selected = result.mask.binary() > 0
        record = dict(id=key, batch=batch.name, method=result.mask.method,
                      before_recall=float(old_selection[truth].mean()),
                      before_precision=float(truth[old_selection].mean()) if old_selection.any() else 0.,
                      before_selected_fraction=float(old_selection.mean()),
                      after_recall=float(selected[truth].mean()),
                      after_precision=float(truth[selected].mean()) if selected.any() else 0.,
                      strong_recall=float(selected[truth_alpha >= .08].mean()),
                      missed_pixels=int((truth & ~selected).sum()),
                      false_positive_pixels=int((~truth & selected).sum()),
                      selected_fraction=float(selected.mean()),
                      before_mae=float(np.abs(old_output.astype(float)-clean).mean(2)[truth].mean()),
                      after_mae=float(np.abs(result.output.image.astype(float)-clean).mean(2)[truth].mean()),
                      clean_area_mae=float(np.abs(result.output.image.astype(float)-clean).mean(2)[~truth].mean()),
                      elapsed_ms=result.verdict.elapsed_ms,
                      processing=result.processing,
                      residual_coverage=result.verdict.to_dict()['metrics']['residual_coverage'])
        record['repair_mode'] = result.mask.repair_mode
        record['outside_selection_unchanged'] = bool(np.array_equal(result.source.image[~selected], result.output.image[~selected]))
        extra_panel = []
        if key in comparison:
            previous_folder, previous_page = comparison[key]
            previous_mask = np.array(Image.open(previous_folder/previous_page['files']['mask'])) > 0
            record['broad_version'] = dict(recall=float(previous_mask[truth].mean()),
                precision=float(truth[previous_mask].mean()), selected_fraction=float(previous_mask.mean()),
                false_positive_pixels=int((previous_mask & ~truth).sum()))
            extra_panel = [('错误长条版', rgb(previous_folder/previous_page['files']['selection_overlay']))]
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in ('processing', 'batch')}), flush=True)
        (args.out/'逐图指标与复核结论.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
        diagnostic = result.source.image.copy().astype(float)
        missed, extra = truth & ~selected, selected & ~truth
        diagnostic[missed] = diagnostic[missed]*.3 + np.array([0, 130, 255])*.7
        diagnostic[extra] = diagnostic[extra]*.3 + np.array([255, 150, 0])*.7
        Image.fromarray(np.uint8(diagnostic)).save(args.out/f'第{len(records):02d}张_漏选与多选标记.png')
        panels = [('原图', result.source.image), ('原始选区', rgb(folder/before_page['files']['selection_overlay']))] + extra_panel + [
                  ('新笔画选区', overlay_mask(result.source.image, result.mask.alpha)), ('新处理结果', result.output.image)]
        sheet = Image.new('RGB', (360*len(panels), 650), '#f1f4f2')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 20)
        for i, (title, image) in enumerate(panels):
            im = Image.fromarray(image)
            im.thumbnail((348, 590))
            sheet.paste(im, (i*360+(360-im.width)//2, 48))
            draw.text((i*360+12, 10), title, font=font, fill='#173d31')
        sheet.save(args.out/f'第{len(records):02d}张_原图旧选区与新结果对照.jpg', quality=94)

    worker = BatchWorker(jobs)
    worker.update.connect(event, Qt.DirectConnection)
    worker.run()
    if len(records)+len(errors) != worker.total:
        raise RuntimeError('Some worker results were not evaluated')
    for batch, _, _ in contexts.values():
        pipeline.write_results([batch.results[i] for i in sorted(batch.results)], args.out/batch.name)
    (args.out/'处理异常记录.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2))
    current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    if current != hashes:
        raise RuntimeError('Inference code changed during verification')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in original_hashes.items()), 'Input export changed'
    (args.out/'原始导出完整性校验.json').write_text(json.dumps(original_hashes,ensure_ascii=False,indent=2))
    rows = []
    sections = []
    for number, r in enumerate(records, 1):
        key = escape(r['id'])
        rows.append(f'<tr><td><a href="#{key}">{key}</a></td><td>{r["before_recall"]:.2%}</td>'
                    f'<td>{r["after_recall"]:.2%}</td><td>{r["before_precision"]:.2%} → {r["after_precision"]:.2%}</td>'
                    f'<td>{r["missed_pixels"]}</td><td>{r["false_positive_pixels"]}</td><td>{r["selected_fraction"]:.2%}</td></tr>')
        sections.append(f'<section id="{key}"><h2>{key}</h2><p>水印区域与无水印参考的平均像素误差：'
                        f'{r["before_mae"]:.2f} → {r["after_mae"]:.2f}；原本无水印区域的平均误差：'
                        f'{r["clean_area_mae"]:.2f}。该误差同时包含残留和重建差异，不等于水印残留率。</p>'
                        f'<a href="第{number:02d}张_原图旧选区与新结果对照.jpg"><img src="第{number:02d}张_原图旧选区与新结果对照.jpg"></a>'
                        f'<details><summary>漏选与多选：蓝色为漏选，橙色为额外选中的背景</summary>'
                        f'<a href="第{number:02d}张_漏选与多选标记.png"><img src="第{number:02d}张_漏选与多选标记.png"></a></details></section>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>两批水印笔画验证</title><style>body{font:16px/1.7 system-ui;background:#f1f4f2;color:#253c35;max-width:1800px;margin:24px auto;padding:0 20px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #dbe5df;text-align:left}section{background:white;border-radius:12px;padding:20px;margin:24px 0}h2{font-size:18px;overflow-wrap:anywhere}img{max-width:100%}a{color:#17764f}</style>
<h1>两批水印笔画验证</h1><p>使用应用实际 BatchWorker 队列，开启“补全水印笔画”。只有用户导出中的原图进入识别和修复；真实水印标注与无水印参考在处理完成后才载入评分。</p>
<p>OCR 框和重复行只作搜索范围，最终选区依据颜色/对比像素和经验证的重复字形，保留字洞与字间空白。可信的透明度反解和周期纹理恢复继续保留。外扩参数为 0；未覆盖的淡笔画和复杂背景交叠仍按漏检报告，不以填框或扩大斜带补数值。</p>
<p>召回率分母为真实标注透明度大于 0 的全部水印像素，精确率为选中像素中真实水印像素的比例。重建误差、背景变形、选区质量需一起检查；无法可靠估算照片残留时仍显示未评估。</p>
<table><tr><th>图片</th><th>原召回率</th><th>新召回率</th><th>精确率 原 → 新</th><th>漏选像素</th><th>多选背景像素</th><th>新选区占图</th></tr>'''
    (args.out/'两批14张_水印笔画选区修复对照报告.html').write_text(html+''.join(rows)+'</table>'+''.join(sections)+
        '<p><a href="逐图指标与复核结论.json">测量数据</a> · <a href="算法版本校验.json">验证代码版本</a></p></html>', encoding='utf-8')
    print(f'Finished {len(records)} results, {len(errors)} errors: {args.out}', flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
