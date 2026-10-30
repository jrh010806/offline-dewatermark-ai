"""Score sealed source-only dot runs, then build the Chinese comparison report."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_image(path):
    return np.asarray(Image.open(path).convert('RGB'))


def score(source, output, selected, clean, alpha):
    truth = alpha > 0
    strong = alpha >= 21/255
    change = np.any(source != output, axis=2)
    error = abs(output.astype(float)-clean).mean(2)
    before = abs(source.astype(float)-clean).mean(2)
    # This is a geometric hole diagnostic, not semantic character annotation.
    padded = np.pad(truth.astype(np.uint8), 1)
    filled = padded.copy()
    cv2.floodFill(filled, None, (0, 0), 1)
    holes = (~truth) & (filled[1:-1, 1:-1] == 0)
    dark = clean.mean(2) < 100
    return dict(recall_all_alpha_gt_zero=float((selected & truth).sum()/truth.sum()) if truth.any() else None,
                recall_alpha_ge_21_255=float((selected & strong).sum()/strong.sum()) if strong.any() else None,
                precision=float((selected & truth).sum()/selected.sum()) if selected.any() else None,
                missed_pixels=int((truth & ~selected).sum()), false_positive_pixels=int((selected & ~truth).sum()),
                selected_pixels=int(selected.sum()), selected_fraction=float(selected.mean()),
                enclosed_hole_false_pixels=int((selected & holes).sum()), semantic_whitespace_false_pixels=None,
                watermark_mae=float(error[truth].mean()) if truth.any() else None,
                source_watermark_mae=float(before[truth].mean()) if truth.any() else None,
                background_mae=float(error[~truth].mean()),
                dark_content_mae=float(error[dark].mean()) if dark.any() else None,
                dark_content_error_gt3_pixels=int((error[dark] > 3).sum()),
                outside_selection_changed_pixels=int(change[~selected].sum()), changed_pixels=int(change.sum()),
                residual_rate=None, recognition_accuracy=None)


def main():
    ap = argparse.ArgumentParser(description='推理结束后评分，生成文档点阵新旧对照')
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    out = args.run.resolve()
    assert (out/'主对照推理结束_未读取参考.json').is_file()
    seal = json.loads((out/'候选冻结校验.json').read_text())
    for name, digest in seal['files'].items():
        assert hashlib.sha256((out/'候选冻结'/name).read_bytes()).hexdigest() == digest
    inventory = json.loads((out/'完整样本清单_评分参考不传推理.json').read_text())
    cases = {x['id']: x for x in inventory['cases']}
    review_path = out/'人工复核与结论.json'
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    review_by_id = {r['id']: r for r in reviews.get('records', [])}
    versions = {}
    for label, directory in (('原版', '原版实际运行'), ('候选', '候选实际运行')):
        versions[label] = {x['key']: x for p in (out/directory).glob('*/inference.json')
                           for x in json.loads(p.read_text())}
    assert set(versions['原版']) == set(versions['候选'])
    rows = []
    from wm import documents
    figures = out/'逐张核对图'
    figures.mkdir(exist_ok=True)
    for key, old in versions['原版'].items():
        new = versions['候选'][key]
        row = dict(key=key, id=old['id'], style=cases[old['id']]['style'],
                   source_group=cases[old['id']]['source_group'], versions={}, numerical_failures=[])
        row['manual_review'] = review_by_id.get(old['id'])
        if old['status'] != 'completed' or new['status'] != 'completed':
            row.update(error={'old': old, 'new': new})
            rows.append(row)
            continue
        case = cases[old['id']]
        source = documents.load_pages(case['input'], dpi=case['dpi'])[old['page']-1].image
        ref = case['references'][old['page']-1]
        clean = load_image(ref['clean'])
        if Path(ref['alpha']).suffix == '.npz':
            alpha = np.load(ref['alpha'])['alpha']
        else:
            alpha = np.array(Image.open(ref['alpha']).convert('L'), dtype=float)/255
        assert alpha.shape == source.shape[:2] == clean.shape[:2]
        for label, record in (('原版', old), ('候选', new)):
            selected = np.array(Image.open(record['files']['笔画选区.png'])) > 0
            output = load_image(record['files']['处理结果.png'])
            row['versions'][label] = score(source, output, selected, clean, alpha) | {
                'seconds': record['elapsed_ms']/1000, 'method': record['method'],
                'repair_mode': record['repair_mode'], 'repairs': record['repairs'], 'exceptions': None}
        a, b = row['versions']['原版'], row['versions']['候选']
        old_selected = np.array(Image.open(old['files']['笔画选区.png'])) > 0
        new_selected = np.array(Image.open(new['files']['笔画选区.png'])) > 0
        old_error = abs(load_image(old['files']['处理结果.png']).astype(float)-clean).mean(2)
        new_error = abs(load_image(new['files']['处理结果.png']).astype(float)-clean).mean(2)
        row['true_pixels_lost_from_old_selection'] = int((old_selected & ~new_selected & (alpha > 0)).sum())
        row['true_pixels_added_to_selection'] = int((~old_selected & new_selected & (alpha > 0)).sum())
        row['watermark_pixels_error_increased_gt3'] = int(((new_error-old_error > 3) & (alpha > 0)).sum())
        limits = json.loads((out/'预登记方案与门槛.json').read_text())['gates']['limits']
        for name, limit in (('recall_all_alpha_gt_zero', limits['recall_drop']),
                            ('recall_alpha_ge_21_255', limits['strong_recall_drop']),
                            ('precision', limits['precision_drop'])):
            if a[name] is not None and b[name] is not None and b[name] < a[name]-limit:
                row['numerical_failures'].append(name+'下降超限')
        for name, limit in (('false_positive_pixels', 0), ('enclosed_hole_false_pixels', 0),
                            ('background_mae', limits['background_mae_increase']),
                            ('watermark_mae', limits['watermark_mae_increase'])):
            if a[name] is not None and b[name] is not None and b[name] > a[name]+limit:
                row['numerical_failures'].append(name+'增加超限')
        row['output_identical'] = old['output_sha256'] == new['output_sha256']
        row['mask_identical'] = old['mask_sha256'] == new['mask_sha256']
        row['route_identical'] = (old['method'], old['repair_mode']) == (new['method'], new['repair_mode'])
        row['timing_failures'] = []
        if new['elapsed_ms'] > max(old['elapsed_ms']*(1+limits['elapsed_increase_ratio']),
                                   old['elapsed_ms']+limits['elapsed_increase_ms']):
            row['timing_failures'].append('相对耗时增加超限')
        if new['elapsed_ms'] > limits['time_budget_ms']:
            row['timing_failures'].append('超过120秒预算')
        if not row['output_identical']:
            if b['outside_selection_changed_pixels']:
                row['numerical_failures'].append('候选选区外有改动')
            if b['dark_content_mae'] is not None and b['dark_content_mae'] > a['dark_content_mae']:
                row['numerical_failures'].append('深色正文/图表平均误差增加')
            if b['dark_content_error_gt3_pixels'] > a['dark_content_error_gt3_pixels']:
                row['numerical_failures'].append('深色内容误差大于3的像素增加')
        if case['id'].endswith('document_032__dot_grid.png'):
            row['target_gates'] = {
                'recall_improved_at_least_15pp': b['recall_all_alpha_gt_zero']-a['recall_all_alpha_gt_zero'] >= .15,
                'precision_at_least_99pct': b['precision'] is not None and b['precision'] >= .99,
                'watermark_mae_reduced_at_least_30pct': b['watermark_mae'] <= a['watermark_mae']*.7}
        stem = hashlib.sha256(key.encode()).hexdigest()[:16]
        pred = np.array(Image.open(new['files']['笔画选区.png'])) > 0
        diagnostic = source.copy()
        diagnostic[(alpha > 0) & ~pred] = [30, 110, 255]
        diagnostic[(alpha == 0) & pred] = [255, 140, 20]
        path = figures/(stem+'_漏选多选核对.png')
        Image.fromarray(diagnostic).save(path)
        row['diagnostic'] = str(path)
        rows.append(row)
    payload = dict(records=rows, independent_validation=False, default_changed=False,
                   reference_loaded_only_after_inference=True,
                   note='开发来源；像素误差不是残留率；时间比较受环境影响，保留超限，不能用均值抵消单图退化。')
    (out/'逐张评分与门槛.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

    def url(path):
        return quote(Path(os.path.relpath(path, out)).as_posix(), safe='/')

    def number(value, percent=False):
        return '未评估' if value is None else f'{value*100:.2f}%' if percent else f'{value:.3f}'

    table, cards = [], []
    for row in rows:
        if row.get('error'):
            table.append('<tr><td>'+escape(row['id'])+'</td><td colspan="7">运行异常，保留记录</td></tr>')
            continue
        a, b = row['versions']['原版'], row['versions']['候选']
        result = '选区、输出和路径相同' if all(row[n] for n in ('output_identical','mask_identical','route_identical')) else '有变化，须逐张复核'
        problems = row['numerical_failures']+row['timing_failures']
        problems += [name+'未达到' for name, passed in row.get('target_gates', {}).items() if not passed]
        table.append('<tr><td>'+escape(Path(row['id']).name)+'</td>'+''.join(
            '<td>'+escape(f'{number(a[n], p)} → {number(b[n], p)}')+'</td>'
            for n, p in (('recall_all_alpha_gt_zero', True), ('precision', True),
                         ('false_positive_pixels', False), ('watermark_mae', False),
                         ('background_mae', False), ('seconds', False)))
            +'<td>'+escape(result+'；'+'；'.join(problems))+'</td></tr>')
        key = row['key']; old, new = versions['原版'][key], versions['候选'][key]
        figures_html = []
        source_case = cases[row['id']]
        source_path = out/'输入快照'/(source_case['input_sha256']+Path(source_case['input']).suffix)
        assert source_path.is_file()
        for title, path in [('原尺寸原图', source_path),
                            ('原版选区', old['files']['选区预览.jpg']),
                            ('候选选区', new['files']['选区预览.jpg']),
                            ('原版实际输出', old['files']['处理结果.png']),
                            ('候选实际输出', new['files']['处理结果.png']),
                            ('事后核对：蓝色漏选、橙色多选', row['diagnostic'])]:
            u = url(path)
            figures_html.append(f'<figure><a href="{u}"><img loading="lazy" src="{u}"></a><figcaption>{title}</figcaption></figure>')
        cards.append('<article><h2>'+escape(Path(row['id']).name)+'</h2><p>'+escape(result+'；'+'；'.join(problems))
                     +'</p><p>'+escape((row.get('manual_review') or {}).get('finding', '尚未填写人工复核'))
                     +'</p><div class="images">'+''.join(figures_html)+'</div><details><summary>完整指标与门槛</summary><pre>'
                     +escape(json.dumps(row, ensure_ascii=False, indent=2))+'</pre></details></article>')
    summary = ('目标文档未达到预登记改善幅度，候选不接入默认流程。'
               + f'新旧各{len(rows)}页；'+str(sum(not r.get('output_identical', True) for r in rows))
               +'页输出改变。数值退化、时间异常和未评估项逐张保留。')
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文档错位点阵专项实验 · 2026-10-29</title><style>body{font:15px/1.8 system-ui;color:#233c4b;background:#f2f5f7;max-width:1500px;margin:30px auto;padding:20px}article,header{background:white;border:1px solid #d7e0e5;padding:22px;margin:20px 0;border-radius:10px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #d7e0e5;text-align:left;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{width:100%;height:340px;object-fit:contain;background:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:800px){.images{grid-template-columns:1fr}table{font-size:12px}}</style><header><h1>文档错位点阵专项实验</h1><p>2026-10-29 · 默认未接入 · 源图推理与参考评分分开</p><p>'''+escape(summary)+'''</p><p>研究进程先保留既有成功周期模型；候选从输入中的小点测量横纵间距、隔行偏移和实际点形，逐点检查局部证据。平坦纸面使用原图邻域插值；不同平坦底色足以确认透明叠加时才反解。正文/表格边缘证据不足的点保留，不调用LaMa。周期位置不是最终掩码。</p><p>完整工程230项通过；固定数据260个文件、43个输入/59页校验通过。全部应用输入适用性扫描另列。未进行完整发布放行及新的独立真实来源验证，不能据此声称全量不退化或通用效果提升。</p><p>全部召回以alpha&gt;0为分母，明显档为alpha≥21/255。深色内容统计包含正文和图表，不能替代逐字正确率。残留率、识别准确率及语义字间空白误选均未评估。</p><p><a href="预登记方案与门槛.json">预登记门槛</a> · <a href="逐张评分与门槛.json">逐张完整指标</a> · <a href="全样式源图适用性扫描_未读取参考.json">全部输入适用性扫描</a> · <a href="工程测试日志.txt">工程测试</a> · <a href="固定数据校验.txt">固定数据</a></p></header><div style="overflow:auto"><table><tr><th>样本</th><th>全部召回</th><th>精确率</th><th>背景误选</th><th>水印区参考误差</th><th>背景误差</th><th>秒</th><th>状态</th></tr>'''+''.join(table)+'</table></div>'+''.join(cards)+'</html>'
    report = out/'文档错位点阵专项实验_新旧实际对照_20261029.html'
    local_findings = '；'.join(
        f"{Path(row['id']).name}：{row['true_pixels_lost_from_old_selection']}个原已选中的真实点像素未被候选保留，"
        f"{row['watermark_pixels_error_increased_gt3']}个水印像素对参考误差增加超过3"
        for row in rows if not row.get('output_identical', True))
    supplements = ('<h2>逐张复核及其他检查的范围</h2><p>'+escape(local_findings)
                   +'。总召回和平均误差改善不能掩盖局部变差；本轮保持未通过，不合并新旧选区补造通过。'
                   +'</p><p><a href="人工复核与结论.json">实图复核记录</a></p>')
    for name, title in [('模式探针对照.json', '关闭补全和CLI的8页次模式探针'),
                        ('默认路径跨样式日常检查/分层检查结果.json', '默认路径日常检查（未启用本候选）')]:
        path = out/name
        if path.exists():
            value = json.loads(path.read_text())
            supplements += '<details><summary>'+escape(title)+'</summary><p><a href="'+url(path)+'">完整记录</a></p><pre>'+escape(json.dumps(value, ensure_ascii=False, indent=2))+'</pre></details>'
    stress_path = out/'冻结后压力探针/负例探针结果.json'
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        supplements += ('<h2>无水印负例暴露的误选风险</h2><p>冻结后另做合法印刷点状底纹的合成负例，候选形成'
                        + str(stress['false_positive_pixels'])
                        +'像素错误选区。它不能区分合法浅色错位底纹与水印点阵，不能默认全自动启用。此项仅检查候选，不冒充相对原版完整流程的新增退化统计，也不是独立真实来源验证。</p>'
                        +'<p><a href="'+url(stress_path)+'">负例记录</a> · <a href="'
                        +url(stress_path.parent/'无水印_合法浅色错位点状底纹.png')+'">实际负例</a> · <a href="'
                        +url(stress_path.parent/'候选错误选区.png')+'">错误选区</a></p>')
    supplements += '<p>模式探针及默认路径日常检查不能代替候选启用后的完整跨样式实际输出对照。候选仅主对照和模式探针实际启用；全样式扫描只验证源图适用条件。候选已未通过开发门槛，因此未执行发布放行、不修改默认基线。</p>'
    body = body.replace('</header>', supplements+'</header>', 1)
    report.write_text(body)
    print(report)


if __name__ == '__main__':
    main()
