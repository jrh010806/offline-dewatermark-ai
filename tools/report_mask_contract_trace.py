"""Score completed source-only trace runs and publish inspectable local evidence."""
from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(Path(path).read_text())


def rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def binary(path):
    return np.array(Image.open(path).convert('L')) > 0


def existing_boundary_note(root):
    path = root/'其他模式原有范围越界核对.json'
    if not path.exists():
        return ''
    data = read(path)
    return ('<section><h2>另行发现的原有范围不一致（尚未修复）</h2>'
            f'<p>关闭补全和CLI中的document_032：阈值选区{data["selected_pixels"]:,}像素，'
            f'footprint与实际模型输入{data["actual_lama_input_pixels"]:,}像素，'
            f'阈值选区外实际修改{data["actual_changed_outside_threshold_selection"]:,}像素。'
            '原版与候选相同；不能因此认为范围安全。这与目标图在补全阶段丢笔画是不同问题，未在本轮接入默认修复。</p>'
            '<p><a href="其他模式原有范围越界核对.json">最终核对与原因</a> · '
            '<a href="原版追踪/desktop_protected/逐阶段_document_032__heavy_dark/逐阶段记录.json">原始阶段记录与实际模型张量索引</a></p></section>')


def quality(selected, alpha, holes):
    truth, strong = alpha > 0, alpha >= .08
    tp = int((selected & truth).sum())
    return dict(selected_pixels=int(selected.sum()), fraction=float(selected.mean()),
                recall=tp/int(truth.sum()) if truth.any() else None,
                strong_recall=float(selected[strong].mean()) if strong.any() else None,
                precision=tp/int(selected.sum()) if selected.any() else None,
                missed_pixels=int((truth & ~selected).sum()),
                false_positive_pixels=int((selected & ~truth).sum()),
                hole_selected_pixels=int((selected & holes).sum()))


def main():
    ap = argparse.ArgumentParser(description='推理后独立评分与逐阶段证据报告')
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    report = root / '选区逐阶段追踪与修复候选对照_20261029.html'
    if report.exists():
        ap.error('报告已存在，禁止覆盖历史轮次')
    inventory = {r['id']: r for r in read(root/'评分清单_不进入推理.json')}
    folders = sorted(root.glob('*追踪/*/task.json'))
    # Completeness is checked before opening any reference or alpha image.
    for task_path in folders:
        task = read(task_path)
        records = read(task_path.parent/'inference.json')
        if len(records) != len(task['inputs']) or any(r['status'] != 'completed' for r in records):
            raise ValueError(f'推理不完整，不能评分：{task_path}')
        if {r['id'] for r in records} != {r['id'] for r in task['inputs']}:
            raise ValueError('样本不完整')
    records, sections = [], []

    def href(path):
        return quote(Path(os.path.relpath(path, root)).as_posix(), safe='/')

    def picture(path, caption):
        return (f'<figure><a href="{href(path)}"><img loading="lazy" src="{href(path)}" '
                f'alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>')

    for task_path in folders:
        mode, variant = task_path.parent.name, task_path.parent.parent.name
        trace_index = {r['id']: r for r in read(task_path.parent/'逐阶段记录索引.json')}
        for inferred in read(task_path.parent/'inference.json'):
            item = inventory[inferred['id']]
            trace = trace_index[item['id']]
            dest = Path(trace['directory']).resolve()
            source = rgb(dest/'00_实际输入.png')
            output = rgb(dest/'07_实际处理结果.png')
            ref = item['references'][0]
            clean = rgb(ref['clean'])
            alpha = np.array(Image.open(ref['alpha']).convert('L'), np.float32)/255
            if clean.shape != source.shape or alpha.shape != source.shape[:2]:
                raise ValueError('几何不一致，禁止伪造配准成绩')
            truth = alpha > 0
            n, labels = cv2.connectedComponents((~truth).astype(np.uint8), connectivity=8)
            internal = np.ones(n, bool)
            internal[np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = False
            internal[0] = False
            holes = internal[labels]
            stages, previous = [], None
            for s in trace['stages']:
                selected = binary(s['selection'])
                q = quality(selected, alpha, holes)
                row = dict(s, **q)
                if previous is not None:
                    lost, added = previous & ~selected, selected & ~previous
                    row['transition'] = dict(lost_pixels=int(lost.sum()), added_pixels=int(added.sum()),
                        lost_true_pixels=int((lost & truth).sum()), lost_background_pixels=int((lost & ~truth).sum()),
                        added_true_pixels=int((added & truth).sum()), added_background_pixels=int((added & ~truth).sum()))
                stages.append(row)
                previous = selected
            for call in trace['lama_calls']:
                path = dest/f'06_LaMa调用{call["index"]}_模型实际输入选区.png'
                actual = binary(path)
                lost, added = previous & ~actual, actual & ~previous
                stages.append(dict(name=f'05_LaMa实际输入选区_调用{call["index"]}',
                    selection=str(path), method='实际ONNX张量映回原图', repair_mode='实际模型调用',
                    **quality(actual, alpha, holes), transition=dict(lost_pixels=int(lost.sum()),
                    added_pixels=int(added.sum()), lost_true_pixels=int((lost & truth).sum()),
                    lost_background_pixels=int((lost & ~truth).sum()),
                    added_true_pixels=int((added & truth).sum()), added_background_pixels=int((added & ~truth).sum()))))
                previous = actual
            selected = binary(dest/'04_修复入口选区.png')
            error = np.abs(output.astype(float)-clean).mean(2)
            changed = np.any(source != output, axis=2)
            row = dict(id=item['id'], variant=variant, mode=mode, **quality(selected, alpha, holes),
                stages=stages, changed_pixels=int(changed.sum()), outside_selection_changed=int((changed & ~selected).sum()),
                watermark_mae=float(error[truth].mean()) if truth.any() else None,
                background_mae=float(error[~truth].mean()),
                dark_reference_content_turned_bright=int((truth & (clean.mean(2)<100) & (output.mean(2)>200)).sum()),
                elapsed_ms=inferred['elapsed_ms'], trace_repair_elapsed_ms=trace['repair_elapsed_ms'],
                method=inferred['method'], repair_mode=inferred['repair_mode'], repairs=inferred['repairs'],
                mask_sha256=inferred['mask_sha256'], output_sha256=inferred['output_sha256'],
                channel_rule=trace.get('channel_rule'), channels=trace['channels'], lama_calls=trace['lama_calls'],
                strategy=trace.get('strategy'),
                residual_quality=None, recognition_quality=None, whitespace_quality=None,
                source_group=item['source_group'], split_group=item['split_group'], base_id=item['base_id'],
                independent_validation=False, exception=None, directory=str(dest))
            records.append(row)
            marker = source.astype(float)
            marker[truth & ~selected] = marker[truth & ~selected]*.25+np.array([0, 130, 255])*.75
            marker[selected & ~truth] = marker[selected & ~truth]*.25+np.array([255, 150, 0])*.75
            Image.fromarray(np.uint8(marker)).save(dest/'评分后_漏选蓝色多选橙色.png')
            removed = binary(dest/'01_初始检测选区.png') & ~binary(dest/'02_补全后选区.png')
            marker = source.copy()
            marker[removed & truth] = (0, 130, 255)
            marker[removed & ~truth] = (255, 150, 0)
            Image.fromarray(marker).save(dest/'评分后_补全丢失真实笔画蓝色背景橙色.png')
            if Path(item['input']).stem == 'charts_015__heavy_dark' and mode == 'desktop':
                gallery = picture(dest/'00_实际输入.png', '实际输入')
                for s in stages:
                    gallery += picture(s['selection'], s['name']+'（真实捕获）')
                for call in trace['lama_calls']:
                    gallery += picture(dest/f'06_LaMa调用{call["index"]}_模型实际输入选区.png', '实际模型张量选区映回原图坐标')
                policy = trace.get('strategy')
                if policy:
                    gallery += picture(policy['risk_file'], '策略暂缓区域；保留检测，不代表确定不透明或真实标注')
                gallery += picture(dest/'08_实际修改像素.png', '实际修改像素；不是检测选区')
                gallery += picture(dest/'07_实际处理结果.png', '实际处理结果')
                gallery += picture(dest/'评分后_补全丢失真实笔画蓝色背景橙色.png', '事后标注核对：补全丢失的真实笔画蓝色、背景橙色')
                gallery += picture(dest/'评分后_漏选蓝色多选橙色.png', '事后标注核对：最终漏选蓝色、多选橙色')
                table = ''.join('<tr><td>'+escape(s['name'])+'</td>'+''.join(
                    f'<td>{s[k]:.2%}</td>' if k in ('recall','precision') and s[k] is not None
                    else f'<td>{"未评估" if s[k] is None else s[k]}</td>' for k in ('selected_pixels','recall','precision','missed_pixels','false_positive_pixels','hole_selected_pixels'))+'</tr>' for s in stages)
                status = ('<p><strong>本页未自动修复：没有调用LaMa，没有实际模型输入图。</strong>完整检测选区保留，输出与原图相同。'
                          +escape(policy['reason'])+'</p>' if policy and policy['abstained'] else '')
                sections.append(f'<section><h2>{escape(variant+" / "+mode)} · 目标图逐阶段</h2>'+status+
                    '<div class="scroll"><table><tr><th>阶段</th><th>选中</th><th>全部召回</th><th>精确率</th><th>漏选</th><th>背景多选</th><th>字洞误选</th></tr>'
                    +table+'</table></div><div class="gallery">'+gallery+'</div><details><summary>实际通道分数与阶段记录</summary><pre>'
                    +escape(json.dumps(dict(channels=row['channels'],transitions=[s.get('transition') for s in stages],lama_calls=row['lama_calls']),ensure_ascii=False,indent=2))+'</pre></details></section>')

    comparisons = []
    lookup = {(r['variant'],r['mode'],r['id']):r for r in records}
    limits = read(root/'预登记.json')['acceptance']['limits']
    for old in records:
        if old['variant'] != '原版追踪':
            continue
        for variant in sorted({r['variant'] for r in records if r['variant'].startswith('候选')}):
            new = lookup.get((variant,old['mode'],old['id']))
            if new is None:
                continue
            failures=[]
            for key, limit in [('recall','recall_drop'),('strong_recall','strong_recall_drop'),('precision','precision_drop')]:
                if old[key] is not None and new[key] is not None and old[key]-new[key] > limits[limit]:
                    failures.append(key+'下降')
            for key, limit in [('false_positive_pixels','false_positive_increase_pixels'),('hole_selected_pixels','hole_selected_increase_pixels'),('watermark_mae','watermark_mae_increase'),('background_mae','background_mae_increase')]:
                if new[key]-old[key] > limits[limit]:
                    failures.append(key+'增加')
            if new['outside_selection_changed']:
                failures.append('外扩0时选区外修改')
            if new['elapsed_ms']>limits['time_budget_ms'] or (new['elapsed_ms']>old['elapsed_ms']*(1+limits['elapsed_increase_ratio']) and new['elapsed_ms']-old['elapsed_ms']>limits['elapsed_increase_ms']):
                failures.append('耗时超限')
            comparisons.append(dict(id=old['id'],mode=old['mode'],variant=variant,before=old,after=new,failures=failures,
                output_identical=old['output_sha256']==new['output_sha256'],mask_identical=old['mask_sha256']==new['mask_sha256'],
                note='逐张数值比较；变化仍需目视复核，无独立验证不具备默认放行资格'))

    rows=''
    for c in comparisons:
        a,b=c['before'],c['after']
        label = escape(Path(c['id']).stem)
        if c['mode']=='desktop':
            folder = '核心候选逐张实际对照' if c['variant']=='候选追踪' else '边缘与正文策略逐张实际对照'
            sheet = root/folder/(Path(c['id']).stem+'_实际对照.jpg')
            if sheet.exists():
                label = f'<a href="{href(sheet)}">{label} · 实际对照</a>'
        rows+=f'<tr><td>{label}</td><td>{c["mode"]}<br>{escape(c["variant"])}</td>'
        for key in ('recall','precision'):
            rows+=f'<td>{a[key]:.2%} → {b[key]:.2%}</td>' if a[key] is not None and b[key] is not None else '<td>未评估</td>'
        for key in ('false_positive_pixels','hole_selected_pixels','watermark_mae','background_mae','elapsed_ms'):
            rows+=f'<td>{a[key]:.3f} → {b[key]:.3f}</td>'
        rows+=f'<td>{escape("；".join(c["failures"]) or ("选区与输出相同" if c["output_identical"] and c["mask_identical"] else "数值门槛内，变化待复核"))}</td></tr>'
    summary=dict(records=records, comparisons=comparisons, thresholds=limits,
        independent_validation=False, this_task_changed_default=False, release_passed=False,
        residual_quality=None, recognition_quality=None,
        note='同一开发批次与历史回归来源，不构成独立泛化验证。像素误差不是残留率。')
    issue = root/'其他模式原有范围越界核对.json'
    if issue.exists():
        summary['existing_boundary_issue'] = read(issue)
    (root/'逐张评分与阶段差异_20261029.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    target = next(r for r in records if r['variant']=='原版追踪' and r['mode']=='desktop'
                  and Path(r['id']).stem=='charts_015__heavy_dark')
    a, b = target['stages'][:2]
    lead = (f'<section><h2>已确认的丢失环节</h2><p>目标图初始选区{a["selected_pixels"]:,}像素，'
            f'全部水印召回{a["recall"]:.2%}；补全后{b["selected_pixels"]:,}像素，召回{b["recall"]:.2%}。'
            f'补全丢失真实水印{b["transition"]["lost_true_pixels"]:,}像素，并新增背景误选'
            f'{b["transition"]["added_background_pixels"]:,}像素。在这次原版运行中，后续先验策略、修复入口及模型实际输入没有再删像素。</p>'
            f'<p>原版目标图实际修改{target["changed_pixels"]:,}像素，选区外修改{target["outside_selection_changed"]}像素；'
            '实际修改像素另存，不把它当作检测选区或真实水印标注。</p>'
            '<p>现场通道记录显示：亮度暗笔画得分约189，颜色通道约15；代码只要找到合格的正向颜色候选，就优先选择颜色通道，'
            '最终用新region替换初始选区。这是本图已确认的原因，不推断所有失败图都由同一原因造成。</p></section>')
    for c in comparisons:
        if c['mode']=='desktop' and Path(c['id']).stem=='charts_015__heavy_dark':
            b=c['after']
            lead += (f'<p><strong>{escape(c["variant"])}</strong>：检测选区召回{b["recall"]:.2%}、精确率{b["precision"]:.2%}，'
                     f'背景多选{b["false_positive_pixels"]:,}，水印区参考误差{b["watermark_mae"]:.3f}。'
                     f'门槛：{escape("；".join(c["failures"]) or "数值门槛内，仍需目视与独立验证")}。</p>')
            if b.get('strategy',{}):
                lead += '<p><strong>正文策略仅暂停生成，保留原图与检测；此页未自动去水印。</strong>原图OCR未找到被遮挡的198、306，不能声称已经准确划出不可恢复像素。</p>'
    extra = '<p><a href="工程与跨样式核对_20261029.html">工程测试、候选跨样式探针及日常检查的实际结论</a></p>'
    lead += existing_boundary_note(root)
    report.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>选区逐阶段追踪与修复候选对照 · 2026-10-29</title><style>body{font:15px/1.8 system-ui;margin:30px auto;padding:20px;max-width:1450px;background:#f5f7fa;color:#23364c}section{background:white;padding:22px;margin:25px 0;border-radius:14px}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}figure{margin:0}img{width:100%;display:block}figcaption{padding:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#1467a8}.warn{background:#fff1db;padding:18px}</style>
<h1>选区逐阶段追踪与修复候选对照</h1><p>2026-10-29 · 深色图表大字失败图及原批次全部9张 · 外扩0 · 当前冻结代码。</p>'''+lead+extra+'''
<p class="warn"><strong>本轮实验候选未接入默认算法；不具备默认放行结论。</strong>本批已用于分析，独立验证不成立；残留率、正文识别质量与字间空白语义保持未评估。其他任务同期对工作目录的默认改动不计入本轮图像验收。</p>
<p>以下阶段图来自实际运行捕获；漏选／多选彩色图在推理结束后读取标注生成，明确为事后核对图。搜索范围仅作观测，不作为修复许可。“策略后”对应真实with_prior返回，当前应用没有单独统一的策略裁剪模块。实际模型输入记录了传给ONNX的张量及图块坐标，重叠图块汇总不重复计数。</p>
<p>三模式均保留实际文件，下面展开桌面补全开启的逐阶段图像；其他模式的完整数值和文件目录见JSON。所有图像对照沿用各自早先冻结代码。运行期间其他任务修改了工作目录，后来的代码不计入这轮图像验收；较晚的工程测试使用另一个完整冻结快照。</p>
<p>全部召回分母为alpha&gt;0的全部标注像素；明显笔画为alpha≥0.08（8位标注相当于≥21/255）。字洞指标只统计标注封闭孔洞，不代表字间空白检查。耗时含捕获开销，不能当作默认应用速度。</p>
<p><a href="预登记.json">原版冻结、模型与预定门槛</a> · <a href="候选方案预登记_评分前.json">核心候选方案</a> · <a href="边缘候选方案预登记_评分前.json">边缘候选与继续开发披露</a> · <a href="正文策略方案预登记_20261029.json">正文策略与未知风险</a> · <a href="逐张评分与阶段差异_20261029.json">完整逐张与阶段指标</a></p>
<p><a href="核心候选逐张目视核对_20261029.json">核心候选逐张目视记录</a> · <a href="边缘与正文策略逐张目视核对_20261029.json">边缘与正文策略逐张目视记录</a> · <a href="运行期间其他任务工作目录变更记录.json">其他任务并行修改的版本边界</a></p>
<section><h2>原批次逐张新旧比较</h2><div class="scroll"><table><tr><th>样本</th><th>模式</th><th>全部召回</th><th>精确率</th><th>背景多选</th><th>字洞多选</th><th>水印区像素误差</th><th>背景像素误差</th><th>总耗时ms</th><th>门槛与差异</th></tr>'''+rows+'</table></div></section>'+''.join(sections)+'</html>')
    print(report)


if __name__ == '__main__':
    main()
