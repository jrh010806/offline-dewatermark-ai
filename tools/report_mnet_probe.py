"""Readable local report for the bounded MNet trial; saved PNGs remain native size."""
import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from probe_mnet import NAMES, base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    verification = json.loads((folder / 'verification.json').read_text())
    assert verification['completed'] and verification['cases'] == len(base.IDS)
    rows = json.loads((folder / 'results.json').read_text())
    assert [r['case_id'] for r in rows] == base.IDS
    provenance = json.loads((folder / 'provenance.json').read_text())
    review = json.loads((folder / 'visual_review.json').read_text()) if (folder / 'visual_review.json').exists() else {}
    positives = [r for r in rows if not r['negative'] and r['score_mode'] == 'paired']
    negatives = [r for r in rows if r['negative']]
    aggregates = []
    for key, name in NAMES.items():
        ps = [r['methods'][key] for r in positives]
        ns = [r['methods'][key] for r in negatives]
        aggregates.append(dict(method=key, label=name,
            positive_pass=sum(x['probe_pass'] for x in ps), positive_count=len(ps),
            negative_unchanged=sum(x['probe_pass'] for x in ns), negative_count=len(ns),
            wm_mae=float(np.mean([x['wm_mae'] for x in ps])),
            wm_error_reduction_pct=float(np.mean([x['wm_error_reduction_pct'] for x in ps])),
            outside_mae=float(np.mean([x['outside_mae'] for x in ps])),
            median_seconds=float(np.median([x['seconds'] for x in ps]))))
    (folder / 'aggregate.json').write_text(json.dumps(aggregates, ensure_ascii=False, indent=2))
    fmt = lambda x: '—' if x is None else f'{x:.2f}'
    table = ''.join(f'<tr><td>{r["label"]}</td><td>{r["positive_pass"]}/{r["positive_count"]}</td>'
        f'<td>{r["negative_unchanged"]}/{r["negative_count"]}</td><td>{fmt(r["wm_mae"])}</td>'
        f'<td>{fmt(r["wm_error_reduction_pct"])}%</td><td>{fmt(r["outside_mae"])}</td>'
        f'<td>{fmt(r["median_seconds"])}</td></tr>' for r in aggregates)
    cards = []
    overview = Image.new('RGB', (1280, 330 * len(rows)), '#f1f5f8')
    overview_draw = ImageDraw.Draw(overview)
    for index, row in enumerate(rows):
        cid = row['case_id']
        d = folder / cid
        src = np.array(Image.open(d / 'input.png').convert('RGB'))
        h, w = src.shape[:2]
        # Fixed common crop per case, used only for display after all inference.
        if not row['negative'] and (d / 'truth_mask.png').exists():
            mask = np.array(Image.open(d / 'truth_mask.png')) > 0
            ys, xs = np.where(mask)
            cx, cy = int(np.median(xs)), int(np.median(ys))
        elif row['negative']:
            change = np.maximum.reduce([np.abs(np.array(Image.open(d / (k + '.png')), dtype=np.float32) - src).mean(2)
                                        for k in ['stored_slbr_tiles', 'mnet_global', 'mnet_tiles']])
            heat = cv2.boxFilter(change, -1, (min(400, w), min(280, h)))
            cy, cx = np.unravel_index(heat.argmax(), heat.shape)
        else:
            cx, cy = w // 2, h // 2
        cw, ch = min(480, w), min(340, h)
        x, y = max(0, min(w - cw, cx - cw // 2)), max(0, min(h - ch, cy - ch // 2))
        box = (x, y, x + cw, y + ch)
        keys = ['input', 'stored_slbr_tiles', 'mnet_tiles', 'mnet_global']
        if (d / 'clean.png').exists():
            keys.append('clean')
        contact = Image.new('RGB', (480 * len(keys), 375), '#eef3f6')
        draw = ImageDraw.Draw(contact)
        for i, key in enumerate(keys):
            im = Image.open(d / (key + '.png')).convert('RGB')
            crop = im.crop(box)
            crop.save(d / (key + '_crop.png'))
            draw.text((i * 480 + 12, 10), key, fill='#17364b')
            contact.paste(crop, (i * 480, 35))
        contact.save(d / 'contact.jpg', quality=94)
        overview_draw.text((10, index * 330 + 6), cid, fill='#17364b')
        for i, key in enumerate(['input', 'stored_slbr_tiles', 'mnet_tiles', 'mnet_global']):
            im = Image.open(d / (key + '.png')).convert('RGB')
            im.thumbnail((310, 285))
            overview.paste(im, (i * 320 + (320 - im.width) // 2, index * 330 + 38))
            overview_draw.text((i * 320 + 10, index * 330 + 22), key, fill='#17364b')
        note = html.escape(review.get('cases', {}).get(cid, '目视复核待记录。'))
        metrics = []
        for key, label in NAMES.items():
            m = row['methods'][key]
            status = '无精确真值' if m['probe_pass'] is None else '通过' if m['probe_pass'] else '未通过'
            metrics.append(f'<tr><td>{label}</td><td>{fmt(m.get("wm_mae"))}</td>'
                f'<td>{fmt(m.get("outside_mae"))}</td><td>{fmt(m["seconds"])}</td>'
                f'<td>{m["changed_pixels"]:,}</td><td>{status}</td></tr>')
        ref = (f'<figure><a href="{cid}/clean.png" target="_blank"><img data-key="clean" src="{cid}/clean.png" alt="干净参考"></a><figcaption>干净参考（仅评分用）</figcaption></figure>'
               if (d / 'clean.png').exists() else '')
        kind = '无水印对照：检查正常内容是否被误改' if row['negative'] else '真实水印：没有精确干净参考' if row['score_mode'] != 'paired' else '有干净参考：检查去除与内容保留'
        cards.append(f'''<article id="{cid}"><h2>{cid} · {html.escape(row['label'])}</h2>
<p class="muted">{kind} · {w} × {h} 像素</p><p class="note">{note}</p>
<div class="controls"><label>MNet 方式 <select onchange="method(this,'{cid}')"><option value="mnet_tiles">保持原尺寸分块</option><option value="mnet_global">整图缩小后处理</option></select></label>
<label><input type="checkbox" onchange="zoom(this,'{cid}')"> 看同位置局部放大</label><a href="{cid}/contact.jpg" target="_blank">打开局部对照图</a></div>
<div class="images" data-case="{cid}"><figure><a href="{cid}/input.png" target="_blank"><img data-key="input" src="{cid}/input.png" alt="原图"></a><figcaption>原图</figcaption></figure>
<figure><a href="{cid}/stored_slbr_tiles.png" target="_blank"><img data-key="stored_slbr_tiles" src="{cid}/stored_slbr_tiles.png" alt="SLBR"></a><figcaption>SLBR（历史保存结果）</figcaption></figure>
<figure><a href="{cid}/mnet_tiles.png" target="_blank"><img data-key="mnet" src="{cid}/mnet_tiles.png" alt="MNet"></a><figcaption>MNet（本次实跑）</figcaption></figure>{ref}</div>
<details><summary>查看数值和定位区域</summary><p><a href="{cid}/mnet_tiles_probability.png" target="_blank">MNet 分块预测区域（越白越确信是水印）</a> · <a href="{cid}/mnet_global_probability.png" target="_blank">MNet 整图预测区域</a> · <a href="{cid}/stored_current.png" target="_blank">原有流程历史结果</a></p><div class="scroll"><table><tr><th>方法</th><th>水印区误差 ↓</th><th>区外误差 ↓</th><th>秒</th><th>改动像素数</th><th>实验门槛</th></tr>{''.join(metrics)}</table></div></details></article>''')
    overview.save(folder / 'overview.jpg', quality=92)
    summary = ''.join(f'<li>{html.escape(s)}</li>' for s in review.get('summary', ['实跑完成，目视复核待记录。']))
    source_changes = verification.get('concurrent_source_changes', [])
    concurrent_note = ('<p class="muted">运行期间检测到其他工作修改了 ' + html.escape('、'.join(source_changes))
        + '；本次 MNet 推理不调用这些应用入口，主程序对照使用历史保存输出，不能据此判断最新主程序的表现。</p>') if source_changes else ''
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MNet · 8 张样本试跑</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#20394a;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1500px;margin:auto;padding:32px 24px}header{padding:30px;background:#173c4a;color:#fff;border-radius:14px}h1{font-size:32px;line-height:1.4;margin:8px 0}h2{font-size:23px;line-height:1.5;margin:0 0 10px}.eyebrow{font-size:13px;color:#a8dfd2;letter-spacing:2px}article,.panel{padding:26px;background:#fff;border:1px solid #d8e2e8;border-radius:12px;margin:22px 0}.muted{font-size:14px;color:#637582}.note{padding:12px 16px;background:#fff5df;border-left:4px solid #d49b2e}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.images figure{margin:0}.images img{width:100%;height:390px;object-fit:contain;background:#f0f4f6}figcaption{font-weight:650;margin-top:8px}a{color:#087472}header a{color:#bceee1}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:20px;margin:18px 0}select{font:inherit;padding:8px;border:1px solid #bdccd5;border-radius:5px}summary{cursor:pointer;font-weight:600}details{margin-top:20px}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}th,td{padding:10px;text-align:left;border-bottom:1px solid #dce5ea}th{background:#edf3f6}.scroll{overflow:auto}code{font-size:13px;background:#edf3f6;padding:2px 5px}li{margin:8px 0}@media(max-width:1000px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){main{padding:14px}header,article,.panel{padding:18px}.images{grid-template-columns:1fr}.images img{height:360px}h1{font-size:26px}}
</style></head><body><main><header><div class="eyebrow">LOCAL MNET TRIAL · 2026-10-27</div><h1>MNet 这次实际去得怎么样？</h1><p>同一组 8 张：5 张有精确参考的带水印图、2 张无水印对照、1 张真实醒图导出。固定官方 LOGO-H 权重，一次试跑，无训练、无样本调参。</p></header>'''
    page += f'<section class="panel"><h2>看这几条结论就够了</h2><ul>{summary}</ul><p><a href="overview.jpg" target="_blank">打开全部样本总览</a> · <a href="results.json">逐图数值</a> · <a href="provenance.json">模型与运行记录</a> · <a href="verification.json">完整性检查</a></p></section>'
    page += f'<section class="panel"><h2>同样本数值对比</h2><p class="muted">误差越低越好；误差降低率越高越好。只统计 5 张配对正例。无水印对照要求逐像素不变。结果均为原尺寸保存；整图模式缩小到 256 后映回，并不代表保住了原图细节。历史方法的耗时仅供参考，本次未重跑。</p><div class="scroll"><table><tr><th>方法</th><th>正例门槛</th><th>负例保持原样</th><th>水印区误差 ↓</th><th>误差降低 ↑</th><th>区外误差 ↓</th><th>中位秒</th></tr>{table}</table></div></section>'
    page += ''.join(cards)
    page += f'''<section class="panel"><h2>这次测试的边界</h2><p>推理只读带水印原图。干净参考、真实水印区域只在输出完成后用于评分与展示。原有流程与 SLBR 使用此前保存的同输入结果，输入哈希已核对，并用同一评分函数重算。样本少且照片底图有重复，不能外推总体成功率。</p><p>官方 MNetold 源码未改；严格匹配 {provenance['model_info']['state_keys']} 个权重键。两种预设适配：整图缩小到 256×256；原尺寸分块 256×256、步长192、羽化融合。均使用模型预测的软掩码混合结果。当前使用 Apple MPS、4 个 CPU 线程，未证明 NVIDIA 工作站速度。单独模型加载 {provenance['model_load_seconds']:.2f} 秒、预热 {provenance['warmup_seconds']:.2f} 秒，不计入表中推理耗时。</p><p>实验门槛沿用此前小样本诊断：水印区平均误差≤3、区内误差&gt;10的像素比例≤10%、区外平均误差≤1、区外P99≤4、笔画IoU≥0.90、耗时≤120秒。笔画IoU不能证明数字或文字正确，结论还结合目视复核。</p>{concurrent_note}<p><a href="https://github.com/Aitchson-Hwang/MNet">官方代码与权重入口</a> · <a href="https://doi.org/10.1016/j.neunet.2024.106961">论文</a> · <a href="runner_snapshot.py">本次推理脚本快照</a> · <a href="aggregate.json">汇总数据</a> · <a href="device_parity.json">CPU/MPS 输出像素一致性检查</a></p></section></main>
<script>const methods={{}},cropped={{}};function render(id){{document.querySelector('[data-case="'+id+'"]').querySelectorAll('img').forEach(im=>{{let key=im.dataset.key==='mnet'?(methods[id]||'mnet_tiles'):im.dataset.key;im.src=id+'/'+key+(cropped[id]?'_crop':'')+'.png';im.parentElement.href=id+'/'+key+'.png'}})}}function method(el,id){{methods[id]=el.value;render(id)}}function zoom(el,id){{cropped[id]=el.checked;render(id)}}</script></body></html>'''
    (folder / 'report.html').write_text(page)
    print(folder / 'report.html')


if __name__ == '__main__':
    main()
