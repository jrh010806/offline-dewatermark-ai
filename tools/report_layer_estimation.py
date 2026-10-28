"""Render the frozen layer-estimation diagnostic without rerunning inference."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/layer_estimation/20261027_fixed'
rows = json.loads((OUT / 'results.json').read_text())
prior = {r['case_id']: r for r in json.loads((ROOT / 'out/strategy_probe/20261027_small8_mps/combined_results.json').read_text())}
notes = {
    'core_001': '通过。水印显著减弱，横梁与暗部细节保留；仍有轻微色差。',
    'core_010': '通过。密集淡字明显减弱，墙面与窗框保留；仍有淡痕。这两张通过样本来自同一底图，不能代表普遍有效。',
    'core_015': '失败。大字留下清楚的轮廓和局部色差；拟合检查却予以放行，说明拟合稳定不等于恢复可靠。',
    'core_020': '拒绝处理。候选图产生紫绿色残影；最终输出保留原图。不能把未获采用的候选图误差下降算作成功。',
    'core_016': '拒绝处理。拟合透射率触及下界，最终保留原图；没有恢复完全遮住的信息。',
    'core_023': '非空误检掩膜触发拒绝，输出与输入逐字节一致；空掩膜对照也一致。',
    'core_024': '非空误检掩膜因有效内部像素不足而拒绝，输出与输入逐字节一致；空掩膜对照也一致。',
    'real_003': '未通过视觉验收。白色水印变淡，但残边明显，局部正文笔画变粗、变黑。此图只有近似掩膜，没有干净参考，不计算恢复成功率。',
}
primary = [r for r in rows if 'meets_continue_gate' in r]
table = []
mdtable = ['|样本|SLBR 同位置 MAE|本次最终 MAE|笔画 IoU|结论|', '|---|---:|---:|---:|---|']
for r in primary:
    cid = r['case_id']; a = r['methods']['guarded']; b = r['methods']['slbr_same_mask']
    verdict = '通过' if r['meets_continue_gate'] else ('拒绝，保留原图' if not r['fit']['accepted'] else '失败')
    vals = [r['label'], f"{b['wm_mae']:.2f}", f"{a['wm_mae']:.2f}", f"{a['ink_iou']:.3f}", verdict]
    table.append('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in vals) + '</tr>')
    mdtable.append('|' + '|'.join(vals) + '|')

summary = '''# 已知位置下的水印层估计：固定小样本实验

结论：4 张主测试中 2 张通过，未达到预先冻结的 4/4 门槛。停止这个实现进入自动定位和产品接入；没有扩大样本、训练模型或事后调参。

这次只验证一个问题：提供精确二值位置，但不提供真实透明度和颜色，能否从原图估计水印层并保留底图？估计器使用输入图、二值位置和上轮缓存的 LaMa 背景近似图，拟合统一颜色和内部透明度，再做代数反演。干净图、真实透明度、真实水印颜色只用于事后评分与归因，不进入估计器。

## 实测结果

MAE 为水印区域 RGB 平均绝对误差（0–255，越小越好）。所有主测试都得到精确位置，SLBR 也被限制在同一位置，避免比较条件不一致。这里的通过率不是自动去水印成功率。

TABLE

局部淡字和密集淡字的误差相对同位置 SLBR 分别下降约 57% 和 51%，局部视觉检查也显示改善。这两张来自同一底图，不能推广到其他照片。

深色大字的拟合颜色与平均透明度接近真实值，但实际内部透明度并不恒定：真实内部 alpha 的 P10–P90 为 0.831–0.902，统一估计为 0.856。透射率只有 0.144，反演会放大约 7 倍的像素误差。事后分区统计显示 84% 的总误差在字内部，而非仅在边缘。视觉上残留清楚的轮廓。背景近似误差与逐像素透明度误差尚未被单独消融，因此这是有证据支持的失败机制，不是已证明的唯一原因。

带阴影彩字同时包含彩色字与黑色阴影，颜色和透明度变化很大，单一共享参数无法解释。拟合残差触发拒绝；候选图 MAE 虽为 39.38，但有明显紫绿色残影，最终保留原图，MAE 为 87.55。

两个无水印样本在提供旧模型非空误检位置时均保留原图；完全不透明块也被拒绝。这只说明本次三个控制样本表现符合预期，不证明普遍不会误删。

真实文档仅有 SLBR 近似位置。水印变淡，但仍有残边和正文局部加粗，视觉验收失败；无干净参考，不给成功率。

## 决策与边界

不继续改自动定位或 UI：即便位置已知，困难样本的恢复仍未达标，解决检测不能消除这些失败。现有拟合残差和参数稳定性还会放行深色字的坏结果，不能作为成品质量保证。

若继续同一方向，唯一待验证的问题是“空间变化的透明度/颜色，能否在已知位置下可靠估计”，并且必须同时保住底图文字。此轮没有实现它，也没有证明更复杂的分层模型一定有效。当前证据支持淡色、近似单色水印上的局部可行性，不支持通用自动去水印。

## 验收与复现

测试前冻结 protocol.json。门槛包括 MAE≤3、区域大误差比例≤10%、区域外保真、笔画 IoU≥0.90，且相对同位置 SLBR 至少改善10%；两个负样本需保持不变。笔画 IoU 不是 OCR 正确率，视觉损伤仍可否决。

本轮复用上轮由输入图和二值位置生成的 LaMa 背景近似结果，没有重新运行 LaMa。results.json 的 seconds 是估计器用时，连对照条目也复用了该值，不能用来比较方法速度或解释为端到端时间。报告另列历史背景生成用时和本次估计用时，二者之和仅为估算。

独立理想混合数组验证：真实 alpha=0.4，估计0.40013，恢复 MAE=0.394；空掩膜恒等、平坦背景拒绝、区域外不变均通过。它验证数值实现，不代替真实样本质量验收。

复现：将冻结 protocol.json 放入新的输出目录，运行 `.venv/bin/python tools/probe_layer_estimation.py <新目录>`。依赖同仓库固定数据集及上轮缓存；不能覆盖已有结果。runner_snapshot.py 与 provenance.json 保留运行脚本和输入摘要。生产 wm/ 源码未改动。
'''.replace('TABLE', '\n'.join(mdtable))
(OUT / 'summary.md').write_text(summary)

cards = []
timings = []
for r in rows:
    cid = r['case_id']; fit = r['fit']; a = r['methods']['guarded']
    proxy_method = 'slbr_mask_lama' if r['negative'] or r['score_mode'] != 'paired' else 'known_mask_lama'
    old_sec = prior[cid]['methods'][proxy_method]['seconds']
    timings.append({'case_id': cid, 'cached_proxy_historical_seconds': old_sec, 'new_estimation_seconds': r['estimation_seconds'], 'estimated_sum_seconds': old_sec + r['estimation_seconds'], 'end_to_end_measured_this_run': False})
    links = [('input', '输入'), ('slbr_same_mask', 'SLBR 同位置'), ('candidate', '未经拒绝过滤的候选'), ('guarded', '最终输出'), ('background_proxy', '背景近似'), ('support', '二值位置')]
    if (OUT/cid/'clean.png').exists(): links.append(('clean', '干净参考（仅评分）'))
    linkhtml = ' · '.join(f'<a href="{cid}/{n}.png" target="_blank">{label}</a>' for n,label in links)
    pic = f'<a href="{cid}/comparison.jpg" target="_blank"><img loading="lazy" src="{cid}/comparison.jpg" alt="{cid} 局部对比"></a>' if (OUT/cid/'comparison.jpg').exists() else ''
    diagnostic = {'fit': fit, 'posthoc_only': r.get('posthoc_diagnostics'), 'final_scores': a, 'timing': timings[-1]}
    cards.append(f'<article id="{cid}"><h2>{cid} · {html.escape(r["label"])}</h2><p>{notes[cid]}</p><p>{linkhtml}</p>{pic}<details><summary>拟合诊断、最终评分与用时说明</summary><pre>{html.escape(json.dumps(diagnostic,ensure_ascii=False,indent=2))}</pre></details></article>')
(OUT/'timing_context.json').write_text(json.dumps(timings,indent=2))
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水印层估计 · 固定样本实验</title><style>
body{margin:0;background:#f1f4f6;color:#172a37;font:16px/1.75 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:36px 22px}h1{font-size:32px;line-height:1.3}h2{font-size:23px}p{max-width:1000px}.decision{background:#fff0d5;border-left:5px solid #b66b00;padding:18px 24px}article,.panel{margin:26px 0;background:white;border:1px solid #dce3e7;border-radius:12px;padding:24px}img{width:100%;height:auto;margin:12px 0}a{color:#086a8b}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #dce3e7;padding:12px}pre{white-space:pre-wrap;font-size:13px;overflow-wrap:anywhere}nav{display:flex;gap:16px;flex-wrap:wrap}.muted{color:#5c6f7b}details{margin-top:16px}summary{cursor:pointer}code{background:#edf2f5;padding:2px 5px}@media(max-width:600px){main{padding:18px 12px}article,.panel{padding:15px}td,th{padding:7px;font-size:13px}}
</style><main><p class="muted">2026-10-27 · 已知二值位置 · 单一冻结配置 · 无事后调参</p><h1>淡字有改善，困难样本仍失败</h1><div class="decision"><strong>决定：2/4 通过，停止此实现进入自动定位与产品接入。</strong><br>这次只验证恢复，不把精确位置下的结果包装成自动去水印成功率。</div>
<section class="panel"><h2>固定条件下的比较</h2><p>输入图 + 二值位置 + 输入生成的 LaMa 背景近似 → 估计统一颜色与内部透明度 → 反演。真实 alpha、颜色和干净图仅在估计完成后用于评分、归因。SLBR 同样限制在给定位置。</p><table><thead><tr><th>主测试</th><th>SLBR MAE ↓</th><th>最终 MAE ↓</th><th>笔画 IoU ↑</th><th>结果</th></tr></thead><tbody>TABLE</tbody></table><p class="muted">MAE 范围0–255；笔画 IoU 不是 OCR 正确率。两张通过图片共用底图。彩字候选误差39.38，但被拒绝，最终原图误差87.55。</p></section>
<section class="panel"><h2>为什么到这里停止</h2><p><b>深色字：</b>平均参数接近真实值仍有清楚残边。实际内部透明度变化，低透射率让反演放大约7倍误差；84%的总误差位于字内部。拟合稳定性检查仍放行了这个坏结果，不能据此判定成品可靠。背景近似与逐像素参数误差尚未单独消融。</p><p><b>阴影彩字：</b>颜色与透明度不统一，拟合无法解释，候选出现彩色残影，触发拒绝。<b>真实文档：</b>水印减淡但仍有残边、局部笔画变粗，视觉验收失败。</p><p>位置已知仍不合格，所以现在改定位或界面不能解决核心问题。若继续分层方向，下一项只应验证空间变化的透明度/颜色估计；本轮没有展开开发，也不承诺该方案必然有效。</p></section>
<nav>NAV</nav><p class="muted">对比图：input 输入；slbr_same_mask 同位置基线；candidate 未经拒绝过滤的候选；guarded 最终输出；clean 仅供评分的参考。点击查看大图。</p>CARDS
<section class="panel"><h2>控制、证据与复现</h2><p>2个无水印样本在非空误检位置下均保持原图；不透明块被拒绝。独立理想混合数组恢复 alpha=0.40013（真值0.4），MAE=0.394；空掩膜、平坦背景拒绝、区域外恒等检查通过。这些控制不能替代真实样本验收。</p><p>背景近似复用上一轮缓存。本次计时仅为估计器；results.json 内对照项的 seconds 也复用了此值，不能用于方法速度对比。历史背景用时与本次拟合用时在 <a href="timing_context.json">计时说明</a> 分开记录，相加不是本次端到端实测。</p><p><a href="summary.md">完整文字结论</a> · <a href="protocol.json">预先冻结协议</a> · <a href="results.json">原始评分</a> · <a href="decision.json">门槛决策</a> · <a href="posthoc_regions.json">事后区域归因</a> · <a href="numerical_checks.json">数值检查</a> · <a href="provenance.json">输入与代码摘要</a> · <a href="verification.json">交付复核</a> · <a href="runner_snapshot.py">运行脚本快照</a></p><p>生产 wm/ 源码未修改。结果依赖固定数据集与上一轮缓存；按 summary.md 的命令可在新目录复现。</p></section></main></html>'''
page = page.replace('TABLE',''.join(table)).replace('NAV',''.join(f'<a href="#{r["case_id"]}">{r["case_id"]}</a>' for r in rows)).replace('CARDS',''.join(cards))
(OUT/'report.html').write_text(page)
(OUT/'visual_review.json').write_text(json.dumps({'reviewer':'assistant visual inspection of saved comparison crops','case_notes':notes,'real_003_acceptable':False,'primary_visual_veto':[],'note':'The two numerical passes retain faint residual differences; review does not establish generalization or OCR correctness.'},ensure_ascii=False,indent=2))
print(OUT/'report.html')
