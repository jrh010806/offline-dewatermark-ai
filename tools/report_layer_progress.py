"""Report measured alpha diagnosis and one spatial-projection experiment."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/layer_estimation/20261027_fixed'
old={r['case_id']:r for r in json.loads((OUT/'results.json').read_text())}
new={r['case_id']:r for r in json.loads((OUT/'spatial_alpha/results.json').read_text())}
oracle=json.loads((OUT/'alpha_ablation/results.json').read_text())
names={'estimated_alpha_estimated_color':'估计透明度＋估计颜色','true_alpha_estimated_color':'真实透明度＋估计颜色','estimated_alpha_true_color':'估计透明度＋真实颜色','true_alpha_true_color':'真实透明度＋真实颜色'}
atable=''.join(f'<tr><td>{label}</td><td>{oracle[k]["wm_mae"]:.2f}</td><td>{oracle[k]["ink_iou"]:.3f}</td></tr>' for k,label in names.items())
ntable=''.join(f'<tr><td>{old[c]["label"]}</td><td>{old[c]["methods"]["guarded"]["wm_mae"]:.2f}</td><td>{new[c]["methods"]["guarded"]["wm_mae"]:.2f}</td><td>{"通过" if new[c]["meets_continue_gate"] else "不通过"}</td></tr>' for c in ['core_001','core_010','core_015','core_020'])
review={
 'core_001':'数值门槛通过，但误差比上轮更大。',
 'core_010':'退步：出现彩色噪点、轮廓失真，误差从2.56升到7.58。',
 'core_015':'水印轮廓大幅减弱，但264、313的部分笔画受损；误差3.53未达标，视觉也不通过。',
 'core_020':'原拒绝规则仍拒绝，最终保留带水印输入。',
 'real_003':'正文出现断笔、模糊；水印残留。不可用，无干净参考，不做成功率统计。',
}
(OUT/'spatial_alpha/visual_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
cards=''.join(f'<section><h2>{c} · {old[c]["label"]}</h2><p>{note}</p><a href="spatial_alpha/{c}/comparison.jpg"><img src="spatial_alpha/{c}/comparison.jpg" loading="lazy" alt="{c} 本轮对照"></a></section>' for c,note in review.items())
summary='''# 下一步方向与本轮推进结果

主线：已知位置下，研究能保住底图内容的水印透明度估计。先验收恢复，再投入自动定位。主线尚有依据，但当前两个实现都未达到产品接入条件。

本轮完成两项实验，没有换模型或修改生产代码。

1. 深色大字参数替换诊断：当前误差16.64；仅换真实透明度3.37；仅换真实颜色16.47；两者都真实1.41。只纠正透明度减少约79.7%的误差，只纠正颜色减少约1.0%。透明度是该样本的首要瓶颈，但真实透明度加估计颜色仍未达到MAE≤3，颜色精度仍需控制。这些带真实参数的结果不是自动算法成绩。
2. 一个固定逐像素投影原型：保持共享颜色估计与拒绝规则，只替换透明度公式。局部淡字1.34→2.21；密集淡字2.56→7.58；深色大字16.64→3.53；阴影彩字仍拒绝，输出原图87.55。原有4/4门槛下只1/4通过，弱于上一轮2/4。深色图中的数字笔画仍受损，真实文档也有断笔，不能接入。

为何会退步：像素透明度通过输入与LaMa背景近似之间的差反推，当背景近似损坏文字或结构时，这种做法缺少区分“水印”与“背景补错”的独立约束。可视对照显示数字和正文损伤与背景近似相似。不能把随背景近似变化的像素透明度当成真实透明度。

推进边界：不继续微调这个自由逐像素投影公式，也不因深色图平均误差下降而替换现有流程。保留统一参数版作为淡字实验基线；它也不是产品承诺。

下一项待验证假设：利用水印自身的笔画形状、局部连续性或重复图案建立结构约束，使透明度不能随每个背景补画错误自由变化。重复图案仅适用于存在重复的样本，不能假设深色大字有重复模板。该步骤尚未实现；若不能建立与背景近似相对独立的约束，不再堆叠反演启发式。

下轮继续使用这4张固定主测试，重点同时验证深色图与密集淡字：深色图MAE≤3且数字完整，淡字不比统一版退步，阴影彩字仍如实记为未解决；全部主测试与负样本门槛满足前不进入定位开发。若结构约束仍不能解决退化，应将基于LaMa背景反推的路线判定为不适合高保真文档，并重新评估具备独立水印层监督的方案，不立即开展大规模训练。

诊断边界：core_015有13个全不透明像素、共26个alpha≥0.98像素。四组参数替换统一保留13个全不透明像素，不使用补画兜底，因此诊断基线与上轮误差有约0.002的差别。可恢复区域alpha<0.98的四组误差依次16.438、3.229、16.268、1.276。不得把全不透明区域说成已准确恢复。

所有估计只使用输入、给定二值位置和缓存背景近似。真实alpha/颜色用于明确标记的参数替换诊断；干净参考用于评分。逐像素原型没有读取真实参数。用时不是端到端测试：背景近似复用缓存，诊断scores.seconds=0表示未计时。

复现脚本：tools/probe_layer_ablation.py、tools/probe_spatial_alpha.py。为保护结果，已有results.json时拒绝覆盖。协议、结果与摘要分别保存在alpha_ablation/和spatial_alpha/。
'''
(OUT/'progress.md').write_text(summary)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>推进结果：透明度是瓶颈，背景投影不足</title><style>body{font:16px/1.8 system-ui,sans-serif;background:#f1f4f6;color:#172a37;margin:0}main{max-width:1120px;margin:auto;padding:30px 20px}h1{line-height:1.35}h2{font-size:23px}section{background:white;border:1px solid #dae1e6;border-radius:12px;padding:24px;margin:24px 0}.decision{background:#fff0d5;padding:18px;border-left:5px solid #ba7a13}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dce4e9;text-align:left}img{width:100%;height:auto}a{color:#086a8b}.muted{color:#536774}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:600px){main{padding:18px 12px}section{padding:14px}td,th{font-size:13px;padding:6px}}</style><main><p><a href="report.html">上一轮固定样本结果</a> / 2026-10-27 后续推进</p><h1>透明度是主要瓶颈；直接逐像素投影仍不可用</h1><div class="decision"><b>本轮完成根因对照＋一个最小原型。</b><br>主线继续聚焦透明度估计；淘汰直接跟随补画背景的自由投影实现。尚不进入自动定位或产品接入。</div>
<section><h2>1 · 已明确优先级</h2><p>深色大字中，真实透明度替换减少79.7%的误差；仅真实颜色替换减少1.0%。需要优先解决透明度，但颜色精度也不能忽略。</p><table><tr><th>参数替换诊断</th><th>MAE ↓</th><th>笔画IoU ↑</th></tr>ATABLE</table><p class="muted">后3组使用真实参数，只说明可恢复性及误差来源，不是自动算法成绩。四组统一保留13个完全不透明像素，不补画。原图量化、近不透明像素的误差仍存在。</p><a href="alpha_ablation/comparison.jpg"><img src="alpha_ablation/comparison.jpg" alt="透明度与颜色替换诊断"></a></section>
<section><h2>2 · 最小实现确实跑了，但没有通过</h2><p>保持颜色估计与拒绝规则不变，仅将透明度改成逐像素RGB投影。它只用输入、二值位置与缓存LaMa背景近似，不用真实透明度或颜色。所有参数在评分前冻结，没有追成绩调参。</p><table><tr><th>固定主测试</th><th>统一透明度 MAE</th><th>像素投影 MAE</th><th>本轮结果</th></tr>NTABLE</table><p>只有1/4通过，上轮为2/4。两个负样本仍保留原图。不透明块和阴影彩字仍拒绝。深色图平均误差下降，但数字损伤；密集照片出现彩色噪点，真实文档出现断笔。</p></section>
<section><h2>3 · 推进方向与退出条件</h2><p><b>下一项只验证：带水印结构约束的透明度估计。</b>让水印笔画的形状、局部连续性或存在时的重复图案提供约束，避免把每个背景补画错误都解释成透明度变化。该方案尚未实现，重复模板也不能用于没有重复图案的样本。</p><p>固定样本继续使用。深色字需达到误差门槛并保住数字；淡字不能退步。阴影彩字仍单独记为未解决，全部门槛满足前不做自动定位。若无法建立相对独立的结构约束，就不继续堆反演启发式，重新评估有独立水印层监督的方案。</p><p>图像质量的核心限制仍在恢复阶段。当前方向得到的是“值得定向验证”的证据，没有获得“可以产品化”的证据。</p></section>CARDS
<section><h2>证据</h2><p><a href="progress.md">完整结论与边界</a> · <a href="alpha_ablation/protocol.json">参数诊断协议</a> · <a href="alpha_ablation/results.json">参数诊断评分</a> · <a href="spatial_alpha/protocol.json">原型协议</a> · <a href="spatial_alpha/results.json">原型评分</a> · <a href="progress_verification.json">复核</a></p><p>计时不包括缓存背景生成；参数诊断seconds=0表示未计时。生产wm/源码未修改。原始结果完整保留。</p></section></main></html>'''
page=page.replace('ATABLE',atable).replace('NTABLE',ntable).replace('CARDS',cards)
(OUT/'progress.html').write_text(page)
# A visible entry on the previously opened report, without rewriting prior conclusions.
index=OUT/'report.html';text=index.read_text()
marker='<p class="decision"><a href="progress.html">后续推进：参数根因对照与逐像素原型结果 →</a></p>'
if marker not in text:index.write_text(text.replace('<main>','<main>'+marker,1))
print(OUT/'progress.html')
