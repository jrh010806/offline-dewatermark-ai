#!/usr/bin/env python3
"""Build a compact, browsable HTML report from the saved first-round metrics."""
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out/baseline_20261026"
IM = json.loads((OUT / "metrics.json").read_text())
PDF = json.loads((OUT / "pdf_metrics.json").read_text())
BEFORE = json.loads((ROOT / "out/baseline_20261026_before_photo_guard/metrics.json").read_text())
CORPUS = ROOT / "data/watermark_training_v1"
SPLIT_ROWS = [json.loads(x) for x in (CORPUS / "splits/validation.jsonl").read_text().splitlines()]
ROWS_BY_ID = {r["id"]: r for r in SPLIT_ROWS}


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def status(ok):
    return '<span class="pass">通过</span>' if ok else '<span class="fail">未通过</span>'


def image_status(row):
    if row['is_negative']:
        return '<span class="pass">安全保留</span>' if row.get('negative_control_passed') else '<span class="fail">误改</span>'
    return status(row['passed'])


image_rows = []
for x in IM["results"]:
    image_rows.append(
        "<tr>"
        f"<td>{html.escape(x['label'])}<small>{html.escape(x['subtype'])}</small></td>"
        f"<td>{'负例' if x['is_negative'] else '正例'}</td>"
        f"<td>{image_status(x)}</td><td>{fmt(x['metrics'].get('mask_iou'))}</td>"
        f"<td>{fmt(x['metrics'].get('residual_coverage'))}</td>"
        f"<td>{fmt(x['metrics'].get('fidelity_mean_delta'), 2)}</td>"
        f"<td>{x['mask_coverage']:.3f}</td><td>{'、'.join(html.escape(s) for s in x['failures']) or '—'}</td>"
        "</tr>")

pdf_rows = []
for doc in PDF["results"]:
    passed = sum(p["passed"] for p in doc["pages"])
    count = len(doc["pages"])
    failures = sorted({f for p in doc["pages"] for f in p["failures"]})
    ious = [p["metrics"].get("mask_iou") for p in doc["pages"] if p["metrics"].get("mask_iou") is not None]
    resid = [p["metrics"].get("residual_coverage") for p in doc["pages"] if p["metrics"].get("residual_coverage") is not None]
    pdf_rows.append(
        "<tr>"
        f"<td>{html.escape(doc['label'])}<small>{html.escape(doc['subtype'])}</small></td>"
        f"<td>{'負例' if doc['is_negative'] else '正例'}</td>"
        f"<td>{passed}/{count} {'<span class=\"pass\">全頁通過</span>' if passed == count else '<span class=\"fail\">有頁失敗</span>'}</td>"
        f"<td>{fmt(sum(ious)/len(ious) if ious else None)}</td>"
        f"<td>{fmt(sum(resid)/len(resid) if resid else None)}</td>"
        f"<td>{'、'.join(html.escape(s) for s in failures) or '—'}</td>"
        "</tr>")


def comparison_card(title, subtitle, status_html, input_src, output_src, meta=""):
    title, subtitle, meta = map(lambda s: html.escape(str(s)), (title, subtitle, meta))
    input_src = html.escape(input_src, quote=True)
    output_src = html.escape(output_src, quote=True)
    return f'''<article class="compare-card"><header><b>{title}</b><span>{status_html}</span><small>{subtitle}</small></header>
    <div class="compare-pair"><figure><a href="{input_src}" target="_blank"><img loading="lazy" src="{input_src}" alt="{title} 输入样本"></a><figcaption>输入样本</figcaption></figure>
    <figure><a href="{output_src}" target="_blank"><img loading="lazy" src="{output_src}" alt="{title} 软件处理结果"></a><figcaption>软件处理结果</figcaption></figure></div>
    <p class="compare-meta">{meta}</p></article>'''


def before_after_card(title, subtitle, before_src, after_src, meta=""):
    title, subtitle, meta = map(lambda s: html.escape(str(s)), (title, subtitle, meta))
    before_src = html.escape(before_src, quote=True)
    after_src = html.escape(after_src, quote=True)
    return f'''<article class="compare-card"><header><b>{title}</b><span class="pass">前后对照</span><small>{subtitle}</small></header>
    <div class="compare-pair"><figure><a href="{before_src}" target="_blank"><img loading="lazy" src="{before_src}" alt="{title} 修改前结果"></a><figcaption>修改前输出</figcaption></figure>
    <figure><a href="{after_src}" target="_blank"><img loading="lazy" src="{after_src}" alt="{title} 修改后结果"></a><figcaption>修改后输出</figcaption></figure></div>
    <p class="compare-meta">{meta}</p></article>'''


image_gallery = []
for x in IM["results"]:
    row = ROWS_BY_ID[x["id"]]
    image_gallery.append(comparison_card(
        f"图片 · {x['label']}", f"{x['subtype']} · {'负例' if x['is_negative'] else '正例'} · {x['id']}",
        image_status(x), f"../data/watermark_training_v1/{row['paths']['watermarked']}",
        f"../out/baseline_20261026/{x['id']}__output.png",
        f"掩码 IoU {fmt(x['metrics'].get('mask_iou'))}；残留 {fmt(x['metrics'].get('residual_coverage'))}；"
        f"掩码覆盖 {x['mask_coverage']:.1%}；失败：{'、'.join(x['failures']) or '—'}"))

pdf_gallery = []
for doc in PDF["results"]:
    row = ROWS_BY_ID[doc["id"]]
    for page_result in doc["pages"]:
        page = row["pages"][page_result["page"] - 1]
        pdf_gallery.append(comparison_card(
            f"PDF · {doc['label']} · 第 {page_result['page']} 页",
            f"{doc['subtype']} · {'负例' if doc['is_negative'] else '正例'} · {doc['id']}",
            status(page_result["passed"]), f"../data/watermark_training_v1/{page['paths']['watermarked']}",
            f"../out/baseline_20261026/{doc['id']}__page_{page_result['page']}__output.png",
            f"方法 {page_result['mask_method']}；掩码 IoU {fmt(page_result['metrics'].get('mask_iou'))}；"
            f"残留 {fmt(page_result['metrics'].get('residual_coverage'))}；"
            f"失败：{'、'.join(page_result['failures']) or '—'}"))

before_by_id = {x["id"]: x for x in BEFORE["results"]}
after_by_id = {x["id"]: x for x in IM["results"]}
delta_ids = ["photo_033__local_text", "photo_033__negative",
             "document_025__normal_logo_signature_negative"]
before_after_gallery = []
for sample_id in delta_ids:
    old, new = before_by_id[sample_id], after_by_id[sample_id]
    if old["is_negative"]:
        row = ROWS_BY_ID[sample_id]
        source = np.asarray(Image.open(CORPUS / row["paths"]["watermarked"]).convert("RGB"))
        old_output = np.asarray(Image.open(ROOT / "out/baseline_20261026_before_photo_guard" / f"{sample_id}__output.png").convert("RGB"))
        new_output = np.asarray(Image.open(OUT / f"{sample_id}__output.png").convert("RGB"))
        old_changed = np.any(source != old_output, axis=2).mean()
        new_changed = np.any(source != new_output, axis=2).mean()
        explanation = (f"误改像素占比：{old_changed:.2%} → "
                       f"{new_changed:.2%}；掩码面积："
                       f"{old['mask_coverage']:.2%} → {new['mask_coverage']:.2%}。")
    else:
        explanation = (f"水印是否去除：仍未去除；掩码面积：{old['mask_coverage']:.2%} → "
                       f"{new['mask_coverage']:.2%}。本次修正只拦截不可靠的大范围重绘，不代表正例去水印成功。")
    before_after_gallery.append(before_after_card(
        f"修改前后 · {new['label']}", f"{new['subtype']} · {sample_id}",
        f"../out/baseline_20261026_before_photo_guard/{sample_id}__output.png",
        f"../out/baseline_20261026/{sample_id}__output.png", explanation))

image_failure = IM["aggregate"]["failure_counts"]
pdf_failure = PDF["aggregate"]["failure_counts"]
html_doc = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>首轮去水印基线评测</title>
<style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f3f5f8;color:#243247;font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}}
aside{{position:fixed;inset:0 auto 0 0;width:230px;padding:28px 18px;background:#14253b;color:#fff;overflow:auto}}aside b{{display:block;font-size:18px;padding:0 10px 18px}}aside a{{display:block;color:#d2deed;text-decoration:none;padding:9px 10px;border-radius:8px;font-size:14px}}aside a:hover{{background:#294360}}
main{{max-width:1440px;margin-left:230px;padding:34px 38px 80px}}section{{scroll-margin-top:20px;background:#fff;border:1px solid #e3e8ee;border-radius:14px;padding:28px;margin-bottom:22px}}h1{{font-size:34px;line-height:1.3;margin:8px 0 12px}}h2{{font-size:22px;margin:0 0 18px}}h3{{font-size:17px;margin:24px 0 8px}}p{{margin:10px 0}}.muted{{color:#6d7b8d;font-size:13px}}.lead{{font-size:18px;color:#52647c}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.stats div{{padding:16px;background:#eff5fb;border-radius:10px}}.stats b{{display:block;font-size:30px;color:#205e9d}}.stats span{{font-size:13px;color:#60748b}}.notice{{padding:16px 18px;border-left:4px solid #d99828;background:#fff7e8;border-radius:5px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;vertical-align:top;padding:10px 11px;border-bottom:1px solid #e3e8ee}}th{{background:#eff3f8;color:#48627f;white-space:nowrap}}td:first-child{{font-weight:600;min-width:150px}}td small{{display:block;color:#78879a;font-weight:400}}.pass{{color:#16744e;font-weight:700}}.fail{{color:#b53c39;font-weight:700}}code{{background:#f0f3f7;padding:2px 5px;border-radius:4px}}.links a{{margin-right:18px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{border:1px solid #e1e7ef;border-radius:10px;padding:12px}}.card img{{width:100%;height:170px;object-fit:contain;background:#eef1f4}}.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.compare-card{{border:1px solid #dfe6ee;border-radius:12px;padding:13px;min-width:0}}.compare-card header{{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:center;margin-bottom:9px}}.compare-card header small{{grid-column:1/-1;color:#718096;font-size:12px;overflow-wrap:anywhere}}.compare-pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}.compare-pair figure{{margin:0;min-width:0}}.compare-pair img{{display:block;width:100%;height:240px;object-fit:contain;background:#eef1f4;border-radius:6px}}.compare-pair figcaption{{font-size:12px;color:#62738a;text-align:center;padding-top:4px}}.compare-meta{{font-size:12px;color:#65758a;margin:8px 0 0;overflow-wrap:anywhere}}@media(max-width:900px){{aside{{width:185px}}main{{margin-left:185px;padding:20px}}.stats{{grid-template-columns:repeat(2,1fr)}}.gallery{{grid-template-columns:1fr}}}}@media(max-width:620px){{aside{{position:relative;width:auto;padding:14px}}aside b{{padding:0 8px 8px}}aside a{{display:inline-block;padding:6px 8px;font-size:12px}}main{{margin:0;padding:12px}}section{{padding:18px}}h1{{font-size:27px}}.cards{{grid-template-columns:1fr}}.compare-pair img{{height:180px}}}}
</style>
<aside><b>首轮基线 · 2026-10-26</b><a href="#summary">01　结果概览</a><a href="#changes">02　修改前后对照</a><a href="#images">03　图片分类结果</a><a href="#pdf">04　PDF 原生结果</a><a href="#cases">05　代表样本</a><a href="#all-results">06　全部输入与输出</a><a href="#limits">07　范围与注意事项</a><a href="#next">08　下一步</a></aside>
<main>
<section id="summary"><p class="muted">使用冻结的 <code>data/watermark_training_v1/splits/validation.jsonl</code>；本轮复测没有把新增样本混入验证集</p><h1>自有去水印流程首轮基线 · 安全修正复测</h1>
<p class="lead">本轮处理了照片场景误检和文档页眉编号误检，并对同一冻结验证集复跑。报告同时给出修改前后输出图片，便于直观看变化。</p>
<div class="stats"><div><b>{IM['aggregate']['positive_passed']}/{IM['aggregate']['positive_total']}</b><span>图片水印正例通过</span></div><div><b>{IM['aggregate']['negative_controls_passed']}/{IM['aggregate']['negative_controls_total']}</b><span>图片负例安全不改动</span></div><div><b>{PDF['aggregate']['passed']}/{PDF['aggregate']['total']}</b><span>PDF 页面通过</span></div><div><b>{IM['aggregate']['median_elapsed_ms']/1000:.1f}s</b><span>图片处理耗时中位数</span></div></div>
<div class="notice"><b>本轮结论：</b>图片水印正例仍为 0/{IM['aggregate']['positive_total']}，表示正例没有成功去除；两个干净负例均未被改动（2/2）。此前照片负例被误画约 17.7% 像素，现已逐像素保持原样。PDF 同批复跑由 7/24 变为 {PDF['aggregate']['passed']}/{PDF['aggregate']['total']} 页通过；本轮同时调整了角标文字拒识，结果变化不能归因于照片门控。验证集水印为项目合成，不能据此宣称覆盖真实应用水印。</div>
<p class="muted">通过条件按项目现有四项评判（残留、背景保真、正文识别、时限）；负例单独依据 negative_control_passed 判断是否安全保留；原评分器的 passed/failures 字段仍保留供追溯，不用“未检测到水印”判定干净负例失败。表格另列掩码质量和原始诊断。</p></section>
<section id="images"><h2>图片：22 种水印样式 + 2 个负例</h2><p>每个水印子类型取验证清单首条，单页检测；不传样式提示。22 种正例目前都来自同一张留出底图，适合看风格识别的初步短板，不代表跨背景泛化。</p>
<div class="table-wrap"><table><thead><tr><th>类别</th><th>样本</th><th>正例四项 / 负例保留</th><th>掩码 IoU</th><th>残留比例</th><th>掩码外平均改动</th><th>预测掩码面积</th><th>触发失败</th></tr></thead><tbody>{''.join(image_rows)}</tbody></table></div>
<p class="muted">图片失败计数：残留 {image_failure.get('residual_watermark',0)} 条、正文回退 {image_failure.get('recognition_regression',0)} 条、背景保真 {image_failure.get('fidelity_breach',0)} 条。记录可能同时命中多个条件。</p></section>
<section id="pdf"><h2>PDF：原生文件处理</h2><p>每个 11 种 PDF 存储/对象子类型取一份验证文档，保留 PDF 原生入口处理，逐页将输出与对应干净 PDF 对比。共 12 份文档、24 页，包含无水印负例。</p>
<div class="table-wrap"><table><thead><tr><th>类别</th><th>样本</th><th>页面结果</th><th>平均掩码 IoU</th><th>平均残留比例</th><th>触发失败</th></tr></thead><tbody>{''.join(pdf_rows)}</tbody></table></div>
<p class="muted">PDF 失败计数：残留 {pdf_failure.get('residual_watermark',0)} 页、正文回退 {pdf_failure.get('recognition_regression',0)} 页。每类只有一份文档，暂不能作为总体成功率。</p></section>
<section id="changes"><h2>本轮修改前后图片对照</h2><p class="muted">左侧为修改前的实际输出，右侧为加入保守检测门控和角标上下文拒识后的实际输出。负例误改得到修复；照片正例被安全拦截但水印仍残留。</p><div class="gallery">{''.join(before_after_gallery)}</div></section>
<section id="cases"><h2>代表输出</h2><p class="muted">每张卡片左侧是原带水印输入，右侧是当前软件输出。完整输出和掩码文件保存在基线目录。</p>
<div class="cards"><div class="card"><b>图片 · 局部文字</b><p><img src="../data/watermark_training_v1/images/validation/photo_033__local_text/watermarked.png" alt="局部文字输入"><img src="../out/baseline_20261026/photo_033__local_text__output.png" alt="去水印输出"></p><small>失败：水印残留；掩码 IoU {fmt(IM['results'][0]['metrics'].get('mask_iou'))}</small></div>
<div class="card"><b>图片 · 不透明 Logo</b><p><img src="../data/watermark_training_v1/images/validation/photo_033__logo_opaque/watermarked.png" alt="不透明Logo输入"><img src="../out/baseline_20261026/photo_033__logo_opaque__output.png" alt="去水印输出"></p><small>失败：水印残留、正文识别回退；当前输出没有改动背景。</small></div>
<div class="card"><b>图片 · 无水印负例</b><p><img src="../data/watermark_training_v1/bases/photos/photo_033.png" alt="无水印输入"><img src="../out/baseline_20261026/photo_033__negative__output.png" alt="负例输出"></p><small>预测掩码面积 {after_by_id['photo_033__negative']['mask_coverage']:.1%}；误改像素 {after_by_id['photo_033__negative'].get('changed_fraction',0):.1%}。</small></div></div></section>
<section id="all-results"><h2>全部测试样本与输出对比</h2><p>共展示图片 {len(image_gallery)} 条、PDF {len(pdf_gallery)} 页。每张图可点击打开原尺寸；卡片按本轮实际输入和落盘输出生成。</p><h3>图片（输入 / 处理结果）</h3><div class="gallery">{''.join(image_gallery)}</div><h3>PDF（输入页 / 处理结果页）</h3><div class="gallery">{''.join(pdf_gallery)}</div></section>
<section id="limits"><h2>范围与注意事项</h2><ul>
<li>图片集每种水印只测 1 张、且正例共用同一底图；PDF 每种子类只测 1 份文档。样本量不足以对外宣称全面效果。</li>
<li>图片计时包含目前较慢的图像检测/修复路径，样本中位数 {IM['aggregate']['median_elapsed_ms']/1000:.1f} 秒；PDF 页面中位数 {PDF['aggregate']['median_elapsed_ms']/1000:.1f} 秒。</li>
<li>负例的误检掩码没有被现有四项通过条件单独拦截，因此报告同时展示掩码 IoU/面积；“通过”不等于可无审核上线。</li>
<li>本轮水印均为项目合成，底图来自留出照片/文档；尚未验证醒图、抖音等真实应用导出样本的表现。本轮只测自有实现，不含竞品对比；验证集与公开样本的重叠也未独立审计。</li>
</ul></section>
<section id="next"><h2>下一步</h2><ol><li>先看失败案例和预测掩码，区分“检测错位”与“修复/阈值策略”问题；优先处理高残留类别。</li><li>增加图片验证抽样：每种样式选不同底图/来源组，纳入纹理、人物、文字背景等场景，再复跑同一协议。</li><li>按失败簇做修复后，只在固定 validation 上复测；调参确认后再用冻结 test 做一次最终评估。</li><li>M03 已收到 2 份底稿共 4 页扫描图；先完成电子源页配准和水印区域标注复核，再判断可采用的训练方式。这两份底稿继承原有来源组，不另算独立新来源，也不混入本轮验证/测试结果。</li></ol><p class="links"><a href="../out/baseline_20261026/metrics.json">图片逐样本 JSON</a><a href="../out/baseline_20261026/pdf_metrics.json">PDF 逐页 JSON</a></p></section>
</main></html>'''
(ROOT / "docs/首轮基线评测.html").write_text(html_doc, encoding="utf-8")
print(ROOT / "docs/首轮基线评测.html")
