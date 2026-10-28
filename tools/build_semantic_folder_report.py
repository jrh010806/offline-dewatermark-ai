"""Build a local, inspectable comparison report for a semantic restoration run."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image


REVIEWS = {
    "charts_015__heavy_dark": {
        "status": "有改善，仍有合成痕迹",
        "note": "AI 局部结果补出了 198、306；现有处理在这两处丢失了数字。大字水印视觉上已去掉，但柱顶与柱边有不平整、轻微色块。AI 整图也重画了线条与字形，不能直接作为无损结果。",
    },
    "charts_016__heavy_dark": {
        "status": "未去净：检测漏选",
        "note": "现有选区漏掉水印尾号 1099，因此现有处理与 AI 局部结果都保留了该尾号。AI 整图去掉了整条水印，局部结果中的 181 较现有处理清楚；柱体边缘仍有合成痕迹。",
    },
    "document_029__heavy_dark": {
        "status": "未处理：选区为空",
        "note": "现有检测返回 0 个像素，最终局部结果与输入完全相同，整条水印保留。AI 整图在视觉上去除了水印并补出文字，但不等于已逐字验证恢复正确。此例不能计为局部修复成功。",
    },
    "document_030__heavy_dark": {
        "status": "未去净：检测漏选",
        "note": "选区只覆盖右侧部分水印和尾号，左侧“测试素”等大字保留。AI 整图去掉了整条水印；局部结果还存在正文接缝，不能用整图预览的干净程度代替最终结果。",
    },
    "document_031__heavy_dark": {
        "status": "未去净：检测漏选",
        "note": "现有选区漏掉水印尾号 1115，最终仍有明显残留。与现有处理相比，AI 局部结果补出的部分表格数字更清楚，但有表格线接缝，且生成数字未按无水印真值核验，不能据此认证数据正确。",
    },
    "document_032__heavy_dark": {
        "status": "未去净：几乎全漏选",
        "note": "现有选区只有 794 个像素，主要落在柱边，几乎没有覆盖水印。最终大字水印基本保留。AI 整图虽然视觉上去掉水印，但正文、数字、柱体细节仍需独立核验。",
    },
    "photo_021__heavy_dark": {
        "status": "结构改善，仍有残留",
        "note": "AI 局部结果中的建筑斜向深色结构带比现有处理连续；现有处理在对应位置有较明显模糊。尾部数字局部漏选，仍能看到残留轮廓；部分玻璃反射与边缘有融合色差。",
    },
    "textures_015__heavy_dark": {
        "status": "未达标：格线与色块不连续",
        "note": "AI 理解了棋盘格背景并去掉水印，但整图格子位置、颜色有变化。放回现有选区后出现格线错位、色块与模糊，不能认定优于现有处理；现有处理也有格子变形。",
    },
    "textures_016__heavy_dark": {
        "status": "未达标：纹理融合明显",
        "note": "AI 整图将原有高频随机噪声重画为另一种纹理。局部融合后出现明显的斜向色块与字形状痕迹；现有处理也存在平滑带。此次 AI 局部结果没有显示出更好的连续性。",
    },
}


def build_report(root: Path) -> Path:
    index = json.loads((root / "index.json").read_text())
    cards, rows, results = [], [], []
    variants = [("source", "输入原图"), ("selection_overlay", "现有检测选区"),
                ("baseline", "现有处理"), ("generated", "AI 整图 · 能力参考"),
                ("result", "AI 局部合成 · 最终")]
    for number, case in enumerate(index["cases"], 1):
        ident = case["id"]
        folder = root / ident
        audit = json.loads((folder / "result.json").read_text())
        source = np.asarray(Image.open(folder / "source.png").convert("RGB"))
        final = np.asarray(Image.open(folder / "result.png").convert("RGB"))
        selected = np.asarray(Image.open(folder / "selection.png").convert("L")) >= 128
        assert source.shape == final.shape
        outside = int(np.count_nonzero(np.any(source != final, axis=2) & ~selected))
        assert outside == audit["changed_outside_selection"] == 0
        review = {**REVIEWS[ident], "review_type": "manual_visual_review",
                  "ground_truth_compared": False, "text_and_numbers_verified": False,
                  "selected_pixels": int(selected.sum()), "outside_changed_pixels": outside}
        (folder / "visual_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))
        results.append({"id": ident, **review, "composition": audit})
        w, h = case["size"]
        box = (int(w * .12), int(h * .33), int(w * .9), int(h * .67)) if ident.startswith("document") else (int(w * .12), int(h * .28), int(w * .9), int(h * .71))
        figures = []
        for name, label in variants:
            image = Image.open(folder / f"{name}.png").convert("RGB")
            if image.size != (w, h):
                image = image.resize((w, h), Image.Resampling.LANCZOS)
            image.crop(box).save(folder / f"{name}_detail.png")
            css = " extra" if name in ("selection_overlay", "generated") else ""
            figures.append(f'<figure class="{css}"><figcaption>{label}</figcaption><a class="image-link" href="{ident}/{name}.png"><img loading="lazy" data-full="{ident}/{name}.png" data-detail="{ident}/{name}_detail.png" src="{ident}/{name}_detail.png" alt="{html.escape(ident)} {label}"></a></figure>')
        baseline = ", ".join(case["baseline_repairs"]) or "未修复"
        cards.append(f'''<section id="{ident}"><header><div class="number">{number:02}</div><div><h2>{html.escape(ident)}</h2><p class="status">{review['status']}</p></div></header>
<p>{review['note']}</p><div class="metrics">选区 {review['selected_pixels']:,} 像素 · AI 合成选区外变化 0 · 现有处理 {html.escape(baseline)}</div>
<div class="images">{''.join(figures)}</div><p class="links"><a href="{ident}/result.png">最终图</a><a href="{ident}/generated.png">AI 整图原始输出</a><a href="{ident}/selection.png">掩码</a><a href="{ident}/prompt.txt">生成提示词</a><a href="{ident}/result.json">像素核验</a></p></section>''')
        rows.append(f'<tr><td><a href="#{ident}">{html.escape(ident)}</a></td><td>{review["selected_pixels"]:,}</td><td>{review["status"]}</td></tr>')
    summary = {"cases_completed": len(results), "outside_pixels_identical_all": True,
               "ground_truth_compared": False, "quality_pass_rate": None,
               "generator": "Codex built-in image_gen.imagegen; model version not exposed",
               "detector": index["detector"], "thresholds": index["thresholds"],
               "selection_modified_for_ai": False, "cases": results}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>深色大字 · 9 张 AI 重建对比</title>
<style>
:root{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#182537;background:#edf1f5;line-height:1.65}*{box-sizing:border-box}body{margin:0}.wrap{max-width:1720px;padding:28px 36px;margin:auto}h1{font-size:32px;line-height:1.25;margin:12px 0}h2{font-size:20px;line-height:1.3;margin:0;overflow-wrap:anywhere}p{margin:10px 0}.intro,section{background:white;border:1px solid #dce3eb;border-radius:12px;padding:24px;margin-bottom:24px}.eyebrow{color:#54667f;font-size:14px}.callout{border-left:4px solid #d48828;padding:12px 16px;background:#fff5e5;margin:20px 0}.muted,.metrics{color:#5d6d80;font-size:14px}table{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #e2e7ed}a{color:#1e5ba6}nav{position:sticky;top:0;z-index:2;background:#edf1f5f5;border-bottom:1px solid #dce3eb;padding:12px 0;display:flex;flex-wrap:wrap;gap:20px;align-items:center;margin-bottom:24px}label{cursor:pointer}input{accent-color:#1e5ba6}section{scroll-margin-top:80px}section header{display:flex;gap:16px;align-items:center}.number{background:#eaf0f8;color:#1e5ba6;border-radius:8px;padding:7px 14px;font-size:22px;font-weight:600}.status{margin:4px 0;color:#985708;font-size:14px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}.images figure{margin:0;background:#f6f8fa;border:1px solid #e1e6ed;border-radius:8px;overflow:hidden}figcaption{padding:9px 12px;font-size:14px;font-weight:600}img{display:block;width:100%;height:auto;border-top:1px solid #e1e6ed}.extra{display:none}body.expanded .extra{display:block}body.expanded .images{grid-template-columns:repeat(5,minmax(0,1fr))}.links{display:flex;gap:20px;flex-wrap:wrap;font-size:13px}dialog{border:0;border-radius:10px;padding:16px;max-width:95vw;max-height:95vh}dialog::backdrop{background:#14202cbb}dialog img{width:auto;max-width:90vw;max-height:85vh;object-fit:contain;border:0}dialog button{float:right;padding:6px 18px;margin-bottom:10px;cursor:pointer}@media(max-width:900px){.wrap{padding:16px}.intro,section{padding:16px}.images,body.expanded .images{display:flex;overflow-x:auto}.images figure{min-width:78vw}table{font-size:12px}td,th{padding:7px 4px}h1{font-size:25px}}
</style><div class="wrap"><div class="intro"><div class="eyebrow">2026-10-28 · 深色大字目录 · 全部 9 张 · 单次生成实验</div><h1>AI 能去掉大字水印，局部修复仍有两个障碍</h1>
<p>一是现有检测漏选；二是 AI 重画的线条、文字和纹理与原图不能完全对齐。此次结果不足以支持“直接替换现有处理”。图表数字和建筑结构有改善，纹理图的局部合成仍不理想。</p>
<div class="callout"><strong>9 / 9 已完成生成、合成与目视检查；9 / 9 最终图的选区外像素完全不变。</strong><br>这只是处理完整性与像素约束的核验，不是修复成功率。文档 029 选区为零，最终图没有发生任何修复。</div>
<p class="muted">做法：沿用当前默认检测配置，对每张图只检测一次；现有处理和 AI 合成使用相同二值选区。AI 接收整张原图和红色选区定位图，生成完整修复图，再将选区内的生成结构通过梯度融合放回原图。没有手工补选、扩大掩码，也没有把无水印参考图提供给 AI。</p>
<p class="muted">生成工具为本次会话内置图像编辑工具，工具未公开具体模型版本；此次不是 Kimi 或 DeepSeek 生成图片，也不代表客户本机离线效果。现有处理按原流程执行，多数样本使用 LaMa，文档 032 同时使用 alpha_unmix 与 LaMa。未与无水印真值逐字逐数核对，不报告内容准确率。</p>
<table><thead><tr><th>样本（点击跳转）</th><th>选区像素</th><th>最终局部结果 · 目视结论</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table></div>
<nav><label><input type="checkbox" id="extended"> 显示检测选区和 AI 整图（5 列）</label><label><input type="checkbox" id="full"> 查看整张图片</label><span class="muted">默认显示水印附近细节；点击图片放大原文件。</span></nav>''' + ''.join(cards) + '''<p class="muted">原输入文件未改动。各图提示词、原始生成图、掩码、最终图和像素核验均保存在同一案例目录。<a href="summary.json">完整核验记录</a> · <a href="index.json">检测与现有处理记录</a></p></div><dialog id="viewer"><button id="close">关闭</button><img id="large" alt="原尺寸预览"></dialog>
<script>
document.getElementById('extended').onchange=e=>document.body.classList.toggle('expanded',e.target.checked);
document.getElementById('full').onchange=e=>document.querySelectorAll('img[data-full]').forEach(im=>im.src=e.target.checked?im.dataset.full:im.dataset.detail);
const viewer=document.getElementById('viewer');document.querySelectorAll('.image-link').forEach(a=>a.onclick=e=>{e.preventDefault();document.getElementById('large').src=a.getAttribute('href');viewer.showModal()});
document.getElementById('close').onclick=()=>viewer.close();viewer.onclick=e=>{if(e.target===viewer)viewer.close()};
</script></html>'''
    target = root / "report.html"
    target.write_text(page)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    print(build_report(parser.parse_args().run_directory).resolve())
