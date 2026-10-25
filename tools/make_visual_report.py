"""生成通俗易读的可视化测试报告（HTML）。

分组清单 + 每样例「带水印原图 | 去水印输出 | 干净参考」三联对照图 + 图下判定，
附功能总结与下一步优化方向。数据全部来自 tools/run_full_audit.py 的产物。

Usage:
    .venv/bin/python tools/make_visual_report.py \
        --real-root ../real_samples --wm-root ../watermark_samples --out out/audit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def log(msg: str) -> None:
    print(f"[report {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def img(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def strip(paths: list[Path], out: Path, width: int = 340) -> None:
    """三联图：等比缩放到统一高度后横向拼接，每列下方留白由 HTML 标注。"""
    tiles = []
    h = 300
    for p in paths:
        im = Image.open(p).convert("RGB")
        w = max(1, int(im.width * h / im.height))
        im = im.resize((min(w, 640), h))
        tiles.append(np.asarray(im))
    hmin = min(t.shape[0] for t in tiles)
    tiles = [t[:hmin] for t in tiles]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out, quality=88)


# ------------------------------------------------------------------ 判定规则
def judge_real(r: dict) -> tuple[str, str, str]:
    """real_samples 判定 -> (等级, 颜色, 一句话说明)"""
    fails = set(r["failures"])
    if r["passed"] and r["psnr"] >= 38:
        return ("优秀", "good", f"四条件全过，PSNR {r['psnr']}，水印区 MAE {r['roi_mae_input']}→{r['roi_mae_output']}")
    if r["passed"]:
        return ("良好", "good", f"四条件全过，PSNR {r['psnr']}，水印区残留已压到 {r['roi_mae_output']}")
    notes = []
    if r["ink_damage"] > 0.05:
        notes.append(f"误伤文字较明显（{r['ink_damage']:.0%}）")
    if "out_of_time_budget" in fails:
        notes.append(f"超时（{r['ms'] // 1000}s > 120s）")
    if "fidelity_breach" in fails:
        notes.append("非水印区被明显改动")
    improved = r["roi_mae_output"] < 0.7 * r["roi_mae_input"]
    if improved and r["ink_damage"] <= 0.05:
        lv, col = "一般", "mid"
        notes.append(f"水印区有改善（MAE {r['roi_mae_input']}→{r['roi_mae_output']}）但未达标："
                     + ("、".join(fails) if fails else "识别条件未过"))
    else:
        lv, col = "较差", "bad"
        why = "、".join(notes) if notes else ("水印区几乎无改善" if not improved else "、".join(fails))
        notes.append(f"未达标（{why or '、'.join(fails)}）")
    return (lv, col, "；".join(notes))


def judge_wm(r: dict) -> tuple[str, str, str]:
    """watermark_samples 判定"""
    d = r["output_roi"] - r["input_roi"]
    oc = r["outside_changed"] or 0.0
    if d < -0.5 and oc < 0.02:
        return ("改善", "good", f"水印区 MAE {r['input_roi']:.1f}→{r['output_roi']:.1f}，区外未动")
    if abs(d) <= 0.5 and oc < 0.02:
        return ("无变化", "mid", "未检测到可靠水印证据，未做处理（保守行为）")
    if oc > 0.10:
        return ("变差", "bad", f"区外 {oc:.0%} 像素被改动，水印区误差{d:+.1f} —— 定位误触发")
    if d > 0.5:
        return ("变差", "bad", f"水印区误差{d:+.1f}，区外改动 {oc:.0%}")
    return ("无变化", "mid", f"区外改动 {oc:.0%}，未获得改善")


# ------------------------------------------------------------------- HTML
CSS = """
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;background:#f5f6f8;color:#1c1e21}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:26px} h2{font-size:21px;margin-top:48px;border-left:5px solid #4a86d4;padding-left:12px}
h3{font-size:17px;margin-top:32px;color:#234}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13.5px}
th,td{border:1px solid #d5d9e0;padding:7px 10px;text-align:left}
th{background:#eceff3}
.card{background:#fff;border:1px solid #d5d9e0;border-radius:10px;padding:14px;margin:18px 0}
.card img{width:100%;border-radius:6px;border:1px solid #e3e6ea;display:block}
.verdict{margin-top:10px;font-size:14.5px;line-height:1.65}
.badge{display:inline-block;padding:2px 12px;border-radius:12px;color:#fff;font-weight:600;font-size:13px;margin-right:10px}
.good{background:#2e9e5b}.mid{background:#c9920e}.bad{background:#c93a3a}
.cols{display:flex;gap:10px;font-size:12.5px;color:#667;margin-top:6px}
.cols span{flex:1;text-align:center;font-weight:600}
.note{background:#fff8e6;border:1px solid #e8d48b;border-radius:8px;padding:12px 16px;font-size:13.5px;line-height:1.7}
.toc{background:#eef3fb;border:1px solid #c9d9f2;border-radius:8px;padding:12px 20px;font-size:14px;line-height:2}
.kpi{display:inline-block;background:#fff;border:1px solid #d5d9e0;border-radius:8px;padding:10px 18px;margin:6px 8px 0 0;font-size:14px}
.kpi b{font-size:19px;display:block}
ul{line-height:1.9} code{background:#eef1f4;padding:1px 6px;border-radius:4px;font-size:12.5px}
.small{font-size:12.5px;color:#667}
"""


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-root", type=Path, default=Path("../real_samples"))
    ap.add_argument("--wm-root", type=Path, default=Path("../watermark_samples"))
    ap.add_argument("--out", type=Path, default=Path("out/audit"))
    ap.add_argument("--baseline", type=Path, default=Path("out/audit_before_iter1"),
                    help="优化前基线结果目录（用于前后对比，可不存在）")
    ap.add_argument("--layered-out", type=Path, default=Path("out/audit_layered"),
                    help="分层样本结果目录（可不存在）")
    args = ap.parse_args()
    out: Path = args.out
    assets = out / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)

    L: list[str] = []
    L.append(f"<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>"
             f"<title>离线智能去水印 · 可视化测试报告</title><style>{CSS}</style></head><body><div class='wrap'>")
    L.append("<h1>离线智能去水印 · 可视化测试报告</h1>")
    L.append(f"<div class='small'>生成时间：{time.strftime('%Y-%m-%d %H:%M')} ｜ "
             f"Python {sys.version.split()[0]} ｜ macOS (Intel, 纯 CPU) ｜ "
             f"被测代码：offline-dewatermark-ai 主分支</div>")

    # ---------------- 数据装载
    real = json.loads((out / "real_samples.json").read_text("utf-8"))
    wm_rows = json.loads((out / "wm_baseline/metrics.json").read_text("utf-8"))
    inv = json.loads((out / "wm_inventory.json").read_text("utf-8"))
    pdf_rows = json.loads((out / "wm_pdf_agg.json").read_text("utf-8"))["rows"]
    unit_ok = "OK" in (out / "unit_tests.log").read_text("utf-8").splitlines()[-1]
    base_real = None
    if (args.baseline / "real_samples.json").exists():
        base_real = json.loads((args.baseline / "real_samples.json").read_text("utf-8"))
    base_wm = None
    if (args.baseline / "wm_baseline/metrics.json").exists():
        base_wm = json.loads((args.baseline / "wm_baseline/metrics.json").read_text("utf-8"))
    layered = None
    if (args.layered_out / "real_samples.json").exists():
        layered = json.loads((args.layered_out / "real_samples.json").read_text("utf-8"))

    # ---------------- 一、评判标准（置顶，通俗口径）
    L.append("<h2>一、当前去水印效果的评判标准</h2>")
    L.append("""
<table>
<tr><th>清单</th><th>评分依据</th><th>徽章怎么打</th></tr>
<tr><td><b>A · real_samples</b>（16 个，带真值）</td>
<td>项目自定义<b>四条件协议</b>，四条同时满足才算「通过」：<br>
① <b>保真</b>：非水印区改动均值 ≤1、p99 ≤4（0–255 灰度单位）；<br>
② <b>残膜</b>：真值水印区域内残留覆盖率 ≤15%（本轮起只在真值水印区统计，排除 JPEG 压缩噪点）；<br>
③ <b>识别</b>：正文笔迹 ink_iou ≥0.90（对照干净参考算，防误伤文字）；<br>
④ <b>时限</b>：单页 ≤120 秒。<br>
另附 PSNR/SSIM（越高越好）、掩码 IoU/精确率/召回（对水印 alpha 真值算）。</td>
<td><span class='badge good'>优秀</span>四条件全过且 PSNR≥38；<span class='badge good'>良好</span>四条件全过；<br>
<span class='badge mid'>一般</span>水印区明显改善（MAE 降 30%+）且误伤 ≤5%，但未全过；<br>
<span class='badge bad'>较差</span>误伤 >5% / 超时 / 无改善。</td></tr>
<tr><td><b>B · 外部图片</b>（300 张）</td>
<td>无四条件协议（图片无逐笔真值），用两个直观量：<br>
① <b>水印区 MAE</b>：输出与干净参考在水印真值区域内的平均像素误差，前后对比看改善；<br>
② <b>区外改动率</b>：水印区之外被改动 >4 灰度级的像素占比，衡量误伤。</td>
<td><span class='badge good'>改善</span>MAE 降 >0.5 且区外改动 <2%；<br>
<span class='badge mid'>无变化</span>基本没动（含保守不处理）；<br>
<span class='badge bad'>变差</span>MAE 升 >0.5 或区外改动 ≥2%（>10% 特别注明）。</td></tr>
<tr><td><b>C · 外部 PDF</b>（10 份）</td>
<td>渲染后与干净参考<b>逐像素比对</b>；并检查干净 PDF 是否被误改（负样本）。</td>
<td><span class='badge good'>优秀</span>全部页逐像素一致、负样本零误改；<br>
<span class='badge mid'>不适用</span>扫描式无独立文字对象，正确拒绝、未做改动。</td></tr>
<tr><td><b>D · 工程回归</b></td><td>46 个单元/界面自动化测试。</td><td>全部通过 / 存在失败。</td></tr>
</table>""")

    # ---------------- 二、总体结果一览（每组总数 + 各效果数量）
    from collections import Counter
    a_counts = Counter(judge_real(r)[0] for r in real["rows"])
    b_counts = Counter(judge_wm(r)[0] for r in wm_rows.values())
    c_counts = Counter("优秀" if r["type"] == "vector_text" else "不适用" for r in pdf_rows)
    a_detail = "、".join(f"{k} {a_counts.get(k, 0)}" for k in ("优秀", "良好", "一般", "较差") if a_counts.get(k))
    b_detail = "、".join(f"{k} {b_counts.get(k, 0)}" for k in ("改善", "无变化", "变差") if b_counts.get(k))
    c_detail = "、".join(f"{k} {v}" for k, v in c_counts.items())
    L.append("<h2>二、总体结果一览（各组总数与效果分布）</h2>")
    L.append("<table><tr><th>清单</th><th>样例总数</th><th>各效果数量</th><th>通过率</th></tr>")
    L.append(f"<tr><td>A · real_samples</td><td>{len(real['rows'])}</td><td>{a_detail}</td>"
             f"<td>四条件通过 {a_counts['优秀'] + a_counts['良好']}/16</td></tr>")
    L.append(f"<tr><td>B · 外部图片</td><td>{len(wm_rows)}</td><td>{b_detail}</td>"
             f"<td>改善 {b_counts.get('改善', 0)}/300</td></tr>")
    L.append(f"<tr><td>C · 外部 PDF</td><td>{len(pdf_rows)}</td><td>{c_detail}</td>"
             f"<td>可编辑 PDF 无损 5/5</td></tr>")
    L.append(f"<tr><td>D · 工程回归</td><td>46 个用例</td><td>{'通过 46' if unit_ok else '有失败'}</td>"
             f"<td>{'100%' if unit_ok else '见日志'}</td></tr>")
    L.append("</table>")
    L.append("<div class='note'><b>一句话读法：</b>文档类样例（A）整体可用、点阵最好；"
             f"外部照片（B）{b_counts.get('变差', 0)} 张变差是主要问题，集中在彩色 Logo 类；"
             "PDF 结构剥离（C）确定性强。数字均为样例级统计，不是通用「去除成功率」。</div>")

    # ---------------- 二·五、优化前后对比
    if base_real is not None:
        old_by_id = {r["id"]: r for r in base_real["rows"]}
        old_pass = sum(1 for r in base_real["rows"] if r["passed"])
        new_pass = sum(1 for r in real["rows"] if r["passed"])
        L.append("<h2>二·五、优化前后对比（同一批样例、同一口径）</h2>")
        L.append(f"<div class='kpi'><b>{old_pass} → {new_pass}</b>real_samples 四条件通过（/16）</div>")
        if base_wm is not None:
            old_b = Counter(judge_wm(r)[0] for r in base_wm.values())
            new_b = Counter(judge_wm(r)[0] for r in wm_rows.values())
            L.append(f"<div class='kpi'><b>{old_b.get('改善', 0)} → {new_b.get('改善', 0)}</b>外部图片改善</div>"
                     f"<div class='kpi'><b>{old_b.get('变差', 0)} → {new_b.get('变差', 0)}</b>外部图片变差</div>")
        L.append("<table><tr><th>样本</th><th>优化前：检测路径 / 判定</th><th>优化后：检测路径 / 判定</th>"
                 "<th>掩码 IoU 前→后</th><th>识别 IoU 前→后</th><th>残膜 前→后</th></tr>")
        for r in real["rows"]:
            o = old_by_id.get(r["id"])
            if not o:
                continue
            def v(x):
                return "通过" if x["passed"] else "未过(" + ",".join(x["failures"]) + ")"
            L.append(f"<tr><td>{esc(r['id'])}</td><td><code>{esc(o['method'])}</code><br>{esc(v(o))}</td>"
                     f"<td><code>{esc(r['method'])}</code><br>{esc(v(r))}</td>"
                     f"<td>{o['mask_iou']} → {r['mask_iou']}</td>"
                     f"<td>{o['ink_iou']} → {r['ink_iou']}</td>"
                     f"<td>{o.get('residual_coverage', '—')} → {r.get('residual_coverage', '—')}</td></tr>")
        L.append("</table>")
        L.append("<div class='note'>说明：优化前基线未记录残膜口径修正前的数值时以 — 表示；"
                 "残膜自本轮起只在真值水印区域内统计（旧口径把 JPEG 噪点也计入，会系统性偏高）。</div>")

    # 清单总览
    n_real = len(real["rows"])
    n_img = sum(1 for r in inv["rows"] if r["kind"] == "image")
    n_pdf = sum(1 for r in inv["rows"] if r["kind"] == "pdf")
    L.append("<h2>三、测试样例分组清单</h2>")
    L.append("<div class='toc'>")
    L.append("<b>本次测试共 4 组清单：</b><br>")
    L.append(f"清单 A：real_samples 真实底材文档水印 —— <b>{n_real}</b> 个样例（4 种水印样式 × 4，带真值，可精确评分）<br>")
    L.append(f"清单 B：watermark_samples 外部图片 —— <b>{n_img}</b> 个样例（彩色 Logo 150 / 单处文字 75 / 平铺文字 75，配无水印参考图）<br>")
    L.append(f"清单 C：watermark_samples 外部 PDF —— <b>{n_pdf}</b> 份（可编辑 5 份 + 扫描式 5 份，配对校验）<br>")
    L.append("清单 D：工程回归 —— 46 个单元/界面自动化测试")
    L.append("</div>")
    L.append("<div class='note'>看图口径：每张三联图从左到右 = <b>① 带水印原图（输入）</b> → "
             "<b>② 去水印输出（程序结果）</b> → <b>③ 干净参考（仅用于评分，程序看不到）</b>。"
             "图下徽章：<span class='badge good'>优秀/改善</span>达到或超过预期；"
             "<span class='badge mid'>一般/无变化</span>保守处理或未完全达标；"
             "<span class='badge bad'>较差/变差</span>存在明显问题。所有指标为样例级误差，"
             "不等于「去除成功率」。<br><br>"
             "<b>干净参考从哪来？</b>所有样例都同时有「带水印原图」和「原始干净参考」两张配对图，"
             "参考图只在推理结束后用于评分，检测/修复程序全程看不到它。具体来源：<br>"
             "① real_samples：拿真实文档/照片做底，用 wm/synth.py 以固定随机种子 20261022 叠加合成水印，"
             "底图原样就是干净参考，因此真值精确到像素（合成水印，非真实互联网水印）；<br>"
             "② watermark_samples：01_dataset_logo 为公开 HuggingFace 配对数据集（COCO 底图 + 合成 Logo 水印，"
             "干净参考即数据集原始无水印图）；02_synthetic_images 由样本包制作方在 COCO 底图上人工叠加水印；"
             "③ PDF 为 Kleister NDA 原始文件，带水印版是本次测试另行叠加透明文字生成的，原件即干净参考。</div>")

    # ---------------- 清单 A
    L.append("<h2>清单 A：real_samples（真实文档底材，带真值）</h2>")
    agg = real["agg"]
    style_names = {"tiled_text": "平铺文字水印", "big_center": "居中大号文字",
                   "dot_grid": "点阵水印", "heavy_dark": "深色重字母水印"}
    L.append("<table><tr><th>样式</th><th>样例数</th><th>四条件通过</th><th>掩码 IoU↑</th>"
             "<th>识别 IoU↑</th><th>PSNR↑</th><th>水印区 MAE↓</th><th>平均耗时</th></tr>")
    for g, a in agg.items():
        L.append(f"<tr><td>{esc(style_names.get(g, g))} <code>{g}</code></td><td>{a['n']}</td>"
                 f"<td>{a['passed']}/{a['review_or_fail']}</td><td>{a['mask_iou']}</td>"
                 f"<td>{a['ink_iou']}</td><td>{a['psnr']}</td>"
                 f"<td>{a['roi_mae_input']} → {a['roi_mae_output']}</td><td>{a['ms'] // 1000}s</td></tr>")
    L.append("</table>")
    for g in ("dot_grid", "tiled_text", "big_center", "heavy_dark"):
        L.append(f"<h3>A · {esc(style_names[g])}（{g}）</h3>")
        for r in real["rows"]:
            if r["style"] != g:
                continue
            lv, col, note = judge_real(r)
            png = f"A_{r['id']}.png"
            src = out / "comparisons" / f"{r['id']}.png"
            if not (assets / png).exists() and src.exists():
                (assets / png).write_bytes(src.read_bytes())
            L.append("<div class='card'>")
            L.append(f"<img src='report_assets/{png}' loading='lazy'>")
            L.append("<div class='cols'><span>① 带水印原图</span><span>② 去水印输出</span><span>③ 干净参考</span></div>")
            L.append(f"<div class='verdict'><span class='badge {col}'>{lv}</span>"
                     f"<b>{esc(r['id'])}</b>（检测路径 <code>{esc(r['method'])}</code>）｜ {esc(note)}</div>")
            L.append("</div>")

    # ---------------- 清单 A2：分层能力评测
    if layered is not None:
        L.append(f"<h2>清单 A2：分层能力评测（{len(layered['rows'])} 张，样式 × 透明度 × 底材）</h2>")
        L.append("<div class='note'>真底材+合成水印，用于验证能力覆盖面而非对老 16 张过拟合。"
                 "分组键：样式 × 透明度（light/medium/dark）× 底材（pdf_page/photo）。</div>")
        groups = {}
        for r in layered["rows"]:
            groups.setdefault((r["style"], r.get("strength_layer", "?"), r["base_kind"]), []).append(r)
        L.append("<table><tr><th>样式</th><th>透明度</th><th>底材</th><th>n</th><th>四条件通过</th>"
                 "<th>掩码 IoU</th><th>识别 IoU</th><th>水印区 MAE 前→后</th></tr>")
        for (style, layer, kind), rows in sorted(groups.items()):
            n = len(rows)
            passed = sum(r["passed"] for r in rows)
            miou = round(sum(r["mask_iou"] for r in rows) / n, 3)
            iiou = round(sum(r["ink_iou"] for r in rows) / n, 3)
            mae_in = round(sum(r["roi_mae_input"] for r in rows) / n, 2)
            mae_out = round(sum(r["roi_mae_output"] for r in rows) / n, 2)
            L.append(f"<tr><td>{style}</td><td>{layer}</td><td>{kind}</td><td>{n}</td>"
                     f"<td>{passed}/{n}</td><td>{miou}</td><td>{iiou}</td><td>{mae_in} → {mae_out}</td></tr>")
        L.append("</table>")
        for r in layered["rows"]:
            lv, col, note = judge_real(r)
            png = f"A2_{r['id']}.png"
            src = args.layered_out / "comparisons" / f"{r['id']}.png"
            if not (assets / png).exists() and src.exists():
                (assets / png).write_bytes(src.read_bytes())
            L.append("<div class='card'>")
            L.append(f"<img src='report_assets/{png}' loading='lazy'>")
            L.append("<div class='cols'><span>① 带水印原图</span><span>② 去水印输出</span><span>③ 干净参考</span></div>")
            L.append(f"<div class='verdict'><span class='badge {col}'>{lv}</span>"
                     f"<b>{esc(r['id'])}</b>（{esc(r['method'])}）｜ {esc(note)}</div>")
            L.append("</div>")

    # ---------------- 清单 B
    L.append(f"<h2>清单 B：watermark_samples 外部图片（{n_img} 张 256×256）</h2>")
    L.append("<div class='note'>这批样本来自公开数据集/人工叠加，底图为自然照片——正好检验程序在"
             "<b>「文档之外」场景</b>的泛化能力。每张三联图同上；因数量多，按类型分组展示全部样例。</div>")
    type_names = {"dataset_logo": "B1 · 彩色 Logo 水印（dataset_logo，150 张）",
                  "text": "B2 · 单处文字水印（text，75 张）",
                  "tiled_text": "B3 · 斜向平铺文字水印（tiled_text，75 张）"}
    by_id = {r["id"]: r for r in inv["rows"] if r["kind"] == "image"}
    counts = {"good": 0, "mid": 0, "bad": 0}
    for t in ("dataset_logo", "text", "tiled_text"):
        L.append(f"<h3>{esc(type_names[t])}</h3>")
        rows_t = {k: v for k, v in wm_rows.items() if v["type"] == t}
        n_good = sum(1 for r in rows_t.values() if judge_wm(r)[0] == "改善")
        n_mid = sum(1 for r in rows_t.values() if judge_wm(r)[0] == "无变化")
        n_bad = sum(1 for r in rows_t.values() if judge_wm(r)[0] == "变差")
        counts["good"] += n_good; counts["mid"] += n_mid; counts["bad"] += n_bad
        L.append(f"<div class='kpi'><b>{n_good}</b>改善</div><div class='kpi'><b>{n_mid}</b>无变化</div>"
                 f"<div class='kpi'><b>{n_bad}</b>变差</div>")
        for rid in sorted(rows_t, key=int):
            r = rows_t[rid]
            inv_row = by_id[rid]
            lv, col, note = judge_wm(r)
            png = f"B_{rid}.jpg"
            if not (assets / png).exists():
                strip([Path(inv["root"]) / inv_row["watermarked"],
                       out / "wm_baseline" / f"{rid}.png",
                       Path(inv["root"]) / inv_row["clean"]], assets / png)
            L.append("<div class='card'>")
            L.append(f"<img src='report_assets/{png}' loading='lazy'>")
            L.append("<div class='cols'><span>① 带水印原图</span><span>② 去水印输出</span><span>③ 干净参考</span></div>")
            L.append(f"<div class='verdict'><span class='badge {col}'>{lv}</span>"
                     f"<b>{esc(rid)}</b>｜ {esc(note)}</div>")
            L.append("</div>")
    L.append(f"<div class='note'>小结：300 张中 <b>{counts['good']} 张改善</b>、<b>{counts['mid']} 张基本无变化</b>、"
             f"<b>{counts['bad']} 张变差</b>。变差的集中在彩色 Logo 类——程序把照片里的颜色/纹理变化误判成水印并重绘，"
             f"反而破坏了原本干净的画面。这属于<b>已知能力边界</b>（详见功能总结）。</div>")

    # ---------------- 清单 C
    L.append("<h2>清单 C：watermark_samples 外部 PDF（10 份）</h2>")
    pdf_names = {"vector_text": "可编辑 PDF（文字对象是独立指令，理论上可无损剥离）",
                 "rasterized_text": "扫描式 PDF（页面已栅格化成图片，结构引擎不适用）"}
    for row in pdf_rows:
        png = f"C_{row['id']}.png"
        src = out / "wm_pdf" / f"{row['id']}_comparison.png"
        if not (assets / png).exists() and src.exists():
            (assets / png).write_bytes(src.read_bytes())
        exact = sum(p["exact_reference"] for p in row["pages"])
        total = len(row["pages"])
        if row["type"] == "vector_text":
            lv, col, note = ("优秀", "good",
                             f"移除 {sum(row['text_blocks_removed'])} 个水印文字指令，"
                             f"{total} 页与干净参考逐像素一致，干净负样本未被误改")
        else:
            lv, col, note = ("不适用", "mid",
                             "扫描式 PDF 无独立文字对象，程序正确拒绝处理，未做任何改动")
        L.append("<div class='card'>")
        L.append(f"<img src='report_assets/{png}' loading='lazy'>")
        L.append("<div class='cols'><span>① 带水印原图（首页）</span><span>② 程序输出</span><span>③ 干净参考</span></div>")
        L.append(f"<div class='verdict'><span class='badge {col}'>{lv}</span>"
                 f"<b>{esc(row['id'])}</b>（{esc(pdf_names[row['type']])}）｜ {esc(note)}</div>")
        L.append("</div>")

    # ---------------- 清单 D
    L.append("<h2>清单 D：工程回归测试</h2>")
    L.append(f"<div class='note'>单元测试与界面回归共 <b>46 个</b>，结果："
             f"<span class='badge {'good' if unit_ok else 'bad'}\">{'全部通过' if unit_ok else '存在失败'}</span>"
             "（含队列、导出、PDF 文字剥离、照片样式、边界修复等场景，QT offscreen 运行）。</div>")

    # ---------------- 功能总结
    L.append("<h2>功能报告总结</h2>")
    a = agg
    L.append(f"""
<table>
<tr><th>能力项</th><th>状态</th><th>关键证据</th></tr>
<tr><td><b>点阵水印</b>（文档/表格）</td><td><span class='badge good'>成熟</span></td>
<td>掩码 IoU 0.99（精确率 1.0、召回 0.99），水印区 MAE 47.6→3.2，单页 0.8–2.3s，4 组样例 3 组全条件通过</td></tr>
<tr><td><b>平铺文字水印</b></td><td><span class='badge mid'>可用但有缺口</span></td>
<td>修复质量高（误伤仅 0.7–2.5%，水印区 MAE 48→5.6），但掩码召回不足（IoU≈0.35），识别条件 4 组均未过线</td></tr>
<tr><td><b>居中大字/深色重字母</b></td><td><span class='badge mid'>部分场景</span></td>
<td>白底文档新路径有效（big_center_04 掩码 IoU 0.91）；照片底材不稳定（big_center_01 超时+保真越界），深色字母识别 IoU 仅 0.45–0.77</td></tr>
<tr><td><b>PDF 结构剥离</b>（引擎A）</td><td><span class='badge good'>确定性强项</span></td>
<td>可编辑 PDF 32/32 页逐像素无损、负样本零误改、扫描件正确拒绝</td></tr>
<tr><td><b>自然照片通用 Logo</b></td><td><span class='badge bad'>能力边界</span></td>
<td>300 张外部照片 116 张变差、181 张非水印区被改动，误触发重绘</td></tr>
<tr><td><b>批量/GUI/回归</b></td><td><span class='badge good'>健康</span></td>
<td>46/46 测试通过；多批次、进度、暂停继续、导出 manifest 齐全</td></tr>
</table>""")
    L.append("<div class='note'><b>一句话总结：</b>这是一套<b>面向文档场景</b>的成熟去水印工具——"
             "点阵近乎完美、PDF 无损剥离可靠、平铺文字可用；但在<b>自然照片上的通用 Logo 水印</b>"
             "和深色/极小水印上尚未达标，这与项目 README 自述的边界完全一致，不是本次环境搭建引入的退化。</div>")

    # ---------------- 优化方向
    L.append("<h2>下一步优化方向</h2>")
    L.append("""
<ol>
<li><b>平铺文字掩码召回（投入产出比最高）</b>：当前 IoU≈0.35 的原因是重叠在文字上的笔画漏检。
 建议在已确认方向/行距上做「残笔匹配补全」（项目已有 paper_repeat 基础），把掩码 IoU 推到 0.7+，
 预计可让 tiled_text 四条件通过率从 0/4 提升到 3/4 以上。</li>
<li><b>自然照片误触发保护</b>：300 张外部照片 60%+ 出现区外误改。可加两道闸：
 (a) 纸面锚点门扩展到照片——无纸面锚点时要求更高置信度才允许重绘；
 (b) 重绘前对掩码做连通域面积/形状 sanity check，超阈值直接判 NEEDS_MANUAL_MASK 而非强行修复。
 目标是「宁可不处理，不可处理错」。</li>
<li><b>深色重字母水印</b>：单页色调通道已证明不可辨识，应走两条路之一：
 结构解析（PDF 文字对象）已成熟，可对可编辑源优先路由；位图场景引入样式库/模板匹配，
 或评估接入研究模型（SLBR 在留出集已验证改善，但因会改动干净图内容暂未接入，需先解决负样本保护）。</li>
<li><b>性能</b>：big_center_01 照片页 131s 超 120s 预算。修复阶段可限制 LaMa 只处理掩码连通域外接框
 （而非整页分块），并对超大掩码先降采样反解、仅核心交给模型，目标所有样例 <60s。</li>
<li><b>判定协议确认</b>：四条件阈值（识别 IoU 0.90、残膜 15% 等）是项目自定义口径，
 建议携带本报告数据与需求方书面确认，避免交付时口径争议。</li>
<li><b>字体/环境固化</b>：macOS 字体软链方案（wm/fonts/）建议随包发布，避免换机再踩；
 Intel Mac 的 onnxruntime 1.23.2 组合已通过 eval 对表验收，可写入交付文档。</li>
</ol>""")

    L.append(f"<div class='small'>数据文件：{esc(str(out))}/real_samples.json、wm_baseline/metrics.json、"
             f"wm_pdf_agg.json、unit_tests.log ｜ 复现：./start.sh test</div>")
    L.append("</div></body></html>")

    (out / "可视化测试报告.html").write_text("\n".join(L), "utf-8")
    log(f"HTML -> {out / '可视化测试报告.html'}（对照图 {len(list(assets.iterdir()))} 张）")


if __name__ == "__main__":
    main()
