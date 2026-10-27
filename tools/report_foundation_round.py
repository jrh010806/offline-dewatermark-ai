"""Build the plain-language report from the saved, paired regression outputs."""
from collections import Counter
from html import escape
from pathlib import Path
import hashlib
import json
import re

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"out/foundation_round_20261026"
DATA = ROOT/"data/watermark_training_v1"
REPORT = ROOT/"docs/基础功能修复测试报告.html"


def main():
    before = json.loads((OUT/"before/all_metrics.json").read_text())["results"]
    payload = json.loads((OUT/"after/all_metrics.json").read_text())
    after = payload["results"]
    selection = {r["id"]: r for r in json.loads((OUT/"selection.json").read_text())}
    assert len(before) == len(after) == 88, (len(before), len(after))
    assert {(r["id"], r["page"]) for r in before} == {(r["id"], r["page"]) for r in after}
    old = {(r["id"], r["page"]): r for r in before}
    for r in after:
        assert r["input_sha256"] == old[r["id"], r["page"]]["input_sha256"]
    for name, digest in payload["code_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    tests = (OUT/"tests_after.log").read_text()
    matched = re.search(r"Ran (\d+) tests", tests)
    assert matched and "\nOK\n" in tests and "FAILED" not in tests
    test_count = int(matched[1])
    counts = {}
    rows = []
    for kind, suite, label in [
        ("image", "original", "原先的22张带水印图片"),
        ("image", "extra", "另外检查的20张带水印图片"),
        ("pdf", "original", "原先的22页带水印PDF"),
        ("pdf", "extra", "另外检查的4页带水印PDF"),
    ]:
        key = kind+"_"+suite
        a = [r for r in before if r["kind"] == kind and r["suite"] == suite and not r["is_negative"]]
        b = [r for r in after if r["kind"] == kind and r["suite"] == suite and not r["is_negative"]]
        n0, n1 = sum(r["passed"] for r in a), sum(r["passed"] for r in b)
        counts[key] = dict(before=n0, after=n1, total=len(b))
        rows.append(f"<tr><td>{label}</td><td>{n0}/{len(a)} 通过</td><td><b>{n1}/{len(b)} 通过</b></td></tr>")
    clean0 = [r for r in before if r["is_negative"]]
    clean1 = [r for r in after if r["is_negative"]]
    safe0, safe1 = sum(r["negative_unchanged"] for r in clean0), sum(r["negative_unchanged"] for r in clean1)
    counts["clean"] = dict(before=safe0, after=safe1, total=len(clean1))
    rows.append(f"<tr><td>20份干净样本（含正常标识、签名）</td><td>{safe0}/20 保持原样</td><td><b>{safe1}/20 保持原样</b></td></tr>")
    counts["unit_tests"] = test_count
    (OUT/"summary.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2))
    assets = OUT/"report_assets"; assets.mkdir(exist_ok=True)

    def example(sid, page, title, text):
        r = selection[sid]
        if r["kind"] == "image":
            input_path = DATA/r["paths"]["watermarked"]
            alpha_path = DATA/r["paths"]["alpha"]
        else:
            input_path = DATA/r["pages"][page-1]["paths"]["watermarked"]
            alpha_path = DATA/r["pages"][page-1]["paths"]["alpha"]
        name = sid+f"__p{page}"
        paths = [input_path, OUT/"before"/f"{name}__output.png", OUT/"after"/f"{name}__output.png"]
        # The same scoring annotation defines a zoom crop only for presentation;
        # it is never fed to the application detector or repairer.
        images = [Image.open(p).convert("RGB") for p in paths]
        if r["kind"] == "pdf" and r["subtype"] == "unicode_text":
            import pymupdf as fitz
            with fitz.open(DATA/r["paths"]["clean"]) as clean_pdf:
                pix = clean_pdf[page-1].get_pixmap(dpi=96, colorspace=fitz.csRGB)
                clean = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            assert np.array_equal(np.asarray(images[2]), clean), (sid, "PDF content changed")
        mask = Image.open(alpha_path).convert("L").resize(images[1].size)
        bbox = mask.getbbox()
        if sid.endswith("normal_logo_signature_negative"):
            # The top table area contains the legitimate label removed before.
            w, h = images[1].size; bbox = (int(.05*w), int(.13*h), int(.55*w), int(.25*h))
        if bbox:
            x0, y0, x1, y1 = bbox
            margin = max(15, int(min(x1-x0, y1-y0)*.25))
            w, h = images[1].size
            box = (max(0, x0-margin), max(0, y0-margin), min(w, x1+margin), min(h, y1+margin))
        else:
            box = (0, 0, *images[1].size)
        figures = []
        for i, (im, label) in enumerate(zip(images, ["原来的图片", "修改前的结果", "这次的结果"])):
            if im.size != images[1].size:
                # PDF input is shown at its stored preview resolution; map the
                # shared crop coordinates without changing inference/scoring.
                sx, sy = im.width/images[1].width, im.height/images[1].height
                crop = im.crop(tuple(round(v*(sx if k%2 == 0 else sy)) for k, v in enumerate(box)))
            else:
                crop = im.crop(box)
            path = assets/f"{name}_{i}.png"; crop.save(path)
            url = "../"+str(path.relative_to(ROOT))
            figures.append(f'<figure><a href="{url}" target="_blank"><img src="{url}" alt="{label}"></a><figcaption>{label}</figcaption></figure>')
        return f'<article><h3>{title}</h3><p>{text}</p><div class="pictures">{"".join(figures)}</div></article>'

    examples = [
        example("photo_008__local_text", 1, "照片上的局部文字：这张通过了", "新增加的照片文字识别在这张图上有效。这里只代表这个样例。"),
        example("document_025__pdf_unicode_text", 1, "PDF中文水印：现在能一起去掉", "过去只删掉斜向英文，留下了中文；这次两部分都去掉了，正文与干净原稿一致。"),
        example("photo_033__center_white", 1, "照片上的居中大字：仍然失败", "这类照片还没有解决，不能因为PDF改善，就说图片去水印已经可用了。"),
    ]
    more = [
        example("document_025__pdf_independent_image", 1, "PDF透明图案：这次通过", "现在只移除叠在正文上的那次图片绘制，保留其他位置使用的图片。"),
        example("document_027__normal_logo_signature_negative", 1, "正常表格编号：现在会保留", "原先把表格里的正常编号擦掉了；这次增加了表格保护。"),
    ]
    reasons = {"no_watermark_detected": "没找到水印", "residual_watermark": "水印仍有残留",
               "recognition_regression": "内容检查未通过", "fidelity_breach": "改动超出允许范围",
               "out_of_time_budget": "用时超过限制", "needs_manual_mask": "需要人工指定位置"}
    detail_rows = []
    for r in after:
        prev = old[r["id"], r["page"]]
        def state(x):
            if x["is_negative"]:
                return "保持原样" if x["negative_unchanged"] else "被误改"
            return "通过" if x["passed"] else "未通过"
        description = "、".join(reasons.get(x, x) for x in r["failures"]) if not r["is_negative"] else ""
        if state(r) == "通过": description = "按原有标准通过"
        detail_rows.append(f'<tr><td>{escape(r["label"])}<small>{escape(r["id"])} · 第{r["page"]}张/页</small></td><td>{state(prev)}</td><td>{state(r)}</td><td>{description}</td></tr>')
    orig = counts["image_original"]; pdf = counts["pdf_original"]; extra = counts["image_extra"]
    report = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>基础功能修复测试报告</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f5f7;color:#202a35;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}main{{max-width:1100px;margin:auto;padding:32px 24px 70px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:23px;margin-top:36px}}h3{{font-size:18px;margin:0}}.date,small{{color:#66717e;font-size:13px}}.conclusion{{padding:20px 24px;background:#fff2d8;border-left:5px solid #bb7918;border-radius:8px;font-size:19px}}section,article{{background:white;border:1px solid #dde1e6;border-radius:10px;padding:22px;margin:20px 0}}article p{{margin:6px 0 16px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;border-bottom:1px solid #e4e7eb;padding:13px 10px;vertical-align:top}}th{{background:#f2f4f7}}.scroll{{overflow:auto}}small{{display:block;overflow-wrap:anywhere}}.pictures{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}figure{{margin:0}}img{{display:block;width:100%;height:230px;object-fit:contain;background:#f1f3f5;border:1px solid #ddd}}figcaption{{text-align:center;font-size:13px;margin-top:7px}}details{{margin:20px 0}}summary{{cursor:pointer;color:#245d92;font-weight:600}}a{{color:#245d92}}.note{{color:#5a6673;font-size:14px}}li{{margin:6px 0}}@media(max-width:650px){{main{{padding:18px 12px}}.pictures{{grid-template-columns:1fr}}img{{height:auto;max-height:380px}}section,article{{padding:16px}}h1{{font-size:26px}}}}
</style></head><body><main><div class="date">2026年10月26日 · 第一轮基础修复</div><h1>PDF有改善，图片仍未过关</h1>
<p class="conclusion"><b>原先22张图片：{orig['before']}→{orig['after']}张通过；原先22页带水印PDF：{pdf['before']}→{pdf['after']}页通过；干净样本：{safe0}→{safe1}份保持原样（共20份）。</b><br>图片只有少量样例改善，目前还不能算基础图片去水印已经解决。</p>
<section><h2 style="margin-top:0">这次改了什么</h2><p>补了照片文字识别和细笔画修补；修了PDF漏掉中文、漏掉透明图案的问题；增加页脚横线、图表和表格编号保护，减少误删。</p><div class="scroll"><table><thead><tr><th>检查内容</th><th>修改前</th><th>修改后</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p><b>程序检查：{test_count}/{test_count}项通过。</b>这只说明已检查的程序行为正常，不等于所有图片都能去水印。</p><p class="note">更正：旧报告把“检查通过”当成“完全没改动”，这不准确。本轮对干净样本单独核对，旧版本20份里有10份被误改；修改后20份都保持原样。</p></section>
<section><h2 style="margin-top:0">为什么图片还没做好</h2><p>一部分水印仍然找不到；另一部分即使找到，修补后也有残留或细节变化。本轮开始时的辅助诊断中，即使给出准确水印位置，原有修补方法也只有4/22张通过，说明识别和修补都需要继续改。</p><p><b>下一步仍应集中解决照片上的居中大字、重复文字，暂时不需要你继续补样本。</b>本轮不能作为通用去水印已经达标的证明。</p></section>
<h2>看三个实际例子</h2><p class="note">三张图使用相同位置放大，点击可查看清晰图。</p>{''.join(examples)}
<details><summary>再看透明图案和正常表格的对照</summary>{''.join(more)}</details>
<details><summary>查看全部88张/页的逐项结果</summary><div class="scroll"><table><thead><tr><th>样本</th><th>修改前</th><th>修改后</th><th>结果说明</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div></details>
<section><h2 style="margin-top:0">这些数字怎么来的</h2><p class="note">共检查42张带水印图片、26页带水印PDF、16张干净图片和4页干净PDF。原先22张带水印图片共用1张底图；另外20张来自5张底图，其中部分属于已有训练材料，用于开发检查，不能当成独立成功率。PDF来自2份底稿。本轮使用已有合成样本，没有把尚未整理的真实扫描图混入统计。</p><p class="note">修改前后使用相同输入；带水印样本沿用原有通过标准，干净样本另外核对是否完全没改。原图和水印位置只用于事后打分，辅助诊断另算，没有提供给正式处理流程。最终保留的测试集没有用于调参。图像处理结果与统计均保存在本地，没有上传样本。</p><p class="note"><a href="../out/foundation_round_20261026/summary.json">简要统计</a> · <a href="../out/foundation_round_20261026/before/all_metrics.json">修改前记录</a> · <a href="../out/foundation_round_20261026/after/all_metrics.json">修改后记录</a> · <a href="../out/foundation_round_20261026/tests_after.log">程序检查记录</a></p></section>
</main></body></html>'''
    REPORT.write_text(report)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
