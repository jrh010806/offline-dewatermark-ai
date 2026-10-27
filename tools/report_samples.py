"""Build a local evidence report from saved inference runs; performs no inference."""
from __future__ import annotations

# Historical data was retired on 2026-10-26; shared imported helpers remain usable.
if __name__ == "__main__":
    raise SystemExit('该脚本依赖的历史样本已退役。当前测试请使用 .venv/bin/python tools/run_min_regression.py --suite all；历史结果仍在 out/，不再覆盖或重跑。')
import base64
from collections import defaultdict
from html import escape
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out/new_samples'


def read(path):
    return json.loads(path.read_text())


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def table(headings, rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headings)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def thumb(path):
    with Image.open(path) as src:
        src = ImageOps.contain(src.convert('RGB'), (420, 450))
        buf = BytesIO(); src.save(buf, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


def gallery(title, columns):
    return '<article><h3>'+escape(title)+'</h3><div class="gallery">'+''.join(
        '<figure><img src="'+thumb(path)+'"><figcaption>'+escape(label)+'</figcaption></figure>'
        for label, path in columns)+'</div></article>'


def main():
    inv = read(OUT/'inventory.json'); base = Path(inv['root'])
    baseline = read(OUT/'baseline/metrics.json')
    model = read(OUT/'slbr_validation/metrics.json')
    pdfs = read(OUT/'pdf_outputs/metrics.json')
    native = read(OUT/'native_regression/after/metrics.json')
    previous = read(ROOT/'out/center_refinement/after/metrics.json')
    assert len(native) == 16 and not any('error' in x for x in native.values()), 'Native run incomplete'
    gate = read(OUT/'paper_gate.json')
    assert len(gate) == 300 and not any(r['selected'] or r['clean_selected'] for r in gate)
    assert len(baseline) == 300 and len(model) == 57
    by_id = {r['id']: r for r in inv['rows'] if r['kind']=='image'}
    unchanged, changed = [], []
    for stem, row in native.items():
        if 'source_sha256' in previous[stem]:
            assert row['source_sha256'] == previous[stem]['source_sha256']
        else:
            assert row['input'] == previous[stem]['input'] and row['shape'] == previous[stem]['shape']
        a = np.array(Image.open(OUT/f'native_regression/after/{stem}.png'))
        b = np.array(Image.open(ROOT/f'out/center_refinement/after/{stem}.png'))
        (unchanged if np.array_equal(a, b) else changed).append(stem)
    sections = ['<h1>水印去除：新样本回归与本轮改进</h1><p>2026-10-23 · 本地离线测试 · 输入素材未修改</p>',
        '<p class="notice">已接入：单图点阵文档修复、PDF 独立透明文字移除。照片专用模型仍为研究实验，没有替换桌面程序默认处理。</p>',
        '<h2>数据与评估口径</h2><p>300 对图片、10 对 PDF，共 620 个配对文件已核对 SHA256、尺寸/页数。图片为 150 个数据集合成 Logo、75 个单处文字、75 个平铺文字；均为 256×256。哈希固定划分 243 张开发集和 57 张项目留出集。预训练模型是否见过相关上游底图尚未审计，因此不能据此宣称通用基准成绩。</p>',
        '<p>推理阶段只读带水印输入；干净参考和人工 mask 只用于推理后的评分。区域 MAE 按参考水印区域扩张 2 像素后固定计算，单位 0–255，越低越好；不得当成去除成功率。原有 16 张回归使用其历史固定区域口径，数值不与新 300 张横向混算。</p>',
        '<h2>已接入：点阵文档</h2>']
    targets = ['dot_grid_01', 'dot_grid_03', 'dot_grid_04']
    sections.append(table(['原尺寸样例', '上版区域 MAE', '新版区域 MAE', '本次耗时', '请求范围外像素'], [
        [stem, f"{previous[stem]['output']['mae_watermark_roi']:.2f}",
         f"{native[stem]['output']['mae_watermark_roi']:.2f}",
         f"{native[stem]['elapsed_ms']/1000:.2f} 秒", '逐像素未改' if native[stem]['outside_mask_unchanged'] else '发生变化'] for stem in targets]))
    sections.append('<p>从单张图内的周期、相位、纸面与墨色估计透明叠加，交叠处反解；远离暗色笔画/彩色印章的小点才清理纸面残影。三个样例的暗字抬亮超过 50 级比例均为 0；这不是 OCR 文字识别正确率。</p>')
    for stem in targets:
        sections.append(gallery(stem, [('原图', ROOT/f'data/real_samples/jpg/{stem}.jpg'),
            ('上版', ROOT/f'out/center_refinement/after/{stem}.png'),
            ('新版', OUT/f'native_regression/after/{stem}.png'),
            ('参考底图（仅评分）', ROOT/f'data/real_samples/clean/{stem}.png')]))
    sections.append('<h2>已接入：可编辑 PDF 透明文字</h2><p>5 份可编辑 PDF 共 32 页，输出在同一 PyMuPDF 渲染器、100 DPI 下逐页与干净参考完全一致；10 份干净 PDF 的原始字节全部未被重写。只省略已确认文字块的文字显示指令，不做整块矩形擦除；混合普通文字的块拒绝删除。该条件仍是透明度/几何启发式，结果需要复核，Form XObject 内文字暂不覆盖。</p>')
    sections.append(table(['PDF', '类型/分组', '页数', '移除文字块', '与参考逐像素一致页', '解析/改写耗时'], [
        [r['id'], r['type']+' / '+r['split'], len(r['pages']), sum(r['text_blocks_removed']),
         sum(p['exact_reference'] for p in r['pages']), f"{r['strip_ms']} ms"] for r in pdfs]))
    sections.append('<p>另 5 份扫描式 PDF 共 24 页均没有触发对象删除，继续图像流程。它们与参考原件包含额外栅格化/JPEG 差异，本表不把原件像素差异计为去水印成败。桌面批次、单页、整文件和流式入口均能使用透明文字路径，并保留逐页进度。</p>')
    for stem in ['0003', '0004']:
        sections.append(gallery('PDF '+stem+'：左原图 / 中输出 / 右参考', [('', OUT/f'pdf_outputs/{stem}_comparison.png')]))
    sections.append('<h2>照片：当前能力缺口与专用模型实验</h2><p>旧流程 300 张基线全部完成，只有 18 张固定水印区域误差比原输入降低；不能把“300 张处理完成”解释为“300 张处理成功”。新点阵文档路径在这 300 张照片及对应干净图上全部拒绝触发，因此不会凭这次点阵更新声称照片质量提升。</p>')
    groups = defaultdict(list)
    for stem, row in model.items():
        groups[row['type']].append((row, baseline[stem]))
    rows = []
    for kind, values in sorted(groups.items()):
        positives = [r for r, _ in values]
        rows.append([kind, len(values), f"{mean(positives, 'input_roi'):.2f}",
            f"{np.mean([b['output_roi'] for _, b in values]):.2f}", f"{mean(positives, 'output_roi'):.2f}",
            sum(r['output_roi'] < b['output_roi'] for r, b in values),
            sum(r['negative_changed_gt4'] > .002 for r, _ in values),
            f"{np.median([r['ms'] for r, _ in values]):.0f} ms"])
    sections.append(table(['项目留出类型', '张数', '输入 ROI MAE', '旧流程', 'SLBR 实验', '好于旧流程张数', '干净图明显修改张数', 'MPS 推理中位数'], rows))
    sections.append('<p>“明显修改”固定定义：干净图中，超过 0.2% 像素的任一 RGB 通道变化超过 4 级，57 张中共 29 张。该计数不代表 29 张整图不可用，但暴露了合法文字/Logo 误删风险。单次耗时是本机原尺寸 256×256、已加载权重的 MPS 网络推理，不含启动和文件 IO，不能推广到高清图片或 CPU。</p>')
    sections.append('<p>模型：<a href="https://github.com/bcmi/SLBR-Visible-Watermark-Removal">官方 SLBR</a>，固定提交 47c665f1855ab6624cd52b28cefa797a9c8b96f7，官方权重 SHA256 be5c0a89513803c8ce56b1768e0c6296900f6ef76df8cbda12900324f9b1caf9（详见 provenance 文件）。正例改进未解决干净图误删，两轮开发集校准未达到要求，校准权重均未部署。研究用 TorchScript 在本机 Python 3.14 会提示兼容性警告，也不应直接作为发布后端。</p>')
    for kind, values in sorted(groups.items()):
        r = sorted((r for r, _ in values), key=lambda r:r['id'])[0]
        stem = r['id']; item = by_id[stem]
        sections.append(gallery(kind+' '+stem+'（按 ID 首个留出样例）', [('原图', base/item['watermarked']),
            ('旧流程', OUT/f'baseline/{stem}.png'), ('研究模型', OUT/f'slbr_validation/{stem}.png'),
            ('参考（仅评分）', base/item['clean'])]))
    for stem in ['0123', '0061']:
        item = by_id[stem]
        sections.append(gallery('误删证据：干净图 '+stem, [('干净输入', base/item['clean']),
            ('模型输出（未加水印的输入也被修改）', OUT/f'slbr_validation/{stem}_negative.png')]))
    sections.append('<h2>原有样式回归与后续重点</h2><p>16 张原尺寸全部复测。与上版逐像素一致：'+str(len(unchanged))+' 张；输出变化：'+escape(', '.join(changed))+'。全部 16 张请求修复范围外像素保持原样。big_center_01/03 仍没有有效改善，部分高清图超时；旧斜向文字与边缘修复维持上版结果。</p><p>后续优先研究照片水印与合法内容的区分，用干净图负例约束定位，再验证高清修复。扫描式 PDF 要走图像路径。已接入的文档改善可以使用，但整套程序尚未达到商用网站通用效果。</p>')
    sections.append('<p class="links">原始指标：<a href="baseline/metrics.json">300 张基线</a> · <a href="slbr_validation/metrics.json">57 张研究留出</a> · <a href="slbr_validation/provenance.json">模型与设备</a> · <a href="native_regression/after/metrics.json">16 张回归</a> · <a href="pdf_outputs/metrics.json">PDF 逐页指标</a></p>')
    css = 'body{font:16px/1.65 system-ui,sans-serif;color:#1a2938;background:#f4f6f8;max-width:1500px;margin:32px auto;padding:0 28px}h1{font-size:30px}h2{margin-top:42px}table{width:100%;border-collapse:collapse;background:white;font-variant-numeric:tabular-nums}td,th{padding:10px 12px;border:1px solid #dbe1e8;text-align:left}th{background:#e8eef4}article{background:white;padding:18px;margin:20px 0;border-radius:10px}.gallery{display:flex;gap:12px;align-items:flex-start;overflow:auto}figure{margin:0;flex:1;min-width:180px;text-align:center}img{max-width:100%;max-height:470px;object-fit:contain}figcaption{font-size:14px;color:#526273}.notice{padding:18px;background:#deefe5;border-left:5px solid #278257}.links{margin:40px 0}a{color:#126caa}'
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>水印样本回归报告</title><style>'+css+'</style><body>'+''.join(sections)+'</body></html>'
    (OUT/'report.html').write_text(html, 'utf-8')
    summary = dict(native_unchanged=unchanged, native_changed=changed,
                   native_outside_unchanged=all(r['outside_mask_unchanged'] for r in native.values()),
                   pdf_exact_pages=sum(p['exact_reference'] for r in pdfs if r['type']=='vector_text' for p in r['pages']),
                   clean_pdf_untouched=all(r['clean_negative_untouched'] for r in pdfs),
                   model_clean_negative_events=sum(r['negative_changed_gt4']>.002 for r in model.values()),
                   model_research_only=True)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(OUT/'report.html'); print(summary)


if __name__ == '__main__':
    main()
