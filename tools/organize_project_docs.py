"""Synchronize the local document status and build its HTML entry point."""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def update(name, replacements):
    path = DOCS / name
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text)


def main():
    update('水印覆盖与竞品评测.html', [
        ('自动补样已完成 · 人工采集待补', '首批人工样本已整理 · 自有基线已完成'),
        ('效果评测待执行', '自有基线已完成 · 竞品待测'),
        ('版本 1.2', '版本 1.3'),
        ('所有正样本水印为项目合成；实扫、翻拍、真实手写及跨软件业务文件仍在人工补充清单中。', 'v1 正样本水印为项目合成。本轮人工增量另有 121 条训练记录、3 组近似配对、5 条参考登记；M03 已收 2 份底稿共 4 页扫描图，待配准及掩码复核。'),
        ('效果评测统一记为待测。已有目录中的历史指标没有纳入本轮结论；本次仅核验样本文件和分类清单，未运行本工具或竞品的去水印推理。', '自有工具已经完成首轮基线及安全修正复测，详情见<a href="首轮基线评测.html">全部输入与输出对比</a>。竞品尚未执行同样本测试，横向比较矩阵仍为待测；自有小样本基线不能替代正式对照。'),
        ('自动基础样本已建，人工待补', '自动基础样本已建；首批人工样本已整理'),
        ('正式清单、来源、标注、分区与整包检查已完成；保留 M01—M06 人工项', '正式清单和首批人工增量已整理；M03 扫描图已收到，配准及标注待完成，细分多样性继续扩充'),
        ('<td>竞品与本工具效果</td><td><span class="status pending">全部待测</span></td><td>没有本轮同样本可复核结果</td>', '<td>竞品与本工具效果</td><td><span class="status pending">自有基线已测；竞品待测</span></td><td><a href="首轮基线评测.html">已保存自有输入、输出和指标</a>；横向比较尚未执行</td>'),
        ('当前没有运行去水印效果评测。', '上述历史库存盘点没有运行去水印推理；后续自有基线结果见<a href="首轮基线评测.html">首轮基线评测</a>。'),
        ('本轮尚未向竞品上传样本，也未执行本工具推理。历史记录未作为本轮效果依据，当前无可报告的质量分数或胜率。', '本轮尚未向竞品上传样本。自有工具已有小规模基线结果：图片正例 0/22，通过安全保留的图片负例 2/2；PDF 页面 8/24 通过（含 2 页无水印负例）。完整输出见<a href="首轮基线评测.html">首轮基线评测</a>。下表保留为正式同样本横向测试计划，尚无可报告的对竞品胜率。'),
        ('本工具与竞品待测项', '正式横向比较待测项（自有基线另列）'),
        ('M03 实际纸张扫描暂缺', 'M03 已收 4 页扫描图，待配准和水印区域标注复核'),
        ('M04 翻拍角度及跨软件文件（M05）仍待补', 'M04 已有手写翻拍；更多角度与跨软件文件（M05）仍可扩充'),
    ])
    status_note = ('<strong>截至 2026-10-26：</strong>原 v1 为 2,432 条记录；人工增量为 121 条已标注训练记录，'
                   '另有 3 组近似配对和 5 条参考登记。M03 已收 2 份底稿共 4 页扫描图，原件和源 PDF 已归档，'
                   '配准与水印区域标注复核待做。以下数量目标是后续扩充目标，收件状态以'
                   '<a href="样本整理与入库.html">样本整理与入库</a>为准。')
    path = DOCS / '训练数据集说明.html'
    text = path.read_text()
    text = re.sub(r'<p class="notice" id="manual-received-status">.*?</p>',
                  '<p class="notice" id="manual-received-status">' + status_note + '</p>', text, flags=re.S)
    text = text.replace('<th>首轮目标</th>', '<th>后续扩充目标（不是当前缺件数）</th>')
    path.write_text(text)

    names = ['水印覆盖与竞品评测.html', '训练数据集说明.html', '样本整理与入库.html', '首轮基线评测.html']
    nav = ('<!-- project-doc-nav:start --><a href="项目文档总览.html" '
           'style="display:block;padding:9px 10px;margin:0 0 14px;border:1px solid #8296af;border-radius:7px;'
           'color:inherit;text-decoration:none;font-size:14px">← 项目文档总览</a><!-- project-doc-nav:end -->')
    for name in names:
        p = DOCS / name
        text = re.sub(r'<!-- project-doc-nav:start -->.*?<!-- project-doc-nav:end -->', '', p.read_text(), flags=re.S)
        text = re.sub(r'(<aside\b[^>]*>)', lambda m: m[1] + nav, text, count=1)
        p.write_text(text)

    summary = json.loads((ROOT / 'data/watermark_training_increment_v1_1/build_summary.json').read_text())
    im = json.loads((ROOT / 'out/baseline_20261026/metrics.json').read_text())
    pdf = json.loads((ROOT / 'out/baseline_20261026/pdf_metrics.json').read_text())
    refs = [json.loads(x) for x in (ROOT / 'data/watermark_training_increment_v1_1/references.jsonl').read_text().splitlines() if x.strip()]
    scan_count = sum(r.get('task') == 'M03' for r in refs)
    assert summary['strong_training_records'] == 121 and scan_count == 4
    links = [
        ('数据集清单与清理.html', '数据集清单与清理记录', '当前仅保留 3 套工作数据集；旧 3 套历史样本及配套格式已退役，附清理记录。'),
        ('最小回归测试集.html', '每次修改必跑的最小回归集', '固定核心 36 个输入，覆盖已有 22 种图片样式和 11 种 PDF 子类及负例；另附 10 个真实来源扩展案例。'),
        ('水印分类样本与最新测试.html', '水印分类、全部样本与最新测试', '按类型列出全部 2,753 张/页样本和可用原图，集中展示最新 48 组完整测试图片及修改方案。'),
        ('水印覆盖与竞品评测.html', '水印覆盖与竞品评测', '12 个验收场景、处理难点、覆盖要求、7 款候选竞品与比较协议。'),
        ('训练数据集说明.html', '训练数据集说明', '原 v1 的构成、标注、来源与训练/验证/测试划分；扩充目标另列。'),
        ('样本整理与入库.html', '样本整理与入库', '首批人工素材的六类收件状态、121 条训练增量、近似配对及扫描图预览。'),
        ('首轮基线评测.html', '首轮基线评测', '已测的 24 条图片和 24 页 PDF 输入/输出、修改前后对照及逐条指标。'),
    ]
    cards = ''.join(f'<a class="card" href="{file}"><b>{title} ↗</b><span>{desc}</span></a>' for file, title, desc in links)
    base = '../data/watermark_training_increment_v1_1/'
    section_data = [
        ('overview', '当前进度', f'''<p class="eyebrow">图片与 PDF 可见水印项目 · 2026-10-26</p><h1>项目文档总览</h1>
<p class="lead">从这里查看覆盖范围、训练数据、人工样本入库和已完成的测试对比。</p>
<div class="stats"><div><b>12</b><span>基础验收场景</span></div><div><b>2,432</b><span>原 v1 记录</span></div><div><b>121</b><span>人工增量训练记录</span></div><div><b>4 页</b><span>新收纸张扫描图</span></div></div>
<p class="notice">首批材料已归档。M03 扫描图已收到，接下来完成配准与水印区域标注复核。竞品横向测试尚未执行。</p>'''),
        ('documents', '文档的用途', f'<div class="cards">{cards}</div>'),
        ('samples', '样本整理到哪一步', f'''<div class="table-wrap"><table><tr><th>集合</th><th>数量与计数单位</th><th>目前用途</th></tr>
<tr><td>原 v1</td><td>2,112 对图片 + 176 对 PDF 文档 + 144 份负例，共 2,432 条</td><td>保持既有分组；train 1,962、validation 235、test 235</td></tr>
<tr><td>人工训练增量</td><td>110 组合成图片正例 + 5 个负例 + 6 个原生 PDF 页面配对，共 121 条</td><td>单独保存，均属训练用途；PDF 按页计，不能直接与 v1 的文档数相加</td></tr>
<tr><td>近似配对</td><td>3 组：2 组手写 + 1 组醒图</td><td>已有配准参考，单列辅助使用</td></tr>
<tr><td>参考登记</td><td>5 条：1 张红色批注、4 页 M03 扫描</td><td>不由默认监督训练加载器读取；两张抖音图及登记已删除</td></tr></table></div>
<h3>六类人工材料</h3><table><tr><th>类型</th><th>接收与准备状态</th></tr>
<tr><td>M01 真实手写</td><td>已收 5 张；整理 2 组近似配对及 1 张单图</td></tr>
<tr><td>M02 设备 / 应用水印</td><td>醒图 1 组近似配对；两张抖音图及登记已删除</td></tr>
<tr><td>M03 实际纸张扫描</td><td>已收 document_004、document_006 共 4 页全能扫描王图片；已关联电子干净与带水印源稿，待配准及掩码复核</td></tr>
<tr><td>M04 实际纸张翻拍</td><td>已有手写翻拍照片，与 M01 交叉归类，不重复计数</td></tr>
<tr><td>M05 原生办公 PDF</td><td>1 份底稿、灰/黑两个变体，共 6 个带水印页面配对</td></tr>
<tr><td>M06 真实背景</td><td>5 张普通照片已制备训练版本；银行原件单独保存，未用作训练背景；人像仍可扩充</td></tr></table>
<p>六类均已有收件或交叉归类记录。每类有样本与各类都具备足量可训练数据，是两个不同进度。</p>
<p><a href="样本整理与入库.html#received">查看扫描图及六类材料明细 ↗</a></p>'''),
        ('results', '已完成的测试在哪里看', f'''<p>自有工具已测 {len(im['results'])} 条图片、{pdf['aggregate']['total']} 页 PDF。图片正例 {im['aggregate']['positive_passed']}/{im['aggregate']['positive_total']} 通过，干净负例 {im['aggregate']['negative_controls_passed']}/{im['aggregate']['negative_controls_total']} 安全保留；PDF {pdf['aggregate']['passed']}/{pdf['aggregate']['total']} 页通过（含无水印负例）。</p>
<p>这是小规模验证基线：22 种图片水印共用一张照片底图，PDF 每种子类只有一份文档。新收到的 4 页扫描图尚未计入这些结果。</p>
<p><a class="button" href="首轮基线评测.html#all-results">查看全部测试输入与结果图片</a></p>
<p><a href="首轮基线评测.html#changes">查看修改前后对照</a> · <a href="../out/baseline_20261026/metrics.json">图片原始指标</a> · <a href="../out/baseline_20261026/pdf_metrics.json">PDF 原始指标</a></p>'''),
        ('next', '接下来按什么顺序做', '''<ol><li><b>先整理扫描图标注：</b>用对应电子源稿做几何配准，投影水印区域，逐页检查正文交叠及纸张弯曲；记录是否适合像素配对或仅辅助训练。document_004/006 继承原来源分组。</li>
<li><b>定位现有失败并扩充验证抽样：</b>结合输入、输出和预测掩码判断检测与修复问题，覆盖不同底图、背景和来源组。</li>
<li><b>修改算法并复测：</b>在固定验证协议上比较修改前后，保留所有输入、输出、失败案例和指标。</li>
<li><b>执行竞品同样本比较：</b>明确工具版本、设置与导出条件，用同一批适用输入比较残留、内容损伤和耗时；最后在冻结测试集上验收。</li></ol>
<p>本次完成文档整理与状态同步；上述步骤是后续执行顺序。</p>'''),
        ('paths', '文件与数据路径', f'''<p>项目根目录：</p><pre>{html.escape(str(ROOT))}</pre>
<table><tr><th>内容</th><th>相对项目根目录的位置</th></tr>
<tr><td>本文档</td><td>docs/项目文档总览.html</td></tr><tr><td>原训练数据</td><td>data/watermark_training_v1/</td></tr>
<tr><td>人工训练增量</td><td>data/watermark_training_increment_v1_1/</td></tr>
<tr><td>4 页扫描原图</td><td>data/watermark_training_v1/manual_incoming/M03_实际纸张扫描/received/</td></tr>
<tr><td>已测输出与指标</td><td>out/baseline_20261026/</td></tr></table>
<p><a href="{base}README.txt">增量使用说明</a> · <a href="{base}manifest.jsonl">121 条训练清单</a> · <a href="{base}weak_pairs.jsonl">近似配对清单</a> · <a href="{base}references.jsonl">参考登记清单</a> · <a href="{base}qa/validation.json">数据校验记录</a></p>'''),
    ]
    toc = ''.join(f'<a href="#{sid}">{i:02d}　{title}</a>' for i, (sid, title, _) in enumerate(section_data, 1))
    sections = ''.join(f'<section id="{sid}"><h2>{title}</h2>{body}</section>' for sid, title, body in section_data)
    css = '''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f8;color:#243247;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}aside{position:fixed;inset:0 auto 0 0;width:240px;padding:30px 18px;background:#14253b;color:white;overflow:auto}aside strong{display:block;font-size:20px;padding:0 10px 22px}aside a{display:block;color:#d0ddec;text-decoration:none;padding:11px 10px;font-size:14px;border-radius:8px}aside a:hover{background:#294360}main{margin-left:240px;max-width:1420px;padding:32px 40px 70px}section{scroll-margin-top:22px;border:1px solid #e0e6ed;background:white;border-radius:14px;padding:30px;margin-bottom:22px}h1{font-size:36px;line-height:1.35;margin:10px 0 16px}h2{font-size:23px;margin:0 0 20px}h3{font-size:18px}.eyebrow{font-size:13px;color:#63758b}.lead{font-size:19px;color:#52647c}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:25px 0}.stats div{border-radius:10px;background:#eff5fb;padding:16px}.stats b{display:block;font-size:31px;color:#205e9d}.stats span{font-size:13px;color:#63758b}.notice{background:#edf6f1;border-left:4px solid #27856b;padding:15px 18px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{padding:22px;background:#f5f8fc;border:1px solid #dce5ef;border-radius:10px;text-decoration:none}.card b{display:block;color:#235b91;font-size:19px}.card span{display:block;color:#63758b;margin-top:10px;font-size:14px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}td,th{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #e1e6ed}th{background:#f1f5f9}td:first-child{min-width:145px}a{color:#205e9d}pre,td{overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f0f4f8;padding:18px;border-radius:8px;font-size:13px}.button{display:inline-block;background:#235f9a;color:white;border-radius:7px;padding:10px 18px;text-decoration:none}li{margin-bottom:14px}@media(max-width:850px){aside{width:200px}main{margin-left:200px;padding:20px}.stats,.cards{grid-template-columns:1fr 1fr}}@media(max-width:600px){aside{position:relative;width:auto;padding:18px}aside a{display:inline-block}main{margin:0;padding:12px}section{padding:20px}.cards{grid-template-columns:1fr}h1{font-size:28px}}@media print{aside{display:none}main{margin:0;padding:0}section{border:0}body{background:white}}'''
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>项目文档总览</title><style>{css}</style></head><body><aside><strong>水印工具 · 文档</strong>{toc}</aside><main>{sections}</main></body></html>'
    (DOCS / '项目文档总览.html').write_text(page)
    from sync_real_training_docs import sync
    sync()
    print('Updated 4 documents and docs/项目文档总览.html')


if __name__ == '__main__':
    main()
