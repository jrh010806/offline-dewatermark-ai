"""Freeze a small source-backed developer regression suite without altering splits."""
from __future__ import annotations
import collections
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT/'data/watermark_training_v1'
INC = ROOT/'data/watermark_training_increment_v1_1'
DEST = ROOT/'data/watermark_regression_min_v1'


def rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def jsave(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    if (ROOT/'config/regression_retirements.json').exists():
        raise SystemExit('历史初始化脚本已停用：当前固定集已按授权移除样本，不能重建旧46个输入。校验用 ./start.sh check-data；更新说明用 .venv/bin/python tools/organize_project_docs.py。')
    if (DEST/'manifest.jsonl').exists():
        raise SystemExit('Frozen suite already exists; validate it instead of overwriting.')
    DEST.mkdir(parents=True,exist_ok=True)
    validation=rows(V1/'splits/validation.jsonl')
    config=json.loads((V1/'build_summary.json').read_text())
    by={(r['kind'],r['subtype'],r['base_id']):r for r in validation}
    # Fixed selection across all nine already-held-out image base groups.
    preferred=['photo_033','document_025','document_026','document_027','document_028',
               'charts_013','charts_014','textures_013','textures_014']
    selected=[]
    for index,style in enumerate(config['image_styles']):
        selected.append(by['image',style,preferred[index%len(preferred)]])
    selected += [by['image','no_watermark','photo_033'],by['image','normal_logo_signature_negative','document_025']]
    for index,style in enumerate(list(config['pdf_styles'])+['no_watermark']):
        selected.append(by['pdf',style,'document_025' if index%2==0 else 'document_026'])
    manifest=[]
    def copy(source, relative):
        target=DEST/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        return relative.as_posix()
    def add_source(row, root, suite, index, override=None):
        case_id=f'{suite}_{index:03d}'
        record=dict(case_id=case_id,suite=suite,label=row.get('label',row.get('id')),subtype=row.get('subtype'),
            scenario=row.get('scenario'),source_id=row['id'],source_group=row['source_group'],
            original_split=row.get('split'),kind=row['kind'],negative=row.get('is_negative',False),
            score_mode='paired',source_manifest=str((root/'manifest.jsonl').relative_to(ROOT)),paths={},pages=[])
        if override: record.update(override)
        source_input=root/row['paths']['watermarked']
        record['paths']['input']=copy(source_input,Path('inputs')/suite/(case_id+source_input.suffix))
        record['source_input_sha256']=sha(source_input)
        for key in ['clean','alpha','mask']:
            if key in row['paths']:
                p=root/row['paths'][key]
                record['paths'][key]=copy(p,Path('references')/case_id/(key+p.suffix))
        if row['kind']=='pdf':
            for p in row['pages']:
                item={'page':p['page'],'paths':{}}
                for key in ['watermarked','clean','alpha','mask']:
                    source=root/p['paths'][key]
                    item['paths'][key]=copy(source,Path('references')/case_id/f"page_{p['page']:02d}_{key}{source.suffix}")
                record['pages'].append(item)
        record['page_count']=len(record['pages']) or 1
        manifest.append(record)
        return record
    for i,r in enumerate(selected,1):add_source(r,V1,'core',i)
    extension_index=0
    for r in rows(INC/'weak_pairs.jsonl'):
        extension_index+=1
        alias=dict(r,kind='image',label='真实手写前后照' if r['task']=='M01' else '醒图真实导出',
                   subtype='real_handwriting' if r['task']=='M01' else 'real_application',
                   scenario='C02' if r['task']=='M01' else 'C04')
        case=add_source(alias,INC,'real',extension_index,dict(score_mode='manual_review',
             original_split='train_reserved',source_manifest='data/watermark_training_increment_v1_1/weak_pairs.jsonl'))
        for role in ['original_clean','aligned_clean','annotation']:
            p=INC/r['paths'][role]
            case['paths'][role]=copy(p,Path('references')/case['case_id']/(role+p.suffix))
    for r in rows(INC/'references.jsonl'):
        if 'local_intake_path' in r:continue
        extension_index+=1
        scan=r.get('task')=='M03'
        alias=dict(r,kind='image',label='实际纸张扫描/采集' if scan else '红色手写批注',
                   subtype='physical_scan' if scan else 'red_handwriting',scenario='C12' if scan else 'C02')
        case=add_source(alias,INC,'real',extension_index,dict(score_mode='manual_review',
             original_split='train_reserved',source_manifest='data/watermark_training_increment_v1_1/references.jsonl'))
        if scan:
            for role,p in r['source_pdfs'].items():
                source=INC/p
                case['paths']['electronic_'+role+'_pdf']=copy(source,Path('references')/case['case_id']/(role+'.pdf'))
            case['note']='Electronic reference is not geometrically aligned to capture; no pixel truth score.'
    pages=rows(INC/'manifest.jsonl')
    for variant in ['gray','black']:
        extension_index+=1
        source_rows=[r for r in pages if r['kind']=='pdf_page' and f'/pdf/{variant}/' in str(INC/r['paths']['pdf_watermarked'])]
        assert len(source_rows)==3
        first=source_rows[0]
        alias=dict(first,kind='pdf',id='native_pdf_'+variant,
             label='真实软件 PDF · '+('灰色' if variant=='gray' else '黑色'),
             paths={'watermarked':first['paths']['pdf_watermarked'],'clean':first['paths']['pdf_clean']},
             pages=[{'page':i+1,'paths':{k:r['paths'][k] for k in ['watermarked','clean','alpha','mask']}} for i,r in enumerate(source_rows)])
        add_source(alias,INC,'real',extension_index,dict(original_split='train',subtype='native_software_'+variant))
    assert len(manifest)==46 and extension_index==10
    core=[r for r in manifest if r['suite']=='core'];real=[r for r in manifest if r['suite']=='real']
    assert len(core)==36 and sum(r['page_count'] for r in core)==48
    assert len({r['subtype'] for r in core if r['kind']=='image' and not r['negative']})==22
    assert len({r['subtype'] for r in core if r['kind']=='pdf' and not r['negative']})==11
    files={}
    for r in manifest:
        for p in list(r['paths'].values())+[p for page in r['pages'] for p in page['paths'].values()]:files[p]=sha(DEST/p)
    frozen=(DEST/'manifest.jsonl')
    frozen.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    missing=[
        {'type':'PDF 文字转曲后的字形路径','status':'missing','reason':'现有 vector_paths 是几何图案，outlined_text 仍是文字对象'},
        {'type':'更多相机原生机型/时间戳与跨应用版本','status':'insufficient_real_sources'},
        {'type':'授权人像背景','status':'missing'},
        {'type':'复杂透明混合、共享资源及跨软件结构组合','status':'not_exhaustively_covered'}]
    coverage=dict(version='1.0.0',purpose='fixed minimal developer regression; not an independent final benchmark',
        core_inputs=len(core),core_pages=sum(r['page_count'] for r in core),real_inputs=len(real),real_pages=sum(r['page_count'] for r in real),
        core_image_positives=22,core_image_negatives=2,core_pdf_positive_documents=11,core_pdf_negative_documents=1,
        core_image_base_groups=len({r['source_group'] for r in core if r['kind']=='image'}),
        coverage_of_available_image_subtypes='22/22',coverage_of_available_pdf_subtypes='11/11',
        original_validation_split_sha256=sha(V1/'splits/validation.jsonl'),manifest_sha256=sha(frozen),
        real_extension_note='Train and train-reserved sources, development regression only; do not report as held-out generalization.',
        missing_or_incomplete=missing,all_types_covered=False)
    jsave(DEST/'coverage.json',coverage)
    jsave(DEST/'file_hashes.json',files)
    readme=f'''最小回归测试集 v1

用途：每次算法修改后，用固定输入比较自动去除与误删；不能用于证明总体成功率或优于竞品。

核心 core：36 个输入文件 / 48 张或页。
22 种图片水印各 1 张 + 2 张图片负例；11 种 PDF 各 1 份两页文档 + 1 份两页 PDF 负例。
图片分散到 9 个既有验证底稿来源组，PDF 使用 2 个既有验证文档组。

真实来源扩展 real：10 个输入文件 / 14 张或页。
2 张黑笔手写、1 张醒图、1 张红笔批注、4 张纸张采集、2 份三页办公软件 PDF。
这些材料属于 train 或 train-reserved 来源，只作开发回归，不能算独立验证/最终测试。
8 张真实图片缺精确配对真值，须人工检查，不能自动冒充去除成功。

输入：inputs/core/ 与 inputs/real/。只把这两个目录里的文件交给产品或竞品处理。
参考：references/。干净图、mask 和 alpha 只能在推理结束后评分时读取，不能提示模型位置。
清单：manifest.jsonl。覆盖与明确缺口：coverage.json。字节校验：file_hashes.json。
不包含抖音未准入素材和银行原件。

每次执行（在项目根目录）：
.venv/bin/python tools/run_min_regression.py --validate-only
.venv/bin/python tools/run_min_regression.py --suite core
.venv/bin/python tools/run_min_regression.py --suite all

结果写入新的 out/min_regression/时间目录，包含 summary.json、逐输入/逐页输出、预测掩码及 report.html。
本套件刚建立，尚未运行去水印推理，不能套用原首轮测试数字；核心选图已改变。
仍缺真正文字转曲字形等类型，不能称为所有真实变化完整覆盖。
原训练数据与原 validation/test 划分不做修改。
'''
    (DEST/'README.txt').write_text(readme)
    def url(p):return os.path.relpath(p,ROOT/'docs').replace(os.sep,'/')
    cards=[]
    for r in manifest:
        imgs=r['pages'] if r['kind']=='pdf' else [{'page':1,'paths':{'watermarked':r['paths']['input'],'clean':r['paths'].get('clean',r['paths'].get('original_clean'))}}]
        figures=[]
        for p in imgs:
            a=DEST/p['paths']['watermarked'];b=DEST/p['paths']['clean'] if p['paths'].get('clean') else None
            paired=f'<figure><a href="{url(a)}" target="_blank"><img loading="lazy" src="{url(a)}"></a><figcaption>输入 · 第 {p["page"]} 页</figcaption></figure>'
            paired+=(f'<figure><a href="{url(b)}" target="_blank"><img loading="lazy" src="{url(b)}"></a><figcaption>干净原图 / 参考（配对性质见上方说明）</figcaption></figure>' if b else '<figure><p>无精确配对干净原图；实扫电子源 PDF 随清单保留。</p></figure>')
            figures.append('<div class="pair">'+paired+'</div>')
        cards.append(f'<article data-suite="{r["suite"]}"><h3>{r["case_id"]} · {html.escape(r["label"])}</h3><p>{r["suite"]} / {r["kind"]} / {r["subtype"]} / {"负例：应保留" if r["negative"] else "正例：应去除"} · 来源 {html.escape(r["source_group"])}</p><p>{"像素配对评分" if r["score_mode"]=="paired" else "实际来源辅助案例，必须人工复核"} · <a href="{url(DEST/r["paths"]["input"])}" target="_blank">打开完整输入文件</a></p>'+''.join(figures)+'</article>')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font:15px/1.8 -apple-system,"PingFang SC",sans-serif;color:#24354a;background:#f2f5f8}aside{position:fixed;inset:0 auto 0 0;width:230px;padding:28px 20px;background:#15283e;color:white}aside a{display:block;color:#d2dfed;text-decoration:none;padding:10px}main{margin-left:230px;max-width:1450px;padding:30px}section,article{border:1px solid #dde5ed;background:white;border-radius:10px;padding:26px;margin-bottom:20px;scroll-margin-top:20px}h1{font-size:30px}h2{font-size:23px}.notice{background:#fff5e4;border-left:4px solid #d69b36;padding:15px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #dfe7ef}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0}figure{margin:0}img{width:100%;height:350px;object-fit:contain;background:#f1f3f5}figcaption{text-align:center;font-size:12px;color:#708198}article p{font-size:12px;color:#60768e;overflow-wrap:anywhere}a{color:#28669f}pre{overflow:auto;background:#edf2f7;padding:18px}li{margin:8px 0}@media(max-width:760px){aside{position:relative;width:auto}aside a{display:inline-block}main{margin:0;padding:12px}.pair{grid-template-columns:1fr}img{height:300px}}'''
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>最小回归测试集</title><style>{css}</style></head><body><aside><h2>最小回归集 v1</h2><a href="#summary">01　集合与用途</a><a href="#run">02　每轮怎么跑</a><a href="#core">03　核心 36 个输入</a><a href="#real">04　真实扩展 10 个输入</a><a href="#missing">05　缺少的类别</a><a href="项目文档总览.html">← 项目文档总览</a></aside><main>
<section id="summary"><h1>每次修改后固定测试哪些数据</h1><p>核心 <b>36 个文件 / 48 张或页</b>；真实扩展 <b>10 个文件 / 14 张或页</b>。共 46 个输入、62 张或页。</p><p>覆盖当前已有 22 种图片样式与 11 种 PDF 子类，包含干净负例，并保留真实手写、应用导出、实扫与办公 PDF 作为扩展检查。</p><p class="notice">这是按“每个已有子类型一例＋负例”确定的最小开发回归集，适合发现改动是否退步。它不覆盖所有背景组合，也不替代完整验证与最终测试。文字转曲等缺项见下方。</p><table><tr><th>集合</th><th>输入组成</th><th>用途</th></tr><tr><td>core：36 文件</td><td>22 张图片正例＋2 张图片负例；11 份 PDF 正例＋1 份 PDF 负例，每份 2 页</td><td>来自冻结的 validation；每轮完整自动跑</td></tr><tr><td>real：10 文件</td><td>8 张真实图片＋2 份三页办公 PDF</td><td>来自 train / train-reserved，只作开发回归；8 张图片需人工核查</td></tr></table><p>原始 validation/test 没有改动。该核心选择与旧首轮抽样不同，原报告数字不能当作本集合的跑分。</p></section>
<section id="run"><h2>每次怎么跑</h2><pre>.venv/bin/python tools/run_min_regression.py --suite core
.venv/bin/python tools/run_min_regression.py --suite all</pre><p>只把 inputs/ 内文件给算法，自动定位；不传位置、不传干净图、不传 mask。推理完成后才读取 references/ 评分。</p><p>每轮保存输入标识、完整输出、预测掩码、逐项指标及 HTML。负例按原图保留单独判断；缺真值的真实图片标为“待人工复核”，不记自动成功。每轮写新目录，保留前一次。</p><p>本轮只建立并校验测试集，尚未运行去水印推理。</p><p><a href="{url(DEST/'README.txt')}">使用说明</a> · <a href="{url(DEST/'manifest.jsonl')}">46 条固定清单</a> · <a href="{url(DEST/'coverage.json')}">覆盖统计与缺口</a></p></section>
<section id="core"><h2>核心 36 个输入文件的全部样本</h2>{''.join(c for c,r in zip(cards,manifest) if r['suite']=='core')}</section><section id="real"><h2>真实来源扩展 10 个输入文件的全部样本</h2><p>已保留原前后照或电子源稿。有原图不代表像素对齐；本节不能用于独立泛化成功率。</p>{''.join(c for c,r in zip(cards,manifest) if r['suite']=='real')}</section><section id="missing"><h2>还没有完全覆盖的类别</h2><ul>{''.join('<li>'+html.escape(m['type'])+'：'+html.escape(m.get('reason',m['status']))+'</li>' for m in missing)}</ul><p>后续补齐时增加套件版本、重新冻结清单，不悄悄替换原样本。</p><p><a href="水印分类样本与最新测试.html">查看全量样本与最近已有结果</a></p></section></main></body></html>'''
    (ROOT/'docs/最小回归测试集.html').write_text(page)
    print(json.dumps(coverage,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
