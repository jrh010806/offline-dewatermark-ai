"""Build the complete local sample catalogue and latest saved test report.

Only preview assets and report files are written. Training manifests, original
samples, model output images and scoring records are read without modification.
"""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
import statistics
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
ASSETS = DOCS / 'assets/watermark_review'
V1 = ROOT / 'data/watermark_training_v1'
INC = ROOT / 'data/watermark_training_increment_v1_1'
OUT = ROOT / 'out/baseline_20261026'
LATEST = ROOT / 'out/foundation_round_20261026'
UPDATED_ON = '2026-10-29'
REPORT = DOCS / '水印分类样本与最新测试.html'
IMAGE_STYLES = json.loads((V1 / 'build_summary.json').read_text())['image_styles']
PDF_STYLES = json.loads((V1 / 'build_summary.json').read_text())['pdf_styles']
SCENARIOS = [
    ('C01', '局部文字、时间戳、网址', '文字贴在边角或局部；细字、浅字和背景纹理容易混淆。', '有 3 种合成样式；真实相机机型、时间戳与不同设备输出仍需扩充。'),
    ('C02', 'Logo、签名、手写标记', '透明与不透明标识、笔画与正文交叠；区分应删除标记与应保留原始签名。', '有 3 种合成样式、2 组真实黑笔近似配对和 1 张红笔单图；真实设备 Logo 及跨背景笔迹仍不足。'),
    ('C03', '居中大字、半透明覆盖', '水印面积大，跨越主体、细节和正文；白字与浅色背景难分离。', '有 2 种合成样式；真实人物背景与更多实际应用导出仍不足。'),
    ('C04', '斜向、平铺和密集文字', '重复周期和旋转角可用于定位；与正文重叠、残缺笔画和非等距布局会漏检。', '有 2 种合成样式、1 组醒图真实导出近似配对；其他应用与版本仍不足。'),
    ('C05', '点阵、线网图案', '细小重复图形与表格线、网格、建筑纹理相似。', '有 2 种合成样式；真实业务产生的点阵和线网尚无已登记专项样本。'),
    ('C06', '极小角标、贴边与裁切水印', '高清图中占比小，水印贴边或被裁掉部分，整图缩放容易丢失。', '有 2 种合成样式；真实设备原尺寸角标和实际多次压缩裁边流程仍不足。'),
    ('C07', '深色、完全不透明遮挡', '原像素被完全覆盖时，需要推测补全；不能核实被遮文字的真实内容。', '有 2 种合成样式；真实不透明业务遮挡与复杂内容交叠样本仍不足。'),
    ('C08', '极淡、渐变、描边、阴影、多层多色', '各部分透明度和颜色不同，混合边缘与压缩会破坏简单颜色分离。', '有 6 种合成样式；多层样式不等于覆盖所有混合模式、压缩和背景组合。'),
    ('C09', 'PDF 独立文字、中文编码、共用内容流', '应按绘制指令识别水印文字；字体编码、正文共用流会造成漏删或误删。', '有 3 种合成结构；另有 macOS 打印导出的 2 个真实软件变体、6 页配对，仅 1 份底稿。'),
    ('C10', 'PDF 独立图片、矢量路径、描边与转曲字', '处理图片透明蒙版和绘制实例；矢量字形与图表线条难区分。', '已有独立图片、几何矢量和描边文字 3 种样本。矢量样本是几何图案，描边字仍是文字对象；文字转曲后的字形路径样本尚缺，跨软件真实导出也不足。'),
    ('C11', 'PDF 嵌套对象、批注图章、可选图层', '需要递归资源、变换和图层结构；同一对象可能被多处复用。', '有 3 种合成结构；复杂多级嵌套、跨软件图层和真实批注流程仍不足。'),
    ('C12', '扫描压平、混合页、纸张实扫与翻拍', '印刷、纸纹、光照、弯曲和透视进入像素，通常不能直接编辑 PDF 对象。', '有 2 种数字结构；已收 4 页实扫采集图和真实手写翻拍，但实扫已做配准及目标框标注，可训练检测，仍不算精确修复配对。'),
]
SPLITS = {'train': '训练', 'validation': '验证', 'test': '测试', 'aux': '辅助 / 参考'}


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def esc(value):
    return html.escape(str(value), quote=True)


def rel(path):
    return quote(os.path.relpath(Path(path), DOCS).replace(os.sep, '/'), safe='/')


def source_link(path, label):
    return f'<a href="{rel(path)}" target="_blank" rel="noopener">{esc(label)}</a>'


def render_pdf(path, page):
    key = hashlib.sha256((str(path.relative_to(ROOT)) + f':{page}').encode()).hexdigest()[:22]
    dest = ASSETS / 'pdf_pages' / f'{key}.png'
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open(path) as doc:
            doc[page - 1].get_pixmap(dpi=110, colorspace=fitz.csRGB, alpha=False).save(dest)
    return dest


def catalogue():
    entries = []
    raw_photos = {r['id']: V1 / r['raw_path'] for r in json.loads((V1/'sources/commons_cc0.json').read_text())}
    user_originals = {r['id']: INC / r['original'] for r in json.loads((INC/'sources/photo_bases.json').read_text())}
    for collection, root in [('v1', V1), ('increment', INC)]:
        for row in read_rows(root / 'manifest.jsonl'):
            kind = row['kind']
            negative = row.get('is_negative', False)
            subtype = row['subtype']
            subtype = 'no_watermark' if subtype == 'no_added_watermark' else subtype
            subtype = 'shared_content_stream' if kind == 'pdf_page' else subtype
            family = 'pdf' if kind in ('pdf', 'pdf_page') else 'image'
            group = f'{family}-{subtype}'
            page_rows = row['pages'] if kind == 'pdf' else [row]
            for page_row in page_rows:
                paths = {key: root / value for key, value in page_row['paths'].items()}
                page = page_row.get('page', row.get('parameters', {}).get('pdf_page_1based'))
                title = row['id'] + (f' · 第 {page} 页' if page else '')
                links = []
                if kind == 'pdf':
                    links = [(root / row['paths']['watermarked'], '带水印 PDF'), (root / row['paths']['clean'], '干净 PDF')]
                elif kind == 'pdf_page':
                    links = [(paths['pdf_watermarked'], '带水印 PDF'), (paths['pdf_clean'], '同版式干净 PDF'), (paths['original_clean_source'], '用户原始源 PDF')]
                raw = (raw_photos if collection == 'v1' else user_originals).get(row.get('base_id'))
                if kind == 'image' and raw:
                    links.append((raw, '未缩放原始照片'))
                entries.append(dict(id=title, record_id=row['id'], family=family, group=group, subtype=subtype,
                    label=row['label'], scenario='N00' if negative else row['scenario'], split=row['split'], collection=collection,
                    source_group=row['source_group'], base_id=row.get('base_id', row['source_group']), background=row.get('background', ''),
                    negative=negative, input=paths['watermarked'], clean=paths['clean'], mask=paths.get('mask'), links=links,
                    note=('真实软件 PDF，干净参考为同导出版式的已知对象移除参考' if kind == 'pdf_page' else
                          '无水印负例：原始内容应保留' if negative else '项目合成水印；干净参考及标注已知'),
                    clean_label='干净原图 / 训练尺寸参考', origin=row.get('watermark_origin', ''), manifest=root / 'manifest.jsonl'))
    for row in read_rows(INC / 'weak_pairs.jsonl'):
        handwritten = row['task'] == 'M01'
        paths = {k: INC / v for k, v in row['paths'].items()}
        entries.append(dict(id=row['id'], record_id=row['id'], family='real', group='real-handwriting' if handwritten else 'real-application',
            subtype='real_handwriting' if handwritten else 'xingtu', label='真实手写前后照' if handwritten else '醒图真实导出',
            scenario='C02' if handwritten else 'C04', split='aux', collection='weak', source_group=row['source_group'],
            background=row.get('background', ''), base_id='', negative=False, input=paths['watermarked'], clean=paths['original_clean'],
            mask=None, links=[(paths['original_clean'], '添加前未配准原始照片'), (paths['aligned_clean'], '已配准参考图'), (paths['annotation'], '区域标注')],
            clean_label='添加前原始照片', note='近似配对，前后照/导出有几何或亮度差异；不进入默认像素监督训练',
            extra_scenarios=['C12'] if handwritten else [], origin='real', manifest=INC / 'weak_pairs.jsonl'))
    for row in read_rows(INC / 'references.jsonl'):
        scan = row.get('task') == 'M03'
        downloaded = 'local_intake_path' in row
        input_path = INC / row['paths']['watermarked'] if row.get('paths') else ROOT / row['local_intake_path']
        links, clean = [], None
        if scan:
            source = INC / row['source_pdfs']['clean']
            page = int(row['id'][-2:])
            clean = render_pdf(source, page)
            links = [(source, '电子干净源 PDF'), (INC / row['source_pdfs']['watermarked'], '带水印打印源 PDF')]
        entries.append(dict(id=row['id'], record_id=row['id'], family='real',
            group='real-paper-scan' if scan else 'real-platform' if downloaded else 'real-handwriting',
            subtype='physical_scan' if scan else 'platform' if downloaded else 'real_red_handwriting',
            label='全能扫描王纸张采集图' if scan else '抖音下载参考（未准入训练）' if downloaded else '红色手写批注单图',
            scenario='C12' if scan else 'C01' if downloaded else 'C02', split='aux', collection='references',
            source_group=row['source_group'], base_id='', background=row.get('background', ''), negative=False,
            input=input_path, clean=clean, mask=None, links=links, clean_label='电子干净源页（未配准）',
            note='电子源页不是同拍摄条件干净真值；纸张透视和弯曲尚未标注处理' if scan else
                 '本地参考展示；来源训练授权待补，无对应干净原图，未复制到训练包' if downloaded else '无添加前原图；单独参考，未进入默认监督训练',
            extra_scenarios=['C02'] if row['id']=='M02_unpaired_001' else ['C12'] if not scan and not downloaded else [],
            origin='physical_scan' if scan else 'third_party_reference' if downloaded else 'real', manifest=INC / 'references.jsonl'))
    training_path=INC/'training_tasks.jsonl'
    if training_path.exists():
        task_qa=json.loads((INC/'qa/training_tasks_validation.json').read_text())
        assert task_qa['status']=='passed' and hashlib.sha256(training_path.read_bytes()).hexdigest()==task_qa['task_manifest_sha256']
        task_rows={r['id']:r for r in read_rows(training_path) if r['category']!='strict_pair'}
        for entry in entries:
            task_row=task_rows.get(entry['record_id'])
            if not task_row:continue
            entry.update(split='train',training_tasks=task_row['tasks'],manifest=training_path)
            if 'restoration' in task_row['tasks']:
                entry.update(clean=INC/task_row['paths']['target'],clean_label='近似干净参考（已配准，训练权重 0.25）',
                             note='已纳入近似配对修复训练；仍有几何和光照差异，不用于精确像素评测')
                entry['links'].append((INC/task_row['paths']['loss_weight'],'有效损失区域（不是水印掩码）'))
            else:
                entry['note']='已纳入检测训练；有经过视觉核对的目标框或红笔标注；没有同拍摄条件修复目标'
                if entry['subtype']=='physical_scan':entry['clean_label']='电子干净底稿页（仅作来源参考）'
            entry['annotation_preview']=INC/'real_training'/entry['record_id']/'annotation_preview.jpg'
            entry['links'].append((entry['annotation_preview'],'训练标注预览'))
    return entries


def latest_results(entries):
    latest_path=LATEST/'after/all_metrics.json'
    if latest_path.exists():
        payload=json.loads(latest_path.read_text())
        before_payload=json.loads((LATEST/'before/all_metrics.json').read_text())
        before={(r['id'],r['page']):r for r in before_payload['results']}
        selected={r['id']:r for r in read_rows_json(LATEST/'selection.json')}
        actual={(r['id'],r['page']) for r in payload['results']}
        assert len(actual)==len(payload['results'])==88 and actual==set(before)
        lookup={(e['record_id'], e.get('page_number',1)):e for e in entries if e['collection']=='v1'}
        # PDF catalogue labels contain their source page number.
        for e in entries:
            if e['collection']=='v1' and e['family']=='pdf':
                pn=int(e['id'].rsplit('第 ',1)[1].split(' 页')[0])
                lookup[e['record_id'],pn]=e
        output=[];image_rows=[];pdf_groups=collections.OrderedDict()
        for raw in payload['results']:
            previous=before[raw['id'],raw['page']]
            source=V1/selected[raw['id']]['paths']['watermarked']
            assert hashlib.sha256(source.read_bytes()).hexdigest()==raw['input_sha256']==previous['input_sha256']
            entry=lookup[raw['id'],raw['page']]
            row=dict(raw,mask_method=raw['method'],negative_control_passed=raw['negative_unchanged'])
            name=raw['name']
            result=dict(entry=entry,metrics=row,previous_metrics=previous,
                        output=LATEST/'after'/f'{name}__output.png',
                        previous_output=LATEST/'before'/f'{name}__output.png',
                        predicted_mask=LATEST/'after'/f'{name}__mask.png')
            assert result['output'].is_file() and result['previous_output'].is_file() and result['predicted_mask'].is_file()
            entry['latest_test']=True
            output.append(result)
            if raw['kind']=='image':image_rows.append(row)
            else:
                group=pdf_groups.setdefault(raw['id'],dict(id=raw['id'],label=raw['label'],
                    subtype=raw['subtype'],is_negative=raw['is_negative'],pages=[]))
                group['pages'].append(row)
        pdf_rows=[r['metrics'] for r in output if r['entry']['family']=='pdf']
        def aggregate(rows):
            positives=[r for r in rows if not r['is_negative']]
            negatives=[r for r in rows if r['is_negative']]
            return dict(total=len(rows),passed=sum(r['passed'] for r in positives)+sum(r['negative_unchanged'] for r in negatives),
                        positive_passed=sum(r['passed'] for r in positives),positive_total=len(positives),
                        negative_controls_passed=sum(r['negative_unchanged'] for r in negatives),negative_controls_total=len(negatives),
                        median_elapsed_ms=statistics.median(r['elapsed_ms'] for r in rows))
        im=dict(created_at=payload['created_at'],results=image_rows,aggregate=aggregate(image_rows))
        pdf=dict(created_at=payload['created_at'],results=list(pdf_groups.values()),aggregate=aggregate(pdf_rows))
        run=dict(name='基础功能修复复测',input_count=88,image_count=len(image_rows),pdf_count=len(pdf_rows),
                 created_at=payload['created_at'],before_at=before_payload['created_at'],
                 metrics_path=latest_path,before_metrics_path=LATEST/'before/all_metrics.json',
                 protocol=payload['protocol'],before_rows=before_payload['results'],after_rows=payload['results'],
                 code_snapshot_matches=all((ROOT/p).is_file() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
                                          for p,h in payload.get('code_sha256',{}).items()),
                 source_paths=[latest_path,LATEST/'before/all_metrics.json',LATEST/'selection.json',LATEST/'summary.json'])
        return im,pdf,output,run
    im = json.loads((OUT / 'metrics.json').read_text())
    pdf = json.loads((OUT / 'pdf_metrics.json').read_text())
    lookup = {e['record_id']: e for e in entries if e['collection'] == 'v1' and e['family'] == 'image'}
    plookup = {(e['record_id'], int(e['id'].rsplit('第 ', 1)[1].split(' 页')[0])): e
               for e in entries if e['collection'] == 'v1' and e['family'] == 'pdf'}
    output = []
    for r in im['results']:
        output.append(dict(entry=lookup[r['id']], metrics=r,
            output=OUT / f"{r['id']}__output.png", predicted_mask=OUT / f"{r['id']}__predicted_mask.png"))
    for doc in pdf['results']:
        for page in doc['pages']:
            r = dict(page, id=doc['id'], subtype=doc['subtype'], label=doc['label'], is_negative=doc['is_negative'])
            output.append(dict(entry=plookup[(doc['id'], page['page'])], metrics=r,
                output=OUT / f"{doc['id']}__page_{page['page']}__output.png", predicted_mask=None))
    for result in output:
        result['entry']['latest_test'] = True
    return im, pdf, output, dict(name='首轮基线',created_at=pdf['created_at'],input_count=len(output),
        image_count=len(im['results']),pdf_count=sum(len(r['pages']) for r in pdf['results']),
        metrics_path=OUT/'metrics.json',before_metrics_path=None,code_snapshot_matches=None,
        before_rows=[],after_rows=[],source_paths=[OUT/'metrics.json',OUT/'pdf_metrics.json'])


def read_rows_json(path):
    return json.loads(path.read_text())


THUMBS = {}


def build_thumbnails(paths):
    directory = ASSETS / 'thumbs'
    directory.mkdir(parents=True, exist_ok=True)
    def one(path):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        token = f'{path.relative_to(ROOT)}:{stat.st_size}:{stat.st_mtime_ns}:preview-v1'
        dest = directory / (hashlib.sha256(token.encode()).hexdigest()[:26] + '.jpg')
        if not dest.exists():
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                image.thumbnail((360, 420), Image.Resampling.LANCZOS)
                image.save(dest, quality=77, optimize=True)
        return path, dest
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (path, dest) in enumerate(pool.map(one, sorted(set(paths))), 1):
            THUMBS[path] = dest
            if index % 600 == 0:
                print(f'Preview files prepared: {index}', flush=True)


def figure(path, label):
    if path is None:
        return f'<figure><div class="missing">没有对应干净原图</div><figcaption>{esc(label)}</figcaption></figure>'
    return (f'<figure><a class="image-link" href="{rel(path)}" data-title="{esc(label)}" target="_blank">'
            f'<img loading="lazy" decoding="async" width="360" height="420" src="{rel(THUMBS[path])}" alt="{esc(label)}"></a>'
            f'<figcaption>{esc(label)} <a href="{rel(path)}" target="_blank" rel="noopener">原尺寸 ↗</a></figcaption></figure>')


def sample_card(entry):
    e = entry
    search = ' '.join(str(e.get(k, '')) for k in ['id', 'label', 'scenario', 'subtype', 'source_group', 'background', 'note']).lower()
    tag = ('真实训练 · '+('修复' if 'restoration' in e['training_tasks'] else '检测')) if e.get('training_tasks') else ('正式 v1' if e['collection'] == 'v1' else '训练增量' if e['collection'] == 'increment' else '近似配对' if e['collection'] == 'weak' else '参考 / 未准入')
    badges = f'<span class="tag">{tag}</span><span class="tag">{SPLITS[e["split"]]}</span>'
    if e.get('latest_test'):
        badges += '<span class="tag tested">本次已测</span>'
    links = list(e['links'])
    if e.get('mask'):
        links.append((e['mask'], '水印区域标注'))
    more = ' · '.join(source_link(p, label) for p, label in links)
    annotation=(f'<div class="annotation-preview">{figure(e["annotation_preview"], "训练标注预览（供训练使用）")}</div>' if e.get('annotation_preview') else '')
    return f'''<article class="sample-card" data-search="{esc(search)}" data-split="{e['split']}" data-collection="{e['collection']}" data-tested="{int(bool(e.get('latest_test')))}">
<header><strong>{esc(e['id'])}</strong><div>{badges}</div></header><div class="image-pair">{figure(e['input'], '样本输入' if e['negative'] else '带水印样本')}{figure(e['clean'], e['clean_label'])}</div>
{annotation}<p class="small">{esc(e['note'])}</p><p class="small source">来源组：{esc(e['source_group'])}</p><p class="small links">{more}</p></article>'''


def test_card(result):
    e, r = result['entry'], result['metrics']
    negative = r['is_negative']
    passed = r.get('negative_control_passed', r['passed']) if negative else r['passed']
    verdict = '安全保留' if negative and passed else '通过' if passed else '未通过'
    vals = r['metrics']
    text = f"残留比例 {vals.get('residual_coverage', 0):.3f} · 掩码 IoU {vals.get('mask_iou', 0):.3f}"
    failures = '、'.join(r.get('failures', [])) or '无'
    previous=result.get('previous_metrics')
    old_state=('保持原样' if previous['negative_unchanged'] else '被误改') if previous and negative else ('通过' if previous and previous['passed'] else '未通过')
    history=(f'<p class="small">同输入对照：修改前 {old_state} → 本次 {verdict}；样本分区 {esc(r.get("split", e["split"]))}；分组 {"原批次" if r.get("suite")=="original" else "扩展开发检查"}。</p>' if previous else '')
    original=(figure(result['previous_output'],'修改前实际输出') if result.get('previous_output') else '')
    extra = ''
    if result['predicted_mask']:
        extra = f'<details class="mask-detail"><summary>查看实际预测掩码与真值掩码</summary><div class="image-pair">{figure(result["predicted_mask"], "实际预测掩码")}{figure(e.get("mask"), "真值区域标注")}</div></details>'
    return f'''<article class="test-card"><header><div><strong>{esc(e['label'])}</strong><p class="small">{esc(e['id'])}</p></div><span class="verdict {'pass' if passed else 'fail'}">{verdict}</span></header>
<div class="test-images {'with-before' if original else ''}">{figure(e['input'], '测试输入')}{original}{figure(result['output'], '最新实际输出')}{figure(e['clean'], '干净原图 / 参考')}</div>
{history}<p class="small">{esc(text)}</p><p class="small">处理方法：{esc(r.get('mask_method', ''))}；原始诊断：{esc(failures)}{'; 负例采用逐像素保持原样判据，原始 passed 字段不参与负例判断' if negative else ''}</p>{extra}</article>'''


def group_label(key):
    family, subtype = key.split('-', 1)
    if family == 'image':
        return IMAGE_STYLES[subtype][1] if subtype in IMAGE_STYLES else {'no_watermark': '无水印负例', 'normal_logo_signature_negative': '合法 Logo、正文签署保护负例'}[subtype]
    if family == 'pdf':
        return PDF_STYLES[subtype][1] if subtype in PDF_STYLES else 'PDF 无水印负例'
    return {'handwriting': '真实手写、签署与红笔批注', 'application': '醒图真实应用导出', 'paper-scan': '纸张实际扫描 / 采集', 'platform': '平台下载参考（未准入训练）'}[subtype]


def evidence_table(entries):
    body = []
    for sid, label, difference, gap in SCENARIOS:
        matching = [e for e in entries if e['scenario'] == sid or sid in e.get('extra_scenarios', [])]
        counts = collections.Counter(e['collection'] for e in matching)
        formal = f"v1 {counts['v1']} 张/页；增量 {counts['increment']} 张/页"
        real_training = sum(bool(e.get('training_tasks')) for e in matching)
        links = sorted({e['group'] for e in matching})
        destinations = ' '.join(f'<a href="#group-{key}">{esc(group_label(key))}</a>' for key in links)
        status = '部分包含：文字转曲字形缺样本' if sid == 'C10' else '基础样本已包含'
        body.append(f'<tr><td><b>{sid}</b><br>{esc(label)}</td><td>{esc(difference)}</td><td>{formal}<br>新增真实训练 {real_training} 条</td><td><span class="tag">{status}</span><br>{esc(gap)}</td><td class="subtype-links">{destinations}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>种类 / 场景</th><th>差别与难点</th><th>全部可视样本数量</th><th>是否覆盖与缺口</th><th>查看该类全部图片</th></tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>'


def build():
    ASSETS.mkdir(parents=True, exist_ok=True)
    entries=catalogue()
    im,pdf,tests,run=latest_results(entries)
    tasks=read_rows(INC/'training_tasks.jsonl')
    task_qa=json.loads((INC/'qa/training_tasks_validation.json').read_text())
    groups=collections.OrderedDict()
    for entry in entries:groups.setdefault(entry['group'],[]).append(entry)
    paths=[e[k] for e in entries for k in ['input','clean','annotation_preview'] if e.get(k)]
    paths += [r[k] for r in tests for k in ['output','previous_output','predicted_mask'] if r.get(k)]
    paths += [r['entry']['mask'] for r in tests if r['entry'].get('mask')]
    print(f'Catalogue: {len(entries)} cards; latest run: {len(tests)} complete comparisons',flush=True)
    build_thumbnails(paths)
    gallery=[];nav=[]
    for key,members in groups.items():
        label=group_label(key)
        count=collections.Counter(e['collection'] for e in members)
        count_text=' / '.join(f'{name} {count[c]}' for c,name in [('v1','v1'),('increment','严格增量'),('weak','近似修复训练'),('references','检测训练')] if count[c])
        gallery.append(f'<details class="sample-group" id="group-{key}" data-group="{key}"><summary><span><b>{esc(label)}</b><small>{esc(count_text)}；全部 {len(members)} 张/页</small></span><span class="group-count">{len(members)}</span></summary><p class="group-note">逐条列出全部样本；原图、参考和训练标注分别标明。</p><div class="sample-grid">'+''.join(sample_card(e) for e in members)+'</div></details>')
        nav.append(f'<a class="type-nav" href="#group-{key}">{esc(label)}<em>{len(members)}</em></a>')
    ia,pa=im['aggregate'],pdf['aggregate']
    image_tests=[r for r in tests if r['entry']['family']=='image']
    pdf_tests=[r for r in tests if r['entry']['family']=='pdf']
    old=run['before_rows'];now=run['after_rows']
    def stat(rows,kind,suite=None,negative=False):
        selected=[r for r in rows if r['kind']==kind and r['is_negative']==negative and (suite is None or r['suite']==suite)]
        return sum(r['negative_unchanged'] if negative else r['passed'] for r in selected),len(selected)
    comparisons=[]
    for kind,suite,label in [('image','original','原批次带水印图片'),('image','extra','新增开发检查图片'),('pdf','original','原批次带水印 PDF 页'),('pdf','extra','新增开发检查 PDF 页')]:
        before,n=stat(old,kind,suite);after,total=stat(now,kind,suite)
        comparisons.append(f'<tr><td>{label}</td><td>{before}/{n}</td><td><b>{after}/{total}</b></td></tr>')
    before_safe=sum(r['negative_unchanged'] for r in old if r['is_negative'])
    after_safe=sum(r['negative_unchanged'] for r in now if r['is_negative'])
    negative_total=sum(r['is_negative'] for r in now)
    comparisons.append(f'<tr><td>全部干净样本，按逐像素不改动判断</td><td>{before_safe}/{negative_total}</td><td><b>{after_safe}/{negative_total}</b></td></tr>')
    subtype_rows=[]
    for subtype in dict.fromkeys(r['metrics']['subtype'] for r in pdf_tests):
        results=[r['metrics'] for r in pdf_tests if r['metrics']['subtype']==subtype]
        previous=[r['previous_metrics'] for r in pdf_tests if r['metrics']['subtype']==subtype]
        success=lambda rows:sum(r['negative_unchanged'] if r['is_negative'] else r['passed'] for r in rows)
        label=PDF_STYLES[subtype][1] if subtype in PDF_STYLES else 'PDF 无水印负例'
        methods='、'.join(sorted({r['mask_method'] for r in results}))
        subtype_rows.append(f'<tr><td>{esc(label)}</td><td>{success(previous)}/{len(previous)}</td><td>{success(results)}/{len(results)}</td><td>{esc(methods)}</td></tr>')
    nonpaper=sum(r['metrics']['mask_method']=='nonpaper_background' for r in image_tests if not r['metrics']['is_negative'])
    image_before,_=stat(old,'image');pdf_before,_=stat(old,'pdf')
    original_images,original_image_total=stat(now,'image','original')
    extra_images,extra_image_total=stat(now,'image','extra')
    snapshot='与本次报告读取的源码一致' if run['code_snapshot_matches'] else '与当前源码存在差异；结果仅代表保存的测试版本'
    source_links=' · '.join(source_link(p,label) for p,label in [(run['metrics_path'],'本次完整指标'),(run['before_metrics_path'],'同输入修改前指标'),(LATEST/'selection.json','88 张／页选样清单')] if p)
    warning='本轮包含 train 和 validation 来源的开发检查样本；不是全部来自独立留出集。图片样式仍集中于少数底图，不能把此处比例当作总体成功率。新增 8 个真实训练样本尚未加入这轮推理结果。'
    content=f'''<div class="notice" id="current-project-entry"><strong>样本目录与历史开发测试：</strong>文件名保留兼容，页内“最新”指下方注明日期的保存结果，不代表当前代码全量验收。<a href="项目现状与后续开发说明.html">查看当前状态</a> · <a href="../out/水印验证报告总目录.html">查看基线、实验和全部报告</a></div><section id="overview"><p class="eyebrow">图片 / PDF · 文档更新 {UPDATED_ON} · 测试记录保留原时间</p><h1>水印分类样本与历史测试</h1>
<p class="lead">按类型查看样本与参考；本页保存的开发测试对应历史版本。当前状态和全量观察基线请从上方入口查看。</p>
<div class="stats"><div><b>12 类</b><span>可见水印验收场景</span></div><div><b>{len(entries):,}</b><span>全部图片／页面卡片</span></div><div><b>{len(tasks)}</b><span>人工增量可训练记录</span></div><div><b>{len(tests)}</b><span>最新测试完整对比</span></div></div>
<div class="notice">人工增量已包含 121 条严格配对、3 组近似修复配对和 5 张检测专用图。最新已保存复测为 {esc(run['created_at'])}，共 58 张图片、30 页 PDF。此次只更新报告，没有重跑推理或训练模型。</div></section>
<section id="training"><h2>当前训练数据状态</h2><p>当前只保留基础训练集、人工训练增量、固定最小回归集三套工作数据。原 v1 的 2,432 条按图片／PDF 文档计；增量 129 条按图片／页计，不能直接相加成独立来源数。固定回归集是派生测试输入，也不另算训练来源。</p>
<div class="table-wrap"><table><thead><tr><th>增量构成</th><th>数量</th><th>当前训练用途</th><th>质量边界</th></tr></thead><tbody>
<tr><td>严格配对</td><td>121 条</td><td>修复与检测</td><td>110 组合成图片、5 个干净负例、6 个已知 PDF 对象页面配对</td></tr>
<tr><td>手写 2 组、醒图 1 组</td><td>3 组</td><td>低权重近似配对修复</td><td>有效损失区域已提供，权重 0.25；不冒充精确像素评测真值</td></tr>
<tr><td>纸张实际扫描／采集</td><td>4 张</td><td>检测框训练</td><td>电子底稿配准和目标框已核对；投影笔画图不是精确分割标签，没有同拍摄条件修复目标</td></tr>
<tr><td>红笔批注单图</td><td>1 张</td><td>笔迹检测／分割训练</td><td>提供前景、背景与忽略边界；没有干净原图，不用于配对修复训练</td></tr></tbody></table></div>
<p><b>统一入口 129 个不重复样本；修复视图 124 条，检测视图 126 条。</b>两个任务共用原 121 条，不能把 124 与 126 相加。任务读取校验已通过，标注仅用于训练；产品推理不要求用户提供水印位置。</p>
<p>{source_link(INC/'training_tasks.jsonl','统一训练清单')} · {source_link(INC/'qa/training_tasks_validation.json','读取与来源检查')} · <a href="真实样本训练标注.html">查看全部 8 个真实训练样本及标注</a></p></section>
<section id="taxonomy"><h2>01　水印种类与覆盖范围</h2>
<p>按图片外观与 PDF 存储结构区分 12 个验收场景，包含已有的 22 种图片样式、11 种 PDF 子类及无水印保护负例。类别允许交叉标记，不能把每行数量相加成独立样本总数。</p>
{evidence_table(entries)}<h3>目前是否已经包含所有需要处理的种类</h3><p><b>还不能说完全覆盖。</b>已有类别覆盖常见可见水印的基础形态，但文字转曲为矢量字形路径的 PDF 专项仍缺样本；跨软件／设备、真实人物、极端压缩与复杂混合模式的多样性仍不足。不可见数字水印不在本工具当前图片／PDF 可见水印的验收范围。</p></section>
<section id="samples"><h2>02　每一类的全部样本、原图与标注</h2>
<p>原 v1 的 2,432 条记录展开 PDF 每一页后，与严格增量 121 条合计 2,745 张／页；再加 3 组近似修复配对、5 张检测训练图，共 <b>{len(entries):,}</b> 张／页卡片，无抽样省略。</p>
<p>严格配对部分：2,222 张图片正例、133 张图片负例、358 页 PDF 正例、32 页 PDF 负例。新增 8 个真实样本都已按相应用途进入训练，不再列为“未准入”。</p>
<div class="filter-panel"><label>查找样本<input id="sample-search" type="search" placeholder="样本 ID、类型、来源组或背景"></label><label>数据集合<select id="sample-collection"><option value="all">全部集合</option><option value="v1">基础 v1</option><option value="increment">严格增量</option><option value="weak">真实近似修复配对</option><option value="references">真实检测训练样本</option></select></label>
<label>用途<select id="sample-split"><option value="all">全部用途</option><option value="train">训练</option><option value="validation">验证</option><option value="test">测试</option></select></label><label class="check"><input id="sample-tested" type="checkbox">只看本轮已测样本</label>
<div class="filter-actions"><button id="expand-all" type="button">展开所有类型</button><button id="collapse-all" type="button">收起所有类型</button><button id="reset-filters" type="button">清空筛选</button></div></div>
<p id="sample-count" role="status">全部 {len(entries):,} 张／页。可按左侧类型跳转，或筛选查看。</p>
<p class="small">可用的干净图并排列出；真实近似配对显示已配准的近似参考，并保留添加前原照链接。扫描图旁的电子页仅作底稿来源参考，不是同条件拍摄真值；其检测标注另行展示。红笔单图没有干净原图。已删除的两张抖音图不再展示；受限银行原件不属于训练样本。</p>
{''.join(gallery)}<p id="no-match" hidden>没有符合条件的样本。</p></section>
<section id="latest"><h2>03　最新一轮测试总结</h2><p class="small">本轮：{esc(run['name'])}；落盘时间（北京时间）：{esc(run['created_at'])}。源码快照：{snapshot}。本次更新只读取已有结果。</p>
<p class="test-summary">带水印图片 <b>{ia['positive_passed']}/{ia['positive_total']}</b> 通过，带水印 PDF <b>{pa['positive_passed']}/{pa['positive_total']}</b> 页通过；全部干净样本 <b>{after_safe}/{negative_total}</b> 保持原样。与相同输入的旧版本相比，图片正例由 {image_before} 张增至 {ia['positive_passed']} 张，PDF 正例由 {pdf_before} 页增至 {pa['positive_passed']} 页。<br>原批次带水印图片为 {original_images}/{original_image_total}；新增开发检查图片为 {extra_images}/{extra_image_total}。PDF 有改善，图片仍未达到可用目标。</p>
<div class="stats"><div><b>{ia['positive_passed']}/{ia['positive_total']}</b><span>图片水印正例通过</span></div><div><b>{ia['negative_controls_passed']}/{ia['negative_controls_total']}</b><span>图片负例保持原样</span></div><div><b>{pa['positive_passed']}/{pa['positive_total']}</b><span>PDF 水印正例页通过</span></div><div><b>{pa['negative_controls_passed']}/{pa['negative_controls_total']}</b><span>PDF 负例页保持原样</span></div></div>
<div class="table-wrap"><table><thead><tr><th>同输入分组</th><th>修改前</th><th>最新保存结果</th></tr></thead><tbody>{''.join(comparisons)}</tbody></table></div>
<p class="notice">{warning}</p><p>正例沿用原评分阈值；负例使用 negative_unchanged 检查输出是否逐像素保持原样。部分负例原始 passed 为 false，是因为通用判据仍记录“没有找到水印”，不能据此把正确保留判为误伤。</p>
<h3>当前失败集中在哪里</h3><p>{ia['positive_total']} 张带水印图片中，{nonpaper} 张返回 nonpaper_background；其余路径也有残留、定位不足或内容损伤。照片文字分支已经加入，仍需同时改进定位与修复。PDF 的矢量路径、嵌套 Form、图章、可选图层和压平扫描页仍有失败，不能把局部结构修复的改善扩大到所有 PDF。</p>
<h3>PDF 子类型结果</h3><div class="table-wrap"><table><thead><tr><th>子类型</th><th>修改前</th><th>最新结果</th><th>实际方法</th></tr></thead><tbody>{''.join(subtype_rows)}</tbody></table></div>
<p class="small">{source_links}。耗时中位数：图片 {ia['median_elapsed_ms']/1000:.2f} 秒／张，PDF {pa['median_elapsed_ms']/1000:.2f} 秒／页。</p></section>
<section id="image-results"><h2>04　图片测试全部 {len(image_tests)} 组完整对比</h2><p>包括 42 张带水印图片和 16 张干净图片；每组依次展示输入、修改前实际输出、本次实际输出、干净参考。全部保留成功、失败和负例，可点击打开完整尺寸。</p><div class="test-grid">{''.join(test_card(r) for r in image_tests)}</div></section>
<section id="pdf-results"><h2>05　PDF 测试全部 {len(pdf_tests)} 页完整对比</h2><p>包括 26 页带水印 PDF 和 4 页干净 PDF，展示完整页面。输入／参考为数据集保存的渲染尺寸，处理输出为 96 DPI；页面未裁切为局部。</p><div class="test-grid">{''.join(test_card(r) for r in pdf_tests)}</div></section>
<section id="changes"><h2>06　下一步怎么改</h2><p>数据接收、相应用途的标注与加载已完成。下一步集中处理当前失败，不再把“扫描图待标注”列为前置任务。</p>
<div class="table-wrap"><table><thead><tr><th>优先级</th><th>本轮证据</th><th>下一步修改</th><th>验收要求</th></tr></thead><tbody>
<tr><td>P0 照片定位</td><td>图片 3/42 通过，26 张仍走背景拒绝路径；白色居中大字 0/6</td><td>覆盖居中大字、平铺文字与浅色区域的自动定位；使用现有精确标签及真实检测标注开发检测分支，推理仅输入图片。</td><td>同一批 88 张／页逐项复测；同时保留掩码与完整输出，记录漏检和误改，20 份干净样本继续保持原样。</td></tr>
<tr><td>P0 照片修复</td><td>局部文字、居中深字和平铺文字各仅 1/6 通过</td><td>分别核对检测范围与修补后残留；用严格配对训练修复，3 组真实近似配对以低权重参与，保护文字、纹理与主体细节。</td><td>报告区域残留、区域外改动与细节损伤；按来源分组检查，开发样本成绩不冒称独立成功率。</td></tr>
<tr><td>P1 PDF 剩余结构</td><td>矢量、嵌套 Form、图章、OCG、压平页各仍 0/2</td><td>跟踪绘制实例、嵌套变换与透明合成；结构可分离时精确处理对象，压平页采用像素路径，保护正常图表与签署。</td><td>保存原生 PDF、完整页图和逐类结果；已有中文与独立图片结构的改善不得回退。</td></tr>
<tr><td>P1 真实数据训练与复测</td><td>129 条增量已经可读取，新增 8 个真实样本尚未进入本轮去水印测试</td><td>接入真实检测框／红笔分割、近似配对损失与有效区域；扫描和红笔单图不伪造修复目标。训练记录、模型权重与运行配置分别保存。</td><td>训练与测试用途分开；真实样本作为开发检查时明确标注，不把已参与训练的图当独立验收集。</td></tr>
<tr><td>P2 完善覆盖与竞品比较</td><td>文字转曲字形路径仍缺专项；尚无竞品同样本输出</td><td>补转曲字样及更多真实来源；使用固定回归集和相同导出条件比较竞品，最后在冻结 test 集验收。</td><td>每类完整输入、输出、失败与耗时均可复核，再判断是否领先。</td></tr></tbody></table></div></section>
<section id="sources"><h2>07　清单、证据与复现</h2><p>本页只生成预览和报告；不修改训练输入、测试输出、评分阈值或模型权重，不上传样本。</p>
<ul><li>{source_link(V1/'manifest.jsonl','基础 v1：2,432 条图片／文档记录')}；{source_link(INC/'training_tasks.jsonl','当前增量：129 个任务训练样本')}；{source_link(INC/'manifest.jsonl','兼容严格配对：121 条')}。</li>
<li>{source_link(INC/'qa/training_tasks_validation.json','训练读取检查')}；{source_link(INC/'qa/real_training_visual_review.json','8 个真实样本视觉核对')}。</li>
<li>{source_links}；{source_link(LATEST/'tests_after.log','该轮程序检查记录')}。</li>
<li>{source_link(ASSETS/'sample_index.json','全量卡片索引与来源哈希')}；{source_link(ASSETS/'validation.json','本页数量与文件检查')}。</li>
<li>生成脚本 tools/build_watermark_review.py；左侧目录可快速跳转，样本支持筛选、展开和原尺寸查看。</li></ul>
<p><a href="项目文档总览.html">项目文档总览</a> · <a href="真实样本训练标注.html">真实样本标注</a> · <a href="最小回归测试集.html">固定最小回归集</a> · <a href="基础功能修复测试报告.html">基础修复报告</a> · <a href="首轮基线评测.html">早期 48 项基线</a></p></section>'''
    css=(ROOT/'tools/watermark_review.css').read_text()
    js=(ROOT/'tools/watermark_review.js').read_text()
    main_nav=f'<a href="#training">当前训练数据状态</a><a href="#taxonomy">01　种类与覆盖缺口</a><a href="#samples">02　全部样本与原图</a><a href="#latest">03　最新测试总结</a><a href="#image-results">04　{len(image_tests)} 组图片完整对比</a><a href="#pdf-results">05　{len(pdf_tests)} 页 PDF 完整对比</a><a href="#changes">06　下一步修改方案</a><a href="#sources">07　清单与证据</a>'
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>水印分类样本与最新测试</title><style>{css}</style></head><body><aside><a class="brand" href="#overview">水印覆盖与实测</a><span class="nav-caption">报告目录</span><nav>{main_nav}</nav><span class="nav-caption">全部样本分类 · 点击展开</span><nav class="types">{''.join(nav)}</nav></aside><main>{content}</main><a class="back-top" href="#overview">回到顶部 ↑</a><dialog id="image-viewer"><div class="viewer-bar"><strong id="viewer-title"></strong><a id="viewer-original" target="_blank">打开原尺寸文件 ↗</a><button id="viewer-zoom" type="button">1:1 放大</button><button id="viewer-close" type="button">关闭 ✕</button></div><div class="viewer-canvas"><img id="viewer-image" alt="完整图片预览"></div></dialog><script>{js}</script></body></html>'''
    REPORT.write_text(page)
    source_paths=[V1/'manifest.jsonl',INC/'manifest.jsonl',INC/'weak_pairs.jsonl',INC/'references.jsonl',INC/'training_tasks.jsonl',INC/'qa/training_tasks_validation.json',*run['source_paths']]
    serial=[{k:(str(v.relative_to(ROOT)) if isinstance(v,Path) else [[str(p.relative_to(ROOT)),label] for p,label in v] if k=='links' else v) for k,v in e.items()} for e in entries]
    index=dict(updated_on=UPDATED_ON,scope='all current datasets; PDF documents expanded per page; real task labels shown',source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},card_count=len(entries),group_count=len(groups),test_input_count=len(tests),latest_run=run['name'],image_test_at=im['created_at'],pdf_test_at=pdf['created_at'],entries=serial)
    (ASSETS/'sample_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2))
    validation=dict(status='passed',updated_on=UPDATED_ON,sample_cards=len(entries),groups=len(groups),strict_paired_image_or_page_cards=sum(e['collection'] in ('v1','increment') for e in entries),real_training_cards=sum(bool(e.get('training_tasks')) for e in entries),increment_unique_training_records=len(tasks),restoration_training_records=task_qa['restoration_records'],detection_training_records=task_qa['detection_records'],latest_image_results=len(image_tests),latest_pdf_page_results=len(pdf_tests),before_after_input_hashes_checked=len(tests),positive_image_passed=ia['positive_passed'],positive_image_total=ia['positive_total'],positive_pdf_passed=pa['positive_passed'],positive_pdf_total=pa['positive_total'],clean_unchanged=after_safe,clean_total=negative_total,source_images_and_previews=len(THUMBS),clean_reference_missing=sum(e['clean'] is None for e in entries),all_input_and_available_clean_images_exist=True,code_snapshot_matches=run['code_snapshot_matches'],scope_note=warning)
    assert len(entries)==2753 and len(tasks)==129 and len(tests)==run['input_count']
    assert validation['real_training_cards']==8 and len(image_tests)==58 and len(pdf_tests)==30
    (ASSETS/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    print(REPORT,flush=True)



if __name__ == '__main__':
    build()
