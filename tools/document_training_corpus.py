"""Create the dataset card, human intake checklist, contact sheets and exports."""
from pathlib import Path
import base64, collections, csv, hashlib, html, io, json, shutil, os
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1];D=Path(os.environ.get('WM_DATASET_ROOT',ROOT/'data/watermark_training_v1'))
F=D/'assets/fonts/NotoSansCJKsc-Regular.otf'
def font(n):return ImageFont.truetype(str(F),n)
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(D))

TASKS=[
 dict(id='M01',name='真实手写签名',scenario='C02',target='10 个自己书写的不同签名/签署笔迹素材',
      steps='可打印目录内的签名采集表，用纸笔实际书写后扫描或拍摄；保留原始照片。可另存透明 PNG，但同时提交原始文件。不要使用程序画线替代。',
      files='原始照片/扫描 + 可选透明 PNG + 来源记录',alignment='素材，无需与底图配准'),
 dict(id='M02',name='设备或真实应用产生的图片水印',scenario='C01/C02/C06',target='至少 10 张，覆盖 2 种设备或应用',
      steps='用自己的手机/相机应用开启日期、位置或自定义文字水印后实际拍照，或使用自有应用正常导出带标识图片。保留原尺寸文件和设备/应用版本。',
      files='原始导出图 + 能保留则附无水印原图 + 版本/设置记录',alignment='没有对应原图时明确标为非配对，不虚构干净参考'),
 dict(id='M03',name='实际纸张扫描',scenario='C12',target='10 份不同底稿，每份至少 2 页',
      steps='可直接打印 manual_incoming/打印素材 中已准备的 10 对底稿，也可使用自己的不同底稿。用扫描仪或扫描应用实际扫描带水印纸张，保留电子干净底稿；条件允许时也打印扫描无水印版。保留彩色/灰度和倾斜等实际差异。',
      files='电子干净底稿 + 带水印打印源 + 原始扫描 PDF/图片 + 扫描设置',alignment='扫描与电子底稿先标未配准，不能直接当逐像素对齐图片对'),
 dict(id='M04',name='纸张实际翻拍',scenario='C12',target='10 份不同底稿，每份至少 2 个拍摄角度',
      steps='可使用目录内已准备的打印素材。拍摄带水印纸张，分别拍正面和倾斜视角；加入自然阴影、反光或轻微模糊。直接保存相机原文件，不要只提交聊天软件压缩后的图片。',
      files='电子底稿 + 带水印打印源 + 相机原图 + 拍摄条件',alignment='先标未配准，保留透视和光照信息'),
 dict(id='M05',name='真实办公软件原生 PDF',scenario='C09/C10/C11/C12',target='至少 15 份不同文档，覆盖 3 个软件/导出流程',
      steps='在实际使用的 WPS、Word、PDF 编辑器等软件中分别生成带水印文件。选择文字水印、图片水印、图章批注等软件实际支持的功能。每份保存加水印前和加水印后的 PDF，以及源文件和软件版本；不要统一截图转 PDF。',
      files='源文档 + 干净 PDF + 带水印 PDF + 软件版本/操作说明',alignment='内部对象类型由接收脚本/后续检查确认，不凭肉眼填写'),
 dict(id='M06',name='真实人物与业务背景补充',scenario='C01—C08',target='至少 10 张自有照片与 10 份可用业务底稿',
      steps='提供有权用于训练的人像、商品、室内等照片，以及可使用的中文合同式、票据式或复杂表格底稿。先提供干净素材即可，后续可自动叠加已有全部水印样式。',
      files='干净原文件 + 来源及训练使用依据',alignment='底素材；不要求你手工逐张添加所有水印'),
]

def hard_negatives(rows):
    """Normal letterhead/logo/signature content is intentionally outside masks."""
    ids={r['id'] for r in rows}
    bases=json.loads((D/'bases/manifest.json').read_text())
    for b in bases:
        if not b['id'].startswith('document_'):continue
        sid=b['id']+'__normal_logo_signature_negative'
        if sid in ids:continue
        folder=D/f'images/{b["split"]}/{sid}';folder.mkdir(parents=True,exist_ok=True)
        im=Image.open(D/b['path']).convert('RGB');draw=ImageDraw.Draw(im);w,h=im.size
        # A letterhead emblem and an explicitly labelled signature in the body.
        draw.rounded_rectangle((w-145,95,w-45,195),radius=16,fill=(25,78,125))
        draw.text((w-133,123),'HX',font=font(38),fill='white')
        draw.text((60,h-150),'经办人签署（正文内容）：',font=font(20),fill=(25,35,45))
        pts=[(340+j*2.5,h-110+np.sin(j*.2+int(b['id'][-3:]))*10+np.sin(j*.7)*4) for j in range(95)]
        draw.line(pts,fill=(30,35,45),width=3)
        p=folder/'clean.png';im.save(p,compress_level=3);m=folder/'zero.png';Image.new('L',im.size).save(m)
        paths=dict(clean=rel(p),watermarked=rel(p),mask=rel(m),alpha=rel(m))
        rows.append(dict(id=sid,kind='image',scenario='N00',subtype='normal_logo_signature_negative',label='正常页眉 Logo 与正文签署负例',
            split=b['split'],source_group=b['source_group'],split_group=b['split_group'],base_id=b['id'],background=b['background'],
            provenance='project_original_document_with_legitimate_body_content',license='LicenseRef-Project-Original',
            size=list(im.size),paths=paths,sha256={k:sha(D/v) for k,v in paths.items()},parameters={},watermark_origin='none',
            is_negative=True,mask_definition='all_zero_normal_header_logo_and_body_signature_are_content',
            review_status='automated_integrity_verified_contact_sheet_review_pending'))
    return rows

def save_manifests(rows):
    (D/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for split in ['train','validation','test']:
        (D/f'splits/{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['split']==split))
    fields=['id','kind','scenario','subtype','split','source_group','split_group','base_id','background','provenance','license','is_negative','clean','watermarked','alpha','mask']
    with (D/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for r in rows:w.writerow({k:r.get(k,r['paths'].get(k,'')) for k in fields})
    pages=[]
    for r in rows:
        if r['kind']=='pdf':
            for p in r['pages']:pages.append(dict(id=r['id']+f'__page_{p["page"]:02d}',parent_id=r['id'],split=r['split'],scenario=r['scenario'],subtype=r['subtype'],source_group=r['source_group'],split_group=r['split_group'],is_negative=r['is_negative'],**p))
    (D/'pdf_pages.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pages))

def contact(rows,kind):
    positive=[r for r in rows if r['kind']==kind and not r['is_negative']]
    styles=list(dict.fromkeys(r['subtype'] for r in positive));cols=4;tw=350;th=330
    canvas=Image.new('RGB',(cols*tw,((len(styles)+cols-1)//cols)*th),'#f4f6f9');draw=ImageDraw.Draw(canvas)
    for i,style in enumerate(styles):
        options=[r for r in positive if r['subtype']==style]
        if kind=='image':
            # Alternate actual photos and authored text/table backgrounds.
            pool=[r for r in options if r['base_id']==('photo_005' if i%2==0 else 'document_003')]
            row=(pool or options)[0];paths=row['paths']
        else:row=options[0];paths=row['pages'][0]['paths']
        im=Image.open(D/paths['watermarked']).convert('RGB')
        mask=Image.open(D/paths['mask']).convert('L');box=mask.getbbox()
        if box:
            x0,y0,x1,y1=box;pad=20;im=im.crop((max(0,x0-pad),max(0,y0-pad),min(im.width,x1+pad),min(im.height,y1+pad)))
        im.thumbnail((tw-20,th-95))
        x=i%cols*tw;y=i//cols*th
        canvas.paste(im,(x+(tw-im.width)//2,y+12))
        draw.text((x+12,y+th-78),row['label'],font=font(18),fill='#1c2d45')
        draw.text((x+12,y+th-49),style,font=font(13),fill='#4b617d')
        draw.text((x+12,y+th-26),'已知水印区域局部预览 · '+row['base_id'],font=font(12),fill='#697a90')
    p=D/f'qa/{kind}_styles.png';canvas.save(p);return p

def printable_intake_materials(rows):
    folder=D/'manual_incoming/打印素材';folder.mkdir(parents=True,exist_ok=True)
    source_ids=[]
    for split,n in [('train',6),('validation',2),('test',2)]:
        source_ids.extend(list(dict.fromkeys(r['base_id'] for r in rows if r['kind']=='pdf' and r['split']==split))[:n])
    styles=['independent_text','independent_image','vector_paths','outlined_text','nested_form','stamp_annotation','optional_content','shared_content_stream','unicode_text','independent_image']
    listing=[]
    for i,(bid,style) in enumerate(zip(source_ids,styles),1):
        r=next(r for r in rows if r['kind']=='pdf' and r['base_id']==bid and r['subtype']==style)
        paths={}
        for role,title in [('clean','干净原稿'),('watermarked','带水印打印稿')]:
            p=folder/f'{i:02d}_{bid}_{title}.pdf';shutil.copy2(D/r['paths'][role],p)
            if sha(p)!=r['sha256'][role]:raise ValueError('Printable source copy mismatch')
            paths[role]=rel(p)
        listing.append(dict(print_id=f'PRINT_{i:02d}',source_sample=r['id'],source_group=r['source_group'],
                            split_group=r['split_group'],locked_split=r['split'],paths=paths,sha256=r['sha256']))
    dump(folder/'打印底稿来源清单.json',listing)
    (folder/'打印说明.txt').write_text('本目录提供 10 对可直接使用的原始 PDF（每份 2 页）。\n扫描任务：打印带水印版并实际扫描；条件允许时也打印扫描干净版。\n翻拍任务：拍摄带水印打印纸，每份正面/倾斜至少各一张。\n文件名保留 PRINT 编号，并记录设备、分辨率、彩色/灰度、拍摄条件。\n这些底稿已属于训练/验证/测试中的固定来源组。扫描/翻拍衍生件必须继承来源清单中的 locked_split，不得重新随机划分。\n原始扫描/翻拍与电子底稿先标未配准，尚不纳入正式训练清单。\n')
    doc=fitz.open();p=doc.new_page(width=595,height=842)
    p.insert_text((42,55),'真实手写笔迹采集表',fontname='china-s',fontsize=21)
    p.insert_text((42,83),'用自己的笔迹在每行书写；扫描或拍摄整页，保留原文件。',fontname='china-s',fontsize=10)
    p.insert_text((42,101),'此页为空白采集模板，不是已经完成的训练样本。',fontname='china-s',fontsize=10)
    for i in range(10):
        y=145+i*58;p.insert_text((42,y+20),f'{i+1:02d}',fontsize=12)
        p.draw_line((85,y+36),(548,y+36),color=(.7,.75,.8),width=.5)
    p.insert_text((42,785),'请在来源登记表中填写笔迹作者及训练使用依据。',fontname='china-s',fontsize=10)
    path=D/'manual_incoming/M01_真实手写签名/签名采集表.pdf';doc.save(path,garbage=4,deflate=True)
    p.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False).save(D/'qa/signature_collection_form.png');doc.close()

def main():
    rows=[json.loads(x) for x in (D/'manifest.jsonl').read_text().splitlines()]
    rows=hard_negatives(rows)
    visual_path=D/'qa/visual_review.json'
    if visual_path.exists() and json.loads(visual_path.read_text()).get('status')=='passed':
        for row in rows:row['review_status']='automated_integrity_verified_base_and_subtype_visual_reviewed_not_individual_review'
    save_manifests(rows)
    (D/'qa').mkdir(exist_ok=True)
    for kind in ['image','pdf']:contact(rows,kind)
    manual=D/'manual_incoming';manual.mkdir(exist_ok=True)
    for t in TASKS:
        p=manual/(t['id']+'_'+t['name']);p.mkdir(exist_ok=True)
        (p/'填写说明.txt').write_text(t['target']+'\n'+t['steps']+'\n提交：'+t['files']+'\n配对：'+t['alignment']+'\n未完成审核前不自动并入训练清单。\n')
    printable_intake_materials(rows)
    dump(D/'manual_collection_tasks.json',TASKS)
    with (manual/'来源登记模板.csv').open('w',encoding='utf-8-sig',newline='') as f:
        csv.writer(f).writerow(['sample_id','task_id','source_group','file_role','relative_path','source_or_creator','training_rights_basis','permission_record_path','software_device_version','watermark_setting','alignment_status','notes'])
    split_counts={s:dict(collections.Counter(r['kind'] for r in rows if r['split']==s)) for s in ['train','validation','test']}
    counts=dict(total=len(rows),image_positive=sum(r['kind']=='image' and not r['is_negative'] for r in rows),
        image_negative=sum(r['kind']=='image' and r['is_negative'] for r in rows),
        pdf_positive=sum(r['kind']=='pdf' and not r['is_negative'] for r in rows),
        pdf_negative=sum(r['kind']=='pdf' and r['is_negative'] for r in rows),
        pdf_pages=sum(r.get('page_count',0) for r in rows),
        source_groups=len({r['source_group'] for r in rows}),split_groups=len({r['split_group'] for r in rows}),
        photo_bases=32,project_original_bases=64,image_subtypes=22,pdf_subtypes=11)
    subtype_rows=[]
    for kind in ['image','pdf']:
        for subtype in dict.fromkeys(r['subtype'] for r in rows if r['kind']==kind and not r['is_negative']):
            rr=[r for r in rows if r['kind']==kind and r['subtype']==subtype]
            subtype_rows.append(dict(kind=kind,subtype=subtype,label=rr[0]['label'],scenario=rr[0]['scenario'],records=len(rr),source_groups=len({r['source_group'] for r in rr})))
    qa=json.loads((D/'qa/validation.json').read_text()) if (D/'qa/validation.json').exists() else {'status':'pending'}
    visual=json.loads((D/'qa/visual_review.json').read_text()) if (D/'qa/visual_review.json').exists() else {'status':'pending'}
    legacy={'status':'reference_only_excluded_from_training_release','records':383,
        'sources':['data/public_watermarks_20261026/labels.json','data/real_samples','data/syn','external watermark_samples'],
        'reason':'Existing imports mix author-test subsets, underlying image/trademark license uncertainty, and samples without complete original provenance. Existing files preserved; none silently promoted to licensed training records.'}
    dump(D/'legacy_exclusions.json',legacy)
    card=dict(name='自建可见水印训练数据集 v1',version='1.0.0',created_on='2026-10-26',
        intended_use='Supervised image watermark localization/restoration training, and PDF structure research; image/PDF negative controls included.',
        counts=counts,splits=split_counts,subtypes=subtype_rows,
        positive_watermark_origin='100% project-synthetic; no positive claimed as naturally occurring business watermark',
        rights_basis={'photographs':'32 per-file CC0-1.0 declarations archived from Wikimedia Commons; original author/page/url/hash recorded',
            'original_material':'64 project-authored document/chart/procedural bases and all watermark assets; LicenseRef-Project-Original; project owner sets redistribution license',
            'font':'Noto Sans CJK SC, SIL OFL-1.1, font file and original notice included'},
        partition_policy='All variants of a base stay together; photographs additionally grouped conservatively by normalized creator credit. PDF/image derivatives share group. Rendering templates may be shared across partitions; this is not an unseen-template split.',
        alpha_and_mask={'image':'Exact 8-bit effective alpha, exact RGBA overlay and binary alpha>0 support. Lossless PNG pairs.',
            'pdf':'Known objects rendered separately at 108 dpi; alpha/support before final raster flattening. Not a guarantee of exact full-page RGB alpha inversion. Stored rendered pairs are training pixels.'},
        backgrounds='32 CC0 landscape/vegetation/architecture/object photos; 32 original bilingual text/table/report PDFs; 16 original charts; 16 procedural textures.',
        validation=qa.get('status'),visual_review=visual.get('status'),
        manual_tasks=TASKS,limitations=['真实手写笔迹尚未采集；已有签名为程序模拟。','数字压平不是实际纸张扫描或翻拍。','PDF 样本由一个生成引擎创建，跨办公软件真实导出仍待补。','32 张真实照片不等于 32 张真实业务水印，正样本水印均为合成。','许可依据为归档的公开声明和项目自制记录；历史来源未明素材被排除。'],
        public_description='已构建自建可见水印训练数据集 v1，含图片配对样本、PDF 专项结构样本及负例，覆盖约定的 12 类基础场景。水印为项目合成，底素材包含公开 CC0 照片与项目自制文档/图表；实扫、翻拍和真实软件原生文件另行补充。',
        training_use_statement='A constructed training dataset. This release itself does not establish that any shipped model has already trained on these samples.')
    dump(D/'dataset_card.json',card)
    summary=json.loads((D/'build_summary.json').read_text());summary.update(total_records=len(rows),kind_counts=dict(collections.Counter(r['kind'] for r in rows)),split_counts=dict(collections.Counter(r['split'] for r in rows)),scenario_counts=dict(collections.Counter(r['scenario'] for r in rows)),hard_negative_records=32)
    dump(D/'build_summary.json',summary)
    # Full source scripts make the corpus usable without chat history.
    scripts=D/'scripts';scripts.mkdir(exist_ok=True)
    for name in ['fetch_cc0_training_bases.py','build_training_corpus.py','document_training_corpus.py','validate_training_corpus.py','load_watermark_training.py']:
        if (ROOT/'tools'/name).exists():shutil.copy2(ROOT/'tools'/name,scripts/name)
    (D/'requirements.txt').write_text('numpy>=2.0\nPillow>=11.0\nPyMuPDF>=1.24\n')
    (D/'LICENSES.txt').write_text('This is a mixed-provenance project dataset, not a blanket Apache-2.0 grant.\n32 accepted photograph sources: CC0-1.0, see sources/commons_cc0.json and per-file evidence.\nOriginal document/chart/texture content and watermark assets: project-created, LicenseRef-Project-Original; the project owner controls redistribution terms.\nNoto Sans CJK font: SIL OFL-1.1, original license notice in licenses/NotoSansCJK-OFL.txt.\nThe compilation must retain these notices and source records. Legacy data is excluded.\n')
    readme=f'''自建可见水印训练数据集 v1 / 1.0.0

数据数量
- 图片正样本 {counts['image_positive']} 对；无水印负例 {counts['image_negative']} 张。
- PDF 正样本 {counts['pdf_positive']} 对；无水印负例 {counts['pdf_negative']} 份；共 {counts['pdf_pages']} 页样本记录。
- {counts['source_groups']} 个底稿来源组；{counts['split_groups']} 个保守划分组（同摄影者照片一起划分）。
- 22 种图片细分、11 种 PDF 细分，映射到 C01—C12；正样本水印全部由项目合成。

怎么使用
1. manifest.jsonl 是主清单；manifest.csv 方便浏览。所有 paths 相对本目录。
2. splits/train.jsonl、validation.jsonl、test.jsonl 已按来源组隔离。
3. 图片：watermarked 与 clean 是 RGB PNG；alpha 为有效透明度，mask 为二值区域；overlay_rgba 可精确重建正样本。
4. PDF：原文件成对保留；pages 内附每页 clean、watermarked、alpha、mask；pdf_pages.jsonl 为展开清单。
5. 无水印负例 clean 与 watermarked 相同、alpha/mask 全零；正常页眉 Logo 与正文签署负例不能标成水印。
6. 使用 Python 及 requirements.txt：
   python scripts/load_watermark_training.py --root /本数据集目录 --split train --kind image --limit 3
   python scripts/load_watermark_training.py --root /本数据集目录 --split train --kind pdf --limit 3
   函数 iter_pairs 返回 HWC float32 RGB [0,1] 和 HW alpha/mask [0,1]；变尺寸样本由训练端做同步裁切，不要单独缩放掩码。

复现（在原项目根目录，使用仓库 .venv）
   .venv/bin/python tools/fetch_cc0_training_bases.py
   .venv/bin/python tools/build_training_corpus.py
   .venv/bin/python tools/document_training_corpus.py
   .venv/bin/python tools/validate_training_corpus.py
   保留已归档照片、字体及许可；不要在重建时删掉 sources 和 assets。
   独立解压本包后，也可在数据集目录使用 WM_DATASET_ROOT 指定目录：
   WM_DATASET_ROOT="$PWD" python scripts/build_training_corpus.py
   WM_DATASET_ROOT="$PWD" python scripts/document_training_corpus.py
   WM_DATASET_ROOT="$PWD" python scripts/validate_training_corpus.py
   脚本、参数与源文件可以重建数据；不同字体/渲染库版本会改变像素，PDF 对象编号/文件哈希也可能变化。以归档版本、渲染页面和结构校验为准。

来源和许可
   真实照片：Commons 每个文件的 CC0 声明、作者、来源 URL、原始下载和 SHA256 均已记录。
   文档、图表、纹理、水印：本项目构造；源文案与数字均为虚构；不含外部商标素材。
   字体：Noto Sans CJK SC，保留 OFL-1.1 原始许可。
   旧数据 383 条仅保留参考，不混入本训练发布版；详见 legacy_exclusions.json。
   许可边界详见 LICENSES.txt。不要把所有底图或数据一律声称为公司独家拍摄。

剩余人工任务
   manual_collection_tasks.json 与 manual_incoming/ 中有 6 项清单及来源登记模板。
   已准备 10 对可直接打印的 PDF 底稿，以及一页空白手写采集表；它们是收集辅助材料，不计入已完成样本数量。
   实扫和翻拍先标未配准；接收检查前不会自动混入训练清单。

对外表述
   {card['public_description']}
   这是已构建的训练数据集；是否已被某个模型用于训练，需要另有实际训练记录。
'''
    (D/'README.txt').write_text(readme)
    style_html=''.join(f'<tr><td>{r["scenario"]}</td><td>{r["kind"]}</td><td>{html.escape(r["label"])}</td><td>{r["records"]}</td><td>{r["source_groups"]}</td></tr>' for r in subtype_rows)
    task_html=''.join(f'<tr><td>{t["id"]}</td><td>{t["name"]}</td><td>{t["target"]}</td><td>{t["steps"]}<br><strong>提交：</strong>{t["files"]}<br><strong>配对：</strong>{t["alignment"]}</td></tr>' for t in TASKS)
    split_html=''.join(f'<tr><td>{s}</td><td>{n.get("image",0)}</td><td>{n.get("pdf",0)}</td><td>{sum(n.values())}</td></tr>' for s,n in split_counts.items())
    def embed(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    sections=[
      ('summary','数据集概况',f'<h1>自建可见水印训练数据集 v1</h1><p>图片与 PDF · 版本 1.0.0 · 2026-10-26</p><div class="stats"><div><b>{counts["image_positive"]}</b>图片配对正样本</div><div><b>{counts["pdf_positive"]}</b>PDF 配对正样本</div><div><b>{counts["image_negative"]+counts["pdf_negative"]}</b>无水印负例</div><div><b>12</b>基础场景</div></div><p class="notice">自动可创建的 12 类基础场景已建立样本；所有正样本水印为项目合成。真实手写、设备水印、实际扫描/翻拍和跨软件原生文件仍按人工清单补充。</p><p>完整性校验：<strong>{qa.get("status")}</strong>；视觉检查：<strong>{visual.get("status")}</strong>。详见数据目录中的 qa/。</p><p>正式清单共 {len(rows)} 条记录；{counts["source_groups"]} 个底稿来源组；按摄影者等合并为 {counts["split_groups"]} 个保守划分组。衍生版本不会被计为新的独立底稿。</p>'),
      ('coverage','样本类型覆盖','<p>每种图片细分使用 96 张底图；每种 PDF 细分使用 16 个文档来源，每份 2 页。数量是本次发布清单，不包含历史参考数据。</p><table><thead><tr><th>场景</th><th>格式</th><th>细分类型</th><th>记录数</th><th>底稿来源数</th></tr></thead><tbody>'+style_html+'</tbody></table>'),
      ('sources','来源与使用依据','<p>32 张底图为逐文件保存 CC0 许可声明的公开真实照片；其余 64 张为项目自制的中英文文档、表格、图表和程序纹理。所有文字、几何 Logo、签名模拟笔迹、水印样式由项目创建。</p><p>保留 sources/commons_cc0.json、每份 API 原始证据、下载字节、原作者/页面 URL、文件 SHA256。字体附 Noto Sans CJK 的 OFL-1.1 原始许可。项目原创建内容的发布许可由项目方决定，未擅自承诺整包为 Apache-2.0。</p><p>历史 383 条记录存在作者测试分区或底层照片/标识授权链尚未齐全等问题，列入 reference-only，不混入正式训练清单。已排除重复图和带有原生 500px 水印的“干净底图”。</p><img src="'+embed(D/'sources/photo_contact_sheet.png')+'" alt="32张已核验CC0照片底图">'),
      ('labels','标注与文件结构','<p>主索引 manifest.jsonl；可浏览索引 manifest.csv；PDF 页面展开索引 pdf_pages.jsonl。记录包含来源组、划分组、背景、类型、尺寸、生成参数、许可依据、文件路径和 SHA256。</p><p>图片保存 clean、watermarked、overlay_rgba、alpha、mask。alpha 是已知叠加层的有效 8 位透明度；mask 严格为 alpha&gt;0 的二值支持区域，二者不混用。无损 PNG 图片可以由 clean 与 overlay_rgba 精确重建。</p><p>PDF 同时保留原文件与逐页图片；其掩码来自单独渲染的已知水印对象。压平样本保存的是最终压平前的对象透明度，不能当成全页像素混合公式。数字压平不标成实扫。</p><p>程序绘制的签名明确标成模拟笔迹；真实照片上合成的水印明确标成合成水印。正常页眉 Logo、正文签署及原有内容作为负例保留。</p>'),
      ('splits','训练验证测试划分','<table><thead><tr><th>划分</th><th>图片记录</th><th>PDF 记录</th><th>合计</th></tr></thead><tbody>'+split_html+'</tbody></table><p>同一底图的所有水印、负例和 PDF 派生件留在同一划分。照片额外按摄影者署名保守分组，防止同一组拍摄素材跨分区。模板可以跨分区共享，因此不把此划分称为“未见模板测试”。</p>'),
      ('preview','样本预览','<p>以下展示带水印样本的局部原始区域。可在浏览器中放大查看；完整分辨率文件在数据目录中。</p><h3>22 种图片样式</h3><img src="'+embed(D/'qa/image_styles.png')+'" alt="22种图片水印样式"><h3>11 种 PDF 样式</h3><img src="'+embed(D/'qa/pdf_styles.png')+'" alt="11种PDF样式">'),
      ('quality','交付检查','<p>逐文件 SHA256、图片尺寸、全部二值掩码与透明度关系、图片叠加层精确重建、来源组隔离和逐份 PDF 对象结构已纳入检查。视觉检查覆盖全部照片底图及各样式代表件，未将每条记录宣称为人工逐张精标。</p><p>校验状态：'+str(qa.get('status'))+'；错误数：'+str(len(qa.get('errors',[])))+'；已检查不同文件：'+str(qa.get('unique_checked_files','待生成'))+'；跨分区来源组：'+str(qa.get('cross_split_source_groups','待生成'))+'。</p>'),
      ('manual','最后需要人工补充','<p>这 6 项保留真实来源属性，不用程序生成数据冒充。每项有独立收集目录与来源登记模板；未经接收检查不会自动加入训练清单。</p><table><thead><tr><th>编号</th><th>类型</th><th>首轮目标</th><th>制作与提交方法</th></tr></thead><tbody>'+task_html+'</tbody></table>'),
      ('usage','如何使用与对外表述','<p>在数据目录安装 requirements.txt 后，执行：</p><pre>python scripts/load_watermark_training.py --root /数据目录 --split train --kind image --limit 3\npython scripts/load_watermark_training.py --root /数据目录 --split train --kind pdf --limit 3</pre><p>加载器返回 RGB float32 图像、有效透明度和二值区域。同步裁切或缩放输入、参考和掩码；分区文件已经隔离来源，无需重新随机拆图。</p><blockquote>'+card['public_description']+'</blockquote><p>以上可以作为“已构建训练数据集”的准确表述。是否已被发布模型实际使用，需要由训练日志记录支持。</p>'),
    ]
    nav=''.join(f'<a href="#{sid}">{i+1:02d}　{title}</a>' for i,(sid,title,_) in enumerate(sections))
    body=''.join(f'<section id="{sid}"><h2>{title}</h2>{content}</section>' for sid,title,content in sections)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>训练数据集说明</title><style>
    *{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;font:15px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;background:#f2f5f9;color:#203047}aside{position:fixed;inset:0 auto 0 0;width:242px;background:white;border-right:1px solid #dce3ec;padding:30px 20px;overflow:auto}aside strong{display:block;font-size:18px;margin-bottom:24px}aside a{display:block;color:#51657e;text-decoration:none;padding:9px 10px;border-radius:6px;font-size:13px}aside a:hover,aside a.active{background:#e9f1ff;color:#245fbb}main{margin-left:242px;max-width:1500px;padding:30px}section{background:white;border:1px solid #dce3ec;border-radius:10px;padding:30px;margin-bottom:24px;scroll-margin-top:24px}h1{font-size:29px;margin-top:8px}h2{font-size:21px;margin:0 0 20px;border-bottom:1px solid #e3e8ef;padding-bottom:14px}h3{font-size:17px}p{color:#4b5c73}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #dce3ec;padding:11px 12px;text-align:left;vertical-align:top}th{background:#f0f5fc;color:#47607e}tr:nth-child(even){background:#fafcff}img{display:block;width:100%;height:auto;border:1px solid #e1e6ed;border-radius:7px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;padding:15px 0}.stats div{background:#f2f6fc;padding:16px;border-radius:7px;color:#5a6e87;font-size:12px}.stats b{display:block;font-size:29px;color:#254c7e}.notice,blockquote{background:#edf4ff;border-left:3px solid #4c79bd;padding:15px 18px;margin:18px 0}pre{background:#f3f6fa;overflow:auto;padding:16px;border-radius:6px;font-size:12px}a{color:#2863bc}@media(max-width:800px){aside{position:static;width:auto;padding:18px;display:flex;flex-wrap:wrap;gap:4px}aside strong{width:100%;margin:0 0 8px}aside a{font-size:12px;padding:5px 8px}main{margin:0;padding:12px}section{padding:19px;overflow:auto}.stats{grid-template-columns:repeat(2,1fr)}table{min-width:650px}}@media print{aside{display:none}main{margin:0;padding:0;max-width:none}section{border:0;padding:0;break-inside:auto}tr{break-inside:avoid}body{background:white}}
    </style><aside><strong>训练数据集 v1</strong>'''+nav+'''</aside><main>'''+body+'''</main><script>const ss=[...document.querySelectorAll('section')],aa=[...document.querySelectorAll('aside a')];function mark(){let x=ss[0];ss.forEach(s=>{if(s.getBoundingClientRect().top<100)x=s});aa.forEach(a=>a.classList.toggle('active',a.hash==='#'+x.id))}addEventListener('scroll',mark,{passive:true});mark();</script></html>'''
    (D/'DATASET_CARD.html').write_text(page)
    if (ROOT/'docs').is_dir():(ROOT/'docs/训练数据集说明.html').write_text(page)
    print(json.dumps(counts,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
