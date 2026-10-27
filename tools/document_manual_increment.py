"""Write the local HTML dataset preparation receipt and portable instructions."""
import base64, csv, hashlib, html, io, json, re, shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/watermark_training_increment_v1_1'
FONT=D/'assets/fonts/NotoSansCJKsc-Regular.otf'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def jsave(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def f(size):return ImageFont.truetype(str(FONT),size)
def picture(path,box):
    im=Image.open(path).convert('RGB');im.thumbnail(box,Image.Resampling.LANCZOS);return im
def sheet(grid,path,cell=(260,360)):
    cols=max(len(r) for r in grid);w,h=cell
    out=Image.new('RGB',(cols*w,len(grid)*h),'#eef2f6');dr=ImageDraw.Draw(out)
    for y,row in enumerate(grid):
        for x,(label,p) in enumerate(row):
            dr.text((x*w+12,y*h+8),label,font=f(16),fill='#24344a')
            im=picture(p,(w-24,h-48));out.paste(im,(x*w+(w-im.width)//2,y*h+38))
    out.save(path,quality=88);return path
def embed(p):
    im=Image.open(p).convert('RGB');im.thumbnail((1050,1600));b=io.BytesIO();im.save(b,'JPEG',quality=82)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def table(headers,body):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in r)+'</tr>' for r in body)+'</tbody></table></div>'

def sync_m03_received():
    source_root=ROOT/'data/watermark_training_v1'
    source=source_root/'manual_incoming/M03_实际纸张扫描/received_samples.jsonl'
    if not source.exists():return 0
    target_dir=D/'references/manual_scans';target_dir.mkdir(parents=True,exist_ok=True)
    refs_path=D/'references.jsonl'
    references=rows(refs_path) if refs_path.exists() else []
    known={r.get('id') for r in references}
    for item in rows(source):
        src=source_root/item['input_path'];name=Path(item['input_path']).name;dst=target_dir/name
        shutil.copyfile(src,dst)
        source_pdf_paths={};source_pdf_hashes={}
        pdf_dir=target_dir/'source_pdfs';pdf_dir.mkdir(parents=True,exist_ok=True)
        for role,key in [('clean','reference_clean_pdf'),('watermarked','reference_watermarked_pdf')]:
            source_pdf=source_root/item[key];source_dest=pdf_dir/source_pdf.name
            shutil.copyfile(source_pdf,source_dest)
            source_pdf_paths[role]=source_dest.relative_to(D).as_posix()
            source_pdf_hashes[role]=hashlib.sha256(source_dest.read_bytes()).hexdigest()
        if item['sample_id'] in known:
            next(r for r in references if r.get('id')==item['sample_id']).update(
                source_pdfs=source_pdf_paths,source_pdf_sha256=source_pdf_hashes)
            continue
        references.append({'id':item['sample_id'],'kind':'unpaired_actual_paper_scan_image','task':'M03',
            'paths':{'watermarked':f'references/manual_scans/{name}'},'clean':None,
            'training_admission':False,'split':None,'reserved_split':'train',
            'source_group':item['source_group'],'split_group':item['split_group'],
            'source_pdfs':source_pdf_paths,'source_pdf_sha256':source_pdf_hashes,
            'reason':'原始扫描图已接收；电子 PDF 未与扫描图逐像素配准，clean scan reference 和 scan-space mask 待完成。',
            'rights':'LicenseRef-User-Authorized-Project-Training','capture_software':item['capture_software'],
            'watermark_subtype':item['watermark_subtype'],'background':'photographed_printed_document_with_perspective_and_desktop',
            'appearance':['actual_paper_scan_capture',item['watermark_subtype']],
            'size':item['size_px'],'sha256':{'watermarked':item['sha256']},
            'alignment_status':'pending_homography_and_mask_review'})
    refs_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in references),encoding='utf-8')
    status_path=D/'manual_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    status['M03']={'state':'4_scan_images_received_pending_alignment_and_masks','underlying_documents':2,
        'page_images':4,'source_groups':['original:document_004','original:document_006'],'training_admission':False}
    jsave(status_path,status)
    return sum(r.get('task')=='M03' for r in references)

def main():
    current_task_readme=(D/'README.txt').read_text() if (D/'qa/training_tasks_validation.json').exists() else None
    m03_reference_count=sync_m03_received()
    strong=rows(D/'manifest.jsonl');weak=rows(D/'weak_pairs.jsonl');summary=json.loads((D/'build_summary.json').read_text())
    qa=json.loads((D/'qa/validation.json').read_text());assert qa['status']=='passed'
    with (D/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as out:
        keys=['id','kind','unit','scenario','subtype','label','split','source_group','split_group','background','appearance','difficulty','label_quality','rights','training_admission','paths']
        wr=csv.DictWriter(out,fieldnames=keys);wr.writeheader()
        for r in strong:wr.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in keys})
    selected=[next(r for r in strong if r['id']=='inc11_'+name+'__'+style)
              for name,style in [('cat_lowlight','logo_transparent'),('food_dish','tiled_diagonal'),
                                 ('food_soup','outlined'),('indoor_signage','center_white'),
                                 ('outdoor_building','mixed_color')]]
    photos=sheet([[ (r['base_id']+' / '+label,D/r['paths'][key]) for label,key in
                    [('原图','clean'),('样式','watermarked'),('精确掩码','mask')]] for r in selected],D/'qa/contact_photos.jpg',(330,320))
    pdfs=sheet([[(r['id']+' / '+label,D/r['paths'][key]) for label,key in
                 [('同版式参考','clean'),('导出页','watermarked'),('对象掩码','mask')]]
                for r in strong if r['kind']=='pdf_page'],D/'qa/contact_pdf.jpg',(360,510))
    wg=[]
    for r in weak:
        ann=json.loads((D/r['paths']['annotation']).read_text())
        marked=Image.open(D/r['paths']['watermarked']).convert('RGB');dr=ImageDraw.Draw(marked)
        for box in ann['boxes']:dr.rectangle(box,outline='#ed7c10',width=4)
        annotated=D/'qa'/(r['id']+'_regions.jpg');marked.save(annotated,quality=90)
        wg.append([(r['id']+' / 原始参考',D/r['paths']['original_clean']),
                   ('配准参考',D/r['paths']['aligned_clean']),('新增区域 / 粗框',annotated)])
    weakpic=sheet(wg,D/'qa/contact_weak.jpg',(350,475))
    stats=''.join(f'<div><b>{n}</b><span>{label}</span></div>' for n,label in [(121,'新增训练记录'),(3,'近似配对辅助件'),(5,'真实照片底图'),(1,'原生 PDF 底稿')])
    receipt=table(['材料','接收原件','本次整理','计数与状态'],[
        ['M01 真实手写','5 张','2 组前后照已配准、标出新增手写区域；1 张红色批注单图','同一讲稿的不同页；辅助集，不进默认像素监督训练'],
        ['M02 实际应用水印','醒图 2 张','醒图 1 组已配准；用户已删除两张抖音图','醒图版本未记录；两张抖音图及来源登记已删除'],
        ['M03 实际纸张扫描','2 份底稿 / 4 页扫描图','全能扫描王采集图原样归档；已关联 document_004 / document_006 干净与带水印 PDF 源稿','扫描/翻拍样本已接收；尚未完成透视配准与水印掩码，不进入默认像素监督训练'],
        ['M04 实际纸张翻拍','3 张带手写标记的照片','与 M01 交叉归类','不重复增加文件数、配对数或来源数'],
        ['M05 原生办公 PDF','3 份 × 3 页','1 份原始底稿 + 灰/黑两个变体；整理 6 个带水印页面配对','1 个文档来源，2 个变体；不是 6 份独立文档'],
        ['M06 真实背景','5 张普通照片 + 1 张业务表格','普通照片已生成 110 个正例、5 个负例；敏感表格原件隔离','5 张照片，按 4 个场景组；当前无人像底图']])
    scan_files=[
        ('document_004 · 第 1 页','scan_document_004_page_01.png'),
        ('document_004 · 第 2 页','scan_document_004_page_02.png'),
        ('document_006 · 第 1 页','scan_document_006_page_01.png'),
        ('document_006 · 第 2 页','scan_document_006_page_02.png'),
    ]
    scan_root='../data/watermark_training_v1/manual_incoming/M03_实际纸张扫描/received/'
    scan_gallery='<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:18px 0">'+''.join(
        f'<figure style="margin:0;border:1px solid #dfe6ee;border-radius:10px;padding:10px"><a href="{scan_root}{name}" target="_blank"><img src="{scan_root}{name}" alt="{label}" style="width:100%;height:420px;object-fit:contain;background:#eef1f4"></a><figcaption>{label}</figcaption></figure>'
        for label,name in scan_files)+'</div><p class="muted">4 张原始接收图均已保存并核验 SHA256。画面展示扫描应用采集到的桌面背景、页面透视和光照差异。已关联项目电子干净页与带水印 PDF；这些电子页不是扫描图的逐像素干净参考，待透视配准和逐页掩码复核后再纳入监督训练。</p>'
    inventory=table(['清单','数量','默认训练读取','标注'],[
        ['manifest.jsonl / manifest.csv','121 条','是，仅 train','110 精确合成图片对 + 5 无水印负例 + 6 PDF 页面对象配对'],
        ['weak_pairs.jsonl','3 组','否，保留训练组归属','2 组纸笔手写 + 1 组醒图；近似配准参考、有效区域；手写粗框'],
        ['references.jsonl','5 条','否','1 张用户红色批注图；4 页 M03 原始扫描图'],
        ['excluded/financial_handling.json','1 份处理记录','否','银行业务原件不入包；只保留新白底上的长网格线示意'],
        ['旧版 watermark_training_v1','2,432 条不变','沿用旧版划分','本增量独立保存，不重写 v1 清单、验证集、测试集或旧 ZIP']])
    baseinfo=json.loads((D/'sources/photo_bases.json').read_text())
    bases=table(['底图','背景 / 难点','来源组'],[[b['id'],b['background'],b['source_group']] for b in baseinfo])
    reg=table(['配对','有效匹配','内点误差中位数 / P95','训练含义'],[
        [r['id'],f"{r['registration']['inliers']} / {r['registration']['matches']}",
         f"{r['registration']['median_px']:.3f} / {r['registration']['p95_px']:.3f} px",
         '特征点配准指标；不能保证页面每个像素、亮度或纸张弯曲完全一致'] for r in weak])
    checktable=table(['检查','结果 / 范围'],[
        ['20 个保留原件','16 个原有文件 + 4 页 M03 扫描原图；SHA256 与接收登记一致，原件未改写'],
        ['110 组合成正例','逐条从干净参考 + RGBA 叠加层精确重建输入；掩码等于 alpha>0'],
        ['5 个负例','输入等于参考；mask / alpha 全零；场景中的合法标识和文字保留'],
        ['6 个 PDF 页面','逐页确认已标注水印对象；对象区域 1px 边缘以外无变化'],
        ['PDF 抗锯齿边缘','隔离 alpha 的 8bit 取整可能留下 1 个灰度级差；已记录，不宣称完全精确的 RGB 逆合成'],
        ['来源隔离','新增 121 条均为训练组；同源衍生件不随机分到验证/测试'],
        ['原始敏感文件与第三方图片','金融原件不进入发布包；两张抖音图已按用户要求删除'],
        ['文件完整性',f"{qa['indexed_distinct_files']} 个不同清单引用文件已检查路径、哈希和尺寸"]])
    sections=[
        ('summary','01','整理完成情况',f'<p class="eyebrow">图片与 PDF · 样本制备记录 · 2026-10-26</p><h1>人工样本整理与入库</h1><p class="lead">收到的素材已经分组、配准、补充标注，并形成可独立读取的训练增量。</p><div class="stats">{stats}</div><div class="notice">本轮新增：全能扫描王扫描/翻拍图 4 页（2 份底稿），来自已有打印底稿 document_004 与 document_006。原始附件和 SHA256 已归档，配对标注待做。<br>“121 条”仍是已标注训练记录数；这 4 页扫描图目前是未配准接收样本，不计入该数，也不进入默认像素监督训练。</div><p>来源使用依据已登记：你明确确认普通照片、讲稿和 PDF 可授权用于本软件训练。该授权不等同于对整包公开再分发的许可。</p>'),
        ('received','02','六类材料的状态',receipt+scan_gallery+'<p class="muted">这里表示本轮已接收的材料，不表示六类的所有设备、软件、笔迹、背景和难度组合已经穷尽覆盖。</p>'),
        ('inventory','03','哪些进入训练',inventory+'<p>新增 121 条为独立增量。旧版的 PDF 清单按文档计数，本次按页面计数，因此不直接把两者相加成“独立样本总量”。</p>'),
        ('labels','04','配对与标注怎么做','<h3>真实照片上的已知合成水印</h3><p>5 张底图各覆盖原有的 22 种图片样式：局部文字、时间戳、网址、模拟签名、透明与不透明 Logo、居中大字、白色大字、斜向平铺、不等距密集文字、点阵、细线网格、小角标、边缘裁切、深色大字、不透明块、极淡、渐变、描边、阴影、多重叠加和多色水印。合成签名仍标记为模拟笔迹。</p><p>保留原始底图；训练版本最长边缩至 1280 像素。每条含 clean、watermarked、overlay_rgba、alpha、mask，以及结构场景、外观、背景、难度、来源组、种子、参数与 SHA256。</p><p>难度字段是内容属性标签，并非模型实测分数；覆盖多种属性也不等于覆盖所有属性组合。</p>'+bases+'<h3>真实软件生成的 PDF</h3><p>原文件完整保存。针对这两个已确认的受控导出件，逐页标记已知的“内部演示稿”文字对象，在相同导出版式下生成参考页和透明对象层；没有把原始 Letter 版与 A4 导出页直接当作逐像素对齐数据。</p><p>灰色水印为不透明灰字；黑色变体的对象不透明度约 84.71%。对象与正文共用内容流。原始字体导出编码影响文字提取，不能承诺与源 PDF 的 Unicode 提取结果完全相同。</p><p>PDF 掩码表示已知对象的渲染覆盖范围；alpha 包含字体抗锯齿覆盖和填充不透明度。不要把它理解成原应用导出的完整编辑图层。</p>'),
        ('weak','05','真实配对的使用边界','<p>两组手写照已做透视配准，保留原始前后照、变换矩阵、有效区域和手写区域粗框。醒图前后导出也已检查并配准。有效区域图只表示参考图能覆盖哪里，<strong>不是水印掩码</strong>。</p><p>这 3 组可用于人工观察、区域标注或后续适配弱监督方法；默认读取器不混入精确像素监督清单。没有用前后差分伪造真实笔迹 alpha，也没有识别或转录签名姓名。原来就有的红点、红三角属于应保留内容。</p>'+reg+f'<img class="sheet" src="{embed(weakpic)}" alt="真实配对原图、配准参考和手写区域粗框">'),
        ('preview','06','训练样本预览','<p>以下展示底图、带水印版本和标注。场景中的地铁标志、路线文字、建筑栏杆等属于原始内容。</p>'+f'<img class="sheet" src="{embed(photos)}" alt="五种照片背景及合成水印标注"><h3>两个 PDF 变体的全部六页</h3><img class="sheet" src="{embed(pdfs)}" alt="全部六个PDF页面的参考、带水印页和对象掩码">'),
        ('quality','07','质量与来源检查',checktable+'<p>自动检查明细见 qa/validation.json；PDF 对象标注明细见 pdf/*/object_annotation.json；来源与授权见 sources/。文件索引与压缩包校验结果随发布记录提供。</p><p>金融原件仍在受限接收目录。网格线示意不含原图像素或文字，且未作为训练背景；要补业务样本时，应另用虚构数据底稿或可用的完整脱敏件。</p>'),
        ('usage','08','拿到后怎么使用','<p>解压后，先阅读 README.txt；JSONL 用于程序读取，UTF-8 CSV 用于表格筛选。默认读取器只读取已入库的 121 条。</p><pre>python scripts/load_manual_increment.py --root /解压目录 --limit 3\npython scripts/load_manual_increment.py --root /解压目录 --kind pdf_page --limit 2\npython scripts/validate_manual_increment.py --root /解压目录</pre><p>图片输入、参考为 RGB uint8；mask 与 alpha 为单通道 uint8。训练时自行归一化到 0～1；PDF 页已提供 150 DPI 图像与原生 PDF 对象版本。weak_pairs.jsonl 和 references.jsonl 不会被默认训练读取器纳入。</p><div class="quote">可用表述：我们建立了图片与 PDF 可见水印训练数据集，结合已知参数的合成配对、真实软件 PDF 对象标注，并单列真实手写与应用导出辅助样本。<br>本轮新增 110 组合成图片正例、5 个负例、6 个真实软件 PDF 页面配对。</div><p>不能据此声称真实场景已经全部覆盖，也不能把未授权第三方下载图算成正式训练数据。</p>'),
        ('remaining','09','剩余项与后续扩充',table(['项目','当前处理'],[
            ['实际纸张扫描','已接收 2 份底稿共 4 页扫描图；透视配准和掩码 QA 完成前不并入默认训练清单。'],
            ['真实手写与翻拍','已有首批；以后扩充不同笔、纸张、底稿、遮挡正文与拍摄条件。'],
            ['设备 / 应用水印','已有醒图受控样本；相机原生日期/机型角标、其他应用及版本信息仍可扩充。'],
            ['办公软件原生 PDF','已有一个 macOS 打印导出流程；其他软件的图片水印、图章和对象结构仍需真实流程样本。'],
            ['人物与业务背景','有动物、餐食、标牌、建筑；当前没有授权人像，银行原件不能直接使用。'],
            ['抖音两张下载图','已按用户要求删除原图、预览和来源登记，不再列为待补材料。']])+'<p>当前工作已完成本轮材料整理与可用数据制备；接下来可以直接交给训练代码读取，或继续按上述细分补样。本页不包含去水印效果结论。</p>')]
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f8;color:#243247;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}aside{position:fixed;inset:0 auto 0 0;width:234px;padding:30px 18px;background:#14253b;color:white;overflow:auto}aside strong{display:block;font-size:18px;padding:0 10px 20px}aside a{display:block;color:#cfdaea;text-decoration:none;padding:10px;border-radius:8px;font-size:14px}aside a:hover,aside a.active{background:#294360;color:white}main{max-width:1340px;margin-left:234px;padding:36px 44px 80px}section{scroll-margin-top:22px;background:white;border:1px solid #e3e8ee;border-radius:14px;padding:32px;margin-bottom:24px}h1{font-size:36px;line-height:1.35;margin:12px 0 16px}h2{font-size:23px;margin:0 0 22px}h3{margin-top:28px;font-size:18px}.eyebrow,.muted{color:#6d7b8d;font-size:14px}.lead{font-size:19px;color:#50627b}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.stats div{background:#f0f5fb;border-radius:10px;padding:18px}.stats b{display:block;color:#205e9d;font-size:34px;line-height:1.4}.stats span{font-size:14px;color:#5e7085}.notice{background:#eaf5f2;border-left:4px solid #27856b;padding:18px 20px;border-radius:6px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e1e7ed;text-align:left;vertical-align:top;padding:13px 14px}th{background:#eff3f8;color:#48627f}td{word-break:break-word}td:first-child{font-weight:600;min-width:135px}.sheet{display:block;max-width:100%;height:auto;margin:20px auto;border:1px solid #e0e6ec;border-radius:8px}pre{overflow:auto;padding:20px;background:#14253b;color:#e9f3ff;font:13px/1.8 ui-monospace,monospace;border-radius:8px}.quote{border-left:4px solid #447ab1;padding:16px 20px;background:#f1f6fc}a{color:#215e99}footer{color:#718196;font-size:13px;padding:5px 10px}@media(max-width:950px){aside{width:190px}main{margin-left:190px;padding:20px}.stats{grid-template-columns:repeat(2,1fr)}section{padding:22px}}@media(max-width:620px){aside{position:relative;width:auto;padding:16px}aside strong{padding-bottom:8px}aside a{display:inline-block;padding:6px 10px;font-size:12px}main{margin:0;padding:12px}h1{font-size:28px}.stats b{font-size:28px}section{padding:18px}td{min-width:140px}}@media print{aside{display:none}main{margin:0;padding:0}section{break-inside:avoid;border:0}body{background:white}}'''
    nav='<aside><strong>样本整理 · v1.1</strong>'+''.join(f'<a href="#{id}">{n}　{title}</a>' for id,n,title,_ in sections)+'</aside>'
    content=''.join(f'<section id="{id}">'+('' if id=='summary' else f'<h2>{title}</h2>')+body+'</section>' for id,_,title,body in sections)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>人工样本整理与入库</title><style>'+css+'</style>'+nav+'<main>'+content+'<footer>来源：用户提交原件与直接授权；data/manual_collections/*/manifest.json；本增量 manifest.jsonl、weak_pairs.jsonl、sources/、qa/。所有计数可核对。</footer></main><script>const links=[...document.querySelectorAll("aside a")];const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.toggle("active",a.hash==="#"+e.target.id));}})},{rootMargin:"-10% 0px -70% 0px"});document.querySelectorAll("section").forEach(s=>io.observe(s));</script></html>'
    (ROOT/'docs/样本整理与入库.html').write_text(page,encoding='utf-8')
    pack_page=page.replace('../data/watermark_training_v1/manual_incoming/M03_实际纸张扫描/received/','references/manual_scans/')
    (D/'DATASET_CARD.html').write_text(pack_page,encoding='utf-8')
    readme='''人工样本训练增量 v1.1 / 2026-10-26

默认训练清单：121 条（110 图片正例、5 负例、6 PDF 页面配对）。
来源：5 张照片（4 个保守场景分组）+1 份 PDF 底稿（2 变体×3 页）。
全部新增记录只进入 train。冻结 v1 的验证集与测试集不变。
这是图片/页粒度，不能冒称 121 份独立文档，也不能直接与 v1 文档粒度相加。

另有 3 组近似真实配对，默认不进入像素监督训练；有效区域不是水印掩码。
references.jsonl：共 5 条，含 1 张用户红色批注单图和 4 页 M03 实扫/翻拍原图。两张第三方下载图及登记已按用户要求删除。
实际纸张扫描 M03：已接收 document_004、document_006 共 4 页全能扫描王图片，原件在 references/manual_scans/；对应的电子干净与带水印 PDF 源稿也一并保存在 references/manual_scans/source_pdfs/。尚未透视配准或完成像素掩码，因此不计入默认 121 条监督训练记录。

使用依据：用户直接确认上述照片、讲稿和 PDF 可授权用于本软件训练。
不包含整包公开再分发授权；字体另见 licenses/NotoSansCJK-OFL.txt。
所有合成文字/几何 Logo/模拟笔迹属于项目构造，不冒称真实设备或真实签名。
金融原件未进入包；两张抖音图及登记已按用户要求删除。excluded/ 的网格线示意不是业务训练底稿。

环境安装：pip install -r requirements.txt
读取：python scripts/load_manual_increment.py --root . --limit 3
校验：python scripts/validate_manual_increment.py --root .
图像：RGB uint8；mask/alpha：L uint8；可除以 255 归一化。
PDF 页面：150DPI；参数包含原生文件、页码和对象标注关系。
PDF 对象 alpha 边缘取整会有最多一个灰度级的邻接差，详见 qa/。
默认读取器自动排除 weak_pairs 与 references。

资料说明：DATASET_CARD.html；清单 manifest.jsonl / manifest.csv。
生成代码存于 scripts/，重建需原本接收目录及冻结 v1 字体资源，不能仅凭增量 ZIP 重建原始接收件。
验证与读取在解压后可独立运行，不依赖原项目路径。
'''
    (D/'README.txt').write_text(readme,encoding='utf-8')
    (D/'requirements.txt').write_text('numpy>=2.0\nPillow>=10\nopencv-python-headless>=4.10\npymupdf>=1.25\npypdf>=5\n')
    for fn in ['load_manual_increment.py','validate_manual_increment.py','document_manual_increment.py']:
        shutil.copyfile(ROOT/'tools'/fn,D/'scripts'/fn)
    note={'role':'executable companion to quality checks','code':'scripts/validate_manual_increment.py',
          'checks':['intake hashes','known synthetic reconstruction','PDF object support','source-group partitions','excluded originals']}
    nb={'cells':[{'cell_type':'markdown','metadata':{},'source':['# 人工样本增量检查\n','验证代码与数据清单随包保存；此笔记本不运行去水印模型。']},
                 {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'source':['from pathlib import Path\n','import sys,json\n','root=Path.cwd()\n',
                            'sys.path.insert(0,str(root/"scripts"))\n','from validate_manual_increment import validate\n',
                            'checks=validate(root)\n','checks\n']}],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'audit':note},'nbformat':4,'nbformat_minor':5}
    jsave(D/'质量检查.ipynb',nb)
    summary['unpaired_references']=len(rows(D/'references.jsonl'))
    summary['validation_status']='passed'
    if not (D/'qa/visual_review.json').exists():summary['visual_review_status']='pending'
    jsave(D/'build_summary.json',summary)
    # Update only live documentation, never the frozen v1 card.
    live=ROOT/'docs/训练数据集说明.html';old=live.read_text()
    banner='<p class="notice" id="manual-received-status"><strong>本轮已整理：</strong>新增 121 条训练记录（110 组合成图片正例、5 个负例、6 个真实软件 PDF 页面配对），另有 3 组近似配对和 5 条非配对参考登记，其中包括 4 页全能扫描王纸张扫描/翻拍图（2 份底稿）。扫描图原件已归档并关联 document_004 / document_006 的电子干净与带水印源稿；透视配准和逐页水印掩码尚待完成，因此暂不计入默认像素监督训练。你已确认本项目素材可用于训练；金融原件不进入训练包，两张抖音图及登记已删除。原 v1 的 2,432 条保持冻结。<a href="样本整理与入库.html">查看本轮整理、标注、来源与剩余项</a>。</p>'
    old,n=re.subn(r'<p class="notice" id="manual-received-status">.*?</p>',lambda m:banner,old,flags=re.S);assert n==1
    old=old.replace('真实手写、设备水印、实际扫描/翻拍和跨软件原生文件仍按人工清单补充。','人工接收与入库状态以《样本整理与入库》为准；M03 已收 2 份底稿共 4 页，原始扫描图待配准和掩码标注。')
    old=old.replace('08　最后需要人工补充','08　人工样本与扩充状态').replace('<h2>最后需要人工补充</h2>','<h2>人工样本与扩充状态</h2>')
    live.write_text(old,encoding='utf-8')
    report=ROOT/'docs/水印覆盖与竞品评测.html';text=report.read_text()
    text=text.replace('真实手写、设备水印、实扫、翻拍、跨软件原生文件及真实业务背景仍按 M01—M06 补充。',
        '本轮人工素材已整理为独立增量：110 组合成图片正例、5 个负例、6 个真实软件 PDF 页面配对，另有 3 组近似配对辅助件；M03 已收到 2 份底稿共 4 页全能扫描王原图，待配准和掩码 QA。<a href="样本整理与入库.html">查看本轮样本整理与入库 ↗</a>。')
    text=text.replace('自动部分已完成：正式训练清单、成对文件、掩码、许可依据、分区、加载器和检查记录；人工 M01—M06 待补',
        '自动部分及本轮人工素材整理已完成；新增 121 条图片/页训练记录，3 组近似配对单列；M03 收到 4 页扫描图，待配准和掩码 QA')
    text=text.replace('自动 12 类基础场景已建；人工项按清单接收后再确认真实来源覆盖',
        '自动 12 类基础场景已建；首批人工素材已接收和登记，纸张扫描图待配准及掩码标注，其他细分仍按清单扩充')
    report.write_text(text,encoding='utf-8')
    if current_task_readme is not None:
        (D/'README.txt').write_text(current_task_readme,encoding='utf-8')
        from sync_real_training_docs import sync
        sync()
    print('HTML and portable instructions written',flush=True)

if __name__=='__main__':main()
