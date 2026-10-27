"""Build paired, provenance-aware image/PDF watermark training corpus v1.

Run with the repository venv. No removal inference. All watermark assets are
project-created. Sources/splits, exact image alpha, and PDF object types are
recorded. Photographs must first be fetched by fetch_cc0_training_bases.py.
"""
from __future__ import annotations
import argparse, collections, hashlib, io, json, math, random, shutil, os, re
from pathlib import Path
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT=Path(__file__).resolve().parents[1]
DEST=Path(os.environ.get('WM_DATASET_ROOT',ROOT/'data/watermark_training_v1'))
FONT=DEST/'assets/fonts/NotoSansCJKsc-Regular.otf'
SEED=20261026
IMAGE_STYLES={
 'local_text':('C01','局部文字'), 'timestamp':('C01','日期时间戳'), 'website':('C01','网址文字'),
 'signature':('C02','模拟手写签名'), 'logo_transparent':('C02','透明几何 Logo'), 'logo_opaque':('C02','不透明几何 Logo'),
 'center_text':('C03','居中大字'), 'center_white':('C03','白色半透明大字'),
 'tiled_diagonal':('C04','斜向平铺文字'), 'tiled_irregular':('C04','非等距密集文字'),
 'dot_grid':('C05','点阵'), 'line_net':('C05','细线网格'),
 'tiny_corner':('C06','高清小角标'), 'cropped_edge':('C06','边缘裁切文字'),
 'heavy_dark':('C07','深色大字'), 'opaque_block':('C07','大面积完全不透明图案'),
 'faint':('C08','极淡文字'), 'gradient':('C08','渐变文字'), 'outlined':('C08','描边文字'),
 'shadow':('C08','带阴影文字'), 'multiple':('C08','多重叠加'), 'mixed_color':('C08','多色文字与图案'),
}
PDF_STYLES={
 'independent_text':('C09','PDF 独立文字'),
 'unicode_text':('C09','PDF 中文编码文字'), 'shared_content_stream':('C09','PDF 水印正文共用内容流'),
 'independent_image':('C10','PDF 独立透明图片'), 'vector_paths':('C10','PDF 矢量路径'),
 'outlined_text':('C10','PDF 描边文字'),
 'nested_form':('C11','PDF 嵌套 Form XObject'), 'stamp_annotation':('C11','PDF 图章批注'),
 'optional_content':('C11','PDF 可选图层 OCG'),
 'raster_flattened':('C12','PDF 数字栅格化压平'), 'mixed_storage':('C12','PDF 文字与图像混合存储'),
}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def savejson(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def font(n):return ImageFont.truetype(str(FONT),max(8,int(n)))
def filepart(path):return str(path.relative_to(DEST))
def split_at(i,n):return 'train' if i<int(n*.75) else 'validation' if i<int(n*.875) else 'test'
def seed_for(s):return int(hashlib.sha256((str(SEED)+':'+s).encode()).hexdigest()[:12],16)
def png_bytes(im):b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()

def clean_pdf(idx,path):
    """Original fictitious text, tables and charts. No imported business docs."""
    rng=random.Random(SEED+idx);doc=fitz.open()
    topics=['设备巡检记录','实验观察与数据登记','季度资源配置说明','项目工作进度记录']
    for pno in range(2):
        p=doc.new_page(width=595,height=842)
        p.draw_rect((0,0,595,842),color=None,fill=(1,1,1))
        p.draw_rect((34,34,561,45),fill=(.12,.26,.44),color=None)
        p.insert_text((40,78),topics[idx%4]+f' / {idx+1:03d}',fontname='china-s',fontsize=20,color=(.10,.18,.3))
        p.insert_text((40,101),f'FICTITIOUS TRAINING MATERIAL | GROUP {idx+1:03d} | PAGE {pno+1}',fontsize=8,color=(.4,.45,.5))
        p.insert_text((40,125),'所有名称、编号与数值均为本项目构造的虚拟内容。',fontname='china-s',fontsize=10)
        if idx%4 in [0,1]:
            for j in range(30):
                y=154+j*18
                if idx%4==0:
                    line=f'{j+1:02d}  第{idx+1}组记录：检查区域{chr(65+j%8)}，读数{rng.randrange(10,980)}，确认时间及备注完整。'
                    p.insert_text((40,y),line,fontname='china-s',fontsize=10)
                else:
                    line=f'{j+1:02d}  Observation group {idx+1}: sample {rng.randrange(100,9999)}, value {rng.random()*100:.3f}.'
                    p.insert_text((40,y),line,fontsize=10)
        elif idx%4==2:
            for j in range(17):
                y=148+j*25
                p.draw_line((40,y),(555,y),color=(.7,.74,.79),width=.5)
                if j<16:
                    p.insert_text((48,y+17),f'Item {j+1:02d}',fontsize=10)
                    p.insert_text((200,y+17),str(rng.randrange(1000,9000)),fontsize=10)
                    p.insert_text((350,y+17),f'{rng.random()*100:.2f} %',fontsize=10)
                    p.insert_text((470,y+17),'Approved',fontsize=9)
            for x in [40,180,330,450,555]:p.draw_line((x,148),(x,548),color=(.7,.74,.79),width=.5)
            p.insert_text((40,590),'数值仅用于构造表格，不对应任何真实个人或组织。',fontname='china-s',fontsize=10)
        else:
            for j in range(9):
                x=50+j*54;ht=rng.randint(60,200)
                p.draw_rect((x,410-ht,x+30,410),fill=(.1+j*.04,.4,.7-j*.035),color=None)
                p.insert_text((x,430),str(j+1),fontsize=9)
            for j in range(10):
                p.insert_text((40,475+j*20),f'区段{j+1}：安排资源{rng.randrange(10,90)}份；记录中保留中文、英文和数字。',fontname='china-s',fontsize=10)
        # Legitimate footer elements are part of the clean content, not masks.
        p.draw_line((40,774),(555,774),color=(.5,.6,.7))
        p.insert_text((40,795),f'INTERNAL SAMPLE {idx+1:03d} / {pno+1:02d}',fontsize=9)
        p.insert_text((410,795),'正文编号 / 非水印',fontname='china-s',fontsize=9)
    path.parent.mkdir(parents=True,exist_ok=True)
    doc.set_metadata({'title':f'Original fictitious training source {idx+1:03d}','author':'Project-created','subject':'No real personal data'})
    doc.save(path,garbage=4,deflate=True);doc.close()

def make_original_base(kind,i):
    rng=np.random.default_rng(SEED+1000+i+(0 if kind=='charts' else 100))
    w,h=1152,864
    if kind=='textures':
        yy,xx=np.mgrid[:h,:w];idx=i%4
        if idx==0:v=(np.sin(xx/(8+i))+np.cos(yy/(11+i)))*25+125
        elif idx==1:v=(np.sin(xx*.03+np.cos(yy*.02)*2)+np.sin((xx+yy)*.04))*35+120
        elif idx==2:v=((xx//(20+i)+yy//(20+i))%2)*100+60
        else:v=rng.normal(128,35,(h,w))
        channels=[np.clip(v+rng.normal(0,8,(h,w))+k*18,0,255) for k in range(3)]
        im=Image.fromarray(np.stack(channels,-1).astype('uint8'))
    else:
        im=Image.new('RGB',(w,h),(245,248,251));d=ImageDraw.Draw(im)
        d.text((65,40),f'项目数据图表 / 构造底图 {i+1:02d}',font=font(32),fill=(30,50,80))
        d.line((70,700,1090,700),fill=(60,80,100),width=2);d.line((70,150,70,700),fill=(60,80,100),width=2)
        for j in range(8):
            x=110+j*115;v=int(rng.integers(120,500))
            d.rectangle((x,700-v,x+65,699),fill=(35+j*12,90+j*6,180-j*8))
            d.text((x,715),f'Q{j+1}',font=font(20),fill=(40,50,65))
            d.text((x,660-v),str(v),font=font(18),fill=(45,60,80))
        pts=[(110+j*115,int(rng.integers(180,450))) for j in range(8)]
        d.line(pts,fill=(200,75,75),width=4)
        d.text((75,790),'构造图例及坐标轴属于正文内容。',font=font(22),fill=(75,85,100))
    return im

def collect_bases():
    photos=json.loads((DEST/'sources/commons_cc0.json').read_text())
    if len(photos)!=32:raise ValueError('Expected exactly 32 CC0 bases')
    bases=[]
    # Conservative grouping by photographer avoids burst / near-duplicate
    # photographs crossing partitions. Exact-file identity is also retained.
    creators={}
    for r in photos:
        creator=re.sub(r'<[^>]+>','',r['author_html']).strip().lower()
        creator=re.sub(r'\s+',' ',creator) or r['source_group']
        r['creator_group']='commons-creator:'+hashlib.sha256(creator.encode()).hexdigest()[:16]
        creators.setdefault(r['creator_group'],[]).append(r)
    ranked=sorted(creators,key=lambda g:hashlib.sha256(g.encode()).hexdigest())
    # Small creator groups are reserved intact; do not force equal image counts.
    ranked=sorted(ranked,key=lambda g:len(creators[g]))
    creator_splits={g:('test' if i==0 else 'validation' if i==1 else 'train') for i,g in enumerate(ranked)}
    if len(ranked)<3:raise ValueError('Need at least three independent photographer groups')
    for i,r in enumerate(photos):
        bases.append(dict(id=r['id'],path=r['normalized_path'],source_group=r['source_group'],
                          split_group=r['creator_group'],split=creator_splits[r['creator_group']],background='photo_'+r['background'],
                          provenance='downloaded_cc0_photo',license='CC0-1.0',source_evidence=r['evidence_path']))
    for i in range(32):
        bid=f'document_{i+1:03d}';pdf=DEST/f'bases/documents/{bid}.pdf';clean_pdf(i,pdf)
        with fitz.open(pdf) as doc:pix=doc[0].get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False)
        path=pdf.with_suffix('.png');pix.save(path)
        bases.append(dict(id=bid,path=filepart(path),pdf=filepart(pdf),source_group='original:'+bid,
                          split=split_at(i,32),background=['document_zh','document_en','table','document_chart'][i%4],
                          provenance='project_original_document',license='LicenseRef-Project-Original'))
    for kind in ['charts','textures']:
        for i in range(16):
            bid=f'{kind}_{i+1:03d}';path=DEST/f'bases/{kind}/{bid}.png';path.parent.mkdir(parents=True,exist_ok=True)
            make_original_base(kind,i).save(path)
            bases.append(dict(id=bid,path=filepart(path),source_group='original:'+bid,split=split_at(i,16),
                              background='chart' if kind=='charts' else 'procedural_texture',
                              provenance='project_original_procedural',license='LicenseRef-Project-Original'))
    for b in bases:
        b.setdefault('split_group',b['source_group'])
        b['sha256']=digest(DEST/b['path'])
        with Image.open(DEST/b['path']) as im:b['size']=list(im.size)
    savejson(DEST/'bases/manifest.json',bases)
    return bases

def watermark_layer(size,style,key,split):
    w,h=size;s=min(w,h);rng=random.Random(seed_for(key));layer=Image.new('RGBA',size)
    color=(rng.randrange(20,160),rng.randrange(20,160),rng.randrange(20,160))
    opacity=rng.choice([.15,.3,.5,.7]);label={'train':'训练','validation':'验证','test':'测试'}[split]
    token=f'{label}素材 {seed_for(key)%10000:04d}'
    params=dict(style=style,seed=seed_for(key),opacity=opacity,color=list(color),text=token,
                asset_partition=split,asset_id=f'{split}:{key}',watermark_origin='project_synthetic')
    def text_tile(text,fs,fill,angle=0,stroke=0):
        f=font(fs);box=f.getbbox(text,stroke_width=stroke);pad=max(8,stroke+4)
        t=Image.new('RGBA',(max(1,box[2]-box[0]+pad*2),max(1,box[3]-box[1]+pad*2)))
        ImageDraw.Draw(t).text((pad-box[0],pad-box[1]),text,font=f,fill=fill,stroke_width=stroke,stroke_fill=fill)
        return t.rotate(angle,Image.Resampling.BICUBIC,expand=True) if angle else t
    def paste(t,x,y):layer.alpha_composite(t,(int(x),int(y)))
    def centered(text=token,fs=None,fill=None,angle=0):
        t=text_tile(text,fs or s*.09,fill or (*color,round(opacity*255)),angle)
        paste(t,(w-t.width)/2,(h-t.height)/2)
    def logo(x,y,side,alpha):
        d=ImageDraw.Draw(layer);cx=x+side/2;cy=y+side/2
        n=rng.randrange(5,10);points=[]
        for j in range(n*2):
            a=j*math.pi/n;r=side*(.48 if j%2==0 else rng.uniform(.18,.30))
            points.append((cx+math.cos(a)*r,cy+math.sin(a)*r))
        d.polygon(points,fill=(*color,alpha));d.ellipse((cx-side*.12,cy-side*.12,cx+side*.12,cy+side*.12),fill=(240,245,250,alpha))
        d.text((x+side*.05,y+side*.74),f'{split[0].upper()}{seed_for(key)%1000:03}',font=font(side*.14),fill=(255,255,255,alpha))
    a=round(opacity*255)
    if style in ['local_text','timestamp','website']:
        text=token if style=='local_text' else f'2026-10-{1+seed_for(key)%28:02d} {rng.randrange(24):02d}:{rng.randrange(60):02d}' if style=='timestamp' else f'https://{split}-{seed_for(key)%1000}.example.test'
        t=text_tile(text,s*.035,(*color,a))
        placement=rng.choice(['bottom_right','top_left','body_overlap'])
        x,y=(w-t.width-s*.03,h-t.height-s*.03) if placement=='bottom_right' else (s*.03,s*.03) if placement=='top_left' else ((w-t.width)/2,h*.45)
        paste(t,x,y);params.update(text=text,placement=placement)
    elif style=='signature':
        d=ImageDraw.Draw(layer);x0=w*.58;y0=h*.84
        pts=[(x0+j*s*.003,y0+math.sin(j*.22)*s*.026+math.sin(j*.71)*s*.009) for j in range(105)]
        d.line(pts,fill=(*color,a),width=max(2,int(s*.003)))
        d.line((x0,y0+s*.04,x0+s*.32,y0+s*.015),fill=(*color,a),width=2)
        params['signature_kind']='procedural_stroke_not_human_handwriting'
    elif style.startswith('logo_'):
        if style=='logo_opaque':params['opacity']=1.;a=255
        logo(w-s*.25,h-s*.25,s*.21,a)
    elif style in ['center_text','center_white','heavy_dark','faint']:
        if style=='center_white':color=(255,255,255)
        if style=='heavy_dark':color=(10,12,18);opacity=.85
        if style=='faint':color=(95,100,110);opacity=rng.choice([.01,.025,.05])
        params.update(color=list(color),opacity=opacity)
        centered(fs=s*.11,fill=(*color,round(opacity*255)),angle=25)
    elif style in ['tiled_diagonal','tiled_irregular']:
        irregular=style=='tiled_irregular';fs=s*(.045 if irregular else .055)
        t=text_tile(token,fs,(*color,a),32)
        step_x=max(30,int(t.width*(.78 if irregular else 1.25)));step_y=max(30,int(t.height*(.7 if irregular else 1.5)))
        for y in range(-t.height,h+t.height,step_y):
            for x in range(-t.width,w+t.width,step_x):paste(t,x+(rng.randrange(-12,13) if irregular else 0),y+(rng.randrange(-10,11) if irregular else 0))
        params['rotation_degrees']=32;params['irregular']=irregular
    elif style=='dot_grid':
        d=ImageDraw.Draw(layer);step=max(18,int(s*.05));rad=max(1,int(s*.0025))
        family=rng.choice(['square','rectangular','staggered']);step_y=int(step*1.4) if family=='rectangular' else step
        for j,y in enumerate(range(step_y//2,h,step_y)):
            shift=step//2 if family=='staggered' and j%2 else 0
            for x in range(step//2+shift,w,step):d.ellipse((x-rad,y-rad,x+rad,y+rad),fill=(*color,a))
        params.update(spacing_px=[step,step_y],radius_px=rad,grid_family=family)
    elif style=='line_net':
        d=ImageDraw.Draw(layer);step=max(24,int(s*.08));lw=max(1,int(s*.0012))
        for x in range(-h,w+h,step):
            d.line((x,0,x+h,h),fill=(*color,a),width=lw)
            d.line((x,0,x-h,h),fill=(*color,a),width=lw)
        params.update(spacing_px=step,line_width_px=lw)
    elif style=='tiny_corner':
        fs=max(9,int(s*.012));t=text_tile(f'{split.upper()}-{seed_for(key)%10000:04}',fs,(*color,170))
        paste(t,w-t.width-4,h-t.height-4);params.update(font_size_px=fs,opacity=170/255)
    elif style=='cropped_edge':
        t=text_tile(token,s*.09,(*color,a),12);paste(t,w-t.width*.55,h-t.height*.6);params['edge_crop']=True
    elif style=='opaque_block':
        params['opacity']=1.;d=ImageDraw.Draw(layer)
        fw,fh=rng.choice([(.45,.25),(.65,.5),(.9,.75)])
        d.rectangle((w*(1-fw)/2,h*(1-fh)/2,w*(1+fw)/2,h*(1+fh)/2),fill=(*color,255));centered(fs=s*.07,fill=(245,245,245,255))
        params['block_fraction_requested']=fw*fh
    elif style in ['gradient','outlined','shadow']:
        t=text_tile(token,s*.1,(*color,210))
        if style=='gradient':
            arr=np.asarray(t).copy();ramp=np.linspace(.02,.85,t.width)[None,:]
            arr[:,:,3]=(arr[:,:,3]*ramp).astype('uint8');t=Image.fromarray(arr);params.update(opacity_min=.02,opacity_max=.85)
        elif style=='outlined':
            f=font(s*.1);box=f.getbbox(token,stroke_width=2);t=Image.new('RGBA',(box[2]-box[0]+20,box[3]-box[1]+20))
            ImageDraw.Draw(t).text((10-box[0],10-box[1]),token,font=f,fill=(0,0,0,0),stroke_width=2,stroke_fill=(*color,160));params.update(outline_px=2)
        else:
            shadow=Image.new('RGBA',(t.width+20,t.height+20));sil=Image.new('RGBA',t.size,(0,0,0,0));sil.putalpha(t.getchannel('A'))
            shadow.alpha_composite(sil,(8,8));shadow=shadow.filter(ImageFilter.GaussianBlur(3));shadow.alpha_composite(t,(0,0));t=shadow
            params.update(shadow_offset_px=[8,8],shadow_blur_px=3)
        paste(t,(w-t.width)/2,(h-t.height)/2)
    elif style in ['multiple','mixed_color']:
        for j in range(3):
            c=tuple(rng.randrange(20,235) for _ in range(3)) if style=='mixed_color' else color
            t=text_tile(f'{token} {j+1}',s*.06,(*c,a),j*13-13)
            paste(t,w*(.1+j*.22),h*(.15+j*.25))
        logo(w*.68,h*.2,s*.22,a)
        params['components']=4;params['multicolor']=style=='mixed_color'
    else:raise ValueError(style)
    alpha=np.asarray(layer.getchannel('A'))
    if not np.any(alpha):raise ValueError('Empty watermark: '+style)
    params.update(alpha_min_nonzero=int(alpha[alpha>0].min()),alpha_max=int(alpha.max()),
                  support_fraction=float(np.count_nonzero(alpha)/alpha.size),
                  alpha_definition='effective_8bit_alpha_of_composited_synthetic_layer')
    return layer,params

def images(bases):
    rows=[]
    for bi,b in enumerate(bases):
        clean=Image.open(DEST/b['path']).convert('RGB');w,h=clean.size
        for style,(scenario,label) in IMAGE_STYLES.items():
            sid=b['id']+'__'+style;folder=DEST/f'images/{b["split"]}/{sid}';folder.mkdir(parents=True,exist_ok=True)
            overlay,params=watermark_layer(clean.size,style,sid,b['split'])
            watermarked=Image.alpha_composite(clean.convert('RGBA'),overlay).convert('RGB')
            paths=dict(clean=b['path'])
            for role,im in [('watermarked',watermarked),('overlay_rgba',overlay),('alpha',overlay.getchannel('A')),
                            ('mask',overlay.getchannel('A').point(lambda a:255 if a else 0))]:
                path=folder/(role+'.png');im.save(path,compress_level=3);paths[role]=filepart(path)
            rows.append(dict(id=sid,kind='image',scenario=scenario,subtype=style,label=label,split=b['split'],
                             source_group=b['source_group'],split_group=b['split_group'],base_id=b['id'],background=b['background'],
                             provenance='project_synthetic_watermark_on_'+b['provenance'],
                             license=b['license'],size=[w,h],paths=paths,sha256={k:digest(DEST/v) for k,v in paths.items()},
                             parameters=params,watermark_origin='project_synthetic',is_negative=False,
                             mask_definition='support_of_known_composited_layer_alpha_gt_zero',
                             review_status='automated_integrity_verified_contact_sheet_review_pending'))
        # A small legitimate footer/logo remains in clean content for negatives.
        folder=DEST/f'images/{b["split"]}/{b["id"]}__negative';folder.mkdir(parents=True,exist_ok=True)
        paths=dict(clean=b['path'],watermarked=b['path'])
        for role in ['mask','alpha']:
            path=folder/(role+'.png');Image.new('L',clean.size).save(path);paths[role]=filepart(path)
        rows.append(dict(id=b['id']+'__negative',kind='image',scenario='N00',subtype='no_watermark',label='无水印负例',
                         split=b['split'],source_group=b['source_group'],split_group=b['split_group'],base_id=b['id'],background=b['background'],
                         provenance=b['provenance'],license=b['license'],size=[w,h],paths=paths,
                         sha256={k:digest(DEST/v) for k,v in paths.items()},parameters={},watermark_origin='none',is_negative=True,
                         mask_definition='all_zero',review_status='automated_integrity_verified_contact_sheet_review_pending'))
        if bi%8==0:print('IMAGE BASES',bi+1,'/',len(bases),'records',len(rows),flush=True)
    return rows

def raster_doc(src):
    d=fitz.open()
    for p in src:
        out=d.new_page(width=p.rect.width,height=p.rect.height)
        out.insert_image(out.rect,stream=p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).tobytes('png'))
    return d

def add_pdf_watermark(doc,style,key,split):
    seed=seed_for(key);color=(.2,.32,.55);alpha=.28
    label='WM '+split.upper()+' '+str(seed%10000)
    params=dict(text=label,opacity=alpha,watermark_origin='project_synthetic',asset_partition=split,seed=seed)
    for i,p in enumerate(doc):
        center=p.rect.tl+(p.rect.width/2,p.rect.height/2)
        if style in ['independent_text','raster_flattened','mixed_storage','shared_content_stream','unicode_text','outlined_text']:
            text=('内部样本 '+str(seed%1000)) if style=='unicode_text' else label
            p.insert_text((80,430),label,fontsize=42,color=color,fill_opacity=alpha,
                          morph=(center,fitz.Matrix(25)),overlay=True,
                          render_mode=1 if style=='outlined_text' else 0,stroke_opacity=.35)
            if style=='unicode_text':
                # Keep Unicode as actual text objects, not an image conversion.
                p.insert_text((100,500),text,fontname='china-s',fontsize=35,color=color,fill_opacity=alpha)
            if style=='shared_content_stream':p.clean_contents()
        elif style=='independent_image':
            over,_=watermark_layer((600,600),'logo_transparent',key,split)
            bb=over.getbbox();over=over.crop(bb)
            p.insert_image(fitz.Rect(160,280,435,555),stream=png_bytes(over))
        elif style=='vector_paths':
            sh=p.new_shape();cx,cy=300,400;pts=[]
            for j in range(16):
                a=j*math.pi/8;r=155 if j%2==0 else 72
                pts.append(fitz.Point(cx+math.cos(a)*r,cy+math.sin(a)*r))
            sh.draw_polyline(pts+[pts[0]]);sh.finish(color=color,fill=(.6,.2,.15),width=3,fill_opacity=.2,stroke_opacity=.35);sh.commit()
        elif style=='nested_form':
            inner=fitz.open();ip=inner.new_page(width=595,height=842)
            ip.insert_text((95,420),label,fontsize=42,color=color,fill_opacity=alpha)
            middle=fitz.open();mp=middle.new_page(width=595,height=842);mp.show_pdf_page(mp.rect,inner,0)
            p.show_pdf_page(p.rect,middle,0);middle.close();inner.close()
        elif style=='stamp_annotation':
            a=p.add_stamp_annot(fitz.Rect(130,330,470,450),stamp=fitz.STAMP_Draft)
            a.set_opacity(.3);a.set_info(content=label);a.update()
        elif style=='optional_content':
            ocg=doc.add_ocg('WATERMARK '+key+' '+str(i),on=True)
            p.insert_text((75,430),label,fontsize=40,color=color,fill_opacity=alpha,oc=ocg)
        else:raise ValueError(style)
    return params

def pdf_structure(doc):
    return dict(page_count=len(doc),ocgs=len(doc.get_ocgs()),
                pages=[dict(text_chars=len(p.get_text().strip()),images=len(p.get_images()),
                            vector_paths=len(p.get_drawings()),xobjects=len(p.get_xobjects()),
                            nested_form=any(x[2]!=0 for x in p.get_xobjects()),
                            annotations=[a.type[1] for a in p.annots() or []]) for p in doc])

def pdfs(bases):
    # All derivatives inherit document source groups, including raster variants.
    candidates=[b for b in bases if b['id'].startswith('document_')]
    chosen=[b for split,n in [('train',12),('validation',2),('test',2)] for b in [x for x in candidates if x['split']==split][:n]]
    rows=[]
    for b in chosen:
        for style,(scenario,label) in PDF_STYLES.items():
            sid=b['id']+'__pdf_'+style;folder=DEST/f'pdf/{b["split"]}/{sid}';folder.mkdir(parents=True,exist_ok=True)
            with fitz.open(DEST/b['pdf']) as source:
                if style=='raster_flattened':clean=raster_doc(source)
                elif style=='mixed_storage':
                    clean=fitz.open();clean.insert_pdf(source,from_page=0,to_page=0)
                    clean[0].insert_image((350,620,545,740),stream=png_bytes(Image.open(DEST/b['path']).convert('RGB').resize((195,120))))
                    p=clean.new_page(width=595,height=842)
                    p.insert_image(p.rect,stream=source[1].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).tobytes('png'))
                else:
                    clean=fitz.open();clean.insert_pdf(source)
            cleanpath=folder/'clean.pdf';clean.save(cleanpath,deflate=True)
            wm=fitz.open();wm.insert_pdf(clean)
            params=add_pdf_watermark(wm,style,sid,b['split'])
            isolated=fitz.open()
            for p in clean:isolated.new_page(width=p.rect.width,height=p.rect.height)
            add_pdf_watermark(isolated,style,sid,b['split'])
            if style=='raster_flattened':
                flat=raster_doc(wm);wm.close();wm=flat
            elif style=='mixed_storage':
                mixed=fitz.open();mixed.insert_pdf(wm,from_page=0,to_page=0)
                p=mixed.new_page(width=595,height=842)
                p.insert_image(p.rect,stream=wm[1].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).tobytes('png'))
                wm.close();wm=mixed
            path=folder/'watermarked.pdf';wm.save(path,garbage=4,deflate=True)
            # Reopen saved objects for both visual render and structure evidence.
            wm.close();wm=fitz.open(path);page_rows=[]
            for n,p in enumerate(clean):
                paths={}
                for role,pg in [('clean',p),('watermarked',wm[n])]:
                    rp=folder/f'page_{n+1:02d}_{role}.png';pg.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False,annots=True).save(rp);paths[role]=filepart(rp)
                pix=isolated[n].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=True,annots=True)
                rgba=Image.frombytes('RGBA',(pix.width,pix.height),pix.samples)
                # MuPDF pixmap RGB is premultiplied: only alpha is used as GT.
                ap=folder/f'page_{n+1:02d}_alpha.png';rgba.getchannel('A').save(ap);paths['alpha']=filepart(ap)
                mp=folder/f'page_{n+1:02d}_mask.png';rgba.getchannel('A').point(lambda a:255 if a else 0).save(mp);paths['mask']=filepart(mp)
                page_rows.append(dict(page=n+1,paths=paths,sha256={k:digest(DEST/v) for k,v in paths.items()},
                                      size=[pix.width,pix.height],mask_definition='support_of_separately_rendered_known_watermark_objects',
                                      alpha_definition='isolated_object_render_alpha_before_final_flattening' if style=='raster_flattened' else 'isolated_object_render_alpha'))
            structure=pdf_structure(wm)
            rows.append(dict(id=sid,kind='pdf',scenario=scenario,subtype=style,label=label,split=b['split'],
                             source_group=b['source_group'],split_group=b['split_group'],base_id=b['id'],background='project_original_document',
                             provenance='project_original_pdf_structural_fixture',license='LicenseRef-Project-Original',
                             watermark_origin='project_synthetic',is_negative=False,paths=dict(clean=filepart(cleanpath),watermarked=filepart(path)),
                             sha256=dict(clean=digest(cleanpath),watermarked=digest(path)),parameters=params,
                             page_count=len(clean),pages=page_rows,structure=structure,
                             review_status='automated_integrity_verified_contact_sheet_review_pending'))
            clean.close();wm.close();isolated.close()
        # PDF negatives inherit the original source, no annotation/watermark.
        negative_pages=[]
        with fitz.open(DEST/b['pdf']) as doc:
            structure=pdf_structure(doc)
            for n,p in enumerate(doc):
                folder=DEST/f'pdf/{b["split"]}/{b["id"]}__pdf_negative';folder.mkdir(parents=True,exist_ok=True)
                pix=p.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
                cp=folder/f'page_{n+1:02d}_clean.png';pix.save(cp)
                mp=folder/f'page_{n+1:02d}_zero.png';Image.new('L',(pix.width,pix.height)).save(mp)
                paths=dict(clean=filepart(cp),watermarked=filepart(cp),alpha=filepart(mp),mask=filepart(mp))
                negative_pages.append(dict(page=n+1,paths=paths,sha256={k:digest(DEST/v) for k,v in paths.items()},size=[pix.width,pix.height],mask_definition='all_zero'))
        rows.append(dict(id=b['id']+'__pdf_negative',kind='pdf',scenario='N00',subtype='no_watermark',label='PDF 无水印负例',
                         split=b['split'],source_group=b['source_group'],split_group=b['split_group'],base_id=b['id'],background=b['background'],
                         provenance='project_original_document',license='LicenseRef-Project-Original',watermark_origin='none',is_negative=True,
                         paths=dict(clean=b['pdf'],watermarked=b['pdf']),sha256=dict(clean=digest(DEST/b['pdf']),watermarked=digest(DEST/b['pdf'])),
                         page_count=2,pages=negative_pages,parameters={},structure=structure,
                         review_status='automated_integrity_verified_contact_sheet_review_pending'))
        print('PDF GROUP',b['id'],'records',len(rows),flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pdf-only',action='store_true');args=parser.parse_args()
    FONT.parent.mkdir(parents=True,exist_ok=True)
    if not FONT.exists():
        shutil.copy2('/Users/kevinli/Library/Fonts/CodexNotoSansCJKsc-Regular.otf',FONT)
    if not (DEST/'licenses/NotoSansCJK-OFL.txt').exists():raise ValueError('Font license notice must accompany the font')
    if args.pdf_only:
        bases=json.loads((DEST/'bases/manifest.json').read_text())
        rows=[json.loads(x) for x in (DEST/'manifest.jsonl').read_text().splitlines() if json.loads(x)['kind']=='image']
    else:
        bases=collect_bases();rows=images(bases)
        (DEST/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    rows+=pdfs(bases)
    (DEST/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for split in ['train','validation','test']:
        (DEST/'splits').mkdir(exist_ok=True)
        (DEST/f'splits/{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['split']==split))
    savejson(DEST/'build_summary.json',dict(seed=SEED,total_records=len(rows),
         kind_counts=dict(collections.Counter(r['kind'] for r in rows)),
         split_counts=dict(collections.Counter(r['split'] for r in rows)),
         scenario_counts=dict(collections.Counter(r['scenario'] for r in rows)),
         image_styles=IMAGE_STYLES,pdf_styles=PDF_STYLES,
         original_base_groups=len(bases),downloaded_cc0_bases=32,
         real_scan_or_photo_of_paper=0,real_business_watermarked_files=0,
         partition_rule='75/12.5/12.5 by base source group; PDF derivatives inherit original group; never random by image',
         note='All positive watermarks are project-synthetic. Digital flattening is not physical scanning.'))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
