"""Prepare the user-authorized manual dataset increment; no model inference.

Run with .venv/bin/python. Frozen v1 and intake originals are never rewritten.
"""
from __future__ import annotations
import collections, hashlib, importlib.util, json, platform, shutil, sys
from pathlib import Path
import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / 'data/manual_collections'
DEST = ROOT / 'data/watermark_training_increment_v1_1'
FROZEN = ROOT / 'data/watermark_training_v1'
AUTH = {
    'date': '2026-10-26',
    'scope': 'five ordinary photographs, component lesson document and native PDF; training in this software',
    'evidence': 'Direct user reply: 上述素材均可授权用于本软件训练',
    'excludes': ['two Douyin downloads', 'sensitive financial statement'],
    'ownership_not_independently_verified': True,
    'public_redistribution_authorized': False,
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rel(p):
    return str(Path(p).relative_to(DEST))

def jsave(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return rel(dst)

def save_image(im, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, compress_level=3)
    return rel(p)

def add_hashes(row):
    row['sha256'] = {k: sha(DEST/v) for k,v in row['paths'].items() if v}
    return row

def verify_intake():
    files = []
    for mpath in sorted(INCOMING.glob('*/manifest.json')):
        m = json.loads(mpath.read_text())
        for key, f in m['files'].items():
            p = mpath.parent/f['path']
            assert sha(p) == f['sha256'], str(p)
            files.append({'batch': m['batch_id'], 'key': key, 'sha256': sha(p),
                          'intake_path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size})
    assert len(files) == 16
    jsave(DEST/'qa/intake_integrity.json', {'status':'passed', 'files':files})

def photo_pairs():
    spec = importlib.util.spec_from_file_location('base_builder', ROOT/'tools/build_training_corpus.py')
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    gen.FONT = DEST/'assets/fonts/NotoSansCJKsc-Regular.otf'
    copy(FROZEN/'assets/fonts/NotoSansCJKsc-Regular.otf', gen.FONT)
    copy(FROZEN/'licenses/NotoSansCJK-OFL.txt', DEST/'licenses/NotoSansCJK-OFL.txt')
    m = json.loads((INCOMING/'background_batch_001/manifest.json').read_text())
    rows, bases = [], []
    for base in m['samples']:
        if base['sensitive_financial_information']: continue
        name = base['file']; source = INCOMING/'background_batch_001'/m['files'][name]['path']
        original = copy(source, DEST/f'sources/photos/{name}.png')
        clean = Image.open(source).convert('RGB')
        native_size = list(clean.size)
        clean.thumbnail((1280,1280), Image.Resampling.LANCZOS)
        cleanpath = save_image(clean, DEST/f'bases/{name}.png')
        bases.append(dict(id=name, original=original, path=cleanpath, original_size=native_size,
                          size=list(clean.size), source_group=base['source_group'],
                          split_group=base['source_group'], background=base['background']))
        for style, (scenario, label) in gen.IMAGE_STYLES.items():
            sid = 'inc11_'+name+'__'+style
            folder = DEST/'images'/sid
            overlay, params = gen.watermark_layer(clean.size, style, sid, 'train')
            marked = Image.alpha_composite(clean.convert('RGBA'), overlay).convert('RGB')
            paths = {'clean': cleanpath}
            for role, im in [('watermarked',marked),('overlay_rgba',overlay),
                             ('alpha',overlay.getchannel('A')),
                             ('mask',overlay.getchannel('A').point(lambda x:255 if x else 0))]:
                paths[role] = save_image(im, folder/(role+'.png'))
            rows.append(add_hashes(dict(id=sid, kind='image', unit='image_pair', scenario=scenario,
                subtype=style, label=label, split='train', source_group=base['source_group'],
                split_group=base['source_group'], base_id=name, background=base['background'],
                appearance=[style], difficulty=['complex_real_background'],
                difficulty_basis='qualitative content tag; not a measured model difficulty score',
                paths=paths, size=list(clean.size), parameters=params, is_negative=False,
                watermark_origin='project_synthetic_on_user_authorized_real_photo',
                alpha_definition='effective_8bit_composited_overlay',
                mask_definition='exact_overlay_alpha_gt_zero', label_quality='exact_constructed_pair',
                training_admission=True, rights='LicenseRef-User-Authorized-Project-Training',
                normal_content_to_preserve='All original scene objects, printed text, logos, signs and textures')))
        sid='inc11_'+name+'__negative'; paths={'clean':cleanpath,'watermarked':cleanpath}
        for role in ['mask','alpha']:
            paths[role]=save_image(Image.new('L',clean.size), DEST/'images'/sid/(role+'.png'))
        rows.append(add_hashes(dict(id=sid, kind='image', unit='image_pair', scenario='N00',
            subtype='no_added_watermark', label='无新增水印负例', split='train',
            source_group=base['source_group'], split_group=base['source_group'], base_id=name,
            background=base['background'], appearance=[], difficulty=[], paths=paths,
            size=list(clean.size), parameters={}, is_negative=True, watermark_origin='none',
            mask_definition='all_zero', label_quality='exact_constructed_pair', training_admission=True,
            rights='LicenseRef-User-Authorized-Project-Training')))
        print('PHOTO',name,len(rows),flush=True)
    jsave(DEST/'sources/photo_bases.json',bases)
    return rows

def text_blocks(ops):
    blocks=[]; start=None
    for i,(args,op) in enumerate(ops):
        if op==b'BT':
            assert start is None; start=i
        elif op==b'ET':
            assert start is not None
            block=ops[start:i+1]
            large=any(o==b'Tm' and max(abs(float(a[0])),abs(float(a[3])))>=40 for a,o in block)
            blocks.append((start,i,large)); start=None
    assert start is None
    assert sum(b[2] for b in blocks)==1, 'Expected exactly one large known watermark text block'
    return blocks

def normalized_pdf(source, output, mode):
    writer=PdfWriter(); writer.clone_document_from_reader(PdfReader(source))
    ledger=[]
    for pn,page in enumerate(writer.pages):
        cs=ContentStream(page.get_contents(),writer); ops=cs.operations
        blocks=text_blocks(ops)
        excluded=set()
        for start,end,is_watermark in blocks:
            if (mode=='clean' and is_watermark) or (mode=='isolated' and not is_watermark):
                excluded.update(range(start,end+1))
        paint={b'S',b's',b'f',b'F',b'f*',b'B',b'B*',b'b',b'b*'}
        out=[]
        for i,(args,op) in enumerate(ops):
            if i in excluded:continue
            if mode=='isolated' and op in paint: out.append(([],b'n'))
            elif mode=='isolated' and op in {b'Do',b'sh'}:continue
            else:out.append((args,op))
        cs.operations=out; page.replace_contents(cs)
        ledger.append({'page':pn+1,'original_operations':len(ops),'output_operations':len(out),
                       'watermark_block':[list(b[:2]) for b in blocks if b[2]][0],
                       'body_text_blocks':sum(not b[2] for b in blocks)})
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as f:writer.write(f)
    return ledger

def render(page, alpha=False):
    pm=page.get_pixmap(matrix=pymupdf.Matrix(150/72,150/72),alpha=alpha)
    return Image.frombytes('RGBA' if alpha else 'RGB',(pm.width,pm.height),pm.samples)

def native_pdfs():
    batch=INCOMING/'native_pdf_batch_001'; m=json.loads((batch/'manifest.json').read_text())
    source_group=m['samples'][0]['source_group']; rows=[]; evidence=[]
    original = copy(batch/m['files']['clean']['path'],DEST/'sources/pdf/original_clean.pdf')
    for variant in ['gray','black']:
        src=batch/m['files'][variant]['path']; folder=DEST/'pdf'/variant
        raw=copy(src,DEST/f'sources/pdf/{variant}_original_export.pdf')
        ledgers={}
        for mode,fn in [('marked','watermarked_normalized.pdf'),('clean','clean_matched_geometry.pdf'),
                        ('isolated','known_watermark_objects.pdf')]:
            ledgers[mode]=normalized_pdf(src,folder/fn,mode)
        docs={key:pymupdf.open(folder/fn) for key,fn in
              [('marked','watermarked_normalized.pdf'),('clean','clean_matched_geometry.pdf'),
               ('isolated','known_watermark_objects.pdf')]}
        for pn in range(3):
            marked=render(docs['marked'][pn]); clean=render(docs['clean'][pn]); isolated=render(docs['isolated'][pn],True)
            a=np.array(isolated.getchannel('A')); mask=(a>0).astype('uint8')*255
            delta=np.max(np.abs(np.array(marked).astype('int16')-np.array(clean).astype('int16')),axis=2)
            outside=int(np.count_nonzero(delta[a==0]))
            # PDF renderer may round sub-1/255 edge alpha to zero in isolation,
            # while its composite changes a neighbouring pixel by one level.
            expanded=cv2.dilate(mask,np.ones((3,3),dtype='uint8'))>0
            outside_far=int(np.count_nonzero((delta>0)&~expanded))
            outside_max=int(delta[a==0].max())
            assert outside_far==0 and outside_max<=1,(variant,pn,outside_far,outside_max)
            sid=f'inc11_pdf_{variant}_p{pn+1:02d}'; paths={}
            for role,im in [('watermarked',marked),('clean',clean),('overlay_rgba',isolated),
                            ('alpha',Image.fromarray(a)),('mask',Image.fromarray(mask))]:
                paths[role]=save_image(im,DEST/'pdf_pages'/sid/(role+'.png'))
            paths.update(pdf_watermarked=rel(folder/'watermarked_normalized.pdf'),
                pdf_clean=rel(folder/'clean_matched_geometry.pdf'),
                pdf_isolated_watermark=rel(folder/'known_watermark_objects.pdf'),
                source_export=raw,original_clean_source=original)
            trace=[t for t in docs['marked'][pn].get_texttrace() if abs(t['size']-48)<.1]
            assert len(trace)==1
            trace_clean=[t for t in docs['clean'][pn].get_texttrace() if abs(t['size']-48)<.1]
            assert not trace_clean
            text=''.join(chr(c[0]) for c in trace[0]['chars']);assert text=='内部演示稿',repr(text)
            fill_opacity=float(trace[0]['opacity'])
            recomposed=Image.alpha_composite(clean.convert('RGBA'),isolated).convert('RGB')
            rdelta=np.abs(np.asarray(recomposed).astype('int16')-np.asarray(marked).astype('int16'))
            params={'appearance':variant+'_diagonal_text','text':text,'fontsize_pt':48,
                    'fill_opacity':fill_opacity,'dpi':150,'pdf_page_1based':pn+1,
                    'page_size_pt':list(docs['marked'][pn].rect)[2:],
                    'geometry_reference':'derived clean in export geometry; original Letter PDF separately preserved',
                    'reference_derivation':'annotated known watermark BT/ET block omitted from normalized export stream',
                    'label_derivation':'render only the known watermark object onto transparent canvas',
                    'raster_recomposition_max_error':int(rdelta.max()),
                    'raster_recomposition_mean_absolute_error':float(rdelta.mean()),
                    'different_pixels_outside_known_object_support':outside,
                    'max_error_outside_known_object_support':outside_max,
                    'different_pixels_outside_1px_object_support_halo':outside_far,
                    'object_alpha_quantization_note':'isolated alpha may round sub-1/255 antialias edge coverage to zero; adjacent composite may differ by one intensity level',
                    'Unicode_extraction_original_vs_export':'not asserted equal; font export mapping differs'}
            rows.append(add_hashes(dict(id=sid, kind='pdf_page',unit='rendered_pdf_page_pair',
                source_group=source_group,split_group=source_group,split='train',scenario='C09',
                subtype='native_software_shared_content_stream',label='真实软件导出 PDF 共流文字水印',
                background='dense_office_document_text_urls',appearance=[variant,'diagonal_text'],
                difficulty=['body_text_overlap','same_content_stream','font_encoding_difference'],
                paths=paths,size=list(marked.size),parameters=params,is_negative=False,
                watermark_origin='user_created_macOS_print_pipeline',
                label_quality='known_object_annotated_pair',mask_definition='known_object_render_alpha_gt_zero',
                alpha_definition='isolated_object_render_coverage_times_fill_opacity_not_original_application_layer',
                training_admission=True,rights='LicenseRef-User-Authorized-Project-Training')))
            evidence.append(dict(id=sid,**params))
        for d in docs.values():d.close()
        jsave(folder/'object_annotation.json',ledgers)
        print('PDF',variant,'3 pages prepared',flush=True)
    jsave(DEST/'qa/pdf_annotation_checks.json',{'status':'passed','pages':evidence})
    return rows

def register_pair(cleanfile,markedfile,key):
    clean=np.asarray(Image.open(cleanfile).convert('RGB'));marked=np.asarray(Image.open(markedfile).convert('RGB'))
    sift=cv2.SIFT_create(nfeatures=8000)
    ka,da=sift.detectAndCompute(cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY),None)
    kb,db=sift.detectAndCompute(cv2.cvtColor(marked,cv2.COLOR_RGB2GRAY),None)
    matches=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[m for m,n in matches if m.distance<.7*n.distance]
    pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(20261026)
    H,selected=cv2.findHomography(pa,pb,cv2.RANSAC,2.5)
    assert H is not None
    errors=np.linalg.norm(cv2.perspectiveTransform(pa.reshape(-1,1,2),H).reshape(-1,2)-pb,axis=1)[selected.ravel()>0]
    h,w=marked.shape[:2]
    warped=cv2.warpPerspective(clean,H,(w,h),borderValue=(255,255,255))
    valid=cv2.warpPerspective(np.ones(clean.shape[:2],dtype='uint8')*255,H,(w,h),flags=cv2.INTER_NEAREST)
    valid=cv2.erode(valid,np.ones((5,5),dtype='uint8'))
    metrics=dict(matches=len(good),inliers=len(errors),inlier_ratio=float(len(errors)/len(good)),
                 median_px=float(np.median(errors)),p95_px=float(np.quantile(errors,.95)),H=H.tolist(),
                 method='SIFT ratio0.7 + RANSAC2.5px homography; before->after; no brightness correction',
                 valid_area_definition='warped source support eroded 2px; NOT a watermark mask')
    assert metrics['inliers']>100 and metrics['median_px']<1
    folder=DEST/'weak_pairs'/key
    paths={'original_clean':copy(cleanfile,folder/'original_clean.png'),
           'watermarked':copy(markedfile,folder/'watermarked_original.png'),
           'aligned_clean':save_image(Image.fromarray(warped),folder/'aligned_clean.png'),
           'valid_area':save_image(Image.fromarray(valid),folder/'valid_area.png')}
    jsave(folder/'registration.json',metrics);paths['registration']=rel(folder/'registration.json')
    return paths,metrics,[w,h]

def weak_and_references():
    hw=INCOMING/'handwriting_batch_001/originals';xt=INCOMING/'xingtu_batch_001/originals'
    weak=[]
    specs=[('handwriting_page02',hw/'page02_before.png',hw/'page02_after.png','M01',
            [[725,610,1140,785],[835,1050,1135,1210]]),
           ('handwriting_page04',hw/'page04_before.png',hw/'page04_after.png','M01',
            [[725,905,1170,1110],[825,1285,1155,1460]]),
           ('xingtu_page03',xt/'clean.png',xt/'watermarked.png','M02',[])]
    for key,c,m,task,boxes in specs:
        paths,metrics,size=register_pair(c,m,key)
        ann={'bbox_format':'xyxy_pixels_in_original_watermarked_frame','boxes':boxes,
             'quality':'coarse_regions_only_not_stroke_masks','signature_transcribed':False,
             'task':task,'no_boxes_reason':'tiled watermark across page; exact support unavailable' if not boxes else None}
        jsave(DEST/'weak_pairs'/key/'annotation.json',ann);paths['annotation']=f'weak_pairs/{key}/annotation.json'
        weak.append(add_hashes(dict(id='inc11_'+key,kind='registered_real_image_pair',unit='approximate_image_pair',
            source_group='user_document_scada_component_lesson_001',split_group='user_document_scada_component_lesson_001',
            reserved_split='train',split=None,task=task,paths=paths,size=size,registration=metrics,
            label_quality='weak_coarse_regions' if boxes else 'unmasked_approximate_pair',
            mask_path=None,alpha_path=None,training_admission=False,
            exclusion_reason='no exact pixel reference/mask; geometry and illumination/export differences remain',
            rights='LicenseRef-User-Authorized-Project-Training',
            background='photographed_Chinese_printed_document',
            appearance=['physical_black_handwriting','signature'] if boxes else ['white_diagonal_tiled_text','edge_clipped'],
            difficulty=['approximate_registration','lighting_or_export_change','body_text_preservation'],
            normal_content_to_preserve='Pre-existing red dots on page02 and red triangle on page04 remain original content')))
    # Handwritten red page is useful as unpaired material, never invent a clean target.
    rp=copy(hw/'page01_red_annotations.png',DEST/'references/user_handwritten_red_page01.png')
    refs=[add_hashes(dict(id='inc11_reference_handwritten_red',paths={'watermarked':rp},clean=None,
        training_admission=False,split=None,reserved_split='train',
        source_group='user_document_scada_component_lesson_001',
        split_group='user_document_scada_component_lesson_001',
        reason='unpaired; no before reference',rights='LicenseRef-User-Authorized-Project-Training',
        appearance=['red_handwriting','underline'],background='photographed_printed_document'))]
    return weak,refs

def financial_layout_trace():
    """Local vector-like table grid only. No source pixels or text survive."""
    src=INCOMING/'background_batch_001/restricted_originals/business_table_sensitive.png'
    im=Image.open(src).convert('RGB').rotate(90,expand=True)
    gray=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY)
    bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    h,w=bw.shape
    lineimg=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,max(150,w//5)),np.uint8))
    lineimg|=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((max(100,h//6),1),np.uint8))
    lines=cv2.HoughLinesP(lineimg,1,np.pi/180,threshold=100,minLineLength=250,maxLineGap=15)
    # Completely new blank canvas; draw only long, nearly straight grid segments.
    blank=Image.new('RGB',im.size,'white');draw=ImageDraw.Draw(blank);count=0
    if lines is not None:
        for x1,y1,x2,y2 in lines.reshape(-1,4):
            if min(abs(int(x2)-int(x1)),abs(int(y2)-int(y1)))<30:
                draw.line((int(x1),int(y1),int(x2),int(y2)),fill=(140,140,140),width=1);count+=1
    save_image(blank,DEST/'excluded/financial_grid_only_surrogate.png')
    jsave(DEST/'excluded/financial_handling.json',dict(status='restricted_original_excluded_from_dataset_and_archive',
        source_hash=sha(src),derived_asset='financial_grid_only_surrogate.png',
        transformation='blank canvas plus long detected grid segments only; no original raster pixels or text retained',
        detected_grid_segments=count,training_admission=False,
        reason='layout trace only; not a useful desensitized business training document; replace with fictitious or approved redacted source',
        raw_original_location='restricted local intake; intentionally no embedded preview or copied original'))

def jsonl(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def main():
    frozen_hash=sha(FROZEN/'manifest.jsonl')
    DEST.mkdir(parents=True,exist_ok=True)
    verify_intake();jsave(DEST/'sources/user_training_authorization.json',AUTH)
    rows=photo_pairs()+native_pdfs()
    weak,refs=weak_and_references();financial_layout_trace()
    jsonl(DEST/'manifest.jsonl',rows);jsonl(DEST/'weak_pairs.jsonl',weak);jsonl(DEST/'references.jsonl',refs)
    tasks={'M01':{'state':'2_approximate_pairs_registered_1_unpaired','underlying_documents':1},
           'M02':{'state':'1_xingtu_pair_registered_2_third_party_references_deleted_by_user','application_version':None},
           'M03':{'state':'deferred_by_user','received':0},
           'M04':{'state':'3_marked_photographs_cross_classified_with_M01','counted_again':False},
           'M05':{'state':'2_native_variants_6_pages_prepared','underlying_documents':1},
           'M06':{'state':'5_photos_prepared_1_financial_original_restricted','portrait_photos':0}}
    jsave(DEST/'manual_status.json',tasks)
    jsave(DEST/'build_summary.json',dict(version='1.1-manual-increment',date='2026-10-26',
        strong_training_records=len(rows),synthetic_image_positives=110,image_negatives=5,native_pdf_page_pairs=6,
        pdf_variants=2,pdf_underlying_documents=1,ordinary_photo_bases=5,ordinary_photo_split_groups=4,
        training_source_groups=len(set(r['source_group'] for r in rows)),
        weak_registered_pairs=len(weak),unpaired_references=len(refs),
        third_party_images_not_packaged=0,third_party_images_deleted_by_user=2,sensitive_originals_not_packaged=1,
        train=len(rows),validation=0,test=0,
        v1_frozen_manifest_sha256=frozen_hash,v1_records_unchanged=2432,
        unit_warning='121 is page/image training records, not 121 independent source documents; PDF count differs from v1 document-level manifest',
        validation_status='pending_independent_validation',public_redistribution_authorized=False))
    jsave(DEST/'environment.json',dict(python=sys.version,platform=platform.platform(),
        opencv=cv2.__version__,pymupdf=pymupdf.VersionBind,numpy=np.__version__))
    copy(Path(__file__),DEST/'scripts/prepare_manual_increment.py')
    copy(ROOT/'tools/build_training_corpus.py',DEST/'scripts/build_training_corpus.py')
    assert sha(FROZEN/'manifest.jsonl')==frozen_hash
    print('PREPARED',len(rows),'strong;',len(weak),'weak;',len(refs),'reference',flush=True)

if __name__=='__main__':main()
