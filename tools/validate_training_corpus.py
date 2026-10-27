"""Full corpus integrity, alpha/composition, source split and PDF structure QA."""
from pathlib import Path
import collections, functools, hashlib, json, sys, argparse, os
import numpy as np
from PIL import Image
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
DEST=Path(os.environ.get('WM_DATASET_ROOT',ROOT/'data/watermark_training_v1'))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@functools.lru_cache(maxsize=4)
def clean_image(path):return Image.open(path).convert('RGB')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['all','pdf'],default='all');args=ap.parse_args()
    all_rows=[json.loads(x) for x in (DEST/'manifest.jsonl').read_text().splitlines()]
    manifest_hash=sha(DEST/'manifest.jsonl')
    rows=all_rows if args.kind=='all' else [r for r in all_rows if r['kind']=='pdf']
    errors=[];checked={};ids=set();groups=collections.defaultdict(set);clean_hash_splits=collections.defaultdict(set)
    counters=collections.Counter();by_scenario=collections.defaultdict(lambda:dict(records=0,source_groups=set(),subtypes=set()))
    def check(ok,msg):
        if not ok:errors.append(msg)
    def files(record):
        for role,path in record['paths'].items():
            p=DEST/path
            check(p.is_file(),'Missing '+path)
            if not p.is_file():continue
            check(p.resolve().is_relative_to(DEST.resolve()),'Escaping dataset path '+path)
            if path not in checked:checked[path]=sha(p)
            check(checked[path]==record['sha256'][role],'Hash mismatch '+path)
    def pixel_check(record,negative,composition=False):
        p=record['paths'];a=Image.open(DEST/p['alpha']).convert('L');m=Image.open(DEST/p['mask']).convert('L')
        cl=clean_image(str(DEST/p['clean']));wm=Image.open(DEST/p['watermarked']).convert('RGB')
        check(cl.size==wm.size==a.size==m.size,'Dimension mismatch '+p['clean'])
        aa=np.asarray(a);mm=np.asarray(m)
        check(np.all((mm==0)|(mm==255)),'Nonbinary support mask '+p['mask'])
        check(np.array_equal(mm>0,aa>0),'Mask/alpha mismatch '+p['mask'])
        check(bool(np.any(aa))!=negative,'Empty positive or nonzero negative '+p['alpha'])
        if composition:
            overlay=Image.open(DEST/p['overlay_rgba']).convert('RGBA')
            check(np.array_equal(np.asarray(overlay.getchannel('A')),aa),'Overlay/alpha mismatch '+p['alpha'])
            composed=Image.alpha_composite(cl.convert('RGBA'),overlay).convert('RGB')
            check(np.array_equal(np.asarray(composed),np.asarray(wm)),'Composition mismatch '+p['watermarked'])
        if negative:check(np.array_equal(np.asarray(cl),np.asarray(wm)),'Negative differs '+p['clean'])
        else:check(bool(np.any(np.asarray(cl)!=np.asarray(wm))),'Invisible positive '+p['watermarked'])
    for idx,r in enumerate(rows):
        check(r['id'] not in ids,'Duplicate id '+r['id']);ids.add(r['id'])
        groups[r.get('split_group',r['source_group'])].add(r['split'])
        clean_hash_splits[r['sha256']['clean']].add(r['split'])
        files(r);counters[r['kind']]+=1;counters['negative']+=r['is_negative']
        item=by_scenario[r['scenario']];item['records']+=1;item['source_groups'].add(r['source_group']);item['subtypes'].add(r['subtype'])
        if r['kind']=='image':pixel_check(r,r['is_negative'],not r['is_negative'])
        else:
            with fitz.open(DEST/r['paths']['watermarked']) as doc:
                check(len(doc)==r['page_count']==len(r['pages']),'PDF page count '+r['id'])
                s=r['subtype']
                if not r['is_negative']:
                    for p in doc:
                        if s in ['independent_text','unicode_text','shared_content_stream','outlined_text']:
                            check('WM '+r['split'].upper() in p.get_text(),'Missing watermark text '+r['id'])
                        if s=='unicode_text':check('内部样本' in p.get_text(),'Unicode text absent '+r['id'])
                        if s=='shared_content_stream':check(len(p.get_contents())==1,'Not shared stream '+r['id'])
                        if s=='independent_image':check(len(p.get_images())>=1,'PDF image absent '+r['id'])
                        if s=='vector_paths':check(any((x.get('fill_opacity') if x.get('fill_opacity') is not None else 1)<1 for x in p.get_drawings()),'Vector watermark absent '+r['id'])
                        if s=='outlined_text':
                            stream=b''.join(doc.xref_stream(x) for x in p.get_contents())
                            check(b'1 Tr' in stream,'Outline render mode absent '+r['id'])
                        if s=='nested_form':check(any(x[2]!=0 for x in p.get_xobjects()),'No nested Form '+r['id'])
                        if s=='stamp_annotation':check(any(a.type[1]=='Stamp' for a in p.annots() or []),'No stamp annotation '+r['id'])
                        if s=='raster_flattened':check(not p.get_text().strip() and len(p.get_images())>0,'Not raster flattened '+r['id'])
                    if s=='optional_content':check(len(doc.get_ocgs())>=1,'No optional layer '+r['id'])
                    if s=='mixed_storage':
                        check(bool(doc[0].get_text().strip()) and bool(doc[0].get_images()),'No same-page image/text mix '+r['id'])
                        check(not doc[1].get_text().strip() and bool(doc[1].get_images()),'Second page not flattened '+r['id'])
            for page in r['pages']:files(page);pixel_check(page,r['is_negative'])
            counters['pdf_pages']+=r['page_count']
        if idx%250==0:print('VALIDATED',idx,'/',len(rows),'errors',len(errors),flush=True)
    check(all(len(v)==1 for v in groups.values()),'Source/creator split leakage')
    check(all(len(v)==1 for v in clean_hash_splits.values()),'Exact clean reference split leakage')
    check(all('C'+str(i).zfill(2) in by_scenario for i in range(1 if args.kind=='all' else 9,13)),'Missing scenario')
    for sid,item in by_scenario.items():
        if sid!='N00':check(len(item['source_groups'])>=10,'Fewer than ten source groups '+sid)
        item['source_groups']=len(item['source_groups']);item['subtypes']=sorted(item['subtypes'])
    # The shipped partition files must exactly equal the primary manifest.
    for split in ['train','validation','test']:
        actual=[json.loads(x) for x in (DEST/f'splits/{split}.jsonl').read_text().splitlines()]
        check(actual==[r for r in all_rows if r['split']==split],'Stale partition file '+split)
    check(manifest_hash==sha(DEST/'manifest.jsonl'),'Manifest changed during validation')
    visual_path=DEST/'qa/visual_review.json'
    visual=json.loads(visual_path.read_text()).get('status','pending') if visual_path.exists() else 'pending'
    report=dict(status='passed' if not errors else 'failed',errors=errors,records=len(rows),manifest_sha256=manifest_hash,
        counts=dict(counters),unique_checked_files=len(checked),source_groups=len({r['source_group'] for r in rows}),
        conservative_split_groups=len(groups),cross_split_source_groups=sum(len(v)>1 for v in groups.values()),
        cross_split_exact_clean_hashes=sum(len(v)>1 for v in clean_hash_splits.values()),
        scenarios=dict(by_scenario),splits=dict(collections.Counter(r['split'] for r in rows)),
        background_counts=dict(collections.Counter(r['background'] for r in rows if r['kind']=='image' and r['subtype']=='local_text')),
        scope='Every indexed file SHA256; all image pairs reconstructed from exact overlay; all masks; every PDF reopened and structure checked. Contact sheets reviewed separately.',
        visual_review=visual,real_scan_count=0,real_business_watermark_count=0)
    (DEST/'qa').mkdir(exist_ok=True)
    (DEST/('qa/validation.json' if args.kind=='all' else 'qa/pdf_validation.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if errors:sys.exit(1)

if __name__=='__main__':main()
