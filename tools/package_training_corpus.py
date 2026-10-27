"""Package only manifest-approved training assets; verify all ZIP CRCs."""
from pathlib import Path
import hashlib, json, platform, sys, zipfile
import numpy, PIL, pymupdf

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/watermark_training_v1'
OUT=ROOT/'out/training_dataset_release';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    qa=json.loads((D/'qa/validation.json').read_text());card=json.loads((D/'dataset_card.json').read_text())
    if qa['status']!='passed' or card['visual_review']!='passed':raise ValueError('Release requires integrity and representative visual QA')
    if sha(D/'manifest.jsonl')!=qa['manifest_sha256']:raise ValueError('Manifest differs from checked version')
    environment=dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,Pillow=PIL.__version__,PyMuPDF=pymupdf.__version__)
    (D/'environment.json').write_text(json.dumps(environment,indent=2))
    files=set()
    rows=[json.loads(x) for x in (D/'manifest.jsonl').read_text().splitlines()]
    for r in rows:
        files.update(r['paths'].values())
        for page in r.get('pages',[]):files.update(page['paths'].values())
    for b in json.loads((D/'bases/manifest.json').read_text()):
        files.add(b['path'])
        if 'pdf' in b:files.add(b['pdf'])
    for r in json.loads((D/'sources/commons_cc0.json').read_text()):files.update([r['raw_path'],r['evidence_path']])
    for folder in ['assets','licenses','scripts','splits','manual_incoming']:
        files.update(str(p.relative_to(D)) for p in (D/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(str(p.relative_to(D)) for p in D.glob('*') if p.is_file())
    files.update(['bases/manifest.json','sources/commons_cc0.json','sources/photo_contact_sheet.png'])
    files.update(str(p.relative_to(D)) for p in (D/'sources').glob('quarantined_*.json'))
    # Only reviewed QA artifacts: omit temporary smoke PDFs and obsolete partial QA.
    files.update('qa/'+n for n in ['validation.json','visual_review.json','image_styles.png','pdf_styles.png','original_bases.png','line_net_detail.png','normal_content_negative.png','document_base.png','signature_collection_form.png'])
    files.discard('SHA256SUMS.txt')
    checksums=[]
    for i,name in enumerate(sorted(files)):
        p=D/name
        if not p.is_file():raise ValueError('Missing release member '+name)
        checksums.append(sha(p)+'  '+name)
        if i%2000==0:print('HASH',i,'/',len(files),flush=True)
    (D/'SHA256SUMS.txt').write_text('\n'.join(checksums)+'\n');files.add('SHA256SUMS.txt')
    archive=OUT/'watermark_training_v1.zip'
    with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
        for i,name in enumerate(sorted(files)):
            compression=zipfile.ZIP_STORED if (D/name).suffix.lower() in ['.png','.jpg','.jpeg','.pdf','.otf'] else zipfile.ZIP_DEFLATED
            z.write(D/name,'watermark_training_v1/'+name,compress_type=compression)
            if i%2000==0:print('ZIP',i,'/',len(files),flush=True)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failure '+bad)
        indexed=set('watermark_training_v1/'+f for f in files)
        if set(z.namelist())!=indexed:raise ValueError('ZIP member mismatch')
        for r in [next(x for x in rows if x['kind']=='image' and not x['is_negative']),next(x for x in rows if x['kind']=='pdf' and not x['is_negative'])]:
            for role,path in r['paths'].items():
                if hashlib.sha256(z.read('watermark_training_v1/'+path)).hexdigest()!=r['sha256'][role]:raise ValueError('Portable sample hash mismatch')
    report=dict(status='passed',archive=str(archive),archive_bytes=archive.stat().st_size,
        sha256=sha(archive),member_count=len(files),records=len(rows),all_zip_crc_verified=True,
        manifest_sha256=sha(D/'manifest.jsonl'),counts=card['counts'],
        note='Approved manifest assets only; quarantined image pixels and temporary smoke files excluded.')
    (OUT/'release.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (OUT/'watermark_training_v1.zip.sha256').write_text(report['sha256']+'  watermark_training_v1.zip\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
