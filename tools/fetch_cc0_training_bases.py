"""Download CC0 photographs with per-file Wikimedia license evidence.

Never imports upstream executable code. Raw bytes, API evidence and hashes are
kept; only CC0-1.0 entries with a sufficiently large JPEG are accepted.
"""
from pathlib import Path
import hashlib, io, json, time, urllib.parse, urllib.request, urllib.error, os
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(os.environ.get('WM_DATASET_ROOT', ROOT / 'data/watermark_training_v1'))
UA = 'WatermarkTrainingCorpus/1.0 (local dataset curation; CC0 provenance)'
QUERIES = [
    ('landscape', 'landscape -painting -drawing -art -map -portrait'),
    ('vegetation', 'forest tree -painting -drawing -portrait'),
    ('architecture', 'building architecture -painting -drawing -portrait'),
    ('objects', 'flower fruit macro -painting -drawing -portrait'),
    ('landscape', 'mountain lake -painting -drawing -portrait'),
    ('objects', 'rock texture -painting -drawing -portrait'),
    ('architecture', 'bridge exterior -painting -drawing -portrait'),
    ('objects', 'flower closeup -painting -drawing -portrait'),
]

def fetch(url, limit=24_000_000):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=40) as r:
                b = r.read(limit + 1)
                if len(b) > limit: raise ValueError('download exceeds configured limit')
                return b
        except urllib.error.HTTPError as e:
            if e.code in (403,429):raise
            if attempt == 2:raise
            time.sleep(2 + attempt)
        except Exception:
            if attempt == 2: raise
            time.sleep(1 + attempt)

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/'sources/raw').mkdir(parents=True, exist_ok=True)
    (DEST/'sources/evidence').mkdir(parents=True, exist_ok=True)
    (DEST/'bases/photos').mkdir(parents=True, exist_ok=True)
    target = DEST/'sources/commons_cc0.json'
    rows = json.loads(target.read_text()) if target.exists() else []
    seen = {x['commons_title'] for x in rows}
    for qfile in DEST.glob('sources/quarantined_*.json'):
        for rejected in json.loads(qfile.read_text()):seen.add(rejected['sample']['commons_title'])
    seen_hash = {x['raw_sha256'] for x in rows}
    counts = {k: sum(r['background'] == k for r in rows) for k in ['landscape','vegetation','architecture','objects']}
    for qi, (category, query) in enumerate(QUERIES):
        if len(rows) >= 32: break
        if counts[category]>=8:continue
        params = dict(action='query', format='json', generator='search',
                      gsrsearch='incategory:"CC-Zero" filetype:bitmap ' + query,
                      gsrnamespace=6, gsrlimit=30, prop='imageinfo',
                      iiprop='url|extmetadata|size|sha1',iiurlwidth=1280)
        url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)
        raw=fetch(url)
        (DEST/f'sources/evidence/search_{qi:02d}.json').write_bytes(raw)
        pages=json.loads(raw).get('query',{}).get('pages',{})
        for page in sorted(pages.values(),key=lambda p:p.get('index',0)):
            if len(rows)>=32 or counts[category]>=8: break
            if page['title'] in seen: continue
            info=page.get('imageinfo',[{}])[0]; meta=info.get('extmetadata',{})
            lic=meta.get('LicenseShortName',{}).get('value','')
            licurl=meta.get('LicenseUrl',{}).get('value','')
            if lic != 'CC0' or 'creativecommons.org/publicdomain/zero/1.0' not in licurl: continue
            if not urllib.parse.urlparse(info.get('url','')).path.lower().endswith(('.jpg','.jpeg')): continue
            if info.get('size',0)>24_000_000 or min(info.get('width',0),info.get('height',0))<768: continue
            try:
                # Wikimedia recommends its standard-sized thumbnail service
                # instead of large original downloads. Never evade rate limits.
                download_url=info.get('thumburl',info['url'])
                imagebytes=fetch(download_url); digest=hashlib.sha256(imagebytes).hexdigest()
                if digest in seen_hash: continue
                image=ImageOps.exif_transpose(Image.open(io.BytesIO(imagebytes))).convert('RGB')
                image.thumbnail((1536,1536),Image.Resampling.LANCZOS)
                # Reject exact and very similar pixels, even if files / pages differ.
                def dhash(im):
                    a=list(im.convert('L').resize((9,8)).getdata())
                    return sum((a[y*9+x]>a[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                candidate_hash=dhash(image)
                if any((candidate_hash ^ dhash(Image.open(DEST/r['normalized_path']))).bit_count()<=3 for r in rows):
                    print('SKIP near duplicate',page['title'],flush=True);continue
                pid=f'photo_{max([int(r["id"].split("_")[1]) for r in rows]+[0])+1:03d}'
                rawpath=DEST/f'sources/raw/{pid}.jpg'; rawpath.write_bytes(imagebytes)
                normalized=DEST/f'bases/photos/{pid}.png'; image.save(normalized)
                evidence=DEST/f'sources/evidence/{pid}.json'; evidence.write_text(json.dumps(page,ensure_ascii=False,indent=2))
                row=dict(id=pid,commons_title=page['title'],page_url=info['descriptionurl'],
                         download_url=download_url,license='CC0-1.0',license_url=licurl,
                         author_html=meta.get('Artist',{}).get('value',''),
                         description_html=meta.get('ImageDescription',{}).get('value',''),
                         source_group='commons:'+str(page['pageid']),background=category,
                         raw_path=str(rawpath.relative_to(DEST)),raw_sha256=digest,
                         normalized_path=str(normalized.relative_to(DEST)),
                         normalized_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
                         evidence_path=str(evidence.relative_to(DEST)),
                         size=list(image.size),downloaded_on='2026-10-26',
                         transformations=['EXIF orientation','RGB','long edge at most 1536; no upscaling'])
                rows.append(row);seen.add(page['title']);seen_hash.add(digest);counts[category]+=1
                target.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
                print(pid,category,page['title'],image.size,flush=True)
            except urllib.error.HTTPError as e:
                if e.code in (403,429):
                    print('PAUSED_HTTP_LIMIT',e.code,'resume later, completed',len(rows),flush=True)
                    return
                print('SKIP',page['title'],str(e)[:180],flush=True)
            except Exception as e:
                print('SKIP',page['title'],str(e)[:180],flush=True)
            time.sleep(2)
    print('DOWNLOADED',len(rows),counts,flush=True)
    if len(rows)<32: raise RuntimeError('Need 32 license-verified photos; incomplete accession retained for resume')

if __name__=='__main__': main()
