"""Dataset integrity/ground-truth checks; independent of the preparation script."""
import argparse, collections, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import pymupdf
from PIL import Image

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readlines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def validate(root,originals_root=None):
    rows=readlines(root/'manifest.jsonl');weak=readlines(root/'weak_pairs.jsonl');refs=readlines(root/'references.jsonl')
    errors=[];files={};pdfchecks=[]
    assert len(rows)==121 and len(weak)==3 and len(refs)==5
    assert len({r['id'] for r in rows+weak+refs})==129
    assert not any('local_intake_path' in r for r in refs)
    for r in rows+weak+refs:
        for role,p in r.get('paths',{}).items():
            if not p:continue
            q=(root/p).resolve();assert q.is_relative_to(root.resolve()),p
            assert q.is_file(),p
            assert digest(q)==r['sha256'][role],(r['id'],role)
            files[p]=digest(q)
        if r.get('task')=='M03':
            for role,p in r.get('source_pdfs',{}).items():
                q=(root/p).resolve();assert q.is_relative_to(root.resolve()),p
                assert q.is_file() and digest(q)==r['source_pdf_sha256'][role],(r['id'],role)
                files[p]=digest(q)
        if r not in rows:continue
        assert r['split']=='train' and r['training_admission'] is True
        ims={k:Image.open(root/r['paths'][k]) for k in ['clean','watermarked','mask','alpha']}
        assert all(list(v.size)==r['size'] for v in ims.values())
        mask=np.asarray(ims['mask']);alpha=np.asarray(ims['alpha'])
        assert set(np.unique(mask))<={0,255}
        assert np.array_equal(mask>0,alpha>0)
        clean=ims['clean'].convert('RGB');marked=ims['watermarked'].convert('RGB')
        if r['is_negative']:
            assert not mask.any() and not alpha.any()
            assert np.array_equal(np.asarray(clean),np.asarray(marked))
        elif r['kind']=='image':
            overlay=Image.open(root/r['paths']['overlay_rgba']).convert('RGBA')
            assert np.array_equal(np.asarray(overlay.getchannel('A')),alpha)
            expected=Image.alpha_composite(clean.convert('RGBA'),overlay).convert('RGB')
            assert np.array_equal(np.asarray(expected),np.asarray(marked)),r['id']
        elif r['kind']=='pdf_page':
            delta=np.max(np.abs(np.asarray(clean).astype('int16')-np.asarray(marked).astype('int16')),axis=2)
            halo=cv2.dilate(mask,np.ones((3,3),dtype='uint8'))>0
            assert not np.any(delta[~halo])
            maxoutside=int(delta[mask==0].max());assert maxoutside<=1
            page=r['parameters']['pdf_page_1based']-1
            with pymupdf.open(root/r['paths']['pdf_watermarked']) as d:
                assert len(d)==3
                wm=[t for t in d[page].get_texttrace() if abs(t['size']-48)<.1];assert len(wm)==1
                assert ''.join(chr(c[0]) for c in wm[0]['chars'])=='内部演示稿'
            with pymupdf.open(root/r['paths']['pdf_clean']) as d:
                assert not [t for t in d[page].get_texttrace() if abs(t['size']-48)<.1]
            pdfchecks.append({'id':r['id'],'outside_1px_halo_changes':0,'outside_alpha_max_error':maxoutside})
    assert collections.Counter(r['kind'] for r in rows)=={'image':115,'pdf_page':6}
    assert len({r['source_group'] for r in rows})==5
    assert len({r['source_group'] for r in weak})==1
    assert {r['reserved_split'] for r in weak}=={'train'}
    for r in weak:
        assert r['training_admission'] is False and r['mask_path'] is None and r['alpha_path'] is None
        assert r['registration']['median_px']<1 and r['registration']['inlier_ratio']>.7
    m03=json.loads((root/'manual_status.json').read_text())['M03']
    assert m03['state']=='4_scan_images_received_pending_alignment_and_masks'
    assert m03['page_images']==4 and m03['training_admission'] is False
    assert json.loads((root/'sources/user_training_authorization.json').read_text())['public_redistribution_authorized'] is False
    excluded_sha=[]
    if originals_root:
        originals_root=Path(originals_root)
        for p in [originals_root/'background_batch_001/restricted_originals/business_table_sensitive.png',
                  *list((originals_root/'douyin_batch_001/originals').glob('*.png'))]:excluded_sha.append(digest(p))
        assert not(set(excluded_sha)&set(files.values()))
        frozen=root.parent/'watermark_training_v1/manifest.jsonl'
        assert digest(frozen)==json.loads((root/'build_summary.json').read_text())['v1_frozen_manifest_sha256']
    summary={'status':'passed','errors':errors,'strong_records':len(rows),'weak_pairs':len(weak),
             'unpaired_references':len(refs),'indexed_distinct_files':len(files),
             'unique_training_source_groups':5,'training_partitions':['train'],
             'synthetic_exact_reconstruction_checked':110,'zero_mask_negatives_checked':5,
             'pdf_known_object_pairs_checked':len(pdfchecks),'pdf_checks':pdfchecks,
             'weak_registration_quality':'3 homographies median inlier error below 1px; not an exact pixel ground truth guarantee',
             'privacy_and_third_party_originals_checked':len(excluded_sha),
             'v1_frozen_manifest_unchanged':bool(originals_root),
             'coverage_limit':'M03 contains four received scan/capture references with linked source PDFs pending alignment and mask review; low independent source diversity; no assertion of universal watermark coverage'}
    (root/'qa/validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--originals-root',type=Path);args=ap.parse_args()
    print(json.dumps(validate(args.root,args.originals_root),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
