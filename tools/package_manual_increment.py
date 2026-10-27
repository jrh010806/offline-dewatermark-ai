"""Package a verified increment and check portable loading after extraction."""
import hashlib, importlib.util, json, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/watermark_training_increment_v1_1'
OUT=ROOT/'out/training_dataset_release'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    qa=json.loads((D/'qa/validation.json').read_text());assert qa['status']=='passed'
    assert json.loads((D/'qa/visual_review.json').read_text())['status']=='passed'
    shutil.copyfile(Path(__file__),D/'scripts/package_manual_increment.py')
    blocked=[ROOT/'data/manual_collections/background_batch_001/restricted_originals/business_table_sensitive.png',
             *list((ROOT/'data/manual_collections/douyin_batch_001/originals').glob('*.png'))]
    forbidden={digest(p) for p in blocked}
    files=sorted(p for p in D.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS.txt')
    inventory={str(p.relative_to(D)):digest(p) for p in files}
    assert not(forbidden&set(inventory.values()))
    assert not any('restricted_originals/' in n or 'douyin' in n.lower() for n in inventory)
    (D/'SHA256SUMS.txt').write_text(''.join(f'{h}  {p}\n' for p,h in inventory.items()))
    files.append(D/'SHA256SUMS.txt');OUT.mkdir(parents=True,exist_ok=True)
    archive=OUT/'watermark_training_increment_v1_1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,arcname=str(Path(D.name)/p.relative_to(D)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        with tempfile.TemporaryDirectory(prefix='wm_increment_verify_') as tmp:
            z.extractall(tmp); extracted=Path(tmp)/D.name
            for name,h in inventory.items():assert digest(extracted/name)==h,name
            output=subprocess.check_output([sys.executable,str(extracted/'scripts/load_manual_increment.py'),
                '--root',str(extracted),'--view','strict','--limit','1'],text=True)
            assert json.loads(output.splitlines()[-1])['eligible_records']==121
            pdf_output=subprocess.check_output([sys.executable,str(extracted/'scripts/load_manual_increment.py'),
                '--root',str(extracted),'--kind','pdf_page','--limit','1'],text=True)
            assert json.loads(pdf_output.splitlines()[-1])['eligible_records']==6
            task_output=subprocess.check_output([sys.executable,str(extracted/'scripts/load_manual_increment.py'),
                '--root',str(extracted),'--view','training','--limit','1'],text=True)
            assert json.loads(task_output.splitlines()[-1])['eligible_records']==129
            # Full independent integrity validation from the extracted portable files.
            subprocess.check_output([sys.executable,str(extracted/'scripts/validate_manual_increment.py'),
                '--root',str(extracted)],text=True)
    result=dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=digest(archive),
        members=len(files),crc='passed',all_extracted_hashes='passed',portable_loader='passed_121_records_6_pdf_pages',
        extracted_validation='passed',excluded_sensitive_and_third_party_originals=3,
        intended_use='user-authorized training in this software; not a public redistribution license')
    receipt=OUT/'watermark_training_increment_v1_1_release.json'
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (OUT/'watermark_training_increment_v1_1.zip.sha256').write_text(result['sha256']+'  '+archive.name+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
