"""Single-image before/after evaluation. References are used only AFTER inference."""
from __future__ import annotations
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wm import documents,masks,ocr,repair
from wm.contract import Thresholds
from tools.diagnose_single import metrics,overlay

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/single_improvement'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results={}
    before=json.loads((ROOT/'out/single_diagnosis/metrics.json').read_text())
    for stem in ('heavy_dark_02','heavy_dark_04'):
        path=ROOT/'data/real_samples/jpg'/f'{stem}.jpg'
        page=documents.load_pages(path)[0]
        start=time.perf_counter()
        t=time.perf_counter();keepout=ocr.text_zones(page.image);ocr_ms=round((time.perf_counter()-t)*1000)
        t=time.perf_counter();mask=masks.estimate_single_page(page.image,keepout=keepout)
        mask_ms=round((time.perf_counter()-t)*1000)
        t=time.perf_counter();output,used=repair.repair(page.image,mask,Thresholds())
        repair_ms=round((time.perf_counter()-t)*1000)
        elapsed_ms=round((time.perf_counter()-start)*1000)
        # No reference can affect detection/repair above.
        clean=np.array(Image.open(ROOT/'data/real_samples/clean'/f'{stem}.png').convert('RGB'))
        gt=np.load(ROOT/'data/real_samples/mask'/f'{stem}_veil.npz')['alpha']
        pred=mask.alpha>=Thresholds().mask_alpha_min;actual=gt>.08
        row=dict(method=mask.method,ocr_ms=ocr_ms,mask_ms=mask_ms,repair_ms=repair_ms,total_ms=elapsed_ms,
                 precision=float((pred&actual).sum()/max(pred.sum(),1)),
                 recall=float((pred&actual).sum()/max(actual.sum(),1)),
                 before=before[stem]['arms']['auto'],after=metrics(output,clean,page.image,gt),
                 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results[stem]=row
        Image.fromarray(output).save(OUT/f'{stem}_after.png')
        Image.fromarray(overlay(page.image,pred)).save(OUT/f'{stem}_mask_after.png')
        (OUT/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),'utf-8')
        print(stem,json.dumps(row,ensure_ascii=False),flush=True)
    # Native files remain available; this sheet is a crop for inspection only.
    font=ImageFont.truetype('/System/Library/Fonts/PingFang.ttc',22) if Path('/System/Library/Fonts/PingFang.ttc').exists() else ImageFont.load_default()
    sheet=Image.new('RGB',(1260,760),'#f4f5f7');draw=ImageDraw.Draw(sheet)
    for j,title in enumerate(['带水印原图','修改前：自动处理','修改后：自动处理']):draw.text((j*420+14,12),title,font=font,fill='#17212c')
    for i,(stem,crop) in enumerate([('heavy_dark_02',(75,220,535,550)),('heavy_dark_04',(130,160,450,305))]):
        files=[ROOT/'data/real_samples/jpg'/f'{stem}.jpg',ROOT/'out/single_diagnosis'/stem/'auto.png',OUT/f'{stem}_after.png']
        y=55+i*350
        draw.text((14,y),'文档：正文与表格' if i==0 else '照片：设备与纹理',font=font,fill='#334155')
        for j,file in enumerate(files):
            img=ImageOps.contain(Image.open(file).crop(crop),(408,290))
            sheet.paste(img,(j*420+6,y+38+(290-img.height)//2))
    sheet.save(OUT/'before_after.png')

if __name__=='__main__':main()
