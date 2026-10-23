"""Create plain before/after evidence and new exports from a saved re-run."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'out/user_optimization/after')
    args=parser.parse_args();run=args.run.resolve();out=run.parent
    rows=json.loads((run/'metrics.json').read_text())
    assert len(rows)==16 and all('error' not in r for r in rows.values()),'Full re-run required'
    groups=defaultdict(list)
    for stem,r in rows.items():
        original=Path(r['exported_path'])
        assert hashlib.sha256(original.read_bytes()).hexdigest()==r['exported_sha256'],'User export changed'
        groups[original.parent.name].append((stem,r))
    for name,values in groups.items():
        values.sort(key=lambda item:int(item[1]["verdict"]["page"].rsplit("#",1)[1]))
        folder=out/'重新处理结果'/name;folder.mkdir(parents=True,exist_ok=True)
        pages=[];queue=[]
        for stem,r in values:
            name=Path(r['exported_path']).name
            shutil.copy2(run/(stem+'.png'),folder/name)
            alpha=np.load(run/(stem+'_mask.npz'))['alpha']
            Image.fromarray(np.uint8(np.clip(alpha,0,1)*255)).save(folder/name.replace('_clean.png','_mask.png'))
            pages.append(r['verdict'])
            v=r['verdict']['verdict']
            queue.append(dict(page=r['verdict']['page'],state='质量未通过' if not v['passed'] else '待复核',error=None))
        manifest=dict(source=None,generated_at=datetime.now().isoformat(timespec='seconds'),
            summary=dict(total=len(values),passed=0,checks_passed=sum(r['verdict']['verdict']['passed'] for _,r in values),
                         needs_review=len(values),quality_evaluated=0,success_rate=None),pages=pages)
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
        (folder/'queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2))
    font=ImageFont.truetype('/Users/kevinli/Library/Fonts/CodexNotoSansCJKsc-Regular.otf',22)
    assets=out/'对照图';assets.mkdir(exist_ok=True)
    def comparison(stem,r):
        paths=[Path(r['source_path']),Path(r['exported_path']),run/(stem+'.png')]
        canvas=Image.new('RGB',(1470,710),'#edf1f5');d=ImageDraw.Draw(canvas)
        for i,(label,path) in enumerate(zip(['原图','你导出的结果','这次修改后'],paths)):
            im=ImageOps.contain(Image.open(path).convert('RGB'),(470,650))
            x=i*490;d.text((x+12,10),label,font=font,fill='#172333');canvas.paste(im,(x+(490-im.width)//2,52))
        canvas.save(assets/(stem+'.jpg'),quality=94)
    for stem,r in rows.items():comparison(stem,r)
    notes={
        'heavy_dark_01':'大片重复水印已清理，页边和正文交叠处仍需检查。',
        'tiled_text_01':'平铺水印明显减少，正文和红色链接仍保留，少量残影待处理。',
        'tiled_text_02':'主要平铺水印已清理，正文附近及页边仍有残留。',
        'tiled_text_03':'白纸区域的大部分重复水印已清理，仍需逐页核对正文。',
        'tiled_text_04':'水印有所减少，但仍有明显文字残留，这张尚未处理好。',
        'heavy_dark_02':'增加连续表格线保护；正文交叠处仍有残痕，尚未完全恢复。',
        'big_center_01':'居中照片水印仍未可靠识别，尚未解决；当前结果还有误改风险。',
        'big_center_03':'居中照片水印仍未可靠识别，尚未解决。',
        'big_center_02':'这次保持原有结果，照片仍有修补痕迹。',
        'heavy_dark_03':'这次保持原有结果，部分水印残留和照片细节损伤仍存在。',
        'heavy_dark_04':'这次保持原有结果，水印位置仍有模糊和修补痕迹。',
        'big_center_04':'保持已有目录文档处理结果，仍需检查淡色残痕。',
    }
    changed=[s for s,r in rows.items() if not r['unchanged_from_export']]
    improved=[s for s,r in rows.items() if r['repair_mode']=='paper_repeat_cleanup']
    chunks=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>这批图片的处理检查与修复</title>',
        '<style>body{font:17px/1.7 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;background:#f3f5f7;color:#1b2838}h1{font-size:30px}h2{font-size:23px}article{padding:20px;background:#fff;border-radius:10px;margin:24px 0}img{width:100%;height:auto}a{color:#1066a6}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6dee8;padding:12px;text-align:left}th{background:#e8eef5}.note{background:white;padding:18px 22px}.time{color:#526275}button{padding:10px 16px;margin-right:12px;border:1px solid #b4c0ce;border-radius:6px;background:#fff;cursor:pointer}</style>',
        '<h1>这批图片的处理检查与修复</h1>',
        '<p class="note">已按你导出的 4 批、16 张图片重新处理。5 张重复文字文档的水印清理有改善，并增加了连续表格线保护。居中照片水印仍未解决，部分文档仍有残影；这批不能标为全部达标。原始素材和你导出的文件均未改动。</p>',
        '<h2>这次改了什么</h2><p>重复水印定位不再只保留最先找到的 12 个位置；用完整文字和分别确认的文字片段定位整页水印。白底文档不再让补图模型重画已识别的文字水印区域，优先保留暗色文字、彩色链接、印章和连续表格线。点阵及没有获得可靠改善的照片处理方式保持原样。</p>',
        '<p>未复核的输出继续标为“待复核”。自动检查通过不能证明水印已经去净，导出成功率在缺少效果验证时记为未知。</p>',
        '<h2>有改动的图片</h2><table><tr><th>图片</th><th>这次导出时的耗时</th><th>本次重新处理耗时</th><th>实际情况</th></tr>']
    for stem in changed:
        r=rows[stem];chunks.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [stem,f"{r['user_elapsed_ms']/1000:.1f} 秒",f"{r['elapsed_ms']/1000:.1f} 秒",notes.get(stem,'查看下面的对照图')])+'</tr>')
    chunks.extend(['</table><p>耗时为本机这几次运行的记录，可能受到模型首次加载和其他任务影响，不是所有图片的速度保证。</p>',
        '<h2>直接看图</h2><p><button onclick="document.querySelectorAll(\'.unchanged\').forEach(e=>e.hidden=true)">只看有改动的图片</button><button onclick="document.querySelectorAll(\'article\').forEach(e=>e.hidden=false)">显示全部 16 张</button></p>'])
    order=improved+['heavy_dark_02']+[s for s in rows if s not in improved and s!='heavy_dark_02']
    for stem in order:
        r=rows[stem];same=r['unchanged_from_export'];file='对照图/'+stem+'.jpg'
        def href(path):return escape(os.path.relpath(path,out),quote=True)
        note=notes.get(stem,'这次保持点阵处理结果，请继续检查页边细小残点。')
        chunks.append(f'<article class="{"unchanged" if same else "changed"}" {"hidden" if same else ""}><h2>{escape(stem)}</h2><p>{escape(note)}</p><img src="{escape(file,quote=True)}"><p><a href="{href(Path(r["source_path"]))}">原尺寸原图</a> · <a href="{href(Path(r["exported_path"]))}">原尺寸你导出的结果</a> · <a href="{href(run/(stem+".png"))}">原尺寸本次结果</a></p></article>')
    chunks.append('<p>复测用数据：<a href="after/metrics.json">逐张处理记录</a>。重处理图片与批次记录在本报告同目录的“重新处理结果”文件夹内。以上图片均来自程序实际输入输出，没有人工修图。</p></html>')
    (out/'结果对照.html').write_text(''.join(chunks),'utf-8')
    summary=dict(total=len(rows),changed=changed,paper_cleanup=improved,
        unchanged=[s for s,r in rows.items() if r['unchanged_from_export']],
        outside_requested_mask_exact=all(r['outside_mask_unchanged'] for r in rows.values()),
        original_exports_preserved=True,photo_failures_remaining=['big_center_01','big_center_03'],
        success_rate_requires_effect_verification=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(out/'结果对照.html');print(summary)


if __name__=='__main__':main()
