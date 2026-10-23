"""Render saved before/after regression results; performs no model inference."""
import json,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[1];out=root/'out/center_refinement';before=json.loads((out/'before/metrics.json').read_text());after=json.loads((out/'after/metrics.json').read_text());comparison={}
if set(before) != set(after):
    raise SystemExit('Before/after suites must contain the same samples')
for stem,row in after.items():
 a=np.array(Image.open(out/'before'/f'{stem}.png'));b=np.array(Image.open(out/'after'/f'{stem}.png'))
 comparison[stem]={'pixel_identical':bool(np.array_equal(a,b)),'before_roi_mae':before[stem]['output']['mae_watermark_roi'],'after_roi_mae':row['output']['mae_watermark_roi'],'input_roi_mae':row['input']['mae_watermark_roi'],'outside_requested_mask_unchanged':row['outside_mask_unchanged'],'remaining_no_improvement_vs_input':bool(row['output']['mae_watermark_roi']>=row['input']['mae_watermark_roi']-.01)}
(out/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),'utf-8')
font=ImageFont.truetype('/System/Library/Fonts/PingFang.ttc',23);small=ImageFont.truetype('/System/Library/Fonts/PingFang.ttc',18)
canvas=Image.new('RGB',(1800,1400),'#f5f6f8');draw=ImageDraw.Draw(canvas)
for row,(stem,top,height) in enumerate([('big_center_04',0,940),('big_center_02',945,410)]):
 paths=[root/'data/real_samples/jpg'/f'{stem}.jpg',out/'before'/f'{stem}.png',out/'after'/f'{stem}.png']
 for col,(label,path) in enumerate(zip(['原图','本轮优化前','本轮优化后'],paths)):
  x=col*600;draw.text((x+16,top+12),f'{stem} · {label}',font=font,fill='#20252b');im=Image.open(path).convert('RGB');im.thumbnail((570,height-85));canvas.paste(im,(x+(600-im.width)//2,top+58))
  if col:
   value=comparison[stem]['before_roi_mae' if col==1 else 'after_roi_mae'];draw.text((x+16,top+height-30),f'水印区域误差：{value:.2f}',font=small,fill='#58616c')
draw.text((18,1370),'误差越低越好；参考图仅用于评分，未参与自动定位和修复。',font=small,fill='#58616c');canvas.save(out/'before_after.png')
# Readable native-pixel crops of the difficult watermark/body overlap.
crop=Image.new('RGB',(1800,300),'white');dr=ImageDraw.Draw(crop)
for col,(label,path) in enumerate(zip(['原图','本轮优化前','本轮优化后'],[root/'data/real_samples/jpg/big_center_04.jpg',out/'before/big_center_04.png',out/'after/big_center_04.png'])):
 im=Image.open(path).convert('RGB').crop((85,290,375,410)).resize((580,240));crop.paste(im,(col*600+10,50));dr.text((col*600+15,10),label,font=font,fill='black')
crop.save(out/'document_detail.png')
style='body{max-width:1250px;margin:36px auto;padding:0 24px;font:16px/1.65 system-ui;color:#20252b}h1{font-size:28px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#f2f4f7}img{max-width:100%}.bad{color:#ab3d12}.good{color:#16704a}.note{background:#f3f5f8;padding:16px}'
total=len(comparison)
unchanged=sum(row['pixel_identical'] for row in comparison.values())
exact=sum(row['outside_requested_mask_unchanged'] for row in comparison.values())
parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>去水印原尺寸回归</title><style>'+style+'</style><h1>居中水印优化 · 原尺寸回归</h1><p>'+f'共 {total} 张，其他 {unchanged} 张输出逐像素一致。{exact}/{total} 张在默认参数下未修改请求修复范围之外的像素。自动化测试请单独运行 unittest 测试集。'+'</p><p class="note">像素误差越低越好，不能解释为去除成功率。检测和修复仅读取待处理原图；无水印参考图与标注只在推理结束后评分。部分高清图仍超过 120 秒；耗时会受模型加载、CPU 与并行诊断影响。本次并非全样式达标验收。</p><img src="before_after.png" alt="前后对照"><h2>正文交叠细节</h2><img src="document_detail.png" alt="正文细节"><h2>全量结果</h2><table><tr><th>样例</th><th>本轮前误差</th><th>本轮后误差</th><th>较原图改善</th><th>本轮变化</th><th>该次耗时</th><th>结果</th></tr>']
for stem,row in comparison.items():
 remaining=row['remaining_no_improvement_vs_input'];cls='bad' if remaining else 'good';run=after[stem];parts.append(f'<tr><td>{html.escape(stem)}</td><td>{row["before_roi_mae"]:.2f}</td><td>{row["after_roi_mae"]:.2f}</td><td class="{cls}">{"尚无有效改善" if remaining else "误差下降，仍需视觉复核"}</td><td>{"输出一致" if row["pixel_identical"] else "本轮改善"}</td><td>{run["elapsed_ms"]/1000:.2f} 秒</td><td><a href="after/{stem}.png">原尺寸 PNG</a></td></tr>')
parts.append('</table><h2>尚待解决</h2><p>big_center_01、big_center_03，以及 dot_grid_01、dot_grid_03、dot_grid_04 的当前水印区域误差未优于输入。它们属于既有问题，本轮没有将其标成通过。居中照片虽有改善，局部补图痕迹仍然可见。其余样例的误差下降也不意味着文字或照片细节完全恢复。</p><p><a href="before/metrics.json">优化前指标</a> · <a href="after/metrics.json">优化后指标</a> · <a href="comparison.json">像素一致性核对</a> · <a href="workbench_document.png">真实批次队列界面</a></p></html>')
(out/'report.html').write_text(''.join(parts),'utf-8')
print('Saved report, full comparison and document detail. Unchanged:',sum(r['pixel_identical'] for r in comparison.values()),'/',len(comparison))
