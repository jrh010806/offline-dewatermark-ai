"""Build local, interactive visual evidence from the saved strategy experiment."""
from pathlib import Path
import argparse,json,html
import numpy as np
import cv2
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
NAMES={'current':'现有自动流程','current_mask_telea':'现有定位 + Telea','current_mask_lama':'现有定位 + LaMa','slbr_global':'SLBR 整图缩放','slbr_tiles':'SLBR 原尺寸分块','slbr_mask_lama':'SLBR 定位 + LaMa','known_mask_telea':'准确位置 + Telea（诊断）','known_mask_lama':'准确位置 + LaMa（诊断）','known_layer_unmix':'准确图层反混合（诊断）'}
NAMES={**{k:v for k,v in NAMES.items() if not k.startswith('known_')},'wdnet_global':'WDNet 整图缩放','wdnet_tiles':'WDNet 原尺寸分块','wdnet_unmix_tiles':'WDNet 预测图层反解',**{k:v for k,v in NAMES.items() if k.startswith('known_')}}
AUTO=[x for x in NAMES if not x.startswith('known_')]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args();folder=args.folder.resolve()
 rows=json.loads((folder/'results.json').read_text())
 if (folder/'wdnet_results.json').exists():
  extra={r['case_id']:r['methods'] for r in json.loads((folder/'wdnet_results.json').read_text())}
  for r in rows:r['methods'].update(extra.get(r['case_id'],{}))
  for r in rows:r['methods']={k:r['methods'][k] for k in NAMES if k in r['methods']}
 (folder/'combined_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 pos=[r for r in rows if not r['negative'] and r['score_mode']=='paired'];neg=[r for r in rows if r['negative']]
 def f(x):return '—' if x is None else f'{x:.2f}'
 def pct(x):return '—' if x is None else f'{x*100:.2f}%'
 tables=[];summary=[]
 for key,label in NAMES.items():
  vals=[r['methods'][key] for r in pos if key in r['methods']];ns=[r['methods'][key] for r in neg if key in r['methods']]
  if not vals:continue
  record=dict(method=key,label=label,positive_pass=sum(v['probe_pass'] for v in vals),positive_count=len(vals),negative_unchanged=sum(v['probe_pass'] for v in ns),negative_count=len(ns),macro_wm_mae=float(np.mean([v['wm_mae'] for v in vals])),macro_reduction=float(np.mean([v['wm_error_reduction_pct'] for v in vals])),macro_outside_mae=float(np.mean([v['outside_mae'] for v in vals])),median_seconds=float(np.median([v['seconds'] for v in vals])))
  summary.append(record);tables.append(f'<tr><td>{label}</td><td>{record["positive_pass"]}/{len(vals)}</td><td>{str(record["negative_unchanged"])+"/"+str(len(ns)) if ns else "不适用"}</td><td>{f(record["macro_wm_mae"])}</td><td>{f(record["macro_reduction"])}%</td><td>{f(record["macro_outside_mae"])}</td><td>{f(record["median_seconds"])}</td></tr>')
 (folder/'aggregate.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 reviews=json.loads((folder/'visual_review.json').read_text()) if (folder/'visual_review.json').exists() else {}
 cards=[]
 for r in rows:
  cid=r['case_id'];d=folder/cid;im=Image.open(d/'input.png').convert('RGB');w,h=im.size
  if (d/'truth_mask.png').exists() and not r['negative']:
   m=np.array(Image.open(d/'truth_mask.png'))>0; yy,xx=np.where(m);cx=int(np.median(xx));cy=int(np.median(yy))
  elif r['negative']:
   changes=np.zeros((h,w),np.float32);inp=np.array(im,dtype=np.float32)
   for key in r['methods']:
    changes=np.maximum(changes,np.abs(np.array(Image.open(d/(key+'.png')),dtype=np.float32)-inp).mean(2))
   heat=cv2.boxFilter(changes,-1,(min(640,w),min(420,h)),normalize=True)
   cy,cx=np.unravel_index(heat.argmax(),heat.shape)
  else:cx=w//2;cy=h//2
  cw=min(640,w);ch=min(420,h);x=max(0,min(w-cw,cx-cw//2));y=max(0,min(h-ch,cy-ch//2));box=(x,y,x+cw,y+ch)
  allkeys=['input']+list(r['methods'])+(['clean'] if (d/'clean.png').exists() else [])
  for key in allkeys:Image.open(d/(key+'.png')).crop(box).save(d/(key+'_crop.png'))
  # Inspection contact sheet uses the exact same crop for all methods.
  sheet=Image.new('RGB',(640*3,460*((len(allkeys)+2)//3)),'#f2f4f6');draw=ImageDraw.Draw(sheet)
  for i,key in enumerate(allkeys):
   ix=(i%3)*640;iy=(i//3)*460;sheet.paste(Image.open(d/(key+'_crop.png')),(ix,iy+32));draw.text((ix+8,iy+8),key,fill='black')
  sheet.save(d/'contact.jpg',quality=90)
  options=''.join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in NAMES.items() if k in r['methods'])
  trs=[]
  for k,v in r['methods'].items():
   trs.append(f'<tr><td>{NAMES[k]}</td><td>{f(v.get("wm_mae"))}</td><td>{f(v.get("wm_error_reduction_pct"))}%</td><td>{f(v.get("outside_mae"))}</td><td>{pct(v.get("mask_recall"))}</td><td>{f(v.get("ink_iou"))}</td><td>{f(v["seconds"])}</td><td>{"待目视" if v["probe_pass"] is None else "通过门槛" if v["probe_pass"] else "未通过"}</td></tr>')
  cards.append(f'''<article id="{cid}"><h2>{cid} · {html.escape(r['label'])}</h2><p>来源：{html.escape(r['source_id'])} · {'无水印对照' if r['negative'] else '真实采集，无精确真值' if r['score_mode']!='paired' else '合成水印，有精确真值'}。现有定位：<code>{r['extras']['current_method']}</code></p>
  <p class="note">{html.escape(reviews.get(cid,""))}</p><div class="controls"><label>比较方案 <select onchange="change(this,'{cid}')">{options}</select></label><label><input type="checkbox" onchange="crop(this,'{cid}')">查看相同位置的局部放大</label></div>
  <div class="images" data-case="{cid}"><figure><a href="{cid}/input.png" target="_blank"><img src="{cid}/input.png" data-role="input"></a><figcaption>输入原图</figcaption></figure><figure><a href="{cid}/current.png" target="_blank"><img src="{cid}/current.png" data-role="method"></a><figcaption data-caption>现有自动流程</figcaption></figure>{f'<figure><a href="{cid}/clean.png" target="_blank"><img src="{cid}/clean.png" data-role="clean"></a><figcaption>干净参考（不参与自动推理）</figcaption></figure>' if (d/'clean.png').exists() else '<figure><p>没有严格对齐的干净参考，不计算恢复率。</p></figure>'}</div>
  <details><summary>全部数值、掩码与对照图</summary><p><a href="{cid}/contact.jpg">同位置九宫格</a> · <a href="{cid}/current_mask.png">现有掩码</a> · <a href="{cid}/slbr_tiles_mask.png">SLBR 掩码</a></p><div class="scroll"><table><thead><tr><th>方案</th><th>水印区 MAE ↓</th><th>误差降低 ↑</th><th>区外 MAE ↓</th><th>定位召回 ↑</th><th>笔画 IoU ↑</th><th>秒 ↓</th><th>实验门槛</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div></details></article>''')
 conclusions=(folder/'conclusions.html').read_text() if (folder/'conclusions.html').exists() else '<p>实测完成后补充目视结论。</p>'
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>8 张最小样本 · 去水印技术实测</title><style>
 :root{color-scheme:light}body{margin:0;background:#eff3f6;color:#163047;font:16px/1.8 system-ui,-apple-system,sans-serif}main{max-width:1460px;margin:auto;padding:40px 28px}header{padding:28px 36px;background:#102c43;color:white;border-radius:18px}h1{font-size:34px;line-height:1.35;margin:8px 0}h2{font-size:24px;line-height:1.4}article,.panel{background:white;border:1px solid #dbe4ec;border-radius:12px;padding:26px;margin:22px 0}.tag{color:#8ce0cc;letter-spacing:2px}a{color:#087e84}header a{color:#8ce0cc}p{max-width:1100px}.note{border-left:4px solid #e6a641;background:#fff7e5;padding:14px 20px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}figure{margin:0}img{width:100%;height:480px;object-fit:contain;background:#eaf0f4}figcaption{font-weight:600;padding-top:10px}.controls{display:flex;gap:24px;align-items:center;flex-wrap:wrap;margin:20px 0}select{padding:10px;border:1px solid #b8c8d3;border-radius:6px;font:inherit}table{width:100%;border-collapse:collapse;font-size:14px;white-space:nowrap}th,td{padding:12px;border-bottom:1px solid #dce5ed;text-align:left}th{background:#f3f6f8}details{margin-top:24px}summary{cursor:pointer;color:#087e84}.scroll{overflow:auto}code{background:#eef2f6;padding:2px 5px}li{margin:8px 0}@media(max-width:800px){.images{grid-template-columns:1fr}main{padding:16px}img{height:360px}h1{font-size:27px}}
 </style><main><header><div class="tag">LOCAL EXPERIMENT / 2026.10.27</div><h1>水印为什么去不掉，换哪条技术路线？</h1><p>固定最小回归集抽取 8 张；5 张有精确真值的正例 + 2 张无水印对照 + 1 张真实醒图导出。9 种自动配置、3 种诊断配置。全部本机实际运行，结果保留原尺寸。</p></header>'''
 page+='<section class="panel"><h2>实测判断</h2>'+conclusions+'</section>'
 page+='<section class="panel"><h2>各方案对比</h2><p>下表是 5 张配对正例的逐图平均误差；误差降低为逐图百分比的平均。无水印安全保留要求逐像素不变。诊断方案使用已知水印标注，不能与自动方案并列宣称可交付能力。</p><div class="scroll"><table><tr><th>方案</th><th>正例门槛</th><th>负例保留</th><th>水印区 MAE ↓</th><th>误差降低 ↑</th><th>区外 MAE ↓</th><th>中位秒 ↓</th></tr>'+''.join(tables)+'</table></div></section>'
 page+='''<section class="panel"><h2>怎样阅读这些结果</h2><p>MAE 使用 0–255 的 RGB 通道绝对误差。所有方案的水印区域固定为真实 alpha&gt;0，区外固定为 alpha=0，避免算法通过缩小自己的掩码美化评分。笔画 IoU 是局部自适应二值笔画一致性，不是 OCR 字符正确率。</p><p class="note">本轮诊断门槛：水印区 MAE ≤3、区内误差&gt;10 的像素比例≤10%、区外 MAE≤1、区外误差 P99≤4、笔画 IoU≥0.90、耗时≤120 秒。它不等同旧版四条件，也不是商业成功率；目视仍须复核。负例放大图选择各方案误改最集中的区域，同一图片的所有方案使用同一裁剪。样本很少、照片底图有重复，不能外推总体表现。</p><p>自动推理只读取原图。准确位置诊断额外读取真值掩码；准确图层诊断额外读取真实 alpha 与颜色，反解 alpha&lt;0.98 的像素，其余交给已知位置的 LaMa。干净参考仅用于评分。所有补图掩码统一外扩 1 像素。SLBR 整图缩放到256×256后映回；分块保持原尺寸、256窗口/192步长/羽化融合。LaMa/OCR 使用 CPU 4线程，SLBR / WDNet 使用 Apple MPS GPU；加载和文件读写不计推理时间，组合方案保守计入前级完整处理。</p><p><a href="combined_results.json">逐图机器可读结果</a> · <a href="aggregate.json">汇总数据</a> · <a href="provenance.json">代码、模型、输入哈希与环境</a> · <a href="wdnet_provenance.json">WDNet模型来源与哈希</a></p></section>'''
 page+=''.join(cards)
 page+='''<section class="panel"><h2>已核对的技术出处</h2><p><a href="https://github.com/bcmi/SLBR-Visible-Watermark-Removal">SLBR 官方实现</a>：定位与背景重建联合网络，作者说明训练尺寸256×256，建议针对自身数据微调。<a href="https://github.com/advimman/lama">LaMa 官方实现</a>：掩码引导的图像补全。<a href="https://github.com/MRUIL/WDNet">WDNet 官方代码及其链接的预训练模型</a>：估计位置、透明度和水印颜色再重建背景；本轮还比较其精修前的预测图层反解。本报告测试的是这些具体权重及项目推理封装，不等于重新训练或测试了所有去水印模型。</p></section></main><script>
 const selected={},cropped={};function render(id){const root=document.querySelector('[data-case="'+id+'"]');root.querySelectorAll('img').forEach(img=>{const key=img.dataset.role==='method'?(selected[id]||'current'):img.dataset.role;img.src=id+'/'+key+(cropped[id]?'_crop':'')+'.png';img.parentNode.href=id+'/'+key+'.png'})}function change(el,id){selected[id]=el.value;document.querySelector('[data-case="'+id+'"] [data-caption]').textContent=el.options[el.selectedIndex].text;render(id)}function crop(el,id){cropped[id]=el.checked;render(id)}
 </script></html>'''
 (folder/'report.html').write_text(page)
 print(folder/'report.html')
if __name__=='__main__':main()
