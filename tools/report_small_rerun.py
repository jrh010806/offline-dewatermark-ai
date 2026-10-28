"""Compact local report: nine-row overview and one inspectable image at a time."""
from pathlib import Path
import html
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/small_dataset_rerun/20261027_fixed9'

def esc(v):return html.escape(str(v),quote=True)

def main():
    rows=json.loads((OUT/'results.json').read_text())
    review=json.loads((OUT/'visual_review.json').read_text())
    table=[];cards=[];options=[]
    for i,r in enumerate(rows):
        cid=r['case_id'];v=review['cases'][cid]
        options.append(f'<option value="{cid}">{esc(r["label"])} · {esc(v["verdict"])}</option>')
        table.append(f'<tr class="{v["status"]}"><td><button class="case-link" onclick="choose(\'{cid}\')">{esc(r["label"])}</button></td><td>{esc(v["verdict"])}</td><td>{esc(v["watermark"])}</td><td>{esc(v["damage"])}</td><td>{r["seconds"]:.1f}s</td></tr>')
        box=r.get('diagnostic_crop',[0,0,1,1]);w,h=box[2]-box[0],box[3]-box[1]
        reference_button='<button onclick="aux(this,\'clean_reference\')">无水印参考</button>' if (OUT/cid/'clean_reference.png').exists() else ''
        if r['status']=='completed':
            comparison=f'''<div class="controls"><button onclick="mode(this,'crop')" class="active">水印局部</button><button onclick="mode(this,'full')">整张图片</button><button onclick="side(this,100)">只看原图</button><button onclick="side(this,0)">只看结果</button></div>
<div class="compare" data-mode="crop" style="aspect-ratio:{w}/{h}"><img class="after" src="{cid}/output_crop.png" alt="自动处理后"><img class="before" src="{cid}/input_crop.png" alt="处理前"><div class="divider"></div><b class="left">原图</b><b class="right">当前结果</b></div>
<input type="range" class="slider" min="0" max="100" value="50" aria-label="拖动比较处理前后" oninput="slide(this)"><p class="hint">拖动滑杆比较同一位置，也可点击“只看结果”。红色选区是程序自行判断的位置。</p>
<div class="controls"><button onclick="aux(this,'selection_overlay')">自动选区</button><button onclick="aux(this,'baseline')">{esc(r['baseline_label'])}</button>{reference_button}</div><figure class="aux" hidden><img alt="辅助检查图"><figcaption></figcaption></figure>'''
        else:comparison=f'<p>运行失败：{esc(r.get("error"))}，保留在本轮分母中。</p>'
        cards.append(f'''<article id="{cid}" data-cid="{cid}" hidden><div class="eyebrow">{i+1} / 9 · {cid}</div><h2>{esc(r['label'])} <span class="badge {v['status']}">{esc(v['verdict'])}</span></h2><p class="finding">{esc(v['finding'])}</p>{comparison}
<div class="notes"><p><strong>原因：</strong>{esc(v['reason'])}</p><p><strong>内容损伤：</strong>{esc(v['damage_detail'])}</p></div><details><summary>运行记录与参考误差</summary><pre>{esc(json.dumps(r,ensure_ascii=False,indent=2))}</pre><p>历史严格指标同时检查水印残留、保真和笔画一致性，与本阶段“去掉明显水印、小范围损伤后修”的标准不同。参考误差不是去除率；没有参考图的文档不报告精确恢复率。</p></details>
<p class="links"><a href="{cid}/input.png" target="_blank">原图 PNG</a><a href="{cid}/output.png" target="_blank">当前结果 PNG</a><a href="{cid}/selection_overlay.png" target="_blank">自动选区 PNG</a></p></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>9 张小合集 · 当前版本回归</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f6;color:#243248;font:15px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1180px;margin:auto;padding:36px 22px 70px}header,article,.picker{padding:26px;background:white;border:1px solid #dbe4ee;border-radius:14px;margin-bottom:22px}h1{font-size:30px;line-height:1.35;margin:7px 0 18px}h2{font-size:23px;line-height:1.5;margin:6px 0 12px}.eyebrow{font-size:12px;color:#64768d}.summary{background:#eef4ff;border-left:4px solid #2b65be;padding:14px 18px;font-size:17px}.counts{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.counts span{padding:9px 14px;background:#f3f6fa;border-radius:8px}.counts strong{font-size:23px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;min-width:640px}td,th{text-align:left;border-bottom:1px solid #e0e7ef;padding:11px 10px}th{background:#f5f7fa}.case-link{padding:0;border:0;text-align:left;color:#225fae;background:none}.pass td:nth-child(2),.safe td:nth-child(2){color:#17603a}.fail td:nth-child(2){color:#a7352b}.limits,.hint,details{font-size:13px;color:#62738b}.limits{margin-bottom:0}.picker{padding:18px 22px;position:sticky;top:0;z-index:5;box-shadow:0 3px 10px #182d4910}.picker label{display:block;font-size:13px;color:#52667f;margin-bottom:6px}select{width:100%;padding:10px;font:inherit;border:1px solid #c3cfdf;border-radius:7px;background:white;color:#213c60}.badge{display:inline-block;font-size:12px;padding:3px 8px;border-radius:6px;vertical-align:middle}.badge.pass,.badge.safe{background:#e1f4e8;color:#196340}.badge.fail{background:#fce7e3;color:#a2372b}.finding{font-size:17px}.controls{display:flex;flex-wrap:wrap;gap:7px;margin:16px 0 12px}button{cursor:pointer;font:inherit;font-size:13px;padding:6px 11px;border:1px solid #c5d2e3;border-radius:6px;background:white;color:#334d6e}button.active{background:#2c64b8;color:white;border-color:#2c64b8}.compare{position:relative;overflow:hidden;width:100%;background:#eef2f6}.compare img{position:absolute;width:100%;height:100%;inset:0;object-fit:contain}.before{clip-path:inset(0 50% 0 0)}.divider{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid #2d70ce}.compare b{position:absolute;top:8px;font-size:12px;background:#173658e6;color:white;padding:3px 8px;border-radius:5px}.left{left:8px}.right{right:8px}.slider{width:100%;accent-color:#2968c4;margin:12px 0 0}.hint{margin:4px 0 14px}.aux{margin:12px 0;padding:14px;background:#f4f7fa}.aux img{display:block;max-width:100%;max-height:1000px;margin:auto}.aux figcaption{font-size:13px;color:#66778e;text-align:center}.notes{background:#f4f7fb;padding:10px 18px;border-radius:8px}.notes p{margin:8px 0}pre{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.links{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}a{color:#255fae}footer{color:#65768d;font-size:13px}[hidden]{display:none!important}@media(max-width:650px){main{padding:16px 10px}header,article{padding:18px}h1{font-size:25px}h2{font-size:20px}.summary{font-size:15px}.picker{padding:12px}.counts{gap:8px}.counts span{padding:6px 9px}}
</style><main><header><div class="eyebrow">2026-10-27 · 同一程序版本 · 固定小合集</div><h1>9 张小合集，重新检查当前效果</h1>'''
    page+='<div class="summary">'+esc(review['conclusion'])+'</div>'
    s=review['summary'];page+=f'<div class="counts"><span>水印阶段暂通过 <strong>{s["phase_passed"]} / 7</strong></span><span>水印未通过 <strong>{s["phase_failed"]}</strong></span><span>无水印安全保留 <strong>{s["negative_safe"]} / 2</strong></span></div>'
    page+='<div class="table-wrap"><table><thead><tr><th>图片 · 点击查看</th><th>本轮判断</th><th>水印残留</th><th>局部损伤</th><th>耗时</th></tr></thead><tbody>'+''.join(table)+'</tbody></table></div>'
    page+='<p><strong>下一步：</strong>'+esc(review['next_step'])+'</p><p class="limits">这是开发回归：原先 8 张小合集，加上前轮失败的建筑照片；包含已调试、已验收的图片。不是全新盲测，不能把本轮比例当作整体成功率。7 张水印图中有 1 张大块完全不透明遮挡，单独保留为能力边界。先评水印是否基本清除，小范围内容损伤另记；“阶段暂通过”是本轮视觉判断，尚未由用户逐张确认。全程没有修改算法或逐图调参数；每张自动运行一次。</p></header>'
    page+='<section class="picker"><label for="case-picker">一次看一张：可从总表点击，也可在这里切换</label><select id="case-picker" onchange="choose(this.value,false)">'+''.join(options)+'</select></section>'
    page+=''.join(cards)
    page+='''<footer><p>所有结果均保留。原始记录中程序 passed 不直接等同于去水印成功；历史严格指标也没有改成阶段视觉结果。无水印安全判定要求输出像素与输入完全相同。</p><p><a href="protocol.json">测试规则</a> · <a href="selection.json">固定样本清单</a> · <a href="results.json">完整运行记录</a> · <a href="visual_review.json">逐张视觉判断</a> · <a href="provenance.json">版本与模型哈希</a> · <a href="verification.json">归档校验</a></p></footer></main><script>
function choose(id,scroll=true){document.querySelectorAll('article').forEach(a=>a.hidden=a.dataset.cid!==id);document.querySelector('#case-picker').value=id;if(scroll)document.querySelector('.picker').scrollIntoView({behavior:'smooth',block:'start'})}
function slide(el){let c=el.previousElementSibling,v=Number(el.value);c.querySelector('.before').style.clipPath=`inset(0 ${100-v}% 0 0)`;c.querySelector('.divider').style.left=v+'%';c.querySelector('.left').hidden=v===0;c.querySelector('.right').hidden=v===100;c.querySelector('.divider').hidden=v===0||v===100}
function side(b,v){let a=b.closest('article'),s=a.querySelector('.slider');s.value=v;slide(s)}
function mode(b,m){let a=b.closest('article'),c=a.querySelector('.compare'),s=m==='crop'?'_crop':'';c.dataset.mode=m;let after=c.querySelector('.after');after.onload=function(){c.style.aspectRatio=this.naturalWidth+'/'+this.naturalHeight};after.src=a.dataset.cid+'/output'+s+'.png';c.querySelector('.before').src=a.dataset.cid+'/input'+s+'.png';a.querySelector('.aux').hidden=true;a.querySelectorAll('.controls button').forEach(x=>x.classList.remove('active'));b.classList.add('active')}
function aux(b,n){let a=b.closest('article'),c=a.querySelector('.compare'),f=a.querySelector('.aux'),s=c.dataset.mode==='crop'?'_crop':'';f.querySelector('img').src=a.dataset.cid+'/'+n+s+'.png';f.querySelector('figcaption').textContent=n==='baseline'?'此前结果，仅供对照':n==='clean_reference'?'无水印参考，仅在全部处理结束后用于评估':'红色区域为程序自行判断的水印候选位置';f.hidden=false}
choose('photo_021__local_text',false);
</script></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(OUT/'report.html')

if __name__=='__main__':main()
