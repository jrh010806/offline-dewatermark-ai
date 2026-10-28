"""Build a local, source-backed before/after report for the frozen new-three run."""
from pathlib import Path
import html
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'out/unseen_three/20261027_fixed'

def esc(v):
    return html.escape(str(v), quote=True)

def main():
    rows = json.loads((OUT/'results.json').read_text())
    review = json.loads((OUT/'visual_review.json').read_text())
    cards = []
    table = []
    for n, row in enumerate(rows, 1):
        cid = row['case_id']; r = review['cases'][cid]
        table.append(f'<tr><td>{esc(r["title"])}</td><td>{esc(r["watermark"])}</td><td>{esc(r["damage"])}</td><td>{row["seconds"]:.1f} 秒</td></tr>')
        box = row['diagnostic_crop']
        w, h = box[2]-box[0], box[3]-box[1]
        cards.append(f'''<article id="case{n}"><div class="eyebrow">新样本 {n} / 3 · {esc(cid)}</div>
<h2>{esc(r['title'])} <span class="badge {esc(r['status'])}">{esc(r['verdict'])}</span></h2>
<p class="lead">{esc(r['finding'])}</p>
<div class="controls"><button onclick="mode(this,'crop')" class="active">看水印局部</button><button onclick="mode(this,'full')">看整张</button><button onclick="aux(this,'selection_overlay')">红色自动选区</button><button onclick="aux(this,'clean_reference')">无水印参考</button></div>
<div class="compare" data-cid="{esc(cid)}" data-mode="crop" style="aspect-ratio:{w}/{h}">
<img class="after" src="{cid}/output_crop.png" alt="自动处理后"><img class="before" src="{cid}/input_crop.png" alt="处理前"><div class="divider"></div><b class="left">处理前</b><b class="right">自动处理后</b></div>
<input class="slider" type="range" min="0" max="100" value="50" aria-label="拖动比较处理前后" oninput="slide(this)"><p class="hint">拖动滑杆对比同一位置。默认展示全部水印范围；原图和结果可点下方链接放大。</p>
<figure class="aux" hidden><img alt="辅助检查图"><figcaption></figcaption></figure>
<div class="notes"><p><strong>为什么：</strong>{esc(r['reason'])}</p><p><strong>局部损伤：</strong>{esc(r['damage_detail'])}</p><p><strong>这张说明什么：</strong>{esc(r['implication'])}</p></div>
<details><summary>查看检测记录与参考误差</summary><pre>{esc(json.dumps({k:row[k] for k in ['method','repair_mode','mask_pixels','changed_pixels','reference_diagnostics']},ensure_ascii=False,indent=2))}</pre><p>参考误差同时包含水印残留和修补损伤，不等于水印去除率；程序内部的 passed 字段也不等于肉眼验收通过。</p></details>
<p class="links"><a href="{cid}/input.png" target="_blank">原图</a><a href="{cid}/output.png" target="_blank">自动处理结果</a><a href="{cid}/selection_overlay.png" target="_blank">自动选区</a><a href="{cid}/clean_reference.png" target="_blank">无水印参考</a></p></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>换三张新图，当前程序能否继续成功？</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#1e293b;font:16px/1.75 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:1120px;margin:0 auto;padding:44px 24px 70px}header,article{background:white;border:1px solid #dce3ed;border-radius:16px;padding:28px;margin-bottom:24px}h1{font-size:32px;line-height:1.35;margin:8px 0 18px}h2{font-size:24px;margin:8px 0 12px}.eyebrow{color:#536680;font-size:13px;letter-spacing:.04em}.summary{background:#eef4ff;border-left:4px solid #346bcb;padding:14px 18px}.badge{font-size:13px;padding:5px 9px;border-radius:6px;vertical-align:middle}.pass{color:#14643b;background:#dff4e8}.fail{color:#9d3026;background:#fce7e3}.partial{color:#8c600b;background:#fff0ce}table{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0}th,td{padding:12px;text-align:left;border-bottom:1px solid #dce3ed}th{background:#f5f7fa}.lead{font-size:17px}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:20px 0 14px}button{font:inherit;font-size:14px;border:1px solid #c8d3e3;padding:7px 12px;background:white;border-radius:7px;cursor:pointer}button.active{background:#275fb6;color:white;border-color:#275fb6}.compare{position:relative;width:100%;overflow:hidden;background:#f0f2f5}.compare img{position:absolute;width:100%;height:100%;object-fit:contain;inset:0}.before{clip-path:inset(0 50% 0 0)}.divider{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid #2871d5;pointer-events:none}.compare b{position:absolute;top:10px;background:#10243de6;color:white;font-size:12px;padding:3px 8px;border-radius:5px}.left{left:10px}.right{right:10px}.slider{width:100%;accent-color:#2767c4;margin:14px 0 0}.hint,details,.links{font-size:13px;color:#596980}.aux{margin:14px 0;padding:12px;background:#f3f6fa}.aux img{display:block;max-width:100%;max-height:900px;margin:auto}.aux figcaption{text-align:center;color:#5d6d82}.notes{background:#f6f8fb;padding:12px 18px;border-radius:10px}.notes p{margin:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace}.links{display:flex;gap:20px;flex-wrap:wrap}a{color:#225cae}.limits{font-size:14px;color:#52627a}footer{font-size:13px;color:#62738b}@media(max-width:650px){main{padding:16px 10px}header,article{padding:18px}h1{font-size:26px}h2{font-size:20px}th,td{padding:7px;font-size:12px}}
</style><main><header><div class="eyebrow">固定版本 · 三张新样本 · 2026-10-27</div><h1>换三张新图，当前程序能否继续成功？</h1>'''
    page += '<div class="summary">'+esc(review['conclusion'])+'</div>'
    page += '<table><thead><tr><th>图片</th><th>水印是否去净</th><th>内容损伤</th><th>耗时</th></tr></thead><tbody>'+''.join(table)+'</tbody></table>'
    page += '<p><strong>下一步：</strong>'+esc(review['next_step'])+'</p>'
    page += '<p class="limits">固定已接受的程序版本。照片、图表各跑一次；文档首次运行因记录脚本检查中断，未保存输出，同配置重跑一次，详见 <a href="runner_incident.json">中断记录</a>。没有提供水印位置、答案图或网站结果，没有针对新图调参数。三张来自已有测试分区，底图来源均不在最小回归集中；使用相同合成模板，不能视为真实图片的整体成功率。新文档是浅绿色重复水印，与此前拍照文档的白色重复水印不同。此前接受的三张结果保持原样。</p></header>'
    page += ''.join(cards)
    page += '''<footer><p>原图为项目已有数据；合成水印参考仅在三张自动推理全部结束后用于诊断和展示。红色为程序自动选择的区域，不是人工提供的选区。判定为本轮人工视觉审阅，尚未由用户逐张确认。</p><p><a href="protocol.json">固定测试规则</a> · <a href="selection.json">样本来源</a> · <a href="results.json">原始运行记录</a> · <a href="provenance.json">代码与输入哈希</a> · <a href="verification.json">归档校验</a> · <a href="visual_review.json">逐张审阅结论</a></p></footer></main><script>
function slide(el){let c=el.previousElementSibling,v=el.value;c.querySelector('.before').style.clipPath=`inset(0 ${100-v}% 0 0)`;c.querySelector('.divider').style.left=v+'%'}
function mode(b,m){let a=b.closest('article'),c=a.querySelector('.compare'),cid=c.dataset.cid,s=m==='crop'?'_crop':'';c.dataset.mode=m;c.querySelector('.before').src=cid+'/input'+s+'.png';c.querySelector('.after').src=cid+'/output'+s+'.png';c.querySelector('.after').onload=function(){c.style.aspectRatio=this.naturalWidth+'/'+this.naturalHeight};a.querySelectorAll('.controls button').forEach(x=>x.classList.remove('active'));b.classList.add('active');a.querySelector('.aux').hidden=true}
function aux(b,name){let a=b.closest('article'),c=a.querySelector('.compare'),f=a.querySelector('.aux'),s=c.dataset.mode==='crop'?'_crop':'';f.querySelector('img').src=c.dataset.cid+'/'+name+s+'.png';f.querySelector('figcaption').textContent=name==='selection_overlay'?'红色区域：程序自动选择的水印候选位置':'真实无水印参考：仅用于事后检查，没有进入自动处理';f.hidden=false}
</script></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    print(OUT/'report.html')

if __name__ == '__main__':
    main()
