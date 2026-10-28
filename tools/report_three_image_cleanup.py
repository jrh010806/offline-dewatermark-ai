"""Human-readable review of three application outputs, including site references."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'out/three_image_cleanup/20261027/run_03')
    args=parser.parse_args();out=args.run.resolve()
    rows=json.loads((out/'results.json').read_text());reviews=json.loads((out/'visual_review.json').read_text())
    site=ROOT/'data/网站测试结果/WatermarkRemover.io/2026-10-27'
    labels={'core_001':('01_淡字照片','淡字照片'),'core_015':('02_深色图表','深色水印图表'),'real_003':('03_白字文档','白色重复水印文档')}
    refs={'core_015':site/'01_深色大字图表_处理后_WatermarkRemover网站结果.png','real_003':site/'02_拍摄中文文档_处理后_WatermarkRemover网站结果.png'}
    crops={'core_001':(510,412,770,500),'core_015':(270,260,880,620),'real_003':(300,650,950,1100)}
    font=ImageFont.truetype(str(ROOT/'data/watermark_training_increment_v1_1/assets/fonts/NotoSansCJKsc-Regular.otf'),20)
    data=[];assets={}
    for row in rows:
        cid=row['case_id'];prefix,title=labels[cid];folder=out/cid
        views={}
        choices=[('input','原图'),('output','本轮自动结果'),('selection_overlay','红色水印候选选区'),('baseline','旧版结果')]
        if cid in refs:
            shutil.copy2(refs[cid],folder/'website.png');choices.append(('website','网站对照结果'))
        for name,label in choices:
            src=folder/(name+'.png');dst=out/(prefix+'_'+label+'.png');shutil.copy2(src,dst)
            cropname=prefix+'_'+label+'_局部.jpg'
            Image.open(src).crop(crops[cid]).save(out/cropname,quality=95)
            views[name]={'full':dst.name,'crop':cropname,'label':label}
            assets[dst.name]=hashlib.sha256(dst.read_bytes()).hexdigest()
        w=610;b=crops[cid];h=round((b[3]-b[1])*w/(b[2]-b[0]))
        sheet=Image.new('RGB',(2*w,2*(h+36)),'#edf1f5');draw=ImageDraw.Draw(sheet)
        for i,(name,label) in enumerate([('input','原图'),('baseline','旧版结果'),('selection_overlay','红色：自动选到的水印候选'),('output','本轮自动结果')]):
            x=i%2*w;y=i//2*(h+36);draw.text((x+8,y+3),label,font=font,fill='black')
            sheet.paste(Image.open(folder/(name+'.png')).crop(b).resize((w,h)),(x,y+36))
        sheet.save(folder/'comparison.jpg',quality=95)
        data.append({'id':cid,'title':title,'seconds':round(row['seconds'],1),'views':views,**reviews[cid]})
    data_json=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三张图 · 本轮自动去水印结果</title><style>
body{margin:0;background:#f2f5f7;color:#172b3a;font:16px/1.7 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:28px 22px}h1{font-size:30px;line-height:1.3}h2{margin:8px 0;font-size:23px}.muted{color:#647582}.notice{background:#e4eef3;border-left:4px solid #327a96;padding:14px 20px}nav,.controls{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}button{background:white;color:#243e50;border:1px solid #b7c9d4;border-radius:7px;padding:9px 16px;font:inherit;cursor:pointer}button.active{background:#165d78;color:white;border-color:#165d78}article{padding:22px;background:white;border:1px solid #d9e2e7;border-radius:12px;margin:18px 0}.review{padding:12px 16px;background:#fff4dc;margin:12px 0}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{min-width:0;border:1px solid #dce3e7;background:#f6f8fa}.panel p{margin:0;padding:8px 12px;background:#eaf0f4;font-weight:600}.panel img{display:block;width:100%;height:auto;cursor:zoom-in}.links a{margin-right:15px}a{color:#0b6688}details{margin-top:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}footer{padding:20px 0;color:#647582}code{overflow-wrap:anywhere}@media(max-width:650px){main{padding:16px 10px}article{padding:13px}.compare{gap:5px}.panel p{font-size:13px;padding:6px}button{padding:7px 11px;font-size:14px}h1{font-size:25px}}
</style><main><p class="muted">2026-10-27 · 三张固定开发样本 · 本地离线自动处理</p><h1>先看水印清没清掉，再看哪里需要精修</h1><div class="notice">这次直接运行程序：没有提前提供水印位置、干净参考或网站结果。网站图片只放在这里作对照。<br>照片和图表的明显水印基本清掉；文档也有明显改善，但仍有残影和局部正文损伤，尚未达到网站效果。最终是否符合你的阶段要求，由你看图验收。</div><nav id="cases"></nav><article><h2 id="title"></h2><div id="review" class="review"></div><p id="reason"></p><p id="damage"></p><p id="time" class="muted"></p><div class="controls" id="views"></div><div class="controls"><button id="full" class="active">看整张</button><button id="crop">看水印局部</button></div><div class="compare"><div class="panel"><p>原图</p><a id="leftLink" target="_blank"><img id="left" alt="原图"></a></div><div class="panel"><p id="rightTitle">本轮自动结果</p><a id="rightLink" target="_blank"><img id="right" alt="处理结果"></a></div></div><p class="muted">点击图片可打开原尺寸。红色显示候选水印范围，文档中的正文会另行保护，并非红色区域全部重画。</p><p class="links" id="downloads"></p></article>
<details><summary>改了什么、验证了什么</summary><p>深色图表：接住原图中已读到的大号斜字，只把相应深色笔画及抗锯齿边缘交给原有补图模型。淡字照片：通过颜色差异先增强淡字，再找位置，补图仍用原有模型。白字文档：增强“比纸面更亮”的重复字，补齐识别不完整的两端，再用原有模型清理亮色笔画，较暗正文保留。</p><p>真实文档的纸色填充尝试仍有明显字影，已保留为run_01/run_02记录；当前展示run_03，不再把估计纸色贴进输出。图表重叠的194、264、313被一起清掉；文档仍有模糊和断笔，不能宣称完全保真。</p><p>83项工程测试通过。另测两张无水印保护样本，输出均逐字节不变；这只是本轮保护检查，不代表所有无水印图片都不会误删。修改候选区外像素不变。三张是反复检查过的开发样本，不是独立测试集，不计算通用成功率。原有严格参考评分完整保留，不能与用户最新阶段目标混为一谈。</p><p><a href="protocol.json">运行协议</a> · <a href="results.json">原始记录与参考评分</a> · <a href="visual_review.json">逐张视觉结论</a> · <a href="guards.json">无水印保护检查</a> · <a href="provenance.json">代码与输入摘要</a> · <a href="verification.json">交付复核</a> · <a href="工程测试记录.txt">工程测试输出</a> · <a href="说明_三张自动结果.md">文字说明</a></p></details><footer>本机本次端到端处理用时包含自动定位、修补和程序检查；不包含报告生成。文件名以01照片、02图表、03文档区分，原图、候选选区、旧版、当前结果和网站结果分别保存。</footer></main><script>
const items=DATA;let current=items[0],view='output',area='full';const $=id=>document.getElementById(id);
function draw(){
 $('cases').replaceChildren(...items.map((r,i)=>{let b=document.createElement('button');b.textContent=(i+1)+' · '+r.title;b.className=r.id===current.id?'active':'';b.onclick=()=>{current=r;view='output';draw()};return b}));
 $('title').textContent=current.title;$('review').textContent=current.watermark;$('reason').textContent='这次为什么有变化：'+current.reason;$('damage').textContent='还需要精修：'+current.damage;$('time').textContent='本机本次自动处理约 '+current.seconds+' 秒。';
 $('views').replaceChildren(...Object.entries(current.views).filter(([k])=>k!=='input').map(([k,v])=>{let b=document.createElement('button');b.textContent=v.label;b.className=k===view?'active':'';b.onclick=()=>{view=k;draw()};return b}));
 const a=current.views.input,b=current.views[view];$('left').src=a[area];$('right').src=b[area];$('leftLink').href=a.full;$('rightLink').href=b.full;$('rightTitle').textContent=b.label;$('full').className=area==='full'?'active':'';$('crop').className=area==='crop'?'active':'';
 $('downloads').replaceChildren(...Object.values(current.views).map(v=>{let a=document.createElement('a');a.href=v.full;a.textContent='打开'+v.label;a.target='_blank';return a}));
}
$('full').onclick=()=>{area='full';draw()};$('crop').onclick=()=>{area='crop';draw()};draw();
</script></html>'''.replace('DATA',data_json)
    (out/'report.html').write_text(page)
    lines=['# 三张图的本轮自动去水印结果','','当前展示run_03。输入、模型均本地使用，原图和竞品文件不修改。网站结果仅用于本页面对照，不进入推理。','']
    for r in data:
        lines += [f'## {r["title"]}',r['watermark'],'原因：'+r['reason'],'待精修：'+r['damage'],f'本机本次端到端用时约{r["seconds"]}秒。','']
    lines += ['## 验收口径','先看明显水印是否清净，局部内容损伤单独记录，按用户意见可以暂缓精修。文档仍有残影和正文损伤，不能算已经达到网站效果。','83项工程测试通过；两张内部无水印保护样本保持原图。三张固定开发样本不构成通用成功率证明。原有严格参考评分保留在results.json。','', '## 复现','HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 WM_ORT_THREADS=4 .venv/bin/python tools/run_three_image_cleanup.py --out <新的输出目录>','已有results.json时拒绝覆盖。依赖项目固定数据、本地OCR和LaMa缓存。页面生成使用本目录visual_review.json：','.venv/bin/python tools/report_three_image_cleanup.py --run <输出目录>','']
    (out/'说明_三张自动结果.md').write_text('\n'.join(lines))
    (out/'assets_sha256.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2))
    print(out/'report.html')


if __name__=='__main__':main()
