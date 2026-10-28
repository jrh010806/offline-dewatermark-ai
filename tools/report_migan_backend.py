"""Render the fixed-mask MI-GAN document comparison from saved outputs only."""
import hashlib
import html
import json
from pathlib import Path
import shutil

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/repair_backend_compare/20261027_migan_fixed'


def main():
    results=json.loads((OUT/'real_003/results.json').read_text())
    decision=json.loads((OUT/'decision.json').read_text())
    website=ROOT/'data/网站测试结果/WatermarkRemover.io/2026-10-27/02_拍摄中文文档_处理后_WatermarkRemover网站结果.png'
    src=OUT/'real_003'
    choices=[('input','文档_原图_白色重复水印.png',src/'input.png','有水印原图'),
             ('lama_native','文档_现有LaMa_原尺寸分块.png',src/'lama_native.png','现有模型 LaMa'),
             ('migan_native','文档_MI-GAN_同样原尺寸分块.png',src/'migan_native.png','MI-GAN 同样分块'),
             ('migan_official','文档_MI-GAN_官方整体缩放.png',src/'migan_official.png','MI-GAN 官方整体缩放'),
             ('website','文档_WatermarkRemover网站结果.png',website,'网站对照结果'),
             ('mask','文档_固定修补选区_红色叠加.png',src/'fixed_mask_overlay.png','固定修补选区')]
    views={};assets={}
    for name,dstname,original,label in choices:
        target=OUT/dstname;shutil.copy2(original,target)
        crop=target.stem+'_中段局部.jpg'
        Image.open(original).crop((300,650,950,1100)).save(OUT/crop,quality=95)
        views[name]={'full':dstname,'crop':crop,'label':label}
        assets[dstname]=hashlib.sha256(target.read_bytes()).hexdigest()
    methods=results['methods'];a=methods['lama_native']['repair_seconds'];b=methods['migan_native']['repair_seconds'];c=methods['migan_official']['repair_seconds']
    table=f'<tr><td>现有 LaMa · 原尺寸分块</td><td>{a:.2f} 秒</td><td>保留作基线；仍有细残影和局部断笔</td></tr><tr><td>MI-GAN · 同样原尺寸分块</td><td>{b:.2f} 秒</td><td>更快；未建立画质优势，部分正文更糊</td></tr><tr><td>MI-GAN · 官方整体缩放</td><td>{c:.2f} 秒</td><td>明显水印轮廓残留，不采用</td></tr>'
    rendered_views=json.dumps(views,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MI-GAN 对照 · 更快，但暂不替换</title><style>
body{margin:0;background:#f2f5f7;color:#193140;font:16px/1.75 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:26px 22px}h1{font-size:30px;line-height:1.35}h2{font-size:23px}.decision{background:#fff1d8;border-left:5px solid #b47b1c;padding:16px 22px}section{background:white;border:1px solid #dbe3e8;border-radius:12px;padding:22px;margin:22px 0}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #dce3e8;text-align:left}.muted{color:#627685}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}button{font:inherit;padding:8px 14px;border:1px solid #bacbd5;border-radius:7px;background:white;color:#22475d;cursor:pointer}button.active{background:#185f79;color:white}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{border:1px solid #dce4e9;min-width:0}.panel p{margin:0;padding:8px 12px;background:#eaf0f4;font-weight:600}.panel img{display:block;width:100%;height:auto}a{color:#076b8c}details{margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}@media(max-width:650px){main{padding:16px 10px}section{padding:13px}td,th{font-size:13px;padding:6px}h1{font-size:25px}.compare{gap:5px}button{font-size:14px;padding:6px 10px}}
</style><main><p class="muted">2026-10-27 · 一张固定文档 · 同一修补选区 · 本地 CPU 4 线程</p><h1>MI-GAN 跑完了：更快，但没有证明更好</h1><div class="decision"><strong>建议：保留当前默认模型。</strong><br>原尺寸分块的修补用时约缩短到原来的1/7.6，但水印没有完全去净，部分正文更模糊。这个替代模型没有解决目前文档的画质差距。</div><section><h2>只换模型后的结果</h2><table><tr><th>方法</th><th>本机本次修补用时</th><th>目视结论</th></tr>TABLE</table><p class="muted">这里只计修补：不含下载、模型加载、水印检测。两种原尺寸方法使用相同的512分块、上下文、64像素重叠和合成方式。官方整体缩放含不同的裁切、缩放和软融合，只作完整封装的实际效果对照。</p><p>MI-GAN 同分块有些地方更平滑，但部分正文细笔画更模糊，尚不能认定画质优于当前结果。官网默认整体缩放处理很快，但较粗的水印轮廓仍清楚可见。用户提供的网站结果整体更干净。</p></section>
<section><h2>直接看图比较</h2><p>左边固定为现有 LaMa，右边切换候选、网站或原图。默认看整张，也可放大中段文字。</p><div class="controls" id="views"></div><div class="controls"><button id="full" class="active">看整张</button><button id="crop">放大中段文字</button></div><div class="compare"><div class="panel"><p>现有 LaMa · 固定选区</p><a id="leftLink" target="_blank"><img id="left" alt="现有模型结果"></a></div><div class="panel"><p id="rightLabel"></p><a id="rightLink" target="_blank"><img id="right" alt="对照结果"></a></div></div><p class="muted">点击图片查看原尺寸。网站图片仅供目视对照，不进入选区或模型推理，不当作干净真值。</p></section>
<section><h2>这次能确定什么</h2><p>现有 LaMa 的复算结果与上一轮保存的程序结果逐像素一致。两种模型用的是同一张输入、同一个实际修补选区；选区外和受保护像素均保持原图。因此，画面差别来自修补方法，而非临时改变水印位置。</p><p>这次没有证明“换一个现成模型就能解决文档”。MI-GAN 可以保留为速度候选，但没有足够理由替换默认处理。按事先规则，没有继续扩大到另外两张；它们此前的程序结果完整保留。</p><p>“水印没有完全去净”和“局部内容受损”分别记录。用户允许小范围损伤暂缓精修，仍不等于残影明显或正文更糊时就可以只凭速度判胜。</p></section>
<details><summary>技术公平性、局限与原始证据</summary><p>模型从 Picsart 官方仓库指向的作者托管地址下载，固定了版本和SHA-256。原尺寸版本仅提取官方ONNX图中的同一网络；权重逐字节一致，独立合成数组上输入预处理与网络输出差异均为0。没有重新训练、调参或更换选区。</p><p>官方封装会将裁切区缩到512，并在边界与原图软融合。对于较细笔画，这是与原尺寸分块不同的处理条件；不能把两者差异全归因于网络权重。</p><p>一张真实文档没有干净真值，本次正文判断是视觉对照，未逐字核验。局部亮笔画对比度只是辅助诊断，不是水印去除率或OCR准确率。本机单次计时不能当作所有图片的速度承诺。生产wm/代码和默认模型未改动。</p><p><a href="protocol.json">冻结协议</a> · <a href="real_003/results.json">原始结果与用时</a> · <a href="decision.json">逐项结论</a> · <a href="model_provenance.json">模型来源与校验</a> · <a href="extraction_check.json">网络提取一致性</a> · <a href="verification.json">交付复核</a> · <a href="说明_固定选区换模型对照.md">文字说明</a></p><p><a href="https://github.com/Picsart-AI-Research/MI-GAN">MI-GAN 官方项目</a> · <a href="../latest_three/report.html">返回三张图片的当前结果</a></p></details></main><script>
const views=VIEWS;let selected='migan_native',area='full';const $=id=>document.getElementById(id);
function render(){
 $('views').replaceChildren(...Object.entries(views).filter(([k])=>k!=='lama_native').map(([k,v])=>{let b=document.createElement('button');b.textContent=v.label;b.className=k===selected?'active':'';b.onclick=()=>{selected=k;render()};return b}));
 const left=views.lama_native,right=views[selected];$('left').src=left[area];$('right').src=right[area];$('leftLink').href=left.full;$('rightLink').href=right.full;$('rightLabel').textContent=right.label;$('full').className=area==='full'?'active':'';$('crop').className=area==='crop'?'active':'';
}
$('full').onclick=()=>{area='full';render()};$('crop').onclick=()=>{area='crop';render()};render();
</script></html>'''.replace('TABLE',table).replace('VIEWS',rendered_views)
    (OUT/'report.html').write_text(page)
    md=f'''# 固定选区换模型：MI-GAN 对照

结论：暂不替换当前默认模型。MI-GAN同分块明显更快，但未建立文档画质优势，水印未完全去净，部分正文细笔画更模糊。

| 方法 | 本机本次修补用时 | 结论 |
|---|---:|---|
|现有LaMa，原尺寸分块|{a:.3f}秒|复现上一轮结果；仍有细残影和局部断笔|
|MI-GAN，同样原尺寸分块|{b:.3f}秒|约快{a/b:.1f}倍；未证明更好的清洁度或正文保留|
|MI-GAN，官方整体缩放|{c:.3f}秒|明显水印轮廓残留，不采用|

只计修补，不含下载、模型加载或选区检测。原尺寸两种模型使用完全相同的512瓦片、上下文、64像素重叠和最终合成。实际修补范围固定为{results['effective_mask_pixels']}像素，候选水印区内另外{results['protected_candidate_pixels']}像素被原有规则保护。全部输出在实际修补区外逐像素保留输入。

官方整体封装另有裁切、缩放和软融合，不能算同分辨率纯网络对照。原尺寸MI-GAN从官方预转换ONNX提取同一核心网络；预处理差异为0、核心网络输出差异为0，权重字节未变。详见extraction_check.json。

本轮固定一张真实文档，不改水印掩膜、不训练、不调整参数。由于没有确立画质优势，按预定条件不继续测另外两张，保留之前结果。网站图仅供事后对照，未进入推理，不是干净真值；没有逐字核验，不给通用去水印成功率或OCR准确率。

生产wm/源码和默认模型未修改。复制保存的模型后，推理可离线复现：

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 WM_ORT_THREADS=4 .venv/bin/python tools/probe_migan_backend.py --case real_003

为防覆盖，已有real_003/results.json时脚本拒绝重跑；需保留本次目录后另设新输出目录，或修改脚本的OUT常量指向新目录并复制冻结协议与模型。

模型来源：https://github.com/Picsart-AI-Research/MI-GAN ，作者托管版本406830d0fa60666da0071c342ad2fbc8f30c5c64。原始下载SHA-256与派生核心网络SHA-256保存在model_provenance.json、extraction_check.json。
'''
    (OUT/'说明_固定选区换模型对照.md').write_text(md)
    (OUT/'report_assets_sha256.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2))
    print(OUT/'report.html')


if __name__=='__main__':main()
