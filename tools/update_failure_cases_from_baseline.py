"""Append a source-backed failure review; never run OCR or image inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.regression_guard import digest

RANK = {'good': 0, 'review': 1, 'poor': 2}
MODES = {'desktop': '桌面默认（补全开启）', 'desktop_protected': '桌面关闭笔画补全', 'cli': 'CLI默认'}
STYLE = {'logo_opaque':'不透明Logo','logo_transparent':'透明Logo','mixed_color':'多色文字与图案',
 'gradient':'渐变水印','multiple':'多重叠加','timestamp':'时间戳','website':'网址文字','faint':'极淡文字',
 'heavy_dark':'深色大字','outlined':'描边文字','center_white':'白色半透明大字','dot_grid':'点阵',
 'line_net':'线网','local_text':'局部文字','cropped_edge':'边缘裁切文字','shadow':'带阴影文字',
 'signature':'签名类','tiny_corner':'小角标','center_text':'居中大字','tiled_irregular':'非等距密集文字',
 'big_center':'历史居中大字','tiled_diagonal':'斜向平铺文字','vector_paths':'PDF图形路径',
 'native_software_gray':'办公软件灰色PDF水印','native_software_black':'办公软件黑色PDF水印','opaque_block':'不透明覆盖块'}
REPS = [
 ('empty', '未形成选区：照片上的不透明Logo', 'inc11_outdoor_building__logo_opaque.png'),
 ('partial', '漏选与误选并存：文档中的非等距密集文字', 'document_032__tiled_irregular.png'),
 ('reconstruction', '笔画基本选到，复杂纹理仍未恢复', 'textures_016__center_text.png'),
 ('damage', '召回100%仍有内容损伤：菜品照片描边文字', 'inc11_food_dish__outlined.png'),
 ('mode', '补全替换选区后，深色图表文字覆盖减少', 'charts_015__heavy_dark.png'),
 ('negative', '无水印照片被修改，且修改范围超出阈值选区', 'photo_021__negative.png'),
]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def representative_rows(rows):
    result = []
    for tag, title, suffix in REPS:
        found = [r for r in rows if r['mode']=='desktop' and r['id'].endswith(suffix)]
        if len(found) != 1:
            raise ValueError(f'代表样本不唯一：{suffix}')
        result.append(dict(tag=tag, title=title, row=found[0]))
    return result


def prepare(out):
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    gallery = ROOT/'out/完整基线效果总览_20261029_V1'
    record_path = gallery/'逐图效果初筛_20261029.json'
    manifest_path = gallery/'效果总览来源与统计_20261029.json'
    rows = json.loads(record_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest['baseline'])
    baseline = json.loads((base/'运行记录.json').read_text())
    frozen = json.loads((base/'样本清单.json').read_text())
    seal = json.loads((base/'封存校验.json').read_text())
    for name in ('运行记录.json','样本清单.json','原始清单.json','验收规则.json'):
        if digest(base/name) != seal[name]:
            raise ValueError(f'封存证据变化：{name}')
    if digest(base/'运行记录.json') != manifest['source_record_sha256']:
        raise ValueError('效果页与基线版本不一致')
    raw = {r['key']:r for r in baseline['records']}
    for row in rows:
        original = raw[row['key']]
        for name in ('id','page','mode','input_sha256','status','selected_pixels','changed_pixels','recall','precision',
                     'method','repair_mode','watermark_mae','outside_selection_changed_pixels'):
            if row.get(name) != original.get(name):
                raise ValueError(f'派生效果记录与原始证据不符：{row["key"]} {name}')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['mode'],row['input_sha256'],row['page']].append(row)
    unique = [max(rs,key=lambda r:(RANK[r['bucket']],r.get('watermark_mae') or 0)) for rs in grouped.values()]
    by_mode = {mode:{(r['input_sha256'],r['page']):r for r in unique if r['mode']==mode} for mode in MODES}
    stats = {}
    for mode in MODES:
        rs=list(by_mode[mode].values());positive=[r for r in rs if not r['negative']];poor=[r for r in positive if r['bucket']=='poor']
        exclusive = Counter()
        for r in poor:
            category=('unchanged' if r['selected_pixels']==0 or r['changed_pixels']==0 else
                      'not_improved' if r['watermark_error_reduction'] is not None and r['watermark_error_reduction']<=0 else
                      'weak' if r['watermark_error_reduction'] is not None and r['watermark_error_reduction']<.3 and r['watermark_mae']>=5 else 'damage_only')
            exclusive[category]+=1
        stats[mode]=dict(raw_records=sum(r['mode']==mode for r in rows),unique_pages=len(rs),positive_pages=len(positive),
            negative_pages=sum(r['negative'] for r in rs),counts=dict(Counter(r['bucket'] for r in positive)),
            negative_unsafe=sum(r['negative'] and r['bucket']=='poor' for r in rs),exclusive_reasons=dict(exclusive),
            empty_poor=sum(r['selected_pixels']==0 for r in poor),
            poor_damage_risk=sum((r['background_damage_mae'] or 0)>1 or (r['background_damage_p99'] or 0)>12 for r in poor),
            outside_selection_pages=sum(r['outside_selection_changed_pixels']>0 for r in rs),
            outside_selection_positive=sum(r['outside_selection_changed_pixels']>0 and not r['negative'] for r in rs),
            over_120s=sum(r['elapsed_ms']>120000 for r in rs),max_seconds=round(max(r['elapsed_ms'] for r in rs)/1000,3),
            errors=sum(r['status']!='completed' for r in rs),by_style=dict(Counter(r['style'] for r in poor)))
    matrix=Counter();changed=[]
    for key,a in by_mode['desktop'].items():
        b=by_mode['desktop_protected'][key]
        if a['negative']:continue
        matrix[a['bucket'],b['bucket']]+=1
        if a['bucket']!=b['bucket']:
            changed.append(dict(key=a['key'],closed_key=b['key'],open_bucket=a['bucket'],closed_bucket=b['bucket']))
    same=sum(raw[r['key']]['output_sha256']==raw[by_mode['cli'][key]['key']]['output_sha256'] and
             raw[r['key']]['mask_sha256']==raw[by_mode['cli'][key]['key']]['mask_sha256']
             for key,r in by_mode['desktop_protected'].items())
    chosen=representative_rows(unique)
    index={c['id']:c for c in frozen['cases']}
    (out/'代表图').mkdir(parents=True,exist_ok=False)
    font_path='/System/Library/Fonts/PingFang.ttc'
    font=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,18)
    for item in chosen:
        r=item['row'];original=raw[r['key']];case=index[r['id']];ref=case['references'][r['page']-1]
        if r['page']!=1 or not ref.get('clean'):
            raise ValueError('代表图必须有已保存的图片参考，禁止补画未捕获环节')
        source=case['input'];mask=original['files']['选区预览.jpg'];output=original['files']['处理结果.png'];clean=ref['clean']
        panels=[('实际输入',source),('实际阈值选区（红色叠加）',mask),('实际输出',output),('干净参考（仅评分）',clean)]
        columns=2
        if item['tag']=='mode':
            closed=by_mode['desktop_protected'][r['input_sha256'],r['page']];closed_raw=raw[closed['key']]
            panels=[('实际输入',source),('补全关闭：实际选区',closed_raw['files']['选区预览.jpg']),('补全关闭：实际输出',closed_raw['files']['处理结果.png']),
                    ('干净参考（仅评分）',clean),('补全开启：实际选区',mask),('补全开启：实际输出',output)]
            columns=3;item['closed_row']=closed
        width=520 if columns==2 else 430;height=350
        canvas=Image.new('RGB',(width*columns,height*2+62),'white');draw=ImageDraw.Draw(canvas)
        draw.text((18,12),item['title'],fill='#213b34',font=font)
        for i,(caption,path) in enumerate(panels):
            x=(i%columns)*width;y=62+(i//columns)*height
            with Image.open(path) as image:
                image=ImageOps.contain(image.convert('RGB'),(width-26,height-52))
                canvas.paste(image,(x+(width-image.width)//2,y+35+(height-52-image.height)//2))
            draw.text((x+13,y+4),caption,fill='#425b50',font=small)
        filename='代表图/'+item['tag']+'_实际输入选区输出与参考_20261029.jpg'
        canvas.save(out/filename,quality=93)
        item.update(image=str((out/filename).relative_to(ROOT)),source_files=panels,
                    raw_repairs=original.get('repairs'),frozen_source=str(base/'代码快照'),
                    source_image_sha256=digest(source),output_image_sha256=digest(output))
    poor_rows=[r for r in unique if r['bucket']=='poor']
    report=dict(version='完整观察基线问题归纳V1_20261029',prepared_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        baseline=str(base),baseline_seal=digest(base/'封存校验.json'),source_completed_at=baseline['completed_at'],
        source_records_sha256=digest(base/'运行记录.json'),effect_data_sha256=digest(record_path),
        active_record_count=len(rows),unique_page_mode_count=len(unique),stats=stats,
        mode_matrix=[dict(open=a,closed=b,pages=n)for(a,b),n in sorted(matrix.items())],
        mode_changes=changed,cli_and_closed_equal_pages=same,representatives=chosen,
        manual_review_scope='仅6个代表样本的实际图像；未逐张目视验收全部样本',
        inference_rerun=False,algorithm_changed=False,independent_validation=False,
        residual_quality=None,recognition_quality=None,
        deduplication='每模式按输入SHA256＋页码合并；不同批次分档不一致时取较差档，原始全部记录保留',
        exclusions=manifest.get('sample_scope_revision'),
        source_files={name:digest(base/'代码快照'/name) for name in ('wm/masks.py','wm/recall.py','wm/lama.py','tools/control_worker.py')})
    save(out/'本轮问题统计与证据_20261029.json',report)
    save(out/'本轮问题逐图明细_20261029.json',poor_rows)
    save(out/'原始批次记录索引_20261029.json',[dict(key=r['key'],bucket=r['bucket'],files=raw[r['key']]['files'])for r in rows])
    if sum(stats['desktop']['exclusive_reasons'].values())!=stats['desktop']['counts']['poor']:
        raise ValueError('互斥问题分类未闭合')
    for mode in MODES:
        s=stats[mode]
        if s['positive_pages']+s['negative_pages']!=s['unique_pages'] or sum(s['counts'].values())!=s['positive_pages']:
            raise ValueError('去重与正负例分母未闭合')
    print(json.dumps({k:v for k,v in report.items() if k in ('stats','mode_matrix','active_record_count','cli_and_closed_equal_pages')},ensure_ascii=False,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--publish-existing',action='store_true');args=ap.parse_args()
    out=args.out.resolve()
    if args.publish_existing:
        publish(out)
        return
    out.mkdir(parents=True,exist_ok=False)
    for name in ('去水印失败案例记录.html','去水印失败案例记录.md'):
        shutil.copyfile(ROOT/'docs'/name,out/('更新前_'+name))
    prepare(out)




# Publication is deliberately separate from preparation, so representative images
# can be inspected before any narrative is added to the long-lived case record.
def publish(out):
    data=json.loads((out/'本轮问题统计与证据_20261029.json').read_text())
    if data['baseline_seal']!='50bb13d0127a0747b9ae0eb5c0fb7fba749ee30f759498c11ee3a51dc9411bd9' or data['active_record_count']!=1269:
        raise ValueError('本轮叙述只适用于已核对的完整观察基线V3与当前移除范围，不能套用其他版本')
    docs=ROOT/'docs';dest=docs/'去水印失败案例记录.html';markdown=docs/'去水印失败案例记录.md'
    old=dest.read_text();old_md=markdown.read_text()
    if 'id="baseline-review-20261029"' in old:
        raise ValueError('本轮补充已存在，禁止重复追加')
    base=Path(data['baseline']);seal=json.loads((base/'封存校验.json').read_text())
    for name in ('wm/masks.py','wm/recall.py','wm/lama.py','wm/photo_observations.py','wm/paper_repeat.py','tools/control_worker.py'):
        value=digest(base/'代码快照'/name)
        if value!=seal['代码快照/'+name]:raise ValueError('冻结代码证据改变')
        data['source_files'][name]=value
    def url(path):return quote(os.path.relpath(path,docs).replace(os.sep,'/'),safe='/')
    def a(path,label):return f'<a href="{url(path)}">{escape(label)}</a>'
    def pct(v):return '未评估' if v is None else f'{v*100:.2f}%'
    def num(v):return '未评估' if v is None else f'{v:,.2f}'
    def table(headers,rows,label):
        return '<div class="table-scroll" role="region" tabindex="0" aria-label="'+escape(label)+'"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>'for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>'for c in row)+'</tr>'for row in rows)+'</tbody></table></div>'
    s=data['stats']['desktop'];reason=s['exclusive_reasons'];codes=base/'代码快照/wm'
    notes={
    'empty':[
      ('已确认现象','照片右下角的白色图案与编号原样保留，选区和实际改动均为0；没有调用修复。最终定位记录为 nonpaper_background。'),
      ('已确认机制','冻结 masks.py 的该标签对应“类似纸面的像素比例低于25%”后返回空选区；专用路径未给出可用结果后，照片没有可继续使用的通用定位。'),
      ('仍待验证','未保存本样本各专用分支的拒绝明细，不能确定Logo候选在哪一道检查被拒绝，也不能据此断言模型没有处理图案的能力。'),
      ('下一步验证','先区分图案、文字及背景结构，补充经像素证据验证的Logo候选；保留普通图标、照片标识负例。与旧案例011的空选区问题关联，不能用全局取消纸面保护解决。')],
    'partial':[
      ('已确认现象','选区只覆盖顶部一小部分重复文字，输出的大部分斜字仍在；全部笔画召回4.20%、精确率44.05%，水印区对参考误差36.27→36.58，未改善。'),
      ('已确认机制','记录进入 paper_repeat_document_overlay / document_overlay_full，只使用透明度反解。冻结 recall.py 对这种物理反解路径保留原模型，不做通常的笔画补全，因此不能期待补全开关自动补齐这一页。'),
      ('仍待验证','覆盖不足已由标注与图像确认；为什么只找到顶部实例，需要进一步记录模板匹配、配准、正文交叠及保护回退。不能把全部残字都归为正文保护，旧案例009的细分原因也不能直接套用。'),
      ('下一步验证','在当前图内核对重复实例、边缘和正文交叠的证据，分开统计未选中与已选中未清理；保护正文、图表和已有透明度模型。关联旧案例009。')],
    'reconstruction':[
      ('已确认现象','大部分字形已选中，实际输出的文字区域变成了较平滑的纹理，仍与干净噪声参考不同。召回95.70%、精确率87.34%，水印区误差40.36→33.90，约减少16%。'),
      ('已确认机制','最终路径是 recall_pixel_strokes / inpaint，修复记录为 LaMa。这里同时存在少量漏选、背景误选及重建差异；“选中”与“恢复原纹理”是两个问题。'),
      ('仍待验证','像素误差不能区分残余字形和重新生成纹理的差异；随机细纹理未必能从遮挡图唯一恢复。没有可靠的残留率或识别质量结果，不能把33.90写成水印残留率。'),
      ('下一步验证','分别检查残余轮廓与局部纹理保真，再判断掩码或修复方法各贡献多少；现有周期纹理成功案例004不能证明随机纹理已解决。')],
    'damage':[
      ('已确认现象','红色选区呈整条矩形，包含大量真实菜品；输出把该处菜品改成平滑色带。召回100%，精确率仅14.22%，非水印区新增平均误差1.91、99分位52.67。'),
      ('已确认的实现风险','该页记录为 photo_confirmed_colour_line / inpaint，使用LaMa。冻结 photo_observations.local_lines 确实存在：颜色检查通过后，将小倾角OCR多边形填满并扩边，再交给 _layer 作为修复范围的分支。搜索框成为整片修复权限是需要优先检查的实现风险。'),
      ('仍待验证','本轮未保存本页OCR倾角和每个候选的分支日志，矩形来源尚未逐项绑定；不能把全部损伤只归给LaMa模型，也不能据此声称所有该标签的样本都走了填框分支。'),
      ('下一步验证','先核对矩形选区的来源与实际笔画像素，沿字形验证修复权限，严格保留字洞和字间背景；把菜品纹理、水平合法文字及阴影作为保护样本。')],
    'mode':[
      ('已确认现象','同一图表，补全关闭的笔画召回99.91%，开启后降至42.44%；关闭时水印大部分去掉，开启后仍残留较多字形。关闭结果的柱形和数值也有损伤，仍属于待复核。'),
      ('已确认机制','两种桌面模式的保存配置只在笔画补全开关上不同。冻结 recall.expand 最终用新 region 替换原 alpha 和 footprint，并将修复模式改为 inpaint；最终选区缩小后，未获修复权限的字形会保留。'),
      ('仍待验证','这对结果支持补全存在双向影响；本轮未记录候选通道分数、OCR范围等内部观测，不能断定本页就是旧案例008、010的颜色优先规则造成。'),
      ('下一步验证','对原选区与补全选区逐实例保留证据，区分有效旧笔画、误选和新发现笔画；同时回归18页分档改善与11页分档变差样本，不能简单全局关闭补全。')],
    'negative':[
      ('已确认现象','这张建筑照片没有水印，程序仍选中61,538像素、修改69,186像素。依据0.08阈值选区计数，有8,145个实际改动像素落在选区外；外扩配置为0。云层和建筑边界被误选，不能记作成功去水印。'),
      ('已确认的实现风险','保存路径为 tonal / hybrid，执行透明度反解和LaMa。控制记录按 mask.binary(0.08) 定义选区；冻结 hybrid 修复先对连续alpha反解，再可能按 footprint 修复，两者没有统一裁回同一个阈值选区。该代码差异解释了为什么需分开核对阈值选区、修复权限和实际修改。'),
      ('仍待验证','该页保存了连续alpha和最终输出，但没有分阶段结果及当时footprint，无法把8,145个像素逐一归到反解或LaMa，也没有精确量化云层损伤。它是观察基线已有问题，不是本次报告更新引入的新退化。'),
      ('下一步验证','优先核对背景误判和0外扩边界；同时保存并比较阈值选区、实际修复权限与修改像素。无水印对照继续保留，不能通过隐藏或删样本放行。')]
    }
    source_links=[(out/'本轮问题统计与证据_20261029.json','本轮统计、版本与代表样本依据'),
        (out/'本轮问题逐图明细_20261029.json','三模式全部问题记录（去重后698条）'),
        (out/'原始批次记录索引_20261029.json','保留范围全部1269条批次索引'),
        (ROOT/'out/完整基线效果总览_20261029_V1/去水印效果分档总览_20261029.html','三档效果页（可切换模式与问题档）'),
        (base/'运行记录.json','原始封存运行记录'),(ROOT/'config/regression_retirements.json','三张手写样本移除记录')]
    summary_rows=[[MODES[m],v['positive_pages'],v['counts'].get('good',0),v['counts'].get('review',0),v['counts'].get('poor',0),f"{v['negative_unsafe']} / {v['negative_pages']}",v['outside_selection_pages']]for m,v in data['stats'].items()]
    direct=[['最终选区为空，输出未变',reason.get('unchanged',0),'定位未提供修复权限；空选区不能评价修复模型。'],
      ['有改动，水印区误差未改善',reason.get('not_improved',0),'漏选、误选及重建差异需要分开核对。'],
      ['有改善，但改善不足30%且误差仍≥5',reason.get('weak',0),'包括遗漏与纹理差异，不等同于残留率。'],
      ['改善通过上述条件，仍因背景新增误差判问题档',reason.get('damage_only',0),'处理水印的同时，正常内容有受损风险。']]
    intro=f'''<article id="baseline-review-20261029" aria-labelledby="baseline-review-title"><header class="case-heading"><div><span class="eyebrow">2026-10-29 · 完整观察基线补充 V1</span><h2 id="baseline-review-title">先补齐定位，再分别检查修复保真与模式差异</h2></div><span class="badge pending">仅分析保存结果 · 未实施修复</span></header>
<h3>本轮结论</h3><p>桌面默认的381页带水印样本中，227页被自动初筛为“问题较多”，其中139页最终选区为空；其余88页发生了改动，但仍有误差改善不足、重建差异或正常内容受损风险。应分别处理定位与修复，不能统一归因于LaMa。[1]</p><p>补全开启相较关闭，18页从问题档进入待复核，11页从待复核进入问题档；同一规则同时存在改善与变差。另有1页无水印对照被修改，桌面默认共72页出现阈值选区以外的改动，实际修复权限与显示选区需要单独核对。[1][2]</p>
<p class="footnote">证据来自2026-10-28 21:36完成的完整观察基线V3，整理日期为2026-10-29。采用移除3张手写样本后的395个登记输入／423页／三模式1269页次；每模式按输入哈希和页码合并重复批次后为403页（381页带水印＋22页无水印）。403页不代表403个独立底图。原始1278条封存不改写，已移除样本不再纳入本节。</p>
{table(['模式','带水印页','较好初筛','待复核','问题较多','误改负例 / 全部负例','阈值选区外有改动页'],summary_rows,'三模式按相同口径比较')}
<h3>227页的直接表现，按顺序互斥归类</h3>{table(['直接表现','页数','可据此判断什么'],direct,'桌面默认问题样本的互斥分解')}
<p>139＋35＋49＋4＝227。分类是结果表现，不是已查明根因的数量。另按背景新增平均误差&gt;1或99分位&gt;12统计，其中13页触发内容损伤风险；这13页与上表重叠，不能相加。[1]</p>
<h3>补全有双向影响，净减少7页问题档不足以代表安全</h3>{table(['补全关闭 → 开启','带水印页数'],[['问题较多 → 待复核',18],['待复核 → 问题较多',11],['问题档保持不变',216],['待复核保持不变',126],['较好初筛保持不变',10]],'同一381页的分档迁移')}
<p>关闭补全与CLI在本基线403个去重页面上的实际输出和选区哈希全部一致，所以不重复放两套图片；这一结论只对应本次版本、参数及样本，不能推及所有批次或跨页先验场景。所有“待复核”均不等于质量合格。</p>
<details class="evidence"><summary>查看各样式问题数量与本轮记录边界</summary>{table(['水印样式','桌面默认问题页','关闭补全问题页','CLI问题页'],[[STYLE.get(style,style),data['stats']['desktop']['by_style'].get(style,0),data['stats']['desktop_protected']['by_style'].get(style,0),data['stats']['cli']['by_style'].get(style,0)]for style,_ in sorted(s['by_style'].items(),key=lambda kv:(-kv[1],kv[0]))],'按样式查看问题分布')}
<p>本节用保存的输入、选区、输出与评分数据整理；只目视检查下列6个代表样本，未逐张验收全部381页。所有三种模式的423条保留批次记录均无运行异常，也没有超过120秒的页次；本次未重跑推理，不与历史诊断时的冷缓存、分块跟踪耗时直接比较。</p><p>全部笔画召回按真实alpha&gt;0计算；精确率空分母保留未评估。像素误差包括重建和JPEG差异，不是残留率。实际水印残留率、正文识别质量及独立泛化能力仍未可靠评估。13页内容损伤风险、72页范围差异是指标观察，需结合实际图像和权限记录判断。</p></details>
<p><strong>阅读顺序：</strong>先看下列6组代表证据，再从逐图明细追溯其他样本。原有001—011的历史指标与状态完整保留。</p></article>'''
    blocks=[intro]
    md=['\n\n<!-- baseline-review-20261029:start -->\n## 2026-10-29 · 完整观察基线补充 V1\n\n状态：仅分析已有结果，未实施修复，未重新运行推理。\n\n证据来自2026-10-28 21:36完成的完整观察基线V3。移除3张手写样本后为395个登记输入／423页／三模式1269页次；按输入哈希及页码去重后每模式403页，其中381页带水印、22页无水印。不是独立底图数量。\n\n桌面默认227页处于问题档，其中139页最终选区为空。另有1页无水印对照被改动，72页存在阈值选区外改动。只目视检查下列6个代表样本，未逐张验收全部样本。像素误差不等于残留率，没有新的独立泛化验证。\n\n| 模式 | 带水印页 | 较好初筛 | 待复核 | 问题较多 | 误改负例 | 阈值选区外改动页 |\n|---|---:|---:|---:|---:|---:|---:|\n']
    for row in summary_rows:md.append('| '+' | '.join(str(x)for x in row)+' |\n')
    md.append('\n直接表现按顺序互斥归类：139页为空选区未改动，35页有改动但水印区误差未改善，49页改善不足30%且误差仍≥5，4页仅因背景新增误差进入问题档，合计227页。这不是根因已查明的数量。另有13页触发背景损伤风险，与上述分类重叠。\n\n补全关闭→开启：18页问题档变为待复核，11页待复核变为问题档；216页维持问题档、126页维持待复核、10页维持较好初筛。关闭补全与CLI的403个去重页面输出和选区哈希一致，仅对应本版本与本样本。\n')
    visual_findings={}
    for i,item in enumerate(data['representatives'],1):
        r=item['row'];tag=item['tag'];anchor=f'baseline-evidence-{tag}-20261029'
        note=notes[tag];visual_findings[tag]=note[0][1]
        metrics=[[pct(r['recall']),pct(r['precision']),f"{num(r['source_watermark_mae'])} → {num(r['watermark_mae'])}",num(r['background_damage_mae']),r['selected_pixels'],r['changed_pixels'],r['outside_selection_changed_pixels']]]
        if tag=='mode':
            c=item['closed_row'];metrics.insert(0,[pct(c['recall']),pct(c['precision']),f"{num(c['source_watermark_mae'])} → {num(c['watermark_mae'])}",num(c['background_damage_mae']),c['selected_pixels'],c['changed_pixels'],c['outside_selection_changed_pixels']])
        image=ROOT/item['image'];img_url=url(image)
        files=' · '.join(a(Path(path),caption)for caption,path in item['source_files'])
        code_names={'empty':['masks.py'],'partial':['paper_repeat.py','recall.py'],'reconstruction':['recall.py','lama.py'],
                    'damage':['photo_observations.py','lama.py'],'mode':['recall.py'],'negative':['masks.py','lama.py']}[tag]
        code_links=' · '.join(a(codes/name,'冻结代码 '+name)for name in code_names)
        block=f'''<article id="{anchor}" aria-labelledby="{anchor}-title"><header class="case-heading"><div><span class="eyebrow">本轮代表证据 {i:02d} · 基线现状</span><h2 id="{anchor}-title">{escape(item['title'])}</h2></div><span class="badge pending">尚未修复 · 待进一步验证</span></header><figure><a class="image-link" href="{img_url}" target="_blank" rel="noopener"><img src="{img_url}" loading="lazy" decoding="async" alt="{escape(item['title'])}；全部面板来自实际输入、保存选区、实际输出和评分参考"></a><figcaption>实际记录的事后拼图；红色是已保存选区，干净参考仅供评分<span>点击查看原尺寸 ↗</span></figcaption></figure>
<ul class="facts">{''.join('<li><strong>'+escape(title)+'：</strong>'+escape(text)+'</li>'for title,text in note)}</ul>
{table(['全部召回','精确率','水印区误差：输入→输出','非水印区新增平均误差','选中像素','实际修改像素','阈值选区外修改'],metrics,'代表样本的已保存指标')}
{'<p>两行依次为补全关闭、补全开启；关闭时仍有图表损伤，不标为修复成功。</p>'if tag=='mode'else ''}
<details class="evidence"><summary>查看原尺寸文件、冻结实现和样本标识</summary><p>{files}</p><p>{code_links}</p><p>样本：<code>{escape(r['key'])}</code>；最终定位 <code>{escape(r['method'])}</code>，修复 <code>{escape(r['repair_mode'])}</code>；实际修复记录 <code>{escape(', '.join(item['raw_repairs']))or '未调用'}</code>。</p></details></article>'''
        blocks.append(block)
        md.extend([f'\n### 本轮代表证据 {i:02d} · {item["title"]}\n\n',f'![{item["title"]}]({os.path.relpath(image,docs)})\n\n',
                   '\n'.join(f'- **{title}：** {text}'for title,text in note),f'\n\n样本：`{r["key"]}`；全部召回{pct(r["recall"])}，精确率{pct(r["precision"])}，水印区误差{num(r["source_watermark_mae"])}→{num(r["watermark_mae"])}。\n'])
    closing=f'''<article id="baseline-next-checks-20261029"><header class="case-heading"><div><span class="eyebrow">后续验证顺序 · 本轮未实施</span><h2>优先守住正常内容与修复边界，再补定位和重建能力</h2></div></header><ol><li>先处理无水印误选、OCR范围成为整片修复权限、阈值选区与实际修改范围不一致的问题；验证外扩0及合法文字、Logo、照片边缘保护。</li><li>对139页空选区按Logo、角标、淡字、白字、点阵等适用证据分组，核对各路径拒绝原因，补充可靠候选；不能全局降阈值或统一大区域补图。</li><li>对补全引起分档变化的29页同时保留改善与变差样本，记录原笔画、候选通道、OCR搜索范围和最终权限；不把个别案例的规则扩大到全部样式。</li><li>定位与范围可靠后，再单独比较透明度反解、周期恢复和LaMa的残留与背景保真；保留模式隔离、逐张门槛和独立来源验证要求。</li></ol><p>上面是后续验证建议，未变更算法、参数、放行门槛或观察基线。没有运行新候选，不能得出跨样式无退化或独立泛化结论。旧案例的已修复状态仍仅代表其注明日期的复测。</p><h3>来源与可追溯记录</h3><p>[1] {a(out/'本轮问题统计与证据_20261029.json','本轮统计与代表证据')}；[2] {a(base/'代码快照/tools/control_worker.py','冻结记录器')}及上方冻结实现链接。原始批次与去重明细如下。</p><ul>{''.join('<li>'+a(path,label)+'</li>'for path,label in source_links)}</ul></article>'''
    blocks.append(closing)
    md.append('\n### 后续验证顺序与来源\n\n1. 先核对无水印误选、OCR搜索框成为修复权限，以及0外扩的实际修改范围。\n2. 对139页空选区按样式检查定位拒绝原因，不全局降阈值。\n3. 同时保留补全改善18页与变差11页，记录旧笔画、新候选和最终权限。\n4. 在定位与范围可靠后，分别评价反解、周期恢复和LaMa的残留与背景保真。\n\n本轮未实施候选，不得宣称跨样式无退化或通用能力提升。原001—011的历史结论保留。\n\n'+' · '.join(f'[{label}]({os.path.relpath(path,docs)})'for path,label in source_links)+'\n<!-- baseline-review-20261029:end -->\n')
    section='<!-- baseline-review-20261029:start -->'+''.join(blocks)+'<!-- baseline-review-20261029:end -->'
    first=old.index('<article')
    page=old[:first]+section+old[first:]
    page=page.replace('2026-10-28 更新 · V3 · 新增4例、8张原始截图与实际队列复现 · 共11个案例','2026-10-29 更新 · V4 · 新增完整基线问题归纳与6组代表证据 · 保留原有11个案例')
    lead=re.search(r'<p class="lead">.*?</p>',page,re.S)
    new_lead='<p class="lead"><strong>本轮从完整观察基线归纳定位不足、修复差异、内容损伤及模式双向影响。</strong>新增6组实际图像证据，区分已确认现象、实现风险和待验证原因；没有修改算法或重新推理。原有11个案例保留其日期与历史状态，旧复测成绩不当作当前通用能力。</p>'
    page=page[:lead.start()]+new_lead+page[lead.end():]
    nav='<a class="overview-link" href="#baseline-review-20261029">本轮完整基线问题总览</a>'+''.join(f'<a class="case-link" href="#baseline-evidence-{item["tag"]}-20261029"><span class="nav-number">本轮{i:02d}</span><span>{escape(item["title"])}<small>已看代表图 · 原因分级</small></span></a>'for i,item in enumerate(data['representatives'],1))
    page=page.replace('<a class="overview-link" href="#top">记录概览 ↑</a>','<a class="overview-link" href="#top">记录概览 ↑</a>'+nav)
    # The previously documented 11 cases remain byte-for-byte unchanged.
    for i in range(1,12):
        pattern=rf'<article id="case-{i:03d}"[^>]*>.*?</article>'
        original=re.search(pattern,old,re.S);new=re.search(pattern,page,re.S)
        if not original or not new or original.group()!=new.group():raise ValueError(f'历史案例被改变：{i}')
    page=page.replace('<span>11 个案例</span>','<span>历史11个案例</span><span>本轮6组代表证据</span>')
    dest.write_text(page)
    intro_md='\n\n最新补充：2026-10-29 完整观察基线问题归纳，保留原11个案例与历史结论。参见文末“完整观察基线补充 V1”。\n'
    markdown.write_text(old_md.replace('持续追加｜2026-10-27 首次整理','持续追加｜2026-10-27 首次整理'+intro_md,1)+''.join(md))
    data.update(representatives_visually_reviewed=True,representative_visual_findings=visual_findings,
                html_rendering_verified=False,html_rendering_note='沿用原HTML版式；文件URL浏览器检查此前被安全策略阻止，未绕过限制。本轮做结构、链接、数据与实际代表图检查。')
    save(out/'本轮问题统计与证据_20261029.json',data)
    print(dest)




if __name__=='__main__':main()
