"""Refresh current project documentation from real manifests without running inference.

Usage: .venv/bin/python tools/organize_project_docs.py [--check] [--out NEW_DIR]
Historical reports, frozen results, inference sources and dataset splits stay in place.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT / 'docs'
VERSION = '项目整理V1_20261029'
CURRENT_DOCS = {'项目现状与后续开发说明.html', '项目文档总览.html',
                '数据集清单与清理.html', '最小回归测试集.html', '代码与工具用途说明.html'}
CSS = '''*{box-sizing:border-box}body{margin:0;background:#f4f6f4;color:#243c32;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:36px 24px 70px}h1{font-size:30px;line-height:1.4}h2{font-size:22px;margin-top:0}section{background:white;border:1px solid #dde5df;border-radius:12px;padding:24px;margin:20px 0}a{color:#176a4d;text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:10px 20px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #dfe5df;overflow-wrap:anywhere}th{background:#f0f4f0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3ee;padding:16px;border-radius:8px}code{font-size:13px;overflow-wrap:anywhere}.note{background:#eaf4ed;border-left:4px solid #287653;padding:16px}.muted{color:#617267;font-size:13px}.table-wrap{overflow:auto}summary{cursor:pointer;color:#176a4d;padding:8px}li{margin:8px 0}@media(max-width:650px){main{padding:20px 12px}section{padding:16px}h1{font-size:25px}td,th{padding:8px}}'''


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(path, label):
    return f'<a href="{escape(str(path), quote=True)}">{escape(str(label))}</a>'


def table(headers, records):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in record) + '</tr>' for record in records) + '</tbody></table></div>'


def page(title, sections, date):
    nav = ' · '.join([link('../README.md', '项目首页'), link('项目现状与后续开发说明.html', '当前状态'), link('项目文档总览.html', '文档总览'), link('../out/水印验证报告总目录.html', '报告目录')])
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head><body><main data-project-overview="{VERSION}"><nav>{nav}</nav><h1>{escape(title)}</h1><p class="muted">更新日期 {date} · 当前说明由实际清单生成 · 历史成绩保留原日期与条件</p>' + ''.join(f'<section id="{sid}"><h2>{heading}</h2>{body}</section>' for sid, heading, body in sections) + '</main></body></html>\n'


def facts():
    from tools.control_inventory import build_inventory
    from tools.regression_retirements import load_profile, project_baseline
    registry = load(ROOT / 'data/dataset_registry.json')
    v1 = rows(ROOT / 'data/watermark_training_v1/manifest.jsonl')
    strict = rows(ROOT / 'data/watermark_training_increment_v1_1/manifest.jsonl')
    tasks = rows(ROOT / 'data/watermark_training_increment_v1_1/training_tasks.jsonl')
    fixed = rows(ROOT / 'data/watermark_regression_min_v1/manifest.jsonl')
    coverage = load(ROOT / 'data/watermark_regression_min_v1/coverage.json')
    pointer = load(ROOT / '.regression-control/baseline.json')
    baseline = Path(pointer['path'])
    if sha(baseline / '封存校验.json') != pointer['seal']:
        raise ValueError('观察基线封存标识不一致，停止更新说明')
    original_run = load(baseline / '运行记录.json')
    original_inventory = load(baseline / '原始清单.json')
    run, active = project_baseline(original_run, original_inventory, load_profile(), pointer['seal'])
    inventory_error = None
    unregistered = []
    try:
        current = build_inventory(ROOT, ROOT / 'data/应用测试输入')
    except ValueError as exc:
        # Documentation can record an incomplete inventory; release checking still
        # uses the original strict tool and remains blocked by unregistered files.
        prefix = '存在未登记的输入或参考，必须先分类：'
        if not str(exc).startswith(prefix):raise
        inventory_error = str(exc)
        for name in ast.literal_eval(str(exc)[len(prefix):]):
            p = ROOT / 'data/应用测试输入' / name
            digest = sha(p)
            duplicates = [r['id'] for r in active['cases'] if r['input_sha256'] == digest]
            unregistered.append(dict(path=str(p.relative_to(ROOT)), sha256=digest,
                                     matches_registered_ids=duplicates))
        current = active  # Registered retained scope only, explicitly labelled below.
    if {(r['id'], r['input_sha256'], r['page_count'], r['batch']) for r in current['cases']} != {(r['id'], r['input_sha256'], r['page_count'], r['batch']) for r in active['cases']}:
        raise ValueError('当前应用输入与保留基线范围不一致，须单独核对，不自动改写样本数量')
    counts = dict(base=len(v1), strict=len(strict), tasks=len(tasks),
                  approximate=sum(r['category'] == 'approximate_pair' for r in tasks),
                  restoration=sum('restoration' in r['tasks'] for r in tasks),
                  detection=sum('detection' in r['tasks'] for r in tasks),
                  detection_only=sum(r['tasks'] == ['detection'] for r in tasks),
                  fixed_inputs=len(fixed), fixed_pages=sum(r['page_count'] for r in fixed),
                  core=sum(r['suite'] == 'core' for r in fixed), real=sum(r['suite'] == 'real' for r in fixed),
                  control_inputs=current['total_inputs'], control_pages=current['total_pages'],
                  mode_pages=len(run['expected_keys']), historical_mode_pages=len(original_run['expected_keys']))
    counts['approximate'] = len(tasks) - len(strict) - counts['detection_only']
    current_sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'wm').rglob('*.py')}
    changed = sorted(name for name in set(current_sources) | {n for n in original_run['code_hashes'] if n.startswith('wm/')} if current_sources.get(name) != original_run['code_hashes'].get(name))
    return dict(counts=counts, registry=registry, fixed=fixed, coverage=coverage,
                baseline=baseline, baseline_seal=pointer['seal'], original_run=original_run,
                current=current, changed_application_files=changed,
                inventory_error=inventory_error, unregistered=unregistered,
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'data/watermark_training_v1/manifest.jsonl', ROOT / 'data/watermark_training_increment_v1_1/manifest.jsonl', ROOT / 'data/watermark_training_increment_v1_1/training_tasks.jsonl', ROOT / 'data/watermark_regression_min_v1/manifest.jsonl']})


def tool_catalog():
    daily = {'daily_check.py', 'regression_control.py', 'regression_guard.py', 'regression_layers.py', 'regression_retirements.py', 'control_inventory.py', 'control_worker.py', 'run_min_regression.py', 'verify_control_chain.py'}
    docs = {'organize_project_docs.py', 'build_report_index.py', 'build_effect_overview.py', 'build_single_flow_demo.py', 'build_watermark_review.py', 'sync_real_training_docs.py', 'update_failure_cases_from_baseline.py'}
    catalog = []
    for p in sorted((ROOT / 'tools').iterdir()):
        if not p.is_file() or p.suffix not in {'.py', '.mjs', '.js', '.css', '.txt'}:
            continue
        source = p.read_text(errors='replace')
        guarded = p.name != Path(__file__).name and any(marker in source for marker in ('Historical data was retired', '该脚本依赖的历史样本已退役', '历史初始化脚本已停用', '历史整理脚本已停用'))
        if guarded:
            group, use = '历史入口已停用', '直接运行会提示退役；保留历史追溯或导入复用，日常使用 start.sh test。'
        elif p.name in daily:
            group, use = '日常检查与管控', '优先通过 start.sh 调用；内部模块以源码职责为准。'
        elif p.name in docs:
            group, use = '当前文档与报告生成', '按对应来源重建说明或展示，不产生新的图像质量成绩。'
        elif 'tender' in p.name:
            group, use = '方案材料', '生成 Word 或演示材料；部分产物和脚本尚未进入 Git。'
        elif 'kimi' in p.name:
            group, use = '可选外部修图辅助', '单独配置与运行，涉及外部服务；不属于默认离线应用链路。'
        elif p.name.startswith(('probe_', 'benchmark_', 'debug_', 'blocking_', 'jpeg_', 'ocr_')):
            group, use = '实验与诊断', '开发时按需使用；不能把候选结果当作默认版本或发布验收。'
        elif any(s in p.name for s in ('training', 'increment', 'min_regression', 'cleanup_datasets')):
            group, use = '数据准备与维护', '可能生成数据、更新清单或导出文件；不是日常效果测试入口。'
        elif p.suffix in {'.js', '.css'}:
            group, use = '报告展示资源', '由报告页面引用。'
        else:
            group, use = '专项复测与历史报告工具', '依赖各自输入和已有结果，使用前核对源码与参数；不替代日常或完整验收。'
        catalog.append((p.name, group, use))
    return catalog


def render(f, date):
    c = f['counts']
    inventory_note = ('<p class="note"><strong>完整清单检查尚未通过。</strong>发现未登记文件，已登记范围仍为下表数量；未登记文件保留原位，不自动纳入或移除。</p><ul>' + ''.join('<li><code>'+escape(r['path'])+'</code>；'+('内容与已登记输入完全相同。' if r['matches_registered_ids'] else '尚无相同已登记输入。')+'</li>' for r in f['unregistered']) + '</ul>') if f['inventory_error'] else '<p>当前应用输入清单已核对；清单一致不表示图像质量合格。</p>'
    overview = '../out/完整基线效果总览_20261029_V1/去水印效果分档总览_20261029.html'
    recent = '../out/检测漏接单项实验_20261029_112047/检测漏接单项实验_新旧结果对比_20261029.html'
    count_table = table(['数据', '当前数量', '用途与限制'], [
        ('基础数据', f'{c["base"]:,} 条图片／PDF 文档记录', '训练准备与开发；有训练、验证、测试分组，不能与按页统计直接相加。'),
        ('人工增量', f'{c["tasks"]} 个不重复任务样本', f'{c["strict"]} 条严格配对＋{c["approximate"]} 组近似配对＋{c["detection_only"]} 张检测专用；修复视图 {c["restoration"]}、检测视图 {c["detection"]}，任务重叠。'),
        ('固定回归', f'{c["fixed_inputs"]} 个输入／{c["fixed_pages"]} 张页', f'核心 {c["core"]}＋真实扩展 {c["real"]}；来自已有数据，供防退化检查，不增加独立来源。'),
        ('完整管控范围', f'{c["control_inputs"]} 个输入场景／{c["control_pages"]} 张页', f'含不同批次的重复文件；三种模式共 {c["mode_pages"]:,} 页次，不是这么多独立图片。')])
    status = page('项目现状与后续开发说明', [
        ('current', '现在做到哪里', '<p class="note"><strong>软件主干和测试管控已建立，去水印效果仍不稳定。</strong>可以继续开发和逐张复核，目前没有通用质量合格或独立泛化验证结论。</p><p>已有离线桌面界面、图片与 PDF 导入、多批次处理、原图结果对比、手动画笔修改选区及结果导出。PDF 会先检查可处理的独立对象，其他页面走图片定位与修复。具体边界见 '+link('去水印处理流程图.md', '实际处理流程')+'。</p>'),
        ('default', '默认功能和实验功能', table(['范围', '现在的状态'], [
            ('桌面默认', '开启补全水印笔画；外扩由用户参数控制。CLI 默认关闭补全，可显式开启。'),
            ('手动画笔', '用户确认完整选区后重新处理并保存来源记录；人工效果与自动效果分开统计。'),
            ('默认关闭的实验', '自动比较原方法与笔画补全、彩色笔画细化、旧笔画保留、斜字方向补试；均未获得默认放行资格。自动比较保留两种实际结果，仍需核对残留与内容损伤。SLBR 等研究模型未接入默认应用。')])+ '<p>操作细节与此前实验记录保留在 '+link('历史开发记录_20261029.md', '整理前开发详细记录')+'；实验结论见 '+link('../out/水印验证报告总目录.html', '报告目录')+'。</p>'),
        ('data', '数据和测试范围', inventory_note+count_table),
        ('evidence', '哪些结果可以作为依据', '<p>本机观察基线是 <code>'+escape(f['baseline'].name)+'</code>，于 2026-10-28 完成三模式记录并封存，原始 '+f'{c["historical_mode_pages"]:,} 页次。按用户要求移除三张手写样本后，当前保留 '+f'{c["mode_pages"]:,} 页次；历史原件未改写。</p><p>当前应用源码与该快照仍有差异，因此<strong>旧基线不能证明当前工作版本全量通过</strong>。'+link(overview, '查看基线效果初筛')+'，'+link('去水印失败案例记录.html', '查看已知失败')+'。分档是自动初筛，未逐张人工验收；残留和识别正确率不能可靠测量时保持未评估。</p><p>10 月 29 日斜字方向实验完成 16 页次开发对照：目标仍未检测成功，新旧选区和输出相同，干净界面耗时增加超限，候选继续默认关闭。'+link(recent, '查看这轮实验原始报告')+'。工程检查通过不能代替图像质量验收。</p><details><summary>当前应用与观察基线不同的文件</summary><pre>'+escape('\n'.join(f['changed_application_files']) or '无')+'</pre></details>'),
        ('limits', '还没有解决什么', '<ul><li>部分照片和淡字仍漏检，选区可能混入背景或漏掉笔画。</li><li>正文、数字和纹理可能被修复损伤；部分 PDF 复杂结构仍需回退处理。</li><li>真实非配对图片缺精确答案，不能自动给出成功率。</li><li>已反复查看和调参的样本属于开发／回归用途；独立泛化验证尚未成立。</li></ul><p>这些问题继续登记在失败记录中，算法修复另立任务，先保存基线再比较。</p>'),
        ('work', '接下来如何使用', '<p>日常先运行 <code>./start.sh test --plan</code>，再运行 <code>./start.sh test</code>。发布或更新默认基线必须另跑 <code>./start.sh test-release</code>，完成必要的独立验证。数据登记、文档整理不会自动更新默认基线。</p><p>'+link('代码与工具用途说明.html', '代码与工具用途')+' · '+link('项目文档总览.html', '全部说明入口')+' · '+link('项目现状与后续开发说明_历史20261027.html', '10 月 27 日现状说明原件')+'</p>')], date)
    catalog = tool_catalog()
    groups = Counter(row[1] for row in catalog)
    directory_rows = [
        ('wm/', '软件本身', 'app.py 是桌面入口；batches.py 负责队列；masks.py 定位；repair.py / lama.py 修复；pipeline.py 串联与输出。'),
        ('tests/', '程序行为检查', '覆盖选区边界、PDF、修复路由、导出、界面和回归管控；不能代替实际图片验收。'),
        ('tools/', '测试、维护、研究和报告脚本', '下表按用途列出；没有删除、移动或擅自接入默认应用。'),
        ('config/', '固定门槛与范围规则', 'regression_control.json、regression_tiers.json、regression_retirements.json；整理说明不改变规则。'),
        ('data/', '输入、原始来源与评分参考', '图片输入与干净参考分开；参考只能在推理后评分时使用。'),
        ('out/', '运行产物与历史证据', '每轮新目录，封存结果原位保留。'),
        ('proposal/', '方案材料与可维护源文件', '有多轮 Word / PPT 产物，目前未纳入 Git；通过报告总目录进入。'),
        ('.regression-control/', '本机观察基线指针', '本机状态，不进 Git；跨机不能直接照搬绝对路径。')]
    guide = page('代码与工具用途说明', [
        ('map', '目录分别做什么', table(['目录', '用途', '说明'], [(escape(a), escape(b), escape(d)) for a,b,d in directory_rows])),
        ('entry', '平时使用哪些入口', '<pre>./start.sh setup                 # 首次准备本地环境\n./start.sh gui                   # 打开桌面程序\n./start.sh check-data            # 校验固定数据，不运行模型\n./start.sh test --plan           # 预览实际修改需要的检查\n./start.sh test                  # 日常按影响检查\n./start.sh test-release          # 完整三模式对照\n.venv/bin/python tools/organize_project_docs.py  # 更新当前说明\n.venv/bin/python tools/organize_project_docs.py --check  # 只核对说明和链接\n.venv/bin/python tools/build_report_index.py    # 更新报告目录</pre><p>已有本机观察基线，日常不要重新执行 freeze-baseline 替换比较依据。需要完整发布证据时走原放行流程。</p>'),
        ('tools', '每个工具的用途', '<p>下面覆盖 tools/ 顶层脚本与展示资源。分类用于导航，专项脚本是否可运行仍取决于其输入、参数和本地资源；没有逐一重跑研究脚本。</p>'+table(['类别', '文件数'],[(escape(k),str(v)) for k,v in sorted(groups.items())])+table(['文件', '类别', '使用说明'],[(link('../tools/'+a,a),escape(b),escape(d)) for a,b,d in catalog])),
        ('backup', 'Git 和本机备份', '<p>程序、测试、维护脚本、README 和已跟踪的文档通过 Git 保存。本轮新增的当前说明允许进入 Git，但本轮不自动提交或推送。</p><p><code>data/</code>、大部分 <code>out/</code>、模型缓存及本机基线指针采用单独资源包或本机备份；部分报告虽在忽略目录内，仍已被 Git 跟踪，不能只看 .gitignore 判断。<code>proposal/</code>、可选修图配置和根目录临时图表文件夹尚未进入 Git。</p><p>换电脑时要另外保存：当前数据与来源、模型、完整封存基线及其资源、尚未入库的方案材料。可选修图配置可能包含密钥，保留在本机，不直接加入仓库。Git 精确状态见本轮整理记录。</p>')], date)
    data = page('数据集清单与清理', [
        ('sets', '当前只有三套工作数据', inventory_note+count_table+'<p>数据名称中的“训练”表示准备用途，不表示已完成模型训练。固定集和应用批次存在来源重叠，不能把各项直接相加为独立样本数。</p>'),
        ('paths', '输入和答案在哪里', table(['位置', '用途'], [(link('../data/dataset_registry.json','数据登记表'), '当前数量与用途，由实际清单更新。'), (link('../data/watermark_training_v1/manifest.jsonl','基础清单'), '按图片／PDF 文档记录；保留原 train、validation、test 划分。'), (link('../data/watermark_training_increment_v1_1/training_tasks.jsonl','129 条统一任务清单'), '严格、近似、检测专用分开；原 121 条 manifest 继续兼容。'), (link('../data/watermark_regression_min_v1/manifest.jsonl','固定回归清单'), '当前 '+str(c['fixed_inputs'])+' 个输入。inputs/ 交给程序；references/ 只供处理后评分。')])+'<p>原始收件、授权记录、应用测试批次、网站参照和历史输出分别保留，不能因目录里仍有历史素材就算入当前固定集。</p>'),
        ('retired', '已经退役和移除的范围', '<p>旧 real_samples、syn、公开 40 对样本及配套旧格式已经退役。应用测试输入里的历史来源副本仍按批次保留，不代表恢复旧默认入口。</p><p>10 月 29 日仅从固定回归和当前效果页移除 real_001、real_002、real_004；原始采集、训练来源和历史封存继续按用途保留。'+link('../config/regression_retirements.json','查看授权范围与哈希')+'。范围变化不算质量提升。</p><p>'+link('数据集清单与清理_历史20261026.html','查看 10 月 26 日历史清理说明')+'；历史清理收据记录当时状态，不能据此承诺目前还能从废纸篓恢复。</p>'),
        ('check', '怎样检查', '<pre>./start.sh check-data\n./start.sh control-inventory\n./start.sh test --plan\n./start.sh test</pre><p>check-data 只验证固定文件和清单；test 按实际修改选择范围；发布用 test-release。当前独立泛化验证尚未成立。</p>')], date)
    case_rows = [(escape(r['case_id']),escape(r.get('label') or r['subtype']),str(r['page_count']), '无水印对照' if r['negative'] else escape(r['score_mode']),link('../data/watermark_regression_min_v1/'+r['paths']['input'],'输入')) for r in f['fixed']]
    fixed = page('最小回归测试集', [
        ('scope', '当前测试范围', f'<p class="note"><strong>{c["fixed_inputs"]} 个输入／{c["fixed_pages"]} 张页：核心 {c["core"]} 个＋真实扩展 {c["real"]} 个。</strong>这是开发回归集，用来发现退化，不能证明通用成功率。</p><p>核心包含 22 种图片样式、11 种 PDF 子类及无水印对照。真实扩展属于开发来源；缺精确配对的图片须人工复核。</p>'),
        ('run', '日常检查和固定集诊断', '<pre>./start.sh check-data\n./start.sh test --plan\n./start.sh test\n# 只诊断固定集，不代替完整防退化验收\nHF_HUB_OFFLINE=1 .venv/bin/python tools/run_min_regression.py --suite all</pre><p>三种模式分别是桌面默认、桌面关闭补全、CLI 默认。固定集诊断不自动开启桌面默认补全；要按记录的开关判断覆盖范围。完整发布仍用 test-release。</p>'),
        ('cases', '完整输入清单', table(['ID','样本','页数','评分条件','原图'],case_rows)),
        ('missing', '仍缺少的覆盖', '<ul>'+''.join('<li>'+escape(r['type'])+'：'+escape(r.get('reason') or r['status'])+'</li>' for r in f['coverage']['missing_or_incomplete'])+'</ul><p>'+link('../data/watermark_regression_min_v1/coverage.json','覆盖与缺口清单')+' · '+link('../data/watermark_regression_min_v1/file_hashes.json','固定文件哈希')+'</p>')], date)
    portal_rows = [
        ('当前说明','项目现状与后续开发说明.html','项目现状','现在的功能、限制、默认与实验边界。'),
        ('当前说明','去水印处理流程图.md','实际处理流程','按实际代码维护，从导入到导出。'),
        ('当前说明','代码与工具用途说明.html','代码与工具用途','目录职责、全部顶层工具、运行与备份入口。'),
        ('当前说明','数据集清单与清理.html','数据用途与数量','三套数据、来源与评分边界、历史清理入口。'),
        ('当前说明','最小回归测试集.html','固定回归范围',f'{c["fixed_inputs"]} 个输入、{c["fixed_pages"]} 张页与明确缺口。'),
        ('观察基线',overview,'基线效果总览','10 月 28 日封存结果，10 月 29 日按保留范围展示；不是当前代码全量验收。'),
        ('问题记录','去水印失败案例记录.html','已知失败记录','历史与基线代表证据，尚未解决的问题。'),
        ('实验与全部报告','../out/水印验证报告总目录.html','报告总目录','按观察基线、管控、诊断、实验、历史、方案分类。'),
        ('样本目录与历史测试','水印分类样本与最新测试.html','样本与分类目录','文件名保留兼容；页内测试对应历史版本，“最新”不能代表当前验收。'),
        ('训练准备','真实样本训练标注.html','真实样本标注','标注用于训练准备；检测框与近似配对不能当精确修复答案。'),
        ('历史说明','项目现状与后续开发说明_历史20261027.html','10 月 27 日现状原件','旧数字与旧判断保留当时条件。'),
        ('历史记录','历史开发记录_20261029.md','整理前 README 完整记录','保留此前功能细节、试验说明和历史命令，当前使用以首页为准。'),
        ('历史结果','首轮基线评测.html','10 月 26 日首轮测试','早期小范围结果，不迁移为当前成绩。'),
        ('数据准备资料','训练数据集说明.html','基础数据集详细说明','构成与准备过程；当前数量以数据登记表为准。'),
        ('数据准备资料','样本整理与入库.html','人工素材接收记录','保留实际来源、标注和收件过程。'),
        ('评测方案','水印覆盖与竞品评测.html','覆盖与比较方案','方案与历史调查，不代表已完成同口径竞品比较。')]
    portal_rows += [
        ('历史工程报告','基础功能修复测试报告.html','基础功能修复测试报告','早期开发复测，保留当时范围与结论。'),
        ('实际流程实例','单图去水印流程演示_20261028.html','10 月 28 日单图流程演示','来自当次真实运行与记录可视化，开发实例，不是独立验证。'),
        ('历史技术说明','项目技术文档.md','10 月 21 日技术文档','保留早期设计与能力判断，不能替代当前状态或实际流程。'),
        ('历史功能清单','功能矩阵.md','早期编号功能矩阵','历史编号和证据定位；当前效果与完成状态需看当前说明。'),
        ('历史评测','评测记录-2026-10-22-真实素材.md','10 月 22 日真实素材记录','早期样本和环境条件，保留追溯。'),
        ('开发研究','去水印算法最小对比_2026-10-27.md','10 月 27 日算法比较','研究与路线判断，不表示候选已接入。'),
        ('历史评测方案','水印覆盖与竞品评测方案_2026-10-26.docx','10 月 26 日评测方案 Word','既有方案原件，不能当作已完成比较的成绩。'),
        ('历史迁移说明','../MIGRATION.md','早期跨机迁移说明','历史环境与旧数据入口供追溯，当前安装以资源放置说明为准。'),
        ('历史交接资料','../knowme.md','早期交接资料','含旧状态与旧样本口径，当前状态以现状说明为准。'),
        ('后续任务草案','../下一步优化任务.md','算法优化任务草案','后续研究计划，不代表已经实现或验收。'),
        ('建设计划','独立验证集建设清单_20261030.html','独立验证集怎么建','高考卷式的独立数据建设计划：排除范围、12 场景清单、四条铁律；计划待确认，不是已完成的验证。'),
        ('环境与资源','../资源放置说明.md','当前资源放置说明','说明旧资源包与当前清单的差别，换电脑须单独保留后续产物。')]
    portal = page('项目文档总览', [
        ('start','从这里开始','<p class="note">了解现状先看“项目现状”；查看效果先看“基线效果总览”；改代码前看“代码与工具用途”。历史文件保留原位，分类标签说明它代表什么。</p>'),
        ('documents','每份文档的用途',table(['类型','入口','说明'],[(escape(a),link(b,d),escape(e)) for a,b,d,e in portal_rows])),
        ('maintenance','以后怎样保持清楚','<p>实际样本清单决定数量，版本记录决定成绩对应的代码，README 只保留入口。维护工具从清单生成当前状态、数据说明和工具目录；不从旧报告复制数字。</p><pre>.venv/bin/python tools/organize_project_docs.py\n.venv/bin/python tools/organize_project_docs.py --check\n.venv/bin/python tools/build_report_index.py</pre>')], date)
    return {'项目现状与后续开发说明.html':status,'项目文档总览.html':portal,
            '数据集清单与清理.html':data,'最小回归测试集.html':fixed,'代码与工具用途说明.html':guide}


def registry_and_readmes(f, date):
    c=f['counts']; registry=deepcopy(f['registry']); registry['as_of']=date
    detail={'training_v1':('base','图片/文档记录','基础训练准备，保留原分组；不表示已完成模型训练。'),
            'training_increment':('tasks','不重复图片/页任务训练记录',f'{c["strict"]} 严格配对＋{c["approximate"]} 近似配对＋{c["detection_only"]} 检测专用；修复{c["restoration"]}、检测{c["detection"]}，任务重叠。'),
            'regression_min':('fixed_inputs','输入文件',f'核心{c["core"]}＋真实扩展{c["real"]}，共{c["fixed_pages"]}张页；从既有数据派生，不增加独立来源。')}
    for item in registry['datasets']:
        key,unit,desc=detail[item['id']]; item.update(count=c[key],unit=unit,detail=desc)
        item['bytes']=sum(p.stat().st_size for p in (ROOT/item['path']).rglob('*') if p.is_file())
        item['bytes_note']='当前目录文件字节总和，包含说明和派生资料；不是独立样本量。'
        if item['id']=='regression_min': item.update(pages=c['fixed_pages'],retirement_profile='config/regression_retirements.json')
    registry['count_sources']={name:value for name,value in f['source_sha256'].items()}
    registry['independent_validation']=False
    registry['application_inventory']=dict(verified=f['inventory_error'] is None,
        registered_inputs=c['control_inputs'],registered_pages=c['control_pages'],
        error=f['inventory_error'],unregistered_files=f['unregistered'])
    for item in registry.get('retired_datasets',[]):
        item['detail']=item['detail'].replace('现有回归脚本仍引用','历史回归曾引用，已停用旧直接入口').replace('启动/工程测试仍引用','早期启动/测试曾引用，当前使用固定集')
    readme=f'''数据目录总入口（{date}）

当前只有 3 套工作数据，数量来自实际清单：
1. watermark_training_v1：{c['base']} 条图片/PDF文档记录；训练准备用途。
2. watermark_training_increment_v1_1：{c['tasks']} 个不重复任务样本。
   {c['strict']} 严格配对＋{c['approximate']} 近似配对＋{c['detection_only']} 检测专用；修复视图{c['restoration']}、检测视图{c['detection']}，不能相加。
3. watermark_regression_min_v1：{c['fixed_inputs']} 个输入／{c['fixed_pages']} 张页，核心{c['core']}＋真实扩展{c['real']}。
   派生开发回归集，不增加独立来源，不能证明通用成功率。

完整应用管控另包括批次与历史来源，共{c['control_inputs']}个输入场景／{c['control_pages']}张页；三模式{c['mode_pages']}页次，包含重复来源。
应用完整清单状态：{'发现未登记文件，检查未通过；详见 dataset_registry.json 的 application_inventory' if f['inventory_error'] else '清单核对通过，不等于效果合格'}。
原始收件 manual_collections、base_photos、20250227.pdf 保留；应用测试输入、网站参照另存，不另算训练集。
旧 real_samples、syn、公开40对及配套旧格式已退役；当前固定集已移除三张手写样本，原始来源和封存记录按用途保留。
输入与评分答案分开。训练清单或检测框不是当前单图推理的输入。
训练数据已准备不代表已经训练模型；目前没有独立泛化验证结论。

日常检查：./start.sh test --plan，然后 ./start.sh test。
固定数据校验：./start.sh check-data。发布：./start.sh test-release。
登记表：dataset_registry.json；说明：../docs/数据集清单与清理.html。
更新说明：.venv/bin/python tools/organize_project_docs.py。
历史清理收据保持原记录，不能据此承诺废纸篓目前可恢复。
'''
    fixed=f'''最小回归测试集（{date}）

当前 {c['fixed_inputs']} 个输入／{c['fixed_pages']} 张页；核心{c['core']}＋真实扩展{c['real']}。
用途：逐张发现修改后的退化，不用于证明通用成功率。
核心含22种图片样式、11种PDF子类和无水印对照。
真实扩展属于开发来源；缺精确配对的图片只作人工复核。
按用户要求移除 real_001、real_002、real_004，历史原件与原始来源按用途保留。

输入：inputs/core/、inputs/real/。参考：references/，只能在推理结束后评分使用。
清单：manifest.jsonl；覆盖与缺口：coverage.json；固定文件哈希：file_hashes.json。
日常：./start.sh test --plan，然后 ./start.sh test；发布：./start.sh test-release。
固定集诊断：HF_HUB_OFFLINE=1 .venv/bin/python tools/run_min_regression.py --suite all。
固定集诊断不是完整三模式防退化验收，也不保证启用桌面默认笔画补全。
说明与原图入口：../../docs/最小回归测试集.html。
'''
    return registry,readme,fixed


def readme_counts(f):
    c = f['counts']
    block = f'''<!-- current-dataset-counts:start -->
| 数据 | 当前数量 | 用途 |
|---|---|---|
| 基础数据 | {c['base']:,} 条图片／PDF 文档记录 | 训练准备与开发，保留原分组 |
| 人工增量 | {c['tasks']} 个不重复任务样本 | {c['strict']} 严格配对＋{c['approximate']} 近似配对＋{c['detection_only']} 检测专用；原{c['strict']}条清单保留兼容 |
| 固定回归 | {c['fixed_inputs']} 个输入／{c['fixed_pages']} 张页 | 核心{c['core']}＋真实扩展{c['real']}，用来检查退化 |

完整管控覆盖{c['control_inputs']}个输入场景／{c['control_pages']}张页，三种模式共{c['mode_pages']:,}页次，包含批次重复来源。以上数量不能直接相加为独立图片数，也不代表已完成模型训练。
{'当前发现未登记副本，完整清单检查未通过；文件保留原位，详情见当前说明。' if f['inventory_error'] else '当前应用清单已核对，清单一致不表示效果合格。'}
<!-- current-dataset-counts:end -->'''
    old = (ROOT / 'README.md').read_text()
    new, count = re.subn(r'<!-- current-dataset-counts:start -->.*?<!-- current-dataset-counts:end -->', lambda _: block, old, flags=re.S)
    if count != 1:
        raise ValueError('README 数量区块缺失或重复，停止更新')
    return new


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src') and v)


def check_links(paths):
    missing=[]; tails=[]; checked=0
    for path in paths:
        source=path.read_text(); parser=Links()
        if path.suffix=='.md': links=re.findall(r'\]\(([^)]+)\)',source)
        else: parser.feed(source);links=parser.links
        for value in links:
            value=value.strip('<>'); parts=urlsplit(value)
            if parts.scheme or not parts.path:continue
            target=path.parent/unquote(parts.path);checked+=1
            if not target.exists():missing.append(dict(file=str(path.relative_to(ROOT)),link=value))
            if target.suffix in {'.html','.md','.pdf','.docx','.json','.txt'} and (parts.fragment or parts.query):tails.append(dict(file=str(path.relative_to(ROOT)),link=value))
    return dict(checked=checked,missing=missing,document_link_tails=tails)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');ap.add_argument('--out',type=Path);args=ap.parse_args(argv)
    date=datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(); f=facts(); pages=render(f,date)
    paths=[DOCS/name for name in pages]
    if args.check:
        drift=[str(p.relative_to(ROOT)) for p in paths if not p.is_file() or p.read_text()!=pages[p.name]]
        registry,readme,fixed=registry_and_readmes(f,date)
        # Byte totals include the generated readmes, and settle after generation.
        registry_drift=f['registry']!=registry
        for p,content in [(ROOT/'data/README.txt',readme),(ROOT/'data/watermark_regression_min_v1/README.txt',fixed)]:
            if not p.exists() or p.read_text()!=content:drift.append(str(p.relative_to(ROOT)))
        if (ROOT/'README.md').read_text() != readme_counts(f):drift.append('README.md')
        result=dict(status='passed',scope='仅文档一致性与链接',counts=f['counts'],generated_document_drift=drift,registry_drift=registry_drift,inventory_error=f['inventory_error'],unregistered_files=f['unregistered'],links=check_links(paths+[ROOT/'README.md']))
        if drift or registry_drift or result['links']['missing'] or result['links']['document_link_tails']:result['status']='failed'
        print(json.dumps(result,ensure_ascii=False,indent=2));return int(result['status']!='passed')
    out=(args.out or ROOT/'out/项目整理'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f_说明更新')).resolve();out.mkdir(parents=True,exist_ok=False)
    for p in paths+[ROOT/'README.md',ROOT/'data/dataset_registry.json',ROOT/'data/README.txt',ROOT/'data/watermark_regression_min_v1/README.txt']:
        if p.exists():
            target=out/'更新前备份'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in ['项目现状与后续开发说明','数据集清单与清理']:
        old_date='20261027' if '现状' in name else '20261026'; archived=DOCS/f'{name}_历史{old_date}.html'
        if not archived.exists():shutil.copy2(DOCS/(name+'.html'),archived)
    for name,content in pages.items():(DOCS/name).write_text(content)
    (ROOT/'README.md').write_text(readme_counts(f))
    registry,readme,fixed=registry_and_readmes(f,date)
    (ROOT/'data/README.txt').write_text(readme);(ROOT/'data/watermark_regression_min_v1/README.txt').write_text(fixed)
    registry,_,_=registry_and_readmes(f,date)
    (ROOT/'data/dataset_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
    result=dict(version=VERSION,date=date,counts=f['counts'],source_sha256=f['source_sha256'],baseline_seal=f['baseline_seal'],changed_application_files=f['changed_application_files'],inventory_error=f['inventory_error'],unregistered_files=f['unregistered'],links=check_links(paths+[ROOT/'README.md']),inference_run=False,independent_validation=False,algorithm_changed=False,history_preserved=True)
    (out/'整理范围与来源.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(out=str(out),counts=f['counts'],links=result['links']),ensure_ascii=False,indent=2))
    return int(bool(result['links']['missing'] or result['links']['document_link_tails']))


if __name__=='__main__':raise SystemExit(main())
