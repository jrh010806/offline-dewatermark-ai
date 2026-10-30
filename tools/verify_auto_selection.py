"""Reproducible development comparison; never promotes the default baseline.

The run directory must already contain the preregistration and original frozen
snapshot. Source-only child processes finish before reference scoring starts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import regression_control as control
from tools.regression_guard import canonical, compare_page, digest


def freeze(run, name):
    dest = run / name
    dest.mkdir(exist_ok=False)
    snapshot = dest / '代码快照'
    for folder in ('wm', 'tools', 'config', 'tests'):
        shutil.copytree(ROOT/folder, snapshot/folder, copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ('README.md', 'AGENTS.md', 'requirements.txt', 'requirements.lock', 'start.sh'):
        if (ROOT/filename).is_file():
            shutil.copyfile(ROOT/filename, snapshot/filename)
    identity, _ = control.runtime()
    before = control.load(run/'原版冻结/运行记录.json')
    if identity != before['runtime_identity']:
        raise ValueError('模型或环境与修改前冻结不同，不能开始比较')
    control.save(dest/'冻结记录.json', dict(runtime_identity=identity,
        source_hashes={str(p.relative_to(snapshot)): digest(p)
                       for p in snapshot.rglob('*') if p.is_file()},
        preregistration_sha256=digest(run/'本轮范围与验收.json'),
        source_only=True, candidate_enabled=True, independent_real_validation=False))
    print(dest, flush=True)


def verify_snapshot(run, name):
    if name == '原版冻结':
        control.verify_frozen(run/name)
        return
    snap = run/name
    record = control.load(snap/'冻结记录.json')
    for path, expected in record['source_hashes'].items():
        if digest(snap/'代码快照'/path) != expected:
            raise ValueError('候选冻结代码发生变化：'+path)
    if control.runtime()[0] != record['runtime_identity']:
        raise ValueError('候选推理环境发生变化')


def cases(run):
    rows = control.load(run/'原版冻结/样本清单.json')['cases']
    target = control.load(run/'补充输入与范围.json')
    target.update(batch='用户图表', group='用户提供的开发样本', style='unlabelled', negative=False,
                  references=[{}], reference_identity=canonical([{}]))
    return [target] + rows


def jobs(run, scope):
    rows = cases(run)
    if scope in ('target', 'all'):
        for mode in ('desktop', 'desktop_protected', 'cli'):
            yield mode, '用户图表', rows[:1]
    if scope in ('regression', 'all'):
        grouped = defaultdict(list)
        for row in rows[1:]:
            grouped[row['batch']].append(row)
        for batch, members in grouped.items():
            yield 'desktop', batch, members


def infer(run, candidate, snapshot, scope):
    verify_snapshot(run, snapshot)
    frozen = run/snapshot/'代码快照'
    original = control.load(run/'原版冻结/运行记录.json')
    out = run/('候选实际运行_'+snapshot if candidate else '原版实际运行V2')
    env = dict(os.environ, HF_HUB_OFFLINE='1', QT_QPA_PLATFORM='offscreen', PYTHONHASHSEED='0',
               WM_LAMA_ONNX=original['model_files']['lama'])
    for mode, batch, rows in jobs(run, scope):
        relative = Path('用户图表')/mode if batch == '用户图表' else Path(canonical(batch+mode)[:20])
        dest = out/relative
        if (dest/'inference.json').is_file():
            existing = control.load(dest/'inference.json')
            expected = {(r['id'], p) for r in rows for p in range(1, r['page_count']+1)}
            if {(r['id'], r['page']) for r in existing} == expected:
                print('已有记录，保留：', mode, batch, flush=True)
                continue
            raise ValueError('旧目录有缺页；保留现场并另开轮次，禁止覆盖：'+str(dest))
        dest.mkdir(parents=True, exist_ok=False)
        task = dict(mode=mode, settings=original['profiles'][mode],
                    inputs=[{k: r[k] for k in ('id', 'input', 'input_sha256', 'dpi', 'page_count')}
                            for r in rows])
        control.save(dest/'task.json', task)
        command = ([sys.executable, str(frozen/'tools/run_auto_selection_probe.py'),
                    '--task', str(dest/'task.json'), '--candidate'] if candidate else
                   [sys.executable, str(frozen/'tools/control_worker.py'), str(dest/'task.json')])
        print('开始：', mode, batch, len(rows), '个输入', flush=True)
        with (dest/'运行日志.txt').open('w') as log:
            process = subprocess.Popen(command, cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT)
            last_progress, count = time.monotonic(), 0
            timed_out = False
            while process.poll() is None:
                time.sleep(.5)
                try:
                    current = len(control.load(dest/'inference.json')) if (dest/'inference.json').exists() else 0
                except json.JSONDecodeError:
                    continue
                if current > count:
                    last_progress, count = time.monotonic(), current
                    print('已完成：', mode, batch, count, flush=True)
                if time.monotonic()-last_progress > 240:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
        control.save(dest/'进程结果.json', dict(returncode=process.returncode, timeout=timed_out,
                                              automatic_selection=candidate, snapshot=snapshot,
                                              source_only=True))
        # Preserve missing pages and failed subprocesses in a separate receipt.
        actual = control.load(dest/'inference.json') if (dest/'inference.json').exists() else []
        observed = {(r['id'], r['page']) for r in actual}
        missing = [dict(id=r['id'], page=p, mode=mode, status='error', settings=task['settings'],
                        key=f'{mode}:{r["id"]}:{p}', error='推理进程退出或超时，缺页保留')
                   for r in rows for p in range(1, r['page_count']+1) if (r['id'], p) not in observed]
        control.save(dest/'缺页与进程异常.json', missing)
    verify_snapshot(run, snapshot)
    control.save(out/'推理阶段完成_未读取参考.json', dict(scope=scope, snapshot=snapshot,
                                                       source_only=True, scoring_started=False))


def collect(folder):
    records = []
    for path in sorted(folder.rglob('inference.json')):
        records.extend(control.load(path))
    for path in sorted(folder.rglob('缺页与进程异常.json')):
        records.extend(control.load(path))
    return records


def report(run, snapshot):
    before_dir, after_dir = run/'原版实际运行V2', run/('候选实际运行_'+snapshot)
    for folder in (before_dir, after_dir):
        if not (folder/'推理阶段完成_未读取参考.json').is_file():
            raise ValueError('尚未结束独立推理阶段，不能开始评分：'+str(folder))
    rows = {r['id']: r for r in cases(run)}
    old = {r['key']: r for r in collect(before_dir)}
    new = {r['key']: r for r in collect(after_dir)}
    expected = {f'{mode}:{r["id"]}:{p}' for mode, _, group in jobs(run, 'all')
                for r in group for p in range(1, r['page_count']+1)}
    if set(old) != expected or set(new) != expected:
        raise ValueError('新旧结果缺页或有额外记录，必须保留并完成范围核对')
    comparisons = []
    policy = control.load(run/'原版冻结/验收规则.json')
    for key in sorted(expected):
        a, b = deepcopy(old[key]), deepcopy(new[key])
        for record in (a, b):
            control.score(record, rows[record['id']])
        gate = compare_page(b, a, policy)
        comparisons.append(dict(key=key, before=a, after=b, gate=gate))
    dest = run/('评分与报告_'+snapshot)
    dest.mkdir(exist_ok=False)
    control.save(dest/'逐张新旧指标与原门槛.json', comparisons)
    summary = dict(pages=len(comparisons), changed=sum(c['before'].get('output_sha256') !=
        c['after'].get('output_sha256') for c in comparisons),
        failed_pages=sum(bool(c['gate']['failures']) for c in comparisons),
        review_pages=sum(bool(c['gate']['review']) for c in comparisons),
        independent_real_validation=False, full_release_passed=False, default_promoted=False,
        all_known_failures_preserved=True, note='局部开发对照；像素误差不是残留率，不提供项目成功率')
    control.save(dest/'范围与结论.json', summary)
    def link(path, label):
        relative = os.path.relpath(path, dest)
        return '<a href="'+quote(relative, safe='/')+'">'+escape(label)+'</a>'
    sections = []
    for item in comparisons:
        a, b, gate = item['before'], item['after'], item['gate']
        auto = b.get('processing', {}).get('automatic_selection', {})
        figures = []
        for record, field, label in ((a, '原图预览.jpg', '实际原图预览'),
                (a, '选区预览.jpg', '原版实际选区'), (b, '选区预览.jpg', '候选实际选区'),
                (a, '处理结果.png', '原版实际结果'), (b, '处理结果.png', '自动比较实际结果'),
                (b, '漏选多选核对.jpg', '处理后标注核对：蓝色漏选／橙色多选')):
            path = record.get('files', {}).get(field)
            if path:
                url = quote(os.path.relpath(path, dest), safe='/')
                figures.append('<figure><figcaption>'+label+'</figcaption><a href="'+url+'">'
                               '<img loading="lazy" src="'+url+'" alt="'+label+'"></a></figure>')
        metrics = [{k: r.get(k) for k in ('recall', 'strong_recall', 'precision', 'missed_pixels',
                    'false_positive_pixels', 'selected_pixels', 'selected_fraction', 'hole_selected_pixels',
                    'watermark_mae', 'background_mae', 'outside_selection_changed_pixels', 'elapsed_ms',
                    'method', 'repair_mode', 'status', 'error')} for r in (a, b)]
        detail = Path(b.get('files', {}).get('处理结果.png', '.')).parent/'自动比较详情/水印选区与处理结果复核.html'
        sections.append('<section><h2>'+escape(item['key'])+'</h2><p>'+escape(auto.get('reason', '沿用PDF结构路径'))
                        +'</p><p>'+escape('；'.join(gate['failures']) or '本页未发现超过原数值门槛的退化；仍需核对实际图片')
                        +'</p>'+ (link(detail, '查看两种实际候选和导出记录') if detail.is_file() else '')
                        +'<div class="images">'+''.join(figures)+'</div><details><summary>逐张指标、异常和选择证据</summary><pre>'
                        +escape(json.dumps(dict(before_after=metrics, comparison=auto, gate=gate), ensure_ascii=False, indent=2))
                        +'</pre></details></section>')
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动效果选择改造实际对照 · 2026-10-29</title>'
            '<style>body{font:15px/1.7 system-ui;max-width:1400px;margin:24px auto;padding:20px;background:#f3f5f4;color:#243d33}'
            'section{background:white;padding:20px;margin:24px 0}h2{font-size:18px;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}'
            'figure{margin:0}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#076a4b}</style>'
            '<h1>自动效果选择改造：实际新旧对照</h1><p>2026-10-29 · 原图独立推理后才读取参考评分。'
            '保留全部失败、未检出、超时和待复核。当前为试用候选，未完成完整发布放行或真实独立来源验证，普通启动保持原行为。</p>'
            '<p>下面全部来自实际输入、保存选区、实际输出及处理后标注核对；没有补画程序中间步骤。用户图表无干净参考，召回、精确率及完整恢复质量为未评估。</p>'
            '<pre>'+escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre>'
            +link(run/'本轮范围与验收.json', '修改前预登记标准')+' · '+link(dest/'逐张新旧指标与原门槛.json', '完整逐张指标')
            +''.join(sections)+'</html>')
    (dest/'自动效果选择改造_实际新旧对照_20261029.html').write_text(html)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(dest/'自动效果选择改造_实际新旧对照_20261029.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description='冻结并验证自动效果比较；不会更新默认基线')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=('freeze', 'before', 'after', 'report'), required=True)
    parser.add_argument('--snapshot', default='候选冻结V1')
    parser.add_argument('--scope', choices=('target', 'regression', 'all'), default='all')
    args = parser.parse_args()
    run = args.run.resolve()
    if args.phase == 'freeze':
        freeze(run, args.snapshot)
    elif args.phase == 'report':
        report(run, args.snapshot)
    else:
        infer(run, args.phase == 'after', args.snapshot if args.phase == 'after' else '原版冻结', args.scope)


if __name__ == '__main__':
    main()
