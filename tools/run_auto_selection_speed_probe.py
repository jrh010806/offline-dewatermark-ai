"""Frozen, source-only A/B run of the explicitly requested single chart.

No algorithm edits, references, broad inventory or baseline promotion occur here.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(folder):
    return {str(p.relative_to(folder)): digest(p) for p in folder.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def verify(run, name):
    frozen = run/name
    if hashes(frozen/'代码快照') != load(frozen/'冻结记录.json')['source_hashes']:
        raise RuntimeError('冻结代码变化：'+name)


def invoke(command, folder, env, timeout):
    started = time.monotonic()
    with (folder/'运行日志.txt').open('w') as log:
        child = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            code, timed_out = child.returncode, True
    receipt = dict(returncode=code, timeout=timed_out,
                   wall_seconds=round(time.monotonic()-started, 3))
    save(folder/'进程记录.json', receipt)
    return receipt


def report(run, runs, integrity, unit_receipt):
    from PIL import Image
    import numpy as np
    metrics = dict(scope='同一开发图表、桌面默认参数、实际BatchWorker、试用开启',
                   source_only=True, independent_validation=False,
                   full_engineering_suite=False, cross_style_regression=False,
                   integrity=integrity, focused_tests=unit_receipt,
                   recall=None, precision=None, residual_quality=None,
                   content_restoration_quality=None, rows=runs)
    completed = all(r.get('record', {}).get('status') == 'completed' for r in runs)
    if completed:
        before, after = [r['record'] for r in runs]
        old_image = np.asarray(Image.open(before['files']['处理结果.png']))
        new_image = np.asarray(Image.open(after['files']['处理结果.png']))
        old_mask = np.asarray(Image.open(before['files']['笔画选区.png']))
        new_mask = np.asarray(Image.open(after['files']['笔画选区.png']))
        metrics.update(output_difference_pixels=int(np.any(old_image != new_image, axis=2).sum()),
                       mask_difference_pixels=int(np.count_nonzero(old_mask != new_mask)),
                       speed_improvement=1-after['elapsed_ms']/max(1, before['elapsed_ms']))
        metrics['speed_target_passed'] = bool(
            metrics['speed_improvement'] >= .20 and metrics['output_difference_pixels'] == 0
            and metrics['mask_difference_pixels'] == 0 and after['outside_selection_changed_pixels'] == 0
            and all(integrity.values()) and unit_receipt['returncode'] == 0)
    else:
        metrics['speed_target_passed'] = False
    metrics['image_quality_verified'] = False
    save(run/'单图最终记录.json', metrics)
    def link(path, label):
        return '<a href="'+quote(os.path.relpath(path, run))+'">'+escape(label)+'</a>'
    def figure(path, label):
        return '<figure><figcaption>'+escape(label)+'</figcaption><a href="'+quote(os.path.relpath(path, run))+'"><img src="'+quote(os.path.relpath(path, run))+'"></a></figure>'
    cards = figure(run/'用户输入原件.png', '用户输入原图')
    table = ''
    for item in runs:
        r = item.get('record', {})
        a = r.get('processing', {}).get('automatic_selection', {})
        count = (a.get('repair_count', sum(x['status'] == 'completed' for x in a.get('records', [])))
                 if r.get('status') == 'completed' else '未知（无完整记录）')
        table += '<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in (
            item['label'], r.get('status', '缺页'), r.get('elapsed_ms'), count,
            a.get('chosen'), r.get('selected_pixels'), r.get('outside_selection_changed_pixels'),
            a.get('status'), item['receipt']['timeout']))+'</tr>'
        if r.get('status') == 'completed':
            cards += figure(Path(r['files']['处理结果.png']), item['label']+'：实际处理结果')
            cards += figure(Path(r['files']['笔画选区.png']), item['label']+'：实际选区')
    conclusion = ('提速目标通过，图像质量仍未达标。' if metrics['speed_target_passed'] else
                  '本轮未通过预先记录的提速及结果保持门槛。')
    stats = ('处理结果不同像素：'+str(metrics.get('output_difference_pixels'))+
             '；选区不同像素：'+str(metrics.get('mask_difference_pixels'))+
             '；耗时降低：'+(f"{metrics['speed_improvement']:.1%}" if 'speed_improvement' in metrics else '未完成比较'))
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自动选择提速单图验证 · 2026-10-29</title>
<style>body{font:16px/1.6 system-ui;margin:32px;color:#24332f;background:#f4f7f5}main{max-width:1500px;margin:auto}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #ccd8d2}section{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}figure{margin:0;background:white;padding:12px}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#007454}</style><main><h1>自动选择提速单图验证</h1>'''
    html += '<p><b>'+conclusion+'</b></p><p>'+escape(stats)+'</p>'
    html += '<p>仅测试用户指定的深色大字图表，使用本轮同一输入、模型与参数的新旧冻结代码，各运行一次。测试中不改代码、不因结果调参。模型/环境、代码和输入校验见记录。完整工程测试、跨样式回归及独立验证本轮均未运行。</p>'
    html += '<p>已知问题：旧结果中的194、264、313消失及淡水印残留，仍须逐图核对。相同输出只能证明未改变这一结果，不能证明去水印质量合格。自动观测不是真值；真实召回、精确率、残留和正文恢复质量保持未评估。候选默认关闭，不更新观察基线。</p>'
    html += '<table><tr>'+''.join('<th>'+s+'</th>' for s in ('版本','状态','耗时毫秒','实际修复次数','选择','选区像素','选区外改动','自动风险状态','进程超时'))+'</tr>'+table+'</table><h2>实际运行捕获</h2><section>'+cards+'</section>'
    html += '<h2>可复现记录</h2><p>'+link(run/'本轮范围与验收.json', '运行前门槛')+' · '+link(run/'单图最终记录.json', '完整结果及选择依据')+' · '+link(run/'测试冻结及校验.json', '测试前后完整性校验')+' · '+link(run/'候选单元检查/运行日志.txt', '相关工程检查日志')+'</p>'
    html += '<h2>首次选择与按需补试记录</h2><pre>'+escape(json.dumps([r.get('record', {}).get('processing', {}).get('automatic_selection') for r in runs], ensure_ascii=False, indent=2))+'</pre></main></html>'
    (run/'自动选择提速单图验证_20261029.html').write_text(html)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='冻结后仅测指定单图，结果无论好坏均保留')
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not (run/'修改前冻结/冻结记录.json').is_file() or (run/'候选冻结').exists():
        parser.error('需要本轮已保存的修改前冻结，且不得覆盖候选冻结')
    snapshot = run/'候选冻结/代码快照'
    for folder in ('wm', 'tools', 'config', 'tests'):
        shutil.copytree(ROOT/folder, snapshot/folder, copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for filename in ('README.md', 'AGENTS.md', 'start.sh', 'requirements.txt'):
        if (ROOT/filename).is_file():
            shutil.copyfile(ROOT/filename, snapshot/filename)
    from tools.regression_control import runtime
    identity, weights = runtime()
    baseline = load(run/'修改前冻结/冻结记录.json')
    if identity != baseline['runtime_identity']:
        raise RuntimeError('本轮模型/环境身份变化，不能启动比较')
    save(run/'候选冻结/冻结记录.json', dict(source_hashes=hashes(snapshot),
         runtime_identity=identity, model_files={k: str(v) for k, v in weights.items()}))
    working_before = {f: digest(ROOT/f) for f in ('wm/auto_selection.py', 'wm/export_review.py',
                       'tests/test_auto_selection.py', 'tools/run_auto_selection_speed_probe.py')}
    input_task = load(run/'源图任务.json')
    for item in input_task['inputs']:
        if digest(Path(item['input'])) != item['input_sha256']:
            raise RuntimeError('源图校验失败')
    save(run/'测试开始记录.json', dict(started=datetime.now().isoformat(),
         working_hashes=working_before, code_frozen=True, no_edits_during_test=True))
    env = dict(os.environ, HF_HUB_OFFLINE='1', QT_QPA_PLATFORM='offscreen', PYTHONHASHSEED='0',
               WM_LAMA_ONNX=baseline['model_files']['lama'], PYTHONPATH=str(snapshot))
    unit = run/'候选单元检查'
    unit.mkdir()
    print('冻结完成，开始相关工程检查；后续不改代码', flush=True)
    unit_receipt = invoke([sys.executable, '-m', 'unittest', 'discover', '-s', str(snapshot/'tests'),
                           '-p', 'test_auto_selection.py', '-v'], unit, env, 90)
    rows = []
    for name, label in (('修改前冻结', '原版：两次修复'), ('候选冻结', '候选：先选后修复')):
        verify(run, name)
        folder = run/label
        folder.mkdir()
        save(folder/'task.json', input_task)
        env['PYTHONPATH'] = str(run/name/'代码快照')
        print('开始单图实际BatchWorker：'+label, flush=True)
        receipt = invoke([sys.executable, str(run/name/'代码快照/tools/run_auto_selection_probe.py'),
                          '--task', str(folder/'task.json'), '--candidate'], folder, env, 210)
        records = load(folder/'inference.json') if (folder/'inference.json').is_file() else []
        rows.append(dict(label=label, receipt=receipt,
                         record=records[0] if len(records) == 1 else dict(status='missing', records=records)))
        print(label+' 完成：'+str(receipt), flush=True)
    for name in ('修改前冻结', '候选冻结'):
        verify(run, name)
    integrity = dict(frozen_sources_unchanged=True,
         working_sources_unchanged=all(digest(ROOT/f) == value for f, value in working_before.items()),
         models_and_environment_unchanged=runtime()[0] == identity,
         input_unchanged=all(digest(Path(i['input'])) == i['input_sha256'] for i in input_task['inputs']))
    save(run/'测试冻结及校验.json', integrity)
    metrics = report(run, rows, integrity, unit_receipt)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('rows',)}, ensure_ascii=False), flush=True)
    print(run/'自动选择提速单图验证_20261029.html', flush=True)
    return 0 if metrics['speed_target_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
