"""Freeze observations and enforce a reproducible, per-page regression gate.

This tool never changes detection/repair. Each run executes a private source
snapshot. References are scored only after the source-only worker exits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.control_inventory import build_inventory
from tools.regression_guard import apply_reviews, canonical, compare_runs, compare_subset, digest
from tools.regression_layers import jobs_for, selected_cases
from tools.regression_retirements import load_profile, project_baseline

STATE = ROOT / '.regression-control/baseline.json'
POLICY = ROOT / 'config/regression_control.json'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def profiles():
    from wm.contract import Thresholds
    from wm.batches import Batch
    from wm.app import KNOB_DEFAULTS
    base = dict(close_px=KNOB_DEFAULTS['close_px'], ocr_enabled=True, watermark_style=None)
    return {
        'desktop': dict(base, thresholds=asdict(Thresholds(watermark_recall_priority=Batch.__dataclass_fields__['recall_priority'].default))),
        'desktop_protected': dict(base, thresholds=asdict(Thresholds(watermark_recall_priority=False))),
        'cli': dict(base, thresholds=asdict(Thresholds())),
    }


def runtime():
    import platform
    import rapidocr_onnxruntime
    import onnxruntime
    from wm.lama import _resolve_model_path, _providers
    from wm import ocr
    if not ocr.available():
        raise RuntimeError('当前 OCR 不可用，不能以静默降级结果建立桌面默认基线')
    weights = {'lama': Path(_resolve_model_path()).resolve()}
    weights.update({'ocr/' + p.name: p.resolve() for p in
                    sorted(Path(rapidocr_onnxruntime.__file__).parent.rglob('*.onnx'))})
    packages = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()
                if d.metadata['Name']}
    identity = dict(python=sys.version, platform=platform.platform(), packages=packages,
                    providers=_providers(), available_providers=onnxruntime.get_available_providers(),
                    models={name: digest(p) for name, p in weights.items()},
                    environment={k: os.environ.get(k) for k in ('WM_LAMA_CUDA', 'WM_ORT_THREADS', 'OMP_NUM_THREADS')})
    return identity, weights


def copy_object(path, out):
    path = Path(path)
    target = out / '固定资源' / (digest(path) + path.suffix.lower())
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        # Copy bytes only: macOS system-font flags cannot be set on a new file.
        shutil.copyfile(path, target)
    if digest(target) != digest(path):
        raise ValueError(f'资源复制校验失败：{path}')
    return str(target)


def prepare(out, inventory, scope, selection_plan=None):
    os.environ['HF_HUB_OFFLINE'] = '1'
    out.mkdir(parents=True, exist_ok=False)
    save(out / '准备状态.json', dict(status='preparing', created_at=stamp()))
    policy = load(POLICY)
    # Capture dirty/untracked application code as well as the Git commit.
    snapshot = out / '代码快照'
    shutil.copytree(ROOT / 'wm', snapshot / 'wm', copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snapshot / 'tools').mkdir()
    for name in ('control_worker.py',):
        shutil.copyfile(ROOT / 'tools' / name, snapshot / 'tools' / name)
    for path in (POLICY, ROOT / 'start.sh', ROOT / 'AGENTS.md', ROOT / 'README.md',
                 ROOT / 'requirements.txt', ROOT / 'requirements.lock', ROOT / 'config/regression_tiers.json',
                 ROOT / 'config/regression_retirements.json'):
        if path.is_file():
            dest = snapshot / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    for name in ('regression_control.py', 'regression_guard.py', 'regression_layers.py', 'regression_retirements.py', 'control_inventory.py', 'run_min_regression.py'):
        shutil.copyfile(ROOT / 'tools' / name, snapshot / 'tools' / name)
    code_hashes = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob('*') if p.is_file()}
    identity, weights = runtime()
    model_files = {name: copy_object(p, out) for name, p in weights.items()}
    frozen = json.loads(json.dumps(inventory))
    if selection_plan:
        names = {name for values in selection_plan['mode_case_ids'].values() for name in values}
        frozen['cases'] = [r for r in frozen['cases'] if r['id'] in names]
    for row in frozen['cases']:
        row['original_input'] = row['input']
        row['input'] = copy_object(row['input'], out)
        for ref in row['references']:
            for role in ('clean', 'alpha', 'clean_pdf'):
                if ref.get(role):
                    ref[role] = copy_object(ref[role], out)
    # Identity refers to original hashes/metadata, not machine-specific frozen paths.
    save(out / '样本清单.json', frozen)
    save(out / '验收规则.json', policy)
    save(out / '原始清单.json', inventory)
    selections = selected_cases(frozen, dict(scope=scope, selection_plan=selection_plan))
    settings = profiles()
    if selection_plan:
        settings = {m: settings[m] for m in selection_plan['mode_case_ids']}
    expected = (selection_plan['expected_keys'] if selection_plan else
                [f"{m}:{r['id']}:{p}" for m in settings for r in selections for p in range(1, r['page_count'] + 1)])
    run = dict(schema_version=1, created_at=stamp(), scope=scope,
               inventory_sha256=canonical(inventory), policy_sha256=canonical(policy),
               profiles=settings, expected_keys=expected, records=[], complete=False,
               integrity_verified=False, runtime_identity=identity, model_files=model_files,
               code_hashes=code_hashes, completed_jobs=[], independent_validation=False,
               baseline_kind='当前观察基线；包含既有缺陷，不代表质量合格',
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
               note='输入和参考已复制冻结；仅原图进入独立推理进程。运行失败与超时不删样本。')
    if selection_plan:
        run['selection_plan'] = selection_plan
    save(out / '运行记录.json', run)
    save(out / '运行计划.json', {k: v for k, v in run.items()
          if k not in ('records', 'complete', 'integrity_verified', 'completed_jobs')})
    save(out / '准备状态.json', dict(status='prepared', prepared_at=stamp(),
                                   application_code_sha256=canonical(code_hashes)))
    save(out / '冻结完整性.json', {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                                 if p.is_file() and p.name != '运行记录.json'})
    return run


def verify_frozen(out):
    for name, value in load(out / '冻结完整性.json').items():
        path = out / name
        if not path.is_file() or digest(path) != value:
            raise ValueError(f'冻结内容被修改或缺失：{name}')


def score(record, case):
    """Called only in the controller, after the inference process has returned."""
    import numpy as np
    from PIL import Image
    import cv2
    record.update(style=case['style'], group=case['group'], negative=case['negative'],
                  reference_identity=case['reference_identity'])
    for key in ('recall', 'precision', 'strong_recall', 'missed_pixels', 'false_positive_pixels',
                'hole_selected_pixels', 'background_mae', 'watermark_mae', 'outside_truth_changed_pixels'):
        record[key] = None
    record.update(residual_quality=None, recognition_quality=None,
                  whitespace_quality=None, strong_alpha_threshold=0.08)
    if record['status'] != 'completed':
        return
    ref = case['references'][record['page'] - 1]
    if not ref.get('clean') or not ref.get('alpha'):
        record['evaluation_note'] = '无严格配对标注，质量指标未评估；变化必须人工复核'
        return
    output = np.asarray(Image.open(record['files']['处理结果.png']).convert('RGB'))
    selected = np.asarray(Image.open(record['files']['笔画选区.png']).convert('L')) > 0
    if Path(ref['alpha']).suffix == '.npz':
        with np.load(ref['alpha'], allow_pickle=False) as data:
            alpha = data['alpha'].copy()
            if 'strength' in data:
                alpha *= float(data['strength'])
        record['alpha_basis'] = '历史导出的覆盖形状乘透明度；仅在推理结束后的评分中读取'
    else:
        # These known labels describe the same geometric page. PDF fixed suite
        # is rendered at its existing 96 DPI, so resample the alpha to that grid.
        alpha = np.asarray(Image.open(ref['alpha']).convert('L').resize(
            (output.shape[1], output.shape[0]), Image.Resampling.LANCZOS), np.float32) / 255
    if ref.get('clean_pdf'):
        import pymupdf
        with pymupdf.open(ref['clean_pdf']) as doc:
            pix = doc[ref['page_number'] - 1].get_pixmap(dpi=ref['dpi'], alpha=False)
            clean = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    else:
        clean = np.asarray(Image.open(ref['clean']).convert('RGB'))
    if clean.shape != output.shape or alpha.shape != selected.shape:
        raise ValueError('参考与输出几何不一致，不能伪造配准成绩')
    truth = alpha > 0
    strong = alpha >= .08
    tp = int((selected & truth).sum())
    fp = int((selected & ~truth).sum())
    fn = int((~selected & truth).sum())
    count, labels = cv2.connectedComponents((~truth).astype('uint8'), connectivity=8)
    internal = np.ones(count, bool)
    internal[np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = False
    internal[0] = False
    error = np.abs(output.astype(np.float32) - clean).mean(2)
    record.update(recall=tp / (tp + fn) if tp + fn else None,
                  precision=tp / (tp + fp) if tp + fp else None,
                  strong_recall=float(selected[strong].mean()) if strong.any() else None,
                  missed_pixels=fn, false_positive_pixels=fp,
                  hole_selected_pixels=int((selected & internal[labels]).sum()),
                  background_mae=float(error[~truth].mean()) if (~truth).any() else None,
                  watermark_mae=float(error[truth].mean()) if truth.any() else None,
                  evaluation_note='像素误差不是残留率；字洞为真值封闭空洞，字间空白与正文语义需目视复核')
    if Path(case['input']).suffix.lower() != '.pdf':
        original = np.asarray(Image.open(case['input']).convert('RGB'))
    else:
        import pymupdf
        with pymupdf.open(case['input']) as doc:
            pix = doc[record['page'] - 1].get_pixmap(dpi=case['dpi'], alpha=False)
            original = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    if original.shape != output.shape:
        raise ValueError('评分重读原图几何不一致')
    record['outside_truth_changed_pixels'] = int(np.any(original != output, axis=2)[~truth].sum())
    # Save a diagnostic from annotations only after processing has finished.
    source = np.asarray(Image.open(record['files']['原图预览.jpg']).convert('RGB'))
    diagnostic = cv2.resize(source, (output.shape[1], output.shape[0]))
    diagnostic[truth & ~selected] = (0, 130, 255)
    diagnostic[~truth & selected] = (255, 150, 0)
    image = Image.fromarray(diagnostic)
    image.thumbnail((1000, 1000))
    path = Path(record['files']['处理结果.png']).parent / '漏选多选核对.jpg'
    image.save(path, quality=90)
    record['files'][path.name] = str(path)


def execute(out):
    verify_frozen(out)
    run = load(out / '运行记录.json')
    if runtime()[0] != run['runtime_identity']:
        raise ValueError('运行期间依赖、模型或推理环境改变，结果不能混合作为同版本验证')
    for key, value in load(out / '运行计划.json').items():
        if run.get(key) != value:
            raise ValueError(f'运行计划被改动：{key}')
    frozen = load(out / '样本清单.json')
    policy = load(out / '验收规则.json')
    cases = selected_cases(frozen, run)
    index = {r['id']: r for r in cases}
    groups = {}
    for case in cases:
        groups.setdefault(case['batch'], []).append(case)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', QT_QPA_PLATFORM='offscreen', PYTHONHASHSEED='0',
               WM_LAMA_ONNX=run['model_files']['lama'])
    jobs = jobs_for(frozen, run)
    for number, (mode, batch, rows) in enumerate(jobs, 1):
        job_key = mode + ':' + batch
        if job_key in run['completed_jobs']:
            continue
        dest = out / '逐批运行' / canonical(job_key)[:20]
        # An interrupted unfinished job is retained and retried in a new folder.
        if dest.exists():
            dest = dest.with_name(dest.name + '_续跑_' + stamp())
        dest.mkdir(parents=True)
        settings = dict(run['profiles'][mode], batch_context=canonical([
            (r['id'], r['input_sha256'], r['page_count']) for r in rows]))
        task = dict(mode=mode, settings=settings, inputs=[{k: r[k] for k in
                    ('id', 'input', 'input_sha256', 'dpi', 'page_count')} for r in rows])
        # No references, true masks, style labels, synthetic parameters or clean pages.
        save(dest / 'task.json', task)
        print(f'[{number}/{len(jobs)}] {mode} / {batch}', flush=True)
        with (dest / '运行日志.txt').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(out / '代码快照/tools/control_worker.py'),
                                     str(dest / 'task.json')], cwd=out / '代码快照',
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            progress, last_count = time.monotonic(), 0
            while proc.poll() is None:
                time.sleep(0.5)
                file = dest / 'inference.json'
                count = len(load(file)) if file.exists() else 0
                if count > last_count:
                    progress, last_count = time.monotonic(), count
                if time.monotonic() - progress > policy['page_timeout_seconds']:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    log.write('\n管控器：单页无进展超时，已终止本批；缺页全部记失败。\n')
                    break
        records = load(dest / 'inference.json') if (dest / 'inference.json').exists() else []
        seen = {r['key'] for r in records}
        for case in rows:
            for p in range(1, case['page_count'] + 1):
                key = f"{mode}:{case['id']}:{p}"
                if key not in seen:
                    records.append(dict(key=key, id=case['id'], page=p, mode=mode, status='error',
                                        error=f'工作进程退出或超时，退出码 {proc.returncode}', settings=settings))
        for record in records:
            try:
                score(record, index[record['id']])
            except Exception as exc:
                record.update(status='error', error=f'评分/几何校验失败：{type(exc).__name__}: {exc}')
        run['records'].extend(records)
        run['completed_jobs'].append(job_key)
        save(out / '运行记录.json', run)
        write_report(out, run)
    verify_frozen(out)
    if runtime()[0] != run['runtime_identity']:
        raise ValueError('运行期间依赖、模型或推理环境改变，不能封存为同版本验证')
    # Missing/extra/duplicate pages are rejected later as well as here.
    run['complete'] = (len(run['records']) == len(run['expected_keys']) and
                       {r['key'] for r in run['records']} == set(run['expected_keys']))
    run['integrity_verified'] = True
    run['completed_at'] = stamp()
    save(out / '运行记录.json', run)
    write_report(out, run)
    seal(out)
    return run


def seal(out):
    save(out / '封存校验.json', {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                               if p.is_file() and p.name != '封存校验.json' and '__pycache__' not in p.parts})


def verified_run(out):
    if not (out / '封存校验.json').is_file():
        raise ValueError(f'运行尚未封存：{out}')
    for name, value in load(out / '封存校验.json').items():
        path = out / name
        if not path.is_file() or digest(path) != value:
            raise ValueError(f'已封存产物损坏或被修改：{path}')
    return load(out / '运行记录.json')


def write_report(out, run, old=None, gate=None):
    from urllib.parse import quote
    old_rows = {r['key']: r for r in (old or {}).get('records', [])}
    checks = {r['key']: r for r in (gate or {}).get('cases', [])}
    def link(path):
        return quote(os.path.relpath(path, out), safe='/')
    cards = []
    for row in run['records']:
        figures = []
        before = old_rows.get(row['key'], {})
        for item, filename, title in ((row, '原图预览.jpg', '原图（缩略）'),
                (before, '选区预览.jpg', '基线选区'), (row, '选区预览.jpg', '本版选区'),
                (before, '处理结果.png', '基线结果'), (row, '处理结果.png', '本版结果'),
                (row, '漏选多选核对.jpg', '蓝色漏选／橙色多选（评分后标记）')):
            p = item.get('files', {}).get(filename)
            if p:
                figures.append(f'<figure><a href="{link(p)}"><img loading="lazy" src="{link(p)}"></a><figcaption>{title}</figcaption></figure>')
        values = {k: row.get(k) for k in ('status', 'recall', 'strong_recall', 'precision',
                  'missed_pixels', 'false_positive_pixels', 'hole_selected_pixels', 'selected_pixels',
                  'selected_fraction', 'background_mae', 'watermark_mae', 'changed_pixels',
                  'outside_selection_changed_pixels', 'method', 'repair_mode', 'elapsed_ms', 'error')}
        cards.append(f'<article><h2>{escape(row["key"])}</h2><p>{escape(str(checks.get(row["key"], {})))}</p>'
                     '<div class="figs">' + ''.join(figures) + '</div><pre>' +
                     escape(json.dumps(values, ensure_ascii=False, indent=2)) + '</pre></article>')
    failed = sum(r['status'] != 'completed' for r in run['records'])
    summary = (f'已记录 {len(run["records"])} / {len(run["expected_keys"])} 页次，运行失败 {failed}。'
               '这是当前观察基线，包含原有效果缺陷，不代表质量合格。')
    if gate:
        summary = f'防退化门槛：{"通过" if gate["gate_passed"] else "未放行"}；退化/异常 {gate["failed_pages"]} 页次；变化待复核 {gate["review_pages"]} 页次。'
        if gate.get('scope', 'all') != 'all':
            summary = f'日常局部检查：{"通过" if gate["check_passed"] else "未通过"}；范围 {gate["scope"]}；退化/异常 {gate["failed_pages"]}，变化待复核 {gate["review_pages"]}。本结果不能用于更新默认基线。'
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>项目防退化控制报告</title><style>body{font:15px/1.7 system-ui;max-width:1600px;margin:30px auto;padding:20px;background:#f2f5f7;color:#243b46}article{background:white;padding:20px;margin:20px 0;border-radius:10px}.figs{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:200px}img{width:100%;height:320px;object-fit:contain}pre{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:17px;overflow-wrap:anywhere}</style><h1>项目防退化控制报告 · ' + escape(run['created_at']) + '</h1><p>' + escape(summary) + '</p><p>模式：桌面默认、桌面关闭增强、CLI 默认。逐张比较，任何明显退化阻断；输出、选区或路径变化待复核。像素误差不是水印残留率，未知质量保持未评估；所有样本均不作独立泛化证明。</p><pre>' + escape(json.dumps({k: v for k, v in (gate or {}).items() if k != 'cases'}, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(cards) + '</html>'
    (out / '项目防退化控制报告.html').write_text(content)


def compare(candidate, baseline, out, reviews=None):
    new, old = verified_run(candidate), verified_run(baseline)
    scope_revision = None
    if new['inventory_sha256'] != old['inventory_sha256']:
        old, _ = project_baseline(old, load(baseline / '原始清单.json'),
            load_profile(candidate / '代码快照/config/regression_retirements.json'), digest(baseline / '封存校验.json'))
        scope_revision = old.get('sample_scope_revision')
    policy = load(candidate / '验收规则.json')
    if new.get('selection_plan'):
        if new['selection_plan'].get('baseline_seal') != digest(baseline / '封存校验.json'):
            raise ValueError('局部计划绑定的基线与本次比较不一致')
        rules = load(candidate / '代码快照/config/regression_tiers.json')
        result = compare_subset(new, old, policy, load(candidate / '原始清单.json'), rules)
    else:
        result = compare_runs(new, old, policy)
    if scope_revision:
        result['sample_scope_revision'] = scope_revision
    result.update(candidate=str(candidate), baseline=str(baseline),
                  candidate_seal=digest(candidate / '封存校验.json'),
                  baseline_seal=digest(baseline / '封存校验.json'))
    template = dict(candidate_seal=result['candidate_seal'], baseline_seal=result['baseline_seal'],
                    instructions='逐张查看原图、新旧选区及结果后填写。decision 可为 accept_change 或 regression；不能豁免数值退化。',
                    cases=[dict(key=r['key'], decision='pending', reviewer='', reason='')
                           for r in result['cases'] if r['review']])
    if reviews:
        result = apply_reviews(result, load(reviews), result['candidate_seal'], result['baseline_seal'])
        result['reviews_sha256'] = digest(reviews)
    out.mkdir(parents=True, exist_ok=False)
    save(out / '防退化检查.json', result)
    save(out / '人工复核模板.json', template)
    if reviews:
        shutil.copy2(reviews, out / '本轮人工复核记录.json')
    write_report(out, new, old, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description='项目防退化管控：只保存和检查，不修改算法')
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('inventory', help='核对截图中的全部输入与参考')
    p.add_argument('--out', type=Path)
    p = sub.add_parser('promote', help='仅在完整门槛通过后，显式更新本机观察基线')
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reviews', type=Path)
    for name in ('freeze', 'check'):
        p = sub.add_parser(name, help='冻结当前观察基线' if name == 'freeze' else '全模式运行并强制对照基线')
        p.add_argument('--out', type=Path)
        p.add_argument('--scope', choices=['all', 'quick'], default='all')
        p.add_argument('--prepare-only', action='store_true', help='只冻结，不运行；不能验收')
        if name == 'check':
            p.add_argument('--baseline', type=Path)
    p = sub.add_parser('resume', help='继续未封存的运行；保留中断现场')
    p.add_argument('run', type=Path)
    p = sub.add_parser('compare', help='对封存结果重新核对，不重跑推理')
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--reviews', type=Path)
    args = ap.parse_args(argv)
    if args.command == 'compare':
        out = (args.out or ROOT / 'out/防退化对照' / stamp()).resolve()
        result = compare(args.candidate.resolve(), args.baseline.resolve(), out, args.reviews)
        print(out / '项目防退化控制报告.html')
        return 0 if result.get('check_passed', result['gate_passed']) else 1
    if args.command == 'promote':
        if not STATE.exists():
            raise ValueError('缺少当前观察基线，先执行 freeze-baseline')
        pointer = load(STATE)
        baseline = Path(pointer['path'])
        if digest(baseline / '封存校验.json') != pointer['seal']:
            raise ValueError('当前观察基线封存标识改变')
        candidate = args.candidate.resolve()
        out = ROOT / 'out/防退化对照' / (stamp() + '_基线更新检查')
        result = compare(candidate, baseline, out, args.reviews)
        if not result['gate_passed']:
            print(f'门槛未通过，基线保持原样。见 {out}', file=sys.stderr)
            return 1
        history = STATE.with_name('baseline_' + stamp() + '.json')
        shutil.copy2(STATE, history)
        save(STATE, dict(path=str(candidate), seal=digest(candidate / '封存校验.json'),
                         previous=pointer, gate_report=str(out / '防退化检查.json'),
                         role='经相对退化检查更新的观察基线，不代表独立质量验收'))
        print(f'观察基线已更新；旧版本保留：{history}')
        return 0
    if args.command == 'resume':
        out = args.run.resolve()
        if (out / '封存校验.json').exists():
            raise ValueError('已封存运行不可追加；请新建运行目录')
        run = execute(out)
        # Resuming a check must also compare; a raw completed run is never a release pass.
        if run.get('baseline'):
            result = compare(out, Path(run['baseline']), out.with_name(out.name + '_对照_' + stamp()))
            return 0 if result.get('check_passed', result['gate_passed']) else 1
        if not STATE.exists() and run['scope'] == 'all' and run['complete']:
            STATE.parent.mkdir(exist_ok=True)
            save(STATE, dict(path=str(out), seal=digest(out / '封存校验.json'),
                             role='当前观察基线，不代表质量合格'))
        return 0 if run['complete'] and not any(r['status'] != 'completed' for r in run['records']) else 1
    inventory = build_inventory(ROOT, ROOT / 'data/应用测试输入')
    if args.command == 'inventory':
        print(json.dumps({k: v for k, v in inventory.items() if k != 'cases'}, ensure_ascii=False, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            if args.out.exists():
                raise ValueError('不覆盖已有清单')
            save(args.out, inventory)
        return 0
    baseline = None
    if args.command == 'check':
        baseline = args.baseline or (Path(load(STATE)['path']) if STATE.exists() else None)
        if baseline is None:
            print('未建立观察基线。先执行 ./start.sh freeze-baseline；缺基线时禁止放行。', file=sys.stderr)
            return 2
        verified_run(baseline.resolve())
        if args.baseline is None and digest(baseline / '封存校验.json') != load(STATE)['seal']:
            raise ValueError('当前观察基线封存标识改变')
    out = (args.out or ROOT / 'out/项目防退化控制' / (stamp() + '_' + args.command)).resolve()
    new_directory = not out.exists()
    try:
        run = prepare(out, inventory, args.scope)
    except Exception as exc:
        if new_directory and out.is_dir() and not (out / '封存校验.json').exists():
            # Do not overwrite an existing run when exist_ok=False rejected it.
            if not (out / '运行记录.json').exists():
                save(out / '准备失败记录.json', dict(status='preparation_failed',
                    error=f'{type(exc).__name__}: {exc}', recorded_at=stamp(), inference_started=False))
        raise
    if baseline:
        run['baseline'] = str(baseline.resolve())
        save(out / '运行记录.json', run)
        # The comparison target is immutable once the candidate is prepared.
        plan = load(out / '运行计划.json')
        plan['baseline'] = run['baseline']
        save(out / '运行计划.json', plan)
        hashes = load(out / '冻结完整性.json')
        hashes['运行计划.json'] = digest(out / '运行计划.json')
        save(out / '冻结完整性.json', hashes)
    print(f'冻结完成：{out}', flush=True)
    if args.prepare_only:
        print('尚未运行，不具备防退化验收结论。')
        return 2
    run = execute(out)
    print(out / '项目防退化控制报告.html', flush=True)
    if baseline:
        result = compare(out, baseline.resolve(), out.with_name(out.name + '_逐张对照'))
        return 0 if result['gate_passed'] else 1
    if not STATE.exists() and args.scope == 'all' and run['complete']:
        STATE.parent.mkdir(exist_ok=True)
        save(STATE, dict(path=str(out), seal=digest(out / '封存校验.json'),
                         role='当前观察基线，不代表质量合格'))
    return 0 if run['complete'] and not any(r['status'] != 'completed' for r in run['records']) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'管控阻断：{exc}', file=sys.stderr)
        raise SystemExit(2)
