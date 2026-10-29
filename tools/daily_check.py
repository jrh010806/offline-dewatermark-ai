"""Fast daily checks by impact; full baseline promotion stays a separate action."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.regression_control import (STATE, compare, execute, load, main as control_main,
                                     prepare, save, stamp, verified_run)
from tools.regression_guard import digest
from tools.regression_layers import changed_files, make_plan
from tools.regression_retirements import load_profile, project_baseline


def watch_sources():
    files = []
    for folder in ('wm', 'tools', 'tests', 'config'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.sh', '.mjs'))
    files.extend(ROOT / name for name in ('AGENTS.md', 'README.md', 'start.sh', 'requirements.txt', 'requirements.lock') if (ROOT / name).is_file())
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def document_links(files, baseline):
    checked, missing, inherited = [], [], []
    for name in files:
        path = ROOT / name
        if path.suffix not in ('.md', '.html') or not path.is_file():
            continue
        text = path.read_text(errors='replace')
        links = re.findall(r'\]\(([^)]+)\)', text) if path.suffix == '.md' else re.findall(r'href=["\']([^"\']+)', text)
        old = Path(baseline) / '代码快照' / name
        original = old.read_text(errors='replace') if old.is_file() else ''
        for link in links:
            link = link.strip('<>')
            value = urlsplit(link)
            if value.scheme or not value.path or '$' in link or ' ' in link:
                continue
            target = path.parent / unquote(value.path)
            checked.append(link)
            if not target.exists():
                (inherited if link in original else missing).append(dict(file=name, link=link))
    return dict(checked=len(checked), missing=missing, inherited_broken_links=inherited)


def write_daily_report(out, plan, result):
    label = {'static': '文档与链接', 'unit': '工程检查', 'targeted': '定向回归', 'cross': '跨样式回归', 'full': '完整放行检查'}[plan['level']]
    rows = ''.join(f'<tr><td>{escape(r["file"])}</td><td>{escape(r["level"])}</td><td>{escape(r["reason"])}</td></tr>' for r in plan['reasons'])
    comparison = f'<p><a href="{escape(result["comparison"], quote=True)}">逐张对照与检查结论</a></p>' if result.get('comparison') else ''
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>分层防退化检查 · {label}</title><style>body{{font:15px/1.8 system-ui;max-width:1100px;margin:30px auto;padding:25px;color:#243d33;background:#f3f6f3}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px}}a{{color:#176a4d}}</style><h1>分层防退化检查 · {label}</h1><p>本轮范围 {plan['page_mode_count']} 页次，估算 {plan['estimated_seconds']} 秒；估算来自旧基线，不是速度承诺。</p><p><strong>只有完整三模式门槛通过才具备放行资格；日常局部通过不能更新默认基线。</strong>原有数值门槛未改变，原观察基线继续复用。</p><h2>本轮结果</h2><pre>{escape(json.dumps(result,ensure_ascii=False,indent=2))}</pre>{comparison}<h2>为何选择这一级</h2><table><thead><tr><th>文件</th><th>最低检查层</th><th>原因</th></tr></thead><tbody>{rows}</tbody></table><p><a href="分层检查计划.json">固定检查计划与样本范围</a> · <a href="分层检查结果.json">结构化结果</a></p></html>'''
    (out / '分层防退化检查报告.html').write_text(report)


def main(argv=None):
    ap = argparse.ArgumentParser(description='日常按影响分层检查；完整放行请使用 test-release')
    ap.add_argument('--level', choices=['auto', 'static', 'unit', 'targeted', 'cross', 'full'], default='auto')
    ap.add_argument('--files', nargs='*', default=[], help='补充影响文件；不能覆盖自动发现的应用变更')
    ap.add_argument('--cases', nargs='*', default=[], help='补充已登记的目标难例；自动保留完整原批次上下文')
    ap.add_argument('--plan', action='store_true', help='仅列出计划，不运行模型或测试')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    if not STATE.is_file():
        raise ValueError('尚无可复用的观察基线，请先建立；不会自动启动全量建基线')
    pointer = load(STATE)
    baseline = Path(pointer['path'])
    if digest(baseline / '封存校验.json') != pointer['seal']:
        raise ValueError('观察基线的封存标识改变')
    old = load(baseline / '运行记录.json')
    inventory = load(baseline / '原始清单.json')
    old, inventory = project_baseline(old, inventory, load_profile(), pointer['seal'])
    rules = load(ROOT / 'config/regression_tiers.json')
    files = changed_files(ROOT, old, args.files)
    plan = make_plan(files, inventory, old, rules, args.level, args.cases)
    plan['baseline_seal'] = pointer['seal']
    plan['baseline'] = str(baseline)
    if old.get('sample_scope_revision'):
        plan['sample_scope_revision'] = old['sample_scope_revision']
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.plan:
        print('仅计划；未运行测试或推理，未形成验收结论。')
        return 0
    out = (args.out or ROOT / 'out/分层防退化检查' / (stamp() + '_' + plan['level'])).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / '分层检查计划.json', plan)
    before = watch_sources()
    static = document_links(files, baseline)
    result = dict(scope=plan['level'], release_eligible=False, gate_passed=False,
                  document_links=static, unit_tests=None, control_smoke=None,
                  inference_pages=0, check_passed=not static['missing'], independent_validation=False)
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', HF_HUB_OFFLINE='1')
    if plan['level'] != 'static' and result['check_passed']:
        with (out / '工程测试日志.txt').open('w') as log:
            code = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'],
                                  cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        result['unit_tests'] = dict(returncode=code, log='工程测试日志.txt')
        result['check_passed'] = code == 0
        controls = ('tools/regression_', 'tools/daily_check.py', 'tools/control_', 'config/regression_')
        if result['check_passed'] and any(name.startswith(controls) for name in files):
            with (out / '管控链路自检日志.txt').open('w') as log:
                code = subprocess.run([sys.executable, 'tools/verify_control_chain.py'], cwd=ROOT,
                                      env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            result['control_smoke'] = dict(returncode=code, log='管控链路自检日志.txt',
                                          note='仅空白夹具，不能作为真实图片质量成绩')
            result['check_passed'] = code == 0
    if watch_sources() != before:
        result.update(check_passed=False, error='检查期间源码或配置发生变化，本轮不能作为单一版本结果')
    if result['check_passed'] and plan['level'] == 'full':
        # Preflight engineering checks precede the unchanged full gate.
        result.update(check_passed=False, phase='工程预检通过，完整图像对照尚未完成')
        save(out / '分层检查结果.json', result)
        write_daily_report(out, plan, result)
        code = control_main(['check', '--baseline', str(baseline), '--out', str(out / '完整候选记录')])
        result.update(check_passed=code == 0, gate_passed=code == 0, release_eligible=code == 0,
                      full_gate_returncode=code, comparison='完整候选记录_逐张对照/项目防退化控制报告.html',
                      inference_pages=plan['page_mode_count'])
        save(out / '分层检查结果.json', result)
        write_daily_report(out, plan, result)
        return code
    if result['check_passed'] and plan['mode_case_ids']:
        # Validate current data only when real-image inference is actually needed.
        from tools.control_inventory import build_inventory
        current = build_inventory(ROOT, ROOT / 'data/应用测试输入')
        from tools.regression_guard import canonical
        if canonical(current) != old['inventory_sha256']:
            raise ValueError('输入或参考清单改变，不能继续沿用旧基线比较')
        candidate = out / '局部候选记录'
        prepare(candidate, current, plan['level'], plan)
        execute(candidate)
        comparison = compare(candidate, baseline, out / '逐张对照')
        result.update(check_passed=comparison['check_passed'], inference_pages=plan['page_mode_count'],
                      comparison='逐张对照/项目防退化控制报告.html',
                      failed_pages=comparison['failed_pages'], review_pages=comparison['review_pages'])
    save(out / '分层检查结果.json', result)
    write_daily_report(out, plan, result)
    print(out / '分层防退化检查报告.html', flush=True)
    return 0 if result['check_passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'分层检查阻断：{exc}', file=sys.stderr)
        raise SystemExit(2)
