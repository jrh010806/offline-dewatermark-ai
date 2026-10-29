"""Select daily checks without weakening the full release gate or batch context."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import subprocess

from tools.regression_guard import canonical, digest


def changed_files(root: Path, baseline: dict, explicit=()):
    changes = set(explicit)
    # Compare every application source to the actual observation snapshot,
    # including pre-existing uncommitted work and newly added modules.
    application = {str(p.relative_to(root)) for p in (root / 'wm').rglob('*.py')}
    application |= {p for p in baseline['code_hashes'] if p.startswith('wm/') and p.endswith('.py')}
    for name in application:
        path = root / name
        if not path.is_file() or digest(path) != baseline['code_hashes'].get(name):
            changes.add(name)
    tracked = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '-z'], cwd=root).decode().split('\0')
    others = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    for name in tracked + others:
        if not name or name.startswith(('out/', 'data/', 'proposal/', '.chart-data-', '.kimi-code/')):
            continue
        path = root / name
        if name.startswith('wm/'):
            continue  # Already checked against the source snapshot, not HEAD.
        if name in baseline['code_hashes'] and path.is_file() and digest(path) == baseline['code_hashes'][name]:
            continue
        if path.suffix.lower() in ('.py', '.sh', '.json', '.md', '.html', '.mjs', '.js', '.toml', '.txt') or name == 'start.sh':
            changes.add(name)
    return sorted(changes)


def minimum_level(files, rules):
    level, reasons, styles = 'static', [], set()
    ranks = {name: i for i, name in enumerate(rules['levels'])}
    for name in files:
        if name in rules['cross_files'] or name.startswith(('requirements', 'models/')):
            current, reason = 'cross', '涉及公共处理、默认参数或运行环境，检查跨样式与模式差异'
        elif name in rules['target_styles']:
            current, reason = 'targeted', '检查相关样式、固定保护样本及模式探针'
            styles.update(rules['target_styles'][name])
        elif name in rules['unit_files'] or name.startswith(('tests/', 'tools/', 'config/')):
            current, reason = 'unit', '工程与控制工具先跑相关测试，不重复全量图片'
        elif name.startswith('wm/'):
            current, reason = 'cross', '未知应用模块按公共影响处理，不能默认当作文档'
        elif name.endswith(('.md', '.html', '.txt')):
            current, reason = 'static', '文档或展示文件，检查新增本地链接与内容'
        else:
            current, reason = 'unit', '未明确归类的工程变更，保守运行工程测试'
        if ranks[current] > ranks[level]:
            level = current
        reasons.append(dict(file=name, level=current, reason=reason))
    return level, reasons, sorted(styles)


def expand_batches(seeds, cases):
    index = {r['id']: r for r in cases}
    missing = set(seeds) - set(index)
    if missing:
        raise ValueError(f'未登记的目标或保护样本：{sorted(missing)}；须先补可比基线，不能默默跳过')
    batches = {index[name]['batch'] for name in seeds}
    return [r['id'] for r in cases if r['batch'] in batches]


def make_plan(files, inventory, baseline, rules, requested='auto', targets=()):
    minimum, reasons, styles = minimum_level(files, rules)
    ranks = {name: i for i, name in enumerate(rules['levels'])}
    if requested != 'auto' and ranks[requested] < ranks[minimum]:
        raise ValueError(f'变更至少需要 {minimum}，不能降级为 {requested}')
    level = minimum if requested == 'auto' else requested
    if targets and ranks[level] < ranks['targeted']:
        level = 'targeted'
    cases = inventory['cases']
    if level == 'full':
        selections = {m: [r['id'] for r in cases] for m in baseline['profiles']}
    elif level in ('targeted', 'cross'):
        fixed = [r for r in cases if r['group'] == '固定回归']
        affected = [r['id'] for r in fixed if r['style'] in styles]
        seeds = list(rules['protection_seeds']) + list(targets) + affected
        probes = rules['cross_mode_probes' if level == 'cross' else 'targeted_mode_probes']
        if level == 'cross':
            seeds += [r['id'] for r in fixed] + list(probes)
        selections = {'desktop': expand_batches(seeds, cases)}
        selections.update({m: expand_batches(probes, cases) for m in ('desktop_protected', 'cli')})
    else:
        selections = {}
    expected = [f"{mode}:{r['id']}:{page}" for mode, names in selections.items()
                for r in cases if r['id'] in set(names) for page in range(1, r['page_count'] + 1)]
    baseline_rows = {r['key']: r for r in baseline['records']}
    if set(expected) - set(baseline_rows):
        raise ValueError('选中的样本在当前基线没有对应页次，不能跳过缺基线问题')
    infer_ms = sum(baseline_rows[key].get('elapsed_ms', 0) for key in expected)
    jobs = sum(len({r['batch'] for r in cases if r['id'] in names}) for names in selections.values())
    return dict(schema_version=1, level=level, minimum_level=minimum, files=files, reasons=reasons,
                affected_styles=styles, target_cases=list(targets), mode_case_ids=selections,
                expected_keys=expected, page_mode_count=len(expected), batch_count=jobs,
                estimated_seconds=round(infer_ms / 1000 + jobs * 3 + (30 if level != 'static' else 0)),
                estimate_note='根据原基线逐页耗时加批次启动与工程检查估算，不是速度承诺',
                tiers_version=rules['version'], tiers_sha256=canonical(rules),
                release_eligible=False, independent_validation=False,
                note='日常局部检查不代表完整回归或质量达标；默认版本更新仍须单独完整检查')


def selected_cases(inventory, run):
    plan = run.get('selection_plan')
    if plan:
        names = {name for selected in plan['mode_case_ids'].values() for name in selected}
        return [r for r in inventory['cases'] if r['id'] in names]
    return [r for r in inventory['cases'] if run['scope'] == 'all' or r['group'] == '01_快速20张']


def jobs_for(inventory, run):
    cases = selected_cases(inventory, run)
    result = []
    for mode in run['profiles']:
        names = set(run.get('selection_plan', {}).get('mode_case_ids', {}).get(mode, [r['id'] for r in cases]))
        groups = defaultdict(list)
        for row in cases:
            if row['id'] in names:
                groups[row['batch']].append(row)
        result.extend((mode, batch, rows) for batch, rows in groups.items())
    return result
