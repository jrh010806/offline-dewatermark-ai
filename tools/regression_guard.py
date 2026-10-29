"""Fail-closed comparison of sealed inference records, never aggregate away damage."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def compare_page(new: dict, old: dict | None, policy: dict) -> dict:
    failures, review, known = [], [], []
    if new.get('status') != 'completed':
        return dict(failures=['候选运行失败或缺页'], review=[], known=[])
    if old is None or old.get('status') != 'completed':
        return dict(failures=['基线缺失或运行失败，无法证明未退化'], review=[], known=[])
    for key in ('input_sha256', 'reference_identity', 'settings'):
        if key not in new or key not in old or new[key] != old[key]:
            failures.append(f'比较条件不一致：{key}')
    limits = policy['limits']
    for key, threshold in (
        ('recall', 'recall_drop'), ('strong_recall', 'strong_recall_drop'),
        ('precision', 'precision_drop'),
    ):
        a, b = new.get(key), old.get(key)
        if b is not None and (a is None or not math.isfinite(a)):
            failures.append(f'{key} 丢失，不能跳过原可评估指标')
        elif a is not None and b is not None and b - a > limits[threshold] + 1e-9:
            failures.append(f'{key} 下降超过预定门槛')
    for key, threshold in (
        ('false_positive_pixels', 'false_positive_increase_pixels'),
        ('hole_selected_pixels', 'hole_selected_increase_pixels'),
        ('background_mae', 'background_mae_increase'),
        ('watermark_mae', 'watermark_mae_increase'),
    ):
        a, b = new.get(key), old.get(key)
        if b is not None and (a is None or not math.isfinite(a)):
            failures.append(f'{key} 丢失，不能跳过原可评估指标')
        elif a is not None and b is not None and a - b > limits[threshold] + 1e-9:
            failures.append(f'{key} 增加超过预定门槛')
    for key in ('outside_selection_changed_pixels', 'outside_truth_changed_pixels'):
        a, b = new.get(key), old.get(key)
        if b is not None:
            if a is None or a > b:
                failures.append(f'{key} 新增越界改动')
            if b:
                known.append(f'基线已有 {key}={b}；本版 {a}')
    if new.get('negative'):
        safe = old.get('selected_pixels') == 0 and old.get('changed_pixels') == 0
        if safe and (new.get('selected_pixels') != 0 or new.get('changed_pixels') != 0):
            failures.append('原本安全的无水印负例新增选区或改动')
        if not safe:
            known.append('基线负例已经存在误选或改动')
    elapsed, before = new.get('elapsed_ms'), old.get('elapsed_ms')
    if elapsed is None or before is None:
        failures.append('缺少耗时')
    else:
        if elapsed > limits['time_budget_ms'] >= before:
            failures.append('新增单页超时')
        if elapsed - before > max(limits['elapsed_increase_ms'], before * limits['elapsed_increase_ratio']):
            failures.append('耗时增加超过预定门槛')
        if before > limits['time_budget_ms']:
            known.append('基线已经超过单页时间预算')
    for key in ('output_sha256', 'mask_sha256'):
        if not new.get(key) or not old.get(key):
            failures.append(f'缺少实际产物：{key}')
        elif new[key] != old[key] and policy['changed_output_requires_review']:
            review.append(f'{key} 变化，需核对实际图像')
    for key in ('method', 'repair_mode', 'repairs'):
        if key not in new or key not in old:
            failures.append(f'缺少分流记录：{key}')
        elif new[key] != old[key] and policy['route_change_requires_review']:
            review.append(f'{key} 改变，需复核是否切错路径')
    if new.get('manual_repair_regression'):
        failures.append('人工确认背景、正文或修复效果退化')
    return dict(failures=failures, review=review, known=known)


def compare_runs(new: dict, old: dict, policy: dict) -> dict:
    blockers = []
    for key in ('inventory_sha256', 'policy_sha256', 'profiles', 'scope'):
        if key not in new or new.get(key) != old.get(key):
            blockers.append(f'运行条件不一致：{key}')
    if new.get('scope') != 'all':
        blockers.append('仅局部检查，不能作为完整防退化验收')
    if set(new.get('profiles', {})) != set(policy['required_modes']):
        blockers.append('未覆盖全部实际使用模式')
    for run, title in ((new, '候选'), (old, '基线')):
        if not run.get('complete') or not run.get('integrity_verified'):
            blockers.append(f'{title}未完整运行或完整性校验失败')
    def index(run):
        records = run.get('records', [])
        values = {r['key']: r for r in records}
        if len(values) != len(records):
            blockers.append('存在重复结果键')
        if set(values) != set(run.get('expected_keys', [])):
            blockers.append('预期清单与实际结果不一致')
        return values
    n, b = index(new), index(old)
    if not n or not b:
        blockers.append('空结果不能通过防退化验收')
    if set(n) != set(b):
        blockers.append('基线与候选图片/页/模式不齐全')
    checks = []
    for key in sorted(set(n) | set(b)):
        row = n.get(key, {'status': 'missing'})
        checks.append(dict(key=key, style=row.get('style'), mode=row.get('mode'),
                           **compare_page(row, b.get(key), policy)))
    # Dependency/model changes need explicit visual review even when hashes match.
    if (not new.get('runtime_identity') or not old.get('runtime_identity') or
            new.get('runtime_identity') != old.get('runtime_identity')):
        blockers.append('依赖、模型或推理后端改变，需要另行验证运行环境')
    failed = sum(bool(r['failures']) for r in checks)
    pending = sum(bool(r['review']) for r in checks)
    groups = {}
    for row in checks:
        group = groups.setdefault(f"{row['mode']} / {row['style']}", dict(total=0, failed=0, review=0))
        group['total'] += 1
        group['failed'] += bool(row['failures'])
        group['review'] += bool(row['review'])
    passed = not blockers and not failed and not pending
    return dict(gate_passed=passed, check_passed=passed, scope='all', release_eligible=passed,
                blockers=blockers, failed_pages=failed,
                review_pages=pending, cases=checks, by_style_and_mode=groups,
                independent_validation=False, quality_approved=False,
                note='只判断相对基线的退化与变化；通过不代表去水印达标或独立泛化验证。')


def compare_subset(new: dict, old: dict, policy: dict, inventory: dict, rules: dict) -> dict:
    """Daily evidence may pass locally; it can never authorize baseline promotion."""
    from tools.regression_layers import make_plan
    blockers = []
    plan = new.get('selection_plan', {})
    if plan.get('level') not in ('targeted', 'cross'):
        raise ValueError('局部对照只接受明确的定向或跨样式计划')
    required = make_plan(plan['files'], inventory, old, rules, plan['level'], plan['target_cases'])
    for key in ('expected_keys', 'mode_case_ids', 'tiers_sha256'):
        if plan.get(key) != required[key]:
            blockers.append(f'局部范围与预定计划不一致：{key}')
    if set(new.get('expected_keys', [])) != set(required['expected_keys']):
        blockers.append('缺少预先固定的目标、保护样本或模式探针')
    for key in ('inventory_sha256', 'policy_sha256', 'runtime_identity'):
        if not new.get(key) or new.get(key) != old.get(key):
            blockers.append(f'比较条件不一致：{key}')
    for version, title in ((new, '候选'), (old, '基线')):
        if not version.get('complete') or not version.get('integrity_verified'):
            blockers.append(f'{title}未完整记录或完整性失败')
    for mode in required['mode_case_ids']:
        if new.get('profiles', {}).get(mode) != old.get('profiles', {}).get(mode):
            blockers.append(f'{mode} 参数不一致')
    source_paths = {p for run in (new, old) for p in run.get('code_hashes', {})
                    if p.startswith('wm/') and p.endswith('.py')}
    changed = {p for p in source_paths if new.get('code_hashes', {}).get(p) != old.get('code_hashes', {}).get(p)}
    if changed - set(plan['files']):
        blockers.append('计划遗漏实际改变的应用源码，不能靠 --files 隐藏影响')
    rows = new.get('records', [])
    n = {r['key']: r for r in rows}
    b = {r['key']: r for r in old.get('records', [])}
    if len(n) != len(rows) or set(n) != set(required['expected_keys']) or not n:
        blockers.append('实际结果缺页、多页、重复或为空')
    checks = [dict(key=key, mode=n.get(key, {}).get('mode'), style=n.get(key, {}).get('style'),
                   **compare_page(n.get(key, {'status': 'missing'}), b.get(key), policy))
              for key in sorted(set(required['expected_keys']) | set(n))]
    groups = {}
    for row in checks:
        group = groups.setdefault(f"{row['mode']} / {row['style']}", dict(total=0, failed=0, review=0))
        group['total'] += 1
        group['failed'] += bool(row['failures'])
        group['review'] += bool(row['review'])
    failed = sum(bool(r['failures']) for r in checks)
    pending = sum(bool(r['review']) for r in checks)
    passed = not blockers and not failed and not pending
    return dict(scope=plan['level'], check_passed=passed, gate_passed=False, release_eligible=False,
                blockers=blockers, failed_pages=failed, review_pages=pending, cases=checks,
                by_style_and_mode=groups, independent_validation=False, quality_approved=False,
                note='只检查本次固定范围。局部通过不代表完整回归通过，不能用于更新默认基线。')


def apply_reviews(result: dict, reviews: dict, candidate_seal: str, baseline_seal: str) -> dict:
    """Reviews can resolve visual changes, never numerical failures or missing cases."""
    import copy
    result = copy.deepcopy(result)
    if reviews.get('candidate_seal') != candidate_seal or reviews.get('baseline_seal') != baseline_seal:
        raise ValueError('复核记录不是针对这两个封存版本')
    rows = reviews.get('cases', [])
    index = {r['key']: r for r in rows}
    if len(rows) != len(index):
        raise ValueError('复核记录存在重复键')
    expected = {r['key'] for r in result['cases'] if r['review']}
    if set(index) - expected:
        raise ValueError('复核记录包含不属于本轮变化的图片')
    for check in result['cases']:
        row = index.get(check['key'], {})
        if row.get('decision') == 'regression':
            check['failures'].append('人工确认修复退化')
        if (check['review'] and row.get('decision') == 'accept_change' and
                str(row.get('reviewer', '')).strip() and str(row.get('reason', '')).strip()):
            check['review_resolution'] = row
            check['review'] = []
    result['failed_pages'] = sum(bool(r['failures']) for r in result['cases'])
    result['review_pages'] = sum(bool(r['review']) for r in result['cases'])
    result['check_passed'] = not result['blockers'] and not result['failed_pages'] and not result['review_pages']
    result['gate_passed'] = result['check_passed'] and result.get('scope', 'all') == 'all'
    result['release_eligible'] = result['gate_passed']
    for label in result['by_style_and_mode']:
        mode, style = label.split(' / ', 1)
        rows = [r for r in result['cases'] if r['mode'] == mode and str(r['style']) == style]
        result['by_style_and_mode'][label].update(failed=sum(bool(r['failures']) for r in rows),
                                                review=sum(bool(r['review']) for r in rows))
    return result
