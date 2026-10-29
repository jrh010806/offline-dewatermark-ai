"""Explicit user-requested sample removal; never used by image inference."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from tools.regression_guard import canonical

PROFILE = Path(__file__).resolve().parents[1] / 'config/regression_retirements.json'


def load_profile(path=PROFILE):
    return json.loads(path.read_text()) if path.is_file() else None


def project_baseline(run, inventory, profile, baseline_seal):
    """Project verified historical evidence onto the explicitly retained scope.

    The caller must still compare inventory/runtime/settings and all retained pages.
    Original sealed files and their baseline pointer are never rewritten.
    """
    if not profile:
        return run, inventory
    removed = {r['inventory_id']: r for r in profile['cases']}
    present = {r['id']: r for r in inventory['cases'] if r['id'] in removed}
    if not present:
        return run, inventory  # A later baseline already uses the active scope.
    if set(present) != set(removed) or baseline_seal != profile['baseline_seal']:
        raise ValueError('样本移除记录与历史基线不匹配，不能扩大或改变比较范围')
    for key, row in present.items():
        if row['input_sha256'] != removed[key]['input_sha256']:
            raise ValueError(f'移除样本哈希不一致：{key}')
    active = deepcopy(inventory)
    active['cases'] = [r for r in active['cases'] if r['id'] not in removed]
    active.update(groups=dict(Counter(r['group'] for r in active['cases'])),
                  total_inputs=len(active['cases']),
                  total_pages=sum(r['page_count'] for r in active['cases']),
                  unique_input_hashes=len({r['input_sha256'] for r in active['cases']}))
    observed = deepcopy(run)
    removed_keys = {r['key'] for r in observed['records'] if r['id'] in removed}
    observed['records'] = [r for r in observed['records'] if r['id'] not in removed]
    observed['expected_keys'] = [key for key in observed['expected_keys'] if key not in removed_keys]
    observed['inventory_sha256'] = canonical(active)
    observed['sample_scope_revision'] = dict(version=profile['version'], removed_ids=sorted(removed),
        source_baseline_seal=baseline_seal, note='用户明确移除三张手写样本；保留样本沿用原封存结果，不代表重新推理或效果提升')
    return observed, active
