"""Exercise freeze -> real worker -> seal -> compare on an explicit blank fixture.

This is a controller smoke test, never a watermark quality/generalization score.
It does not create or replace the user's baseline pointer.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PIL import Image
from tools.regression_control import compare, execute, load, prepare, stamp
from tools.regression_guard import canonical, digest
from tools.regression_layers import make_plan


def main():
    out = ROOT / 'out/项目防退化控制' / (stamp() + '_管控链路自检')
    out.mkdir(parents=True, exist_ok=False)
    source = out / '自检空白输入.png'
    Image.new('RGB', (128, 128), 'white').save(source)
    rules = load(ROOT / 'config/regression_tiers.json')
    names = list(dict.fromkeys(rules['protection_seeds'] + rules['targeted_mode_probes'] + rules['cross_mode_probes']))
    # IDs mirror the selection contract only; every input here is the same blank fixture.
    records = [dict(id=name, input=str(source), input_sha256=digest(source),
                  batch='工具自检' if name != 'control_fixture_extra' else '未选中夹具批次',
                  group='工具自检', style='无水印夹具', negative=True,
                  references=[{}], reference_identity=canonical([{}]), page_count=1, dpi=200)
               for name in names + ['control_fixture_extra']]
    inventory = dict(schema_version=1, cases=records, total_inputs=len(records), total_pages=len(records),
                     independent_validation=False, note='管控链路夹具，不是用户样本或质量评估')
    prepare(out / '冻结自检', inventory, 'all')
    run = execute(out / '冻结自检')
    if not run['complete'] or any(r['status'] != 'completed' for r in run['records']):
        raise RuntimeError('管控链路自检有失败，禁止启动正式基线')
    result = compare(out / '冻结自检', out / '冻结自检', out / '同版本对照自检')
    if not result['gate_passed']:
        raise RuntimeError('相同封存版本未能通过管控对照')
    application = {str(p.relative_to(ROOT)) for p in (ROOT / 'wm').rglob('*.py')}
    application |= {p for p in run['code_hashes'] if p.startswith('wm/') and p.endswith('.py')}
    changed = [name for name in sorted(application) if not (ROOT / name).is_file()
               or digest(ROOT / name) != run['code_hashes'].get(name)]
    plan = make_plan(['tools/daily_check.py', *changed], inventory, run, rules, 'cross')
    plan['baseline_seal'] = digest(out / '冻结自检/封存校验.json')
    prepare(out / '局部自检', inventory, 'cross', plan)
    partial = execute(out / '局部自检')
    local = compare(out / '局部自检', out / '冻结自检', out / '局部对照自检')
    if not local['check_passed'] or local['gate_passed'] or local['release_eligible']:
        raise RuntimeError('局部检查未通过，或错误获得完整放行资格')
    if len(partial['records']) >= len(run['records']):
        raise RuntimeError('局部链路未缩小到计划范围')
    print(f'管控自检通过：三种模式、冻结、实际工作进程、封存、完整同版本对照及局部不可放行。{out}')


if __name__ == '__main__':
    main()
