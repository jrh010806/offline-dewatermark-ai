"""Check per-case, predeclared quality gates; no average can hide a regression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_case(candidate: dict, baseline: dict | None, limits: dict) -> list[str]:
    if candidate.get('status') != 'completed':
        return ['运行失败']
    reasons = []
    if candidate.get('outside_selection_changed_pixels', 0):
        reasons.append('选区外发生改动（须同时核对是否为基线已有问题）')
    if baseline is None:
        return reasons + ['缺少基线，不能判断退化']
    if baseline.get('status') == 'error':
        return reasons + ['基线已失败，不能证明未退化']
    for key, limit, label in (
        ('recall', limits['recall_drop'], '召回率下降超过预定门槛'),
        ('precision', limits['precision_drop'], '精确率下降超过预定门槛'),
    ):
        new, old = candidate.get(key), baseline.get(key)
        if new is not None and old is not None and old-new > limit+1e-9:
            reasons.append(label)
    increase = candidate['false_positive_pixels']-baseline['false_positive_pixels']
    absolute, relative = limits['false_positive_increase_pixels_and_ratio']
    if increase > absolute and increase > baseline['false_positive_pixels']*relative:
        reasons.append('多选背景增加超过预定门槛')
    for key, limit, label in (
        ('background_mae', limits['background_mae_increase'], '背景误差增加超过预定门槛'),
        ('watermark_mae', limits['watermark_mae_increase'], '水印区误差增加超过预定门槛'),
    ):
        new, old = candidate.get(key), baseline.get(key)
        if new is not None and old is not None and new-old > limit+1e-9:
            reasons.append(label)
    if (candidate.get('negative') and baseline.get('selected_pixels') == 0
            and baseline.get('changed_pixels') == 0
            and (candidate['selected_pixels'] or candidate['changed_pixels'])):
        reasons.append('原本安全的负例新增选区或修改')
    if candidate.get('hole_selected_pixels', 0) > baseline.get('hole_selected_pixels', 0):
        reasons.append('字洞误选增加')
    if candidate.get('manual_repair_regression'):
        reasons.append('人工复核确认实际修复退化，选区指标改善不能抵消')
    return reasons


def evaluate(records: list[dict], limits: dict) -> dict:
    checks = [{'id': row['id'], 'label': row.get('label'),
               'failures': check_case(row, row.get('baseline'), limits)} for row in records]
    return {'quality_regression_gate_passed': not any(row['failures'] for row in checks),
            'cases': checks,
            'note': '仅检查质量退化门槛。通过也不代表修复达标或完成独立泛化验证。'}


def main() -> int:
    parser = argparse.ArgumentParser(description='逐张检查笔画优化的质量退化门槛')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--limits', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    limits = json.loads(args.limits.read_text())['per_case_regression_limits']
    result = evaluate(records, limits)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    failed = sum(bool(row['failures']) for row in result['cases'])
    print(f'质量退化门槛：{len(records)} 张，{failed} 张存在问题；完整结果：{args.out}')
    return int(not result['quality_regression_gate_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
