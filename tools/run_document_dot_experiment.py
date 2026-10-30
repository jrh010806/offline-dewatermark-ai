"""Explicit source-only research worker; task JSON must not contain references.

Usage: python tools/run_document_dot_experiment.py --task task.json [--candidate]
Run from a frozen code directory.  The unchanged application worker supplies
real BatchWorker processing and complete per-task batch context.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='文档错位点阵独立实验；不改变默认应用')
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if ((args.task.parent/'inference.json').exists()
            or (args.task.parent/'候选源图证据.json').exists()):
        parser.error('该目录已有实验结果；请复制任务到新目录，禁止覆盖旧轮次。')
    task = json.loads(args.task.read_text())
    assert set(task) == {'mode', 'inputs', 'settings'}, '只接受源图任务'
    for item in task['inputs']:
        assert set(item) == {'id', 'input', 'input_sha256', 'dpi', 'page_count'}
        assert hashlib.sha256(Path(item['input']).read_bytes()).hexdigest() == item['input_sha256']
    from tools import control_worker
    from tools.experimental_document_dots import enabled
    trace = []
    sys.argv = [str(ROOT/'tools/control_worker.py'), str(args.task)]
    with enabled(trace) if args.candidate else nullcontext():
        code = control_worker.main()
    (args.task.parent/'候选源图证据.json').write_text(json.dumps(
        {'enabled': args.candidate, 'observations': trace, 'reference_read': False},
        ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
