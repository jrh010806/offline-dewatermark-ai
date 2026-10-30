"""Source-only probe for an explicitly enabled candidate; never passes references."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description='自动效果比较独立验证入口')
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    if (args.task.parent / 'inference.json').exists():
        parser.error('目录已有结果；必须使用新目录保留每轮记录')
    task = json.loads(args.task.read_text())
    if set(task) != {'mode', 'inputs', 'settings'}:
        parser.error('任务只允许源图、模式和参数')
    for row in task['inputs']:
        if set(row) != {'id', 'input', 'input_sha256', 'dpi', 'page_count'}:
            parser.error('输入中包含源图字段以外的数据')
        if hashlib.sha256(Path(row['input']).read_bytes()).hexdigest() != row['input_sha256']:
            parser.error('输入校验不一致')
    from wm import pipeline
    from wm.batches import BatchWorker
    from tools import control_worker
    sys.argv = [str(ROOT / 'tools/control_worker.py'), str(args.task)]
    if not args.candidate:
        return control_worker.main()
    original = BatchWorker.__init__

    def initialise(worker, jobs, parent=None):
        original(worker, [tuple(job[:6]) + (True,) for job in jobs], parent)

    with patch.object(BatchWorker, '__init__', initialise), patch.object(
            pipeline, 'process_file', partial(pipeline.process_file, automatic_selection=True)):
        return control_worker.main()


if __name__ == '__main__':
    raise SystemExit(main())
