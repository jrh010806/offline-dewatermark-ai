"""Run the isolated stroke candidate through the real source-only worker."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description='隔离笔画候选跨样式探针；源图任务，不改变默认入口')
    ap.add_argument('--task', type=Path, required=True)
    ap.add_argument('--candidate', action='store_true')
    ap.add_argument('--candidate-edges', action='store_true')
    args = ap.parse_args()
    task = json.loads(args.task.read_text())
    if set(task) != {'mode', 'inputs', 'settings'}:
        ap.error('只允许源图任务字段')
    if (args.task.parent/'inference.json').exists():
        ap.error('目录已有结果，不得覆盖')
    for row in task['inputs']:
        if set(row) != {'id', 'input', 'input_sha256', 'dpi', 'page_count'}:
            ap.error('不允许参考或标注进入推理')
        if hashlib.sha256(Path(row['input']).read_bytes()).hexdigest() != row['input_sha256']:
            ap.error('输入哈希改变')
    from tools import control_worker
    from tools.experimental_mask_contract import enabled
    observations = []
    sys.argv = [str(ROOT/'tools/control_worker.py'), str(args.task)]
    active = args.candidate or args.candidate_edges
    with enabled(observations, edge_support=args.candidate_edges) if active else nullcontext():
        code = control_worker.main()
    (args.task.parent/'候选源图记录.json').write_text(json.dumps(
        dict(enabled=active, edge_support=args.candidate_edges, observations=observations, reference_read=False),
        ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
