"""标定块效应（blocking artifact）检测量：无损页与 JPEG 页能否被分开。

JPEG 在 8x8 块边界上产生周期性不连续。只统计平坦区域的梯度，因为文字笔画的
内部梯度会淹没块边界信号。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm.cli import load_doc


def blocking_ratio(img: np.ndarray) -> float:
    g = img.astype(np.float32).mean(2)
    ratios = []
    for axis in (1, 0):
        d = np.abs(np.diff(g, axis=axis))
        quiet = d < 3.0
        n = d.shape[1] if axis == 1 else d.shape[0]
        at_boundary = ((np.arange(n) + 1) % 8 == 0)
        # select per-axis: build the boolean grid then mask with quiet
        grid = np.zeros(n, bool)
        grid[at_boundary] = True
        if axis == 1:
            sel = np.broadcast_to(grid, d.shape)
        else:
            sel = np.broadcast_to(grid[:, None], d.shape)
        b = d[sel & quiet]
        i = d[~sel & quiet]
        if b.size and i.size:
            ratios.append(float(b.mean() / max(i.mean(), 1e-3)))
    return float(np.mean(ratios)) if ratios else 1.0


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        vals = []
        for d in sorted(Path(arg).glob("doc*")):
            pages, _ = load_doc(d, dpi=200)
            per = [blocking_ratio(p.image) for p in pages]
            vals.append(np.mean(per))
            print(f"{d.name:22} {np.mean(per):.3f}  min={min(per):.3f} max={max(per):.3f}")
        print(f"  -> {arg}: {min(vals):.3f} .. {max(vals):.3f}\n")
