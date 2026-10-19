"""JPEG 空基线（null floor）测量：把"我们没擦干净"和"JPEG 自己弄坏的"分开。

    python ../tools/jpeg_floor.py            # 相对 Desktop 项目根目录运行
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm import masks as mask_mod
from wm import pipeline, repair
from wm.cli import load_doc
from wm.contract import MaskLayer, Thresholds
from wm.synth import degrade
import random


def floor_for(doc_dir: Path) -> dict:
    pages, gt = load_doc(doc_dir, dpi=200)
    thr = Thresholds()
    results = pipeline.process_pages(pages, thr, gt=gt)
    rows = []
    for page, result in zip(pages, results):
        clean, alpha, (s_vec, t_vec) = gt[page.key]
        source = page.image
        codec = degrade(clean, random.Random(1), jpeg_q=75, noise=0)
        oracle = repair.alpha_unmix(source, MaskLayer(
            alpha=alpha.astype(np.float32), veil_strength=s_vec, veil_offset=t_vec))
        rows.append({
            "codec_mean": float(np.abs(codec.astype(np.int16) - clean.astype(np.int16)).mean()),
            "codec_carried": float((np.abs(codec.astype(np.int16) - clean.astype(np.int16)).mean(2)
                                    >= 8.0).mean()),
            "resid_identity": mask_mod.residual_visible_fraction(source, source, clean),
            "resid_ours": mask_mod.residual_visible_fraction(source, result.output.image, clean),
            "resid_oracle": mask_mod.residual_visible_fraction(source, oracle, clean),
        })
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/syn_jpeg")
    print(f"{'doc':22} {'codec误差':>9} {'超阈值像素':>10} {'不动它':>8} "
          f"{'完美掩码':>8} {'我们':>8}")
    for d in sorted(root.glob("doc*")):
        f = floor_for(d)
        print(f"{d.name:22} {f['codec_mean']:9.2f} {f['codec_carried']*100:9.2f}% "
              f"{f['resid_identity']*100:7.1f}% {f['resid_oracle']*100:7.1f}% "
              f"{f['resid_ours']*100:7.1f}%")
