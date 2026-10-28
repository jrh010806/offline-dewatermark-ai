"""Blend AI-generated structure into an existing selection without changing other pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg, spsolve


def blend_generated_background(original: np.ndarray, generated: np.ndarray,
                               selection: np.ndarray) -> np.ndarray:
    """Use generated gradients inside selection and original colors on its boundary.

    This is a composition step, not a new background generator. The generated
    image must already be aligned; this cannot correct an AI-redrawn object.
    """
    if original.shape != generated.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("原图与生成图必须为同尺寸 RGB 图像。")
    if selection.shape != original.shape[:2]:
        raise ValueError("选区尺寸必须与原图一致。")
    if original.dtype != np.uint8 or generated.dtype != np.uint8:
        raise ValueError("合成输入必须为 uint8 RGB。")
    mask = selection.astype(bool)
    output = original.copy()
    if not mask.any():
        return output
    if mask.all():
        return generated.copy()

    y, x = np.where(mask)
    count = len(y)
    ids = np.full(mask.shape, -1, dtype=np.int32)
    ids[y, x] = np.arange(count)
    rows, cols, values = [], [], []
    rhs = np.zeros((count, 3), dtype=np.float64)
    degree = np.zeros(count, dtype=np.float64)
    gen = generated.astype(np.float64)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        valid = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
        p = np.flatnonzero(valid)
        ny, nx = ny[valid], nx[valid]
        degree[p] += 1
        rhs[p] += gen[y[p], x[p]] - gen[ny, nx]
        q = ids[ny, nx]
        selected = q >= 0
        rows.append(p[selected])
        cols.append(q[selected])
        values.append(-np.ones(int(selected.sum())))
        rhs[p[~selected]] += original[ny[~selected], nx[~selected]]
    rows.append(np.arange(count))
    cols.append(np.arange(count))
    values.append(degree)
    matrix = coo_matrix((np.concatenate(values),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(count, count)).tocsc()
    if count <= 150_000:
        solution = spsolve(matrix, rhs)
    else:
        # Avoid a large direct factorization for a dense document selection.
        preconditioner = diags(1.0 / degree)
        channels = []
        for c in range(3):
            result, info = cg(matrix, rhs[:, c], M=preconditioner,
                              rtol=1e-6, atol=1e-4, maxiter=2000)
            if info != 0:
                raise ValueError("选区融合未收敛，保留生成原图供检查，未输出合成结果。")
            channels.append(result)
        solution = np.column_stack(channels)
    if not np.isfinite(solution).all():
        raise ValueError("融合结果包含非有限数值，未输出合成结果。")
    output[y, x] = np.clip(solution, 0, 255).round().astype(np.uint8)
    return output


def compose_files(original_path: Path, generated_path: Path, selection_path: Path,
                  output_path: Path) -> dict:
    if output_path.exists():
        raise ValueError("输出文件已经存在，请选择新文件名。")
    original = ImageOps.exif_transpose(Image.open(original_path)).convert("RGB")
    generated = ImageOps.exif_transpose(Image.open(generated_path)).convert("RGB")
    selection = Image.open(selection_path).convert("L")
    if selection.size != original.size:
        raise ValueError("选区尺寸不一致。")
    generated_size = generated.size
    ratio_change = (generated.width / generated.height) / (original.width / original.height)
    if abs(ratio_change - 1) > 0.01:
        raise ValueError("生成图纵横比发生变化，需要先检查构图和对齐，未自动拉伸。")
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.Resampling.LANCZOS)
    source = np.asarray(original)
    mask = np.asarray(selection) >= 128
    result = blend_generated_background(source, np.asarray(generated), mask)
    changed = np.any(result != source, axis=2)
    outside_changed = int(np.count_nonzero(changed & ~mask))
    if outside_changed:
        raise AssertionError("选区外像素发生变化，未保存结果。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path, format="PNG")
    report = {
        "original": str(original_path.resolve()),
        "generated": str(generated_path.resolve()),
        "selection": str(selection_path.resolve()),
        "result": str(output_path.resolve()),
        "original_size": list(original.size), "generated_size": list(generated_size),
        "selection_pixels": int(mask.sum()),
        "changed_inside_selection": int(np.count_nonzero(changed & mask)),
        "changed_outside_selection": outside_changed,
        "outside_selection_identical": outside_changed == 0,
        "method": "generated_gradients_with_original_boundary",
        "quality_verified": False,
        "note": "AI 重建提供内部结构，原图边界约束颜色融合。只保证选区外不变，不保证内容恢复正确。",
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compose_files(args.original, args.generated, args.selection, args.out),
                     ensure_ascii=False, indent=2))
