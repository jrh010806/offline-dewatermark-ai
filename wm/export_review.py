"""Lossless selection evidence and a portable, offline review index."""
from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import documents
from .contract import PageResult, Thresholds


def overlay_mask(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Use exactly the same alpha-weighted red overlay as the desktop viewer."""
    tint = np.array([255, 40, 40], np.float32)
    strength = (alpha[..., None] * 0.55).astype(np.float32)
    return np.clip(image.astype(np.float32) * (1 - strength) + tint * strength,
                   0, 255).astype(np.uint8)


def write_selection(result: PageResult, out: Path, name: str) -> dict:
    mask = result.mask
    threshold = result.processing.get("thresholds", {}).get(
        "mask_alpha_min", Thresholds.mask_alpha_min)
    selected = mask.binary(threshold)
    images = {
        "source": result.source.image,
        "selection_overlay": overlay_mask(result.source.image, mask.alpha),
        "mask": selected,
        "mask_alpha": np.rint(np.clip(mask.alpha, 0, 1) * 255).astype(np.uint8),
    }
    # Preserve native float values: PNG alpha quantization must not change a
    # later threshold experiment or obscure low-confidence missed strokes.
    arrays = {"alpha": mask.alpha}
    for field in ("footprint", "text_keepout", "overlay_opacity",
                  "veil_strength", "veil_offset"):
        value = getattr(mask, field)
        if value is not None:
            arrays[field] = value
    for field in ("footprint", "text_keepout"):
        if field in arrays:
            images[field] = (arrays[field].astype(bool).astype(np.uint8) * 255)
    files = {}
    for kind, image in images.items():
        files[kind] = f"{name}_{kind}.png"
        documents.save_png(out / files[kind], image)
    files["mask_data"] = f"{name}_mask_data.npz"
    np.savez_compressed(out / files["mask_data"], **arrays)
    return {
        "files": files,
        "selection": {
            "threshold": float(threshold),
            "threshold_source": "processing" if "mask_alpha_min" in
                result.processing.get("thresholds", {}) else "default",
            "selected_pixels": int(np.count_nonzero(selected)),
            "coverage": float(np.count_nonzero(selected) / selected.size),
            "width": result.source.width,
            "height": result.source.height,
            "repair_mode": mask.repair_mode,
            "overlay": "alpha-weighted red; identical to desktop selection view",
            "binary_mask": "white = alpha >= threshold; detection selection, not guaranteed repair extent",
        },
    }


def write_index(out: Path, pages: list[dict]) -> None:
    sections = []
    labels = (("source", "原始图片"), ("selection_overlay", "水印选区（红色）"),
              ("mask", "黑白选区（白色为选中）"), ("clean", "处理结果"))
    for page in pages:
        files = page["files"]
        selection = page["selection"]
        figures = []
        for kind, label in labels:
            url = escape(quote(files[kind]), quote=True)
            figures.append(f'<figure><figcaption>{label}</figcaption><a href="{url}">'
                           f'<img loading="lazy" src="{url}" alt="{label}"></a></figure>')
        links = " · ".join(f'<a href="{escape(quote(path), quote=True)}">{escape(kind)}</a>'
                           for kind, path in files.items())
        sections.append(
            f'<section><h2>{escape(page["page"])}</h2><p>'
            f'{selection["width"]} × {selection["height"]} · '
            f'定位方式：{escape(page["mask_method"])} · '
            f'判定阈值：{selection["threshold"]:.1%} · '
            f'选区占比：{selection["coverage"]:.2%}</p>'
            f'<div class="images">{"".join(figures)}</div><details>'
            f'<summary>诊断文件</summary><p>{links}</p>'
            f'<p>处理参数：{escape(str(page["processing"]))}</p></details></section>')
    (out / "水印选区与处理结果复核.html").write_text('''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>水印选区逐图复核</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f1f4f2;color:#253c35;margin:24px}
section{background:white;border-radius:12px;padding:20px;margin:24px 0}
h1{font-size:26px}h2{font-size:18px;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
figure{margin:0}figcaption{margin-bottom:8px}img{width:100%;height:auto}a{color:#16734e}
details{margin-top:16px;overflow-wrap:anywhere}@media(max-width:1000px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><h1>水印选区逐图复核</h1>
<p>逐张对照原图、选区和处理结果；点击图片查看原尺寸。原图中可见的水印没有被选中，属于漏检；已选中但结果仍有水印，需要进一步检查修复过程。</p>
<p>红色深浅与应用的「水印选区」一致；黑白图使用该次处理保存的阈值。选区不代表真实水印标注，也不保证覆盖全部水印。footprint 是修复候选范围，text_keepout 是正文保护区域，均不等同于最终修改范围；原始数值在 mask_data.npz 中。</p>
<p>统计与完整参数见 <a href="manifest.json">manifest.json</a>。复核资料完全本地保存。</p>
''' + "".join(sections) + "</html>", encoding="utf-8")
