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


def overlay_selection(image: np.ndarray, selection: np.ndarray) -> np.ndarray:
    """Uniform brush permission overlay; opacity is a display choice only."""
    return overlay_mask(image, selection.astype(np.float32)*(105/255/.55))


def write_selection(result: PageResult, out: Path, name: str) -> dict:
    mask = result.mask
    threshold = result.processing.get("thresholds", {}).get(
        "mask_alpha_min", Thresholds.mask_alpha_min)
    selected = mask.binary(threshold)
    images = {
        "source": result.source.image,
        "selection_overlay": (overlay_selection(result.source.image, mask.selection)
                              if mask.selection is not None else overlay_mask(result.source.image, mask.alpha)),
        "mask": selected,
        "mask_alpha": np.rint(np.clip(mask.alpha, 0, 1) * 255).astype(np.uint8),
    }
    # Preserve native float values: PNG alpha quantization must not change a
    # later threshold experiment or obscure low-confidence missed strokes.
    arrays = {"alpha": mask.alpha}
    if mask.selection is not None:
        arrays['selection'] = mask.selection
        if result.selection_overrides is not None:
            arrays['selection_overrides'] = result.selection_overrides
        images['actual_changes'] = np.any(result.output.image != result.source.image, axis=2).astype(np.uint8)*255
    elif result.processing.get('automatic_selection'):
        images['actual_changes'] = np.any(result.output.image != result.source.image, axis=2).astype(np.uint8)*255
    for field in ("footprint", "text_keepout", "overlay_opacity", "repeat_stroke_support",
                  "veil_strength", "veil_offset", "shadow_reconstruction"):
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
    if result.comparison_evidence:
        files['comparison_observations'] = f'{name}_原图评分观测_不是修复选区.npz'
        np.savez_compressed(out / files['comparison_observations'], **result.comparison_evidence)
    evidence = {
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
    evidence['processing_mode'] = 'human_assisted' if result.automatic_result is not None else 'automatic'
    if mask.shadow_evidence is not None:
        evidence['selection']['shadow_layers'] = mask.shadow_evidence
        if mask.repair_mode == 'shadow_layer_inverse':
            evidence['selection']['alpha_note'] = '二值笔画许可；不是测得的水印透明度'
    if result.comparison_results:
        comparisons = []
        for index, candidate in enumerate(result.comparison_results, 1):
            candidate_name = f'{name}_方法{index:02d}'
            item = write_selection(candidate, out, candidate_name)
            item['files']['clean'] = candidate_name + '_clean.png'
            documents.save_png(out / item['files']['clean'], candidate.output.image)
            item['name'] = candidate.processing['candidate_name']
            item['verdict'] = candidate.verdict.to_dict()
            comparisons.append(item)
        evidence['automatic_candidates'] = comparisons
    if mask.selection is not None:
        evidence['selection'].update(
            overlay='uniform red; complete edited selection, not physical alpha',
            binary_mask='white = complete committed selection; manual expansion is zero',
            alpha_note='仅保留自动观测强度；人工选中不代表测得水印透明度',
            selected_pixels=int(mask.selection.sum()),
            actual_changed_pixels=int(np.any(result.output.image != result.source.image, axis=2).sum()))
    if result.automatic_result is not None:
        baseline_name = name + '_自动基线'
        baseline = write_selection(result.automatic_result, out, baseline_name)
        baseline['files']['clean'] = baseline_name + '_clean.png'
        documents.save_png(out / baseline['files']['clean'], result.automatic_result.output.image)
        baseline['verdict'] = result.automatic_result.verdict.to_dict()
        baseline['processing'] = result.automatic_result.processing
        evidence['automatic_baseline'] = baseline
        steps = []
        for index, selected_step in enumerate(result.assistance_masks, 1):
            edited = result.processing['human_assistance'][index-1].get('operation') == 'edit_complete_selection'
            path = f'{name}_辅助第{index:02d}次_' + ('完整选区.png' if edited else '确认笔画.png')
            documents.save_png(out / path, selected_step.astype(np.uint8)*255)
            steps.append(dict(mask=path, **result.processing['human_assistance'][index-1]))
        evidence['assistance_steps'] = steps
    return evidence


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
        assistance = ''
        auto = page.get('processing', {}).get('automatic_selection')
        if auto:
            assistance += ('<p><b>自动比较试用：'+('未达标' if auto['status'] == 'unmet' else '待复核')
                           +'</b> '+escape(auto['reason'])
                           +'。以下仅保留实际算出的结果；未运行的方法见记录，较好候选不等于已验证成功。</p>')
            for candidate in page.get('automatic_candidates', []):
                label = '原方法' if candidate['name'] == 'original' else '笔画补全'
                for kind, suffix in [('clean', '实际结果'), ('mask', '实际选区')]:
                    url = escape(quote(candidate['files'][kind]), quote=True)
                    caption = label+' · '+suffix
                    figures.append(f'<figure><figcaption>{caption}</figcaption><a href="{url}">'
                                   f'<img loading="lazy" src="{url}" alt="{caption}"></a></figure>')
        if page.get('processing_mode') == 'human_assisted':
            baseline = page['automatic_baseline']['files']
            assistance = '<p><b>人工辅助结果：不计为全自动处理效果。</b> 自动基线与每次确认的笔画已单独保存，仍需检查残留及背景损伤。</p>'
            for kind, label in [('clean', '自动处理基线'), ('selection_overlay', '自动检测选区')]:
                url = escape(quote(baseline[kind]), quote=True)
                figures.append(f'<figure><figcaption>{label}</figcaption><a href="{url}">'
                               f'<img loading="lazy" src="{url}" alt="{label}"></a></figure>')
            assistance += '<p>' + ' · '.join(
                f'<a href="{escape(quote(s["mask"]), quote=True)}">辅助第{i}次' +
                ('完整选区' if s.get('operation') == 'edit_complete_selection' else '确认笔画') + '</a>'
                for i, s in enumerate(page['assistance_steps'], 1)) + '</p>'
        links = " · ".join(f'<a href="{escape(quote(path), quote=True)}">{escape(kind)}</a>'
                           for kind, path in files.items())
        sections.append(
            f'<section><h2>{escape(page["page"])}</h2><p>'
            f'{selection["width"]} × {selection["height"]} · '
            f'定位方式：{escape(page["mask_method"])} · '
            f'判定阈值：{selection["threshold"]:.1%} · '
            f'选区占比：{selection["coverage"]:.2%}</p>'
            f'{assistance}<div class="images">{"".join(figures)}</div><details>'
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
<p>手动画笔处理的黑白图表示编辑后的完整选区，红色只表示选中，不表示透明度。mask_alpha 保留自动观测，不能代替手工选区；actual_changes 单独记录相对原图实际变化的位置。每次处理从原图计算，未处理的编辑不导出。</p>
<p>统计与完整参数见 <a href="manifest.json">manifest.json</a>。复核资料完全本地保存。</p>
''' + "".join(sections) + "</html>", encoding="utf-8")
