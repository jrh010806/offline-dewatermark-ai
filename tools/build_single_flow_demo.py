"""Build an offline, source-backed walkthrough of one desktop BatchWorker run.

Read-only observation wrappers record calls without changing their results.
No reference image, true mask, or evaluation annotation is loaded.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime
from functools import wraps
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('WM_ORT_THREADS', '4')

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_image(folder, name, array):
    Image.fromarray(np.asarray(array, np.uint8)).save(folder / (name + '.png'))


def observe_run(source, out):
    from PySide6.QtCore import QCoreApplication
    from wm import masks, ocr, recall, lama, contrast_text
    from wm.batches import Batch, BatchWorker
    from wm.contract import Thresholds, PageImage, MaskLayer
    from wm.export_review import overlay_mask
    import importlib

    out.mkdir(parents=True, exist_ok=False)
    code_before = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'wm').glob('*.py')}
    source_hash = sha(source)
    image = np.array(Image.open(source).convert('RGB'))
    trace, observations, captures, patches = [], [], {}, []
    inside_faint = [False]

    def patch(module, name, replacement):
        patches.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def observe(module, name, label):
        original = getattr(module, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            start = perf_counter()
            result = original(*args, **kwargs)
            entry = {'function': module.__name__ + '.' + name, 'label': label,
                     'seconds': perf_counter() - start}
            if isinstance(result, MaskLayer):
                entry.update(found=bool(result.binary().any()), method=result.method,
                             repair_mode=result.repair_mode,
                             selected_pixels=int(np.count_nonzero(result.binary())))
            else:
                entry['found'] = result is not None
            trace.append(entry)
            return result
        patch(module, name, wrapped)

    for module_name, function, label in [
        ('paper_grid', 'estimate', '文档规则小点'),
        ('periodic', 'estimate', '图内规则小点'),
        ('corner_id', 'estimate', '角落编号'),
        ('observed_text', 'repeated_photo', '照片重复字形'),
        ('observed_text', 'shadow_paper', '纸面阴影文字'),
        ('photo_observations', 'repeated_lines', '照片重复文字行'),
        ('photo_observations', 'local_lines', '照片局部文字'),
        ('photo_text', 'estimate', '照片文字补充判断'),
        ('repeated', 'estimate', '重复斜向文字'),
        ('paper_repeat', 'estimate', '纸面重复文字'),
        ('document_overlay', 'estimate', '文档透明覆盖'),
        ('chromatic', 'estimate', '彩色大字'),
        ('contrast_text', 'dark_oblique', '深色斜字'),
        ('contrast_text', 'bright_repeated', '白色重复字'),
    ]:
        module = importlib.import_module('wm.' + module_name)
        observe(module, function, label)

    read_lines = ocr.read_lines

    @wraps(read_lines)
    def read_observed(view):
        result = read_lines(view)
        if inside_faint[0]:
            observations.append({'view': view.copy(), 'lines': result or []})
        return result
    patch(ocr, 'read_lines', read_observed)

    faint = contrast_text.faint_chromatic

    @wraps(faint)
    def faint_observed(*args, **kwargs):
        inside_faint[0] = True
        start = perf_counter()
        try:
            result = faint(*args, **kwargs)
        finally:
            inside_faint[0] = False
        trace.append({'function': 'wm.contrast_text.faint_chromatic',
                      'label': '放大淡字的颜色差异', 'seconds': perf_counter() - start,
                      'found': result is not None,
                      'method': result.method if result else None})
        return result
    patch(contrast_text, 'faint_chromatic', faint_observed)

    estimate = masks._estimate_single_page

    @wraps(estimate)
    def base_observed(*args, **kwargs):
        result = estimate(*args, **kwargs)
        captures['base_alpha'] = result.alpha.copy()
        captures['base_method'] = result.method
        return result
    patch(masks, '_estimate_single_page', base_observed)

    expand = recall.expand

    @wraps(expand)
    def expand_observed(*args, **kwargs):
        before = args[1].binary() > 0
        start = perf_counter()
        result = expand(*args, **kwargs)
        after = result.binary() > 0
        captures['completion'] = {'enabled': True, 'seconds': perf_counter() - start,
                                  'added_pixels': int(np.count_nonzero(after & ~before)),
                                  'removed_pixels': int(np.count_nonzero(before & ~after)),
                                  'method_after': result.method}
        return result
    patch(recall, 'expand', expand_observed)

    tile = lama._run_tile
    tiles = []

    @wraps(tile)
    def tile_observed(session, view, region):
        tiles.append({'width': int(view.shape[1]), 'height': int(view.shape[0]),
                      'selected_pixels': int(np.count_nonzero(region))})
        return tile(session, view, region)
    patch(lama, '_run_tile', tile_observed)

    thr = Thresholds(watermark_recall_priority=True)
    app = QCoreApplication.instance() or QCoreApplication([])
    batch = Batch('单图流程演示', [PageImage(image, 96, 'image', 0, 'image')])
    worker = BatchWorker([(batch, [0], thr, 11, True, None)])
    updates = []
    worker.update.connect(updates.append)
    start = perf_counter()
    try:
        worker.run()
    finally:
        for module, name, original in reversed(patches):
            setattr(module, name, original)
    elapsed = perf_counter() - start
    pages = [u for u in updates if u.get('kind') == 'page']
    if not pages or pages[-1].get('error'):
        raise RuntimeError(pages[-1].get('error') if pages else 'No page result')
    result = pages[-1]['result']
    selected = result.mask.binary(thr.mask_alpha_min) > 0
    output = result.output.image
    changed = np.any(output != image, axis=2)
    base = captures['base_alpha'] >= thr.mask_alpha_min
    save_image(out, '原始图片', image)
    save_image(out, '初步水印选区', base * 255)
    save_image(out, '最终水印选区', selected * 255)
    save_image(out, '红色选区叠加', overlay_mask(image, result.mask.alpha))
    save_image(out, '实际修改像素', changed * 255)
    save_image(out, '处理结果', output)
    np.savez_compressed(out / '选区原始数值.npz', alpha=result.mask.alpha,
                        base_alpha=captures['base_alpha'], changed=changed)

    # Choose the actually observed OCR view that best overlaps the final selection.
    def overlap(observation):
        domain = np.zeros(image.shape[:2], np.uint8)
        for box, _, _ in observation['lines']:
            cv2.fillConvexPoly(domain, np.rint(box).astype(np.int32), 1)
        return int(np.count_nonzero(domain.astype(bool) & selected))

    observed = max(observations, key=overlap) if observations else None
    lines_json = []
    if observed:
        save_image(out, '实际颜色增强画面', observed['view'])
        framed = Image.fromarray(observed['view'])
        draw = ImageDraw.Draw(framed)
        for box, text, confidence in observed['lines']:
            pts = [tuple(map(float, xy)) for xy in box]
            draw.line(pts + [pts[0]], fill=(0, 120, 165), width=3)
            lines_json.append({'box': box.tolist(), 'text': text, 'confidence': confidence})
        framed.save(out / '实际文字识别范围.png')

    code_after = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'wm').glob('*.py')}
    assert code_before == code_after, 'Inference code changed during run'
    assert sha(source) == source_hash, 'Input file changed during run'
    if thr.lama_mask_dilate == 0:
        assert not np.any(changed & ~selected), 'Unexpected changes outside selection'
    ys, xs = np.nonzero(selected)
    crop = ([max(0, int(xs.min()) - 75), max(0, int(ys.min()) - 55),
             min(image.shape[1], int(xs.max()) + 76),
             min(image.shape[0], int(ys.max()) + 56)] if xs.size else
            [0, 0, image.shape[1], image.shape[0]])
    model = Path(lama._resolve_model_path())
    metadata = {
        'title': '一张照片怎样去水印', 'version': 'V1',
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'input': str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        'input_sha256': source_hash, 'width': image.shape[1], 'height': image.shape[0],
        'entry': 'wm.batches.BatchWorker.run', 'parameters': asdict(thr),
        'ocr_enabled': True, 'style': None, 'close_px': 11, 'batch_size': 1,
        'method': result.mask.method, 'repair_mode': result.mask.repair_mode,
        'repairs': [str(x) for x in result.verdict.repairs],
        'seconds': elapsed, 'stage_ms': result.verdict.stage_ms,
        'selected_pixels': int(selected.sum()), 'base_pixels': int(base.sum()),
        'coverage': float(selected.mean()), 'changed_pixels': int(changed.sum()),
        'changed_outside_selection': int(np.count_nonzero(changed & ~selected)),
        'completion': captures.get('completion'), 'crop': crop,
        'observed_ocr_view_count': len(observations), 'displayed_ocr_lines': lines_json,
        'trace': trace, 'model_tiles': tiles, 'verdict': result.verdict.to_dict(),
        'reference_metrics': None, 'independent_validation': False,
        'reference_policy': 'No clean reference, true alpha, manifest or website result read by this run.',
        'code_sha256': code_before,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'model': {'name': model.name, 'sha256': sha(model),
                  'providers': lama._sess.get_providers() if lama._sess else []},
        'assets_sha256': {p.name: sha(p) for p in out.glob('*.png')},
    }
    (out / '运行记录.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ('method', 'seconds', 'selected_pixels',
          'changed_pixels', 'changed_outside_selection', 'completion', 'model_tiles')}, ensure_ascii=False), flush=True)
    return metadata


def render(out, destination):
    meta = json.loads((out / '运行记录.json').read_text())
    # This narrative explains this example's route; do not silently reuse it
    # for another repair route after an algorithm change or different input.
    if meta['method'] != 'contrast_faint_chromatic' or meta['repairs'] != ['lama']:
        raise ValueError('实际路径已变化，须更新讲解模板后再生成，不能沿用本例的固定说明。')
    required = {'原始图片', '实际颜色增强画面', '实际文字识别范围',
                '红色选区叠加', '最终水印选区', '实际修改像素', '处理结果'}
    if not required.issubset({p.stem for p in out.glob('*.png')}):
        raise ValueError('缺少演示所需的真实中间画面。')
    assets = {}
    crop = tuple(meta['crop'])
    for path in out.glob('*.png'):
        # Historical runs may retain pedagogical views. Only embed actual run
        # inputs/outputs and clearly identified overlays derived from them.
        if path.stem == '灰度辅助示意':
            continue
        im = Image.open(path)
        for suffix, image in (('', im), ('_crop', im.crop(crop))):
            buf = BytesIO()
            image.save(buf, format='WEBP', lossless=True, method=4)
            # Roundtrip check keeps the embedded evidence pixel-exact.
            decoded = Image.open(BytesIO(buf.getvalue()))
            assert np.array_equal(np.array(decoded.convert(image.mode)), np.array(image))
            assets[path.stem + suffix] = 'data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode()
    data = json.dumps({'meta': meta, 'assets': assets,
                      'presentation': {'version': 'V2',
                                       'rendered_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                                       'source_record': str(out.relative_to(ROOT) / '运行记录.json') if out.is_relative_to(ROOT) else str(out / '运行记录.json'),
                                       'display_policy': 'Actual captured inputs/outputs; OCR boxes, selection tint and change map are explicitly labelled visualizations of recorded data. No extra grayscale teaching view.'}},
                      ensure_ascii=False).replace('</', '<\\/')
    template = (ROOT / 'tools/templates/single_flow_demo.html').read_text(encoding='utf-8')
    assert template.count('__DEMO_DATA__') == 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace('__DEMO_DATA__', data), encoding='utf-8')
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/watermark_regression_min_v1/inputs/core/core_001.png')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if not args.render_only:
        observe_run(args.input.resolve(), args.out.resolve())
    render(args.out.resolve(), args.html.resolve())


if __name__ == '__main__':
    main()
