"""Source-only inference process. Evaluation references never enter this process."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PIL import Image


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temp.replace(path)


def pixels(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    from dataclasses import asdict
    from wm import documents, pipeline
    from wm.contract import Thresholds
    task_path = Path(sys.argv[1])
    task = json.loads(task_path.read_text())
    folder = task_path.parent
    records = []
    settings = task['settings']
    thr = Thresholds(**settings['thresholds'])

    def store(result, item, page):
        key = f"{task['mode']}:{item['id']}:{page}"
        dest = folder / hashlib.sha256(key.encode()).hexdigest()[:20]
        dest.mkdir(exist_ok=False)
        selected = result.mask.binary(thr.mask_alpha_min) > 0
        changed = np.any(result.source.image != result.output.image, axis=2)
        Image.fromarray(result.output.image).save(dest / '处理结果.png')
        Image.fromarray(selected.astype('uint8') * 255).save(dest / '笔画选区.png')
        preview = Image.fromarray(result.source.image)
        preview.thumbnail((1000, 1000))
        preview.save(dest / '原图预览.jpg', quality=90)
        overlay = result.source.image.copy()
        overlay[selected] = (overlay[selected].astype(float) * .55 + np.array([255, 0, 0]) * .45).astype('uint8')
        preview = Image.fromarray(overlay)
        preview.thumbnail((1000, 1000))
        preview.save(dest / '选区预览.jpg', quality=90)
        np.savez_compressed(dest / '检测数值.npz', alpha=result.mask.alpha)
        if result.comparison_results:
            pipeline.write_results([result], dest / '自动比较详情')
        record = dict(key=key, id=item['id'], page=page, mode=task['mode'], status='completed',
                      input_sha256=item['input_sha256'], settings=settings,
                      output_sha256=pixels(result.output.image), mask_sha256=pixels(selected),
                      selected_pixels=int(selected.sum()), selected_fraction=float(selected.mean()),
                      changed_pixels=int(changed.sum()),
                      outside_selection_changed_pixels=int(changed[~selected].sum()),
                      elapsed_ms=result.verdict.elapsed_ms, method=result.mask.method,
                      repair_mode=result.mask.repair_mode, repairs=[str(r) for r in result.verdict.repairs],
                      automatic_verdict=result.verdict.to_dict(),
                      processing=result.processing,
                      files={p.name: str(p) for p in dest.iterdir() if p.is_file()})
        records.append(record)
        save_json(folder / 'inference.json', records)
        print(key, flush=True)

    def error(item, page, message):
        key = f"{task['mode']}:{item['id']}:{page}"
        records.append(dict(key=key, id=item['id'], page=page, mode=task['mode'],
                            status='error', error=message, settings=settings))
        save_json(folder / 'inference.json', records)

    if task['mode'] == 'cli':
        for item in task['inputs']:
            try:
                results = pipeline.process_file(item['input'], thr=thr, dpi=item['dpi'], use_ocr=True)
                if len(results) != item['page_count']:
                    raise ValueError('实际页数与固定清单不一致')
                for i, result in enumerate(results, 1):
                    store(result, item, i)
            except Exception:
                for i in range(1, item['page_count'] + 1):
                    if not any(r['id'] == item['id'] and r['page'] == i for r in records):
                        error(item, i, traceback.format_exc())
    else:
        from PySide6.QtCore import Qt
        from wm.batches import Batch, BatchWorker
        pages, lookup = [], []
        for item in task['inputs']:
            try:
                loaded = documents.load_pages(item['input'], dpi=item['dpi'])
                if len(loaded) != item['page_count']:
                    raise ValueError('实际页数与固定清单不一致')
                pages.extend(loaded)
                lookup.extend((item, i) for i in range(1, len(loaded) + 1))
            except Exception:
                for i in range(1, item['page_count'] + 1):
                    error(item, i, traceback.format_exc())
        batch = Batch('固定管控批次', pages)
        # Per-page callback errors must be recorded explicitly; Qt otherwise swallows them.
        def event(payload):
            if payload['kind'] != 'page':
                return
            item, page = lookup[payload['index']]
            try:
                if payload['result'] is None:
                    error(item, page, payload['error'])
                else:
                    store(payload['result'], item, page)
            except Exception:
                error(item, page, traceback.format_exc())
        worker = BatchWorker([(batch, list(range(len(pages))), thr,
                               settings['close_px'], settings['ocr_enabled'], None)])
        worker.update.connect(event, Qt.DirectConnection)
        worker.run()
    return int(any(r['status'] != 'completed' for r in records))


if __name__ == '__main__':
    raise SystemExit(main())
