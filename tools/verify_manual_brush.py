"""Source-only desktop regression and explicit brush-operation replay.

This is an opt-in diagnostic, never an application entry or a quality benchmark.
It reads only source paths/metadata from the saved inventory, never references.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['HF_HUB_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replay-only', action='store_true', help='只重跑目录文档所属4张批次与手工回放，非全范围验收')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.snapshot.resolve()))
    import numpy as np
    from PIL import Image
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest
    from wm import documents, pipeline
    from wm.app import make_app, MainWindow
    from wm.batches import Batch, BatchWorker
    from wm.contract import Thresholds
    from wm.selection_edit import SelectionEdit, apply_selection

    app = make_app()
    baseline = json.loads((args.baseline/'运行记录.json').read_text())
    inventory = json.loads((args.baseline/'样本清单.json').read_text())
    old = {r['key']: r for r in baseline['records']}
    groups = defaultdict(list)
    for row in inventory['cases']:
        if row['id'].startswith('历史图片/') or row['batch'] == '01_快速20张/04_非等距密集文字':
            groups[row['batch']].append(row)
    records, replay = [], []

    def save():
        (args.out/'桌面逐张对照.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
        (args.out/'画笔操作回放.json').write_text(json.dumps(replay, ensure_ascii=False, indent=2))

    for mode in (['desktop'] if args.replay_only else ['desktop', 'desktop_protected']):
        profile = baseline['profiles'][mode]
        thresholds = Thresholds(**profile['thresholds'])
        for batch_name, rows in groups.items():
            if args.replay_only and batch_name != '历史图片/big_center':
                continue
            pages = []
            for row in rows:
                pages.extend(documents.load_pages(row['input'], dpi=row['dpi']))
            batch = Batch(batch_name, pages)

            def receive(payload):
                if payload['kind'] != 'page':
                    return
                index = payload['index']; row = rows[index]
                result = payload['result']; key = f'{mode}:{row["id"]}:1'
                record = dict(key=key, status='error', error=payload['error'])
                if result is not None:
                    batch.results[index] = result
                    batch.states[index] = '待复核'
                    selected = result.mask.binary(thresholds.mask_alpha_min) > 0
                    expected = old[key]
                    record.update(status='completed', error=None,
                                  output_identical=sha256(result.output.image.tobytes()).hexdigest() == expected['output_sha256'],
                                  selection_identical=sha256(selected.tobytes()).hexdigest() == expected['mask_sha256'],
                                  route_identical=result.mask.repair_mode == expected['repair_mode'],
                                  selected_pixels=int(selected.sum()), elapsed_ms=result.verdict.elapsed_ms,
                                  previous_elapsed_ms=expected['elapsed_ms'],
                                  timing_gate_passed=result.verdict.elapsed_ms <= max(
                                      expected['elapsed_ms']*1.5, expected['elapsed_ms']+2000))
                    folder = args.out/mode/Path(row['id']).name
                    folder.mkdir(parents=True, exist_ok=False)
                    Image.fromarray(result.source.image).save(folder/'原图.png')
                    Image.fromarray(result.output.image).save(folder/'自动结果.png')
                    Image.fromarray(selected.astype('uint8')*255).save(folder/'自动完整选区.png')
                    record['folder'] = str(folder)
                records.append(record); save()
                print(key, record.get('output_identical'), flush=True)

            worker = BatchWorker([(batch, list(range(len(pages))), thresholds,
                                   profile['close_px'], profile['ocr_enabled'], None)])
            worker.update.connect(receive, Qt.DirectConnection)
            worker.run()
            if mode != 'desktop' or batch_name != '历史图片/big_center':
                continue
            # Replay on the screenshot's document. Coordinates are UI test
            # actions only, not truth masks or evidence of removal quality.
            index = next(i for i, row in enumerate(rows) if row['id'].endswith('big_center_04'))
            if index not in batch.results:
                continue
            w = MainWindow(); w.batches = [batch]; w.resize(1600, 1000)
            w.refresh_batches(); w.batch_list.setCurrentRow(0)
            w.page_list.setCurrentRow(index); w.show(); app.processEvents()
            result = batch.results[index]
            w.assist_select_btn.click(); w.brush_size.setValue(5)
            rect = w.canvas.image_rect()
            # The top of the visible diagonal watermark, on this source only.
            point = QPoint(round(rect.left()+rect.width()*.155), round(rect.top()+rect.height()*.29))
            w.canvas.zoom_at(point.toPointF(), 2)
            QTest.mouseMove(w.canvas, point); app.processEvents()
            w.grab().save(str(args.out/'01_实际界面_原图与画笔放大镜.png'))
            QTest.mousePress(w.canvas, Qt.LeftButton, pos=point)
            QTest.mouseMove(w.canvas, point+QPoint(6, 8))
            QTest.mouseRelease(w.canvas, Qt.LeftButton, pos=point+QPoint(6, 8))
            draft = w._selection_edit
            w.set_view_mode('水印选区'); app.processEvents()
            w.grab().save(str(args.out/'02_实际界面_编辑后的完整选区.png'))
            wanted = draft.selected.copy()
            started = time.perf_counter()
            w.assist_confirm_btn.click()
            while w._assist_busy and time.perf_counter()-started < 240:
                app.processEvents(); QTest.qWait(10)
            if w._assist_busy:
                # Preserve the still-running state and fail this diagnostic;
                # do not destroy a running QThread or call it a successful page.
                raise TimeoutError('画笔回放处理超过240秒')
            processed = batch.results[index]
            committed = processed is not result
            row = dict(page=rows[index]['id'], committed=committed,
                       source_only=True, reference_images_read=False,
                       note='固定画笔操作仅验证交互与范围，不是真实水印标注；未评估修复质量')
            if committed:
                actual = processed.mask.binary() > 0
                outside = np.any(processed.output.image != processed.source.image, axis=2) & ~actual
                row.update(complete_selection_identical=bool(np.array_equal(actual, wanted)),
                           outside_selected_changed_pixels=int(outside.sum()))
                pipeline.write_results([processed], args.out/'手工处理导出')
                # Identical complete masks must not accumulate reconstruction.
                repeated = apply_selection(processed, SelectionEdit(processed).request(thresholds))
                row['repeat_from_source_identical'] = bool(np.array_equal(repeated.output.image, processed.output.image))
                w.set_view_mode('处理结果'); app.processEvents()
                w.grab().save(str(args.out/'03_实际界面_按选区处理结果.png'))
            else:
                row['error'] = w.status.text()
            replay.append(row); save()
            batch.dirty.clear(); w.close()
    save()
    failed = [r for r in records if r['status'] != 'completed' or not all(
        r.get(k) for k in ['output_identical', 'selection_identical', 'route_identical'])]
    summary = dict(pages=len(records), exact_failures=len(failed),
                   timing_warnings=sum(not r.get('timing_gate_passed', False) for r in records),
                   effective_thresholds=asdict(thresholds),
                   manual_replay=replay, independent_validation=False,
                   note='已登记来源的补充诊断，不代替完整test验收；不读取参考或测量真实水印召回。')
    (args.out/'范围与结论.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(bool(failed) or not replay or any(not r.get('committed') or
               r.get('outside_selected_changed_pixels') != 0 or
               not r.get('repeat_from_source_identical') for r in replay))


if __name__ == '__main__':
    raise SystemExit(main())
