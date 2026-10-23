import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import Qt
from wm.app import MainWindow, make_app
from wm.batches import Batch, BatchWorker
from wm.contract import PageImage, MaskLayer, Thresholds


def page(index, shape=(20, 30, 3)):
    return PageImage(np.full(shape, 240, np.uint8), 200, 'sample', index)


def result(src, mask, thr, **_kwargs):
    return SimpleNamespace(source=src, mask=mask, verdict=SimpleNamespace(
        passed=True, review_needed=True, stage_ms={}, elapsed_ms=1))


class QueueTests(unittest.TestCase):
    def test_single_target_uses_whole_size_group_and_reuses_prior(self):
        batch = Batch('a', [page(0), page(1), page(2, (10, 10, 3))])
        mask = MaskLayer(np.zeros((20, 30), np.float32))
        jobs = [(batch, [1], Thresholds(), 11, False)]
        with patch('wm.batches.masks.estimate_crosspage', return_value=mask) as prior, \
             patch('wm.batches.pipeline._process_one', side_effect=result) as repair:
            BatchWorker(jobs).run()
            BatchWorker(jobs).run()
            self.assertEqual(prior.call_count, 1)
            self.assertEqual(len(prior.call_args.args[0]), 2)
            self.assertIs(repair.call_args.args[0], batch.pages[1])
            batch.invalidate()
            BatchWorker(jobs).run()
            self.assertEqual(prior.call_count, 2)

    def test_mixed_sizes_isolation_error_continuation_and_immediate_updates(self):
        a = Batch('a', [page(0), page(1), page(2, (10, 10, 3))])
        b = Batch('b', [page(0), page(1)])
        events = []
        def process(src, mask, thr, **_kwargs):
            # First result must have been emitted before starting the second image.
            if src is a.pages[1]:
                self.assertEqual(len([e for e in events if e['kind']=='page']), 1)
                raise ValueError('bad image')
            return result(src, mask, thr)
        worker = BatchWorker([(a, [0,1,2], Thresholds(), 11, False),
                              (b, [0,1], Thresholds(), 11, False)])
        worker.update.connect(events.append)
        with patch('wm.batches.masks.estimate_crosspage', side_effect=lambda imgs,*a,**kw: MaskLayer(np.zeros(imgs[0].shape[:2]))) as prior, \
             patch('wm.batches.masks.estimate_single_page', return_value=MaskLayer(np.zeros((10,10)))) as single, \
             patch('wm.batches.pipeline._keepout', return_value=None), \
             patch('wm.batches.pipeline._process_one', side_effect=process):
            worker.run()
        self.assertEqual(prior.call_count, 2)
        self.assertEqual(single.call_count, 5)  # every page retains its own baseline
        pages = [e for e in events if e['kind']=='page']
        self.assertEqual([e['done'] for e in pages], [1,2,3,4,5])
        self.assertEqual(pages[1]['error'], 'bad image')

    def test_resume_releases_pause(self):
        worker = BatchWorker([])
        worker.pause(True)
        reached = threading.Event()
        worker.update.connect(lambda e: reached.set(), Qt.DirectConnection)
        allowed = []
        thread = threading.Thread(target=lambda: allowed.append(worker._checkpoint()), daemon=True)
        thread.start()
        self.assertTrue(reached.wait(2))
        worker.pause(False)
        thread.join(2)
        self.assertEqual(allowed, [True])

    def test_pause_resume_and_stop_at_page_boundary(self):
        batch = Batch('a', [page(0), page(1)])
        worker = BatchWorker([(batch,[0,1],Thresholds(),11,False)])
        worker.pause(True)
        reached = threading.Event()
        worker.update.connect(lambda e: reached.set() if e['kind']=='paused' else None, Qt.DirectConnection)
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        self.assertTrue(reached.wait(2))
        self.assertEqual(worker.done, 0)
        worker.stop()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        worker = BatchWorker([(batch,[0,1],Thresholds(),11,False)])
        worker.update.connect(lambda e: worker.stop() if e['kind']=='page' else None)
        with patch('wm.batches.masks.estimate_crosspage', return_value=MaskLayer(np.zeros((20,30)))), \
             patch('wm.batches.pipeline._process_one', side_effect=result):
            worker.run()
        self.assertEqual(worker.done, 1)


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def test_parameter_isolation_invalidation_and_busy_lock(self):
        w = MainWindow()
        w.batches = [Batch('a',[page(0)]), Batch('b',[page(0)])]
        w.refresh_batches()
        w.batch_list.setCurrentRow(0)
        w._sliders['close_px'].setValue(15)
        w.batch.priors['test'] = object()
        w._sliders['lama_context_margin'].setValue(320)
        self.assertIn('test', w.batch.priors)
        w._sliders['close_px'].setValue(17)
        self.assertFalse(w.batch.priors)
        w.batch_list.setCurrentRow(1)
        self.assertEqual(w.current_close_px(), 11)
        w.batch_list.setCurrentRow(0)
        self.assertEqual(w.current_close_px(), 17)
        w.set_busy(True)
        self.assertFalse(w.param_box.isEnabled())
        self.assertFalse(w.open_btn.isEnabled())
        self.assertTrue(w.batch_list.isEnabled())
        w.set_busy(False)
        w.close()

    def test_export_current_image_keeps_other_results_dirty(self):
        from wm.contract import PageResult, VerdictReport, EngineKind
        w = MainWindow()
        batch = Batch('a/b', [page(0), page(1)])
        for i, src in enumerate(batch.pages):
            report = VerdictReport(src.key, EngineKind.RASTER, [], {}, {}, [], 1)
            batch.results[i] = PageResult(src, src, MaskLayer(np.zeros(src.image.shape[:2])), report)
            batch.dirty.add(i)
        w.batches = [batch]
        w.refresh_batches()
        w.batch_list.setCurrentRow(0)
        w.page_list.setCurrentRow(1)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('wm.app.QInputDialog.getItem', return_value=('当前图片', True)), \
             patch('wm.app.QFileDialog.getExistingDirectory', return_value=tmp):
            w.export_results()
            self.assertEqual(len(list(Path(tmp).rglob('*_clean.png'))), 1)
            self.assertEqual(len(list(Path(tmp).rglob('queue.json'))), 1)
            self.assertEqual(batch.dirty, {0})
        batch.dirty.clear()
        w.close()


class OptionalPriorTests(unittest.TestCase):
    def test_bad_prior_does_not_block_single_image_processing(self):
        batch=Batch('a',[page(0),page(1)])
        single=MaskLayer(np.zeros((20,30),np.float32))
        worker=BatchWorker([(batch,[0,1],Thresholds(),11,False)])
        with patch('wm.batches.masks.estimate_crosspage',side_effect=ValueError('unusable prior')) as prior, \
             patch('wm.batches.masks.estimate_single_page',return_value=single), \
             patch('wm.batches.pipeline._process_one',side_effect=result), \
             self.assertLogs('wm.batches',level='WARNING'):
            worker.run()
        self.assertEqual(worker.done,2)
        self.assertEqual(worker.outcomes,['待复核','待复核'])
        self.assertEqual(prior.call_count,1)

if __name__ == '__main__':
    unittest.main()
