import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import patch

from wm.batch_timing import BatchTiming, format_duration
from wm.batches import Batch, BatchWorker
from wm.app import MainWindow, make_app
from wm.contract import Thresholds, MaskLayer
from test_batches import page, result
import numpy as np


class BatchTimingTests(unittest.TestCase):
    def test_pause_live_average_and_finished_time(self):
        now = [0.0]
        with patch('wm.batch_timing.perf_counter', side_effect=lambda: now[0]):
            timing = BatchTiming(2)
            now[0] = 3
            timing.complete_page()
            now[0] = 5
            timing.pause()
            now[0] = 100
            paused = timing.snapshot()
            self.assertEqual(paused.elapsed_seconds, 5)
            self.assertEqual(paused.average_seconds, 3)
            timing.resume()
            now[0] = 102
            timing.complete_page()
            timing.finish(stopped=False)
            now[0] = 999
            done = timing.snapshot()
            self.assertEqual(done.elapsed_seconds, 7)
            self.assertEqual(done.average_seconds, 3.5)
            self.assertEqual(done.state, '已完成')

    def test_stop_before_first_completed_page_has_no_average(self):
        now = [0.0]
        with patch('wm.batch_timing.perf_counter', side_effect=lambda: now[0]):
            timing = BatchTiming(2)
            now[0] = 4
            timing.finish(stopped=True)
            self.assertEqual(timing.snapshot().elapsed_seconds, 4)
            self.assertIsNone(timing.snapshot().average_seconds)
            self.assertEqual(timing.snapshot().state, '已停止')

    def test_worker_counts_failure_and_resets_only_for_whole_batch_runs(self):
        batch = Batch('a', [page(0), page(1)])
        now = [0.0]
        def process(src, mask, thr, **kwargs):
            now[0] += 2
            if src is batch.pages[1]:
                raise ValueError('failed page')
            return result(src, mask, thr)
        mask = MaskLayer(np.zeros((20, 30), np.float32))
        with patch('wm.batch_timing.perf_counter', side_effect=lambda: now[0]), \
             patch('wm.batches.structure.process_text_page', return_value=None), \
             patch('wm.batches.masks.estimate_crosspage', return_value=mask), \
             patch('wm.batches.masks.estimate_single_page', return_value=mask), \
             patch('wm.batches.pipeline._process_one', side_effect=process):
            jobs = [(batch, [0, 1], Thresholds(), 11, False)]
            BatchWorker(jobs, record_batch_timing=True).run()
            first = batch.timing
            self.assertEqual(first.snapshot().elapsed_seconds, 4)
            self.assertEqual(first.snapshot().average_seconds, 2)
            self.assertEqual(first.snapshot().completed, 2)
            self.assertEqual(first.snapshot().failed, 1)
            BatchWorker([(batch, [0], Thresholds(), 11, False)]).run()
            self.assertIs(batch.timing, first)
            self.assertEqual(first.snapshot().elapsed_seconds, 4)
            self.assertEqual(first.snapshot().failed, 1)
            BatchWorker(jobs, record_batch_timing=True).run()
            self.assertIsNot(batch.timing, first)
            self.assertEqual(batch.timing.snapshot().elapsed_seconds, 4)
            self.assertEqual(batch.timing.snapshot().failed, 1)

    def test_batch_queue_does_not_charge_other_batches_or_replace_unstarted_run(self):
        batches = [Batch('a', [page(0)]), Batch('b', [page(0)])]
        now = [0.0]
        def process(*args, **kwargs):
            now[0] += 5
            return result(*args, **kwargs)
        jobs = [(b, [0], Thresholds(), 11, False) for b in batches]
        with patch('wm.batch_timing.perf_counter', side_effect=lambda: now[0]), \
             patch('wm.batches.structure.process_text_page', return_value=None), \
             patch('wm.batches.masks.estimate_single_page', return_value=MaskLayer(np.zeros((20, 30)))), \
             patch('wm.batches.pipeline._process_one', side_effect=process):
            BatchWorker(jobs, record_batch_timing=True).run()
            for batch in batches:
                self.assertEqual(batch.timing.snapshot().elapsed_seconds, 5)
            previous = batches[1].timing
            worker = BatchWorker(jobs, record_batch_timing=True)
            worker.update.connect(lambda event: worker.stop() if event['kind'] == 'page' else None)
            worker.run()
            self.assertIs(batches[1].timing, previous)

    def test_time_units_and_rounding_carry(self):
        self.assertEqual(format_duration(0), '0.0 秒')
        self.assertEqual(format_duration(59.99), '1 分 0.0 秒')
        self.assertEqual(format_duration(272.3), '4 分 32.3 秒')
        self.assertEqual(format_duration(3602.1), '1 小时 0 分 2.1 秒')


class BatchTimingUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = make_app()

    def test_empty_completed_switch_and_deleted_batch(self):
        window = MainWindow()
        self.addCleanup(window.close)
        self.assertEqual(window.batch_time_hint.text(), '已处理：—\n批次总用时：—\n平均每张：—')
        a, b = Batch('a', [page(0)]), Batch('b', [page(0)])
        with patch('wm.batch_timing.perf_counter', return_value=0):
            a.timing = BatchTiming(1)
        with patch('wm.batch_timing.perf_counter', return_value=2):
            a.timing.complete_page()
            a.timing.finish(stopped=False)
        window.batches = [a, b]
        window.refresh_batches()
        window.batch_list.setCurrentRow(0)
        self.assertEqual(window.batch_time_hint.text(), '已处理：1 / 1 张\n批次总用时：2.0 秒\n平均每张：2.0 秒')
        window.batch_list.setCurrentRow(1)
        self.assertEqual(window.batch_time_hint.text(), '已处理：0 / 1 张\n批次总用时：—\n平均每张：—')
        window.select_batch(-1)
        self.assertEqual(window.batch_time_hint.text(), '已处理：—\n批次总用时：—\n平均每张：—')

    def test_progress_failure_pause_stop_and_new_round_ignore_old_errors(self):
        window = MainWindow()
        self.addCleanup(window.close)
        batch = Batch('a', [page(0), page(1), page(2)])
        batch.errors = {0: 'old failure', 1: 'old failure', 2: 'old failure'}
        window.batches = [batch]
        window.refresh_batches()
        window.batch_list.setCurrentRow(0)
        self.assertIn('已处理：0 / 3 张', window.batch_time_hint.text())
        self.assertNotIn('失败', window.batch_time_hint.text())
        batch.timing = BatchTiming(3)
        batch.timing.complete_page(failed=True)
        batch.timing.pause()
        window.refresh_batch_timing()
        self.assertIn('已处理：1 / 3 张（失败 1 张）', window.batch_time_hint.text())
        self.assertIn('已暂停', window.batch_time_hint.text())
        batch.timing.resume()
        batch.timing.complete_page()
        batch.timing.finish(stopped=True)
        window.refresh_batch_timing()
        self.assertIn('已处理：2 / 3 张（失败 1 张）', window.batch_time_hint.text())
        self.assertIn('已停止', window.batch_time_hint.text())
        batch.timing = BatchTiming(3)
        window.refresh_batch_timing()
        self.assertIn('已处理：0 / 3 张', window.batch_time_hint.text())
        self.assertNotIn('失败', window.batch_time_hint.text())

    def test_single_page_button_does_not_record_whole_batch_time(self):
        window = MainWindow()
        self.addCleanup(window.close)
        window.batches = [Batch('a', [page(0)])]
        window.refresh_batches()
        window.batch_list.setCurrentRow(0)
        with patch('wm.app.BatchWorker') as worker:
            window.start_run(only_page=0, force_automatic=True)
            self.assertFalse(worker.call_args.kwargs['record_batch_timing'])
        window.worker = None
        with patch('wm.app.BatchWorker') as worker:
            window.start_run(scope='batch')
            self.assertTrue(worker.call_args.kwargs['record_batch_timing'])
        window._batch_time_timer.stop()
        window.worker = None
