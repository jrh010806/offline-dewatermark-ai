import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from wm import auto_selection as auto, pipeline
from wm.contract import EngineKind, MaskLayer, PageImage, PageResult, Thresholds, VerdictReport


def record(name, persistence, *, missing=0, outside=0, line_loss=0., pixels=100):
    return dict(name=name, status='completed', assessment=dict(
        missing_readable_items=missing, outside_permission_pixels=outside,
        source_line_loss=line_loss, observed_pattern_persistence=persistence,
        selected_pixels=pixels))


def result(page, mask, thr, **kwargs):
    return PageResult(page, replace(page, image=page.image.copy()), mask,
                      VerdictReport(page.key, EngineKind.RASTER, [], {}, {}, [], 5),
                      processing={'thresholds': {'mask_alpha_min': thr.mask_alpha_min}})


class DecisionTests(unittest.TestCase):
    def test_visible_residue_outranks_smaller_selection(self):
        records = [record('original', .05, pixels=1000),
                   record('stroke_completion', .9, pixels=10)]
        self.assertEqual(auto.choose(records, 'stroke_completion')[0], 'original')

    def test_better_completion_is_also_selected(self):
        self.assertEqual(auto.choose([record('original', .8), record('stroke_completion', .1)],
                                     'original')[0], 'stroke_completion')

    def test_better_pattern_score_cannot_hide_missing_numbers(self):
        self.assertEqual(auto.choose([record('original', .05, missing=2),
                                      record('stroke_completion', .8)], 'stroke_completion')[0],
                         'stroke_completion')

    def test_better_pattern_score_cannot_hide_damaged_chart_lines(self):
        self.assertEqual(auto.choose([record('original', .05, line_loss=.2),
                                      record('stroke_completion', .8)], 'stroke_completion')[0],
                         'stroke_completion')

    def test_unknown_or_small_improvement_keeps_original_mode(self):
        for a, b in ((None, .8), (.75, .8)):
            self.assertEqual(auto.choose([record('original', a), record('stroke_completion', b)],
                                         'stroke_completion')[0], 'stroke_completion')
        self.assertEqual(auto.choose([record('original', .05, missing=None),
                                      record('stroke_completion', .8)], 'stroke_completion')[0],
                         'stroke_completion')

    def test_permission_violation_is_disqualified_even_with_best_score(self):
        self.assertEqual(auto.choose([record('original', .01, outside=1),
                                      record('stroke_completion', .8)], 'original')[0],
                         'stroke_completion')
        with self.assertRaises(RuntimeError):
            auto.choose([record('original', .01, outside=1)], 'original')


class ObservationTests(unittest.TestCase):
    def test_native_hough_line_shapes_and_table_edges(self):
        source = np.full((160, 300, 3), 230, np.uint8)
        cv2.line(source, (20, 70), (260, 70), (0, 0, 0), 2)
        empty = MaskLayer(np.zeros(source.shape[:2], np.float32))
        with patch('wm.auto_selection.ocr.read_lines', return_value=[]), \
             patch('wm.auto_selection.recall._lines', return_value=[]):
            evidence = auto.observe(source, [empty], .08)
            self.assertGreater(len(evidence.lines), 0)
            for shape in ((1, 4), (1, 1, 4)):
                segments = np.array([20, 68, 260, 68], np.int32).reshape(shape)
                with patch('wm.auto_selection.cv2.HoughLinesP', return_value=segments):
                    self.assertGreater(len(auto.observe(source, [empty], .08).lines), 0)

    def test_common_source_pattern_catches_leftover_outside_smaller_candidate(self):
        source = np.full((100, 250, 3), 230, np.uint8)
        glyph = np.zeros((100, 250), np.uint8)
        cv2.putText(glyph, 'DEMO', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        source[glyph > 0] = 20
        box = np.array([[5, 20], [240, 20], [240, 80], [5, 80]], np.float32)
        full = MaskLayer((glyph > 0).astype(np.float32))
        small = MaskLayer(((glyph > 0) & (np.indices(glyph.shape)[1] < 35)).astype(np.float32))
        with patch('wm.auto_selection.ocr.read_lines', return_value=[]), \
             patch('wm.auto_selection.recall._lines', return_value=[(box, 'DEMO', .99)]):
            evidence = auto.observe(source, [full, small], .08)
            self.assertGreater(len(evidence.patterns), 0)
            page = PageImage(source, 72)
            clean = result(page, full, Thresholds())
            clean.output = replace(page, image=np.full_like(source, 230))
            dirty = result(page, small, Thresholds())
            a = auto.assess(source, clean, evidence, Thresholds())
            b = auto.assess(source, dirty, evidence, Thresholds())
        self.assertLess(a['observed_pattern_persistence'], .1)
        self.assertGreater(b['observed_pattern_persistence'], .9)
        np.testing.assert_array_equal(full.alpha, (glyph > 0).astype(np.float32))

    def test_readable_number_is_checked_inside_candidate_permission(self):
        source = np.full((80, 160, 3), 240, np.uint8)
        box = np.array([[20, 20], [80, 20], [80, 50], [20, 50]], np.float32)
        evidence = auto.Evidence([], [(box, '194', 30.)], [], True)
        page = PageImage(source, 72)
        candidate = result(page, MaskLayer(np.ones((80, 160), np.float32)), Thresholds())
        with patch('wm.auto_selection.ocr.read_lines', return_value=[(box, '184', .99)]):
            assessment = auto.assess(source, candidate, evidence, Thresholds())
        self.assertEqual(assessment['missing_readable_items'], 1)

    def test_zero_expansion_detects_one_pixel_outside_permission(self):
        page = PageImage(np.full((20, 20, 3), 200, np.uint8), 72)
        alpha = np.zeros((20, 20), np.float32)
        alpha[5:10, 5:10] = 1
        candidate = result(page, MaskLayer(alpha), Thresholds())
        candidate.output.image[5, 10] = 0
        evidence = auto.Evidence([], [], [], False)
        self.assertEqual(auto.assess(page.image, candidate, evidence, Thresholds())['outside_permission_pixels'], 1)
        self.assertEqual(auto.assess(page.image, candidate, evidence,
                                    Thresholds(lama_mask_dilate=1))['outside_permission_pixels'], 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.page = PageImage(np.full((40, 60, 3), 200, np.uint8), 72, 'arbitrary-name')
        a = np.zeros((40, 60), np.float32)
        a[10:20, 10:20] = 1
        self.mask = MaskLayer(a, footprint=a > 0, veil_strength=np.ones(3, np.float32),
                              veil_offset=np.zeros(3, np.float32), method='contrast_dark_oblique')

    def test_identical_candidates_run_repair_once_and_remain_unverified(self):
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', side_effect=lambda image, mask, *a, **k: mask), \
             patch('wm.pipeline._process_one', side_effect=result) as process:
            chosen = auto.process(self.page, Thresholds(watermark_recall_priority=True), use_ocr=False)
        self.assertEqual(process.call_count, 1)
        self.assertTrue(chosen.verdict.review_needed)
        self.assertFalse(chosen.processing['automatic_selection']['verified_success'])
        self.assertEqual(len(chosen.comparison_results), 2)
        self.assertTrue(all(c is not chosen and not c.comparison_results for c in chosen.comparison_results))

    def test_source_evidence_selects_once_without_running_weaker_method(self):
        smaller = deepcopy(self.mask)
        smaller.alpha[12:20] = 0
        smaller.footprint = smaller.alpha > 0
        y, x = np.nonzero(self.mask.alpha)
        evidence = auto.Evidence([(y, x, 0, cv2.MORPH_BLACKHAT, 9,
                                   np.full(len(y), 100.))], [], [], False)
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', return_value=smaller), \
             patch('wm.auto_selection.observe', return_value=evidence), \
             patch('wm.pipeline._process_one', side_effect=result) as repair:
            chosen = auto.process(self.page, Thresholds(watermark_recall_priority=True))
        metadata = chosen.processing['automatic_selection']
        self.assertEqual(metadata['chosen'], 'original')
        self.assertEqual(repair.call_count, 1)
        self.assertEqual(metadata['repair_count'], 1)
        self.assertEqual(metadata['records'][1]['status'], 'not_run')
        self.assertEqual(len(chosen.comparison_results), 1)

    def test_unknown_source_keeps_mode_and_repairs_once(self):
        smaller = deepcopy(self.mask)
        smaller.alpha[12:20] = 0
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', return_value=smaller), \
             patch('wm.auto_selection.observe', return_value=auto.Evidence([], [], [], False)), \
             patch('wm.pipeline._process_one', side_effect=result) as repair:
            chosen = auto.process(self.page, Thresholds(watermark_recall_priority=True))
        self.assertEqual(chosen.processing['automatic_selection']['chosen'], 'stroke_completion')
        self.assertEqual(repair.call_count, 1)

    def test_failed_first_repair_tries_alternative_and_records_failure(self):
        smaller = deepcopy(self.mask)
        smaller.alpha[12:20] = 0
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', return_value=smaller), \
             patch('wm.auto_selection.observe', return_value=auto.Evidence([], [], [], False)), \
             patch('wm.pipeline._process_one', side_effect=[RuntimeError('first failed'),
                   result(self.page, smaller, Thresholds())]) as repair:
            chosen = auto.process(self.page, Thresholds())
        metadata = chosen.processing['automatic_selection']
        self.assertEqual(repair.call_count, 2)
        self.assertEqual(metadata['chosen'], 'stroke_completion')
        self.assertEqual(metadata['records'][0]['status'], 'error')
        self.assertTrue(metadata['fallback_reason'])
        self.assertEqual(metadata['status'], 'unmet')

    def test_second_repair_needs_source_support_not_only_residual_warning(self):
        r = record('original', .95)
        score = dict(source_pattern_coverage=.9, readable_overlap=0., line_overlap=0.)
        weaker = dict(score, source_pattern_coverage=.2)
        self.assertIsNone(auto.retry_reason(r, score, weaker))
        self.assertIsNotNone(auto.retry_reason(r, weaker, score))

    def test_preselection_does_not_trade_away_observed_body_text(self):
        y, x = np.nonzero(self.mask.alpha)
        smaller = deepcopy(self.mask)
        smaller.alpha[:] = 0
        box = np.array([[10, 10], [19, 10], [19, 19], [10, 19]], np.float32)
        evidence = auto.Evidence([(y, x, 0, cv2.MORPH_BLACKHAT, 9, np.ones(len(y)))],
                                 [(box, '194', 9.)], [], False)
        name, _, _ = auto.preselect([('original', self.mask), ('stroke_completion', smaller)],
                                     evidence, .08, 'stroke_completion')
        self.assertEqual(name, 'stroke_completion')

    def test_model_failure_is_not_reported_as_a_successful_original_image(self):
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', side_effect=lambda image, mask, *a, **k: mask), \
             patch('wm.pipeline._process_one', side_effect=RuntimeError('model missing')):
            with self.assertRaisesRegex(RuntimeError, 'model missing'):
                auto.process(self.page, Thresholds(), use_ocr=False)

    def test_all_permission_violations_retain_source_with_explicit_failure(self):
        def leaking(page, mask, thr, **kwargs):
            candidate = result(page, mask, thr)
            candidate.output.image[0, 0] = 0
            return candidate
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', side_effect=lambda image, mask, *a, **k: mask), \
             patch('wm.pipeline._process_one', side_effect=leaking):
            chosen = auto.process(self.page, Thresholds(), use_ocr=False)
        np.testing.assert_array_equal(chosen.output.image, self.page.image)
        self.assertTrue(chosen.processing['automatic_selection']['source_retained'])
        self.assertEqual(chosen.processing['automatic_selection']['status'], 'unmet')
        self.assertFalse(chosen.verdict.passed)

    def test_cancel_between_candidates_stops_without_delivering_partial_result(self):
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', side_effect=lambda image, mask, *a, **k: mask), \
             patch('wm.pipeline._process_one', side_effect=result):
            with self.assertRaises(InterruptedError):
                auto.process(self.page, Thresholds(), use_ocr=False, checkpoint=lambda: False)

    def test_unrequested_interruption_is_a_page_error_and_queue_continues(self):
        from wm.batches import Batch, BatchWorker
        batch = Batch('两页原始上下文', [self.page, replace(self.page, page_index=1)])
        worker = BatchWorker([(batch, [0, 1], Thresholds(), 11, False)])
        events = []
        worker.update.connect(events.append)
        with patch('wm.batches.structure.process_text_page', return_value=None), \
             patch('wm.batches.masks.estimate_crosspage', return_value=None), \
             patch('wm.batches.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.batches.pipeline._process_one', side_effect=[InterruptedError('读图中断'),
                    result(batch.pages[1], self.mask, Thresholds())]):
            worker.run()
        pages = [e for e in events if e['kind'] == 'page']
        self.assertEqual(pages[0]['error'], '读图中断')
        self.assertIsNotNone(pages[1]['result'])

    def test_candidate_mutation_does_not_change_cached_prior(self):
        prior = deepcopy(self.mask)
        before = prior.alpha.copy()
        def mutate(page, mask, thr, **kwargs):
            mask.alpha[:] = 0
            return result(page, mask, thr)
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', side_effect=lambda image, mask, *a, **k: mask), \
             patch('wm.masks.with_prior', return_value=prior), \
             patch('wm.pipeline._process_one', side_effect=mutate):
            auto.process(self.page, Thresholds(), prior=prior, use_ocr=False)
        np.testing.assert_array_equal(prior.alpha, before)

    def test_references_are_not_accessed_before_choice(self):
        selected = result(self.page, self.mask, Thresholds())
        selected.processing['automatic_selection'] = {'chosen': 'original'}
        class EvaluationOnly(dict):
            def __bool__(self):
                self.assert_chosen()
                return False
        gt = EvaluationOnly()
        gt.assert_chosen = lambda: self.assertTrue(process.called)
        with patch('wm.pipeline.structure.process_text_page', return_value=None), \
             patch('wm.auto_selection.process', return_value=selected) as process:
            pipeline.process_pages([self.page], gt=gt, automatic_selection=True)

    def test_export_keeps_both_actual_candidates_and_no_false_success(self):
        with patch('wm.masks.estimate_single_page', return_value=self.mask), \
             patch('wm.auto_selection.recall.expand', side_effect=lambda image, mask, *a, **k: mask), \
             patch('wm.pipeline._process_one', side_effect=result):
            chosen = auto.process(self.page, Thresholds(), use_ocr=False)
        with tempfile.TemporaryDirectory() as directory:
            import json
            pipeline.write_results([chosen], directory)
            manifest = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertIsNone(manifest['summary']['success_rate'])
            for candidate in manifest['pages'][0]['automatic_candidates']:
                for path in candidate['files'].values():
                    self.assertTrue((Path(directory)/path).is_file())


class DesktopEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wm.app import make_app
        cls.qt = make_app()

    def test_trial_entry_hides_completion_choice_and_submits_automatic_job(self):
        from wm.app import MainWindow
        from wm.batches import Batch
        win = MainWindow(automatic_selection=True)
        try:
            self.assertTrue(win.recall_box.isHidden())
            win.batches.append(Batch('任意批次', [PageImage(np.full((20, 30, 3), 240, np.uint8), 72)]))
            win.refresh_batches()
            win.batch_list.setCurrentRow(0)
            with patch('wm.app.BatchWorker') as worker:
                win.start_run()
                self.assertTrue(worker.call_args.args[0][0][6])
        finally:
            win.worker = None
            win.close()

    def test_existing_entry_preserves_its_original_default(self):
        from wm.app import MainWindow
        win = MainWindow()
        try:
            self.assertFalse(win.automatic_selection)
            self.assertFalse(win.recall_box.isHidden())
            self.assertTrue(win.recall_box.isChecked())
        finally:
            win.close()

    def test_headless_entries_honor_the_trial_flag(self):
        from wm import app
        for option, target in (('--shot', 'shot'), ('--smoke', 'smoke')):
            args = ['wm.app', '--experimental-auto-selection', option]
            if option == '--shot':
                args.append('temporary-output')
            with patch('sys.argv', args), patch.object(app, target, return_value=0) as entry:
                with self.assertRaises(SystemExit):
                    app.main()
                self.assertTrue(entry.call_args.kwargs['automatic_selection'])


if __name__ == '__main__':
    unittest.main()
