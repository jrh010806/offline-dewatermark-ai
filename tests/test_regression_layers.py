"""Tier scope tests: batch context, protected samples, and no partial promotion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import ExitStack, redirect_stdout
from io import StringIO

from tools.regression_guard import apply_reviews, compare_subset
from tools.regression_layers import changed_files, expand_batches, jobs_for, make_plan, minimum_level

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / 'config/regression_tiers.json').read_text())
POLICY = json.loads((ROOT / 'config/regression_control.json').read_text())


def fixture():
    cases = [dict(id=name, batch=batch, group=group, style=style, page_count=pages,
                  input_sha256=name, negative=style == 'no_watermark')
             for name, batch, group, style, pages in (
        ('fixed/a', 'a', '固定回归', 'dot_grid', 1),
        ('fixed/negative', 'n', '固定回归', 'no_watermark', 1),
        ('fixed/pdf', 'pdf', '固定回归', 'independent_text', 2),
        ('manual/1', 'manual', '开发', 'tiled_irregular', 1),
        ('manual/2', 'manual', '开发', 'tiled_irregular', 1))]
    rules = copy.deepcopy(RULES)
    rules.update(protection_seeds=['fixed/a', 'fixed/negative', 'fixed/pdf'],
                 targeted_mode_probes=['fixed/negative', 'fixed/pdf'],
                 cross_mode_probes=['fixed/negative', 'fixed/pdf', 'manual/1'])
    profiles = {m: {'recall': m == 'desktop'} for m in POLICY['required_modes']}
    rows = []
    for mode in profiles:
        for c in cases:
            for p in range(1, c['page_count'] + 1):
                rows.append(dict(key=f"{mode}:{c['id']}:{p}", mode=mode, style=c['style'],
                    status='completed', input_sha256=c['id'], reference_identity='ref', settings=profiles[mode],
                    recall=.99, strong_recall=1., precision=.99, false_positive_pixels=1,
                    hole_selected_pixels=0, background_mae=0., watermark_mae=1., selected_pixels=50,
                    changed_pixels=50, outside_selection_changed_pixels=0, outside_truth_changed_pixels=0,
                    elapsed_ms=1000, output_sha256='output', mask_sha256='mask',
                    method='tonal', repair_mode='hybrid', repairs=['alpha_unmix'], negative=False))
    old = dict(scope='all', records=rows, expected_keys=[r['key'] for r in rows], complete=True,
               integrity_verified=True, profiles=profiles, inventory_sha256='inv', policy_sha256='policy',
               code_hashes={'wm/paper_grid.py': 'old'}, runtime_identity={'model': 'fixed'})
    return dict(cases=cases), old, rules


def candidate():
    inv, old, rules = fixture()
    plan = make_plan(['wm/paper_grid.py'], inv, old, rules)
    new = copy.deepcopy(old)
    new.update(scope=plan['level'], selection_plan=plan,
               records=[r for r in new['records'] if r['key'] in plan['expected_keys']],
               expected_keys=plan['expected_keys'], code_hashes={'wm/paper_grid.py': 'new'})
    return inv, old, rules, new


class LayerTests(unittest.TestCase):
    def test_full_gate_requires_engineering_preflight(self):
        from tools import daily_check
        for unit_code in (0, 1):
            with self.subTest(unit_code=unit_code), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                directory = Path(tmp)
                state = directory / 'state.json'
                state.write_text('{}')
                plan = dict(level='full', files=['README.md'], page_mode_count=3,
                            mode_case_ids={'desktop': ['fixture']}, estimated_seconds=30, reasons=[])
                def mocked_load(path):
                    return {'path': str(directory / 'baseline'), 'seal': 'sealed'} if path == state else {}
                stack.enter_context(patch.object(daily_check, 'STATE', state))
                stack.enter_context(patch.object(daily_check, 'load_profile', return_value=None))
                stack.enter_context(patch.object(daily_check, 'load', side_effect=mocked_load))
                stack.enter_context(patch.object(daily_check, 'digest', return_value='sealed'))
                stack.enter_context(patch.object(daily_check, 'changed_files', return_value=['README.md']))
                stack.enter_context(patch.object(daily_check, 'make_plan', return_value=plan))
                stack.enter_context(patch.object(daily_check, 'watch_sources', return_value={}))
                stack.enter_context(patch.object(daily_check, 'document_links', return_value={'missing': []}))
                events = []
                def unit(*args, **kwargs):
                    events.append('engineering')
                    return SimpleNamespace(returncode=unit_code)
                def full(*args, **kwargs):
                    events.append('full')
                    return 0
                stack.enter_context(patch.object(daily_check.subprocess, 'run', side_effect=unit))
                stack.enter_context(patch.object(daily_check, 'control_main', side_effect=full))
                stack.enter_context(redirect_stdout(StringIO()))
                code = daily_check.main(['--level', 'full', '--out', str(directory / 'run')])
                self.assertEqual(events, ['engineering', 'full'] if unit_code == 0 else ['engineering'])
                self.assertEqual(code, unit_code)

    def test_change_classes_and_unknown_application_files(self):
        for path, expected in [('README.md', 'static'), ('wm/viewer.py', 'unit'),
                               ('tools/regression_guard.py', 'unit'), ('wm/paper_grid.py', 'targeted'),
                               ('wm/pipeline.py', 'cross'), ('wm/new_detector.py', 'cross')]:
            with self.subTest(path=path):
                self.assertEqual(minimum_level([path], RULES)[0], expected)

    def test_explicit_level_cannot_hide_broader_change(self):
        inv, old, rules = fixture()
        with self.assertRaises(ValueError):
            make_plan(['wm/pipeline.py'], inv, old, rules, 'unit')

    def test_target_addition_keeps_entire_original_batch(self):
        inv, old, rules = fixture()
        plan = make_plan(['wm/paper_grid.py'], inv, old, rules, targets=['manual/1'])
        self.assertIn('manual/1', plan['mode_case_ids']['desktop'])
        self.assertIn('manual/2', plan['mode_case_ids']['desktop'])
        self.assertIn('fixed/negative', plan['mode_case_ids']['desktop'])
        self.assertIn('fixed/pdf', plan['mode_case_ids']['desktop'])

    def test_missing_target_is_an_error_not_silently_skipped(self):
        inv, old, rules = fixture()
        with self.assertRaises(ValueError):
            make_plan([], inv, old, rules, targets=['unknown'])

    def test_jobs_preserve_per_mode_plan_and_batch_order(self):
        inv, old, rules = fixture()
        plan = make_plan(['wm/pipeline.py'], inv, old, rules)
        run = dict(profiles=old['profiles'], scope='cross', selection_plan=plan)
        jobs = jobs_for(inv, run)
        manual = [rows for mode, batch, rows in jobs if batch == 'manual']
        self.assertEqual(len(manual), 3)
        self.assertTrue(all([r['id'] for r in rows] == ['manual/1', 'manual/2'] for rows in manual))

    def test_full_plan_preserves_all_existing_modes_and_pages(self):
        inv, old, rules = fixture()
        plan = make_plan([], inv, old, rules, 'full')
        self.assertEqual(set(plan['expected_keys']), set(old['expected_keys']))

    def test_partial_can_pass_locally_but_never_release(self):
        inv, old, rules, new = candidate()
        result = compare_subset(new, old, POLICY, inv, rules)
        self.assertTrue(result['check_passed'])
        self.assertFalse(result['gate_passed'])
        self.assertFalse(result['release_eligible'])

    def test_removing_a_protected_sample_or_probe_blocks(self):
        inv, old, rules, new = candidate()
        new['records'].pop()
        new['expected_keys'] = [r['key'] for r in new['records']]
        result = compare_subset(new, old, POLICY, inv, rules)
        self.assertFalse(result['check_passed'])

    def test_forging_plan_to_remove_a_protected_sample_blocks(self):
        inv, old, rules, new = candidate()
        new['selection_plan']['mode_case_ids']['desktop'].remove('fixed/a')
        self.assertFalse(compare_subset(new, old, POLICY, inv, rules)['check_passed'])

    def test_actual_unlisted_application_change_blocks(self):
        inv, old, rules, new = candidate()
        new['code_hashes']['wm/pipeline.py'] = 'hidden change'
        self.assertFalse(compare_subset(new, old, POLICY, inv, rules)['check_passed'])

    def test_visual_review_cannot_turn_partial_result_into_release(self):
        inv, old, rules, new = candidate()
        new['records'][0]['output_sha256'] = 'changed output'
        result = compare_subset(new, old, POLICY, inv, rules)
        reviews = dict(candidate_seal='a', baseline_seal='b', cases=[dict(
            key=new['records'][0]['key'], decision='accept_change', reviewer='复核者', reason='逐图核对')])
        result = apply_reviews(result, reviews, 'a', 'b')
        self.assertTrue(result['check_passed'])
        self.assertFalse(result['gate_passed'])

    def test_runtime_or_numerical_regression_cannot_be_waived(self):
        inv, old, rules, new = candidate()
        new['records'][0]['recall'] = .5
        self.assertFalse(compare_subset(new, old, POLICY, inv, rules)['check_passed'])
        new['runtime_identity'] = {'model': 'other'}
        self.assertTrue(compare_subset(new, old, POLICY, inv, rules)['blockers'])

    def test_no_inference_for_static_or_unit_scope(self):
        inv, old, rules = fixture()
        for files in (['README.md'], ['wm/viewer.py']):
            plan = make_plan(files, inv, old, rules)
            self.assertEqual(plan['page_mode_count'], 0)
            self.assertEqual(plan['mode_case_ids'], {})


if __name__ == '__main__':
    unittest.main()
