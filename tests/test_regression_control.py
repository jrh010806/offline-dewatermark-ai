"""Controller safety tests: unchanged samples, failures, missing data and reviews."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.regression_guard import apply_reviews, compare_page, compare_runs
from tools.regression_control import save, seal, verified_run, verify_frozen

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / 'config/regression_control.json').read_text())


def page(key='desktop:sample:1'):
    return dict(key=key, mode=key.split(':')[0], style='文字', status='completed',
                input_sha256='source', reference_identity='truth', settings={'recall': True},
                recall=.9, precision=.98, strong_recall=.95, false_positive_pixels=20,
                hole_selected_pixels=0, background_mae=.1, watermark_mae=5,
                selected_pixels=100, changed_pixels=100, outside_selection_changed_pixels=0,
                outside_truth_changed_pixels=20, elapsed_ms=1000,
                output_sha256='output', mask_sha256='mask', method='repeated',
                repair_mode='paper', repairs=['paper_cleanup'], negative=False)


def run():
    rows = [page(m + ':sample:1') for m in POLICY['required_modes']]
    return dict(inventory_sha256='inventory', policy_sha256='policy', scope='all',
                profiles={m: {} for m in POLICY['required_modes']}, records=rows,
                expected_keys=[r['key'] for r in rows], complete=True, integrity_verified=True,
                runtime_identity={'models': 'frozen'})


class GuardTests(unittest.TestCase):
    def test_identical_observation_passes_without_claiming_quality_or_independence(self):
        result = compare_runs(run(), run(), POLICY)
        self.assertTrue(result['gate_passed'])
        self.assertFalse(result['quality_approved'])
        self.assertFalse(result['independent_validation'])

    def test_one_regression_cannot_be_cancelled_by_other_improvements(self):
        before = run()
        after = copy.deepcopy(before)
        after['records'][0]['recall'] = .88
        after['records'][1]['recall'] = 1.
        result = compare_runs(after, before, POLICY)
        self.assertFalse(result['gate_passed'])
        self.assertEqual(result['failed_pages'], 1)

    def test_safe_negative_must_not_change(self):
        before = page()
        before.update(negative=True, selected_pixels=0, changed_pixels=0)
        after = dict(before, changed_pixels=1)
        self.assertTrue(compare_page(after, before, POLICY)['failures'])

    def test_unknown_metrics_do_not_become_zero(self):
        before = page()
        for k in ('recall', 'precision', 'strong_recall', 'background_mae', 'watermark_mae',
                  'false_positive_pixels', 'hole_selected_pixels', 'outside_truth_changed_pixels'):
            before[k] = None
        after = dict(before, output_sha256='changed')
        result = compare_page(after, before, POLICY)
        self.assertTrue(result['review'])
        self.assertIsNone(after['watermark_mae'])

    def test_missing_previously_measured_metric_blocks(self):
        self.assertTrue(compare_page(dict(page(), recall=None), page(), POLICY)['failures'])

    def test_existing_damage_is_disclosed_but_new_damage_blocks(self):
        before = dict(page(), outside_selection_changed_pixels=9)
        same = compare_page(before, before, POLICY)
        self.assertFalse(same['failures'])
        self.assertTrue(same['known'])
        after = dict(before, outside_selection_changed_pixels=10)
        self.assertTrue(compare_page(after, before, POLICY)['failures'])

    def test_any_output_mask_or_route_change_requires_visual_review(self):
        for key, value in [('output_sha256', 'new'), ('mask_sha256', 'new'),
                           ('method', 'new'), ('repair_mode', 'new'), ('repairs', ['lama'])]:
            with self.subTest(key=key):
                result = compare_page(dict(page(), **{key: value}), page(), POLICY)
                self.assertTrue(result['review'])

    def test_missing_extra_and_duplicate_pages_block(self):
        for kind in ('missing', 'extra', 'duplicate'):
            after = run()
            if kind == 'missing':
                after['records'].pop()
            else:
                row = page('cli:extra:1') if kind == 'extra' else copy.deepcopy(after['records'][0])
                after['records'].append(row)
            self.assertFalse(compare_runs(after, run(), POLICY)['gate_passed'])

    def test_errors_in_either_version_block(self):
        for side in ('new', 'old'):
            a, b = run(), run()
            (a if side == 'new' else b)['records'][0]['status'] = 'error'
            self.assertFalse(compare_runs(a, b, POLICY)['gate_passed'])

    def test_partial_unsealed_or_mode_mismatch_cannot_pass(self):
        for key, value in [('complete', False), ('integrity_verified', False),
                           ('scope', 'quick'), ('profiles', {'cli': {}}),
                           ('inventory_sha256', 'other'), ('policy_sha256', 'changed'),
                           ('runtime_identity', {'models': 'other'})]:
            with self.subTest(key=key):
                self.assertFalse(compare_runs(dict(run(), **{key: value}), run(), POLICY)['gate_passed'])

    def test_empty_run_cannot_pass(self):
        a = dict(run(), records=[], expected_keys=[])
        self.assertFalse(compare_runs(a, a, POLICY)['gate_passed'])

    def test_batch_and_parameter_changes_cannot_be_compared(self):
        self.assertTrue(compare_page(dict(page(), settings={'recall': False}), page(), POLICY)['failures'])

    def test_slowdown_and_new_budget_breach_block(self):
        self.assertTrue(compare_page(dict(page(), elapsed_ms=10000), page(), POLICY)['failures'])
        self.assertTrue(compare_page(dict(page(), elapsed_ms=120001), page(), POLICY)['failures'])

    def test_review_is_version_bound_and_cannot_override_numeric_damage(self):
        a, b = run(), run()
        a['records'][0].update(output_sha256='changed', recall=.8)
        result = compare_runs(a, b, POLICY)
        reviews = dict(candidate_seal='candidate', baseline_seal='baseline', cases=[
            dict(key=a['records'][0]['key'], decision='accept_change', reviewer='复核者', reason='检查过')])
        checked = apply_reviews(result, reviews, 'candidate', 'baseline')
        self.assertEqual(checked['review_pages'], 0)
        self.assertFalse(checked['gate_passed'])
        with self.assertRaises(ValueError):
            apply_reviews(result, reviews, 'different', 'baseline')

    def test_visual_review_requires_named_reviewer_and_reason(self):
        a, b = run(), run()
        a['records'][0]['output_sha256'] = 'changed'
        result = compare_runs(a, b, POLICY)
        review = dict(candidate_seal='a', baseline_seal='b', cases=[dict(
            key=a['records'][0]['key'], decision='accept_change', reviewer='', reason='')])
        self.assertFalse(apply_reviews(result, review, 'a', 'b')['gate_passed'])
        review['cases'][0].update(reviewer='复核者', reason='逐图检查无新增损伤')
        self.assertTrue(apply_reviews(result, review, 'a', 'b')['gate_passed'])

    def test_deleted_or_modified_sealed_artifact_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            save(out / '运行记录.json', run())
            artifact = out / 'output.png'
            artifact.write_bytes(b'original')
            seal(out)
            self.assertEqual(verified_run(out), run())
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verified_run(out)
            artifact.unlink()
            with self.assertRaises(ValueError):
                verified_run(out)

    def test_frozen_code_cannot_change_during_run(self):
        with tempfile.TemporaryDirectory() as directory:
            from tools.regression_guard import digest
            out = Path(directory)
            code = out / 'algorithm.py'
            code.write_text('version 1')
            save(out / '冻结完整性.json', {'algorithm.py': digest(code)})
            verify_frozen(out)
            code.write_text('version 2')
            with self.assertRaises(ValueError):
                verify_frozen(out)


if __name__ == '__main__':
    unittest.main()
