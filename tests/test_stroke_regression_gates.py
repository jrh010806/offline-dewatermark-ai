import copy
import unittest

from tools.check_stroke_regressions import evaluate


LIMITS = dict(recall_drop=.01, precision_drop=.01,
              false_positive_increase_pixels_and_ratio=[50, .05],
              background_mae_increase=.05, watermark_mae_increase=.5)


def pair(identifier, recall=.9):
    baseline = dict(status='completed', recall=recall, precision=.95,
                    false_positive_pixels=100, hole_selected_pixels=0,
                    background_mae=.1, watermark_mae=3,
                    selected_pixels=1000, changed_pixels=900)
    return dict(id=identifier, baseline=baseline, **copy.deepcopy(baseline))


class StrokeRegressionGateTests(unittest.TestCase):
    def test_another_cases_gain_cannot_cancel_one_cases_regression(self):
        good, bad = pair('gain'), pair('regression')
        good['recall'] = 1.
        bad['recall'] = .88
        result = evaluate([good, bad], LIMITS)
        self.assertFalse(result['quality_regression_gate_passed'])
        self.assertEqual(result['cases'][0]['failures'], [])
        self.assertIn('召回率', result['cases'][1]['failures'][0])

    def test_previously_safe_negative_cannot_be_modified_even_without_selection(self):
        row = pair('clean')
        row.update(negative=True, recall=None, precision=None,
                   selected_pixels=0, changed_pixels=1, false_positive_pixels=0)
        row['baseline'].update(recall=None, precision=None,
                               selected_pixels=0, changed_pixels=0, false_positive_pixels=0)
        result = evaluate([row], LIMITS)
        self.assertFalse(result['quality_regression_gate_passed'])
        self.assertTrue(any('负例' in message for message in result['cases'][0]['failures']))

    def test_better_pixel_metrics_do_not_override_confirmed_repair_damage(self):
        row = pair('broken-line')
        row.update(recall=1., precision=1., false_positive_pixels=0,
                   manual_repair_regression=True)
        self.assertFalse(evaluate([row], LIMITS)['quality_regression_gate_passed'])

    def test_hole_false_positives_cannot_be_hidden_in_a_small_total(self):
        row = pair('filled-hole')
        row['hole_selected_pixels'] = 1
        row['false_positive_pixels'] = 1
        self.assertFalse(evaluate([row], LIMITS)['quality_regression_gate_passed'])


if __name__ == '__main__':
    unittest.main()
