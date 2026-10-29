"""Only explicitly removed samples may be projected out of sealed evidence."""
import copy
import unittest
from tools.regression_guard import canonical, compare_runs
from tools.regression_retirements import project_baseline
from tests.test_regression_layers import fixture, POLICY


class RetirementTests(unittest.TestCase):
    def versions(self):
        inventory, run, _ = fixture()
        for row in run['records']:
            row['id'] = row['key'].split(':')[1]
        inventory.update(groups={'固定回归':3,'开发':2},total_inputs=5,total_pages=6,unique_input_hashes=5)
        run['inventory_sha256'] = canonical(inventory)
        profile = dict(version='explicit',baseline_seal='sealed',cases=[dict(inventory_id='manual/1',input_sha256='manual/1')])
        return inventory, run, profile

    def test_projection_keeps_other_pages_and_original_immutable(self):
        inventory, run, profile = self.versions()
        saved = copy.deepcopy((run,inventory))
        projected, active = project_baseline(run,inventory,profile,'sealed')
        self.assertEqual((run,inventory),saved)
        self.assertEqual(len(projected['records']),len(run['records'])-3)
        self.assertEqual(set(projected['expected_keys']),{r['key'] for r in projected['records']})
        self.assertIn('manual/2',{r['id'] for r in active['cases']})
        self.assertIn('fixed/negative',{r['id'] for r in active['cases']})
        self.assertEqual(projected['inventory_sha256'],canonical(active))

    def test_wrong_baseline_or_input_hash_is_rejected(self):
        inventory, run, profile = self.versions()
        with self.assertRaises(ValueError):project_baseline(run,inventory,profile,'other')
        profile['cases'][0]['input_sha256']='different'
        with self.assertRaises(ValueError):project_baseline(run,inventory,profile,'sealed')

    def test_already_active_baseline_is_unchanged(self):
        inventory, run, profile = self.versions()
        projected,active=project_baseline(run,inventory,profile,'sealed')
        self.assertEqual(project_baseline(projected,active,profile,'newer'),(projected,active))

    def test_retained_inventory_change_still_blocks(self):
        inventory, run, profile = self.versions()
        projected,_=project_baseline(run,inventory,profile,'sealed')
        candidate=copy.deepcopy(projected)
        self.assertTrue(compare_runs(candidate,projected,POLICY)['gate_passed'])
        candidate['inventory_sha256']='unrelated inventory change'
        self.assertFalse(compare_runs(candidate,projected,POLICY)['gate_passed'])

    def test_missing_retained_result_still_blocks(self):
        inventory, run, profile = self.versions()
        run['records'].pop()
        projected,_=project_baseline(run,inventory,profile,'sealed')
        self.assertFalse(compare_runs(projected,projected,POLICY)['gate_passed'])


if __name__ == '__main__':unittest.main()
