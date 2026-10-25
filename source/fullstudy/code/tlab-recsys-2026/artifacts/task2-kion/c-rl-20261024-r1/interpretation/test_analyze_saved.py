"""Tiny source/synthetic checks: never open campaign arrays/models or raw data."""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name('analyze_saved.py')
spec = importlib.util.spec_from_file_location('saved_i0', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SavedAuditTest(unittest.TestCase):
    def test_rank_metrics_full_denominator(self):
        # Covered first, unknown, offpool and covered rank11: all remain in N.
        ranks = [1, 0, 0, 11]
        self.assertEqual(sum(mod.metric(rank)['map10'] for rank in ranks)/4, .25)
        self.assertEqual(mod.metric(10)['ndcg10'], 1/math.log2(11))
        self.assertEqual(mod.metric(11)['recall10'], 0)

    def test_identical_lists_have_ten_effective_items(self):
        result = mod.exposure([list(range(10))]*3, [1., 1., 1.], 100)
        self.assertEqual(result['top1_max_item_share'], 1.)
        self.assertAlmostEqual(result['top10_hhi'], .1)
        self.assertAlmostEqual(result['effective_catalog_items'], 10)
        self.assertEqual(result['catalog_coverage'], .1)

    def test_equal_user_weights_change_concentration(self):
        tops = [list(range(10)), list(range(10)), list(range(10, 20))]
        query = mod.exposure(tops, [1, 1, 1], 100)
        user = mod.exposure(tops, [.5, .5, 1], 100)
        self.assertAlmostEqual(query['top1_max_item_share'], 2/3)
        self.assertAlmostEqual(user['top1_max_item_share'], .5)
        self.assertLess(user['top10_hhi'], query['top10_hhi'])

    def test_duplicate_top10_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid top10'):
            mod.exposure([[0]*10], [1], 20)

    def test_case_selection_is_deterministic_distinct_and_shortfall_explicit(self):
        rows = [{'row_index': j, 'C_rr10': .5, 'B_rr10': 0 if j < 10 else 1,
                 'GPT_rr10': 0 if j % 2 else 1} for j in range(40)]
        selected = mod.select_cases(rows)
        self.assertEqual(selected, mod.select_cases(list(reversed(rows))))
        ids = [entry['row']['row_index'] for entry in selected['cases']]
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)
        # First cell follows predetermined hash order, not gain magnitude.
        expected = sorted(range(10), key=lambda j: hashlib.sha256(
            ('c3-I0-cases-v1|' + str(j)).encode()).hexdigest())[:3]
        self.assertEqual(ids[:3], expected)
        empty = mod.select_cases([])
        self.assertTrue(all(cell['shortfall'] == 3 for cell in empty['cells']))

    def test_old_audit_uses_own_cohort_latest_and_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._old_fixture(root)
            results = mod.old_audit(root)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['selected_update'], 0)
            self.assertAlmostEqual(results[0]['latest_minus_update0_ndcg10'], -.1)
            self.assertEqual(results[0]['additional_behavior_metrics']['state'], 'unavailable')

    def test_old_cross_cohort_comparison_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._old_fixture(root, mismatch=True)
            with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
                mod.old_audit(root)

    @staticmethod
    def _old_fixture(root, mismatch=False):
        for name in ('run-rl-r1', 'run-rl-recovered-r1'):
            path = root/'rl'/name/'worker'/'training'/'screening.json'
            path.parent.mkdir(parents=True)
            history = []
            for step, ndcg in ((0, .2), (1000, .1)):
                history.append({'update': step, 'evaluation': {'policy': {'ndcg@10': ndcg},
                    'cohort_sha256': 'other' if mismatch and step == 1000 else 'same-cohort',
                    'protocol': 'own frozen128'}})
            path.write_text(json.dumps({'state': {'history': history, 'selected_update': 0,
                                                 'updates': 1000}}), encoding='utf-8')

    def test_runtime_import_boundary(self):
        import ast
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        forbidden = {'torch', 'catboost', 'shap', 'captum', 'subprocess', 'requests'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(forbidden & {alias.name.split('.')[0] for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], forbidden)
        self.assertNotIn('numpy', __import__('sys').modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
