"""Pure stdlib toy tests: no real arrays, NumPy, models or native jobs."""
import ast
import importlib.util
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('analyze_rl.py')
spec = importlib.util.spec_from_file_location('rl_analysis', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class RLAnalysisTests(unittest.TestCase):
    def test_raw_id_tie_order_ignores_pool_column_and_token(self):
        self.assertEqual(mod.rank_oracle([10, 5, 8], [2., 2., 1.], [200, 100, 3]), [5, 10, 8])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            mod.rank_oracle([3], [math.nan], [50])

    def test_unknown_offpool_and_rank11_stay_zero(self):
        top = list(range(3, 13))
        self.assertEqual(mod.row_metrics(top, 3, False), (0., 0., 0.))
        self.assertEqual(mod.row_metrics(top, 13, True), (0., 0., 0.))
        rows = [mod.row_metrics(top, target, known) for target, known in ((3, True), (1, False), (99, True), (13, True))]
        self.assertEqual(sum(row[0] for row in rows)/4, .25)

    def test_query_vs_equal_user_exposure(self):
        top = [list(range(10)), list(range(10)), list(range(10, 20))]
        query, user = mod.exposure(top, [1, 1, 1], 100), mod.exposure(top, [.5, .5, 1], 100)
        self.assertAlmostEqual(query['top1_max_item_share'], 2/3)
        self.assertAlmostEqual(user['top1_max_item_share'], .5)
        self.assertAlmostEqual(user['effective_catalog_items'], 20)
        self.assertEqual(user['unique_top10_items'], 20)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            mod.exposure([[1]*10], [1], 100)

    def test_cluster_ratio_keeps_all_queries_and_user_multiplicities(self):
        # User A has2 queries, B has1. This is not equal-user averaging.
        users, values = ['A', 'A', 'B'], [1., 0., -1.]
        self.assertAlmostEqual(mod.cluster_oracle(users, values, ['A', 'B']), 0.)
        self.assertAlmostEqual(mod.cluster_oracle(users, values, ['A', 'A']), .5)
        self.assertAlmostEqual(mod.cluster_oracle(users, values, ['B', 'B']), -1.)
        self.assertAlmostEqual(mod.cluster_oracle(users, [0., 0., 0.], ['A', 'B']), 0.)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            mod.cluster_oracle(users, values, ['A', 'C'])

    def test_actual_latest_and_selected0_are_separate(self):
        training = {'state': 'completed', 'updates': 1000, 'supervised_updates': 0, 'frozen_encoder': True,
                    'checkpoints': [{'sha256': 'zero', 'path': 'head-0.pt', 'step': 0},
                                    {'sha256': 'last', 'path': 'head-1000.pt', 'step': 1000}]}
        evaluation = {'checkpoints': {'latest': {'sha256': 'last', 'path': 'head-1000.pt'},
                                      'selected': {'sha256': 'zero', 'path': 'head-0.pt'}}}
        self.assertEqual(mod.checkpoint_steps(evaluation, training), {'latest': 1000, 'selected': 0})
        evaluation['checkpoints']['latest'] = evaluation['checkpoints']['selected']
        with self.assertRaisesRegex(ValueError, 'actual final'):
            mod.checkpoint_steps(evaluation, training)

    def test_selection_can_be_trained_checkpoint(self):
        training = {'state': 'completed', 'updates': 1000, 'supervised_updates': 0, 'frozen_encoder': True,
                    'checkpoints': [{'sha256': 'middle', 'path': 'head-500.pt', 'step': 500},
                                    {'sha256': 'last', 'path': 'head-1000.pt', 'step': 1000}]}
        evaluation = {'checkpoints': {'latest': {'sha256': 'last', 'path': 'head-1000.pt'},
                                      'selected': {'sha256': 'middle', 'path': 'head-500.pt'}}}
        self.assertEqual(mod.checkpoint_steps(evaluation, training)['selected'], 500)

    def test_numerical_import_is_lazy_and_no_ML_process_import(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        top_imports = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        self.assertNotIn('numpy', top_imports)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({'torch', 'shap', 'catboost', 'subprocess', 'requests'} & {a.name for a in node.names})
        self.assertEqual((mod.REPLICATES, mod.SEED), (2000, 20261024))


if __name__ == '__main__':
    unittest.main(verbosity=2)
