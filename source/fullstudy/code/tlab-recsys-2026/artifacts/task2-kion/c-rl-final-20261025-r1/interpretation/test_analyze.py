"""Stdlib-only synthetic/source tests; never read real arrays or import NumPy."""
import ast
import importlib.util
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name('analyze.py')
spec = importlib.util.spec_from_file_location('scientific3_analysis', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def evaluation(stage='screen'):
    n = 256 if stage == 'screen' else 4096
    return {'contract': 'kion-scientific3-evaluation-v1', 'state': 'completed', 'stage': stage,
            'rows': n, 'metric_denominator': n, 'candidate_width': 1024,
            'temperature': 1., 'tie_break': 'score descending, raw item ID ascending',
            'final_test_reads': 0,
            'no_target_injection': True, 'no_seen_filter': True, 'unknown_and_offpool_are_zero': True,
            'original_C_parity': {'score_tolerance_passed': True, 'changes_with_margin_above_tolerance': 0},
            'checkpoints': {arm: {'step': 1000, 'sha256': 'a' * 64} for arm in mod.ARMS[1:]},
            'metrics': {arm: {} for arm in mod.ARMS}}


class AnalysisTests(unittest.TestCase):
    def test_raw_id_tie_is_independent_of_pool_order_and_token(self):
        self.assertEqual(mod.rank_oracle([10, 5, 8], [2., 2., 1.], [200, 100, 3]), [5, 10, 8])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            mod.rank_oracle([10, 10], [2., 1.], [2, 1])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            mod.rank_oracle([3], [math.nan], [50])

    def test_complete_target_ranks_and_zero_denominator_cases(self):
        order = list(range(3, 1027))
        self.assertIsNone(mod.target_rank(order, 3, False))
        self.assertIsNone(mod.target_rank(order, 9999, True))
        self.assertEqual(mod.target_rank(order, 1026, True), 1024)
        self.assertEqual(mod.row_metrics(1), (1., 1., 1.))
        self.assertEqual(mod.row_metrics(11), (0., 0., 0.))
        self.assertEqual(mod.row_metrics(None), (0., 0., 0.))
        rows = [mod.row_metrics(rank) for rank in (1, None, None, 11)]
        self.assertEqual(sum(row[0] for row in rows) / 4, .25)
        self.assertEqual(mod.rank_bin(None, False), 'unknown')
        self.assertEqual(mod.rank_bin(None, True), 'offpool')
        for rank, expected in ((1, '1'), (3, '2–3'), (10, '4–10'), (50, '11–50'), (100, '51–100'), (1024, '101–1024')):
            self.assertEqual(mod.rank_bin(rank, True), expected)

    def test_fixed_delta_bins_boundary_counts(self):
        values = [-1., -.5, -.1, -.01, 0., .01, .1, .5, 1.]
        expected = ['[-1,-0.5]', '[-1,-0.5]', '(-0.5,-0.1]', '(-0.1,0)', '0',
                    '(0,0.1]', '(0,0.1]', '(0.1,0.5]', '(0.5,1]']
        self.assertEqual([mod.delta_bin(value) for value in values], expected)

    def test_query_and_equal_user_exposure_have_distinct_denominators(self):
        tops = [list(range(10)), list(range(10)), list(range(10, 20))]
        query = mod.exposure(tops, [1, 1, 1], 100)
        user = mod.exposure(tops, [.5, .5, 1], 100)
        self.assertAlmostEqual(query['top1_max_item_share'], 2 / 3)
        self.assertAlmostEqual(user['top1_max_item_share'], .5)
        self.assertAlmostEqual(user['effective_catalog_items'], 20)
        self.assertAlmostEqual(user['top10_hhi'], .05)
        self.assertEqual(user['unique_top10_items'], 20)
        self.assertAlmostEqual(user['catalog_coverage'], .2)
        self.assertAlmostEqual(sum(item['share'] for item in user['sorted_top10_exposure']), 1.)
        self.assertIsNone(mod.exposure(tops, [1, 1, 1])['catalog_coverage'])
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            mod.exposure([[1] * 10], [1], 100)

    def test_cluster_ratio_keeps_all_queries_and_user_multiplicity(self):
        users, values = ['A', 'A', 'B'], [1., 0., -1.]
        self.assertAlmostEqual(mod.cluster_oracle(users, values, ['A', 'B']), 0.)
        self.assertAlmostEqual(mod.cluster_oracle(users, values, ['A', 'A']), .5)
        self.assertAlmostEqual(mod.cluster_oracle(users, values, ['B', 'B']), -1.)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            mod.cluster_oracle(users, values, ['A', 'C'])

    def test_supported_neutral_centering_uses_C_not_each_arm(self):
        ref, utility, support = [.2, .3, .5], [.8, .2, math.nan], [True, True, False]
        advantage = mod.neutral_oracle(ref, utility, support)
        self.assertAlmostEqual(advantage[0], .36)
        self.assertAlmostEqual(advantage[1], -.24)
        self.assertEqual(advantage[2], 0.)
        self.assertAlmostEqual(sum(p * a for p, a in zip(ref, advantage)), 0.)
        self.assertEqual(mod.neutral_oracle(ref, [math.nan] * 3, [False] * 3), [0., 0., 0.])

    def test_exact_item_contribution_includes_supported_mass_shift(self):
        ref, right, utility, support = [.2, .3, .5], [.4, .5, .1], [.8, .2, 0.], [True, True, False]
        contributions = mod.shift_oracle(ref, ref, right, utility, support)
        self.assertAlmostEqual(contributions[0], .072)
        self.assertAlmostEqual(contributions[1], -.048)
        self.assertEqual(contributions[2], 0.)
        self.assertAlmostEqual(sum(contributions), .024)
        # The unsupported action's mass changed: subtracting conditional means is a different estimand.
        conditional_difference = (.4 * .8 + .5 * .2) / .9 - (.2 * .8 + .3 * .2) / .5
        self.assertFalse(math.isclose(sum(contributions), conditional_difference))
        reverse = mod.shift_oracle(ref, right, ref, utility, support)
        self.assertAlmostEqual(sum(reverse), -.024)

    def test_stage_metadata_and_latest_1000_are_strict(self):
        self.assertEqual(mod.validate_metadata(evaluation()), ('screen', 256))
        self.assertEqual(mod.validate_metadata(evaluation('full4096')), ('full4096', 4096))
        bad = evaluation('full4096')
        bad['rows'] = 256
        with self.assertRaisesRegex(ValueError, 'denominator'):
            mod.validate_metadata(bad)
        bad = evaluation()
        bad['checkpoints']['RL-residual']['step'] = 0
        with self.assertRaisesRegex(ValueError, 'latest1000'):
            mod.validate_metadata(bad)
        bad = evaluation()
        bad['original_C_parity']['changes_with_margin_above_tolerance'] = 1
        with self.assertRaisesRegex(ValueError, 'parity'):
            mod.validate_metadata(bad)

    def test_import_and_source_never_execute_numerical_or_ML_work(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        imports = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
        self.assertNotIn('numpy', imports)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse({'torch', 'catboost', 'shap', 'subprocess', 'requests'} & {a.name for a in node.names})
        self.assertEqual((mod.REPLICATES, mod.SEED), (2000, 20261024))
        self.assertEqual(len(mod.PAIRS), 3)
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn('differences.shape == (len(users), 9)', source)
        self.assertIn("protocol_pin['sha256'] == EVALUATION_PROTOCOL_SHA256", source)
        self.assertIn('identity_absolute_error', source)
        self.assertIn('math.isclose(total, expected, abs_tol=1e-12, rel_tol=0)', source)
        self.assertIn("evaluation.get('candidate_source_columns')", source)
        self.assertIn('Recomputed screen diagnostic differs', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
