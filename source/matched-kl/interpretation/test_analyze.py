"""Source-only standard-library tests. Never call run() or load real arrays."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('analyze.py')
spec = importlib.util.spec_from_file_location('multiseed_analyzer_under_test', SOURCE)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def metadata():
    pin = {'path': 'synthetic-only', 'sha256': 'a' * 64}
    checkpoints = {utility + '-' + str(seed): {**pin, 'training': dict(pin), 'seed': seed,
                    'utility': utility, 'step': 1000} for seed in analyzer.SEEDS for utility in ('completion', 'residual')}
    checkpoints['C'] = {**pin, 'step': None, 'lineage': 'original C'}
    return {'contract': analyzer.CONTRACT, 'state': 'completed', 'stage': 'screen', 'rows': 256,
            'metric_denominator': 256, 'candidate_width': 1024, 'temperature': 1.,
            'tie_break': 'score descending, raw item ID ascending', 'no_target_injection': True,
            'no_seen_filter': True, 'unknown_and_offpool_are_zero': True, 'final_test_reads': 0,
            'original_C_parity': {'max_abs_error': 0., 'top10_changed_rows': 0,
                'score_tolerance_passed': True, 'changes_with_margin_above_tolerance': 0},
            'checkpoints': checkpoints, 'arms': list(checkpoints),
            'metrics': {arm: {metric: 0. for metric in analyzer.METRICS} for arm in ('C', *checkpoints)}}


class AnalyzeTests(unittest.TestCase):
    def test_source_compiles_and_import_is_lazy(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        compile(tree, SOURCE.as_posix(), 'exec')
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('numpy' in ast.unparse(node) or 'torch' in ast.unparse(node) for node in top_imports))

    def test_rank_entire1024_and_raw_id_ties(self):
        pool = list(range(3, 1027))
        scores = [0.] * 1024
        raw = list(reversed(range(1000, 2024)))
        order = analyzer.rank_oracle(pool, scores, raw)
        self.assertEqual(order, list(reversed(pool)))
        self.assertEqual(analyzer.target_rank(order, 3, True), 1024)
        scores[0] = 2.
        self.assertEqual(analyzer.rank_oracle(pool, scores, raw)[0], 3)

    def test_unknown_offpool_and_cutoff_are_zero(self):
        for rank in (None, 11, 1024):
            self.assertEqual(analyzer.row_metrics(rank), (0., 0., 0.))
        self.assertIsNone(analyzer.target_rank([3, 4], 3, False))
        self.assertIsNone(analyzer.target_rank([3, 4], 5, True))
        self.assertEqual(analyzer.row_metrics(1), (1., 1., 1.))
        self.assertAlmostEqual(analyzer.row_metrics(10)[0], .1)

    def test_rank_rejects_duplicate_and_nonfinite(self):
        for pool, scores, raw in (([3, 3], [0., 1.], [1, 2]), ([3, 4], [0., 1.], [1, 1]),
                                  ([3, 4], [math.nan, 1.], [1, 2])):
            with self.assertRaises(ValueError):
                analyzer.rank_oracle(pool, scores, raw)

    def test_fixed_schema_dynamic_arms_and_plan(self):
        doc = metadata()
        stage, rows, arms, pairs = analyzer.validate_metadata(doc)
        self.assertEqual((stage, rows, len(arms)), ('screen', 256, 7))
        plan = analyzer.comparison_plan(pairs)
        self.assertEqual(len(plan), 12)
        primary = next(item for item in plan if item['name'] == 'three-seed-average/residual-minus-completion')
        self.assertEqual(primary['terms'], [('completion-' + str(seed), 'residual-' + str(seed)) for seed in analyzer.SEEDS])
        self.assertEqual(primary['equal_seed_weight'], 1 / 3)
        doc.update(stage='full4096', rows=4096, metric_denominator=4096)
        self.assertEqual(analyzer.validate_metadata(doc)[1], 4096)

    def test_exact_C_parity_is_required(self):
        for field, bad in (('max_abs_error', 1e-15), ('top10_changed_rows', 1),
                           ('score_tolerance_passed', False), ('changes_with_margin_above_tolerance', 1)):
            doc = metadata()
            doc['original_C_parity'][field] = bad
            with self.assertRaises(ValueError, msg=field):
                analyzer.validate_metadata(doc)

    def test_dynamic_safe_arm_names_use_explicit_seed_utility(self):
        doc = metadata()
        rename = {arm: ('custom_' + str(index) if arm != 'C' else arm)
                  for index, arm in enumerate(doc['checkpoints'])}
        doc['checkpoints'] = {rename[arm]: value for arm, value in doc['checkpoints'].items()}
        doc['metrics'] = {rename[arm]: value for arm, value in doc['metrics'].items()}
        doc['arms'] = [rename[arm] for arm in doc['arms']]
        _, _, arms, pairs = analyzer.validate_metadata(doc)
        self.assertEqual(set(arms), set(rename.values()))
        self.assertEqual(pairs[20261024]['completion'], rename['completion-20261024'])
        self.assertEqual(analyzer.comparison_plan(pairs)[0]['right'], rename['completion-20261024'])
        doc['arms'][-1] = doc['arms'][0]
        with self.assertRaises(ValueError):
            analyzer.validate_metadata(doc)

    def test_rejects_windows_casefold_arm_collisions(self):
        for renamed in (('c', 'residual-20261024'), ('Foo', 'foo')):
            doc = metadata()
            rename = {'completion-20261024': renamed[0], 'residual-20261024': renamed[1]}
            doc['checkpoints'] = {rename.get(arm, arm): value for arm, value in doc['checkpoints'].items()}
            doc['metrics'] = {rename.get(arm, arm): value for arm, value in doc['metrics'].items()}
            doc['arms'] = [rename.get(arm, arm) for arm in doc['arms']]
            with self.assertRaisesRegex(ValueError, 'case-insensitive'):
                analyzer.validate_metadata(doc)

    def test_rejects_wrong_checkpoint_binding(self):
        for field, bad in (('seed', 20261027), ('step', 500), ('utility', 'other'), ('training', {})):
            doc = metadata()
            doc['checkpoints']['completion-20261024'][field] = bad
            with self.assertRaises(ValueError, msg=field):
                analyzer.validate_metadata(doc)

    def test_queryweighted_cluster_ratio_not_equal_user_means(self):
        # Two users, query counts 3 and 1. One copy of each retains mean 1/4,
        # while equal-user means would be 1/2. Repeated user carries all queries.
        users, delta = ['A', 'A', 'A', 'B'], [0., 0., 0., 1.]
        self.assertEqual(analyzer.cluster_oracle(users, delta, ['A', 'B']), .25)
        self.assertEqual(analyzer.cluster_oracle(users, delta, ['A', 'A']), 0.)
        self.assertEqual(analyzer.cluster_oracle(users, delta, ['B', 'B']), 1.)

    def test_seed_average_before_common_cluster_draw(self):
        users = [1, 1, 2]
        seeds = [[1., 0., 0.], [0., 0., 1.], [-1., 0., 1.]]
        average = [sum(values) / 3 for values in zip(*seeds)]
        for draw in ([1, 1], [1, 2], [2, 2]):
            self.assertAlmostEqual(analyzer.cluster_oracle(users, average, draw),
                                   sum(analyzer.cluster_oracle(users, values, draw) for values in seeds) / 3)

    def test_descriptive_seed_spread_sample_std(self):
        result = analyzer.descriptive([1., 2., 3.])
        self.assertEqual((result['mean'], result['min'], result['max'], result['sample_std']), (2., 1., 3., 1.))

    def test_matching_absent_and_separate_flags(self):
        doc = metadata()
        pairs = analyzer.validate_metadata(doc)[3]
        absent = analyzer.matching_metadata({}, pairs)
        self.assertIsNone(absent['all_three_train_KL_matches'])
        self.assertEqual(list(absent['train_KL_match_by_seed'].values()), [None, None, None])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.json'
            cache = Path(directory) / 'cache.json'
            reference = Path(directory) / 'reference.json'
            analyzer.write(cache, {'rows': 512, 'synthetic': True})
            analyzer.write(reference, {'synthetic': True, 'state': 'completed', 'mode': 'evaluate_train512',
                'rows': 512, 'cache_sha256': analyzer.sha(cache), 'results': {'old-completion': {
                    'checkpoint': {'sha256': doc['checkpoints']['completion-20261024']['sha256']},
                    'metrics': {'completion': {'exact_KL': .1}}}}})
            receipt = {'selected_checkpoints': {arm: value['sha256'] for arm, value in doc['checkpoints'].items() if arm != 'C'},
                'target_train_KL': .1, 'tolerance': [.8, 1.2],
                'train_KL_by_arm': {arm: (.2 if arm == 'residual-20261025' else .1) for arm in doc['checkpoints'] if arm != 'C'},
                'train_KL_match_by_seed': {'20261024': True, '20261025': False, '20261026': True},
                'cache': {'path': cache.as_posix(), 'sha256': analyzer.sha(cache)},
                'reference': {'path': reference.as_posix(), 'sha256': analyzer.sha(reference)}}
            analyzer.write(path, receipt)
            request = {'matching_receipt': {'path': path.as_posix(), 'sha256': analyzer.sha(path)}}
            with patch.object(analyzer, 'TRAIN_CACHE_SHA', analyzer.sha(cache)):
                result = analyzer.matching_metadata(request, pairs, doc['checkpoints'])
            self.assertFalse(result['all_three_train_KL_matches'])
            self.assertFalse(result['train_KL_match_by_seed']['20261025'])
            self.assertEqual(result['train_KL_residual_over_completion_by_seed']['20261025'], 2.)
            with self.assertRaisesRegex(ValueError, 'cache pin differs'):
                analyzer.matching_metadata(request, pairs, doc['checkpoints'])
            with self.assertRaises(ValueError):
                analyzer.pinned({'path': path.as_posix(), 'sha256': '0' * 64})

            old_completion_disagreement = dict(receipt['train_KL_by_arm'], **{'completion-20261024': .11})
            mutations = [('selected_checkpoints', {'wrong-arm': 'a' * 64}), ('target_train_KL', 0.),
                         ('target_train_KL', 1e-10), ('train_KL_by_arm', old_completion_disagreement),
                         ('target_train_KL', None), ('target_train_KL', .11), ('tolerance', [.7, 1.3]),
                         ('train_KL_by_arm', {'wrong-arm': .1}),
                         ('train_KL_match_by_seed', {str(seed): True for seed in analyzer.SEEDS})]
            for index, (key, value) in enumerate(mutations):
                changed = copy.deepcopy(receipt)
                changed[key] = value
                altered = Path(directory) / ('changed-' + str(index) + '.json')
                analyzer.write(altered, changed)
                with patch.object(analyzer, 'TRAIN_CACHE_SHA', analyzer.sha(cache)):
                    with self.assertRaises(ValueError, msg=key):
                        analyzer.matching_metadata({'matching_receipt': {'path': altered.as_posix(),
                            'sha256': analyzer.sha(altered)}}, pairs, doc['checkpoints'])
            outside_target = copy.deepcopy(receipt)
            outside_target['train_KL_by_arm'].update({'completion-20261026': .2, 'residual-20261026': .2})
            outside_path = Path(directory) / 'outside-target.json'
            analyzer.write(outside_path, outside_target)
            with patch.object(analyzer, 'TRAIN_CACHE_SHA', analyzer.sha(cache)):
                outside = analyzer.matching_metadata({'matching_receipt': {'path': outside_path.as_posix(),
                    'sha256': analyzer.sha(outside_path)}}, pairs, doc['checkpoints'])
            self.assertTrue(outside['train_KL_match_by_seed']['20261026'])
            self.assertEqual(analyzer.BOOTSTRAP_SEED, 20261024)


if __name__ == '__main__':
    unittest.main()
