"""Synthetic-only stdlib tests: no campaign data, arrays, models or numerical jobs."""
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('verify.py')
spec = importlib.util.spec_from_file_location('independent_csv_verifier_test', SOURCE)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def pin(path):
    return {'path': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class Fixture:
    """Use independent fixture aggregation and one covered rank pattern per arm."""
    def __init__(self, root, stage='screen'):
        self.root = root
        self.stage = stage
        self.n = 256 if stage == 'screen' else 4096
        self.folder = root / 'interpretation' / ('screen-r1' if stage == 'screen' else 'full-r1')
        self.folder.mkdir(parents=True)
        self.analysis_path = self.folder / 'analysis.json'
        self.csv_path = self.folder / 'per-query.csv'
        self.match_path = root / 'protocols/matching.json'
        self.evaluation_path = root / 'policy/fake-evaluation.json'
        self.cache_path = root / 'policy/fake-cache.json'
        self.reference_path = root / 'policy/fake-reference.json'
        self.final_path = root / 'policy/fake-six.json'
        self.arms = ['C'] + [u + '-' + str(s) for s in verifier.SEEDS for u in ('completion', 'residual')]
        self.cp = {'C': {'path': (root / 'unopened/C.pt').as_posix(), 'sha256': 'a' * 64}}
        for index, arm in enumerate(self.arms[1:], 1):
            utility, seed = arm.split('-')
            self.cp[arm] = {'path': (root / ('unopened/' + arm + '.pt')).as_posix(),
                            'sha256': hashlib.sha256(arm.encode()).hexdigest(), 'step': 1000,
                            'seed': int(seed), 'utility': utility,
                            'training': {'path': (root / ('unopened/' + arm + '.json')).as_posix(), 'sha256': 'b' * 64}}
        self.header = ['row_index', 'user_id', 'known', 'covered'] + [
            arm + '_' + key for arm in self.arms for key in ('target_rank', 'map10', 'ndcg10', 'recall10')]
        self.rows, self.values = [], {arm: [] for arm in self.arms}
        for row in range(self.n):
            known, covered = row % 8 != 0, row % 8 not in (0, 1)
            cells = [row * 2, row // 3, known, covered]
            for index, arm in enumerate(self.arms):
                rank = (1, 2, 10, 11, 1024, 7, 3)[(row + index) % 7] if covered else None
                result = ((1 / rank, 1 / math.log2(1 + rank), 1.0)
                          if rank is not None and rank < 11 else (0.0, 0.0, 0.0))
                self.values[arm].append(result)
                cells += [rank, *result]
            self.rows.append(cells)
        self.write_csv()
        self.evaluation = {'contract': 'kion-matched-kl-multiseed-evaluation-v1', 'state': 'completed',
            'stage': stage, 'rows': self.n, 'metric_denominator': self.n, 'candidate_width': 1024,
            'temperature': 1.0, 'tie_break': 'score descending, raw item ID ascending', 'no_target_injection': True,
            'no_seen_filter': True, 'unknown_and_offpool_are_zero': True, 'final_test_reads': 0,
            'original_C_parity': {'max_abs_error': 0.0, 'top10_changed_rows': 0,
                'score_tolerance_passed': True, 'changes_with_margin_above_tolerance': 0},
            'checkpoints': self.cp, 'arms': self.arms,
            'metrics': {arm: {'rows': self.n, 'known_targets': self.n * 7 // 8,
                       'unknown_target_misses': self.n // 8, 'offpool_known_misses': self.n // 8,
                       'candidate_coverage_count': self.n * 3 // 4,
                       **{metric: statistics.mean(row[j] for row in self.values[arm])
                       for j, metric in enumerate(verifier.METRICS)}} for arm in self.arms}}
        save(self.evaluation_path, self.evaluation)
        self.cache = {'state': 'completed', 'cohort': 'train', 'rows': 512, 'candidate_width': 1024,
                      'temperature': 1, 'no_target_injection': True, 'no_seen_filter': True,
                      'C_checkpoint_sha256': self.cp['C']['sha256'],
                      'arrays': {'never-open': {'path': 'absent.npy', 'sha256': 'd' * 64}}}
        save(self.cache_path, self.cache)
        self.kls = {arm: value for arm, value in zip(self.arms[1:], (.1, .08, .2, .3, .4, .4))}
        self.ratios = {str(s): self.kls['residual-' + str(s)] / self.kls['completion-' + str(s)] for s in verifier.SEEDS}
        self.flags = {seed: .8 <= ratio <= 1.2 for seed, ratio in self.ratios.items()}
        self.train_base = {'state': 'completed', 'mode': 'evaluate_train512', 'rows': 512,
            'cache_sha256': pin(self.cache_path)['sha256'], 'final_test_reads': 0, 'learning_updates': 0,
            'optimizer_created': False, 'update0_actor_reference_exact': True, 'update0_cached_scores_exact': True,
            'update0_top10_exact': True, 'update0_max_abs_error': 0.0}
        self.final = {**self.train_base, 'results': {'original-C': {}}}
        for arm in self.arms[1:]:
            self.final['results'][arm] = {'checkpoint': {**self.cp[arm], 'name': arm},
                'metrics': {u: {'rows': 512, 'exact_KL': self.kls[arm]} for u in ('completion', 'residual')}}
        save(self.final_path, self.final)
        reference = {**self.train_base, 'results': {'original-C': {}, 'reference-completion':
                     copy.deepcopy(self.final['results']['completion-20261024'])}}
        save(self.reference_path, reference)
        self.matching = {'contract': 'completed-train-KL-match-binding-v1', 'no_dev_selection': True,
            'selected_checkpoints': {arm: self.cp[arm]['sha256'] for arm in self.arms[1:]},
            'target_train_KL': .1, 'tolerance': [.8, 1.2], 'train_KL_by_arm': self.kls,
            'ratio_by_seed': self.ratios, 'train_KL_match_by_seed': self.flags,
            'cache': pin(self.cache_path), 'reference': pin(self.reference_path), 'final_train_diagnostics': pin(self.final_path)}
        save(self.match_path, self.matching)
        self.analysis = {'contract': 'kion-matched-kl-multiseed-analysis-v1', 'state': 'completed', 'stage': stage,
            'rows': self.n, 'metric_denominator': self.n, 'candidate_width': 1024, 'seeds': list(verifier.SEEDS),
            'arms': self.arms, 'checkpoints': self.cp, 'evaluation': pin(self.evaluation_path),
            'original_C_parity': self.evaluation['original_C_parity'],
            'summaries': {arm: {'metrics': {metric: self.evaluation['metrics'][arm][metric]
                                          for metric in verifier.METRICS}} for arm in self.arms},
            'unique_users': len({row[1] for row in self.rows}), 'unknown_target_misses': self.n // 8,
            'offpool_known_misses': self.n // 8, 'candidate_coverage_count': self.n * 3 // 4,
            'primary_comparison': 'three-seed-average/residual-minus-completion', 'primary_metric': 'map10',
            'secondary_metrics': ['ndcg10', 'recall10'],
            'bootstrap': {'replicates': 2000, 'seed': 20261024, 'batch_size': 32, 'columns': 36,
                'common_draw_schedule': True, 'multiple_comparison_adjustment': False,
                'CI': 'unadjusted percentile 2.5/97.5, NumPy linear quantiles',
                'unit': 'unique user clusters sampled with replacement; all queries follow their user',
                'estimand': 'query-weighted ratio of resampled cluster sums to resampled query counts',
                'seed_average': 'equal weight 1/3 on the three fitted heads, before user resampling'},
            'training_KL_matching': {'receipt': pin(self.match_path), 'cache': pin(self.cache_path),
                'reference': pin(self.reference_path), 'receipt_metadata': self.matching,
                'train_KL_match_by_seed': self.flags, 'all_three_train_KL_matches': all(self.flags.values()),
                'train_KL_residual_over_completion_by_seed': self.ratios,
                'train_KL_distance_from_calibration_target_by_arm': {arm: kl - .1 for arm, kl in self.kls.items()}}}
        comparisons = {}
        for seed in verifier.SEEDS:
            c, r = 'completion-' + str(seed), 'residual-' + str(seed)
            for kind, left, right in zip(verifier.KINDS, ('C', 'C', c), (c, r, r)):
                name = str(seed) + '/' + kind
                comparisons[name] = {'name': name, 'kind': kind, 'seed': seed, 'left': left, 'right': right,
                                     'terms': [[left, right]]}
        for kind in verifier.KINDS:
            name = 'three-seed-average/' + kind
            comparisons[name] = {'name': name, 'kind': kind, 'seed': None, 'equal_seed_weight': 1 / 3,
                                'terms': [item['terms'][0] for item in comparisons.values() if item['kind'] == kind]}
        for item in comparisons.values():
            item['metrics'] = {}
            for j, metric in enumerate(verifier.METRICS):
                delta = [sum(self.values[right][row][j] - self.values[left][row][j] for left, right in item['terms']) /
                         len(item['terms']) for row in range(self.n)]
                item['metrics'][metric] = {'difference': statistics.mean(delta), 'gains': sum(x > 0 for x in delta),
                    'losses': sum(x < 0 for x in delta), 'ties': sum(x == 0 for x in delta),
                    'user_cluster_95ci': [.3, .4]}
        self.analysis['comparisons'] = comparisons
        self.analysis['seed_spread'] = {kind: {} for kind in verifier.KINDS}
        self.analysis['prospective_two_seed_mean'] = {kind: {} for kind in verifier.KINDS}
        for kind in verifier.KINDS:
            for metric in verifier.METRICS:
                vals = [comparisons[str(s) + '/' + kind]['metrics'][metric]['difference'] for s in verifier.SEEDS]
                self.analysis['seed_spread'][kind][metric] = {'mean': statistics.mean(vals), 'min': min(vals),
                    'max': max(vals), 'sample_std': statistics.stdev(vals), 'seed_count': 3}
                self.analysis['prospective_two_seed_mean'][kind][metric] = statistics.mean(vals[1:])
        self.refresh()

    def write_csv(self):
        with self.csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(self.header)
            writer.writerows(self.rows)

    def refresh(self):
        save(self.analysis_path, self.analysis)
        self.request = {'analysis': pin(self.analysis_path), 'per_query': pin(self.csv_path),
                        'matching_receipt': pin(self.match_path),
                        'output': (self.root / 'review' / ('numeric-' + self.stage + '.json')).as_posix()}

    def run(self, check=None):
        with patch.object(verifier, 'ROOT', self.root), patch.object(verifier, 'TRAIN_CACHE_SHA', pin(self.cache_path)['sha256']):
            return verifier.run(self.request, check)


class VerifyTests(unittest.TestCase):
    def test_only_stdlib_imports_and_no_analyzer_import(self):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports += [(node.module or '').split('.')[0]]
        self.assertTrue(set(imports) <= sys.stdlib_module_names | {'__future__'})
        self.assertNotIn('analyze', imports)
        self.assertNotIn('numpy', imports)
        self.assertNotIn('torch', imports)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                             node.func.id in ('eval', 'exec', '__import__') for node in ast.walk(tree)))

    def test_end_to_end_fake_both_stages_no_array_or_checkpoint_needed(self):
        for stage in ('screen', 'full4096'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory), stage)
                calls = []
                report = fixture.run(lambda **fields: calls.append(fields))
                self.assertTrue(report['verified'])
                self.assertEqual(report['state'], 'completed')
                self.assertEqual(report['rows'], fixture.n)
                self.assertEqual(len(report['comparisons']), 12)
                self.assertFalse(report['training_KL_matching']['all_three_train_KL_matches'])
                self.assertFalse(report['bootstrap_intervals_independently_resampled'])
                self.assertEqual(len(report['inputs']), 7)
                self.assertEqual(calls[-1]['stage'], 'independent-csv-verification-complete')
                self.assertEqual(json.loads(Path(fixture.request['output']).read_text(encoding='utf-8')), report)
                with self.assertRaisesRegex(ValueError, 'Fresh exact'):
                    fixture.run()

    def test_rejects_bad_csv_evidence(self):
        cases = ('duplicate', 'schema', 'unknown_rank', 'unknown_metric', 'covered_unknown',
                 'rank_missing', 'rank_too_large', 'rank_fraction', 'nan', 'metric', 'short', 'wide')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if case == 'duplicate': fixture.rows[1][0] = fixture.rows[0][0]
                elif case == 'schema': fixture.header[-1] = fixture.header[-2]
                elif case == 'unknown_rank': fixture.rows[0][4] = 2
                elif case == 'unknown_metric': fixture.rows[0][5] = 1e-15
                elif case == 'covered_unknown': fixture.rows[0][3] = True
                elif case == 'rank_missing': fixture.rows[2][4] = None
                elif case == 'rank_too_large': fixture.rows[2][4] = 1025
                elif case == 'rank_fraction': fixture.rows[2][4] = '2.0'
                elif case == 'nan': fixture.rows[2][5] = 'nan'
                elif case == 'metric': fixture.rows[2][5] = .999
                elif case == 'short': fixture.rows.pop()
                elif case == 'wide': fixture.rows[0].append('extra')
                fixture.write_csv()
                fixture.refresh()
                with self.assertRaises(ValueError): fixture.run()
                self.assertFalse(Path(fixture.request['output']).exists())

    def test_rejects_bad_aggregate_and_protocol_claims(self):
        mutations = [
            lambda a: a.update(metric_denominator=255),
            lambda a: a.update(unique_users=1),
            lambda a: a.update(arms=list(reversed(a['arms']))),
            lambda a: a['bootstrap'].update(common_draw_schedule=False),
            lambda a: a['summaries']['C']['metrics'].update(map10=.99),
            lambda a: a['comparisons']['20261024/residual-minus-C']['metrics']['recall10'].update(difference=.9),
            lambda a: a['comparisons']['three-seed-average/residual-minus-C']['metrics']['map10'].update(ties=0),
            lambda a: a['comparisons']['20261024/residual-minus-C'].update(terms=[['C', 'completion-20261024']]),
            lambda a: a['comparisons']['20261024/residual-minus-C']['metrics']['map10'].update(user_cluster_95ci=[.4, .3]),
            lambda a: a['seed_spread']['completion-minus-C']['map10'].update(sample_std=.8),
            lambda a: a['prospective_two_seed_mean']['completion-minus-C'].update(map10=.8),
            lambda a: a['training_KL_matching'].update(all_three_train_KL_matches=True)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                mutate(fixture.analysis)
                fixture.refresh()
                with self.assertRaises(ValueError): fixture.run()
                self.assertFalse(Path(fixture.request['output']).exists())

    def test_final_train_diagnostics_and_checkpoint_bindings_are_checked(self):
        for case in ('scalar', 'checkpoint', 'cache', 'reference', 'flag'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if case in ('scalar', 'checkpoint'):
                    arm = fixture.final['results']['residual-20261026']
                    if case == 'scalar': arm['metrics']['completion']['exact_KL'] = .7
                    else: arm['checkpoint']['sha256'] = 'f' * 64
                    save(fixture.final_path, fixture.final)
                    fixture.matching['final_train_diagnostics'] = pin(fixture.final_path)
                elif case == 'cache':
                    fixture.cache['C_checkpoint_sha256'] = 'f' * 64
                    save(fixture.cache_path, fixture.cache)
                    fixture.matching['cache'] = pin(fixture.cache_path)
                elif case == 'reference':
                    reference = json.loads(fixture.reference_path.read_text(encoding='utf-8'))
                    reference['results']['reference-completion']['metrics']['completion']['exact_KL'] = .11
                    save(fixture.reference_path, reference)
                    fixture.matching['reference'] = pin(fixture.reference_path)
                else: fixture.matching['train_KL_match_by_seed']['20261025'] = True
                save(fixture.match_path, fixture.matching)
                fixture.analysis['training_KL_matching']['receipt'] = pin(fixture.match_path)
                fixture.refresh()
                with self.assertRaises(ValueError): fixture.run()

    def test_nonfinite_json_and_duplicate_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for raw in ('{"x":NaN}', '{"x":1,"x":2}'):
                path.write_text(raw, encoding='utf-8')
                with self.assertRaises(ValueError): verifier.json_pin(pin(path), 'fake')

    def test_tolerance_is_absolute_and_miss_is_exact_zero(self):
        verifier.close(.1, .1 + 5e-13, 'synthetic')
        with self.assertRaises(ValueError): verifier.close(.1, .1 + 2e-12, 'synthetic')
        with self.assertRaises(ValueError): verifier.close(True, 1, 'synthetic')

    def test_digest_and_output_path_fail_closed(self):
        for case in ('analysis_digest', 'csv_digest', 'output'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if case == 'analysis_digest': fixture.request['analysis']['sha256'] = '0' * 64
                elif case == 'csv_digest': fixture.request['per_query']['sha256'] = '0' * 64
                else: fixture.request['output'] = (fixture.root / 'foreign.json').as_posix()
                with self.assertRaises(ValueError): fixture.run()


if __name__ == '__main__':
    unittest.main()
