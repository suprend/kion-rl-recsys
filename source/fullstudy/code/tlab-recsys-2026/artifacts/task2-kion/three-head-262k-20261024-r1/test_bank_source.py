"""Standard-library-only source/selection tests; never imports the ML runtime."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_bank_source_test', HERE / 'bank.py')
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)


class BankSourceTests(unittest.TestCase):
    def test_proportional_exact_remainder_and_ties(self):
        self.assertEqual(bank.proportional_quotas({12: 1, 10: 1, 11: 1}, 2), {12: 0, 10: 1, 11: 1})
        self.assertEqual(bank.proportional_quotas({10: 7, 11: 3}, 6), {10: 4, 11: 2})

    def test_selection_is_known_unique_date_sorted_and_deterministic(self):
        targets = [3, 4, 0, 5, 6, 1, 7, 8, 10, 9]
        dates = [20, 10, 10, 20, 10, 20, 10, 20, 10, 20]
        selected, counts, quotas = bank.select_rows(targets, dates, 10, maximum=6, seed=19)
        self.assertEqual(counts, {10: 3, 20: 4})
        self.assertEqual(sum(quotas.values()), 6)
        self.assertEqual(selected, bank.select_rows(targets, dates, 10, maximum=6, seed=19)[0])
        self.assertEqual(len(set(selected)), 6)
        self.assertEqual([(dates[i], i) for i in selected], sorted((dates[i], i) for i in selected))
        self.assertTrue(all(3 <= targets[i] < 10 for i in selected))

    def test_no_coverage_input_and_every_known_row_can_be_selected(self):
        targets, dates = [3, 4, 5, 6], [10, 11, 10, 11]
        selected, _, _ = bank.select_rows(targets, dates, 7, maximum=4)
        self.assertEqual(set(selected), {0, 1, 2, 3})
        self.assertEqual(bank.select_rows.__code__.co_varnames[:6], ('targets', 'dates', 'vocab', 'maximum', 'seed', 'reuse_rows'))

    def test_nested_selection_retains_every_old_row_and_exact_date_quotas(self):
        targets, dates = [3] * 200, [20] * 120 + [10] * 80
        old, _, _ = bank.select_rows(targets, dates, 4, maximum=24, seed=9)
        rows, counts, quotas = bank.select_rows(targets, dates, 4, maximum=96, seed=9, reuse_rows=old)
        self.assertTrue(set(old) <= set(rows))
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(len(set(rows) - set(old)), 72)
        self.assertEqual(quotas, {20: 58, 10: 38})
        self.assertEqual({d: sum(dates[i] == d for i in rows) for d in counts}, quotas)
        self.assertEqual(rows, bank.select_rows(targets, dates, 4, maximum=96, seed=9, reuse_rows=reversed(old))[0])
        self.assertEqual([(dates[i], i) for i in rows], sorted((dates[i], i) for i in rows))

    def test_nested_reuse_quota_mismatch_and_invalid_ids_fail(self):
        with self.assertRaisesRegex(ValueError, 'date quota'):
            bank.select_rows([3] * 10, [10] + [20] * 9, 4, maximum=2, reuse_rows=[0])
        for rows in ([1, 1], [-1], [10], [0]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                bank.select_rows([0] + [3] * 9, [10] * 10, 4, maximum=4, reuse_rows=rows)

    def test_reuse_manifest_rejects_configuration_and_lineage_changes(self):
        request = {'teacher': {'sha256': bank.TEACHER_SHA, 'config_sha256': bank.CONFIG_SHA},
                   'input_hashes': {(bank.DATA / 'metadata.json').as_posix(): 'fixture-metadata'}}
        manifest = dict(contract='kion-three-head-pilot-bank-v1', state='completed', rows=32768,
                        train_rows=32768, split='train', seed=bank.SEED, teardown_completed=True,
                        model_training=False, optimizer_created=False, final_test=False,
                        teacher=copy.deepcopy(request['teacher']), feature_schema_sha256=bank.SCHEMA_SHA,
                        source_columns=bank.SOURCE_COLUMNS, candidate_width=32, feature_width=1068,
                        context_width=256, natural_covered_only=False, same_day_positives_excluded=True,
                        context_signature=bank.context_signature(request),
                        arrays={name: {} for name in [*bank.LAYOUT, 'item_table']})
        bank.validate_reuse_manifest(manifest, request)
        for field, value in [('seed', 1), ('context_signature', 'changed'),
                             ('feature_schema_sha256', 'changed'), ('rows', 32767)]:
            bad = copy.deepcopy(manifest)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bank.validate_reuse_manifest(bad, request)
        bad = copy.deepcopy(manifest)
        bad['teacher']['config_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'config lineage'):
            bank.validate_reuse_manifest(bad, request)
        request_bad = copy.deepcopy(request)
        request_bad['teacher']['config_sha256'] = 'changed'
        with self.assertRaises(ValueError):
            bank.validate_reuse_manifest(manifest, request_bad)

    def test_insufficient_rows_fails(self):
        with self.assertRaises(ValueError):
            bank.select_rows([0, 3], [10, 10], 4, maximum=2)

    def test_fixed_layout(self):
        self.assertEqual(bank.CONTRACT, 'kion-three-head-262k-bank-v1')
        self.assertEqual(bank.LAYOUT['raw_features'], ('float32', (262144, 32, 1068)))
        self.assertEqual(bank.LAYOUT['source_values'], ('float32', (262144, 32, 2)))
        self.assertEqual(bank.LAYOUT['encoded'], ('float32', (262144, 256)))
        self.assertEqual(bank.MAP_WINDOW_ROWS, 1024)
        self.assertEqual(bank.SOURCE_COLUMNS, ['current_detached_gpt_logit', 'log1p_strict7d_count'])

    def test_only_stdlib_imports_and_no_top_level_calls(self):
        tree = ast.parse((HERE / 'bank.py').read_text(encoding='utf-8'))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module.split('.')[0])
        self.assertTrue(set(imported) <= sys.stdlib_module_names)
        self.assertFalse({'numpy', 'torch', 'scipy', 'pandas'} & set(imported))
        self.assertFalse(any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) for node in tree.body))
        self.assertFalse(any(isinstance(node, ast.If) for node in tree.body))

    def test_close_attempts_remaining_owners_after_failure(self):
        calls = []
        class Broken:
            def close(self):
                calls.append('broken')
                raise RuntimeError('fixture')
        class Good:
            def close(self):
                calls.append('good')
        ctx = {'candidates': Broken(), 'provider': Good(), 'events': {}}
        result = bank.close_context(ctx)
        self.assertEqual(calls, ['broken', 'good'])
        self.assertFalse(result['teardown_completed'])
        self.assertEqual(result['errors'][0]['owner'], 'candidates')
        self.assertIs(bank.close_context(ctx), result)

    def test_validation_graph_build_is_explicit_and_train_readonly(self):
        class FakeCandidates:
            validation_origin = 738011
            def __init__(self, root):
                self.root, self.max_disk_bytes = root, 5 << 30
                self.builds = []
            def _paths(self, origin):
                return self.root / f'{origin}.json', {'neighbors': self.root / f'{origin}.npy'}
            def _disk_check(self):
                pass
            def _build(self, origin, descriptor, arrays):
                self.builds.append((origin, descriptor, arrays, self.max_disk_bytes))
        with tempfile.TemporaryDirectory() as temporary:
            previous = bank.HERE
            bank.HERE = Path(temporary).resolve()
            try:
                train_root = bank.HERE / 'train-cache'
                candidates = FakeCandidates(train_root)
                with self.assertRaises(ValueError):
                    bank.configure_validation_graph(candidates, {'stage': 'bank', 'allow_validation_graph_build': True}, lambda **k: None)
                evidence = bank.configure_validation_graph(candidates,
                    {'stage': 'evaluate', 'allow_validation_graph_build': True, 'graph_cache': str(train_root)}, lambda **k: None)
                self.assertTrue(evidence['train_cache_read_only'])
                self.assertEqual(candidates._paths(737900)[0].parent, train_root)
                descriptor, arrays = candidates._paths(738011)
                self.assertEqual(descriptor.parent, bank.HERE / 'cache/validation-graph')
                with self.assertRaises(ValueError):
                    candidates._build(737900, *candidates._paths(737900))
                candidates._build(738011, descriptor, arrays)
                self.assertEqual(candidates.builds[0][3], 64 << 20)
                self.assertEqual(candidates.root, train_root)
            finally:
                bank.HERE = previous

    def test_existing_validation_graph_routes_readonly_and_never_builds(self):
        class FakeCandidates:
            def __init__(self, root):
                self.root = root
            def _paths(self, origin):
                return self.root / f'{origin}.json', {'neighbors': self.root / f'{origin}.npy'}
        with tempfile.TemporaryDirectory() as temporary:
            previous = bank.REUSE_DIR
            bank.REUSE_DIR = Path(temporary) / 'cache/train'
            root = bank.REUSE_DIR.parent / 'validation-graph'
            root.mkdir(parents=True)
            try:
                candidates = FakeCandidates(Path(temporary) / 'train-cache')
                evidence = bank.configure_validation_graph(candidates,
                    {'stage': 'evaluate', 'validation_graph_cache': str(root),
                     'allow_validation_graph_build': False}, lambda **k: None)
                self.assertTrue(evidence['validation_cache_read_only'])
                self.assertEqual(candidates._paths(738011)[0].parent, root.resolve())
                self.assertEqual(candidates._paths(737900)[0].parent, candidates.root)
                with self.assertRaises(RuntimeError):
                    candidates._build(738011, *candidates._paths(738011))
            finally:
                bank.REUSE_DIR = previous


if __name__ == '__main__':
    unittest.main()
