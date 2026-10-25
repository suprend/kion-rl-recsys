"""Synthetic stdlib-only tests; never open real arrays/checkpoints/providers."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('_test_eval_' + name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev, analysis = load('evaluate'), load('analyze')


class EvaluationTests(unittest.TestCase):
    def test_frozen_selection(self):
        a, b = ev.selection(), ev.selection()
        self.assertEqual(a, b)
        self.assertEqual(a['row_indices'], sorted(set(a['row_indices'])))
        self.assertEqual(len(a['row_indices']), 4096)
        self.assertLess(a['row_indices'][-1], 158760)

    def test_freeze_is_idempotent_but_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'cohort.json'
            first = ev.freeze_selection(path)
            self.assertEqual(first['sha256'], '660100ad03709ed2b1ddb2a62c83c3b9e51de3fc22cabdadfde0735467aa3707')
            self.assertEqual(first, ev.freeze_selection(path))
            self.assertEqual(ev.validate_selection(path, first['sha256']), ev.selection()['row_indices'])
            path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                ev.freeze_selection(path)

    def test_pin_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'cohort.json'
            ev.freeze_selection(path)
            with self.assertRaises(ValueError):
                ev.validate_selection(path, '0'*64)

    def test_unknown_and_offpool_are_misses(self):
        self.assertEqual(ev.row_metrics([3, 4], 3, False), dict(map10=0., recall10=0., ndcg10=0.))
        self.assertEqual(ev.row_metrics([3, 4], 5, True), dict(map10=0., recall10=0., ndcg10=0.))

    def test_single_positive_metrics(self):
        self.assertEqual(ev.row_metrics([5, 3, 4], 3, True)['map10'], .5)
        self.assertEqual(ev.row_metrics(list(range(3, 14)), 13, True)['recall10'], 0.)

    def test_raw_item_id_tie_and_permutation(self):
        self.assertEqual(ev.rank_top([3, 4, 5], [1., 1., 2.], [200, 100, 300]), [5, 4, 3])
        self.assertEqual(ev.rank_top([5, 3, 4], [2., 1., 1.], [300, 200, 100]), [5, 4, 3])

    def test_rrf60_rank_recovery_and_missing_sources(self):
        source = [0.] * 14
        source[3:6] = [1., 0., 1.]
        source[6:9] = [1/2049, 1., 2048/2049]
        source[9:12] = [0., 1., 0.]
        self.assertAlmostEqual(ev.rrf_row(source), 1/61 + 1/2108)
        source[3:6], source[6:9], source[9:12] = [0., 0., 0.], [1., 1., 1.], [1., 1., 1.]
        self.assertEqual(ev.rrf_row(source), 0.)

    def test_rrf_rejects_inconsistent_rank_flags(self):
        source = [0.] * 14
        source[3:6], source[6:9], source[9:12] = [1., 0., 0.], [1., 1., 1.], [0., 1., 1.]
        with self.assertRaises(ValueError):
            ev.rrf_row(source)
        source[6] = 0.
        with self.assertRaises(ValueError):
            ev.rrf_row(source)

    def test_invalid_scores_and_duplicates(self):
        with self.assertRaises(ValueError):
            ev.rank_top([3, 4], [float('nan'), 0], [100, 200])
        with self.assertRaises(ValueError):
            ev.rank_top([3, 3], [1, 0], [100, 100])

    def test_cluster_oracle_weights_rows_not_users(self):
        values = analysis.bootstrap_oracle([10, 10, 20], [1, 1, 0], [[10, 20], [10, 10], [20, 20]])
        self.assertEqual(values, [2/3, 1., 0.])

    def test_cluster_oracle_pairing(self):
        forward = analysis.bootstrap_oracle([1, 1, 2], [1, -1, .5], [[1, 2]])
        reverse = analysis.bootstrap_oracle([1, 1, 2], [-1, 1, -.5], [[1, 2]])
        self.assertEqual(forward[0], -reverse[0])

    def test_imports_are_lazy_and_analysis_gate_precedes_numpy(self):
        for name in ('evaluate', 'analyze'):
            tree = ast.parse((HERE/(name+'.py')).read_text(encoding='utf-8'))
            top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
            imports = [alias.name for node in top_imports if isinstance(node, ast.Import) for alias in node.names]
            self.assertFalse({'torch', 'numpy', 'pandas', 'scipy'} & set(imports))
        function = next(node for node in ast.parse((HERE/'analyze.py').read_text(encoding='utf-8')).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'analyze')
        gate_line = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == 'gate')
        prior_line = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name) and node.func.id == 'prior_gate')
        numpy_line = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Import)
                          and any(alias.name == 'numpy' for alias in node.names))
        self.assertLess(gate_line, numpy_line)
        self.assertLess(prior_line, numpy_line)

    def _record(self):
        receipt = dict(contract=ev.HEAD_CONTRACT, status='completed', epochs_requested=3, epochs_completed=3,
                       rows_per_epoch=262144, batch_size=128,
                       steps_per_head=6144, head_steps={name: 6144 for name in 'ABC'})
        return dict(contract=ev.CONTRACT, state='completed', rows=4096, metric_denominator=4096,
                    candidate_width=1024, arm_names=list(ev.ARMS), no_seen_filter=True, no_target_injection=True,
                    unknown_and_offpool_are_zero=True, source_columns=[12, 13],
                    head_receipts={name: dict(receipt) for name in 'ABC'})

    def test_exact_scale_gate_rejects_pilot_partial_and_wrong_budgets(self):
        valid = self._record()['head_receipts']['A']
        ev.validate_completed_receipt(valid)
        changes = [('contract', 'kion-three-head-pilot-head-v1'), ('rows_per_epoch', 32768),
                   ('batch_size', 64), ('epochs_requested', 1), ('epochs_completed', 1),
                   ('steps_per_head', 768), ('head_steps', {name: 6143 for name in 'ABC'})]
        for key, value in changes:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    ev.validate_completed_receipt(dict(valid, **{key: value}))

    def test_wrong_completed_scale_cannot_open_provider(self):
        receipt = self._record()['head_receipts']['A']
        receipt['rows_per_epoch'] = 32768
        bank = SimpleNamespace(open_context=mock.Mock(side_effect=AssertionError('Provider opened')))
        head = SimpleNamespace(preflight_heads=mock.Mock(return_value=receipt))
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary)/'selection.json'
            ev.freeze_selection(selection)
            request = dict(deadline_utc='2099-01-01T00:00:00Z', validation_selection=str(selection),
                           evaluation_dir=str(Path(temporary)/'evaluation'), bank_manifest_sha256='a'*64)
            with mock.patch.object(ev, 'load_module', side_effect=[bank, head]):
                with self.assertRaises(ValueError):
                    ev.evaluate(request)
        bank.open_context.assert_not_called()

    def test_fixed_analysis_scope_and_margin(self):
        self.assertEqual(analysis.COMPARISONS, (('C', 'B'), ('B', 'A'), ('C', 'A')))
        self.assertEqual(analysis.MINIMUM_USEFUL_MAP_GAIN, .002)
        self.assertEqual(ev.ARMS, ('A', 'B', 'C', 'GPT', 'RRF'))

    def test_analysis_gate_rejects_partial_and_unequal_heads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evaluation.json'
            document = self._record()
            ev.write_json(path, document)
            ev.write_json(path.with_name('cleanup.json'), self._cleanup())
            self.assertEqual(analysis.gate(path), document)
            document['head_receipts']['C']['status'] = 'stopped'
            path.write_text(json.dumps(document), encoding='utf-8')
            with self.assertRaises(ValueError):
                analysis.gate(path)

    def test_analysis_gate_rejects_target_injection_and_changed_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evaluation.json'
            for key, value in [('no_target_injection', False), ('metric_denominator', 3982)]:
                document = self._record()
                document[key] = value
                path.write_text(json.dumps(document), encoding='utf-8')
                with self.assertRaises(ValueError):
                    analysis.gate(path)

    def test_analysis_gate_rejects_equal_but_old_scale_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evaluation.json'
            for key, value in [('contract', 'kion-three-head-pilot-head-v1'), ('rows_per_epoch', 32768),
                               ('epochs_completed', 1), ('steps_per_head', 768)]:
                document = self._record()
                for receipt in document['head_receipts'].values():
                    receipt[key] = value
                path.write_text(json.dumps(document), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    analysis.gate(path)

    def _cleanup(self):
        return dict(evaluation_completed=True, evidence=dict(teardown_completed=True, errors=[]))

    def test_analysis_gate_rejects_missing_or_failed_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evaluation.json'
            ev.write_json(path, self._record())
            with self.assertRaises(FileNotFoundError):
                analysis.gate(path)
            cleanup_path = path.with_name('cleanup.json')
            for cleanup in [dict(evaluation_completed=False, evidence=self._cleanup()['evidence']),
                            dict(evaluation_completed=True, evidence=dict(teardown_completed=False, errors=[])),
                            dict(evaluation_completed=True, evidence=dict(teardown_completed=True, errors=['failure'])),
                            dict(evaluation_completed=True, evidence={})]:
                cleanup_path.write_text(json.dumps(cleanup), encoding='utf-8')
                with self.subTest(cleanup=cleanup), self.assertRaises(ValueError):
                    analysis.gate(path)
            cleanup_path.write_text(json.dumps(self._cleanup()), encoding='utf-8')
            self.assertEqual(analysis.gate(path), self._record())

    def test_evaluation_cleanup_failure_records_evidence_and_raises(self):
        receipt = self._record()['head_receipts']['A']
        failure = dict(teardown_completed=False, errors=[dict(owner='provider', error='synthetic')])
        head = SimpleNamespace(preflight_heads=mock.Mock(return_value=receipt))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'evaluation'
            selection = Path(temporary)/'selection.json'
            ev.freeze_selection(selection)
            def open_context(request, checked):
                output.mkdir()
                # No numeric runtime is loaded; fail at the first context lookup.
                return {}
            bank = SimpleNamespace(open_context=open_context, close_context=mock.Mock(return_value=failure))
            request = dict(deadline_utc='2099-01-01T00:00:00Z', validation_selection=str(selection),
                           evaluation_dir=str(output), bank_manifest_sha256='a'*64)
            with mock.patch.object(ev, 'load_module', side_effect=[bank, head]):
                with self.assertRaisesRegex(ValueError, 'Evaluation context teardown failed'):
                    ev.evaluate(request)
            self.assertEqual(ev.read_json(output/'cleanup.json'),
                             dict(evaluation_completed=False, evidence=failure))
            bank.close_context.assert_called_once_with({})

    def _paired_records(self):
        current = self._record()
        current.update(selection=dict(path='copied/selection.json', sha256='s'*64), population=158760,
                       seed=ev.SEED, batch_size=128, candidate_chunk_size=64,
                       rrf=dict(constant=60), tie_break='score descending, raw item ID ascending',
                       frozen_teacher=dict(sha256='t'*64), feature_schema_sha256='f'*64,
                       primary_comparison='C-B', candidate_contract=dict(natural_quotas=[512, 256, 256],
                       cache_root='new/cache', validation_graph=dict(root='original/graph', origin=738011,
                       train_cache_read_only=True, validation_cache_read_only=True, rebuild_forbidden=True)),
                       arrays={name: dict(path='new/'+name+'.npy', sha256='a'*64, bytes=128,
                                         dtype='int32', shape=[4096]) for name in analysis.PRIOR_INVARIANT_ARRAYS},
                       metrics={name: dict(map10=.1, candidate_coverage_count=3358) for name in ev.ARMS})
        prior = json.loads(json.dumps(current))
        prior['contract'] = ev.SELECTION_CONTRACT
        prior['selection']['path'] = 'original/selection.json'
        prior['candidate_contract']['cache_root'] = 'old/cache'
        prior['candidate_contract']['validation_graph'] = dict(root='original/graph', origin=738011,
            train_cache_read_only=True, maximum_bytes=67108864, original_build_implementation=True)
        for value in prior['arrays'].values():
            value['path'] = 'old/'+Path(value['path']).name
        for receipt in prior['head_receipts'].values():
            receipt.update(contract='kion-three-head-pilot-head-v1', rows_per_epoch=32768,
                           steps_per_head=768, head_steps={name: 768 for name in 'ABC'})
        return prior, current

    def test_prior_gate_pins_metadata_and_accepts_only_operational_differences(self):
        prior, current = self._paired_records()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'prior.json'
            ev.write_json(path, prior)
            request = dict(prior_evaluation=str(path), prior_evaluation_sha256=ev.sha(path))
            reference = analysis.prior_gate(request, current)
            self.assertEqual(reference['metrics'], prior['metrics'])
            self.assertEqual(reference['identical_descriptor_arrays'], list(analysis.PRIOR_INVARIANT_ARRAYS))
            self.assertIn('descriptive comparison only', reference['limitation'])
            self.assertIn('Same global row-shuffle policy', reference['training_order'])
            with self.assertRaises(ValueError):
                analysis.prior_gate({}, current)
            request['prior_evaluation_sha256'] = '0'*64
            with self.assertRaises(ValueError):
                analysis.prior_gate(request, current)

    def test_prior_gate_rejects_scope_math_baseline_and_descriptor_changes(self):
        prior, original = self._paired_records()
        mutations = [('selection', lambda d: d['selection'].update(sha256='changed')),
                     ('teacher', lambda d: d['frozen_teacher'].update(sha256='changed')),
                     ('quota', lambda d: d['candidate_contract'].update(natural_quotas=[256, 256, 512])),
                     ('unknown_candidate_field', lambda d: d['candidate_contract'].update(new_math_policy=True)),
                     ('graph_origin', lambda d: d['candidate_contract']['validation_graph'].update(origin=738012)),
                     ('graph_root', lambda d: d['candidate_contract']['validation_graph'].update(root='changed')),
                     ('baseline', lambda d: d['metrics']['GPT'].update(map10=.2))]
        for name in analysis.PRIOR_INVARIANT_ARRAYS:
            for key, value in [('sha256', 'changed'), ('dtype', 'float64'), ('bytes', 256), ('shape', [1024])]:
                mutations.append((name+'.'+key, lambda d, n=name, k=key, v=value: d['arrays'][n].update({k: v})))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'prior.json'
            ev.write_json(path, prior)
            request = dict(prior_evaluation=str(path), prior_evaluation_sha256=ev.sha(path))
            for label, mutate in mutations:
                current = json.loads(json.dumps(original))
                mutate(current)
                with self.subTest(label=label), self.assertRaises(ValueError):
                    analysis.prior_gate(request, current)

    def test_preflight_precedes_provider_open(self):
        tree = ast.parse((HERE/'evaluate.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        preflight = next(node.lineno for node in calls if node.func.attr == 'preflight_heads')
        provider = next(node.lineno for node in calls if node.func.attr == 'open_context')
        self.assertLess(preflight, provider)


if __name__ == '__main__':
    unittest.main()
