"""Synthetic stdlib contract tests; no real arrays, models or provider are opened."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('evaluate.py')
spec = importlib.util.spec_from_file_location('scientific3_evaluate_tested', PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def rank(ids, scores, raw, k=10):
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), int(raw[i])))[:k]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.c_sha = 'c'*64
        self.entries = {}
        for i, arm in enumerate(ev.ARMS[1:]):
            checkpoint = self.root/(arm+'.pt')
            checkpoint.write_bytes(('synthetic-weight-'+str(i)).encode())
            receipt = self.root/(arm+'.json')
            entry = {'path': checkpoint.as_posix(), 'sha256': ev.sha(checkpoint), 'step': 1000}
            document = {'state': 'completed', 'updates': 1000, 'supervised_updates': 0,
                        'frozen_encoder': True, 'C_checkpoint_sha256': self.c_sha,
                        'checkpoints': [entry.copy()], 'contract': arm+'-contract'}
            receipt.write_text(json.dumps(document), encoding='utf-8')
            entry['training'] = {'path': receipt.as_posix(), 'sha256': ev.sha(receipt)}
            self.entries[arm] = entry

    def tearDown(self):
        self.temp.cleanup()

    def request(self):
        return {'scientific_checkpoints': self.entries}

    def mutate_receipt(self, arm, field, value):
        entry = self.entries[arm]['training']
        path = Path(entry['path'])
        receipt = ev.read(path)
        receipt[field] = value
        path.write_text(json.dumps(receipt), encoding='utf-8')
        entry['sha256'] = ev.sha(path)

    def test_distinct_actual1000_lineages(self):
        result = ev.checkpoint_receipts(self.request(), self.c_sha)
        self.assertEqual(set(result), set(ev.ARMS[1:]))
        self.assertEqual(result['RL-residual']['training_contract'], 'RL-residual-contract')
        self.assertEqual(result['RL-completion']['updates'], 1000)

    def test_selected_alias_is_rejected(self):
        self.entries['selected'] = self.entries.pop('RL-residual')
        with self.assertRaisesRegex(ValueError, 'separately named'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_selected0_cannot_replace_trained_arm(self):
        self.entries['RL-completion']['step'] = 0
        with self.assertRaisesRegex(ValueError, 'latest1000'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_wrong_ancestor_rejected(self):
        self.mutate_receipt('RL-residual', 'C_checkpoint_sha256', 'wrong')
        with self.assertRaisesRegex(ValueError, 'ancestor'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_unfinished_training_rejected(self):
        self.mutate_receipt('RL-residual', 'state', 'running')
        with self.assertRaisesRegex(ValueError, 'Completed'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_supervised_or_unfrozen_training_rejected(self):
        for field, value in [('supervised_updates', 1), ('frozen_encoder', False)]:
            with self.subTest(field=field):
                self.mutate_receipt('RL-residual', field, value)
                with self.assertRaisesRegex(ValueError, 'Frozen'):
                    ev.checkpoint_receipts(self.request(), self.c_sha)
                self.mutate_receipt('RL-residual', field, 0 if field == 'supervised_updates' else True)

    def test_source_or_receipt_tamper_rejected(self):
        Path(self.entries['RL-completion']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'pin differs'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_legacy_subrequest_paths_are_normalized_without_mutation(self):
        source = self.root/'synthetic-module.py'
        source.write_text('VALUE = 1\n', encoding='utf-8')
        pin = {'path': source.as_posix(), 'sha256': ev.sha(source)}
        request = {'policy_module': pin, 'utility_module': pin, 'source_hashes': {}}
        result = ev.adapter_request(request)
        self.assertEqual(result['policy_module'], source.as_posix())
        self.assertEqual(result['utility_module'], source.as_posix())
        self.assertEqual(result['source_hashes'][source.as_posix()], pin['sha256'])
        self.assertEqual(request['source_hashes'], {})
        self.assertIs(request['policy_module'], pin)
        request['source_hashes'][source.as_posix()] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            ev.adapter_request(request)

    def test_exact_screen_parity_has_no_rounding(self):
        self.assertTrue(ev.exact_legacy_parity({'map10': .125, 'rows': 256}, {'map10': .125, 'rows': 256})['exact'])
        with self.assertRaisesRegex(ValueError, 'exact parity failed'):
            ev.exact_legacy_parity({'map10': .125000000000001}, {'map10': .125})

    def test_rank_tie_changes_inside_tolerance_are_counted(self):
        parity = ev.new_parity({'parity_atol': .01, 'parity_rtol': 0.})
        ev.update_rank_parity(parity, [[1., 1.]], [[1., 1.001]], [[3, 4]], [[10, 20]], rank)
        self.assertEqual(parity['top10_changed_rows'], 1)
        self.assertEqual(parity['changes_with_margin_above_tolerance'], 0)

    def test_material_rank_reversal_rejected_by_margin(self):
        parity = ev.new_parity({'parity_atol': .001, 'parity_rtol': 0.})
        ev.update_rank_parity(parity, [[1., 0.]], [[0., 1.]], [[3, 4]], [[10, 20]], rank)
        self.assertEqual(parity['changes_with_margin_above_tolerance'], 2)

    def test_nonfinite_parity_rejected(self):
        parity = ev.new_parity({'parity_atol': 0., 'parity_rtol': 0.})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            ev.update_rank_parity(parity, [[1.]], [[float('nan')]], [[3]], [[10]], rank)

    def test_nonfinite_or_negative_tolerance_rejected(self):
        for value in [-1., float('nan'), float('inf')]:
            with self.assertRaisesRegex(ValueError, 'tolerances'):
                ev.new_parity({'parity_atol': value, 'parity_rtol': 0.})

    def test_fresh_outputs_cannot_escape_or_overwrite(self):
        with patch.object(ev, 'HERE', self.root):
            self.assertEqual(ev.fresh_output(self.root/'new'), self.root/'new')
            with self.assertRaisesRegex(ValueError, 'below'):
                ev.fresh_output(self.root.parent/'escape')
            existing = self.root/'old'
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, 'Fresh'):
                ev.fresh_output(existing)

    def test_completed_schema_has_honest_stage_and_denominator(self):
        record = ev.base_record({'policy_module': {}, 'residual_module': {}, 'source_hashes': {}}, 'screen', 256, {})
        self.assertEqual(record['metric_denominator'], 256)
        self.assertEqual(record['arms'], ['C', 'RL-completion', 'RL-residual'])
        self.assertTrue(record['unknown_and_offpool_are_zero'])
        self.assertEqual(record['temperature'], 1.)
        self.assertNotIn('selected', record)

    def test_import_does_not_load_ml_dependencies(self):
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('torch', sys.modules)

    def test_full_feature_call_is_shared_outside_head_loop(self):
        tree = ast.parse(PATH.read_text(encoding='utf-8'))
        full = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'full')
        calls = [node for node in ast.walk(full) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'candidate_features']
        self.assertEqual(len(calls), 1)
        model_loops = [node for node in ast.walk(full) if isinstance(node, ast.For)
                       and isinstance(node.target, ast.Tuple) and any(isinstance(item, ast.Name) and item.id == 'head'
                                                                    for item in node.target.elts)]
        self.assertEqual(len(model_loops), 1)
        self.assertNotIn(calls[0], list(ast.walk(model_loops[0])))
        self.assertEqual((ev.N, ev.WIDTH, ev.BATCH, ev.CHUNK), (4096, 1024, 128, 64))


if __name__ == '__main__':
    unittest.main()
