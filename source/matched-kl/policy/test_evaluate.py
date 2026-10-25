"""Synthetic source-contract tests; no real array, checkpoint or model is opened."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name('evaluate.py')
spec = importlib.util.spec_from_file_location('kl_multiseed_evaluate_tested', PATH)
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
        for seed in ev.SEEDS:
            for utility in ('completion', 'residual'):
                arm = f'{utility}-{seed}'
                checkpoint = self.root/(arm+'.pt')
                checkpoint.write_bytes(('synthetic-weight-'+arm).encode())
                receipt = self.root/(arm+'.json')
                entry = dict(path=checkpoint.as_posix(), sha256=ev.sha(checkpoint), step=1000,
                             seed=seed, utility=utility)
                document = dict(state='completed', updates=1000, supervised_updates=0,
                                frozen_encoder=True, C_checkpoint_sha256=self.c_sha, beta=.5,
                                checkpoints=[entry.copy()], sampler_seed=seed, utility=utility)
                receipt.write_text(json.dumps(document), encoding='utf-8')
                entry['training'] = dict(path=receipt.as_posix(), sha256=ev.sha(receipt))
                self.entries[arm] = entry

    def tearDown(self):
        self.temp.cleanup()

    def request(self):
        return dict(scientific_checkpoints=self.entries, policy_module={}, residual_module={}, source_hashes={})

    def mutate(self, arm, field, value):
        pin = self.entries[arm]['training']
        path = Path(pin['path'])
        receipt = ev.read(path)
        if value is None:
            receipt.pop(field, None)
        else:
            receipt[field] = value
        path.write_text(json.dumps(receipt), encoding='utf-8')
        pin['sha256'] = ev.sha(path)

    def test_six_final_lineages_three_paired_seeds(self):
        result = ev.checkpoint_receipts(self.request(), self.c_sha)
        self.assertEqual(len(result), 6)
        self.assertEqual(ev.arm_names(self.request())[0], 'C')
        self.assertEqual({(x['seed'], x['utility']) for x in result.values()},
                         {(s, u) for s in ev.SEEDS for u in ('completion', 'residual')})

    def test_missing_pair_or_duplicate_seed_rejected(self):
        self.entries['residual-20261026']['seed'] = 20261025
        with self.assertRaisesRegex(ValueError, 'pair'):
            ev.arm_names(self.request())
        self.entries.pop('residual-20261026')
        with self.assertRaisesRegex(ValueError, 'six'):
            ev.arm_names(self.request())

    def test_unsafe_arm_name_rejected(self):
        self.entries['../escape'] = self.entries.pop('residual-20261026')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            ev.arm_names(self.request())

    def test_seed_type_and_declared_beta_cannot_drift(self):
        self.entries['completion-20261024']['seed'] = 20261024.0
        with self.assertRaisesRegex(ValueError, 'Integer'):
            ev.arm_names(self.request())
        self.entries['completion-20261024']['seed'] = 20261024
        self.entries['completion-20261024']['beta'] = .25
        with self.assertRaisesRegex(ValueError, 'Declared beta'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_windows_casefold_collisions_are_rejected(self):
        self.entries['c'] = self.entries.pop('completion-20261024')
        with self.assertRaisesRegex(ValueError, 'Windows case'):
            ev.arm_names(self.request())
        self.entries['completion-20261024'] = self.entries.pop('c')
        self.entries['COMPLETION-20261024'] = self.entries.pop('residual-20261024')
        with self.assertRaisesRegex(ValueError, 'Windows case'):
            ev.arm_names(self.request())

    def test_seed_and_utility_mislabel_rejected(self):
        self.mutate('completion-20261025', 'sampler_seed', 20261026)
        with self.assertRaisesRegex(ValueError, 'seed differs'):
            ev.checkpoint_receipts(self.request(), self.c_sha)
        self.mutate('completion-20261025', 'sampler_seed', 20261025)
        self.mutate('completion-20261025', 'utility', 'residual')
        with self.assertRaisesRegex(ValueError, 'utility differs'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_invalid_explicit_utility_cannot_fall_back_to_legacy(self):
        self.mutate('completion-20261024', 'utility', 'completion-typo')
        with self.assertRaisesRegex(ValueError, 'utility differs'):
            ev.checkpoint_receipts(self.request(), self.c_sha)
        self.mutate('completion-20261024', 'utility', None)
        with self.assertRaisesRegex(ValueError, 'Legacy completion'):
            ev.checkpoint_receipts(self.request(), self.c_sha)
        self.mutate('completion-20261024', 'contract', 'kion-current-C1070-natural1024-H1-v1')
        self.mutate('completion-20261024', 'objective', 'exact supported centered advantage minus beta KL')
        self.assertEqual(ev.checkpoint_receipts(self.request(), self.c_sha)['completion-20261024']['utility'], 'completion')

    def test_old_seed_must_be_bound_to_pinned_training_preregistration(self):
        arm = 'completion-20261024'
        self.mutate(arm, 'sampler_seed', None)
        with self.assertRaisesRegex(ValueError, 'seed provenance'):
            ev.checkpoint_receipts(self.request(), self.c_sha)
        path = self.root/'prereg.json'
        path.write_text(json.dumps(dict(sampler_seed=20261024)), encoding='utf-8')
        pin = dict(path=path.as_posix(), sha256=ev.sha(path))
        self.entries[arm]['seed_provenance'] = pin
        self.mutate(arm, 'preregistration_sha256', pin['sha256'])
        self.assertEqual(ev.checkpoint_receipts(self.request(), self.c_sha)[arm]['seed'], 20261024)
        self.mutate(arm, 'preregistration_sha256', 'wrong')
        with self.assertRaisesRegex(ValueError, 'not bound'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_latest1000_no_selected_or_partial_arm(self):
        for field, value, match in [('updates', 500, 'Completed'), ('state', 'running', 'Completed'),
                                    ('supervised_updates', 1, 'Frozen'), ('frozen_encoder', False, 'Frozen'),
                                    ('C_checkpoint_sha256', 'bad', 'ancestor'), ('beta', 0, 'beta')]:
            with self.subTest(field=field):
                pin = self.entries['residual-20261024']['training']
                original = Path(pin['path']).read_text(encoding='utf-8')
                self.mutate('residual-20261024', field, value)
                with self.assertRaisesRegex(ValueError, match):
                    ev.checkpoint_receipts(self.request(), self.c_sha)
                Path(pin['path']).write_text(original, encoding='utf-8')
                pin['sha256'] = ev.sha(pin['path'])
        self.entries['completion-20261024']['step'] = 0
        with self.assertRaisesRegex(ValueError, 'latest1000'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_checkpoint_tamper_rejected(self):
        Path(self.entries['completion-20261024']['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'pin differs'):
            ev.checkpoint_receipts(self.request(), self.c_sha)

    def test_adapter_does_not_mutate_request_and_rejects_conflicts(self):
        path = self.root/'module.py'
        path.write_text('VALUE = 1\n', encoding='utf-8')
        pin = dict(path=path.as_posix(), sha256=ev.sha(path))
        request = dict(policy_module=pin, utility_module=pin, source_hashes={})
        result = ev.adapter_request(request)
        self.assertEqual(result['policy_module'], path.as_posix())
        self.assertEqual(request['source_hashes'], {})
        request['source_hashes'][path.as_posix()] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            ev.adapter_request(request)

    def test_fresh_outputs_and_honest_manifest(self):
        with patch.object(ev, 'HERE', self.root):
            self.assertEqual(ev.fresh_output(self.root/'new'), self.root/'new')
            with self.assertRaisesRegex(ValueError, 'below'):
                ev.fresh_output(self.root.parent/'escape')
            existing = self.root/'existing'
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, 'Fresh'):
                ev.fresh_output(existing)
        record = ev.base_record(self.request(), 'screen', 256, {})
        self.assertEqual(record['metric_denominator'], 256)
        self.assertEqual(len(record['arms']), 7)
        self.assertTrue(record['no_seen_filter'] and record['no_target_injection'])

    def test_exact_legacy_screen_diagnostic_parity(self):
        self.assertTrue(ev.exact_legacy_parity({'x': .125}, {'x': .125})['exact'])
        with self.assertRaisesRegex(ValueError, 'exact parity failed'):
            ev.exact_legacy_parity({'x': .1250000000001}, {'x': .125})

    def test_rank_ties_use_raw_id(self):
        self.assertEqual(rank([3, 4], [1, 1], [20, 10]), [1, 0])
        parity = ev.new_parity(dict(parity_atol=0., parity_rtol=0.))
        ev.update_rank_parity(parity, [[1., 1.]], [[1., 1.001]], [[3, 4]], [[10, 20]], rank)
        self.assertEqual(parity['top10_changed_rows'], 1)

    def test_no_numeric_dependency_imported(self):
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('torch', sys.modules)

    def test_shared_feature_call_and_strict_c_gate_are_in_source(self):
        source = PATH.read_text(encoding='utf-8')
        tree = ast.parse(source)
        full = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'full')
        calls = [n for n in ast.walk(full) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'candidate_features']
        loops = [n for n in ast.walk(full) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                 and any(isinstance(x, ast.Name) and x.id == 'head' for x in n.target.elts)]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(loops), 1)
        self.assertNotIn(calls[0], list(ast.walk(loops[0])))
        self.assertIn("parity['max_abs_error'] == 0.", source)
        self.assertIn("parity['top10_changed_rows'] == 0", source)
        self.assertEqual((ev.N, ev.WIDTH, ev.BATCH, ev.CHUNK), (4096, 1024, 128, 64))


if __name__ == '__main__':
    unittest.main()
