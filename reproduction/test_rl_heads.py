"""Synthetic mocked checkpoints only; never deserialize a real checkpoint."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rl_heads as h
from rl_bindings import BindingResolver

SOURCE = Path(__file__).resolve().parents[1] / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/three-head-262k-20261024-r1/heads.py'


class Fixture:
    def __init__(self, root):
        self.root, self.entries = root, []

    def add(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(data)
        logical = 'Z:/frozen/' + name
        digest = hashlib.sha256(data).hexdigest()
        self.entries.append(dict(logical=logical, physical=path.as_posix(), sha256=digest, bytes=len(data)))
        return digest


class HeadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))
        self.module = h.load_frozen_heads(SOURCE)
        self.records = [dict(epoch=i + 1, tiny_fixture=True) for i in range(3)]
        self.module.epoch_record = lambda i: self.records[i]
        self.request = dict(head_dir='Z:/frozen/heads', output_dir='Z:/unused', bank_dir='Z:/frozen/bank',
            seed=20261024, batch_size=128, epochs=3, teacher={'path': 'X:/opaque/teacher', 'sha256': 'f' * 64},
            feature_schema_sha256=self.module.FEATURE_SCHEMA_SHA256, source_columns=self.module.SOURCE_COLUMNS)
        self.bank = dict(contract=self.module.BANK_CONTRACT, state='completed', split='train', train_rows=262144,
            natural_covered_only=False, same_day_positives_excluded=True,
            **{k: self.request[k] for k in ('teacher', 'feature_schema_sha256', 'source_columns')})

    def prepare(self, mutate_receipt=None, mutate_bank=None, mutate_payload=None):
        bank = copy.deepcopy(self.bank)
        if mutate_bank:
            mutate_bank(bank)
        pin = self.f.add('bank/bank.json', bank)
        self.request['bank_manifest_sha256'] = pin
        receipt = dict(contract=self.module.CONTRACT, status='completed', epochs_requested=3, epochs_completed=3,
            steps_per_head=6144, head_steps={n: 6144 for n in 'ABC'}, rows_per_epoch=262144,
            batch_size=128, seed=20261024, common_batches=True, shuffle=self.module.SHUFFLE,
            raw_gather=self.module.RAW_GATHER, trained_from='fresh-common-init-final-zero',
            epoch_records=self.records, bank_manifest_sha256=pin,
            **{k: self.request[k] for k in ('teacher', 'feature_schema_sha256', 'source_columns')})
        if mutate_receipt:
            mutate_receipt(receipt)
        hashes = {}
        for name in 'ABC':
            payload = dict(contract=self.module.CONTRACT, name=name, complete=True, config=self.module.CONFIG,
                raw_indices=list(self.module.mask_indices(name)), receipt=receipt, state_dict={'synthetic': name})
            if mutate_payload:
                mutate_payload(name, payload)
            hashes[name] = self.f.add('heads/' + name + '/head.pt', payload)
        self.f.add('heads/training.json', dict(receipt, checkpoint_sha256=hashes))
        return BindingResolver(self.f.entries)

    def test_frozen_import_is_stdlib_and_rejects_changed_source(self):
        self.assertEqual(self.module.CONFIG['mlp_input_width'], 800)
        path = self.f.root / 'bad.py'
        path.write_text('raise AssertionError("must never execute")', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            h.load_frozen_heads(path)

    def test_preflight_relocated_and_opaque_teacher(self):
        bindings = self.prepare()
        with patch.object(h, 'load_frozen_heads', return_value=self.module):
            self.assertEqual(h.preflight_heads_bound(self.request, bindings, SOURCE)['head_steps']['A'], 6144)

    def test_missing_A_and_changed_checkpoint(self):
        bindings = self.prepare()
        (self.f.root / 'heads/A/head.pt').write_text('changed', encoding='utf-8')
        with patch.object(h, 'load_frozen_heads', return_value=self.module), self.assertRaises(ValueError):
            h.preflight_heads_bound(self.request, bindings, SOURCE)

    def test_bank_teacher_mismatch(self):
        bindings = self.prepare(mutate_bank=lambda b: b.update(teacher={'wrong': True}))
        with patch.object(h, 'load_frozen_heads', return_value=self.module), self.assertRaisesRegex(ValueError, 'lineage'):
            h.preflight_heads_bound(self.request, bindings, SOURCE)

    def test_training_budget_mismatch(self):
        bindings = self.prepare(mutate_receipt=lambda r: r.update(head_steps={'A': 1, 'B': 6144, 'C': 6144}))
        with patch.object(h, 'load_frozen_heads', return_value=self.module), self.assertRaisesRegex(ValueError, 'budget'):
            h.preflight_heads_bound(self.request, bindings, SOURCE)

    def backend(self):
        calls = []
        class Model:
            def load_state_dict(self, state, strict):
                calls.append(('state', state, strict))
            def parameters(self):
                return [1]
            def eval(self):
                calls.append(('eval',))
        def load(stream, map_location, weights_only):
            self.assertEqual(map_location, 'cpu')
            self.assertIs(weights_only, True)
            calls.append(('load',))
            return json.load(stream)
        self.module.make_heads = lambda torch, seed, device: {n: Model() for n in 'ABC'}
        return SimpleNamespace(load=load, isfinite=lambda p: SimpleNamespace(all=lambda: True)), calls

    def test_load_all_three_strict_and_eval_without_real_torch(self):
        bindings = self.prepare()
        backend, calls = self.backend()
        with patch.object(h, 'load_frozen_heads', return_value=self.module):
            models = h.load_heads_bound(self.request, bindings, backend, SOURCE)
        self.assertEqual(set(models), set('ABC'))
        self.assertEqual(calls.count(('load',)), 3)
        self.assertEqual(calls.count(('eval',)), 3)
        self.assertTrue(all(c[2] is True for c in calls if c[0] == 'state'))

    def test_bad_B_payload_fails_no_partial_return(self):
        bindings = self.prepare(mutate_payload=lambda name, p: p.update(complete=False) if name == 'B' else None)
        backend, calls = self.backend()
        with patch.object(h, 'load_frozen_heads', return_value=self.module), self.assertRaisesRegex(ValueError, 'incomplete'):
            h.load_heads_bound(self.request, bindings, backend, SOURCE)
        self.assertEqual(calls.count(('load',)), 2)


class ProofTests(unittest.TestCase):
    def test_all_proof_fields_and_gradients(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = Fixture(Path(root))
            cache_pin = fixture.add('cache/cache.json', {'synthetic': True})
            proof = dict(state='completed', contract='kion-current-C1070-natural1024-H1-v1',
                implementation='producer-shape-slot-v2', implementation_source_sha256=h.POLICY_SHA256,
                cache_sha256=cache_pin, C_checkpoint_sha256='c' * 64, states=512,
                all_actor_reference_exact=True, all_cached_allclose=True, all_cached_top10_exact=True,
                optimizer_created=False, learning_updates=0, actor_reference_weights_unchanged=True,
                gradient_fixture=dict(logical_rows=16, dummy_output_gradient_max=0,
                    gradients={'weight': dict(finite=True, nonzero=True)}))
            proof_pin = fixture.add('proof.json', proof)
            resolver = BindingResolver(fixture.entries)
            request = dict(cache_dir='Z:/frozen/cache', parity_proof=dict(path='Z:/frozen/proof.json', sha256=proof_pin))
            manifest = dict(C_checkpoint_sha256='c' * 64)
            self.assertEqual(h.validate_parity_proof_bound(request, manifest, resolver), proof)
            for field, bad in [('implementation_source_sha256', 'bad'), ('states', 511),
                               ('all_cached_top10_exact', False), ('learning_updates', 1)]:
                changed = dict(proof, **{field: bad})
                with patch.object(resolver, 'read_bound', return_value=changed), self.assertRaises(ValueError):
                    h.validate_parity_proof_bound(request, manifest, resolver)
            bad = copy.deepcopy(proof)
            bad['gradient_fixture']['gradients']['weight']['nonzero'] = False
            with patch.object(resolver, 'read_bound', return_value=bad), self.assertRaisesRegex(ValueError, 'nonzero'):
                h.validate_parity_proof_bound(request, manifest, resolver)

    def test_preregistration_preserves_historical_beta(self):
        value = dict(contract='kion-C-residual-RL-preregistered-v1', chosen_utility='rm-minus-frozen-item-mean',
            formula='r_RM(s,i)-m_fit(i)', coefficient=1., beta=.5, temperature=1., sampler_seed=20261024,
            checkpoint_steps=[0, 250, 500, 1000], start_from='original-C', original_results_known=True,
            one_additional_fit_only=True, primary_contrast='fixed-latest1000', no_gpt_unfreeze=True, no_rm_fit=True,
            no_sft=True, optimizer=dict(name='Adam', updates=1000, batch_size=16, lr=1e-5, grad_clip=1, weight_decay=0, alpha_ce=0))
        with tempfile.TemporaryDirectory() as root:
            fixture = Fixture(Path(root))
            pin = fixture.add('prereg.json', value)
            resolver = BindingResolver(fixture.entries)
            request = dict(preregistration='Z:/frozen/prereg.json', preregistration_sha256=pin)
            self.assertEqual(h.validate_preregistration_bound(request, resolver), value)
            value['beta'] = .16274030138555867
            with patch.object(resolver, 'read_bound', return_value=value), self.assertRaises(ValueError):
                h.validate_preregistration_bound(request, resolver)


if __name__ == '__main__':
    unittest.main()
