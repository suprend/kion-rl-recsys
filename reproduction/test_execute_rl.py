"""Small stdlib fixtures only: never decode a model or launch a training backend."""
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import execute_rl as rl


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 10, 25, tzinfo=timezone.utc)

    def guard(self, **changes):
        args = dict(deadline=(self.now + timedelta(hours=1)).isoformat(),
                    cancel_file=self.root / 'STOP', cap_gib=4,
                    snapshot=lambda: dict(rss=1024, available=3 * 2**30), now=lambda: self.now)
        args.update(changes)
        return rl.RuntimeGuard(**args)

    def test_deadline_cancel_resources(self):
        self.guard()(stage='admission')
        for deadline in (self.now.isoformat(), '2026-10-25T03:00:00',
                         (self.now + timedelta(days=2)).isoformat()):
            with self.assertRaises(ValueError):
                self.guard(deadline=deadline)
        for stats in (dict(rss=5 * 2**30, available=3 * 2**30),
                      dict(rss=1024, available=2**30),
                      dict(rss=1024, available=3 * 2**30, gpu_allocated=5 * 2**30)):
            with self.assertRaises(ValueError):
                self.guard(snapshot=lambda: stats)(stage='over-budget')
        guard = self.guard()
        self.now += timedelta(hours=2)
        with self.assertRaisesRegex(ValueError, 'deadline'):
            guard()
        (self.root / 'STOP').touch()
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            self.guard()()

    def test_exclusive_lock_and_durable_abort(self):
        with self.assertRaisesRegex(RuntimeError, 'fixture abort'):
            with rl.ExclusiveRun(self.root, 'abc') as lease:
                self.assertTrue(lease.lock.exists())
                with self.assertRaises(FileExistsError):
                    with rl.ExclusiveRun(self.root, 'other'):
                        self.fail('Concurrent reservation admitted')
                lease.event('fit-reserved', fit_ordinal=1)
                raise RuntimeError('fixture abort')
        self.assertFalse((self.root / '.rl-execution.lock').exists())
        rows = [json.loads(v) for v in (self.root / '.rl-execution-ledger.jsonl').read_text().splitlines()]
        self.assertEqual([r['event'] for r in rows], ['reserved', 'fit-reserved', 'aborted'])
        self.assertEqual(len({r['reservation_id'] for r in rows}), 1)

    def preparation(self):
        value = dict(contract=rl.prep.CONTRACT, scientific_configuration=rl.prep.SCIENCE,
                     output=str(self.root / 'output'), bindings=[], source_pins={})
        value['plan_sha256'] = rl.prep.canonical_hash(value)
        return value

    def test_plan_selection_digest_and_default_no_backend(self):
        preparation = self.preparation()
        six = rl.execution_plan(preparation)
        self.assertEqual(len(six['fits']), 6)
        one = rl.execution_plan(preparation, 'residual', 20261025)
        self.assertEqual(one['fits'], [dict(utility='residual', seed=20261025, beta=.16274030138555867)])
        rl.verify_digest(one, one['plan_sha256'])
        one['fits'][0]['beta'] = .1627403
        with self.assertRaises(ValueError):
            rl.verify_digest(one, one['plan_sha256'])
        for kwargs in (dict(utility='residual'), dict(utility='other', seed=20261024)):
            with self.assertRaises(ValueError):
                rl.execution_plan(preparation, **kwargs)
        path = self.root / 'preparation.json'
        rl.write_new(path, preparation)
        # A sentinel module cannot be used as an actual backend; default only hashes source files.
        with patch.dict(sys.modules, {'torch': None, 'numpy': None, 'psutil': None}), redirect_stdout(io.StringIO()) as stream:
            rl.main(['--plan', str(path)])
        self.assertFalse(json.loads(stream.getvalue())['training_launched'])
        self.assertFalse((self.root / 'output').exists())

    def test_missing_execution_flags_reject_before_dispatch(self):
        path = self.root / 'plan.json'
        rl.write_new(path, self.preparation())
        with patch.object(rl, 'execute') as execute, self.assertRaises(SystemExit):
            rl.main(['--plan', str(path), '--execute-training'])
        execute.assert_not_called()

    def test_lease_precedes_backend_and_abort_survives(self):
        def backend(*args):
            self.assertTrue((self.root / '.rl-execution.lock').exists())
            raise RuntimeError('backend unavailable')
        with patch.object(rl, 'HERE', self.root), patch.object(rl, 'validate_plan'), \
                patch.object(rl, '_execute_locked', side_effect=backend), self.assertRaises(RuntimeError):
            rl.execute({'run_state': str(self.root)}, 'abc', 'deadline', self.root / 'STOP', 4)
        ledger = (self.root / '.rl-execution-ledger.jsonl').read_text()
        self.assertIn('backend unavailable', ledger)

    def test_no_overwrite(self):
        path = self.root / 'receipt.json'
        rl.write_new(path, {'original': True})
        with self.assertRaises(FileExistsError):
            rl.write_new(path, {'original': False})
        self.assertEqual(rl.read(path), {'original': True})

    def test_artifact_root_reconstruction(self):
        artifacts = self.root / 'artifacts'
        cache = artifacts / 'c-rl-20261024-r1/policy/train-cache-r1/cache.json'
        bank = self.root / 'bank.json'
        repo = self.root / 'repo'
        preparation = self.preparation()
        preparation.update(bindings=[dict(role='metadata:cache', physical=str(cache)),
                                     dict(role='metadata:bank', physical=str(bank))],
                           source_pins={(repo / 'source/matched-kl/policy/policy.py').as_posix(): 'x'})
        plan = dict(preparation=preparation, contract=rl.CONTRACT, fits=[dict(utility='completion', seed=20261024)])
        with patch.object(rl, 'verify_digest'), patch.object(rl, 'execution_plan', return_value=plan), \
             patch.object(rl.prep, 'plan', return_value=preparation) as fresh, \
             patch.object(rl.bound, 'BindingResolver'):
            rl.validate_plan(plan, 'digest')
        fresh.assert_called_once_with(artifacts, str(bank), preparation['output'], root=repo)

    def run_fake_fit(self, skip=False):
        class Flag:
            def __init__(self, value): self.value = value
            def __bool__(self): return bool(self.value)
            def all(self): return self
            def any(self): return self
        class Number:
            def __init__(self, value): self.value, self.grad = value, None
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return Number(self.value)
            def requires_grad_(self, flag): return self
            def sum(self): return self
            def backward(self): pass
            def __float__(self): return float(self.value)
            def __sub__(self, other): return Number(self.value - other.value)
            def __pow__(self, value): return Number(self.value ** value)
            def __radd__(self, value): return Number(value + self.value)
            def __ne__(self, value): return Flag(self.value != value)
        class Model:
            def __init__(self):
                self.weight = Number(1.)
                self.weight.grad = Number(.1)
            def eval(self): return self
            def parameters(self): return [self.weight]
            def state_dict(self): return {'weight': self.weight}
        model = Model()
        optimizer_args = []
        class Optimizer:
            def __init__(self, parameters, **kwargs):
                optimizer_args.append(kwargs)
                self.parameters = list(parameters)
            def zero_grad(self, **kwargs): pass
            def step(self):
                for parameter in self.parameters: parameter.value += .001
        class Scaler:
            def __init__(self, *args, **kwargs): self.value = kwargs['init_scale']
            def scale(self, value): return value
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.value
            def step(self, optimizer):
                if not skip: optimizer.step()
            def update(self):
                if skip: self.value /= 2
        class Advantages:
            def __getitem__(self, positions): return [1.] * len(positions)
        torch = SimpleNamespace(device=lambda value: value, as_tensor=lambda *a, **kw: Number(1),
            ones=lambda *a, **kw: Number(1), float32='float32', isfinite=lambda value: Flag(True),
            equal=lambda a, b: a.value == b.value, optim=SimpleNamespace(Adam=Optimizer),
            amp=SimpleNamespace(GradScaler=Scaler), nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda *a, **kw: Number(.1))),
            save=lambda payload, stream: stream.write(b'fake-checkpoint'))
        np = SimpleNamespace(array=lambda value, **kw: value)
        policy = SimpleNamespace(producer_layout=lambda origins: {}, distribution=lambda *a: ([], []),
            effective_advantage=lambda *a: (Advantages(), [True] * 512), diagnostics=lambda *a: {'fixture': True},
            prepare_shaped_inputs=lambda *a: None, precision=lambda *a: nullcontext(), exact_loss=lambda *a: Number(1),
            score_shaped=lambda *a: Number(1))
        plan = dict(plan_sha256='abc', adapter_sources={}, preparation=dict(source_pins={}, bindings=[]))
        output = self.root / ('skip-fit' if skip else 'success-fit')
        with patch.object(rl, 'score_all', return_value=[]):
            result = rl.fit_one(plan, dict(utility='completion', seed=20261024, beta=.5), output, {}, None,
                dict(C_checkpoint_sha256='original'), dict(item_table=[], origins=[], reference_scores=Advantages()),
                [], [], [], SimpleNamespace(load_original_heads=lambda *a: {'C': model}), policy, torch, np, self.guard(), {})
        return result, optimizer_args

    def test_fake_success_fixed_updates_checkpoints_sampler(self):
        result, options = self.run_fake_fit()
        self.assertEqual(result['updates'], 1000)
        self.assertEqual(result['actual_optimizer_updates'], 1000)
        self.assertEqual([c['step'] for c in result['checkpoints']], [0, 250, 500, 1000])
        self.assertEqual(options, [dict(lr=1e-5, weight_decay=0)])
        self.assertTrue(result['reference_weights_unchanged'])
        rng, digest = random.Random(20261024), hashlib.sha256()
        for _ in range(1000):
            positions = [rng.choice(list(range(512))) for _ in range(16)]
            digest.update(json.dumps(positions, separators=(',', ':')).encode('ascii') + b'\n')
        self.assertEqual(result['sampler_positions_sha256'], digest.hexdigest())

    def test_fake_amp_skipped_update_aborts_without_retry(self):
        with self.assertRaisesRegex(ValueError, 'AMP skipped update'):
            self.run_fake_fit(skip=True)
        receipt = rl.read(self.root / 'skip-fit/aborted.json')
        self.assertEqual(receipt['updates'], 0)
        self.assertEqual([c['step'] for c in receipt['checkpoints']], [0])

    def test_fresh_parity_failure_prevents_optimizer_and_preserves_receipt(self):
        class Model:
            def eval(self): return self
            def parameters(self): return []
            def state_dict(self): return {}
        torch = MagicMock()
        heads = SimpleNamespace(load_original_heads=lambda *args: {'C': Model()})
        np = SimpleNamespace(array=lambda value, **kw: value)
        plan = dict(plan_sha256='abc', adapter_sources={}, preparation=dict(source_pins={}, bindings=[]))
        output = self.root / 'fit'
        with patch.object(rl, 'score_all', side_effect=ValueError('fresh parity failure')), \
                self.assertRaisesRegex(ValueError, 'fresh parity'):
            rl.fit_one(plan, dict(utility='completion', seed=20261024, beta=.5), output, {}, None,
                       dict(C_checkpoint_sha256='original'), dict(item_table=[], origins=[]), None, None, None,
                       heads, SimpleNamespace(producer_layout=lambda value: {}), torch, np, self.guard(), {})
        torch.optim.Adam.assert_not_called()
        receipt = rl.read(output / 'aborted.json')
        self.assertEqual(receipt['updates'], 0)
        self.assertEqual(receipt['state'], 'aborted')


if __name__ == '__main__':
    unittest.main()
