"""Tiny mocked loop and admission tests; no original model or arrays loaded."""
from contextlib import contextmanager
import ast
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import train_gpt as t


def config():
    return {'contract': {'architecture': {**{k: t.FIXED[k] for k in
        ('hidden_size', 'layers', 'heads', 'bottleneck', 'person_dim', 'seed', 'text_variant', 'people_variants')},
        'context_length': 64, 'feature_columns': {k: ['fixture'] * n for k, n in
        [('item', 5736), ('user', 3252), ('candidate', 1068)]}},
        'training_schedule': dict(eligible_count=3477437, updates_per_epoch=27235,
            mode='origin-grouped-v1', microbatch=128, accumulation=1, seed=2026),
        'training': {k: t.FIXED[k] for k in ('schedule_total_epochs', 'batch_size', 'accumulation',
            'seed', 'hard_negatives', 'uniform_negatives', 'auxiliary_weight', 'training_order')}},
        'runtime': {'cpu_threads': 2}}


class Census:
    def __init__(self, size): self.size = size
    def __len__(self): return self.size


class TrainingTests(unittest.TestCase):
    def test_request_builder_pins_inventory_and_bindings_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name, value):
                path = root / name
                data = json.dumps(value).encode()
                path.write_bytes(data)
                return path, hashlib.sha256(data).hexdigest()
            manifest, manifest_sha = save('manifest.json', {'data': {'path': 'D:/logical/data'}})
            scope, _ = save('scope.json', {})
            cfg = config()
            cfg['contract']['integration'] = {'scope_decision': {'path': 'D:/logical/scope.json', 'sha256': t.SCOPE_SHA}}
            cfg_path, cfg_sha = save('config.json', cfg)
            bindings = {'D:/logical/features/manifest.json': str(manifest), 'D:/logical/scope.json': str(scope)}
            bp, bs = save('bindings.json', bindings)
            entries = [dict(logical=logical, physical=physical, sha256=manifest_sha if logical.endswith('manifest.json') else t.SCOPE_SHA,
                            kind='metadata') for logical, physical in bindings.items()]
            ip, ins = save('inventory.json', dict(schema_version=1, payloads_hashed=False, entries=entries))
            kwargs = dict(inventory_path=ip, inventory_sha256=ins, bindings_path=bp, bindings_sha256=bs,
                expected_config=cfg_path, expected_config_sha256=cfg_sha, output=root/'model-future',
                request_path=root/'request.json')
            with patch.object(t, 'PROVIDER_MANIFEST_SHA', manifest_sha):
                result = t.prepare_request(**kwargs)
                self.assertEqual(result['state'], 'request-prepared-not-admitted')
                saved = json.loads((root/'request.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['data'], 'D:/logical/data')
                self.assertFalse((root/'model-future').exists())
                with self.assertRaises(ValueError): t.prepare_request(**kwargs)
                kwargs['request_path'] = root/'tampered-request.json'
                kwargs['inventory_sha256'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'pin differs'): t.prepare_request(**kwargs)
                self.assertFalse((root/'tampered-request.json').exists())

    def test_retained_run_h5_two_validation_stop_with_fake_backend(self):
        # Execute the exact retained run body, not a reimplementation. Every
        # numerical dependency is fake and the fixture has two synthetic rows.
        path = t.ROOT / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_complete_train.py'
        source = path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '16180fc30db79f6a5b5c0fc82080eb89658d4b466c65f3a96f66d313c85746d9')
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        lr_calls, updates, validations = [], [], []
        class Schedule:
            eligible = [0, 1]
            updates_per_epoch = 27235  # Test the unchanged denominator seam.
            def __init__(self, *a): pass
            def validate_cursor(self, *a): pass
            def groups(self, epoch, offset): yield 0, [[0, 1]], 0, 1
        class Model:
            def to(self, device): return self
            def parameters(self): return []
            def train(self): pass
        class Timing:
            def __init__(self, *a): pass
            def finish_train(self, *a, **kw): pass
            def checkpoint(self, callback, device): return callback()
            def validation(self, callback, device): return callback()
            def fields(self, *a): return {}
        def learning_rate(step, total, *a):
            lr_calls.append((step, total))
            return 1e-4
        def update(*a, **kw): updates.append(kw); return {}, 0
        def evaluate(*a, **kw): validations.append(kw); return {'ndcg@10': .1}
        noop = lambda *a, **kw: None
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=noop),
            device=lambda name: SimpleNamespace(type=name), manual_seed=noop, set_num_threads=noop,
            backends=SimpleNamespace(cudnn=SimpleNamespace()),
            optim=SimpleNamespace(AdamW=lambda *a, **kw: SimpleNamespace(param_groups=[{}])),
            amp=SimpleNamespace(GradScaler=lambda *a, **kw: None))
        ns = dict(time=time, datetime=datetime, timezone=timezone, math=__import__('math'),
            Path=Path, json=json, print=noop, torch=torch, random=SimpleNamespace(seed=noop),
            np=SimpleNamespace(random=SimpleNamespace(seed=noop)), TrainingSchedule=Schedule,
            make_model=lambda *a: Model(), make_config=lambda *a, **kw: {},
            integration_contract=lambda *a: {}, write_json=noop, EpochTiming=Timing,
            checkpoint=noop, publish_selected=lambda *a: 'fixture', sha256=lambda *a: 'fixture',
            shared=SimpleNamespace(learning_rate=learning_rate, _atomic_copy=noop),
            update=update, evaluate=evaluate, EvaluationStopped=RuntimeError)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(**t.FIXED, output=Path(tmp) / 'model', stop_file=None, deadline_utc=None)
            result = ns['run'](args, object())
        self.assertEqual(lr_calls, [(0, 136175), (1, 136175)])
        self.assertEqual(len(updates), 2)
        self.assertEqual(len(validations), 2)
        self.assertEqual(result['reason'], 'runtime_stop_after_validation')
        self.assertEqual(result['state']['epoch'], 2)
        self.assertEqual(result['predictions_status'], 'deferred-runtime-stop')

    def test_horizon_and_recipe_rejections(self):
        t.validate_config(config())
        for where, key, bad in [('training', 'schedule_total_epochs', 2),
            ('training_schedule', 'eligible_count', 12), ('architecture', 'context_length', 32)]:
            c = config()
            c['contract'][where][key] = bad
            with self.assertRaises(ValueError): t.validate_config(c)

    def test_original_loop_seams_and_restoration(self):
        expected = config()
        events = []
        provider = SimpleNamespace(metadata={'context': 64}, splits={
            'train': {'y': Census(3477437)}, 'val': {'y': Census(158760)}})
        class Sampler:
            CONTRACT = 'test-original-sampler'
            class InstalledSampler:
                def __init__(self, train):
                    self.replacement = SimpleNamespace(audit={'calls': 2})
                def install(self): events.append('install')
                def restore(self): events.append('restore')
        train = SimpleNamespace(integration_contract=lambda *a: None,
            make_config=lambda *a, **kw: copy.deepcopy(expected),
            update=lambda *a: events.append('update'), evaluate=lambda *a: events.append('evaluate'),
            checkpoint=lambda *a: None, sha256=lambda *a: 'hash')
        originals = vars(train).copy()
        def run(args, p):
            self.assertEqual(args.schedule_total_epochs, 5)
            self.assertEqual(args.stop_after_validation, 2)
            self.assertFalse(args.resume)
            train.make_config(p, None, args, None)
            train.update()
            train.evaluate()
            return dict(status='stopped', reason='runtime_stop_after_validation', state=dict(
                epoch=2, offset=0, group_index=0, updates=54470, examples=6954874))
        train.run = run
        result = t.run_bound(train, Sampler, provider, SimpleNamespace(**t.FIXED), expected,
                             lambda: events.append('check'))
        self.assertEqual(result['sampler_audit'], {'calls': 2})
        self.assertEqual(events[-1], 'restore')
        self.assertGreater(events.count('check'), 4)
        for name, original in originals.items(): self.assertIs(getattr(train, name), original)

    def test_cancelled_before_numerics_and_restores(self):
        class Sampler:
            class InstalledSampler:
                def __init__(self, train): pass
                def install(self): pass
                def restore(self): pass
        names = ('integration_contract', 'make_config', 'update', 'evaluate', 'checkpoint', 'sha256')
        original = lambda *a: None
        train = SimpleNamespace(**{k: original for k in names})
        def cancel(): raise InterruptedError('test cancellation')
        with self.assertRaises(InterruptedError):
            t.run_bound(train, Sampler, None, None, None, cancel)
        for name in names: self.assertIs(getattr(train, name), original)

    def test_default_cli_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'request.json'
            path.write_text('{}', encoding='utf-8')
            with patch.object(t, 'build_plan', return_value={'training_executed': False}) as plan, \
                    patch.object(t, 'execute', side_effect=AssertionError('numerical execution')):
                t.main(['--request', str(path)])
                plan.assert_called_once_with({})

    def test_guard_stop_deadline_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = dict(output=tmp, max_seconds=60, min_free_disk_gib=1, max_rss_gib=12)
            with patch.object(t, 'rss_bytes', return_value=1), \
                    patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=2**40)):
                guard = t.Guard(r)
                guard()
                guard.started -= 61
                with self.assertRaises(TimeoutError): guard()
                guard = t.Guard(r)
                (Path(tmp) / 'stop-request.json').write_text('{}', encoding='utf-8')
                with self.assertRaises(InterruptedError): guard()

    def test_plan_bounded_read_only_preserves_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / 'config.json'
            cfg.write_text(json.dumps(config()), encoding='utf-8')
            scope = root / 'scope.json'
            scope.write_text('{}', encoding='utf-8')
            payload = root / 'never-open.bin'
            payload.write_bytes(b'payload')
            manifest = root / 'manifest.json'
            manifest.write_text('{}', encoding='utf-8')
            r = dict(contract=t.CONTRACT, data='D:/logical/data', feature_dir='D:/logical/features',
                scope_decision='D:/logical/scope.json', manifest_sha256='a'*64,
                bindings=[dict(logical='D:/logical/scope.json', physical=str(scope), sha256=t.SCOPE_SHA, role='metadata'),
                    dict(logical='D:/logical/features/manifest.json', physical=str(manifest), sha256='a'*64, role='metadata'),
                    dict(logical='D:/logical/array.npy', physical=str(payload), sha256='b'*64, role='payload')],
                expected_config=str(cfg), expected_config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest(),
                output=str(root / 'fresh'), max_seconds=60, min_free_disk_gib=1, max_rss_gib=12)
            before = copy.deepcopy(r)
            original_open = Path.open
            def limited_open(path, *a, **kw):
                if path == payload: raise AssertionError('plan opened payload')
                return original_open(path, *a, **kw)
            with patch.object(Path, 'open', limited_open): result = t.build_plan(r)
            self.assertEqual(result['schedule_denominator'], 136175)
            self.assertEqual(result['stop_updates'], 54470)
            self.assertEqual(r, before)
            self.assertFalse((root / 'fresh').exists())
            self.assertTrue(any(row['logical'] == 'D:/logical/array.npy' for row in result['bindings']))


if __name__ == '__main__': unittest.main()
