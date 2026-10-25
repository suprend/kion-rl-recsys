"""Tiny/source-only tests; never activate a provider or production arrays."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import execute_head_bank as app
from gpt_provider import BoundIO, FileBinding


class BankExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def spec(self):
        return dict(mode=app.MODE, bindings=[], output=str(self.root / 'out'),
                    reuse_root=str(self.root / 'old'), graph_root=str(self.root / 'graphs'),
                    max_seconds=60, manifest_sha256=app.MANIFEST_SHA)

    def test_plan_no_numerical_import_or_payload_read(self):
        original = __import__
        def guarded(name, *a, **kw):
            if name.split('.')[0] in ('numpy', 'torch', 'psutil'):
                raise AssertionError('numerical import in planner')
            return original(name, *a, **kw)
        spec = self.spec()
        before = copy.deepcopy(spec)
        with patch('builtins.__import__', guarded), patch.object(BoundIO, 'verify', side_effect=AssertionError):
            plan = app.build_plan(spec)
        self.assertEqual(spec, before)
        self.assertFalse(plan['ready'])
        self.assertTrue(plan['blocked_input'])
        self.assertEqual((plan['rows'], plan['retained_rows'], plan['new_rows']), (262144, 32768, 229376))
        self.assertFalse(Path(spec['output']).exists())

    def test_gate_precedes_plan(self):
        with patch.object(app, 'build_plan', side_effect=AssertionError):
            with self.assertRaisesRegex(ValueError, 'explicit'):
                app.execute({})

    def test_unknown_lineage_and_budget_rejected(self):
        for field, value in [('mode', 'fresh-new-E2'), ('manifest_sha256', '0' * 64),
                             ('max_seconds', float('nan')), ('max_seconds', True)]:
            spec = self.spec()
            spec[field] = value
            with self.assertRaises(ValueError):
                app.build_plan(spec)

    def test_existing_or_overlapping_output_rejected(self):
        spec = self.spec()
        Path(spec['output']).mkdir()
        with self.assertRaises(ValueError):
            app.build_plan(spec)
        spec['output'] = str(self.root / 'old' / 'child')
        with self.assertRaises(ValueError):
            app.build_plan(spec)

    def test_changed_historical_pin_rejected(self):
        spec = self.spec()
        spec['bindings'] = [dict(logical=app.BANK, physical=str(self.root / 'bank.py'), sha256='0' * 64, role='source')]
        with self.assertRaisesRegex(ValueError, 'historical pin'):
            app.build_plan(spec)

    def test_original_selection_and_draw_source(self):
        source = app.ROOT / 'source' / app.plan_source.SOURCE
        calls = []
        io = BoundIO([FileBinding(app.BANK, source.as_posix(), app.plan_source.SOURCE_SHA, 'source')],
                     lambda: calls.append(True))
        bank = app.source_module(io, app.BANK, app.plan_source.SOURCE_SHA)
        rows, counts, quotas = bank.select_rows([3] * 20, [1] * 10 + [2] * 10, 4,
                                               maximum=8, reuse_rows=[2, 12])
        self.assertEqual(len(rows), 8)
        self.assertTrue({2, 12} <= set(rows))
        self.assertEqual(quotas, {1: 4, 2: 4})
        self.assertEqual(rows, sorted(rows))
        self.assertTrue(calls)

    def test_run_calls_original_body_with_real_check(self):
        request = app.historical_request()
        before = copy.deepcopy(request)
        spec = self.spec()
        checks = []
        def guard(**fields):
            checks.append(fields)
        seen = {}
        def run(req, check):
            seen.update(req=req, check=check)
            check(stage='spy')
            return {'state': 'fake-only'}
        bank = SimpleNamespace(close_context=lambda ctx: {}, open_reuse=lambda *a: None,
                               write_new=lambda *a: None, run=run)
        result = app.run_bound(bank, request, spec, None, guard)
        self.assertEqual(request, before)
        self.assertEqual(result['state'], 'fake-only')
        self.assertIs(seen['check'], guard)
        self.assertEqual(seen['req']['teacher'], before['teacher'])
        self.assertEqual(seen['req']['source_hashes'], before['source_hashes'])
        self.assertEqual(seen['req']['bank_output'], (Path(spec['output']) / 'train').as_posix())
        self.assertEqual(checks, [{'stage': 'spy'}])

    def test_guard_cancel_deadline_and_stability(self):
        spec = self.spec()
        output = Path(spec['output'])
        output.mkdir()
        guard = app.Guard(spec)
        guard.until = 0
        with self.assertRaises(TimeoutError):
            guard()
        guard = app.Guard(spec)
        (output / 'STOP').write_text('', encoding='utf-8')
        with self.assertRaises(InterruptedError):
            guard()

    def test_digest_is_real_and_cancellable(self):
        path = self.root / 'tiny'
        path.write_bytes(b'abc')
        calls = []
        self.assertEqual(app.checked_digest(path, lambda **kw: calls.append(kw)), hashlib.sha256(b'abc').hexdigest())
        self.assertTrue(calls)
        def cancel(**kw):
            raise InterruptedError()
        with self.assertRaises(InterruptedError):
            app.checked_digest(path, cancel)

    def test_missing_graph_fails_before_rebuild(self):
        tiny = self.root / 'tiny'
        tiny.write_bytes(b'x')
        io = BoundIO([FileBinding('D:/tiny', str(tiny), hashlib.sha256(b'x').hexdigest(), 'payload')], lambda: None)
        cls = app.graph_class(SimpleNamespace(LiveCandidates=object), io, 'E:/graph', self.root)
        obj = cls()
        obj.cache_key = 'a' * 64
        with self.assertRaises(KeyError):
            obj._paths(123)
        with self.assertRaisesRegex(RuntimeError, 'reconstruction forbidden'):
            obj._build()

    def test_builder_profiles_source_autobinding_and_missing_metadata(self):
        with patch.object(BoundIO, 'verify', side_effect=AssertionError('payload hash forbidden')):
            report = app.assemble_spec({}, input_root=self.root / 'bundle', output=self.root / 'out',
                                       reuse_root=self.root / 'old', graph_root=self.root / 'graphs', max_seconds=60)
        self.assertFalse(report['metadata_complete'])
        self.assertFalse(report['payloads_hashed'])
        self.assertEqual(len(report['metadata_prerequisites']), 3)
        rows = {r['logical']: r for r in report['inventory']}
        self.assertTrue(rows[app.BANK]['exists'])
        self.assertEqual(rows[app.BANK]['sha256'], app.plan_source.SOURCE_SHA)
        self.assertTrue(rows[app.BANK]['physical'].startswith(app.ROOT.as_posix()))
        missing = next(r for r in report['inventory'] if r['role'] == 'payload' and not r['exists'])
        self.assertIn('/bundle/inputs/', missing['physical'])
        self.assertEqual(app.build_plan(report['spec'])['execution_mode'], app.MODE)

    def test_builder_rejects_unknown_override(self):
        with self.assertRaisesRegex(ValueError, 'undeclared explicit'):
            app.assemble_spec({'D:/unknown': str(self.root / 'unknown')}, input_root=self.root,
                              output=self.root / 'out', reuse_root=self.root / 'old',
                              graph_root=self.root / 'graphs', max_seconds=60)

    def test_builder_cli_writes_only_fresh_spec(self):
        target = self.root / 'spec.json'
        argv = ['--build-spec', '--input-root', str(self.root / 'bundle'), '--output-dir', str(self.root / 'run'),
                '--reuse-root', str(self.root / 'old'), '--graph-root', str(self.root / 'graphs'),
                '--max-seconds', '60', '--spec-output', str(target)]
        with patch('builtins.print'):
            self.assertEqual(app.main(argv), 0)
        spec = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(spec['mode'], app.MODE)
        self.assertGreater(len(spec['bindings']), 100)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            app.main(argv)
        with self.assertRaisesRegex(ValueError, 'never executes'):
            app.main(argv + ['--execute-bank'])

    def test_builder_expands_named_metadata_without_payload_reads(self):
        request = app.historical_request()
        oldroot, graphroot = self.root / 'old', self.root / 'graphs'
        oldroot.mkdir()
        graphroot.mkdir()
        def write(path, value):
            data = json.dumps(value).encode()
            path.write_bytes(data)
            return hashlib.sha256(data).hexdigest()
        signature = dict(context='tiny-context', adapter_sha256=request['source_hashes'][request['candidate_source_path']])
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        old = dict(state='completed', train_rows=32768, teacher=request['teacher'], context_signature='tiny-context',
                   candidate_contract=dict(cache_key=key), date_counts={'123': 300000},
                   arrays={name: dict(path=name + '.npy', sha256='a' * 64) for name in app.plan_source.ARRAYS})
        oldpin = write(oldroot / 'bank.json', old)
        request['reuse_bank_manifest_sha256'] = oldpin
        request['input_hashes'][request['reuse_bank_dir'] + '/bank.json'] = oldpin
        graph = dict(state='completed', signature=signature, cache_key=key, origin=123, cutoff=123,
                     arrays={name: dict(path=key + '-123-' + name + '.npy', sha256='b' * 64)
                             for name in ('neighbors', 'weights', 'global_counts', 'recent_counts')})
        write(graphroot / (key + '-123.json'), graph)
        from gpt_provider import canonical_hash
        options = request['provider_hook']['options']
        coverage = dict(state='completed', mode='full', split_rows={'train':3477437, 'val':158760},
                        store_path=options['store_path'], store_source_pins_sha256=canonical_hash(options['store_source_pins']),
                        store_sha256='c' * 64)
        coverage_path, sidepath = self.root / 'coverage.json', self.root / 'latest.json'
        options['coverage_receipt']['sha256'] = write(coverage_path, coverage)
        write(sidepath, dict(sha256=request['teacher']['sha256']))
        bindings = {options['coverage_receipt']['path']: str(coverage_path),
                    request['teacher']['path'].rsplit('.', 1)[0] + '.json': str(sidepath)}
        with patch.object(app, 'historical_request', return_value=request), patch.object(app, 'PROFILE_PINS', {}), \
                patch.object(BoundIO, 'verify', side_effect=AssertionError('payload read')):
            report = app.assemble_spec(bindings, input_root=self.root / 'bundle', output=self.root / 'run',
                                       reuse_root=oldroot, graph_root=graphroot, max_seconds=60)
        self.assertTrue(report['metadata_complete'])
        rows = {r['logical']: r for r in report['inventory']}
        self.assertEqual(rows[options['store_path']]['sha256'], 'c' * 64)
        self.assertEqual(rows[request['graph_cache'] + '/' + key + '-123-neighbors.npy']['sha256'], 'b' * 64)
        self.assertEqual(rows[request['reuse_bank_dir'] + '/item_table.npy']['sha256'], 'a' * 64)


if __name__ == '__main__':
    unittest.main()
