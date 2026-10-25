"""Small read-only fixtures; no historical payloads, inference or GPU."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace, MethodType
import unittest

from gpt_provider import BoundIO, FileBinding, canonical_hash
from head_bank_hooks import coverage, prepared_query, HookHandle, EXPECTED_ROWS


class FakeIO:
    def __init__(self, receipt):
        self.receipt, self.verifies = receipt, []

    def read_json(self, logical, *, expected):
        assert logical == 'receipt' and expected == 'receipt-pin'
        return self.receipt

    def verify(self, logical, role, expected):
        self.verifies.append((logical, role, expected))
        return Path('relocated/store.sqlite')


class Tests(unittest.TestCase):
    def test_full_coverage_logical_identity_and_pin(self):
        options = dict(enabled=True, purpose='full-E1', allow_missing_fallback=False,
                       coverage_receipt={'path': 'receipt', 'sha256': 'receipt-pin'},
                       store_path='E:/old/store.sqlite', store_source_pins={'logical-source': 'pin'})
        receipt = dict(state='completed', mode='full', split_rows=EXPECTED_ROWS,
                       store_path=options['store_path'], store_sha256='store-pin',
                       store_source_pins_sha256=canonical_hash(options['store_source_pins']))
        io = FakeIO(receipt)
        self.assertEqual(coverage(io, options), Path('relocated/store.sqlite'))
        self.assertEqual(io.verifies, [('E:/old/store.sqlite', 'payload', 'store-pin')])
        for key, value in [('state', 'partial'), ('mode', 'pilot'), ('split_rows', {'train': 1}),
                           ('store_path', 'relocated/store.sqlite'), ('store_source_pins_sha256', 'bad'),
                           ('store_sha256', '')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                coverage(FakeIO(dict(receipt, **{key: value})), options)
        with self.assertRaises(ValueError):
            coverage(io, dict(options, allow_missing_fallback=True))

    def test_prepared_ast_extracts_only_query_with_original_globals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.py'
            source = b'raise RuntimeError("never run factory/module")\nclass CompleteProvider:\n    def _query(self, split, row, *, prepared_profile=None):\n        return marker, prepared_profile\n'
            path.write_bytes(source)
            pin = {'path': 'D:/retained/source.py', 'sha256': hashlib.sha256(source).hexdigest()}
            io = BoundIO([FileBinding(pin['path'], str(path), pin['sha256'], 'source')], lambda: None)
            method = prepared_query(io, pin, SimpleNamespace(marker='original'))
            self.assertEqual(method(None, 'train', 0, prepared_profile=123), ('original', 123))
            path.write_bytes(source + b'# tamper')
            with self.assertRaises(ValueError):
                prepared_query(io, pin, SimpleNamespace(marker='original'))

    def test_handle_restores_query_closes_reader_but_not_provider(self):
        events = []
        provider = SimpleNamespace(close=lambda: events.append('provider'))
        handle = HookHandle(provider)
        handle.query = MethodType(lambda self: None, provider)
        provider._query = handle.query
        handle.reader = SimpleNamespace(stats=lambda: {'rows': 1}, close=lambda: events.append('reader'))
        handle.performance = SimpleNamespace(evidence=lambda: {}, uninstall=lambda: events.append('performance'))
        handle.dynamic = SimpleNamespace(stats=lambda: {}, uninstall=lambda: events.append('dynamic'))
        with handle:
            self.assertEqual(handle.evidence()['reader'], {'rows': 1})
        handle.uninstall()
        self.assertEqual(events, ['dynamic', 'performance', 'reader'])
        self.assertNotIn('_query', vars(provider))
        self.assertFalse(handle.evidence()['allow_missing_fallback'])

    def test_handle_rejects_query_owner_change(self):
        provider = SimpleNamespace()
        handle = HookHandle(provider)
        handle.query = object()
        provider._query = object()
        with self.assertRaisesRegex(BaseExceptionGroup, 'teardown failed'):
            handle.uninstall()

    def test_teardown_attempts_every_owner_after_evidence_and_dynamic_failure(self):
        events = []
        provider = SimpleNamespace()
        handle = HookHandle(provider)
        handle.query = object()
        provider._query = handle.query
        def fail_evidence():
            raise ValueError('evidence failed')
        def fail_dynamic():
            events.append('dynamic')
            raise RuntimeError('dynamic failed')
        handle.reader = SimpleNamespace(stats=fail_evidence, close=lambda: events.append('reader'))
        handle.dynamic = SimpleNamespace(uninstall=fail_dynamic)
        handle.performance = SimpleNamespace(uninstall=lambda: events.append('performance'))
        with self.assertRaises(BaseExceptionGroup) as caught:
            handle.uninstall()
        self.assertEqual([str(e) for e in caught.exception.exceptions],
                         ['evidence failed', 'dynamic failed'])
        self.assertEqual(events, ['dynamic', 'performance', 'reader'])
        self.assertNotIn('_query', vars(provider))
        self.assertFalse(handle.closed)

    def test_teardown_preserves_foreign_query_but_closes_owned_reader(self):
        events = []
        replacement = object()
        provider = SimpleNamespace(_query=replacement)
        handle = HookHandle(provider)
        handle.query = object()
        handle.reader = SimpleNamespace(stats=lambda: {}, close=lambda: events.append('reader'))
        handle.performance = SimpleNamespace(evidence=lambda: {}, uninstall=lambda: events.append('performance'))
        with self.assertRaises(BaseExceptionGroup) as caught:
            handle.uninstall()
        self.assertEqual(len(caught.exception.exceptions), 1)
        self.assertIs(provider._query, replacement)
        self.assertEqual(events, ['performance', 'reader'])
        self.assertFalse(handle.closed)

    def test_retained_reader_missing_never_prepares(self):
        path = Path(__file__).resolve().parents[1] / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/precompute-e1-20261022-r1/preparation/store_hook.py'
        spec = importlib.util.spec_from_file_location('_test_store_hook', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        provider = SimpleNamespace(splits={'train': {'user_id': [1], 'lengths': [1],
                                                     'x': [SimpleNamespace(tobytes=lambda: b'x')]}},
                                   _origin=lambda split, row: 2, _profiles={}, personal=object())
        counts = {}
        def forbidden(*args, **kwargs):
            raise AssertionError('Fallback forbidden')
        query = module.reader_query(forbidden, forbidden,
                                    SimpleNamespace(get=lambda *a, **k: None), object,
                                    allow_missing=False, counters=counts)
        with self.assertRaises(KeyError):
            query(provider, 'train', 0)
        self.assertEqual(counts['store_misses'], 1)
        self.assertEqual(counts['fallback_prepares'], 0)

    def test_retained_reader_tail_mismatch_evicts_inserted_query(self):
        path = Path(__file__).resolve().parents[1] / 'source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/precompute-e1-20261022-r1/preparation/store_hook.py'
        spec = importlib.util.spec_from_file_location('_test_store_tail', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        provider = SimpleNamespace(splits={'train': {'user_id': [1], 'lengths': [1],
                                                     'x': [SimpleNamespace(tobytes=lambda: b'x')]}},
                                   _origin=lambda split, row: 2, _profiles={}, personal=object())
        key = module.query_key(provider, 'train', 0)
        def prepared(self, split, row, *, prepared_profile):
            self._profiles[key] = 'just inserted'
            return {'user': SimpleNamespace(dtype='f4', shape=(1,), tobytes=lambda: b'bad')}
        tail = SimpleNamespace(dtype='f4', shape=(1,), tobytes=lambda: b'good')
        counts = {}
        query = module.reader_query(None, prepared,
                                    SimpleNamespace(get=lambda *a, **k: (object(), tail)), object,
                                    allow_missing=False, counters=counts)
        with self.assertRaisesRegex(ValueError, 'tail differs'):
            query(provider, 'train', 0)
        self.assertNotIn(key, provider._profiles)
        self.assertEqual(counts['tail_checks'], 0)


if __name__ == '__main__':
    unittest.main()
