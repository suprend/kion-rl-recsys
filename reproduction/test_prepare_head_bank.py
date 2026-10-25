"""Stdlib-only adapter and original tiny selection fixtures; no model calls."""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_head_bank as bank


def original_functions(relative, digest, names):
    data = (bank.ROOT / 'source' / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise AssertionError('Original source changed')
    tree = ast.parse(data)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError('Missing pure source helpers')
    context = {'Counter': Counter, 'hashlib': hashlib, 'random': random, 'ROWS': 262144, 'SEED': 20261024}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<verified-pure-functions>', 'exec'), context)
    return context


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_default_no_backend_payload_or_mutation(self):
        before = set(sys.modules)
        with patch('subprocess.Popen', side_effect=AssertionError('No process')):
            plan = bank.build_plan({}, self.root / 'fresh')
        self.assertFalse(plan['ready'])
        self.assertFalse(plan['preparation_executed'])
        self.assertEqual(plan['selection']['slots'], 32)
        self.assertEqual(plan['selection']['new_rows'], 229376)
        self.assertFalse((self.root / 'fresh').exists())
        self.assertFalse({'torch', 'numpy'} & (set(sys.modules) - before))
        self.assertTrue(all(r['status'] == 'blocked_input:unbound' for r in plan['requirements']))

    def test_payload_bound_stat_only_and_identity_unchanged(self):
        request = bank.metadata(bank.REQUEST, bank.REQUEST_SHA)
        original = copy.deepcopy(request)
        teacher = request['teacher']['path']
        payload = self.root / 'opaque.pt'
        payload.write_bytes(b'not a model')
        real_open = Path.open
        def guarded(path, *args, **kwargs):
            if path == payload:
                raise AssertionError('Payload opened')
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            plan = bank.build_plan({teacher: payload.as_posix()}, self.root / 'fresh')
        row = next(r for r in plan['requirements'] if r['logical_path'] == teacher)
        self.assertEqual(row['status'], 'stat-only')
        self.assertFalse(row['payload_verified'])
        self.assertEqual(plan['historical_request'], original)

    def test_exact_request_pin(self):
        path = self.root / 'request.json'
        request = bank.metadata(bank.REQUEST)
        request['seed'] = 1
        path.write_text(json.dumps(request), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pin mismatch'):
            bank.build_plan({}, self.root / 'new', path)

    def test_binding_alias_unknown_and_traversal_rejected(self):
        a, b = 'D:/input/a', 'D:/input/b'
        with self.assertRaises(ValueError):
            bank.bindings_map({a: str(self.root), b: str(self.root)}, {a, b})
        with self.assertRaises(ValueError):
            bank.bindings_map({a: str(self.root)}, {b})
        with self.assertRaises(ValueError):
            bank.logical_path('D:/a/../b')
        with self.assertRaises(ValueError):
            bank.bindings_map({a: 'relative'}, {a})

    def test_metadata_duplicates_and_bound(self):
        path = self.root / 'm.json'
        path.write_text('{"a":1,"a":2}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            bank.metadata(path)
        with self.assertRaisesRegex(ValueError, 'bounded'):
            bank.metadata(path, maximum=2)

    def test_outputs_and_execution_fail_closed(self):
        with self.assertRaises(ValueError):
            bank.build_plan({}, self.root)
        with self.assertRaises(ValueError):
            bank.build_plan({}, bank.ROOT / 'source' / 'new')
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            bank.execute({'ready': True}, execute_preparation=True)

    def test_missing_graphs_and_arrays_are_explicit_not_built(self):
        request = bank.metadata(bank.REQUEST)
        root = self.root / 'old'
        root.mkdir()
        manifest = root / 'bank.json'
        manifest.write_text('{}', encoding='utf-8')
        graph = self.root / 'graphs'
        graph.mkdir()
        old = {'teacher': request['teacher'], 'train_rows': 32768, 'state': 'completed',
               'arrays': {name: {'path': name + '.npy', 'sha256': 'a' * 64, 'bytes': 128}
                          for name in bank.ARRAYS},
               'date_counts': {'737000': 3477437}, 'candidate_contract': {'cache_key': 'b' * 64}}
        bindings = {request['reuse_bank_dir']: root, request['reuse_bank_dir'] + '/bank.json': manifest,
                    request['graph_cache']: graph}
        with patch.object(bank, 'metadata', return_value=old):
            entries, blockers = bank.reuse_inventory(request, bindings)
        self.assertEqual(len(entries), 15)
        self.assertTrue(all(e['status'] == 'blocked_input:missing' for e in entries))
        self.assertTrue(blockers)
        self.assertEqual(list(graph.iterdir()), [])

    def test_symlink_binding_rejected(self):
        link = self.root / 'link'
        target = self.root / 'target'
        target.mkdir()
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            self.skipTest(str(error))
        with self.assertRaises(ValueError):
            bank.bindings_map({'D:/a': str(link)}, {'D:/a'})

    def test_original_nested_quota_and_retention(self):
        funcs = original_functions(bank.SOURCE, bank.SOURCE_SHA, ['require', 'proportional_quotas', 'select_rows'])
        targets = [3] * 12
        dates = [10] * 4 + [20] * 8
        rows, counts, quotas = funcs['select_rows'](targets, dates, 10, maximum=6, reuse_rows=[1, 5])
        self.assertTrue({1, 5}.issubset(rows))
        self.assertEqual(quotas, {10: 2, 20: 4})
        self.assertEqual(rows, sorted(rows, key=lambda i: (dates[i], i)))
        self.assertEqual(len(set(rows)), 6)
        self.assertEqual(funcs['select_rows'](targets, dates, 10, maximum=6, reuse_rows=[1, 5])[0], rows)
        with self.assertRaises(ValueError):
            funcs['select_rows'](targets, dates, 10, maximum=6, reuse_rows=[0, 1, 2])

    def test_original_sampler_positive_first_same_origin_exclusion(self):
        source = 'fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/full-joint-e1-20261024-r1/candidates.py'
        funcs = original_functions(source, '6923f4e63ce6278de3297449c8d59fa498b9117f2cdc85537d7e9bd4e0acc09d', ['require', 'draw'])
        pool = list(range(3, 1027))
        chosen = funcs['draw'](pool, 2000, [3, 4, 5], 20261024, 123)
        self.assertEqual(chosen[0], 2000)  # Natural miss is retained.
        self.assertEqual(len(set(chosen)), 32)
        self.assertFalse(set(chosen) & {3, 4, 5})
        self.assertEqual(chosen, funcs['draw'](pool, 2000, [3, 4, 5], 20261024, 123))
        self.assertNotEqual(chosen, funcs['draw'](pool, 2000, [3, 4, 5], 20261024, 124))


if __name__ == '__main__':
    unittest.main()
