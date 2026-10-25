"""Stdlib-only source/contract tests. No real arrays, models or provider calls."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_final_rollout_tests', HERE/'rollout.py')
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class PrefixTests(unittest.TestCase):
    def test_original_peers_size_slot_and_max_trim(self):
        peers = [[1, 3, 4], [1], [1, 2, 9, 11]]
        padded, lengths = live.producer_prefix_batch(peers, [1, 7], 3, 1)
        self.assertEqual(padded, [[1, 3, 4, 0], [1, 7, 0, 0], [1, 2, 9, 11]])
        self.assertEqual(lengths, [3, 2, 4])
        grown, lengths = live.producer_prefix_batch(peers, [1, 7, 8, 9, 10], 3, 1)
        self.assertEqual(len(grown[0]), 5)
        self.assertEqual(lengths, [3, 5, 4])
        self.assertEqual(peers[1], [1])

    def test_partial_producer_nonzero_slot(self):
        peers = [[1, i+3] for i in range(9)]
        padded, lengths = live.producer_prefix_batch(peers, [1, 99], 9, 8)
        self.assertEqual(padded[8], [1, 99])
        self.assertEqual(padded[:8], peers[:8])
        self.assertEqual(lengths, [2]*9)

    def test_invalid_producer_contracts(self):
        for size, slot in ((0, 0), (17, 0), (1, 1), (1, -1), (True, 0), (2, 0)):
            with self.subTest(size=size, slot=slot), self.assertRaises(ValueError):
                live.producer_prefix_batch([[1, 3]], [1, 3], size, slot)

    def test_observed_padding_and_h63_append(self):
        self.assertEqual(live.observed_prefix([1, 2, 7, 0], 3), [1, 2, 7])
        initial = [1, *range(3, 66)]
        self.assertEqual(live.append_generated(initial, 90), [1, *range(4, 66), 90])
        for tokens, length in (([1, 3, 0], 1), ([1, 0, 3], 3), ([3], 1), ([1], 0)):
            with self.assertRaises(ValueError):
                live.observed_prefix(tokens, length)
        for token in (0, 1, 2):
            with self.assertRaises(ValueError):
                live.append_generated([1, 3], token)


class DistributionTests(unittest.TestCase):
    def test_mask_and_stable_exact_kl(self):
        p, lp = live.distribution([10000, -10000, 9999], excluded={1})
        _, lq = live.distribution([10000, -10000, 9999], excluded={1})
        metrics = live.policy_metrics(p, lp, lq, [.2, math.nan, .9], [True, False, True], [0, 0, 0],
            [-.1, math.nan, .6], [.3, math.nan, .3], [0, 0, 0])
        self.assertEqual(metrics['kl_nats'], 0)
        self.assertEqual(p[1], 0)
        self.assertIsNone(lp[1])
        self.assertAlmostEqual(metrics['supported_mass'], 1)

    def test_unsupported_is_neutral_with_null_conditional(self):
        p, lp = live.distribution([0, 0])
        metrics = live.policy_metrics(p, lp, lp, [math.nan]*2, [False]*2, [0]*2,
                                     [math.nan]*2, [math.nan]*2, [0]*2)
        self.assertEqual(metrics['unsupported_mass'], 1)
        self.assertIsNone(metrics['expected_supported_reward_conditional'])
        self.assertIsNone(metrics['expected_supported_residual_conditional'])
        self.assertEqual(metrics['expected_neutral_residual_proxy'], 0)

    def test_seed_and_raw_id_tie_break(self):
        self.assertEqual(live.select_index([.5, .5], 'greedy', random.Random(1), [9, 3]), 1)
        def picks(seed):
            rng = random.Random(seed)
            return [live.select_index([.2, 0, .8], 'categorical', rng) for _ in range(25)]
        self.assertEqual(picks(3), picks(3))
        self.assertNotEqual(picks(3), picks(4))
        self.assertNotIn(1, picks(3))

    def test_invalid_distribution_and_rollout_config(self):
        for kwargs in ({'logits': []}, {'logits': [math.inf]}, {'logits': [0], 'temperature': 0},
                       {'logits': [0], 'excluded': {0}}, {'logits': [0], 'excluded': {2}}):
            with self.assertRaises(ValueError):
                live.distribution(**kwargs)
        for kwargs in ({'steps': 6}, {'steps': 0}, {'repeat_mask': 'seen'}, {'selection': 'target'},
                       {'temperature': 0}, {'gamma': 2}, {'seed': True}):
            with self.assertRaises(ValueError):
                live.rollout_state({}, None, None, {}, [], [], **kwargs)


class SourceTests(unittest.TestCase):
    def test_provider_teardown_runs_after_cache_close_failure(self):
        calls = []
        def fail_close(arrays):
            calls.append('cache')
            raise BufferError('synthetic mmap close error')
        def close_context(ctx):
            calls.append('provider')
            return dict(teardown_completed=True, errors=[])
        with self.assertRaises(BufferError):
            live.close_resources(SimpleNamespace(close_cache=fail_close), {},
                                 SimpleNamespace(close_context=close_context), {})
        self.assertEqual(calls, ['cache','provider'])

    def test_every_module_descriptor_requires_matching_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = (Path(temporary)/'source.py').resolve()
            source.write_text('# stdlib source fixture\n', encoding='utf-8')
            pin = dict(path=source.as_posix(),sha256=live.sha(source))
            request = dict(residual_module=pin,source_hashes={source.as_posix():pin['sha256']})
            self.assertEqual(live.verify_module_pin(request, 'residual_module'),source.resolve())
            request['source_hashes'] = {}
            with self.assertRaises(ValueError):
                live.verify_module_pin(request,'residual_module')

    def test_import_is_stdlib_and_no_target_or_training_reads(self):
        tree = ast.parse((HERE/'rollout.py').read_text(encoding='utf-8'))
        imports = {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertLessEqual(imports, {'__future__', 'math', 'random', 'hashlib', 'importlib', 'json', 'pathlib', 'sys'})
        keys = {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)}
        self.assertFalse(keys & {'y', 'target_ids', 'targets', 'target_item_id', 'watched_pct'})
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {'backward', 'step', 'candidate_features', 'batch'})

    def test_full_projection_inside_every_step_and_old_parity_tolerances(self):
        source = (HERE/'rollout.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'step']
        self.assertEqual(len(loops), 1)
        retrieval = [node for node in ast.walk(loops[0]) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'retrieval_logits']
        self.assertEqual(len(retrieval), 1)
        self.assertEqual([arg.id for arg in retrieval[0].args], ['q_group', 'table'])
        allclose = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'allclose']
        self.assertEqual(len(allclose), 1)
        self.assertEqual({k.arg: k.value.value for k in allclose[0].keywords}, {'atol': .003, 'rtol': .001})
        self.assertIn("if 'reference_scores' not in state:", source)
        self.assertIn('bound = .006 + .001*', source)

    def test_labelblind_request_is_fixed_before_native_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            selection = Path(temporary)/'selection.json'
            selection.write_text(json.dumps(dict(positions=list(live.POSITIONS), seed=live.SEED,
                original_val_rows=list(live.ORIGINAL_ROWS),
                before_RL_training=True, steps=5, selection_modes=list(live.MODES), repeat_masks=list(live.MASKS))), encoding='utf-8')
            request = dict(rollout_positions=list(live.POSITIONS), seed=live.SEED,
                scientific_checkpoints={arm: {} for arm in live.ARMS[1:]},
                preregistration={'path': selection.as_posix(), 'sha256': live.sha(selection)},
                rollout_selection_pin={'path': selection.as_posix(), 'sha256': live.sha(selection)},
                rollout_output=(HERE/'test-unused-output.json').as_posix())
            self.assertEqual(live.validate_request(request).name, 'test-unused-output.json')
            request['rollout_positions'] = list(reversed(live.POSITIONS))
            with self.assertRaises(ValueError):
                live.validate_request(request)


if __name__ == '__main__':
    unittest.main()
