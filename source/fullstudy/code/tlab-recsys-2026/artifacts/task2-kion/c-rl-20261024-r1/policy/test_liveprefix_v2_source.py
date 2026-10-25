"""Stdlib-only live-prefix contract checks; no data, torch or NumPy imports."""
import ast
import importlib.util
import math
from pathlib import Path
import random
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_liveprefix_v2_source_test", HERE / "liveprefix_v2.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class PrefixTests(unittest.TestCase):
    def test_original_peers_and_active_slot_are_preserved(self):
        peers = [[1, 3, 4], [1], [1, 2, 9, 11]]
        padded, lengths = live.producer_prefix_batch(peers, [1, 7], 3, 1)
        self.assertEqual(padded, [[1, 3, 4, 0], [1, 7, 0, 0], [1, 2, 9, 11]])
        self.assertEqual(lengths, [3, 2, 4])
        self.assertEqual(peers, [[1, 3, 4], [1], [1, 2, 9, 11]])

    def test_peer_max_trim_width_and_active_growth(self):
        peers = [[1, 3], [1, 7, 8]]
        padded, lengths = live.producer_prefix_batch(peers, [1, 3], 2, 0)
        self.assertEqual(len(padded[0]), 3)
        grown, lengths = live.producer_prefix_batch(peers, [1, 3, 9, 10], 2, 0)
        self.assertEqual(grown, [[1, 3, 9, 10], [1, 7, 8, 0]])
        self.assertEqual(lengths, [4, 3])

    def test_partial_producer_and_nonzero_last_slot(self):
        peers = [[1, i + 3] for i in range(9)]
        padded, lengths = live.producer_prefix_batch(peers, [1, 99], 9, 8)
        self.assertEqual(len(padded), 9)
        self.assertEqual(padded[8], [1, 99])
        self.assertEqual(padded[:8], peers[:8])
        self.assertEqual(lengths, [2] * 9)

    def test_invalid_producer_layout_is_rejected(self):
        for size, slot in ((0, 0), (17, 0), (1, 1), (1, -1), (True, 0), (2, 0)):
            with self.subTest(size=size, slot=slot), self.assertRaises(ValueError):
                live.producer_prefix_batch([[1, 3]], [1, 3], size, slot)

    def test_bos_only_and_right_padding(self):
        self.assertEqual(live.observed_prefix([1, 0, 0], 1), [1])
        self.assertEqual(live.observed_prefix([1, 2, 7, 0], 3), [1, 2, 7])

    def test_invalid_padding_or_length(self):
        for tokens, length in (([1, 7, 0], 1), ([1, 0, 7], 3), ([7, 0], 1),
                               ([1, 1], 2), ([1], 0), ([1], 2), ([1] + [3] * 64, 65)):
            with self.subTest(tokens=tokens, length=length), self.assertRaises(ValueError):
                live.observed_prefix(tokens, length)

    def test_append_preserves_bos_and_latest63(self):
        original = [1, *range(3, 66)]
        grown = live.append_generated(original, 90)
        self.assertEqual(grown, [1, *range(4, 66), 90])
        self.assertEqual(len(grown), 64)
        self.assertEqual(original, [1, *range(3, 66)])
        self.assertEqual(live.append_generated([1], 9), [1, 9])

    def test_generated_specials_forbidden(self):
        for token in (0, 1, 2):
            with self.assertRaises(ValueError):
                live.append_generated([1, 8], token)


class DistributionTests(unittest.TestCase):
    def test_shift_invariance_and_temperature(self):
        p, _ = live.distribution([1, 2, 3])
        shifted, _ = live.distribution([1001, 1002, 1003])
        self.assertEqual(p, shifted)
        self.assertAlmostEqual(math.fsum(p), 1)
        warm, _ = live.distribution([1, 2, 3], temperature=2)
        self.assertLess(warm[2], p[2])

    def test_mask_shared_for_exact_kl(self):
        p, lp = live.distribution([1, 2, 3], excluded={1})
        _, lq = live.distribution([1, 2, 3], excluded={1})
        result = live.policy_metrics(p, lp, lq, [0.2, math.nan, 0.9], [True, False, True], [0, 0, 0])
        self.assertEqual(p[1], 0)
        self.assertIsNone(lp[1])
        self.assertEqual(result["kl_nats"], 0)
        self.assertEqual(result["unsupported_mass"], 0)
        self.assertAlmostEqual(result["supported_mass"], 1)

    def test_extreme_logits_are_stable(self):
        p, lp = live.distribution([10000, -10000])
        result = live.policy_metrics(p, lp, lp, [0.4, math.nan], [True, False], [0, 0])
        self.assertEqual(result["kl_nats"], 0)
        self.assertEqual(result["entropy_nats"], 0)

    def test_unknown_reward_is_not_observed_zero(self):
        p, lp = live.distribution([0, 0])
        result = live.policy_metrics(p, lp, lp, [math.nan, math.nan], [False, False], [0, 0])
        self.assertEqual(result["unsupported_mass"], 1)
        self.assertIsNone(result["expected_supported_reward_conditional"])
        self.assertEqual(result["expected_neutral_proxy"], 0)

    def test_bad_softmax_contracts(self):
        for kwargs in ({"logits": []}, {"logits": [math.inf]}, {"logits": [math.nan]},
                       {"logits": [0], "excluded": {0}}, {"logits": [0], "temperature": 0},
                       {"logits": [0], "temperature": math.inf}, {"logits": [0], "excluded": {2}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                live.distribution(**kwargs)

    def test_greedy_tie_and_seeded_sampling(self):
        self.assertEqual(live.select_index([0.5, 0.5, 0], "greedy", random.Random(9)), 0)
        def trajectory(seed):
            rng = random.Random(seed)
            return [live.select_index([0.2, 0.0, 0.8], "categorical", rng) for _ in range(25)]
        self.assertEqual(trajectory(19), trajectory(19))
        self.assertNotIn(1, trajectory(19))
        self.assertNotEqual(trajectory(19), trajectory(20))


class SourceBoundaryTests(unittest.TestCase):
    def test_imports_are_stdlib_only(self):
        tree = ast.parse((HERE / "liveprefix_v2.py").read_text(encoding="utf-8"))
        modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertEqual(set(modules), {"__future__", "math", "random"})

    def test_no_target_reads_or_provider_rm_calls(self):
        tree = ast.parse((HERE / "liveprefix_v2.py").read_text(encoding="utf-8"))
        keys = {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)}
        self.assertTrue({"candidate_ids", "raw_features", "source_values", "user_features", "x", "lengths"} <= keys)
        self.assertFalse(keys & {"y", "target_ids", "target_item_id", "targets", "provider", "reward_model"})
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"batch", "candidate_features", "sample", "backward", "step"})
        self.assertIn("encode_context", calls)
        self.assertIn("retrieval_logits", calls)
        self.assertNotIn("linear", calls)
        self.assertTrue({'producer_x', 'producer_lengths', 'producer_user_features', 'producer_size',
                         'producer_slot', 'score_live_head'} <= keys)

    def test_initial_parity_remains_required_and_tolerance_unchanged(self):
        source = (HERE / 'liveprefix_v2.py').read_text(encoding='utf-8')
        self.assertIn("if 'reference_scores' not in state:", source)
        self.assertNotIn("if step == 0 and 'reference_scores' in state:", source)
        tree = ast.parse(source)
        comparisons = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and node.func.attr == 'allclose']
        self.assertEqual(len(comparisons), 1)
        self.assertEqual({keyword.arg: keyword.value.value for keyword in comparisons[0].keywords},
                         {'atol': .003, 'rtol': .001})
        self.assertIn("bound = .006 + .001*", source)

    def test_retrieval_is_inside_step_loop_and_not_cached_state(self):
        tree = ast.parse((HERE / 'liveprefix_v2.py').read_text(encoding='utf-8'))
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'step']
        self.assertEqual(len(loops), 1)
        calls = [node for node in ast.walk(loops[0]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'retrieval_logits']
        self.assertEqual(len(calls), 1)
        self.assertEqual([arg.id for arg in calls[0].args], ['q_group', 'table'])
        assigned = [node for node in ast.walk(loops[0]) if isinstance(node, ast.Assign)]
        for node in assigned:
            if any(isinstance(target, ast.Name) and target.id in ('q', 'base') for target in node.targets):
                names = {name.id for name in ast.walk(node.value) if isinstance(name, ast.Name)}
                self.assertFalse(names & {'initial_q', 'initial_base', 'state'})

    def test_invalid_config_rejected_before_numeric_context(self):
        for kwargs in ({"steps": 6}, {"steps": 0}, {"repeat_mask": "seen"}, {"selection": "target"},
                       {"temperature": 0}, {"gamma": 2}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                live.rollout_state({}, None, None, {}, [], [], **kwargs)


if __name__ == "__main__":
    unittest.main()
