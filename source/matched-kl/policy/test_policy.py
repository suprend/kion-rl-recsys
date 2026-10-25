"""Source-only and fake-object tests. Never import NumPy/Torch or open arrays/models."""
import ast
import builtins
import copy
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'policy.py'


def load():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('numpy', 'torch'):
            raise AssertionError('Numeric import in source-only check: ' + name)
        return original(name, *args, **kwargs)
    spec = importlib.util.spec_from_file_location('_policy_source_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch('builtins.__import__', guarded):
        spec.loader.exec_module(module)
    return module


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = load()
        cls.text = SOURCE.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.text)

    def request(self):
        return dict(utility='completion', beta=.5, seed=20261024, precision='cuda-fp16',
                    device='cuda', cpu_threads=4, reserve_gib=2, cap_gib=8)

    def test_supported_parameters(self):
        for utility in ('completion', 'residual'):
            for seed in self.p.SEEDS:
                request = self.request() | dict(utility=utility, beta=.0123, seed=seed)
                self.assertEqual(self.p.parameters(request), (utility, .0123, seed))

    def test_nonfit_evaluation_requires_zero_rl_updates(self):
        request = self.request() | dict(mode='evaluate_train512', rl_updates=0)
        self.assertEqual(self.p.parameters(request), ('completion', .5, 20261024))
        with self.assertRaises(ValueError):
            self.p.parameters(request | dict(rl_updates=1000))
        with self.assertRaises(ValueError):
            self.p.parameters(request | dict(mode='train', rl_updates=0))

    def test_reject_missing_nonfinite_and_invalid(self):
        for beta in (None, 0, -1, math.inf, math.nan, True, '0.5'):
            with self.assertRaises(ValueError):
                self.p.parameters(self.request() | dict(beta=beta))
        for key, value in [('utility', 'both'), ('seed', 20261027), ('seed', True), ('cpu_threads', 8),
                           ('cap_gib', 16), ('reserve_gib', 0), ('rl_updates', 250), ('batch_size', 8),
                           ('learning_rate', 1e-4), ('optimizer', 'SGD'), ('grad_clip', 2),
                           ('start_from', 'latest'), ('precision', 'fp32'), ('weight_decay', .01)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.p.parameters(self.request() | {key: value})

    def test_frozen_sources_and_metadata_closure(self):
        request = self.p.read(self.p.FINAL / 'request-residual-fit-r1.json')
        request['source_hashes'][SOURCE.as_posix()] = self.p.sha(SOURCE)
        utility, policy, residual = self.p.load_dependencies(request)
        self.assertEqual(policy.IMPLEMENTATION, 'producer-shape-slot-v2')
        self.assertEqual(utility.FORMULA, 'r_RM(s,i)-m_fit(i)')
        self.assertEqual(residual.CONTRACT, 'kion-current-C1070-residual-H1-v1')
        for key in ('cache_dir', 'reward_dir', 'utility_dir'):
            changed = copy.deepcopy(request)
            changed[key] = str(HERE / 'wrong')
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.p.load_dependencies(changed)

    def test_sampler_exact_replacement_and_reproducibility(self):
        eligible = [2, 5, 11]
        for seed in self.p.SEEDS:
            expected_rng = random.Random(seed)
            expected = [expected_rng.choice(eligible) for _ in range(16)]
            actual = self.p.sample_positions(random.Random(seed), eligible)
            self.assertEqual(actual, expected)
            self.assertLess(len(set(actual)), len(actual))

    def test_original_head_metadata_separate_from_rl_seed_and_batch(self):
        calls = []
        class FakeHeads:
            def load_heads(self, request, torch, device):
                self_outer.assertEqual((request['seed'], request['batch_size'], request['epochs']), (20261024, 128, 3))
                calls.append(request)
                return {'C': 'fake-head'}
        class FakePolicy:
            def load(self, request, name):
                self_outer.assertEqual(name, 'heads')
                return FakeHeads()
        self_outer = self
        for seed in self.p.SEEDS:
            request = self.request() | dict(seed=seed, batch_size=16)
            self.assertEqual(self.p.original_head(request, FakePolicy(), None, 'fake-device'), 'fake-head')
            self.assertEqual((request['seed'], request['batch_size']), (seed, 16))
        self.assertEqual(len(calls), 3)

    def test_output_rejects_frozen_roots(self):
        for path in (self.p.OLD / 'policy/new', self.p.FINAL / 'policy/new', self.p.ROOT / 'policy'):
            with self.assertRaises(ValueError):
                self.p.owned_output({'rl_output': path}, 'rl_output')

    def test_fit_budget_root_callback_required(self):
        admission = dict(maximum_new_fits=8, fit_ordinal=1, reservation_id='fake-only',
                         ledger_path='fake-ledger.json', ledger_sha256='a' * 64)
        calls = []
        self.assertEqual(self.p.require_fit_admission(dict(fit_admission=admission),
                         lambda **fields: calls.append(fields)), admission)
        self.assertEqual(calls[0]['stage'], 'kl-seeds-fit-admission')
        for ordinal in (0, 9, True):
            with self.assertRaises(ValueError):
                self.p.require_fit_admission(dict(fit_admission=admission | dict(fit_ordinal=ordinal)), lambda **kw: None)
        with self.assertRaises(ValueError):
            self.p.require_fit_admission(dict(fit_admission=admission), None)

    def test_train_shared_math_and_optimizer_contract(self):
        funcs = {node.name: ast.get_source_segment(self.text, node) for node in self.tree.body if isinstance(node, ast.FunctionDef)}
        train = funcs['train']
        for seam in ('prepare_shaped_inputs', 'score_shaped', 'effective_advantage', 'exact_loss'):
            self.assertNotIn(seam, funcs)
            self.assertIn('policy.' + seam + '(', train)
        for fragment in ('range(1, 1001)', 'torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)',
                         'clip_grad_norm_(model.parameters(), 1.', 'random.Random(seed)',
                         'policy.exact_loss(torch, scores, refs, adv, beta, valid)',
                         'np.array_equal(actual, expected)', 'scale_before = scaler.get_scale()'):
            self.assertIn(fragment, train)
        self.assertLess(train.index('Update0 top10 mismatch'), train.index('torch.optim.Adam'))
        self.assertEqual(self.p.STEPS, (0, 250, 500, 1000))

    def test_evaluation_has_no_training_or_feature_entrypoints(self):
        node = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_train512')
        text = ast.get_source_segment(self.text, node)
        for forbidden in ('optimizer', 'prepare_cache(', 'open_frozen_context(', 'candidate_features(', '.backward(', '.step('):
            if forbidden == 'optimizer':
                self.assertIn('optimizer_created=False', text)
            else:
                self.assertNotIn(forbidden, text)
        self.assertIn('range(0, 512, 16)', text)
        self.assertIn("('completion', reward), ('residual', derived)", text)
        self.assertIn('np.array_equal(values, expected)', text)

    def test_run_rejects_unknown_mode_before_numeric_work(self):
        with self.assertRaises(ValueError):
            self.p.run(dict(mode='prepare_cache'))


if __name__ == '__main__':
    unittest.main()
