"""Stdlib synthetic/source checks. Never import NumPy or open real payloads."""
import ast
import importlib.util
import math
from pathlib import Path
import random
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('utility_under_test', HERE / 'utility.py')
utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility)


class UtilityTests(unittest.TestCase):
    def test_prior_schema(self):
        utility.validate_prior_records([7, 9], [1, 0], [.4, .5])
        for ids, counts, means in (([7, 7], [1, 2], [.1, .2]), ([7], [-1], [.1]),
                                   ([7], [1.5], [.1]), ([7], [True], [.1]),
                                   ([7], [1], [math.nan]), ([7], [1], [1.1]), ([], [], [])):
            with self.subTest(ids=ids, counts=counts, means=means), self.assertRaises(ValueError):
                utility.validate_prior_records(ids, counts, means)

    def test_varying_prior_flips_preference(self):
        values = [utility.residual_value(.8, .7), utility.residual_value(.6, .1)]
        self.assertLess(values[0], values[1])
        self.assertEqual(utility.pairwise_preference_changes([.8, .6], values), (1, 1))

    def test_constant_offset_cancels_under_centering(self):
        weights, values = [.2, .3, .5], [.8, .5, .2]
        residual = [utility.residual_value(r, .4) for r in values]
        center = lambda items: [v - sum(p * r for p, r in zip(weights, items)) for v in items]
        for a, b in zip(center(values), center(residual)):
            self.assertAlmostEqual(a, b)
        self.assertEqual(utility.pairwise_preference_changes(values, residual), (0, 3))

    def test_unsupported_is_neutral_storage(self):
        self.assertEqual(utility.residual_value(.8, .1, False), 0.)
        with self.assertRaises(ValueError):
            utility.residual_value(math.nan, .1, False)

    def test_preference_count_matches_independent_bruteforce_including_ties(self):
        rng = random.Random(91)
        sign = lambda x: (x > 0) - (x < 0)
        for length in range(1, 26):
            for _ in range(8):
                rewards = [rng.randrange(6) for _ in range(length)]
                utilities = [rng.randrange(8) for _ in range(length)]
                expected = sum(sign(rewards[i] - rewards[j]) != sign(utilities[i] - utilities[j])
                               for i in range(length) for j in range(i))
                self.assertEqual(utility.pairwise_preference_changes(rewards, utilities),
                                 (expected, length * (length - 1) // 2))

    def test_numerical_imports_are_deferred(self):
        tree = ast.parse((HERE / 'utility.py').read_text(encoding='utf-8'))
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in statement.names] if isinstance(statement, ast.Import) else [statement.module]
                self.assertFalse(any(name.split('.')[0] in {'numpy', 'torch', 'catboost'} for name in names))

    def test_archive_extraction_reads_only_three_named_fields(self):
        source = (HERE / 'utility.py').read_text(encoding='utf-8')
        self.assertIn("for key in ('item_ids', 'counts', 'item_mean')", source)
        self.assertNotIn('CatBoostRegressor', source)
        self.assertNotIn('model.cbm', source)
        self.assertIn("np.array_equal(arrays['candidate_ids'], indices + 3)", source)
        self.assertIn('np.array_equal(counts[indices] > 0, support)', source)
        self.assertIn("require(summaries['train']['nondegenerate_prior']", source)

    def test_utility_range_contract_separate_from_rm(self):
        self.assertNotEqual(utility.CONTRACT, 'current-C-local-RM-r2-bridge-v1')
        self.assertEqual(utility.residual_value(0., 1.), -1.)
        self.assertEqual(utility.residual_value(1., 0.), 1.)
        for reward, prior in ((-.1, .2), (.1, 1.1), (math.inf, .2)):
            with self.assertRaises(ValueError):
                utility.residual_value(reward, prior)


if __name__ == '__main__':
    unittest.main()
