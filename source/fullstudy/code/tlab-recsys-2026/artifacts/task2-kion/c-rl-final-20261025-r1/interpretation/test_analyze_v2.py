"""Stdlib toy precision seam. No NumPy, real arrays, native jobs or models."""
import ast
import importlib.util
import math
from pathlib import Path
import struct
import unittest

SOURCE = Path(__file__).with_name('analyze_v2.py')
spec = importlib.util.spec_from_file_location('scientific3_analysis_v2', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


class TinyArray:
    def __init__(self, values, dtype):
        self.values = values
        self.dtype = dtype

    def astype(self, dtype):
        cast = f32 if dtype == 'float32' else float
        return TinyArray([[cast(value) for value in row] for row in self.values], dtype)

    def sum(self):
        # The precision fixture has only two elements: NumPy's two-value sum
        # is exactly addition rounded once to the array's accumulation dtype.
        result = math.fsum(value for row in self.values for value in row)
        return f32(result) if self.dtype == 'float32' else result


class TinyNP:
    float32, float64 = 'float32', 'float64'

    @staticmethod
    def where(mask, values, other):
        return TinyArray([[value if supported else other for value, supported in zip(row, flags)]
                          for row, flags in zip(values.values, mask.values)], values.dtype)

    @staticmethod
    def take_along_axis(values, order, axis):
        assert axis == 1
        return TinyArray([[row[j] for j in indices] for row, indices in zip(values.values, order)], values.dtype)

    @staticmethod
    def array_equal(left, right):
        return left.values == right.values


class PrecisionRegressionTests(unittest.TestCase):
    def test_reproduce_parent_float32_top10_mean_under_strict_guard(self):
        raw = TinyArray([[f32(.1), f32(.2)]], 'float64')
        support = TinyArray([[True, True]], 'bool')
        actual = mod.producer_top10_raw_reward(TinyNP, raw, support, [[0, 1]], 'float32')
        producer = f32(f32(.1) + f32(.2)) / 2
        self.assertTrue(math.isclose(actual, producer, abs_tol=1e-12, rel_tol=1e-12),
                        f'Recomputed screen diagnostic differs C/supported_top10_raw_reward: {actual} vs {producer}')

    def test_float64_scientific_mean_stays_distinct_from_producer_check(self):
        raw = TinyArray([[f32(.1), f32(.2)]], 'float64')
        support = TinyArray([[True, True]], 'bool')
        scientific = raw.sum() / support.sum()
        producer = mod.producer_top10_raw_reward(TinyNP, raw, support, [[0, 1]], 'float32')
        self.assertFalse(math.isclose(scientific, producer, abs_tol=1e-12, rel_tol=1e-12))
        self.assertEqual(raw.dtype, 'float64')
        self.assertEqual(raw.values, [[f32(.1), f32(.2)]])

    def test_parent_dtype_export_dtype_and_nonroundtrip_values_fail_closed(self):
        raw = TinyArray([[f32(.1), f32(.2)]], 'float64')
        support = TinyArray([[True, True]], 'bool')
        with self.assertRaisesRegex(ValueError, 'Parent RM reward dtype'):
            mod.producer_top10_raw_reward(TinyNP, raw, support, [[0, 1]], 'float64')
        with self.assertRaisesRegex(ValueError, 'Exported raw RM reward dtype'):
            mod.producer_top10_raw_reward(TinyNP, raw.astype('float32'), support, [[0, 1]], 'float32')
        # The invalid value is unsupported: the guard must still check it.
        bad = TinyArray([[f32(.1), .2]], 'float64')
        with self.assertRaisesRegex(ValueError, 'Every raw RM export value'):
            mod.producer_top10_raw_reward(TinyNP, bad, TinyArray([[True, False]], 'bool'), [[0, 1]], 'float32')

    def test_support_denominator_and_mask_preserve_original_producer_semantics(self):
        raw = TinyArray([[f32(.1), f32(.2)]], 'float64')
        support = TinyArray([[True, False]], 'bool')
        self.assertEqual(mod.producer_top10_raw_reward(TinyNP, raw, support, [[0, 1]], 'float32'), f32(.1))
        self.assertIsNone(mod.producer_top10_raw_reward(TinyNP, raw, TinyArray([[False, False]], 'bool'), [[0, 1]], 'float32'))

    def test_parent_metadata_is_shape_cache_and_dtype_bound(self):
        parent = {'contract': 'current-C-local-RM-r2-bridge-v1', 'ready': True, 'states': 256,
                  'candidates_per_state': 1024, 'rows': 256 * 1024, 'cache': {'sha256': 'a' * 64},
                  'files': {'rewards.npy': {'dtype': 'float32', 'shape': [256, 1024]}}}
        self.assertEqual(mod.validate_parent_reward_metadata(parent, 256, 'a' * 64), 'float32')
        with self.assertRaisesRegex(ValueError, 'cohort/cache'):
            mod.validate_parent_reward_metadata(parent, 256, 'b' * 64)
        parent['files']['rewards.npy']['dtype'] = 'float64'
        with self.assertRaisesRegex(ValueError, 'Parent RM reward dtype'):
            mod.validate_parent_reward_metadata(parent, 256, 'a' * 64)

    def test_core_scientific_functions_and_all_other_producer_checks_unchanged(self):
        original = ast.parse(SOURCE.with_name('analyze.py').read_text(encoding='utf-8'))
        current = ast.parse(SOURCE.read_text(encoding='utf-8'))
        old_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
        new_functions = {node.name: node for node in current.body if isinstance(node, ast.FunctionDef)}
        for name in ('bootstrap', '_screen', '_distribution', '_neutral', 'rank_oracle', 'target_rank', 'row_metrics',
                     'rank_bin', 'exposure', 'cluster_oracle', 'neutral_oracle', 'shift_oracle', 'load_array', 'validate_metadata'):
            self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]), name)
        def comparisons(tree):
            assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
                              any(isinstance(target, ast.Name) and target.id == 'comparisons_to_producer' for target in node.targets))
            return {key.value: ast.dump(value) for key, value in zip(assignment.value.keys, assignment.value.values)}
        old_checks, new_checks = comparisons(original), comparisons(current)
        self.assertEqual(set(old_checks), set(new_checks))
        self.assertEqual([key for key in old_checks if old_checks[key] != new_checks[key]], ['supported_top10_raw_reward'])
        source = SOURCE.read_text(encoding='utf-8')
        self.assertIn('math.isclose(value, expected[key], abs_tol=1e-12, rel_tol=1e-12)', source)
        self.assertIn('screen_diagnostic_precision', source)
        top_imports = [alias.name for node in current.body if isinstance(node, ast.Import) for alias in node.names]
        self.assertNotIn('numpy', top_imports)


if __name__ == '__main__':
    unittest.main(verbosity=2)
