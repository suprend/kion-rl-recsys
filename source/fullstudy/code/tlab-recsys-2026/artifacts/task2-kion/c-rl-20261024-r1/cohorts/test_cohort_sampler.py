"""Pure synthetic checks; no numpy, real arrays, providers or SQLite reads."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

spec = importlib.util.spec_from_file_location('cohort_sampler', Path(__file__).with_name('cohort_sampler.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CohortTests(unittest.TestCase):
    def test_native_wrapper_checks_inner_pin_and_forwards_result(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Path(folder) / 'inner.json'
            c.write_new(cfg, {'contract': 'synthetic-inner'})
            request = dict(contract='synthetic-outer', cohort_request=str(cfg),
                cohort_request_sha256=c.sha(cfg), cohort_output=str(Path(folder) / 'out.json'))
            stages = []
            with patch.object(c, 'run', return_value={'state': 'completed'}) as worker:
                result = c.probe(request, lambda **item: stages.append(item['stage']))
                self.assertEqual(result, {'state': 'completed'})
                worker.assert_called_once_with({'contract': 'synthetic-inner'}, Path(request['cohort_output']))
            self.assertEqual(stages, ['cohort-probe-before', 'cohort-probe-after'])
            request['cohort_request_sha256'] = '0'*64
            with patch.object(c, 'run') as worker:
                with self.assertRaisesRegex(ValueError, 'request pin changed'):
                    c.probe(request)
                worker.assert_not_called()

    def test_excludes_actual_users_and_deduplicates_tied_states(self):
        day = c.TRAIN_START
        rows, counts = c.select_train([1, 1, 2, 3, 4, 5], [day, day, day, day-1, c.DEV_START, day+1], {2}, 2)
        self.assertEqual(rows, [0, 5])
        self.assertEqual(counts['duplicate_state_rows'], 1)
        self.assertEqual(counts['excluded_RM_user_rows'], 1)
        self.assertEqual(counts['outside_train_window'], 2)

    def test_no_silent_undersize(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            c.select_train([1], [c.TRAIN_START], {1}, 1)

    def test_selection_does_not_use_outcomes_targets_or_support(self):
        users, days = list(range(100)), [c.TRAIN_START] * 100
        a = c.select_train(users, days, set(), 10)[0]
        self.assertEqual(a, c.select_train(users, days, set(), 10)[0])
        self.assertEqual(len(set(a)), 10)
        self.assertEqual(a, sorted(a))

    def test_frozen_screen_retains_any_unknown_offpool_positions(self):
        rows = list(range(0, 8192, 2))
        selected, positions = c.select_screen(rows)
        self.assertEqual(len(selected), 256)
        self.assertEqual(selected, [rows[p] for p in positions])
        self.assertEqual((selected, positions), c.select_screen(rows))
        with self.assertRaises(ValueError):
            c.select_screen(rows[:-1])

    def test_state_key_stable_across_integer_storage_width(self):
        x = [1, 2, 3] + [0]*61
        self.assertEqual(c.context_hash(2, c.TRAIN_START, 3, x), c.context_hash(2, c.TRAIN_START, 3, tuple(x)))
        self.assertNotEqual(c.context_hash(2, c.TRAIN_START, 3, x), c.context_hash(2, c.TRAIN_START+1, 3, x))
        for bad in ([0]*64, [1, 3, 0]+[0]*61, [1, 3, 3]+[0]*60):
            with self.assertRaises(ValueError):
                c.context_hash(2, c.TRAIN_START, 3, bad)


if __name__ == '__main__':
    unittest.main()
