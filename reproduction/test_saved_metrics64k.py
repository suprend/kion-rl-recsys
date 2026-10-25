"""Synthetic tests of ranking, denominators, shared cluster draws and validation."""
import copy
import unittest

import saved_metrics as old
import saved_metrics64k as sm
import export_saved_metrics64k as exporter


def fixture():
    return {"contract": sm.CONTRACT, "eligibility": [0, 1, 2, 2, 2],
            "clusters": [0, 0, 1, 2, 2],
            "ranks": {arm: [0, 0, 1, 10, 11] for arm in sm.ARMS}}


class ReplayTests(unittest.TestCase):
    def test_unknown_offpool_and_outside_top10_stay_in_denominator(self):
        got = sm.calculate(fixture(), rows=5)
        self.assertAlmostEqual(got["per_arm"]["C"]["map10"], 1.1/5)
        self.assertEqual(got["per_arm"]["C"]["recall10"], 2/5)
        self.assertEqual(got["counts"], {"known": 4, "unknown": 1, "offpool_known": 1, "covered": 3})

    def test_family_average_not_best_seed_or_extra_observations(self):
        payload = fixture()
        for seed, rank in zip(old.SEEDS, (1, 2, 10)):
            payload["ranks"][f"completion-{seed}"][2] = rank
        got = sm.calculate(payload, rows=5)
        self.assertAlmostEqual(got["main_table"]["completion"]["map10"], ((1+.5+.1)/3+.1)/5)
        self.assertEqual(got["unique_users"], 3)

    def test_invalid_rank_cluster_and_eligibility_rejected(self):
        for key, value in (("clusters", [0, 0, 2, 3, 3]), ("clusters", [True, 0, 1, 2, 2]),
                           ("eligibility", [0, 1, 2, 2, 0])):
            payload = fixture()
            payload[key] = value
            with self.assertRaises(old.ValidationError):
                sm.calculate(payload, rows=5)
        for rank in (True, -1, 1025, 0, 1.0):
            payload = fixture()
            payload["ranks"]["C"][2] = rank
            with self.assertRaises(old.ValidationError):
                sm.calculate(payload, rows=5)

    def test_cluster_duplicates_row_weighting_and_shared_draws(self):
        np = sm.numpy()
        users = np.array([10, 10, 20, 30, 30])
        values = np.array([1., 3., 7., 11., 13.])
        differences = np.stack([values, 2*values, -values], axis=1)
        got, count = sm.paired_cluster_bootstrap(np, users, differences)
        self.assertEqual(count, 3)
        # Independent literal expansion of the same random cluster counts.
        rng = np.random.default_rng(20261024)
        weights = rng.multinomial(3, np.full(3, 1/3), size=2000)
        expected = np.array([(w[0]*4+w[1]*7+w[2]*24)/(w[0]*2+w[1]+w[2]*2) for w in weights])
        np.testing.assert_allclose(got[:, 0], expected, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(got[:, 1], 2*got[:, 0])
        np.testing.assert_array_equal(got[:, 2], -got[:, 0])
        # Sorted dense anonymization preserves draw-to-cluster association.
        dense, _ = sm.paired_cluster_bootstrap(np, [0, 0, 1, 2, 2], differences)
        np.testing.assert_array_equal(got, dense)
        row_bootstrap, _ = sm.paired_cluster_bootstrap(np, np.arange(5), differences)
        self.assertFalse(np.array_equal(row_bootstrap, got))

    def test_tied_saved_order_preserved_and_bad_top_or_permutation_rejected(self):
        np = sm.numpy()
        # First two items had tied scores; producer raw-ID ascending puts token8 first.
        pool = np.array([[7, 8, 9], [7, 8, 9]])
        ranks = np.array([[2, 1, 3], [2, 1, 3]])
        top = np.array([[8, 7], [8, 7]])
        target = np.array([7, 9])
        flags = np.array([True, True])
        got = exporter.extract_target_ranks(np, pool, target, flags, flags, ranks, top)
        self.assertEqual(got, [2, 3])
        with self.assertRaisesRegex(old.ValidationError, "disagreement"):
            exporter.extract_target_ranks(np, pool, target, flags, flags, ranks, top[:, ::-1])
        with self.assertRaisesRegex(old.ValidationError, "permutation"):
            exporter.extract_target_ranks(np, pool, target, flags, flags, np.ones_like(ranks), top)

    def test_expected_values_cannot_replace_computation(self):
        got = sm.calculate(fixture(), rows=5)
        expected = copy.deepcopy(got)
        expected["per_arm"]["C"]["map10"] += .01
        with self.assertRaisesRegex(old.ValidationError, "scalar mismatch"):
            sm.compare(got, expected)


if __name__ == "__main__":
    unittest.main()
