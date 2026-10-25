"""Stdlib synthetic compact-metadata tests; never import Matplotlib or render."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("figures.py")
spec = importlib.util.spec_from_file_location("figures_under_test", SOURCE)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def fixture(directory):
    root = Path(directory)
    checkpoints = {"C": {"path": (root / "C.pt").as_posix(), "sha256": "c" * 64}}
    for index, seed in enumerate(figures.SEEDS):
        for kind, offset in (("completion", 0), ("residual", 1)):
            name = kind + "-" + str(seed)
            checkpoints[name] = {"path": (root / (name + ".pt")).as_posix(),
                "sha256": str(index * 2 + offset + 1) * 64, "seed": seed, "utility": kind, "step": 1000}
    matching = {"selected_checkpoints": {name: item["sha256"] for name, item in checkpoints.items() if name != "C"},
        "target_train_KL": .1, "tolerance": [.8, 1.2],
        "train_KL_by_arm": {name: (.12 if name == "residual-20261026" else .1) for name in checkpoints if name != "C"},
        "train_KL_match_by_seed": {str(seed): True for seed in figures.SEEDS},
        "cache": {"path": (root / "cache.json").as_posix(), "sha256": figures.TRAIN_CACHE_SHA},
        "reference": {"path": (root / "reference.json").as_posix(), "sha256": "f" * 64}}
    matching_pin = {"path": (root / "protocols/matching.json").as_posix(), "sha256": "b" * 64}
    contrasts = {}
    differences = [-.001, .002, .002]
    for index, seed in enumerate((*figures.SEEDS, None)):
        name = (str(seed) if seed is not None else "three-seed-average") + "/" + figures.CONTRAST
        terms = [["completion-" + str(item), "residual-" + str(item)]
                 for item in (figures.SEEDS if seed is None else (seed,))]
        contrasts[name] = {"name": name, "seed": seed, "kind": figures.CONTRAST, "terms": terms,
            "metrics": {"map10": {"difference": differences[index] if seed is not None else .001,
                "user_cluster_95ci": [-.003, .004], "gains": 50, "losses": 46, "ties": 4000}}}
        if seed is None:
            contrasts[name]["equal_seed_weight"] = 1 / 3
    analysis = {"contract": figures.ANALYSIS_CONTRACT, "state": "completed", "stage": "full4096",
        "rows": 4096, "metric_denominator": 4096, "candidate_width": 1024, "seeds": list(figures.SEEDS),
        "primary_metric": "map10", "primary_comparison": "three-seed-average/" + figures.CONTRAST,
        "source": {"path": figures.ANALYZER.as_posix(), "sha256": figures.sha(figures.ANALYZER)},
        "evaluation": {"path": (root / "evaluation.json").as_posix(), "sha256": "a" * 64},
        "checkpoints": checkpoints, "unique_users": 3700, "comparisons": contrasts,
        "training_KL_matching": {"receipt": dict(matching_pin), "receipt_metadata": copy.deepcopy(matching),
            "train_KL_match_by_seed": matching["train_KL_match_by_seed"].copy(), "all_three_train_KL_matches": True,
            "train_KL_residual_over_completion_by_seed": {str(seed): matching["train_KL_by_arm"]["residual-" + str(seed)] / .1
                for seed in figures.SEEDS}},
        "bootstrap": {"replicates": 2000, "seed": 20261024, "common_draw_schedule": True,
            "multiple_comparison_adjustment": False, "CI": "unadjusted percentile 2.5/97.5, NumPy linear quantiles"},
        "summaries": {name: {"exact_KL_to_C": 50} for name in checkpoints}}
    sources = {SOURCE.resolve().as_posix(): figures.sha(SOURCE), figures.ANALYZER.resolve().as_posix(): figures.sha(figures.ANALYZER)}
    return matching, analysis, matching_pin, sources


def sync_matching(matching, analysis):
    copied = analysis["training_KL_matching"]
    copied["receipt_metadata"] = copy.deepcopy(matching)
    copied["train_KL_match_by_seed"] = matching["train_KL_match_by_seed"].copy()
    copied["all_three_train_KL_matches"] = all(matching["train_KL_match_by_seed"].values())
    copied["train_KL_residual_over_completion_by_seed"] = {}
    for seed in figures.SEEDS:
        c, r = (matching["train_KL_by_arm"][kind + "-" + str(seed)] for kind in figures.KINDS)
        copied["train_KL_residual_over_completion_by_seed"][str(seed)] = r / c if c else None


class FigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.matching, self.analysis, self.pin, self.sources = fixture(self.temp.name)

    def project(self, average=True):
        return figures.figure_data(self.matching, self.analysis, self.pin, self.sources, average)

    def test_import_and_tests_use_no_numerical_packages(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        compile(tree, SOURCE.as_posix(), "exec")
        allowed = {"__future__", "argparse", "hashlib", "json", "math", "pathlib", "sys"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        imported = [alias.name for node in ast.walk(run) if isinstance(node, ast.Import) for alias in node.names]
        self.assertEqual(imported, ["matplotlib", "matplotlib.pyplot"])
        self.assertFalse(any(name in sys.modules for name in ("matplotlib", "numpy", "torch")))

    def test_scope_roles_and_descriptive_average(self):
        data = self.project()
        self.assertEqual([row["role"] for row in data["train"]], ["calibration", "prospective", "prospective"])
        self.assertEqual(len(data["outcomes"]), 4)
        self.assertEqual(data["outcomes"][-1]["difference"], .001)
        self.assertEqual(data["cohorts"], {"train_matching_queries": 512, "development_queries": 4096,
            "development_users": 3700, "candidate_actions": 1024})
        self.assertIn("not training-seed-population", data["CI_interpretation"])
        self.assertEqual(len(self.project(False)["outcomes"]), 3)
        with self.assertRaises(ValueError):
            self.project("yes")

    def test_dev_KL_never_determines_train_match(self):
        before = self.project()["train"]
        for summary in self.analysis["summaries"].values():
            summary["exact_KL_to_C"] = 999999.
        self.assertEqual(self.project()["train"], before)
        self.assertAlmostEqual(before[2]["ratio"], 1.2)

    def test_inclusive_band_and_separate_flags(self):
        values = self.matching["train_KL_by_arm"]
        values["completion-20261025"], values["residual-20261025"] = 1., .8
        values["completion-20261026"], values["residual-20261026"] = 1., 1.2
        sync_matching(self.matching, self.analysis)
        self.assertTrue(all(row["train_KL_match"] for row in self.project()["train"]))
        values["residual-20261025"] = .799
        self.matching["train_KL_match_by_seed"]["20261025"] = False
        sync_matching(self.matching, self.analysis)
        rows = self.project()["train"]
        self.assertEqual([row["train_KL_match"] for row in rows], [True, False, True])

    def test_zero_completion_has_undefined_ratio_and_fails_match(self):
        self.matching["train_KL_by_arm"]["completion-20261025"] = 0.
        self.matching["train_KL_match_by_seed"]["20261025"] = False
        sync_matching(self.matching, self.analysis)
        row = self.project()["train"][1]
        self.assertIsNone(row["ratio"])
        self.assertFalse(row["train_KL_match"])

    def test_zero_calibration_target_rejected(self):
        self.matching["target_train_KL"] = 0
        sync_matching(self.matching, self.analysis)
        with self.assertRaises(ValueError):
            self.project()

    def test_dynamic_checkpoint_names(self):
        names = {name: "new" + str(index) if name != "C" else "C"
                 for index, name in enumerate(self.analysis["checkpoints"])}
        self.analysis["checkpoints"] = {names[name]: value for name, value in self.analysis["checkpoints"].items()}
        for key in ("selected_checkpoints", "train_KL_by_arm"):
            self.matching[key] = {names[name]: value for name, value in self.matching[key].items()}
        for comparison in self.analysis["comparisons"].values():
            comparison["terms"] = [[names[left], names[right]] for left, right in comparison["terms"]]
        self.analysis["training_KL_matching"]["receipt_metadata"] = copy.deepcopy(self.matching)
        self.assertEqual(len(self.project()["train"]), 3)

    def test_checkpoint_binding_and_seed_are_not_inferred(self):
        self.matching["selected_checkpoints"]["residual-20261025"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint bindings"):
            self.project()
        self.matching, self.analysis, self.pin, self.sources = fixture(self.temp.name)
        self.analysis["checkpoints"]["residual-20261025"]["seed"] = 20261026
        with self.assertRaises(ValueError):
            self.project()

    def test_stale_copied_receipt_or_flag_rejected(self):
        for field, value in (("receipt_metadata", {}), ("train_KL_match_by_seed", {}),
                             ("all_three_train_KL_matches", False),
                             ("train_KL_residual_over_completion_by_seed", {})):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.analysis)
                changed["training_KL_matching"][field] = value
                with self.assertRaises(ValueError):
                    figures.figure_data(self.matching, changed, self.pin, self.sources)

    def test_false_claimed_match_rejected(self):
        self.matching["train_KL_by_arm"]["residual-20261025"] = .2
        sync_matching(self.matching, self.analysis)
        with self.assertRaisesRegex(ValueError, "TRAIN flag differs"):
            self.project()

    def test_nonfinite_negative_and_boolean_numbers_rejected(self):
        for value in (math.nan, math.inf, -1, True):
            with self.subTest(value=value):
                matching = copy.deepcopy(self.matching)
                matching["train_KL_by_arm"]["completion-20261025"] = value
                with self.assertRaises(ValueError):
                    figures.figure_data(matching, self.analysis, self.pin, self.sources)

    def test_nonfinite_ratio_from_extreme_finite_KLs_rejected(self):
        self.matching["train_KL_by_arm"].update({"completion-20261025": 1e-320, "residual-20261025": 1e308})
        with self.assertRaisesRegex(ValueError, "Finite number"):
            self.project()

    def test_wrong_stage_or_protocol_rejected(self):
        for key, value in (("stage", "screen"), ("rows", 256), ("candidate_width", 100),
                           ("state", "pending"), ("unique_users", 1), ("primary_metric", "recall10")):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.analysis)
                changed[key] = value
                with self.assertRaises(ValueError):
                    figures.figure_data(self.matching, changed, self.pin, self.sources)

    def test_changed_bootstrap_protocol_rejected(self):
        for key, value in (("replicates", 999), ("seed", 1), ("common_draw_schedule", False),
                           ("multiple_comparison_adjustment", True), ("CI", "seed-population")):
            changed = copy.deepcopy(self.analysis)
            changed["bootstrap"][key] = value
            with self.assertRaises(ValueError):
                figures.figure_data(self.matching, changed, self.pin, self.sources)

    def test_CI_ranges_and_query_counts(self):
        name = "20261025/" + figures.CONTRAST
        for field, value in (("difference", math.nan), ("difference", 1.1), ("user_cluster_95ci", [.1, -.1]),
                             ("user_cluster_95ci", [0, math.inf]), ("user_cluster_95ci", [-1.1, 0]),
                             ("gains", True), ("ties", 0)):
            changed = copy.deepcopy(self.analysis)
            changed["comparisons"][name]["metrics"]["map10"][field] = value
            with self.assertRaises(ValueError):
                figures.figure_data(self.matching, changed, self.pin, self.sources)

    def test_percentile_interval_need_not_contain_point(self):
        self.analysis["comparisons"]["20261025/" + figures.CONTRAST]["metrics"]["map10"]["user_cluster_95ci"] = [.003, .004]
        self.assertEqual(self.project()["outcomes"][1]["difference"], .002)

    def test_mean_must_equal_three_paired_estimates(self):
        self.analysis["comparisons"]["three-seed-average/" + figures.CONTRAST]["metrics"]["map10"]["difference"] = .1
        with self.assertRaisesRegex(ValueError, "descriptive mean"):
            self.project()

    def test_json_rejects_NaN_and_overflow(self):
        path = Path(self.temp.name) / "invalid.json"
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            path.write_text('{"value":' + token + '}', encoding="utf-8")
            with self.assertRaises(ValueError):
                figures.read_json(path)

    def write_inputs(self):
        root = Path(self.temp.name)
        (root / "reports").mkdir()
        (root / "protocols").mkdir()
        (root / "interpretation/full-r1").mkdir(parents=True)
        matching_path = root / "protocols/matching.json"
        matching_path.write_text(json.dumps(self.matching, allow_nan=False), encoding="utf-8")
        self.pin["sha256"] = figures.sha(matching_path)
        self.analysis["training_KL_matching"]["receipt"] = dict(self.pin)
        analysis_path = root / "interpretation/full-r1/analysis.json"
        analysis_path.write_text(json.dumps(self.analysis, allow_nan=False), encoding="utf-8")
        return {"analysis": {"path": analysis_path.as_posix(), "sha256": figures.sha(analysis_path)},
            "matching_receipt": dict(self.pin), "source_hashes": dict(self.sources),
            "output_dir": (root / "reports/figures-r1").as_posix()}

    def test_prepare_checks_real_source_hashes_without_output_or_plot(self):
        request = self.write_inputs()
        with patch.object(figures, "ROOT", Path(self.temp.name) / "reports"):
            output, data, inputs = figures.prepare(request)
        self.assertFalse(output.exists())
        self.assertEqual(len(data["train"]), 3)
        self.assertEqual(inputs["matching_receipt"]["sha256"], self.pin["sha256"])
        self.assertFalse("matplotlib" in sys.modules)

    def test_missing_stale_and_non_source_pins_rejected(self):
        request = self.write_inputs()
        altered = copy.deepcopy(request)
        altered["source_hashes"][SOURCE.resolve().as_posix()] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source hash differs"):
            figures.source_pins(altered)
        altered["source_hashes"] = {}
        with self.assertRaises(ValueError):
            figures.source_pins(altered)
        Path(request["matching_receipt"]["path"]).write_text("{}", encoding="utf-8")
        with patch.object(figures, "ROOT", Path(self.temp.name) / "reports"):
            with self.assertRaisesRegex(ValueError, "Pinned source differs"):
                figures.prepare(request)

    def test_existing_or_other_output_refused(self):
        request = self.write_inputs()
        with patch.object(figures, "ROOT", Path(self.temp.name) / "reports"):
            changed = dict(request, output_dir=(Path(self.temp.name) / "elsewhere").as_posix())
            with self.assertRaisesRegex(ValueError, "Fresh reports"):
                figures.prepare(changed)
            Path(request["output_dir"]).mkdir()
            with self.assertRaisesRegex(ValueError, "Fresh reports"):
                figures.prepare(request)


if __name__ == "__main__":
    unittest.main()
