"""Pinned compact-JSON figure builder; only run() imports Matplotlib.

No models, arrays, bootstrap, raw data or selection are accessed here. Root owns
admission and the single controlled native CPU worker. Import/tests are stdlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve()
ANALYZER = ROOT.parent / "interpretation" / "analyze.py"
SEEDS = (20261024, 20261025, 20261026)
ANALYSIS_CONTRACT = "kion-matched-kl-multiseed-analysis-v1"
CONTRACT = "kion-matched-kl-figure-v1"
TRAIN_CACHE_SHA = "e76c365d1245355d5314e383ba53499800eb3c099fa0af7de106cced5fad15e5"
KINDS = ("completion", "residual")
CONTRAST = "residual-minus-completion"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value, label, lower=None, upper=None):
    require(type(value) in (int, float) and math.isfinite(value), "Finite number required: " + label)
    require(lower is None or value >= lower, "Below range: " + label)
    require(upper is None or value <= upper, "Above range: " + label)
    return float(value)


def valid_pin(entry, label):
    require(isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"],
            "Path pin required: " + label)
    digest = entry.get("sha256")
    require(isinstance(digest, str) and len(digest) == 64 and
            all(char in "0123456789abcdef" for char in digest), "SHA256 pin required: " + label)
    require(Path(entry["path"]).is_absolute(), "Absolute pinned path required: " + label)
    return Path(entry["path"]).resolve()


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    require(isinstance(value, dict), "JSON object required")
    # Also reject overflow such as 1e999, which JSON parse_constant does not see.
    def walk(node):
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
        elif isinstance(node, float):
            require(math.isfinite(node), "Nonfinite JSON number")
    walk(value)
    return value


def small_json(entry, expected, label):
    path = valid_pin(entry, label)
    require(path == expected.resolve() and path.suffix == ".json", "Unexpected compact input: " + label)
    require(path.is_file() and path.stat().st_size <= 2 * 1024 * 1024,
            "Small completed JSON required: " + label)
    require(sha(path) == entry["sha256"], "Pinned source differs: " + label)
    value = read_json(path)
    return value, {"path": path.as_posix(), "sha256": entry["sha256"], "bytes": path.stat().st_size}


def source_pins(request):
    entries = request.get("source_hashes")
    require(isinstance(entries, dict) and 2 <= len(entries) <= 32, "Explicit source_hashes required")
    result = {}
    for name, digest in entries.items():
        path = valid_pin({"path": name, "sha256": digest}, "source")
        require(path.suffix == ".py" and path.is_file() and path.stat().st_size <= 2 * 1024 * 1024,
                "Small Python source required")
        require(path.as_posix() not in result, "Duplicate resolved source pin")
        require(sha(path) == digest, "Source hash differs: " + path.as_posix())
        result[path.as_posix()] = digest
    require(SOURCE.as_posix() in result and ANALYZER.resolve().as_posix() in result,
            "Pin both figures.py and interpretation/analyze.py")
    return result


def matching_rows(matching, analysis, matching_pin):
    """Recompute only compact TRAIN512 ratios; never use evaluation-cohort KL."""
    checkpoints = analysis.get("checkpoints")
    require(isinstance(checkpoints, dict) and len(checkpoints) == 7 and "C" in checkpoints,
            "Exactly C and six fitted checkpoints required")
    require(all(isinstance(name, str) for name in checkpoints) and
            len({name.casefold() for name in checkpoints}) == 7, "Checkpoint names collide")
    pairs = {seed: {} for seed in SEEDS}
    selected = {}
    for name, checkpoint in checkpoints.items():
        valid_pin(checkpoint, name)
        if name == "C":
            continue
        seed, kind = checkpoint.get("seed"), checkpoint.get("utility")
        require(type(seed) is int and seed in SEEDS and kind in KINDS and checkpoint.get("step") == 1000,
                "Unexpected checkpoint seed/utility/step")
        require(kind not in pairs[seed], "Duplicate seed/utility")
        pairs[seed][kind] = name
        selected[name] = checkpoint["sha256"]
    require(all(set(pair) == set(KINDS) for pair in pairs.values()), "Missing fitted pair")
    require(matching.get("selected_checkpoints") == selected, "Matching checkpoint bindings differ")
    require(matching.get("tolerance") == [.8, 1.2], "Registered TRAIN ratio band must be [0.8, 1.2]")
    target = finite(matching.get("target_train_KL"), "target_train_KL")
    require(target > 1e-8, "Calibration target must exceed 1e-8")
    require(valid_pin(matching.get("cache"), "TRAIN cache") and
            matching["cache"]["sha256"] == TRAIN_CACHE_SHA, "Frozen TRAIN512 cache pin differs")
    valid_pin(matching.get("reference"), "TRAIN reference")
    values = matching.get("train_KL_by_arm")
    require(isinstance(values, dict) and set(values) == set(selected), "Six TRAIN KL values required")
    values = {name: finite(value, "TRAIN KL/" + name, 0) for name, value in values.items()}
    require(values[pairs[20261024]["completion"]] == target, "Seed24 calibration TRAIN KL differs")
    flags = matching.get("train_KL_match_by_seed")
    require(isinstance(flags, dict) and set(flags) == {str(seed) for seed in SEEDS}, "Three TRAIN flags required")
    normalized, rows = {}, []
    for seed in SEEDS:
        completion, residual = (values[pairs[seed][kind]] for kind in KINDS)
        ratio = residual / completion if completion else None
        if ratio is not None:
            finite(ratio, "TRAIN ratio/" + str(seed), 0)
        flag = ratio is not None and .8 <= ratio <= 1.2
        require(type(flags[str(seed)]) is bool and flags[str(seed)] == flag, "TRAIN flag differs: " + str(seed))
        normalized[str(seed)] = flag
        rows.append({"seed": seed, "role": "calibration" if seed == SEEDS[0] else "prospective",
                     "completion_train_KL": completion, "residual_train_KL": residual,
                     "ratio": ratio, "train_KL_match": flag})
    copied = analysis.get("training_KL_matching")
    require(isinstance(copied, dict), "Analysis TRAIN matching provenance required")
    receipt = copied.get("receipt")
    require(valid_pin(receipt, "analysis matching receipt") == Path(matching_pin["path"]).resolve() and
            receipt["sha256"] == matching_pin["sha256"], "Analysis matching receipt pin differs")
    require(copied.get("receipt_metadata") == matching, "Analysis matching receipt contents differ")
    require(copied.get("train_KL_match_by_seed") == normalized and
            copied.get("all_three_train_KL_matches") is all(normalized.values()), "Analysis matching flags differ")
    require(copied.get("train_KL_residual_over_completion_by_seed") ==
            {str(row["seed"]): row["ratio"] for row in rows}, "Analysis TRAIN ratios differ")
    return rows, pairs


def outcome_row(analysis, seed, pairs):
    prefix = str(seed) if seed is not None else "three-seed-average"
    name = prefix + "/" + CONTRAST
    comparison = analysis.get("comparisons", {}).get(name)
    require(isinstance(comparison, dict) and comparison.get("name") == name and
            comparison.get("kind") == CONTRAST and comparison.get("seed") == seed,
            "Expected paired comparison: " + name)
    expected = [[pairs[item]["completion"], pairs[item]["residual"]]
                for item in (SEEDS if seed is None else (seed,))]
    require(comparison.get("terms") == expected, "Paired comparison arms differ: " + name)
    if seed is None:
        require(comparison.get("equal_seed_weight") == 1 / 3, "Equal three-seed weights required")
    metric = comparison.get("metrics", {}).get("map10")
    require(isinstance(metric, dict), "MAP@10 contrast required")
    difference = finite(metric.get("difference"), name + "/difference", -1, 1)
    interval = metric.get("user_cluster_95ci")
    require(isinstance(interval, list) and len(interval) == 2, "Two user-cluster CI endpoints required")
    low, high = [finite(value, name + "/CI", -1, 1) for value in interval]
    require(low <= high, "Reversed user-cluster CI")
    counts = [metric.get(key) for key in ("gains", "losses", "ties")]
    require(all(type(value) is int and value >= 0 for value in counts) and sum(counts) == 4096,
            "Paired query accounting differs")
    return {"seed": seed, "role": "descriptive three-seed average" if seed is None else
            ("calibration" if seed == SEEDS[0] else "prospective"),
            "difference": difference, "user_cluster_95ci": [low, high]}


def figure_data(matching, analysis, matching_pin, sources, include_average=True):
    """Pure stdlib validation and projection, testable without plotting imports."""
    require(analysis.get("contract") == ANALYSIS_CONTRACT and analysis.get("state") == "completed" and
            analysis.get("stage") == "full4096", "Completed full4096 analysis required")
    require(analysis.get("rows") == analysis.get("metric_denominator") == 4096 and
            analysis.get("candidate_width") == 1024 and analysis.get("seeds") == list(SEEDS),
            "DEV4096 / natural1024 / three-seed scope differs")
    require(analysis.get("primary_metric") == "map10" and
            analysis.get("primary_comparison") == "three-seed-average/" + CONTRAST, "Primary estimand differs")
    source = analysis.get("source")
    require(valid_pin(source, "analysis producer") == ANALYZER.resolve() and
            source["sha256"] == sources.get(ANALYZER.resolve().as_posix()), "Analysis producer source pin differs")
    valid_pin(analysis.get("evaluation"), "evaluation provenance")
    users = analysis.get("unique_users")
    require(type(users) is int and 1 < users <= 4096, "Valid user-cluster count required")
    bootstrap = analysis.get("bootstrap")
    require(isinstance(bootstrap, dict) and bootstrap.get("replicates") == 2000 and
            bootstrap.get("seed") == 20261024 and bootstrap.get("common_draw_schedule") is True and
            bootstrap.get("multiple_comparison_adjustment") is False and
            bootstrap.get("CI") == "unadjusted percentile 2.5/97.5, NumPy linear quantiles",
            "Registered conditional user-cluster interval protocol differs")
    train, pairs = matching_rows(matching, analysis, matching_pin)
    outcomes = [outcome_row(analysis, seed, pairs) for seed in SEEDS]
    average = outcome_row(analysis, None, pairs)
    require(math.isclose(average["difference"], math.fsum(row["difference"] for row in outcomes) / 3,
                         rel_tol=1e-12, abs_tol=1e-12), "Three-seed descriptive mean differs")
    require(type(include_average) is bool, "include_three_seed_average must be boolean")
    return {"train": train, "outcomes": outcomes + ([average] if include_average else []),
            "target_train_KL": matching["target_train_KL"], "tolerance": [.8, 1.2],
            "cohorts": {"train_matching_queries": 512, "development_queries": 4096,
                        "development_users": users, "candidate_actions": 1024},
            "CI_interpretation": "Exploratory unadjusted 95% user-cluster percentile intervals; conditional on six fitted heads and frozen C/encoder/RM/pools; not training-seed-population intervals.",
            "average_interpretation": "Equal-weight descriptive average of calibration seed24 and prospective seeds25/26; conditional user-cluster CI, not a training-seed-population CI.",
            "scope": "Reused DEV4096, all1024 natural actions. TRAIN512 ratios alone determine matching. Offline proxy comparison; no causal user-outcome interpretation."}


def prepare(request):
    """Read/hash only named compact JSON and Python sources; make no outputs."""
    require(isinstance(request, dict), "Request object required")
    sources = source_pins(request)
    analysis, analysis_pin = small_json(request.get("analysis"), ROOT.parent / "interpretation/full-r1/analysis.json", "analysis")
    matching, matching_pin = small_json(request.get("matching_receipt"), ROOT.parent / "protocols/matching.json", "matching")
    output = Path(request.get("output_dir", "")).resolve()
    require(output == (ROOT / "figures-r1").resolve() and not output.exists(),
            "Fresh reports/figures-r1 output required")
    data = figure_data(matching, analysis, matching_pin, sources, request.get("include_three_seed_average", True))
    return output, data, {"analysis": analysis_pin, "matching_receipt": matching_pin, "source_hashes": sources}


def render(plt, data, output):
    """Draw only already validated scalars; plotting is called by root run()."""
    colors = ("#697586", "#007C83", "#2E6595", "#202C3C")
    ys = (3, 2, 1, -.25)
    labels = ["20261024  calibration", "20261025  prospective", "20261026  prospective"]
    if len(data["outcomes"]) == 4:
        labels.append("3-seed average\n(descriptive)")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.edgecolor": "#AAB2BB", "axes.linewidth": .65,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "svg.fonttype": "none",
                         "savefig.facecolor": "white", "figure.facecolor": "white"}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11.8, 6.25), gridspec_kw={"width_ratios": [1, 1.12]})
        fig.subplots_adjust(left=.185, right=.965, top=.755, bottom=.34, wspace=.34)
        fig.suptitle("Train KL matching and development MAP@10", x=.055, y=.965,
                     ha="left", fontsize=17, fontweight="bold", color="#172C40")
        fig.text(.055, .905, "TRAIN512 matching  |  reused DEV4096 outcome cohort  |  all1024 natural actions",
                 ha="left", fontsize=10, color="#526170")
        left.set_title("A   TRAIN512 KL ratio", loc="left", pad=15, fontweight="bold")
        right.set_title("B   DEV4096 paired difference", loc="left", pad=15, fontweight="bold")
        left.axvspan(.8, 1.2, color="#E6E8EB", zorder=0)
        left.axvline(1, color="#A7ADB6", linewidth=.8, linestyle=(0, (3, 3)), zorder=1)
        ratios = [row["ratio"] for row in data["train"] if row["ratio"] is not None]
        left.set_xlim(0, max(1.55, max(ratios, default=1.2) * 1.42))
        for index, row in enumerate(data["train"]):
            value, y = row["ratio"], ys[index]
            if value is None:
                left.text(.05, y, "undefined: completion KL = 0", transform=left.get_yaxis_transform(),
                          va="center", fontsize=8.5, color=colors[index])
            else:
                left.plot(value, y, "o", color=colors[index], markerfacecolor="white" if index == 0 else colors[index],
                          markersize=7, markeredgewidth=1.6, zorder=3)
                left.annotate(f"{value:.3f}" + ("  match" if row["train_KL_match"] else "  outside"),
                              (value, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5)
        left.set_xlabel("KL(residual) / KL(completion)", labelpad=9)
        left.set_yticks(ys[:3], labels[:3])
        left.text(.0, -.27, "Gray band: registered inclusive [0.8, 1.2].\nMatch flags use TRAIN512 only.",
                  transform=left.transAxes, ha="left", va="top", fontsize=9, color="#526170")
        low = min(0., *(100 * row["user_cluster_95ci"][0] for row in data["outcomes"]),
                  *(100 * row["difference"] for row in data["outcomes"]))
        high = max(0., *(100 * row["user_cluster_95ci"][1] for row in data["outcomes"]),
                   *(100 * row["difference"] for row in data["outcomes"]))
        span = max(high - low, .01)
        right.set_xlim(low - .13 * span, high + .13 * span)
        right.axvline(0, color="#8B96A3", linewidth=.85, linestyle=(0, (3, 3)), zorder=1)
        for index, row in enumerate(data["outcomes"]):
            y, value = ys[index], 100 * row["difference"]
            lo, hi = (100 * number for number in row["user_cluster_95ci"])
            # Plot interval and point separately: percentile CIs need not contain the point estimate.
            right.hlines(y, lo, hi, color=colors[index], linewidth=1.6, zorder=2)
            right.vlines([lo, hi], y - .07, y + .07, color=colors[index], linewidth=1.1, zorder=2)
            right.plot(value, y, marker="D" if row["seed"] is None else "o", color=colors[index],
                       markerfacecolor="white" if index == 0 else colors[index], markersize=6.4,
                       markeredgewidth=1.5, zorder=3)
            right.text(1, y + .21, f"{value:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                       transform=right.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color=colors[index])
        right.set_xlabel("Residual minus completion MAP@10 (percentage points)", labelpad=9)
        right.set_yticks(ys[:len(labels)], [""] * len(labels))
        if len(labels) == 4:
            right.axhline(.35, linewidth=.7, color="#D7DBE0")
            right.text(-.035, ys[3], labels[3], transform=right.get_yaxis_transform(),
                       ha="right", va="center", fontsize=8.5, color=colors[3])
        right.text(.0, -.27, "Whiskers: conditional user-cluster 95% CIs.\nPositive values favor residual on this offline metric.",
                   transform=right.transAxes, ha="left", va="top", fontsize=9, color="#526170")
        for axis in (left, right):
            axis.set_ylim(-.8, 3.65)
            axis.grid(axis="x", color="#E8ECF0", linewidth=.6, zorder=0)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0, pad=10)
        mean_note = "The 3-seed mean includes calibration seed24." if len(labels) == 4 else "Seed24 was used for calibration."
        fig.text(.055, .045, "Exploratory intervals condition on six fitted heads and frozen C / encoder / RM / pools;\n"
                 "they do not describe a population of training seeds. " + mean_note + "\n"
                 "Development data were reused. Offline relevance is not a causal user outcome.",
                 ha="left", va="bottom", fontsize=9, linespacing=1.45, color="#526170")
        try:
            for extension in ("png", "svg"):
                fig.savefig(output / ("train-kl-and-map." + extension), dpi=240,
                            metadata={"Creator": "Pinned compact-JSON figure builder"})
        finally:
            plt.close(fig)


def run(request, check=None):
    require(sys.dont_write_bytecode, "Run worker with python -B")
    output, data, inputs = prepare(request)
    if check:
        check(stage="figure-compact-inputs-and-sources-verified")
    # Deliberately absent from import, prepare(), and the standard-library tests.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    # Reverify all input/source bytes immediately before the first output write.
    second_output, second_data, second_inputs = prepare(request)
    require((output, data, inputs) == (second_output, second_data, second_inputs), "Inputs changed before render")
    output.mkdir(exist_ok=False)
    render(plt, data, output)
    if check:
        check(stage="figure-rendered-awaiting-visual-QA")
    outputs = {path.name: {"path": path.as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}
               for path in (output / "train-kl-and-map.png", output / "train-kl-and-map.svg")}
    manifest = {"contract": CONTRACT, "state": "rendered-awaiting-visual-QA", "inputs": inputs,
                "plot_data": data, "outputs": outputs, "matplotlib_version": matplotlib.__version__,
                "visual_QA": "Root must inspect PNG and SVG for clipping, overlap and label legibility before delivery.",
                "access": "Only two compact JSON inputs and pinned Python sources read; no models, raw data, arrays, bootstrap, training or policy selection."}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate compact JSON and pins without Matplotlib/output")
    arguments = parser.parse_args()
    request = read_json(arguments.request)
    if arguments.check:
        output, data, inputs = prepare(request)
        print(json.dumps({"state": "validated-source-only", "output_dir": output.as_posix(), "inputs": inputs}))
    else:
        result = run(request)
        print(json.dumps({"state": result["state"], "outputs": result["outputs"]}))
