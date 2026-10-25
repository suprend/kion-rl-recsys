"""Add single-positive row-query MAP@10 to existing KION validation exports.

CPU-only postprocessing: no model execution, checkpoint selection, training, or
raw/prepared split access. Each query has one next-film positive, so AP@10 is
the reciprocal rank through 10 (zero for a miss or unknown target). The primary
MAP averages all validation rows, not weekly multi-positive user queries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


HELPER_PATH = Path(__file__).resolve().with_name("kion_compare_predictions.py")
EXPECTED_HELPER_SHA256 = "3b4966e6974a4fd52743ab90cf5999e65c0fb38bf9bd9f3b901fcbf65ae638c3"
METRIC_ORDER = ("map@10", "recall@10")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_helper():
    # Bind the reused validation and resampling implementation before execution.
    if sha256(HELPER_PATH) != EXPECTED_HELPER_SHA256:
        raise ValueError("Prediction helper source differs from reviewed implementation")
    spec = importlib.util.spec_from_file_location("_kion_map10_prediction_helper", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if sha256(HELPER_PATH) != EXPECTED_HELPER_SHA256:
        raise ValueError("Prediction helper source changed while loading")
    return helper


paired = _load_helper()


def single_positive_scores(data):
    """Compute [AP@10, hit@10] from an export validated by paired.load_export."""
    top = data["top_items"]
    matches = (top == data["target_token"][:, None]) & data["known_target"][:, None]
    ap = (matches / np.arange(1, top.shape[1] + 1, dtype=np.float64)).sum(axis=1)
    return np.column_stack((ap, matches.any(axis=1).astype(np.float64)))


def compare(baseline, candidate, metadata_path, output, replicates=1000, seed=2026):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("MAP comparison output must not already exist")
    source_path = Path(__file__).resolve()
    source_hash, helper_hash = sha256(source_path), sha256(HELPER_PATH)
    if helper_hash != EXPECTED_HELPER_SHA256:
        raise ValueError("Prediction helper source differs from reviewed implementation")
    metadata_path = Path(metadata_path).resolve()
    metadata, metadata_hash = paired.read_json_snapshot(metadata_path)
    a, _, aggregate_a, bind_a = paired.load_export(baseline, metadata, metadata_hash)
    b, _, _, bind_b = paired.load_export(candidate, metadata, metadata_hash)
    for name in (*paired.ROW_KEYS, "row_index", "target_token", "known_target"):
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f"Baseline/candidate row/label mismatch: {name}")
    contract_a, contract_b = bind_a["declared_contract"], bind_b["declared_contract"]
    if contract_a["inputs_sha256"] != contract_b["inputs_sha256"]:
        raise ValueError("Different prepared split bindings")
    if contract_a["sources_sha256"]["kion_conditioned_train.py"] != contract_b["sources_sha256"]["kion_conditioned_train.py"]:
        raise ValueError("Different whole evaluator source/tie contract")
    if contract_a["architecture"].get("context_length") != contract_b["architecture"].get("context_length"):
        raise ValueError("Different prepared context contract")

    scores_a, scores_b = single_positive_scores(a), single_positive_scores(b)
    users, inverse, sizes = np.unique(a["user_id"], return_inverse=True, return_counts=True)
    sums_a = np.column_stack([np.bincount(inverse, weights=scores_a[:, n]) for n in range(2)])
    sums_b = np.column_stack([np.bincount(inverse, weights=scores_b[:, n]) for n in range(2)])
    row_draws, user_draws = paired.cluster_bootstrap(sums_b - sums_a, sizes, replicates, seed)
    results = {}
    for scope, point_a, point_b, draws in (
        ("row_primary", scores_a.mean(axis=0), scores_b.mean(axis=0), row_draws),
        ("user_macro_companion", (sums_a / sizes[:, None]).mean(axis=0),
         (sums_b / sizes[:, None]).mean(axis=0), user_draws),
    ):
        results[scope] = {
            name: {"baseline": float(point_a[n]), "candidate": float(point_b[n]),
                   "delta_candidate_minus_baseline": float(point_b[n] - point_a[n]),
                   "paired_percentile_95_ci": np.quantile(draws[:, n], [.025, .975], method="linear").tolist()}
            for n, name in enumerate(METRIC_ORDER)
        }

    bindings = [bind[k] for bind in (bind_a, bind_b) for k in ("predictions", "config", "metrics")]
    bindings += [entry for bind in (bind_a, bind_b) for entry in bind["receipts"].values()]
    if (sha256(metadata_path) != metadata_hash or sha256(source_path) != source_hash or
            sha256(HELPER_PATH) != helper_hash or
            any(sha256(entry["path"]) != entry["sha256"] for entry in bindings)):
        raise ValueError("MAP comparison input/source changed while computing")
    report = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": bind_a, "candidate": bind_b,
        "metadata": {"path": metadata_path.as_posix(), "sha256": metadata_hash},
        "source": {"path": source_path.as_posix(), "sha256": source_hash},
        "helper": {"path": HELPER_PATH.as_posix(), "sha256": helper_hash},
        "metric_order": list(METRIC_ORDER),
        "definition": {
            "query_unit": "one validation row with one next-film positive",
            "ap@10": "1/rank if the known target is ranked 1..10, otherwise 0",
            "map@10": "mean AP@10 over the full validation row count, including unknown targets as zero",
            "equivalence": "truncated MRR@10 for these single-positive row queries",
            "scope": "not weekly multi-positive user MAP",
            "user_macro_companion": "mean of each user's mean row AP@10; includes unknown target rows",
            "recall@10": "mean single-positive hit@10 with the same denominators",
        },
        "rows": len(scores_a), "users": len(users), "same_key_domain": True,
        "row_key_domain_sha256": paired.canonical_hash(np.column_stack([a[name] for name in paired.ROW_KEYS]).tolist()),
        "known_targets": aggregate_a["known_targets"],
        "unknown_target_misses": aggregate_a["unknown_target_misses"],
        "candidate_catalog": {"tokens": "3..vocab_size-1", "vocab_size": metadata["vocab_size"],
                              "mapping_sha256": paired.canonical_hash(metadata["item_id_to_token"]),
                              "actual_k": aggregate_a["actual_k"]},
        "cache_absolute_tolerance": paired.CACHE_ATOL,
        "alignment": "sort unique user_id,target_item_id,target_date; equal row_index binding required",
        "bootstrap": {"replicates": replicates, "seed": seed, "unit": "user", "paired": True,
                      "user_order": "ascending numeric user_id",
                      "method": "uniform users with replacement; row ratio of cluster sums; user mean of cluster means",
                      "percentiles": [.025, .975], "quantile_method": "linear", "maximum_draw_batch": 10},
        "results": results,
        "interpretation": "Conditional on these trained and selected models and this development week; user sampling uncertainty only. Does not estimate training-seed variance or remove adaptive checkpoint/feature selection bias. No p-value or confirmatory superiority claim.",
        "safety": {"test_accessed": False, "raw_data_accessed": False,
                   "model_weights_accessed": False, "models_executed": False,
                   "checkpoint_selection_changed": False, "unknown_targets_in_all_denominators": True},
    }
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "bootstrap_deltas.npz"
    np.savez_compressed(artifact, row_delta=row_draws, user_delta=user_draws)
    report["bootstrap_artifact"] = {"file": artifact.name, "sha256": sha256(artifact)}
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.metadata, args.output, args.replicates, args.seed)
    print(json.dumps({"rows": report["rows"], "users": report["users"], "results": report["results"]}, indent=2))


if __name__ == "__main__":
    main()
