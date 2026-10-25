"""Compare full validation exports with paired user-cluster bootstrap on CPU.

Prediction paths require sibling config/metrics/completion JSON and a selected
receipt for conditioned checkpoints. Unique row keys are aligned by sorting;
duplicates or partial validation exports are rejected. No raw/test files are read.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROW_KEYS = ("user_id", "target_item_id", "target_date")
FIELDS = (*ROW_KEYS, "row_index", "target_token", "known_target", "top_items", "hit_at_10", "ndcg_at_10")
CACHE_ATOL = 2e-7


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json_snapshot(path):
    payload = Path(path).read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def checkpoint_binding(path, metrics, metrics_hash, prediction_hash, aggregate):
    """Verify small exporter receipts, without opening or hashing tensor weights."""
    completion_path = path.with_name("completion.json")
    try:
        completion, completion_hash = read_json_snapshot(completion_path)
    except FileNotFoundError as error:
        raise ValueError("Missing completion receipt") from error
    epoch = metrics.get("selected_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Missing or invalid selected checkpoint epoch")
    if (completion.get("state") not in ("completed", "stopped") or
            type(completion.get("selected_epoch")) is not int or completion.get("selected_epoch") != epoch or completion.get("test_accessed") is not False or
            completion.get("validation_rows") != aggregate["examples"]):
        raise ValueError("Completion state/epoch/validation contract mismatch")
    hashes = completion.get("sha256")
    if not isinstance(hashes, dict) or hashes.get(path.name) != prediction_hash or hashes.get("metrics.json") != metrics_hash:
        raise ValueError("Completion export/metrics digest mismatch")
    completion_metrics = completion.get("metrics", {})
    for receipt_key, aggregate_key in (("ndcg10", "ndcg@10"), ("recall10", "recall@10")):
        value = completion_metrics.get(receipt_key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or
                abs(value - aggregate[aggregate_key]) > CACHE_ATOL):
            raise ValueError("Completion aggregate metric mismatch")
    receipts = {"completion": {"path": completion_path.as_posix(), "sha256": completion_hash}}
    if metrics.get("mode") == "legacy_validation_export":
        digest = metrics.get("checkpoint_sha256")
        checkpoint = metrics.get("checkpoint")
        if (not valid_digest(digest) or completion.get("legacy_checkpoint_sha256") != digest or
                not isinstance(checkpoint, str) or not checkpoint):
            raise ValueError("Legacy checkpoint identity/digest mismatch")
        identity = {"path": checkpoint, "sha256": digest, "epoch": epoch, "kind": "legacy"}
    else:
        selected_path = path.with_name("selected.json")
        try:
            selected, selected_hash = read_json_snapshot(selected_path)
        except FileNotFoundError as error:
            raise ValueError("Missing selected checkpoint receipt") from error
        digest = selected.get("sha256")
        filename = selected.get("file")
        if (not valid_digest(digest) or type(selected.get("epoch")) is not int or selected.get("epoch") != epoch or
                filename != f"selected-epoch{epoch:04d}.pt" or hashes.get(filename) != digest or
                hashes.get("selected.json") != selected_hash or selected.get("validation") != metrics.get("validation")):
            raise ValueError("Selected checkpoint identity/epoch/digest mismatch")
        receipts["selected"] = {"path": selected_path.as_posix(), "sha256": selected_hash}
        identity = {"file": filename, "sha256": digest, "epoch": epoch, "kind": "conditioned"}
    identity["verification"] = "consistent exporter receipts; weights not opened or hashed"
    return identity, receipts


def metadata_contract(metadata):
    size, mapping = metadata.get("vocab_size"), metadata.get("item_id_to_token")
    if (metadata.get("special_tokens") != {"PAD": 0, "BOS": 1, "UNK": 2} or
            isinstance(size, bool) or not isinstance(size, int) or size <= 3 or not isinstance(mapping, dict)):
        raise ValueError("Invalid metadata vocabulary/special tokens")
    if any(not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k or
           isinstance(v, bool) or not isinstance(v, int) for k, v in mapping.items()):
        raise ValueError("Invalid item ID/token mapping")
    if len(mapping) != size - 3 or sorted(mapping.values()) != list(range(3, size)):
        raise ValueError("Vocabulary must be one-to-one and contiguous")
    rows = metadata.get("splits", {}).get("val", {}).get("saved")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("Metadata must declare full validation row count")
    if metadata.get("split_dates", {}).get("val") != "2021-08-09..2021-08-15":
        raise ValueError("Only the fixed validation window is supported")
    return size, {int(k): v for k, v in mapping.items()}, rows


def load_export(path, metadata, metadata_hash):
    path = Path(path).resolve()
    config_path, metrics_path = path.with_name("config.json"), path.with_name("metrics.json")
    config, config_hash = read_json_snapshot(config_path)
    metrics, metrics_hash = read_json_snapshot(metrics_path)
    prediction_hash = sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        if not set(FIELDS).issubset(archive.files):
            raise ValueError("Missing prediction members")
        data = {name: archive[name] for name in FIELDS}
    if sha256(path) != prediction_hash:
        raise ValueError("Predictions changed while reading")
    size, mapping, count = metadata_contract(metadata)
    for name in (*ROW_KEYS, "row_index", "target_token"):
        if data[name].dtype != np.dtype("int64") or data[name].shape != (count,):
            raise ValueError(f"Invalid int64 row field: {name}")
    if any(np.any(data[name] < 0) for name in (*ROW_KEYS, "row_index")):
        raise ValueError("Negative row keys/indices")
    if not np.array_equal(np.sort(data["row_index"]), np.arange(count)):
        raise ValueError("row_index must cover the full validation NPZ once")
    if any(value.ndim < 1 or len(value) != count for value in data.values()):
        raise ValueError("Prediction members have inconsistent row counts")
    start, stop = date(2021, 8, 9).toordinal(), date(2021, 8, 16).toordinal()
    if not np.all((data["target_date"] >= start) & (data["target_date"] < stop)):
        raise ValueError("Prediction dates outside validation window")
    order = np.lexsort(tuple(data[name] for name in reversed(ROW_KEYS)))
    data = {name: value[order] for name, value in data.items()}
    keys = np.column_stack([data[name] for name in ROW_KEYS])
    if np.any(np.all(keys[1:] == keys[:-1], axis=1)):
        raise ValueError("Duplicate prediction row keys")
    expected_tokens = np.fromiter((mapping.get(int(item), 2) for item in data["target_item_id"]),
                                  dtype=np.int64, count=count)
    if not np.array_equal(data["target_token"], expected_tokens):
        raise ValueError("Target token disagrees with vocabulary/original item ID")
    known = expected_tokens >= 3
    if data["known_target"].dtype != np.bool_ or data["known_target"].shape != (count,) or not np.array_equal(data["known_target"], known):
        raise ValueError("Invalid known-target flags")
    k = min(10, size - 3)
    top = data["top_items"]
    if top.dtype != np.dtype("int64") or top.shape != (count, k) or np.any((top < 3) | (top >= size)):
        raise ValueError("Top items outside common catalog or invalid shape/dtype")
    if np.any(np.diff(np.sort(top, axis=1), axis=1) == 0):
        raise ValueError("Duplicate top items")
    matches = (top == expected_tokens[:, None]) & known[:, None]
    hit = matches.any(axis=1).astype(np.float64)
    ndcg = (matches / np.log2(np.arange(k, dtype=np.float64) + 2)).sum(axis=1)
    for name, expected in (("hit_at_10", hit), ("ndcg_at_10", ndcg)):
        cached = data[name]
        if cached.shape != (count,) or cached.dtype.kind not in "bfi" or not np.isfinite(cached).all():
            raise ValueError(f"Invalid cached per-row metric: {name}")
        if name == "hit_at_10" and not np.all((cached == 0) | (cached == 1)):
            raise ValueError("Cached hit must be exactly binary")
        if np.any((cached < 0) | (cached > 1)) or not np.allclose(cached, expected, rtol=0, atol=CACHE_ATOL):
            raise ValueError(f"Cached per-row metric disagrees with top items: {name}")
    aggregate = {"examples": count, "known_targets": int(known.sum()),
                 "unknown_target_misses": int((~known).sum()), "actual_k": k,
                 "recall@10": float(hit.mean()), "ndcg@10": float(ndcg.mean())}
    saved = metrics.get("validation")
    if not isinstance(saved, dict):
        raise ValueError("Missing cached aggregate validation metrics")
    ce = saved.get("cross_entropy_known")
    if (aggregate["known_targets"] and
            (isinstance(ce, bool) or not isinstance(ce, (int, float)) or not np.isfinite(ce) or ce < 0)):
        raise ValueError("Invalid cached known-target cross entropy")
    if not aggregate["known_targets"] and ce is not None:
        raise ValueError("Cross entropy must be null with no known targets")
    for name, expected in aggregate.items():
        value = saved.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid cached aggregate metric: {name}")
        if (value != expected if name in ("examples", "known_targets", "unknown_target_misses", "actual_k")
                else abs(value - expected) > CACHE_ATOL):
            raise ValueError(f"Cached aggregate disagrees with recompute: {name}")
    contract = config.get("contract", {})
    inputs = contract.get("inputs_sha256", {})
    if inputs.get("metadata.json") != metadata_hash or set(inputs) != {"metadata.json", "train.npz", "val.npz"}:
        raise ValueError("Config metadata/prepared input binding mismatch")
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in inputs.values()):
        raise ValueError("Invalid declared prepared input hash")
    architecture = contract.get("architecture", {})
    if (architecture.get("vocab_size") != size or architecture.get("context_length") != metadata.get("context") or
            not isinstance(metadata.get("context"), int) or metadata["context"] < 2 or
            contract.get("prediction_row_keys") != list(ROW_KEYS)):
        raise ValueError("Config vocabulary/row-key contract mismatch")
    evaluator = contract.get("sources_sha256", {}).get("kion_conditioned_train.py")
    if (not isinstance(evaluator, str) or len(evaluator) != 64 or
            any(c not in "0123456789abcdef" for c in evaluator)):
        raise ValueError("Missing evaluator source binding")
    if any(sha256(p) != digest for p, digest in ((config_path, config_hash), (metrics_path, metrics_hash))):
        raise ValueError("Config/metrics changed while reading")
    checkpoint, receipts = checkpoint_binding(path, metrics, metrics_hash, prediction_hash, aggregate)
    binding = {"predictions": {"path": path.as_posix(), "sha256": prediction_hash},
               "config": {"path": config_path.as_posix(), "sha256": config_hash},
               "metrics": {"path": metrics_path.as_posix(), "sha256": metrics_hash},
               "declared_contract": contract, "selected_epoch": metrics["selected_epoch"],
               "checkpoint": checkpoint, "receipts": receipts}
    return data, np.column_stack((ndcg, hit)), aggregate, binding


def cluster_bootstrap(cluster_delta, cluster_sizes, replicates=1000, seed=2026, batch_size=10):
    """Uniform paired cluster draws; row estimand uses each draw's row count."""
    if not 1 <= replicates <= 1000 or not 1 <= batch_size <= 20 or seed < 0:
        raise ValueError("Require 1..1000 replicates, 1..20 batch size, nonnegative seed")
    delta, sizes = np.asarray(cluster_delta, np.float64), np.asarray(cluster_sizes, np.int64)
    if delta.ndim != 2 or delta.shape[1] != 2 or sizes.shape != (len(delta),) or not len(sizes) or not np.isfinite(delta).all() or np.any(sizes <= 0):
        raise ValueError("Invalid cluster sums/sizes")
    rng = np.random.default_rng(seed)
    users = len(sizes)
    batch_size = max(1, min(batch_size, 2_000_000 // users))
    row, user = np.empty((replicates, 2)), np.empty((replicates, 2))
    user_delta = delta / sizes[:, None]
    for start in range(0, replicates, batch_size):
        stop = min(replicates, start + batch_size)
        draws = rng.integers(users, size=(stop - start, users), dtype=np.int64)
        row[start:stop] = delta[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
        user[start:stop] = user_delta[draws].mean(axis=1)
    return row, user


def compare(baseline, candidate, metadata_path, output, replicates=1000, seed=2026):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Comparison output must not already exist")
    metadata_path = Path(metadata_path).resolve()
    metadata, metadata_hash = read_json_snapshot(metadata_path)
    source_hash = sha256(Path(__file__))
    a, scores_a, aggregates_a, bind_a = load_export(baseline, metadata, metadata_hash)
    b, scores_b, aggregates_b, bind_b = load_export(candidate, metadata, metadata_hash)
    for name in (*ROW_KEYS, "row_index", "target_token", "known_target"):
        if not np.array_equal(a[name], b[name]):
            raise ValueError(f"Baseline/candidate row/label mismatch: {name}")
    contract_a, contract_b = bind_a["declared_contract"], bind_b["declared_contract"]
    if contract_a["inputs_sha256"] != contract_b["inputs_sha256"]:
        raise ValueError("Different prepared split bindings")
    if contract_a["sources_sha256"]["kion_conditioned_train.py"] != contract_b["sources_sha256"]["kion_conditioned_train.py"]:
        raise ValueError("Different evaluator source/tie contract")
    if contract_a["architecture"].get("context_length") != contract_b["architecture"].get("context_length"):
        raise ValueError("Different prepared context contract")
    users, inverse, sizes = np.unique(a["user_id"], return_inverse=True, return_counts=True)
    sums_a = np.column_stack([np.bincount(inverse, weights=scores_a[:, n]) for n in range(2)])
    sums_b = np.column_stack([np.bincount(inverse, weights=scores_b[:, n]) for n in range(2)])
    row_draws, user_draws = cluster_bootstrap(sums_b - sums_a, sizes, replicates, seed)
    result = {}
    for scope, point_a, point_b, draws in (
        ("row_primary", scores_a.mean(axis=0), scores_b.mean(axis=0), row_draws),
        ("user_macro_companion", (sums_a / sizes[:, None]).mean(axis=0), (sums_b / sizes[:, None]).mean(axis=0), user_draws),
    ):
        result[scope] = {name: {"baseline": float(point_a[n]), "candidate": float(point_b[n]),
                               "delta_candidate_minus_baseline": float(point_b[n] - point_a[n]),
                               "paired_percentile_95_ci": np.quantile(draws[:, n], [.025, .975], method="linear").tolist()}
                         for n, name in enumerate(("ndcg@10", "recall@10"))}
    # Recheck every consumed file and optional source after both models and draws.
    bindings = [bind_a[k] for k in ("predictions", "config", "metrics")] + [bind_b[k] for k in ("predictions", "config", "metrics")]
    bindings += list(bind_a["receipts"].values()) + list(bind_b["receipts"].values())
    if (sha256(metadata_path) != metadata_hash or sha256(Path(__file__)) != source_hash or
            any(sha256(entry["path"]) != entry["sha256"] for entry in bindings)):
        raise ValueError("Comparison input/source changed while computing")
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "baseline": bind_a, "candidate": bind_b, "metadata": {"path": metadata_path.as_posix(), "sha256": metadata_hash},
              "source": {"path": Path(__file__).resolve().as_posix(), "sha256": source_hash},
              "rows": len(scores_a), "users": len(users), "same_key_domain": True,
              "row_key_domain_sha256": canonical_hash(np.column_stack([a[name] for name in ROW_KEYS]).tolist()),
              "known_targets": aggregates_a["known_targets"], "unknown_target_misses": aggregates_a["unknown_target_misses"],
              "candidate_catalog": {"tokens": "3..vocab_size-1", "vocab_size": metadata["vocab_size"],
                                    "mapping_sha256": canonical_hash(metadata["item_id_to_token"]), "actual_k": aggregates_a["actual_k"]},
              "cache_absolute_tolerance": CACHE_ATOL, "alignment": "sort unique user_id,target_item_id,target_date; equal row_index binding required",
              "bootstrap": {"replicates": replicates, "seed": seed, "unit": "user", "paired": True,
                            "user_order": "ascending numeric user_id", "method": "uniform users with replacement; row ratio of cluster sums",
                            "percentiles": [.025, .975], "quantile_method": "linear", "maximum_draw_batch": 10},
              "results": result,
              "interpretation": "Conditional on these trained and selected models and this development week; user sampling uncertainty only. Does not estimate training-seed variance or remove adaptive checkpoint/feature selection bias. No p-value or confirmatory superiority claim.",
              "safety": {"test_accessed": False, "raw_data_accessed": False, "unknown_targets_in_all_denominators": True}}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "bootstrap_deltas.npz", row_delta=row_draws, user_delta=user_draws)
    report["bootstrap_artifact"] = {"file": "bootstrap_deltas.npz", "sha256": sha256(output / "bootstrap_deltas.npz")}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
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
