"""Portable frozen 65,536-query rank and paired user-cluster bootstrap replay.

Requires NumPy 2.5.4 for the historical random stream. No model/data download,
training or inference. Target ranks preserve the producer's score/raw-ID tie
order; replay does not re-sort scores. Dense labels omit original user IDs but
preserve cluster equality and sorted order; no reidentification claim is made.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Set before importing NumPy. No process-wide native pool can exceed two here.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import saved_metrics as old

CONTRACT = "kion-saved-ranks-user-bootstrap-64k-v1"
ROWS = 65536
LIMIT = 5 * 1024 * 1024
ARMS = ("C", "completion-20261024", "residual-20261024", "completion-20261025",
        "residual-20261025", "completion-20261026", "residual-20261026", "A", "B", "GPT", "RRF")
METRICS = ("map10", "ndcg10", "recall10")
LABELS = ("MAP@10", "NDCG@10", "Recall@10")
CONTRASTS = tuple((a, "C") for a in ARMS if a != "C") + (
    ("completion-equal-seed-mean", "C"), ("residual-equal-seed-mean", "C"),
    ("residual-equal-seed-mean", "completion-equal-seed-mean"))
BOOTSTRAP = {"replicates": 2000, "seed": 20261024, "batch_size": 32,
             "unit": "user_id", "paired": True, "shared_draws": True,
             "seeds_are_not_observations": True,
             "estimator": "resample unique user clusters with replacement; mean over all rows in sampled clusters",
             "interval": "percentile 2.5/97.5; no multiple-comparison adjustment"}
SCOPE = {"ranking": "saved target ranks; score sorting and tie breaking are not recomputed",
         "ids": "dense sorted user-cluster labels; source IDs omitted; reidentification not assessed",
         "table": "equal weight of three fitted policies per query, not an ensemble",
         "ci": "actual shared paired user-cluster bootstrap, 2000 replicates",
         "limits": "frozen development evaluation; no training, inference, causal or temporal-leakage validation"}
SOURCE_HASHES = {
    "evaluation.json": "c8be71308e73fba907a08102b44afefcc4f602c152db2264d5d9f9073e11604c",
    "analysis.json": "88d76e5d59996cc2b9a790ee54bad8037ab98ee1c2512a78308f7908fb3fb3d8",
    "ranks.json": "1b48869ff7965b01556456f27b049f87bf3a6f09b1d5e0eb8fae4fedff29628a",
    "analyze64k.py": "a5671134fa0ba29c4c8a1e6cfc45c2c440e360f553948455d0f08fc7438f05b8",
    "export_ranks.py": "c0600637c355ee761416e146730441765451de0da2ae1b714d101ace0cefcbb2",
    "original_bootstrap": "c5ac40bf2ffc215e4ef283b8a661c4239b4f5c9a39d29933cde8e080f4ab3ce7",
}


def validate_sources(sources):
    roles = set(SOURCE_HASHES) | {"export_saved_metrics64k.py", "saved_metrics64k.py"}
    roles |= {"array/"+name for name in ("user_id", "target_token", "known_target", "candidate_coverage", "pool_ids")}
    roles |= {"array/"+arm+"_top10" for arm in ARMS} | {"ranks/"+arm for arm in ARMS}
    old.exact_keys(sources, roles, "sources")
    for role, pin in sources.items():
        old.exact_keys(pin, ("path", "bytes", "sha256"), role)
        old.validate_descriptor(dict(pin, bytes=1), role)
        old.integer(pin["bytes"], 1, 1024**3, role)
        if role in SOURCE_HASHES:
            old.require(pin["sha256"] == SOURCE_HASHES[role], "unpinned source: "+role)


def numpy():
    import numpy as np
    old.require(np.__version__ == "2.5.4", "64k replay requires numpy==2.5.4")
    return np


def validate_payload(payload, rows=ROWS):
    old.exact_keys(payload, ("contract", "eligibility", "clusters", "ranks"), "64k payload")
    old.require(payload["contract"] == CONTRACT, "unsupported 64k contract")
    for name, high in (("eligibility", 2), ("clusters", rows - 1)):
        values = payload[name]
        old.require(type(values) is list and len(values) == rows, name + " length")
        for value in values:
            old.integer(value, 0, high, name)
    labels = set(payload["clusters"])
    old.require(len(labels) > 1 and labels == set(range(len(labels))), "cluster labels must be dense from zero")
    old.exact_keys(payload["ranks"], ARMS, "64k rank arms")
    for arm, ranks in payload["ranks"].items():
        old.require(type(ranks) is list and len(ranks) == rows, "rank length")
        for flag, rank in zip(payload["eligibility"], ranks):
            old.integer(rank, 0, 1024, "rank")
            old.require((flag == 2) == (rank > 0), "rank/eligibility mismatch")


def metric_rows(np, ranks):
    ranks = np.asarray(ranks)
    hit = (ranks >= 1) & (ranks <= 10)
    safe = np.where(hit, ranks, 1)
    return np.stack([hit / safe, hit / np.log2(safe + 1), hit.astype(float)], axis=1)


def paired_cluster_bootstrap(np, users, differences, replicates=2000, seed=20261024, batch_size=32):
    """Same multinomial draw order and row-weighted estimator as analyze.py."""
    unique, inverse = np.unique(users, return_inverse=True)
    old.require(len(unique) > 1 and differences.ndim == 2 and len(differences) == len(users), "cluster inputs differ")
    old.require(replicates == 2000 and 1 <= batch_size <= 32, "fixed 2000 bounded bootstrap required")
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.stack([np.bincount(inverse, weights=differences[:, col], minlength=len(unique))
                     for col in range(differences.shape[1])], axis=1)
    rng = np.random.default_rng(seed)
    result = np.empty((replicates, differences.shape[1]), dtype=np.float64)
    probability = np.full(len(unique), 1.0 / len(unique))
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        weights = rng.multinomial(len(unique), probability, size=stop-start)
        result[start:stop] = (weights @ sums) / (weights @ counts)[:, None]
    return result, len(unique)


def calculate(payload, bootstrap=False, rows=ROWS):
    validate_payload(payload, rows)
    np = numpy()
    metrics = {arm: metric_rows(np, payload["ranks"][arm]) for arm in ARMS}
    for family in ("completion", "residual"):
        metrics[family + "-equal-seed-mean"] = np.mean(
            [metrics[f"{family}-{seed}"] for seed in old.SEEDS], axis=0)
    per_arm = {arm: dict(zip(METRICS, map(float, metrics[arm].mean(0)))) for arm in ARMS}
    main = {a: per_arm[a] for a in ("GPT", "A", "B", "C", "RRF")}
    for family in ("completion", "residual"):
        main[family] = dict(zip(METRICS, map(float, metrics[family + "-equal-seed-mean"].mean(0))))
    status = payload["eligibility"]
    result = {"per_arm": per_arm, "main_table": main,
              "counts": {"known": rows-status.count(0), "unknown": status.count(0),
                         "offpool_known": status.count(1), "covered": status.count(2)},
              "unique_users": len(set(payload["clusters"]))}
    if bootstrap:
        differences = np.concatenate([metrics[a]-metrics[b] for a, b in CONTRASTS], axis=1)
        distribution, _ = paired_cluster_bootstrap(np, payload["clusters"], differences)
        result["contrasts"] = []
        for i, (a, b) in enumerate(CONTRASTS):
            for j, metric in enumerate(LABELS):
                col = 3*i+j
                result["contrasts"].append({"model": a, "reference": b, "metric": metric,
                    "difference": float(differences[:, col].mean()),
                    "ci95": np.quantile(distribution[:, col], [.025, .975]).tolist()})
        result["bootstrap"] = BOOTSTRAP
    return result


def compare(got, expected, label="result"):
    """Recursive finite-value comparison; expected values never feed arithmetic."""
    if type(got) is dict:
        old.exact_keys(expected, got, label)
        return max((compare(v, expected[k], label+"/"+k) for k, v in got.items()), default=0.)
    if type(got) is list:
        old.require(type(expected) is list and len(expected) == len(got), label+" length mismatch")
        return max((compare(a, b, label) for a, b in zip(got, expected)), default=0.)
    if type(got) is float:
        import math
        old.require(type(expected) in (float, int) and math.isfinite(expected), label+" invalid scalar")
        delta = abs(got-expected)
        old.require(delta <= 1e-12, label+" scalar mismatch")
        return delta
    old.require(type(got) is type(expected) and got == expected, label+" mismatch")
    return 0.


def replay(data, bootstrap=False):
    old.require(data.is_dir() and not data.is_symlink() and not getattr(data, "is_junction", lambda: False)(), "invalid data directory")
    old.require({p.name for p in data.iterdir()} == {"manifest.json", "rankings.json"}, "expected exactly manifest.json and rankings.json")
    raw = old.bounded_read(data / "manifest.json", LIMIT)
    manifest = old.decode_json(raw)
    old.exact_keys(manifest, ("contract", "rows", "candidate_width", "payload", "sources", "expected", "bootstrap", "numpy_version", "scope"), "manifest")
    old.require(manifest["contract"] == CONTRACT and manifest["rows"] == ROWS and type(manifest["rows"]) is int
                and manifest["candidate_width"] == 1024, "unsupported cohort")
    old.require(manifest["scope"] == SCOPE and manifest["bootstrap"] == BOOTSTRAP and manifest["numpy_version"] == "2.5.4", "unsupported method")
    # Large source rank arrays are provenance only and are never opened.
    validate_sources(manifest["sources"])
    payload_raw = old.bounded_read(data / "rankings.json", LIMIT-len(raw))
    old.require(manifest["payload"] == old.descriptor("rankings.json", payload_raw), "payload pin mismatch")
    got = calculate(old.decode_json(payload_raw), bootstrap)
    expected = manifest["expected"]
    old.exact_keys(expected, ("per_arm", "main_table", "counts", "unique_users", "contrasts", "bootstrap"), "expected")
    error = compare(got, {key: expected[key] for key in got})
    return {"status": "PASS", "contract": CONTRACT, "rows": ROWS, **got,
            "scalar_max_abs_error": error, "bootstrap_recomputed": bootstrap,
            "scope": dict(SCOPE, ci=SCOPE["ci"] if bootstrap else "not recomputed; run bootstrap command"),
            "inputs": {"manifest": old.descriptor("manifest.json", raw), "rankings": manifest["payload"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "recompute", "bootstrap"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        started = time.monotonic()
        result = replay(args.data, args.command == "bootstrap")
        result["elapsed_seconds"] = time.monotonic()-started
        if args.output:
            old.require(args.output.resolve().parent != args.data.resolve(), "output must be outside input directory")
            old.write_new_json(args.output, result)
        print(old.encoded(result).decode("utf-8"), end="")
        return 0
    except (old.ValidationError, OSError, ImportError, ValueError) as error:
        print(f"saved_metrics64k: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
