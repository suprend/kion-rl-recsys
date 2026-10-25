"""Offline, stdlib replay of point metrics from a bounded saved-rank export.

Ranks are positions of the observed target in previously saved top10 outputs.
This does not sort scores, run inference, fit a model, or replay bootstrap CIs.
Source IDs are excluded from the export; reidentification was not assessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

CONTRACT = "kion-saved-ranks-point-metrics-v1"
ROWS = 4096
WIDTH = 1024
LIMIT = 1024 * 1024  # Combined manifest + payload, not per file.
SEEDS = (20261024, 20261025, 20261026)
ARMS = ("GPT", "A", "B", "C") + tuple(
    f"{kind}-{seed}" for kind in ("completion", "residual") for seed in SEEDS
)
METRICS = ("map10", "ndcg10", "recall10")
MAIN_ARMS = ("GPT", "A", "B", "C", "completion", "residual")
BASES = {
    "supervised": "three-head-262k-20261024-r1/evaluation",
    "matched": "c-kl-seeds-20261025-r1/policy/full-six-r1",
}
FIXED_SOURCES = {
    "evaluation/supervised": (
        BASES["supervised"] + "/evaluation.json", 18013,
        "ebc6cb8b8eb682298723aba8f1d24a3250c286b20e91c70f51e159e7ba000e56"),
    "evaluation/matched": (
        BASES["matched"] + "/evaluation.json", 41162,
        "cc7dd2372a5c80f49aed0d5374deff4cdda8aa739ce41531f5c13245fe760ba5"),
    "historical-request/supervised": (
        "tlab-audit-20261025-r1/operations/request-metrics-supervised.json", 5441,
        "0283bd7d49ad7727079f916c11840a389e9e5208c7dfce5b63bd4fd037fadfd1"),
    "historical-request/matched": (
        "tlab-audit-20261025-r1/operations/request-metrics-matched.json", 6262,
        "09e9fc934620310d7b99365b16b58631b45ee76b27133e97c777260a05ff5edf"),
    "historical-statistics-producer": (
        "tlab-audit-20261025-r1/probes/frozen-r2/statistics-proposal.py", 9841,
        "932b411ce57426b3e507b24214d0a7ebb7674bf06e90d11aab812e59ae932730"),
}
SHARED_ARRAYS = ("row_index", "target_token", "known_target", "candidate_coverage")
SCOPE = {
    "ranking": "positions from saved top10; score sorting is not recomputed",
    "table": "equal-weight mean of three fitted-policy metrics; no ensemble",
    "ci": "historical audit results only; bootstrap CIs are not recomputed",
    "ids": "source IDs and target tokens excluded; reidentification not assessed",
    "limits": "no inference, training, feature, temporal-leakage or causal validation",
}


class ValidationError(ValueError):
    """The input is outside the supported finite replay contract."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def exact_keys(value, names, label):
    require(type(value) is dict and set(value) == set(names), f"{label}: unexpected or missing keys")


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"{label}: invalid integer")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def decode_json(raw):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except ValidationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValidationError("malformed UTF-8 JSON") from error


def bounded_read(path, limit):
    require(not path.is_symlink(), "symlink inputs are not supported")
    require(not getattr(path, "is_junction", lambda: False)(), "junction inputs are not supported")
    require(path.is_file(), f"missing regular input: {path.name}")
    require(0 < path.stat().st_size <= limit, f"input empty or oversized: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit, f"input empty or oversized: {path.name}")
    return raw


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def descriptor(path, raw):
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def validate_descriptor(value, label):
    exact_keys(value, ("path", "bytes", "sha256"), label)
    path = value["path"]
    require(type(path) is str and 0 < len(path) < 240, f"{label}: invalid path")
    parts = PurePosixPath(path).parts
    require(parts and not path.startswith("/") and ":" not in path and "\\" not in path
            and all(part not in (".", "..") for part in parts)
            and PurePosixPath(path).as_posix() == path, f"{label}: unsafe path")
    integer(value["bytes"], 1, 20 * LIMIT, label + " bytes")
    require(type(value["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", value["sha256"]),
            f"{label}: invalid SHA256")


def source_array_names(family):
    arms = ("GPT", "A", "B", "C") if family == "supervised" else ("C",) + ARMS[4:]
    return SHARED_ARRAYS + tuple(arm + "_top10" for arm in arms)


def validate_sources(sources):
    roles = set(FIXED_SOURCES) | {"exporter"}
    for family in BASES:
        roles.update(f"array/{family}/{name}" for name in source_array_names(family))
    exact_keys(sources, roles, "sources")
    for role, item in sources.items():
        validate_descriptor(item, role)
        if role in FIXED_SOURCES:
            path, size, sha = FIXED_SOURCES[role]
            require(item == {"path": path, "bytes": size, "sha256": sha}, f"{role}: unpinned source")
        elif role == "exporter":
            require(item["path"] == "reproduction/export_saved_metrics.py", "unexpected exporter path")
        else:
            _, family, name = role.split("/")
            require(item["path"] == BASES[family] + "/" + name + ".npy", f"{role}: unexpected array path")


def validate_manifest(manifest, raw_payload):
    exact_keys(manifest, ("contract", "rows", "candidate_width", "payload", "counts",
                          "sources", "expected", "scope"), "manifest")
    require(manifest["contract"] == CONTRACT, "unsupported manifest contract")
    integer(manifest["rows"], ROWS, ROWS, "rows")
    integer(manifest["candidate_width"], WIDTH, WIDTH, "candidate width")
    validate_descriptor(manifest["payload"], "payload")
    require(manifest["payload"] == descriptor("rankings.json", raw_payload), "payload pin mismatch")
    exact_keys(manifest["counts"], ("known", "unknown", "offpool_known", "covered"), "counts")
    for name, value in manifest["counts"].items():
        integer(value, 0, ROWS, "count " + name)
    require(manifest["counts"] == {"known": 3974, "unknown": 122, "offpool_known": 616,
                                    "covered": 3358}, "unexpected cohort counts")
    require(manifest["scope"] == SCOPE, "unsupported replay scope")
    validate_sources(manifest["sources"])


def calculate(payload):
    """Compute all estimates without consulting manifest expected scalars."""
    exact_keys(payload, ("contract", "eligibility", "ranks"), "rankings")
    require(payload["contract"] == CONTRACT, "unsupported rankings contract")
    status = payload["eligibility"]
    require(type(status) is list and len(status) == ROWS, "invalid eligibility length")
    for value in status:
        integer(value, 0, 2, "eligibility")
    counts = {"unknown": status.count(0), "offpool_known": status.count(1),
              "covered": status.count(2), "known": ROWS - status.count(0)}
    exact_keys(payload["ranks"], ARMS, "ranks")
    per_arm = {}
    for arm in ARMS:
        ranks = payload["ranks"][arm]
        require(type(ranks) is list and len(ranks) == ROWS, f"{arm}: invalid rank length")
        for flag, rank in zip(status, ranks):
            integer(rank, 0, 10, arm + " rank")
            require(flag == 2 or rank == 0, f"{arm}: ineligible target must be a miss")
        # With exactly one held-out target, AP is reciprocal rank at a hit.
        per_arm[arm] = {
            "map10": math.fsum(1.0 / r for r in ranks if r) / ROWS,
            "ndcg10": math.fsum(1.0 / math.log2(r + 1) for r in ranks if r) / ROWS,
            "recall10": sum(r != 0 for r in ranks) / ROWS,
        }
    main_table = {arm: per_arm[arm].copy() for arm in MAIN_ARMS[:4]}
    for kind in MAIN_ARMS[4:]:
        main_table[kind] = {
            metric: math.fsum(per_arm[f"{kind}-{seed}"][metric] for seed in SEEDS) / len(SEEDS)
            for metric in METRICS
        }
    return {"counts": counts, "per_arm": per_arm, "main_table": main_table}


def compare_expected(got, expected):
    exact_keys(expected, ("per_arm", "main_table"), "expected")
    maximum = 0.0
    for group, arms in (("per_arm", ARMS), ("main_table", MAIN_ARMS)):
        exact_keys(expected[group], arms, "expected " + group)
        for arm in arms:
            exact_keys(expected[group][arm], METRICS, "expected metrics " + arm)
            for metric in METRICS:
                value = expected[group][arm][metric]
                require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                        f"invalid expected scalar: {arm}/{metric}")
                error = abs(got[group][arm][metric] - value)
                require(error <= 1e-12, f"metric mismatch: {arm}/{metric}")
                maximum = max(maximum, error)
    return maximum


def replay(data):
    require(not data.is_symlink() and not getattr(data, "is_junction", lambda: False)(),
            "symlink/junction data directories are not supported")
    require(data.is_dir(), "data directory does not exist")
    require({p.name for p in data.iterdir()} == {"manifest.json", "rankings.json"},
            "data directory must contain exactly manifest.json and rankings.json")
    raw_manifest = bounded_read(data / "manifest.json", LIMIT)
    raw_payload = bounded_read(data / "rankings.json", LIMIT - len(raw_manifest))
    manifest = decode_json(raw_manifest)
    validate_manifest(manifest, raw_payload)
    # The arithmetic always precedes any reading/comparison of expected scalars.
    got = calculate(decode_json(raw_payload))
    require(got["counts"] == manifest["counts"], "eligibility count mismatch")
    error = compare_expected(got, manifest["expected"])
    return {"contract": CONTRACT, "status": "PASS", "rows": ROWS,
            "candidate_width": WIDTH, **got, "scalar_max_abs_error": error,
            "inputs": {"manifest": descriptor("manifest.json", raw_manifest),
                       "rankings": descriptor("rankings.json", raw_payload)}, "scope": SCOPE}


def write_new_json(path, value):
    """Exclusive creation; no mkdir, overwrite or implicit default output."""
    with path.open("xb") as stream:
        stream.write(encoded(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "recompute"):
        child = sub.add_parser(command)
        child.add_argument("--data", type=Path, required=True)
        if command == "recompute":
            child.add_argument("--output", type=Path, help="exclusive new JSON result; default is stdout only")
    args = parser.parse_args(argv)
    try:
        result = replay(args.data)
        output = getattr(args, "output", None)
        if output is not None:
            require(output.resolve().parent != args.data.resolve(), "output must be outside the input directory")
            write_new_json(output, result)
        sys.stdout.write(encoded(result).decode("utf-8"))
        return 0
    except (ValidationError, OSError, OverflowError) as error:
        print(f"saved_metrics: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
