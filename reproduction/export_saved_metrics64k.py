"""Export anonymous target ranks and cluster equality from frozen local artifacts.

Maintainer-only: streams hashes and mmaps saved arrays, never loads models or
scores, never trains or runs inference. Output must be a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import saved_metrics as old
import saved_metrics64k as sm


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_pin(path, relative):
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def load_array(np, path, entry, sources, role, relative):
    old.require(path.is_file() and not path.is_symlink(), "missing regular array")
    old.require(path.stat().st_size == entry["bytes"], "array size mismatch: " + role)
    pin = source_pin(path, relative)
    old.require(pin["sha256"] == entry["sha256"], "array hash mismatch: " + role)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    old.require(list(array.shape) == entry["shape"] and str(array.dtype) == entry["dtype"], "array schema mismatch")
    sources[role] = pin
    return array


def extract_target_ranks(np, pool, target, known, covered, ranks, top):
    """Bounded rows, exact permutation and saved-top10 agreement checks.

    A tie's saved position is authoritative; numeric item IDs are not exported.
    """
    count = len(target)
    output = np.zeros(count, dtype=np.uint16)
    width = pool.shape[1]
    for first in range(0, count, 128):
        stop = min(first+128, count)
        p = pool[first:stop]
        r = ranks[first:stop]
        old.require(np.all(np.sort(r, axis=1) == np.arange(1, width+1)), "rank row is not a permutation")
        old.require(np.all(np.diff(np.sort(p, axis=1), axis=1) != 0), "duplicate candidates")
        matches = (p == target[first:stop, None]) & known[first:stop, None]
        old.require(np.array_equal(matches.any(1), covered[first:stop]), "coverage mismatch")
        output[first:stop] = np.where(matches, r, 0).sum(1)
        # The rank permutation defines a unique order even for equal scores.
        order = np.argsort(r, axis=1)[:, :top.shape[1]]
        old.require(np.array_equal(np.take_along_axis(p, order, axis=1), top[first:stop]), "saved top10/rank disagreement")
    return output.tolist()


def build_export(artifacts, store):
    np = sm.numpy()
    sources = {}
    records = {}
    for name in ("evaluation.json", "analysis.json", "ranks.json", "analyze64k.py", "export_ranks.py"):
        path = artifacts / name
        sources[name] = source_pin(path, "eval64k-20261025-r1/" + name)
        if name.endswith(".json"):
            records[name] = old.decode_json(old.bounded_read(path, sm.LIMIT))
    original = artifacts.parent / "three-head-262k-20261024-r1/analyze.py"
    sources["original_bootstrap"] = source_pin(original, "three-head-262k-20261024-r1/analyze.py")
    evaluation, analysis, ranks = (records[k] for k in ("evaluation.json", "analysis.json", "ranks.json"))
    evaluation_sha = sources["evaluation.json"]["sha256"]
    old.require(evaluation["state"] == "completed" and evaluation["rows"] == sm.ROWS and tuple(evaluation["arms"]) == sm.ARMS, "unsupported evaluation")
    old.require(analysis["evaluation_sha256"] == ranks["evaluation_sha256"] == evaluation_sha
                and sha(store / "full/evaluation.json") == evaluation_sha, "evaluation provenance mismatch")
    old.require(analysis["analysis_source_sha256"] == sources["analyze64k.py"]["sha256"]
                and analysis["original_bootstrap_source"]["sha256"] == sources["original_bootstrap"]["sha256"]
                and ranks["source_sha256"] == sources["export_ranks.py"]["sha256"], "producer provenance mismatch")
    old.require(analysis["bootstrap"] == sm.BOOTSTRAP and analysis["numpy_version"] == "2.5.4", "unsupported bootstrap")
    old.require(ranks["state"] == "completed" and ranks["rank_base"] == 1
                and ranks["tie_break"] == "score descending, raw item ID ascending", "unsupported ranks")
    arrays = {}
    for name in ("user_id", "target_token", "known_target", "candidate_coverage", "pool_ids"):
        arrays[name] = load_array(np, store / "full" / (name + ".npy"), evaluation["arrays"][name],
                                  sources, "array/"+name, "eval64k-20261025-r1/full/"+name+".npy")
    known, covered = arrays["known_target"], arrays["candidate_coverage"]
    old.require(not np.any(covered & ~known), "unknown covered target")
    _, inverse = np.unique(arrays["user_id"], return_inverse=True)
    payload = {"contract": sm.CONTRACT, "clusters": inverse.tolist(),
               "eligibility": np.where(covered, 2, np.where(known, 1, 0)).tolist(), "ranks": {}}
    for arm in sm.ARMS:
        rank = load_array(np, store / "ranks" / (arm+"_ranks.npy"), ranks["arrays"][arm], sources,
                          "ranks/"+arm, "eval64k-20261025-r1/ranks/"+arm+"_ranks.npy")
        name = arm+"_top10"
        top = load_array(np, store / "full" / (name+".npy"), evaluation["arrays"][name], sources,
                         "array/"+name, "eval64k-20261025-r1/full/"+name+".npy")
        payload["ranks"][arm] = extract_target_ranks(np, arrays["pool_ids"], arrays["target_token"], known, covered, rank, top)
        rank._mmap.close()
        top._mmap.close()
    for array in arrays.values():
        array._mmap.close()
    got = sm.calculate(payload, bootstrap=True)
    expected = {"per_arm": {a: {m: evaluation["metrics"][a][m] for m in sm.METRICS} for a in sm.ARMS},
                "counts": {"known": 63790, "unknown": 1746, "offpool_known": 10183, "covered": 53607},
                "unique_users": analysis["unique_users"], "contrasts": analysis["contrasts"], "bootstrap": analysis["bootstrap"]}
    expected["main_table"] = {a: expected["per_arm"][a] for a in ("GPT", "A", "B", "C", "RRF")}
    for family in ("completion", "residual"):
        expected["main_table"][family] = {m: sum(expected["per_arm"][f"{family}-{seed}"][m] for seed in old.SEEDS)/3 for m in sm.METRICS}
    sm.compare(got, expected)
    for name in ("export_saved_metrics64k.py", "saved_metrics64k.py"):
        sources[name] = source_pin(Path(__file__).with_name(name), "reproduction/"+name)
    sm.validate_sources(sources)
    raw = old.encoded(payload)
    manifest = {"contract": sm.CONTRACT, "rows": sm.ROWS, "candidate_width": 1024,
                "payload": old.descriptor("rankings.json", raw), "sources": sources,
                "expected": expected, "numpy_version": "2.5.4", "bootstrap": sm.BOOTSTRAP, "scope": sm.SCOPE}
    old.require(len(raw)+len(old.encoded(manifest)) <= sm.LIMIT, "export too large")
    return manifest, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        started = time.monotonic()
        old.require(not args.output.exists(), "output already exists")
        manifest, payload = build_export(args.artifacts, args.store)
        args.output.mkdir()
        old.write_new_json(args.output / "manifest.json", manifest)
        old.write_new_json(args.output / "rankings.json", payload)
        print(old.encoded({"status": "PASS", "rows": sm.ROWS, "payload_bytes": manifest["payload"]["bytes"],
                           "elapsed_seconds": time.monotonic()-started}).decode(), end="")
        return 0
    except (old.ValidationError, OSError, ImportError, ValueError, KeyError) as error:
        print(f"export_saved_metrics64k: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
