"""Owner-only offline export from the exact historical supervised/matched pins.

Root must admit this operation before invocation. It reads only two pinned
request JSONs, their evaluation metadata, the historical statistics source, and
selected int/bool arrays. No NumPy, pickle, model, scores, user IDs or network.
The root coordinator owns resource limits and an external execution receipt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import stat
import struct
import sys

import saved_metrics as sm

ARRAY_LIMIT = 20 * sm.LIMIT
TOTAL_LIMIT = 10 * sm.LIMIT
DTYPES = {"int64": ("<i8", "<q", 8), "int32": ("<i4", "<i", 4),
          "bool": ("|b1", "B", 1)}


def read_pinned_bytes(path, size, remaining):
    """Check the size and remaining budget before opening; never read ahead."""
    sm.require(type(size) is int and 0 < size <= min(ARRAY_LIMIT, remaining),
               "source exceeds remaining read budget")
    before = path.stat()
    sm.require(stat.S_ISREG(before.st_mode), "source is not a regular file")
    sm.require(before.st_size == size, "source size pin mismatch")
    # Unbuffered exact-size reads also bound OS reads if the file grows after stat.
    # A post-open fstat closes the replacement window before consuming any bytes.
    with path.open("rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        sm.require(stat.S_ISREG(opened.st_mode) and opened.st_size == size,
                   "source changed before read")
        raw = stream.read(size)
        sm.require(len(raw) == size and os.fstat(stream.fileno()).st_size == size,
                   "source changed during read")
    return raw


def read_npy(raw, dtype, shape):
    """Bounded NPY v1/v2 reader for explicit little-endian int/C-order bool."""
    sm.require(type(raw) is bytes and 10 <= len(raw) <= ARRAY_LIMIT, "invalid NPY size")
    sm.require(raw[:6] == b"\x93NUMPY", "invalid NPY magic")
    version = raw[6:8]
    sm.require(version in (b"\x01\x00", b"\x02\x00"), "unsupported NPY version")
    prefix = 10 if version == b"\x01\x00" else 12
    sm.require(len(raw) >= prefix, "truncated NPY prefix")
    header_size = struct.unpack("<H" if prefix == 10 else "<I", raw[8:prefix])[0]
    sm.require(0 < header_size <= 4096 and len(raw) >= prefix + header_size, "invalid NPY header length")
    header_raw = raw[prefix:prefix + header_size]
    sm.require(header_raw.endswith(b"\n"), "NPY header lacks newline")
    try:
        tree = ast.parse(header_raw.decode("ascii").strip(), mode="eval")
        sm.require(type(tree.body) is ast.Dict, "NPY header must be a dict")
        keys = [ast.literal_eval(key) for key in tree.body.keys]
        sm.require(len(keys) == len(set(keys)), "duplicate NPY header key")
        header = ast.literal_eval(tree)
    except (UnicodeError, SyntaxError, ValueError, TypeError, RecursionError) as error:
        raise sm.ValidationError("invalid literal NPY header") from error
    sm.exact_keys(header, ("descr", "fortran_order", "shape"), "NPY header")
    sm.require(dtype in DTYPES, "unsupported array dtype")
    descr, unpack, itemsize = DTYPES[dtype]
    sm.require(header["descr"] == descr and header["fortran_order"] is False,
               "unsupported NPY dtype/endianness/order")
    sm.require(type(header["shape"]) is tuple and all(type(n) is int and n > 0 for n in header["shape"])
               and list(header["shape"]) == shape, "NPY shape mismatch")
    count = 1
    for dimension in shape:
        count *= sm.integer(dimension, 1, sm.ROWS, "array dimension")
    payload = raw[prefix + header_size:]
    sm.require(len(payload) == count * itemsize, "NPY payload length mismatch")
    result = [item[0] for item in struct.iter_unpack(unpack, payload)]
    if dtype == "bool":
        sm.require(all(item in (0, 1) for item in result), "invalid NPY boolean representation")
        return [bool(item) for item in result]
    return result


def safe_source(root, relative):
    """Every source stays inside the supplied artifact root, without links."""
    parts = relative.split("/")
    sm.require(all(p and p not in (".", "..") and ":" not in p and "\\" not in p for p in parts),
               "unsafe source path")
    path = root
    for part in parts:
        path = path / part
        sm.require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(),
                   "linked source paths are not supported")
    sm.require(path.resolve().is_relative_to(root.resolve()), "source escapes artifact root")
    return path


def historical_pin(request, relative):
    sm.require(type(request) is dict and type(request.get("pins")) is dict, "invalid historical request")
    # Historical Windows paths are never executed or used as current paths.
    matches = [sha for path, sha in request["pins"].items()
               if path.replace("\\", "/").endswith("/" + relative)]
    sm.require(len(matches) == 1, "historical request does not pin " + relative)
    return matches[0]


def build_export(root):
    sources = {}
    consumed = 0

    def pinned(role, expected=None):
        nonlocal consumed
        if expected is None:
            relative, size, sha = sm.FIXED_SOURCES[role]
        else:
            relative, size, sha = expected
        item = {"path": relative, "bytes": size, "sha256": sha}
        sm.validate_descriptor(item, role)
        remaining = TOTAL_LIMIT - consumed
        sm.require(size <= remaining, "total source read budget exceeded")
        raw = read_pinned_bytes(safe_source(root, relative), size, remaining)
        sm.require(len(raw) == size and hashlib.sha256(raw).hexdigest() == sha, "source pin mismatch: " + role)
        consumed += len(raw)
        sm.require(consumed <= TOTAL_LIMIT, "total source read budget exceeded")
        sources[role] = item
        return raw

    requests = {family: sm.decode_json(pinned("historical-request/" + family)) for family in sm.BASES}
    pinned("historical-statistics-producer")  # Hash only: never imports or executes it.
    producer_relative, _, producer_sha = sm.FIXED_SOURCES["historical-statistics-producer"]
    data = {}
    expected_per_arm = {}
    for family in sm.BASES:
        evaluation_role = "evaluation/" + family
        relative, _, expected_sha = sm.FIXED_SOURCES[evaluation_role]
        sm.require(historical_pin(requests[family], relative) == expected_sha, "evaluation is not request-pinned")
        sm.require(historical_pin(requests[family], producer_relative) == producer_sha, "statistics source is not request-pinned")
        evaluation = sm.decode_json(pinned(evaluation_role))
        sm.require(evaluation.get("rows") == evaluation.get("metric_denominator") == sm.ROWS
                   and evaluation.get("candidate_width") == sm.WIDTH, "unsupported evaluation cohort")
        sm.require(evaluation.get("tie_break") == "score descending, raw item ID ascending",
                   "unexpected saved ranking convention")
        sm.require(type(evaluation.get("arrays")) is dict, "evaluation has no array descriptors")
        arrays = {}
        for name in sm.source_array_names(family):
            entry = evaluation["arrays"].get(name)
            sm.exact_keys(entry, ("path", "bytes", "sha256", "dtype", "shape"), "array " + name)
            sm.require(entry["path"] == name + ".npy", "unexpected source array filename")
            is_top = name.endswith("_top10")
            dtype = "int32" if is_top else ("bool" if name in ("known_target", "candidate_coverage") else "int64")
            shape = [sm.ROWS, 10] if is_top else [sm.ROWS]
            sm.require(entry["dtype"] == dtype and entry["shape"] == shape, "unexpected source array schema")
            array_relative = sm.BASES[family] + "/" + entry["path"]
            sm.require(historical_pin(requests[family], array_relative) == entry["sha256"],
                       "array is not historical-request-pinned")
            raw = pinned("array/" + family + "/" + name, (array_relative, entry["bytes"], entry["sha256"]))
            arrays[name] = read_npy(raw, dtype, shape)
        sm.require(len(set(arrays["row_index"])) == sm.ROWS, "duplicate source row_index")
        sm.require(all(row >= 0 for row in arrays["row_index"]), "negative source row_index")
        sm.require(all(not coverage or known for known, coverage in
                       zip(arrays["known_target"], arrays["candidate_coverage"])), "unknown covered target")
        if data:
            for name in sm.SHARED_ARRAYS + ("C_top10",):
                sm.require(arrays[name] == data["supervised"][name], "cross-family alignment mismatch: " + name)
        for name in arrays:
            if not name.endswith("_top10"):
                continue
            arm = name[:-6]
            metric_values = evaluation["metrics"][arm]
            expected_values = {metric: metric_values[metric] for metric in sm.METRICS}
            if arm in expected_per_arm:
                sm.require(expected_values == expected_per_arm[arm], "inconsistent C expected metrics")
            expected_per_arm[arm] = expected_values
        data[family] = arrays

    base = data["supervised"]
    eligibility = [2 if covered else 1 if known else 0 for known, covered in
                   zip(base["known_target"], base["candidate_coverage"])]
    ranks = {}
    for arm in sm.ARMS:
        family = "supervised" if arm in sm.ARMS[:4] else "matched"
        top = data[family][arm + "_top10"]
        result = []
        for i, (target, flag) in enumerate(zip(base["target_token"], eligibility)):
            items = top[10 * i:10 * (i + 1)]
            sm.require(len(set(items)) == 10 and all(item >= 3 for item in items), "invalid saved top10")
            sm.require(flag == 0 or target >= 3, "invalid known target token")
            sm.require(flag == 2 or target not in items, "ineligible target occurs in saved top10")
            result.append(items.index(target) + 1 if flag == 2 and target in items else 0)
        ranks[arm] = result
    payload = {"contract": sm.CONTRACT, "eligibility": eligibility, "ranks": ranks}
    got = sm.calculate(payload)
    expected_main = {arm: expected_per_arm[arm].copy() for arm in sm.MAIN_ARMS[:4]}
    for kind in sm.MAIN_ARMS[4:]:
        expected_main[kind] = {metric: sum(expected_per_arm[f"{kind}-{seed}"][metric] for seed in sm.SEEDS) / 3
                               for metric in sm.METRICS}
    expected = {"per_arm": expected_per_arm, "main_table": expected_main}
    sm.compare_expected(got, expected)
    # Include exact export implementation provenance, without requiring source at replay time.
    exporter_path = Path(__file__)
    sources["exporter"] = sm.descriptor("reproduction/export_saved_metrics.py", sm.bounded_read(exporter_path, sm.LIMIT))
    payload_raw = sm.encoded(payload)
    manifest = {"contract": sm.CONTRACT, "rows": sm.ROWS, "candidate_width": sm.WIDTH,
                "payload": sm.descriptor("rankings.json", payload_raw), "counts": got["counts"],
                "sources": sources, "expected": expected, "scope": sm.SCOPE}
    sm.validate_manifest(manifest, payload_raw)
    sm.require(len(sm.encoded(manifest)) + len(payload_raw) <= sm.LIMIT, "export exceeds consumer budget")
    return manifest, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--root-admitted", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        sm.require(args.artifacts_root.is_dir(), "artifact root does not exist")
        sm.require(not args.output.exists(), "output already exists")
        manifest, payload = build_export(args.artifacts_root)
        # Complete all source checks and arithmetic before creating the new output.
        args.output.mkdir()
        sm.write_new_json(args.output / "rankings.json", payload)
        sm.write_new_json(args.output / "manifest.json", manifest)
        result = sm.replay(args.output)
        sys.stdout.write(sm.encoded(result).decode("utf-8"))
        return 0
    except (sm.ValidationError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f"export_saved_metrics: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
