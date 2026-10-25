"""Root-launched CPU pilot/full query preparation. Import has no runtime effects."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import store_hook


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_progress(path, value):
    temporary = Path(path).with_suffix(".pending.json")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def checked(request):
    for path, expected in request["sources"].items():
        store_hook.checked_pin({"path": path, "sha256": expected})
    required = dict(request["sources"])
    for pin in request["metadata"].values():
        path = store_hook.checked_pin(pin)
        required[path.as_posix()] = pin["sha256"]
    if any(request["store_source_pins"].get(path) != expected for path, expected in required.items()):
        raise ValueError("Store bindings must include every request source/input metadata pin")


class Limits:
    def __init__(self, request):
        self.deadline = datetime.fromisoformat(request["deadline_utc"])
        if self.deadline.tzinfo is None:
            raise ValueError("Absolute UTC deadline required")
        self.output = Path(request["output"]).resolve()
        self.reserve = request["free_space_reserve_bytes"]
        self.stop_file = Path(request["stop_file"])
        self.started = time.monotonic()
        self.wall_seconds = request["wall_seconds"]
        if type(self.wall_seconds) is not int or not 1 <= self.wall_seconds <= 86400:
            raise ValueError("Explicit finite preparation wall_seconds required")

    def check(self):
        if self.stop_file.exists():
            raise InterruptedError("Root requested preparation stop")
        if time.monotonic() - self.started >= self.wall_seconds:
            raise TimeoutError("Preparation finite wall cap reached")
        if datetime.now(timezone.utc) >= self.deadline:
            raise TimeoutError("Preparation absolute deadline reached")
        if shutil.disk_usage(self.output.parent).free < self.reserve:
            raise OSError("Preparation free disk reserve breached")


def array_signature(value):
    """Dtype/shape/content signature for composed provider array tuples."""
    digest = hashlib.sha256()
    for array in value if isinstance(value, (tuple, list)) else [value]:
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def key_feed(digest, key):
    # Unambiguous streaming digest; no population of keys is kept in RAM.
    for number in key[:3]:
        digest.update(int(number).to_bytes(8, "little", signed=True))
    digest.update(len(key[3]).to_bytes(8, "little"))
    digest.update(key[3])


def first_groups(provider, train, count=12):
    args = SimpleNamespace(training_order="origin-grouped-v1", seed=2026,
        batch_size=128, accumulation=1, bucket_size=8192)
    schedule = train.TrainingSchedule(provider, args)
    result = []
    for offset, batches, index, origin in schedule.groups(0, 0):
        result.append({"group_index": int(index), "offset": int(offset), "origin": int(origin),
            "rows": [int(row) for batch in batches for row in batch]})
        if len(result) == count:
            break
    if len(result) != count:
        raise ValueError("Could not select first twelve corrected E1 groups")
    return result


def representative_rows(provider, split, *, each=64, chunk=65536):
    """Three origins, short/long rows, bounded selected lists; no global sort."""
    import numpy as np
    origins = provider.origin_keys(split)
    days = set()
    for start in range(0, len(origins), chunk):
        days.update(map(int, np.unique(origins[start:start + chunk])))
    days = sorted(days)
    targets = sorted({days[0], days[len(days) // 2], days[-1]})
    lengths = provider.splits[split]["lengths"]
    result = []
    for day in targets:
        short, long = [], []
        for start in range(0, len(origins), chunk):
            take = np.flatnonzero(origins[start:start + chunk] == day) + start
            # At most one scan chunk is converted to Python, not the full split.
            local = [(int(lengths[row]), int(row)) for row in take]
            short = sorted(short + local)[:each]
            long = sorted(long + local, key=lambda pair: (-pair[0], pair[1]))[:each]
        result.extend(row for _, row in short + long)
    return list(dict.fromkeys(result))


def pilot_selection(provider, train, request):
    groups = first_groups(provider, train)
    rows = {"train": list(dict.fromkeys(row for group in groups for row in group["rows"])), "val": []}
    for split in ("train", "val"):
        rows[split] = list(dict.fromkeys(rows[split] + representative_rows(provider, split)))
    if sum(map(len, rows.values())) > request["pilot_max_profiles"]:
        raise ValueError("Pilot row coverage exceeds explicit profile cap")
    return rows, groups


def iter_rows(provider, mode, selected):
    for split in ("train", "val"):
        values = selected[split] if mode == "pilot" else range(len(provider.splits[split]["y"]))
        for row in values:
            yield split, int(row)


def finalized_recovery(output, codec, provider, request):
    """Recover the finish-to-receipt gap without reopening a completed writer."""
    audit_path = output / "build-completed.json"
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["mode"] != request["mode"]
            or audit["store_source_pins_sha256"] != store_hook.canonical_sha(request["store_source_pins"])):
        raise ValueError("Finalized recovery source/mode binding changed")
    try:
        reader = codec.QueryStoreReader(output / "queries.sqlite3",
            expected_schema_hash=provider.personal.schema_hash, expected_source_pins=request["store_source_pins"])
    except ValueError:
        # A writing database can have the ready-to-finalize audit immediately
        # before finish. Ordinary writer resume replays all keys in that case.
        # A corrupt/completed incompatible store still fails Writer admission.
        return None
    try:
        if (reader.counts != {key: audit["store"][key] for key in reader.counts}
                or reader.binding["max_total_bytes"] != request["max_total_bytes"]
                or reader.binding["free_space_reserve_bytes"] != request["free_space_reserve_bytes"]):
            raise ValueError("Finalized recovery counts/policy differ")
        return audit
    finally:
        reader.close()


def bounded_store_sha(path, limits):
    before = Path(path).stat()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            limits.check()
            digest.update(block)
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Finalized database changed while hashing")
    return digest.hexdigest()


def candidate_batch_rows(provider, width, *, preferred_rows=128):
    """Respect every original provider row/cell/float32-output byte bound."""
    dimension = len(provider.feature_columns["candidate"])
    if width < 1 or dimension < 1 or preferred_rows < 1:
        raise ValueError("Positive parity candidate width/dimension/row preference required")
    count = min(preferred_rows, 128, provider.max_batch_rows,
        provider.max_pair_cells // width, provider.max_pair_bytes // (width * dimension * 4))
    if count < 1:
        raise ValueError("One parity candidate row exceeds admitted provider bounds")
    return int(count)


def parity(provider, original, prepared_method, codec, store_path, pins, selected, limits):
    """Complete batch and C4 train/C128 val bytes through explicit AST _query."""
    import numpy as np
    reader = codec.QueryStoreReader(store_path, expected_schema_hash=provider.personal.schema_hash,
        expected_source_pins=pins)
    original_cls = type(provider)
    profile_type = importlib.import_module("kion_personal_candidates").QueryProfiles
    counts = {}
    query = store_hook.reader_query(original_cls._query, prepared_method, reader, profile_type, counters=counts)
    adapted = type("PilotPreparedProvider", (original_cls,), {"_query": query})
    results = []
    try:
        for split in ("train", "val"):
            # Pilot checks every selected row; full checks the bounded first12
            # plus representative fixture, never retains full split row lists.
            rows = selected[split]
            width = 4 if split == "train" else 128
            step = candidate_batch_rows(provider, width, preferred_rows=64 if split == "val" else 128)
            for start in range(0, len(rows), step):
                limits.check()
                take = np.asarray(rows[start:start + step], np.int64)
                candidates = 3 + ((provider.splits[split]["y"][take, None] + np.arange(width))
                    % (provider.metadata["vocab_size"] - 3))
                snapshots = []
                durations = []
                for cls in (original_cls, adapted):
                    provider.__class__ = cls
                    provider._profiles = OrderedDict()
                    before = time.perf_counter()
                    batch = provider.batch(take, split)
                    batch_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    features = provider.candidate_features(take, candidates, split)
                    durations.append({"batch_seconds": batch_seconds,
                        "candidate_seconds": time.perf_counter() - before})
                    snapshots.append((array_signature(batch), array_signature(features)))
                if snapshots[0] != snapshots[1]:
                    raise ValueError("Complete prepared batch/candidate byte parity failed")
                results.append({"split": split, "rows_sha256": array_signature(take), "count": len(take),
                    "candidate_width": width, "batch_sha256": snapshots[0][0], "candidate_sha256": snapshots[0][1]})
                results[-1]["original_seconds"], results[-1]["prepared_seconds"] = durations
        if counts.get("store_misses", 0) or counts.get("fallback_prepares", 0):
            raise ValueError("Pilot parity unexpectedly missed a required prepared row")
        return {"state": "exact_bytes_equal", "records": results, "query": counts, "reader": reader.stats()}
    finally:
        provider.__class__ = original_cls
        provider._profiles = OrderedDict()
        reader.close()


def validate_request(request):
    mode = request["mode"]
    if request.get("phase") != {"pilot": "cpu-pilot", "full": "cpu-build"}.get(mode):
        raise ValueError("Explicit matching CPU phase/mode required")
    if request.get("expected_rows") != store_hook.EXPECTED_ROWS:
        raise ValueError("Only exact corrected E1 train/val rows are admitted")
    cap = request["max_total_bytes"]
    if type(cap) is not int or not 0 < cap <= (1 << 30 if mode == "pilot" else 30_000_000_000):
        raise ValueError("Explicit total disk cap exceeds pilot/full bound")
    reserve = request["free_space_reserve_bytes"]
    if type(reserve) is not int or reserve < 8 << 30:
        raise ValueError("At least 8GiB free disk reserve is required")
    if mode == "pilot" and not 1536 <= request["pilot_max_profiles"] <= 4096:
        raise ValueError("Pilot row cap must cover first12 and be <=4096")
    if mode == "full":
        decision_path = store_hook.checked_pin(request["full_build_decision"])
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        pilot_path = store_hook.checked_pin(decision["pilot_result"])
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if (decision.get("state") != "GO" or decision.get("mode") != "full"
                or decision.get("store_source_pins_sha256") != store_hook.canonical_sha(request["store_source_pins"])
                or pilot.get("state") != "completed" or pilot.get("mode") != "pilot"
                or pilot.get("parity", {}).get("state") != "exact_bytes_equal"):
            raise ValueError("Full build requires a pinned affirmative root pilot decision")


def run(request):
    validate_request(request)
    checked(request)
    output = Path(request["output"]).resolve()
    resume = request.get("resume", False)
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    output.mkdir(parents=True, exist_ok=resume)
    result_path, progress_path = output / "result.json", output / "progress.json"
    if result_path.exists():
        raise FileExistsError("Completed/previous result cannot be overwritten")
    limits = Limits(request)
    limits.check()
    if __import__("os").environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU preparation requires CUDA_VISIBLE_DEVICES=-1")
    common = Path(request["common_scripts"]).resolve()
    if (common / "kion_ready_runtime.py").as_posix() not in request["sources"]:
        raise ValueError("Frozen configure_paths source pin required")
    sys.path.insert(0, common.as_posix())
    import kion_ready_runtime
    kion_ready_runtime.configure_paths()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import numpy as np
    import kion_complete_pack as original
    import kion_complete_train as train
    codec = store_hook.load_pinned(request["codec_source"], "_preparation_codec")
    prepared_method = store_hook.prepared_query_method(original, request["query_method_source"])
    began = time.perf_counter()
    provider = original.CompleteProvider.load(request["data_dir"], request["feature_dir"],
        scope_decision=request["scope_decision"])
    loaded = time.perf_counter() - began
    writer = None
    store_path = output / "queries.sqlite3"
    try:
        for split, count in store_hook.EXPECTED_ROWS.items():
            if len(provider.splits[split]["y"]) != count:
                raise ValueError("Provider corrected E1 split count mismatch")
        if provider.personal is None:
            raise ValueError("No admitted personal profile index")
        selected, groups = pilot_selection(provider, train, request if request["mode"] == "pilot" else {"pilot_max_profiles": 4096})
        selection = {"first12_epoch0_groups": groups, "selected_rows": selected}
        if resume:
            if json.loads((output / "coverage-selection.json").read_text(encoding="utf-8")) != selection:
                raise ValueError("Resume row-selection binding changed")
        else:
            write_new(output / "coverage-selection.json", selection)
        recovered = finalized_recovery(output, codec, provider, request) if resume else None
        if recovered is None:
            writer = codec.QueryStoreWriter(store_path, schema_hash=provider.personal.schema_hash,
                source_pins=request["store_source_pins"], max_total_bytes=request["max_total_bytes"],
                free_space_reserve_bytes=request["free_space_reserve_bytes"], resume=resume)
        scanned = {"train": 0, "val": 0}
        digests = {split: hashlib.sha256() for split in scanned}
        inserted = duplicates = 0
        compute_seconds = contains_seconds = put_seconds = 0.
        started = time.perf_counter()
        for split, row in (() if recovered is not None else iter_rows(provider, request["mode"], selected)):
            limits.check()
            key = store_hook.query_key(provider, split, row)
            key_feed(digests[split], key)
            scanned[split] += 1
            before = time.perf_counter()
            present = writer.contains(key)
            contains_seconds += time.perf_counter() - before
            if present:
                duplicates += 1
            else:
                before = time.perf_counter()
                state = provider._query(split, row)
                compute_seconds += time.perf_counter() - before
                before = time.perf_counter()
                if not writer.put(key, state["personal"], state["user"]):
                    raise ValueError("Unexpected duplicate after single-owner contains miss")
                put_seconds += time.perf_counter() - before
                inserted += 1
            total = sum(scanned.values())
            if total % 128 == 0:
                stats = writer.checkpoint()
                progress = {"state": "writing", "mode": request["mode"], "split_rows": scanned,
                    "session_inserted": inserted, "session_duplicates": duplicates,
                    "elapsed_seconds": time.perf_counter() - began, "store": stats,
                    "store_source_pins_sha256": store_hook.canonical_sha(request["store_source_pins"])}
                write_progress(progress_path, progress)
                print(json.dumps({"event": "preparation_progress", **progress}), flush=True)
        limits.check()
        checked(request)
        build_seconds = time.perf_counter() - started
        if recovered is None:
            stats = writer.checkpoint()
            audit = {"state": "ready_to_finalize", "mode": request["mode"], "split_rows": scanned,
                "split_ordered_exact_key_sha256": {split: digest.hexdigest() for split, digest in digests.items()},
                "session_inserted": inserted, "session_duplicates": duplicates, "store": stats,
                "compute_seconds": compute_seconds, "contains_seconds": contains_seconds,
                "put_seconds": put_seconds, "build_seconds": build_seconds,
                "store_source_pins_sha256": store_hook.canonical_sha(request["store_source_pins"])}
            write_progress(output / "build-completed.json", audit)
            stats = writer.finish()
            writer = None
        else:
            stats = recovered["store"]
            scanned = recovered["split_rows"]
            inserted, duplicates = recovered["session_inserted"], recovered["session_duplicates"]
            compute_seconds, contains_seconds, put_seconds = (recovered[key] for key in
                ("compute_seconds", "contains_seconds", "put_seconds"))
            build_seconds = recovered["build_seconds"]
        limits.check()
        parity_result = parity(provider, original, prepared_method, codec, store_path,
            request["store_source_pins"], selected, limits)
        checked(request)
        store_sha256 = bounded_store_sha(store_path, limits)
        if torch.cuda.is_initialized():
            raise ValueError("CPU preparation unexpectedly initialized CUDA")
        unique = stats["unique_count"]
        # Extrapolation is explicitly a pilot measurement, never auto-GO.
        total_rows = sum(store_hook.EXPECTED_ROWS.values())
        estimated_all_unique_payload = stats["stored_payload_bytes"] / max(unique, 1) * total_rows
        result = {"state": "completed", "mode": request["mode"], "phase": request["phase"],
            "store_path": store_path.as_posix(), "store_source_pins": request["store_source_pins"],
            "store_sha256": store_sha256, "recovered_finalized_store": recovered is not None,
            "store_source_pins_sha256": store_hook.canonical_sha(request["store_source_pins"]),
            "personal_schema_hash": provider.personal.schema_hash, "split_rows": scanned,
            "split_ordered_exact_key_sha256": (recovered["split_ordered_exact_key_sha256"] if recovered is not None
                else {split: digest.hexdigest() for split, digest in digests.items()}),
            "selection_path": (output / "coverage-selection.json").as_posix(),
            "selection_sha256": store_hook.file_sha(output / "coverage-selection.json"),
            "session_inserted": inserted, "session_duplicates": duplicates, "resumed": resume,
            "store": stats, "provider_load_seconds": loaded, "build_seconds": build_seconds,
            "compute_seconds": compute_seconds, "contains_seconds": contains_seconds,
            "put_seconds": put_seconds, "parity": parity_result,
            "pilot_all_rows_unique_payload_estimate_bytes": estimated_all_unique_payload,
            "pilot_all_rows_unique_database_estimate_bytes": stats["database_bytes"] / max(unique, 1) * total_rows,
            "pilot_all_rows_unique_db_plus_journal_estimate_bytes": 2 * stats["database_bytes"] / max(unique, 1) * total_rows + (1 << 20),
            "estimate_limit": "Sample payload extrapolation; excludes index/freepages/journal and nonrepresentative profiles; no auto full build",
            "elapsed_seconds": time.perf_counter() - began, "CUDA_initialized": False}
        write_new(result_path, result)
        print(json.dumps({"event": "completed", "result": result_path.as_posix(), "store": stats}), flush=True)
        return result
    except BaseException as error:
        if writer is not None:
            # Root's outer hard wall remains authoritative. Cooperative stop
            # leaves committed records in writing state, never a complete store.
            stats = writer.checkpoint()
            write_progress(progress_path, {"state": "incomplete", "mode": request["mode"],
                "error_type": type(error).__name__, "error": str(error), "store": stats,
                "store_source_pins_sha256": store_hook.canonical_sha(request["store_source_pins"])})
        raise
    finally:
        if writer is not None:
            writer.close()
        provider.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    run(request)


if __name__ == "__main__":
    main()
