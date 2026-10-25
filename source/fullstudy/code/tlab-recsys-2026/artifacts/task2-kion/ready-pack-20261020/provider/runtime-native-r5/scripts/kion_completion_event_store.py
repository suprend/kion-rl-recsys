"""Deduplicated completion audit events and exact symbolic H63 query pointers.

One raw retained user/item event has one strict event-relative baseline. This
module never feeds diagnostic fields to the model and never opens test.npz.
Probe stores are explicitly inadmissible for full production coverage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_completion_diagnostics as diagnostics
history = diagnostics.history


CONTRACT = "kion-completion-event-store-v1"
QUERY_JOIN = "exact-immutable-user-day-raw-item-H63-v1"
FULL_QA = "kion-completion-event-store-qa-v1"
FLAG_BITS = {"no_item_support": 0, "no_type_support": 1, "no_global_support": 2,
    "prior_only": 3, "type_unknown": 4, "used_c0": 5, "baseline_valid": 6,
    "scarce": 7, "residual_valid": 8}
EVENT_DTYPES = {"raw_row_index": "int64", "user_id": "int64", "raw_item_id": "int64", "date": "int32",
    "token": "int32", "outcome": "float32", "parser_status": "uint8", "baseline": "float64",
    "residual": "float64", "n_item": "int32", "n_type": "int32", "n_global": "int32",
    "fallback_level": "uint8", "flags": "uint16"}
BYTES_PER_EVENT = sum(np.dtype(value).itemsize for value in EVENT_DTYPES.values())
ROW_KEYS = ("x", "lengths", "y", "user_id", "target_item_id", "target_date")
canonical_hash, file_hash = diagnostics.canonical_hash, diagnostics.file_hash


def _pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": file_hash(path)}


def event_columns(audit, raw_row_indices):
    """Compute unique historical facts; raw UNK identity is retained separately."""
    h = audit.history
    indices = np.asarray(raw_row_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= len(h.users)) or np.any(indices[1:] <= indices[:-1])
            or np.any(h.days[indices] >= h.fit_cutoff)):
        raise ValueError("Event rows must be unique sorted raw indices strictly before the role cap")
    baseline = audit.baseline_rows(h.users[indices], h.items[indices], h.days[indices])
    valid = h.statuses[indices] == 0
    flags = np.zeros(len(indices), np.uint16)
    for name, bit in FLAG_BITS.items():
        values = valid if name == "residual_valid" else (np.ones(len(indices), bool)
            if name == "baseline_valid" else baseline[name])
        flags |= np.asarray(values, np.uint16) << bit
    values = {"raw_row_index": indices, "user_id": h.users[indices], "raw_item_id": h.items[indices],
        "date": h.days[indices], "token": h.tokens[indices], "outcome": h.outcomes[indices],
        "parser_status": h.statuses[indices], "baseline": baseline["baseline"],
        "residual": np.where(valid, h.outcomes[indices] - baseline["baseline"], 0.),
        "n_item": baseline["n_item"], "n_type": baseline["n_type"], "n_global": baseline["n_global"],
        "fallback_level": baseline["fallback_level"], "flags": flags}
    for name in ("n_item", "n_type", "n_global"):
        if np.any(values[name] > np.iinfo(np.int32).max):
            raise ValueError("Support exceeds the declared int32 event-store schema")
    return {name: np.asarray(values[name], dtype=dtype) for name, dtype in EVENT_DTYPES.items()}


def write_probe_store(audit, output, metadata_path, raw_row_indices, *, prepared_pins=None,
                      row_keys_sha256=None):
    """Persist a bounded probe; cannot manufacture full query coverage receipts."""
    output = Path(output)
    if output.exists():
        raise ValueError("Event-store output must be a new directory")
    columns = event_columns(audit, raw_row_indices)
    output.mkdir(parents=True)
    arrays = {}
    for name, values in columns.items():
        path = output / (name + ".npy")
        np.save(path, values, allow_pickle=False)
        arrays[name] = {"path": path.name, "sha256": file_hash(path), "dtype": str(values.dtype), "shape": list(values.shape)}
    h = audit.history
    eligible = np.flatnonzero(h.days < h.fit_cutoff)
    raw_complete = np.array_equal(np.asarray(raw_row_indices), eligible)
    manifest = {"contract": CONTRACT, "state": "probe", "scope": "probe", "events_count": len(raw_row_indices),
        "metadata_sha256": file_hash(metadata_path), "metadata_path": Path(metadata_path).resolve().as_posix(),
        "mapping_sha256": canonical_hash(h.mapping), "effective_metadata_sha256": audit.manifest["effective_metadata_sha256"],
        "parser_sha256": audit.manifest["history_module_sha256"], "type_policy": "mapped token >=3 exact film/series; UNK remains unknown",
        "fit_cutoff": __import__("datetime").date.fromordinal(h.fit_cutoff).isoformat(),
        "clock": "strict_h63_prior_loo", "event_clock": {"baseline": "strictly_before_event_date",
            "global_cap": "min(event_date,role_fit_cap)", "whole_user_exclusion": "all item/type/global tiers"},
        "normalization": "immutable_per_event_no_global_fit", "alpha": history.BASELINE_ALPHA,
        "beta": history.BASELINE_BETA, "c0": history.BASELINE_C0, "flag_bits": FLAG_BITS,
        "prepared_pins": prepared_pins or {}, "row_keys_sha256": row_keys_sha256 or {},
        "query_join": QUERY_JOIN, "coverage": {"raw_events": "complete_before_role_cap" if raw_complete else "probe_subset",
            "query_mapping": "not_checked", "eligible_raw_events": int(len(eligible))},
        "arrays": arrays, "bytes_per_event": BYTES_PER_EVENT,
        "source_pins": {**h.source_pins, "event_store_builder": _pin(__file__),
            "completion_diagnostics": _pin(diagnostics.__file__)},
        "diagnostic_only": True, "metadata_availability": "static_retrospective_catalog_snapshot",
        "outcome_availability": "last_watch_date_assumption_for_retained_aggregate; receipt time unverified"}
    manifest["signature"] = canonical_hash(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    return manifest


class CompletionEventStore:
    def __init__(self, root, manifest, arrays):
        self.root, self.manifest, self.arrays = Path(root), manifest, arrays
        self.provenance = manifest
        self.keys = arrays["user_id"] * history.MAX_ORDINAL + arrays["date"]
        self.users, self.starts = np.unique(arrays["user_id"], return_index=True)
        self.cap = history._day(manifest["fit_cutoff"])
        self.query_arrays = {}
        self.catalog = {}

    def derive_query_events(self, user_ids, query_dates, *, x, lengths, history_cutoffs=None,
                            prepared_split=None, prepared_row_indices=None):
        """Exact [B,63] event IDs, -1 padding; frozen state never uses target day."""
        if self.manifest["coverage"]["raw_events"] != "complete_before_role_cap":
            raise ValueError("A sampled event store cannot derive complete H63")
        users = history._ids(user_ids, "query user IDs")
        origins = np.asarray([history._day(v) for v in query_dates], np.int64)
        cutoffs = origins if history_cutoffs is None else np.asarray([history._day(v) for v in history_cutoffs], np.int64)
        if origins.shape != users.shape or cutoffs.shape != users.shape or np.any(cutoffs > origins) or np.any(cutoffs > self.cap):
            raise ValueError("Query history must align and remain within the event-store role cap")
        positions = np.searchsorted(self.users, users)
        found = positions < len(self.users)
        if len(self.users):
            found &= self.users[np.minimum(positions, len(self.users) - 1)] == users
        starts = np.zeros(len(users), np.int64)
        starts[found] = self.starts[positions[found]]
        ends = np.searchsorted(self.keys, users * history.MAX_ORDINAL + cutoffs, side="left")
        ends[~found] = starts[~found]
        n = np.minimum(ends - starts, 63)
        ids = ends[:, None] - n[:, None] + np.arange(63)[None, :]
        mask = np.arange(63)[None, :] < n[:, None]
        safe = np.where(mask, ids, 0)
        tokens = np.where(mask, self.arrays["token"][safe], 0) if len(self.keys) else np.zeros_like(safe)
        expected_x = np.column_stack((np.ones(len(users), np.int32), tokens))
        x, lengths = np.asarray(x), np.asarray(lengths)
        if (x.dtype.kind not in "iu" or x.shape != expected_x.shape or lengths.dtype.kind not in "iu"
                or lengths.shape != n.shape or not np.array_equal(x, expected_x) or not np.array_equal(lengths, n + 1)):
            raise ValueError("Prepared x/lengths do not match exact stored raw H63")
        if prepared_split is not None or prepared_row_indices is not None:
            if (prepared_split not in ("train", "val") or prepared_split not in self.manifest["prepared_pins"]
                    and prepared_split + ".npz" not in self.manifest["prepared_pins"]):
                raise ValueError("Prepared query split lacks an immutable binding")
            rows = history._ids(prepared_row_indices, "prepared row indices")
            if rows.shape != users.shape:
                raise ValueError("Prepared row indices must align with queries")
            if prepared_split not in self.query_arrays:
                raise ValueError("Exact prepared row lookup is absent from this staged/probe store")
            bound = self.query_arrays[prepared_split]
            if (np.any(rows >= len(bound["user_id"])) or not np.array_equal(users, bound["user_id"][rows])
                    or not np.array_equal(origins, bound["origin"][rows])
                    or not np.array_equal(cutoffs, bound["origin"][rows])
                    or not np.array_equal(lengths, bound["length"][rows])):
                raise ValueError("Prepared row IDs disagree with immutable user/origin/H63 binding")
        return np.where(mask, ids, -1).astype(np.int64)

    def query_diagnostics(self, event_ids):
        """Derive agreed query/group diagnostics from saved unique event facts."""
        ids = np.asarray(event_ids)
        if (ids.ndim != 2 or ids.shape[1] != 63 or ids.dtype.kind not in "iu"
                or np.any(ids < -1) or np.any(ids >= len(self.keys)) or not self.catalog):
            raise ValueError("Diagnostic event pointers/catalog are invalid")
        mask = ids >= 0
        if np.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("H63 event pointers must be a contiguous prefix")
        safe = np.where(mask, ids, 0)
        result = {"history.mask": mask, "history.event_id": ids}
        for name, values in self.arrays.items():
            fill = -1 if name == "raw_row_index" else 0
            result["history." + name] = (np.where(mask, values[safe], fill) if len(values)
                else np.full(ids.shape, fill, values.dtype))
        flags = result["history.flags"]
        prior_only = ((flags >> FLAG_BITS["prior_only"]) & 1).astype(bool) & mask
        valid = ((flags >> FLAG_BITS["residual_valid"]) & 1).astype(bool) & mask
        status = result["history.parser_status"]
        residual = result["history.residual"]
        n, nr = mask.sum(axis=1), valid.sum(axis=1)
        mean = np.divide(residual.sum(axis=1), nr, out=np.zeros(len(ids)), where=nr > 0)
        for name, value in {"n63": n, "n_valid_outcome": nr, "n_residual": nr,
            "n_missing": ((status == 1) & mask).sum(axis=1), "n_invalid": ((status == 2) & mask).sum(axis=1),
            "n_unavailable": np.zeros(len(ids), np.int32), "n_no_baseline": np.zeros(len(ids), np.int32),
            "relative_mean": mean, "relative_empty": nr == 0, "n_prior_only": (prior_only & valid).sum(axis=1),
            "n_type_unknown": ((((flags >> FLAG_BITS["type_unknown"]) & 1) != 0) & valid).sum(axis=1)}.items():
            result["query." + name] = value
        result["query.prior_only_share_fill"] = np.divide(result["query.n_prior_only"], nr,
            out=np.zeros(len(ids)), where=nr > 0)
        tokens = result["history.token"]
        for field in ("genres", "content_type", "countries", "decade"):
            labels = self.catalog["labels." + field][tokens]
            known = self.catalog["known." + field][tokens] & mask
            unseen = self.catalog["unseen." + field][tokens] & mask
            known_residual = (known & valid).sum(axis=1)
            for name, value in {"metadata_known_h63": known.sum(axis=1), "metadata_known_residual": known_residual,
                "metadata_unknown_h63": (~known & mask).sum(axis=1), "metadata_unknown_residual": (~known & valid).sum(axis=1),
                "unseen_h63": unseen.sum(axis=1), "unseen_residual": (unseen & valid).sum(axis=1),
                "has_unseen": unseen.any(axis=1),
                "partial_unknown_h63": (unseen & known).sum(axis=1),
                "partial_unknown_residual": (unseen & known & valid).sum(axis=1),
                "residual_coverage": np.divide(known_residual, nr, out=np.zeros(len(ids)), where=nr > 0)}.items():
                result[f"field.{field}.{name}"] = value
            if field == "decade":
                for name, label in (("year_valid", "valid_year"), ("year_outside", "outside")):
                    values = self.catalog[name][tokens]
                    result[f"field.{field}.{label}_h63"] = (values & mask).sum(axis=1)
                    result[f"field.{field}.{label}_residual"] = (values & valid).sum(axis=1)
            result[f"history.{field}.known"] = known
            result[f"history.{field}.has_unseen"] = unseen
            result[f"history.{field}.labels"] = np.where(mask[:, :, None], labels, -1)
            width = len(self.manifest["group_policy"]["categories"][field])
            support, sums, priors = (np.zeros((len(ids), width), np.float64) for _ in range(3))
            for category in range(width):
                chosen = np.any(labels == category, axis=2) & valid
                support[:, category] = chosen.sum(axis=1)
                sums[:, category] = np.where(chosen, residual, 0).sum(axis=1)
                priors[:, category] = (chosen & prior_only).sum(axis=1)
            mu = np.divide(sums, support, out=np.zeros_like(sums), where=support > 0)
            delta = np.where(support > 0, mu - mean[:, None], 0)
            for name, value in {"n": support.astype(np.int32), "raw_mu_fill": mu, "raw_delta_fill": delta,
                "empty": support == 0, "z": support / (support + self.manifest["group_policy"]["lambda"]) * delta,
                "prior_only_n": priors.astype(np.int32),
                "prior_only_share_fill": np.divide(priors, support, out=np.zeros_like(priors), where=support > 0)}.items():
                result[f"group.{field}.{name}"] = value
        return result

    def close(self):
        for value in self.arrays.values():
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
        for group in self.query_arrays.values():
            for value in group.values():
                if getattr(value, "_mmap", None) is not None:
                    value._mmap.close()
        self.arrays = {}
        self.query_arrays = {}
        self.catalog = {}
        self.keys = self.users = self.starts = np.array([], np.int64)


def load(path, *, expected_metadata_sha256=None, expected_prepared_pins=None,
         expected_row_keys_sha256=None, verify_hashes=True, allow_probe=False):
    """Production rejects partial/probe coverage before allocating an index."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT or canonical_hash({k: v for k, v in manifest.items() if k != "signature"}) != manifest.get("signature"):
        raise ValueError("Event-store manifest signature/contract mismatch")
    full = (manifest.get("state") == "completed" and manifest.get("scope") == "full"
        and manifest.get("coverage", {}).get("raw_events") == "complete_before_role_cap"
        and manifest.get("coverage", {}).get("query_mapping") == "complete_all_prepared_rows")
    if not full and not allow_probe:
        raise ValueError("Full diagnostics require complete raw events and all prepared query mappings")
    if verify_hashes is not True:
        raise ValueError("Event-store source/payload hash validation cannot be disabled")
    if expected_metadata_sha256 is not None and expected_metadata_sha256 != manifest["metadata_sha256"]:
        raise ValueError("Event-store metadata binding mismatch")
    if expected_prepared_pins is not None:
        for name, pin in expected_prepared_pins.items():
            actual = manifest["prepared_pins"].get(name)
            if (actual is None or actual["sha256"] != pin["sha256"]
                    or Path(actual["path"]).resolve() != Path(pin["path"]).resolve()):
                raise ValueError("Event-store prepared file binding mismatch")
    if expected_row_keys_sha256 is not None and expected_row_keys_sha256 != manifest["row_keys_sha256"]:
        raise ValueError("Event-store prepared row-key binding mismatch")
    for pin in manifest["source_pins"].values():
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("Event-store source SHA mismatch")
    if file_hash(manifest["metadata_path"]) != manifest["metadata_sha256"]:
        raise ValueError("Event-store metadata source SHA mismatch")
    if set(manifest["arrays"]) != set(EVENT_DTYPES) or manifest["flag_bits"] != FLAG_BITS:
        raise ValueError("Event-store column/flag schema mismatch")
    arrays = {}
    try:
        for name, dtype in EVENT_DTYPES.items():
            declaration = manifest["arrays"][name]
            target = (root / declaration["path"]).resolve()
            if target.parent != root.resolve() or file_hash(target) != declaration["sha256"]:
                raise ValueError("Event-store payload path/hash mismatch")
            values = np.load(target, mmap_mode="r", allow_pickle=False)
            arrays[name] = values
            if values.shape != (manifest["events_count"],) or str(values.dtype) != dtype or declaration["shape"] != list(values.shape):
                raise ValueError("Event-store payload shape/dtype mismatch")
        for start in range(0, manifest["events_count"], 100000):
            end = min(start + 100000, manifest["events_count"])
            if any(not np.isfinite(value[start:end]).all() for value in arrays.values()):
                raise ValueError("Nonfinite event facts")
            if (np.any(arrays["date"][start:end] >= history._day(manifest["fit_cutoff"]))
                    or not np.isin(arrays["parser_status"][start:end], (0, 1, 2)).all()):
                raise ValueError("Invalid event clock/parser status")
            if (any(np.any(arrays[name][start:end] < 0) for name in
                    ("raw_row_index", "user_id", "raw_item_id", "n_item", "n_type", "n_global"))
                    or np.any(arrays["date"][start:end] < 1) or np.any(arrays["token"][start:end] < 2)
                    or np.any((arrays["outcome"][start:end] < 0) | (arrays["outcome"][start:end] > 1))
                    or np.any((arrays["baseline"][start:end] < 0) | (arrays["baseline"][start:end] > 1))
                    or np.any(np.abs(arrays["residual"][start:end]) > 1)):
                raise ValueError("Invalid event identity/support/value range")
        if len(arrays["raw_row_index"]) > 1 and np.any(np.diff(arrays["raw_row_index"]) <= 0):
            raise ValueError("Event raw-row identities must increase uniquely")
        # Native source ordering is user/date/raw-item, including tied days.
        if len(arrays["user_id"]) > 1:
            u, d, i = arrays["user_id"], arrays["date"], arrays["raw_item_id"]
            if np.any((u[1:] < u[:-1]) | ((u[1:] == u[:-1]) & ((d[1:] < d[:-1])
                    | ((d[1:] == d[:-1]) & (i[1:] <= i[:-1]))))):
                raise ValueError("Event ordering/uniqueness differs from exact H63")
        store = CompletionEventStore(root, manifest, arrays)
        try:
            postproof = manifest.get("coverage", {}).get("query_mapping") == "complete_all_prepared_rows"
            if full or postproof:
                catalog_pin = manifest.get("catalog", {})
                catalog_path = (root / catalog_pin.get("path", "")).resolve()
                if catalog_path.parent != root.resolve() or file_hash(catalog_path) != catalog_pin.get("sha256"):
                    raise ValueError("Event-store diagnostic catalog path/hash mismatch")
                with np.load(catalog_path, allow_pickle=False) as payload:
                    store.catalog = {name: payload[name] for name in payload.files}
                if canonical_hash({name: diagnostics.array_hash(value) for name, value in store.catalog.items()}) != catalog_pin.get("arrays_sha256"):
                    raise ValueError("Event-store effective diagnostic metadata mismatch")
                receipt = manifest.get("query_coverage_receipt", {})
                if receipt.get("contract") != QUERY_JOIN or receipt.get("error_count") != 0 or set(receipt.get("splits", {})) != {"train", "val"}:
                    raise ValueError("Full event store lacks an all-prepared query coverage receipt")
                if set(manifest.get("query_arrays", {})) != {"train", "val"}:
                    raise ValueError("Full event store lacks persisted prepared row lookup")
                for split, declarations in manifest["query_arrays"].items():
                    if set(declarations) != {"user_id", "origin", "length"}:
                        raise ValueError("Prepared query lookup schema mismatch")
                    store.query_arrays[split] = {}
                    for name, declaration in declarations.items():
                        target = (root / declaration["path"]).resolve()
                        if target.parent != root.resolve() or file_hash(target) != declaration["sha256"]:
                            raise ValueError("Prepared query lookup payload hash/path mismatch")
                        values = np.load(target, mmap_mode="r", allow_pickle=False)
                        store.query_arrays[split][name] = values
                        dtype = {"user_id": "int64", "origin": "int32", "length": "uint8"}[name]
                        if (values.shape != (receipt["splits"][split]["rows"],) or str(values.dtype) != dtype
                                or declaration["shape"] != list(values.shape)):
                            raise ValueError("Prepared query lookup shape/dtype mismatch")
                    if receipt["splits"][split]["row_keys_sha256"] != manifest["row_keys_sha256"].get(split):
                        raise ValueError("Coverage receipt/prepared row hash mismatch")
                for pin in manifest["prepared_pins"].values():
                    if file_hash(pin["path"]) != pin["sha256"]:
                        raise ValueError("Prepared source file SHA mismatch")
                if full:
                    qa = manifest.get("final_validation", {})
                    if (qa.get("contract") != FULL_QA or qa.get("status") != "passed" or qa.get("error_count") != 0
                            or qa.get("events_count") != manifest["events_count"]
                            or qa.get("query_counts") != {split: data["rows"] for split, data in receipt["splits"].items()}):
                        raise ValueError("Full store lacks completed final validation; staged data is not admissible")
            return store
        except Exception:
            store.close()
            raise
    except Exception:
        for values in arrays.values():
            if getattr(values, "_mmap", None) is not None:
                values._mmap.close()
        raise


def _header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported prepared NPY version")
    if fortran or dtype.hasobject or dtype.kind not in "iu":
        raise ValueError("Prepared row arrays require C-order integer data")
    return shape, dtype


def _hash_header(shape, dtype):
    header = json.dumps({"dtype": dtype.str, "shape": list(shape)}, sort_keys=True,
        separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n")


def _peak_rss():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(key, ctypes.c_size_t)
            for key in ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    read = ctypes.windll.psapi.GetProcessMemoryInfo
    read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    read.restype = wintypes.BOOL
    if not read(process(), ctypes.byref(memory), memory.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return memory.peak


def _write_manifest(root, manifest):
    manifest.pop("signature", None)
    manifest["signature"] = canonical_hash(manifest)
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(root / "manifest.json")


def validate_prepared_queries(store, history_index, prepared_paths, *, batch_size=4096,
                              check_budget=None, expected_row_keys_sha256=None):
    """Stream all six prepared arrays once; save only 13 bytes/query lookup.

    Exact raw-row-index/item/date/token equality is checked independently against
    HistoryIndex, including UNK raw IDs. A digest binds the full symbolic H63 map.
    """
    if set(prepared_paths) != {"train", "val"} or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Both train/val paths and a positive integer batch size are required")
    result, declarations = {}, {}
    for split, source in prepared_paths.items():
        source = Path(source)
        pin = store.manifest["prepared_pins"][split + ".npz"]
        if source.resolve() != Path(pin["path"]).resolve() or file_hash(source) != pin["sha256"]:
            raise ValueError("Prepared source differs from immutable store declaration")
        with zipfile.ZipFile(source) as archive:
            streams = {name: archive.open(name + ".npy") for name in ROW_KEYS}
            mappings = {}
            try:
                headers = {name: _header(stream) for name, stream in streams.items()}
                total = headers["user_id"][0][0]
                if any(shape != ((total, 64) if name == "x" else (total,)) for name, (shape, _) in headers.items()):
                    raise ValueError("Prepared row-array shapes disagree")
                hashes = {name: _hash_header(*headers[name]) for name in ROW_KEYS}
                mapping_hash = _hash_header((total, 63), np.dtype("int64"))
                raw_mapping_hash = _hash_header((total, 63), np.dtype("int64"))
                for name, dtype in (("user_id", "int64"), ("origin", "int32"), ("length", "uint8")):
                    target = store.root / f"query.{split}.{name}.npy"
                    mappings[name] = np.lib.format.open_memmap(target, mode="w+", dtype=dtype, shape=(total,))
                history_rows = 0
                for start in range(0, total, batch_size):
                    if check_budget is not None:
                        check_budget("query_coverage", split=split, rows_done=start, rows_total=total)
                    stop = min(start + batch_size, total)
                    batch = {}
                    for name, stream in streams.items():
                        shape, dtype = headers[name]
                        wanted = (stop - start) * (64 if name == "x" else 1) * dtype.itemsize
                        payload = stream.read(wanted)
                        if len(payload) != wanted:
                            raise ValueError("Truncated prepared row array")
                        hashes[name].update(payload)
                        batch[name] = np.frombuffer(payload, dtype=dtype).reshape((stop - start, 64) if name == "x" else (stop - start,))
                    origins = batch["target_date"] if split == "train" else np.full(stop - start, store.cap, np.int64)
                    if split == "train" and np.any(origins >= store.cap):
                        raise ValueError("Train query origin violates frozen role cap")
                    ids = store.derive_query_events(batch["user_id"], origins, x=batch["x"], lengths=batch["lengths"])
                    raw_idx, mask, _, n, _ = history_index._history(batch["user_id"], origins,
                        x=batch["x"], lengths=batch["lengths"])
                    selected = ids[mask]
                    for name, values in (("raw_row_index", raw_idx), ("raw_item_id", history_index.items[raw_idx]),
                                         ("date", history_index.days[raw_idx]), ("token", history_index.tokens[raw_idx])):
                        if not np.array_equal(store.arrays[name][selected], values[mask]):
                            raise ValueError("Event pointers do not match historical raw H63 identity")
                    mapping_hash.update(np.ascontiguousarray(ids).tobytes())
                    raw_mapping_hash.update(np.ascontiguousarray(np.where(mask, raw_idx, -1)).tobytes())
                    history_rows += int(n.sum())
                    mappings["user_id"][start:stop], mappings["origin"][start:stop], mappings["length"][start:stop] = (
                        batch["user_id"], origins, batch["lengths"])
                if any(stream.read(1) for stream in streams.values()):
                    raise ValueError("Unexpected trailing prepared array bytes")
                row_hash = canonical_hash({name: digest.hexdigest() for name, digest in hashes.items()})
                if expected_row_keys_sha256 is not None and row_hash != expected_row_keys_sha256[split]:
                    raise ValueError("Prepared full row-key hash differs from expected baseline")
                result[split] = {"rows": total, "validated_rows": total, "error_count": 0,
                    "historical_events_in_queries": history_rows, "row_keys_sha256": row_hash,
                    "symbolic_h63_sha256": mapping_hash.hexdigest(), "raw_h63_source_indices_sha256": raw_mapping_hash.hexdigest()}
                declarations[split] = {}
                for name, values in mappings.items():
                    values.flush()
                    target = store.root / f"query.{split}.{name}.npy"
                    declarations[split][name] = {"path": target.name, "sha256": file_hash(target),
                        "shape": list(values.shape), "dtype": str(values.dtype)}
            finally:
                for stream in streams.values():
                    stream.close()
                for values in mappings.values():
                    values._mmap.close()
        if file_hash(source) != pin["sha256"]:
            raise ValueError("Prepared source changed during coverage validation")
    return {"contract": QUERY_JOIN, "error_count": 0, "splits": result}, declarations


def build_full(audit, output, metadata_path, prepared_paths, *, event_batch_size=100000,
               query_batch_size=4096, max_seconds=900, max_rss_bytes=4 * 1024 ** 3,
               expected_row_keys_sha256=None, final_validation=None):
    """Full authorized build; never mark full until every prepared query passes."""
    if (type(event_batch_size) is not int or event_batch_size < 1 or type(max_seconds) is not int or max_seconds < 1
            or set(prepared_paths) != {"train", "val"} or not callable(final_validation)):
        raise ValueError("Invalid full-build budget/batch/prepared inputs")
    root = Path(output)
    if root.exists():
        raise ValueError("Full event-store output must be new")
    h = audit.history
    eligible = np.flatnonzero(h.days < h.fit_cutoff)
    started = time.perf_counter()
    root.mkdir(parents=True)
    pins = {split + ".npz": _pin(path) for split, path in prepared_paths.items()}
    state = {"contract": CONTRACT, "state": "staged", "scope": "not_full", "events_count": len(eligible),
        "metadata_sha256": file_hash(metadata_path), "metadata_path": Path(metadata_path).resolve().as_posix(),
        "mapping_sha256": canonical_hash(h.mapping), "effective_metadata_sha256": audit.manifest["effective_metadata_sha256"],
        "parser_sha256": audit.manifest["history_module_sha256"], "group_policy": {
            "categories": audit.manifest["categories"], "lambda": history.RELATIVE_LAMBDA,
            "genre_eligibility": audit.manifest["genre_eligibility"],
            "rule": "equal-weight signed residual category mean minus overall mean, n/(n+lambda); zero fill plus empty mask"},
        "type_policy": "mapped token >=3 exact film/series; UNK remains unknown", "fit_cutoff": __import__("datetime").date.fromordinal(h.fit_cutoff).isoformat(),
        "clock": "strict_h63_prior_loo", "event_clock": {"baseline": "strictly_before_event_date",
            "global_cap": "min(event_date,role_fit_cap)", "whole_user_exclusion": "all item/type/global tiers"},
        "normalization": "immutable_per_event_no_global_fit", "alpha": history.BASELINE_ALPHA,
        "beta": history.BASELINE_BETA, "c0": history.BASELINE_C0, "flag_bits": FLAG_BITS,
        "prepared_pins": pins, "row_keys_sha256": {}, "query_join": QUERY_JOIN,
        "coverage": {"raw_events": "incomplete", "query_mapping": "not_checked", "eligible_raw_events": len(eligible)},
        "source_pins": {**h.source_pins, "event_store_builder": _pin(__file__), "completion_diagnostics": _pin(diagnostics.__file__)},
        "arrays": {}, "bytes_per_event": BYTES_PER_EVENT, "diagnostic_only": True,
        "metadata_availability": "static_retrospective_catalog_snapshot",
        "outcome_availability": "last_watch_date_assumption_for_retained_aggregate; receipt time unverified"}
    _write_manifest(root, state)
    last_log, observed_peak = -float("inf"), 0
    def budget(stage, **progress):
        nonlocal last_log, observed_peak
        elapsed = time.perf_counter() - started
        observed_peak = max(observed_peak, _peak_rss())
        if elapsed > max_seconds or observed_peak > max_rss_bytes:
            raise RuntimeError("Event-store actual time/RSS budget exceeded")
        if elapsed - last_log >= 5:
            (root / "progress.json").write_text(json.dumps({"stage": stage, "elapsed_seconds": elapsed,
                "peak_rss_bytes": observed_peak, **progress}, indent=2) + "\n", encoding="utf-8", newline="\n")
            last_log = elapsed
    mappings = {}
    try:
        for name, dtype in EVENT_DTYPES.items():
            mappings[name] = np.lib.format.open_memmap(root / (name + ".npy"), mode="w+", dtype=dtype, shape=(len(eligible),))
        for start in range(0, len(eligible), event_batch_size):
            budget("event_facts", events_done=start, events_total=len(eligible))
            stop = min(start + event_batch_size, len(eligible))
            values = event_columns(audit, eligible[start:stop])
            for name in mappings:
                mappings[name][start:stop] = values[name]
        for name, values in mappings.items():
            values.flush()
            state["arrays"][name] = {"path": name + ".npy", "sha256": file_hash(root / (name + ".npy")),
                "dtype": str(values.dtype), "shape": list(values.shape)}
        state["coverage"]["raw_events"] = "complete_before_role_cap"
        catalog = {**{"labels." + name: audit.genre_labels if name == "genres" else values
            for name, values in h.labels.items()}, **{"known." + name: values for name, values in h.known.items()},
            **{"unseen." + name: values for name, values in h.unseen.items()},
            "types": h.types, "year_valid": h.year_valid, "year_outside": h.year_outside}
        np.savez_compressed(root / "catalog.npz", **catalog)
        state["catalog"] = {"path": "catalog.npz", "sha256": file_hash(root / "catalog.npz"),
            "arrays_sha256": canonical_hash({name: diagnostics.array_hash(value) for name, value in catalog.items()})}
        state["state"] = "events_completed_query_pending"
        _write_manifest(root, state)
    except Exception as error:
        state["state"], state["error"] = "incomplete", str(error)
        _write_manifest(root, state)
        raise
    finally:
        for values in mappings.values():
            values._mmap.close()
    store = load(root, allow_probe=True)
    try:
        receipt, query_arrays = validate_prepared_queries(store, h, prepared_paths,
            batch_size=query_batch_size, check_budget=budget, expected_row_keys_sha256=expected_row_keys_sha256)
        budget("finalize")
        state["query_coverage_receipt"], state["query_arrays"] = receipt, query_arrays
        state["row_keys_sha256"] = {split: declaration["row_keys_sha256"] for split, declaration in receipt["splits"].items()}
        state["coverage"]["query_mapping"] = "complete_all_prepared_rows"
        state["state"] = "events_completed_query_proved_qa_pending"
        _write_manifest(root, state)
        store.close()
        store = load(root, allow_probe=True)
        qa = final_validation(store, h)
        if (not isinstance(qa, dict) or qa.get("contract") != FULL_QA or qa.get("status") != "passed"
                or qa.get("error_count") != 0):
            raise ValueError("Final oracle/diagnostic validation did not pass")
        budget("final_admission")
        state["final_validation"] = {**qa, "events_count": len(eligible),
            "query_counts": {split: data["rows"] for split, data in receipt["splits"].items()}}
        state["state"], state["scope"] = "completed", "full"
        state["actual_measurement"] = {"elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": observed_peak,
            "max_seconds": max_seconds, "max_rss_bytes": max_rss_bytes}
        _write_manifest(root, state)
    except Exception as error:
        state["state"] = ("events_completed_query_proved_qa_failed" if "query_coverage_receipt" in state
            else "events_completed_query_incomplete")
        state["error"] = str(error)
        _write_manifest(root, state)
        raise
    finally:
        store.close()
    return state
