"""Pinned float32 people day banks: bounded writer, owned workers and mmap reader.

Planning never computes catalogue features. Execution requires explicit origins;
the full inventory additionally requires allow_full=True. Each day is immutable
once published. Partial runs retain completed receipts for explicit later reuse.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import ctypes
from datetime import date
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading
import time

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "2"
import numpy as np

if __package__:
    from . import hm_job
    from . import kion_people_dynamics as core
else:
    def _standalone_sibling(name):
        path = Path(__file__).resolve().with_name(name + ".py")
        spec = importlib.util.spec_from_file_location("_people_banks_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    hm_job = _standalone_sibling("hm_job")
    core = _standalone_sibling("kion_people_dynamics")

CONTRACT = "kion-people-dynamics-bank-v1"
GIB = 1024 ** 3
CORE_SHA256 = "6bf97bb98c8fdf0cb858db18066862d3a94cb107334bcb14c74e340f2943911c"


def _sibling(name):
    return Path(__file__).resolve().with_name(name)
IDENTITY_FIELDS = ("contract", "feature_contract", "fit_role", "fit_cutoff_exclusive",
    "role_cap_exclusive", "vocab_size", "feature_count", "dtype", "shape_per_origin",
    "columns", "columns_sha256", "groups", "parameters", "parameters_sha256",
    "mapping_sha256", "index", "graph", "implementation", "origin_inventory")


class Interrupted(RuntimeError):
    pass


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=constant)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": core._file_hash(path), "size_bytes": path.stat().st_size}


def _verify_pin(pin):
    actual = _pin(pin["path"])
    if actual != pin:
        raise ValueError("File pin changed: " + str(pin["path"]))


def _replace(source, target, check=lambda: None, *, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + 1.
    for attempt in range(51):
        check()
        try:
            os.replace(source, target)
            return
        except OSError as error:
            remaining = deadline - clock()
            if getattr(error, "winerror", None) not in (5, 32, 33) or remaining <= 0 or attempt == 50:
                raise
            check()
            sleep(min(.02, remaining))
    raise AssertionError("Unreachable publication retry")


def atomic_json(path, value, check=lambda: None):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    check()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _replace(temporary, path, check)


def _local(root, value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe bank relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Unsafe bank relative path")
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Bank path escapes output tree")
    return path


def identity(manifest):
    return core._hash({key: manifest[key] for key in IDENTITY_FIELDS})


def _origins(inventory):
    origins = inventory.get("origin_ordinals")
    if (not isinstance(origins, list) or not origins or any(type(x) is not int for x in origins)
            or origins != sorted(set(origins)) or any(not 1 <= x < core.DAY_BASE for x in origins)):
        raise ValueError("Origins must be exact sorted unique ordinals")
    actual = hashlib.sha256(np.asarray(origins, dtype=np.int64).tobytes()).hexdigest()
    if actual != inventory.get("origin_array_sha256"):
        raise ValueError("Origin array pin mismatch")
    return origins


def validate_manifest(manifest, *, expected_identity_sha256=None, require_complete=True, root=None):
    if manifest.get("contract") != CONTRACT or manifest.get("feature_contract") != core.CONTRACT:
        raise ValueError("People bank contract mismatch")
    try:
        signature = identity(manifest)
        origins = _origins(manifest["origin_inventory"])
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete bank identity") from error
    if (signature != manifest.get("identity_sha256") or not _sha(signature)
            or expected_identity_sha256 is not None and signature != expected_identity_sha256):
        raise ValueError("People bank identity pin mismatch")
    columns, parameters = manifest["columns"], manifest["parameters"]
    vocab, width = manifest["vocab_size"], manifest["feature_count"]
    if (type(vocab) is not int or vocab < 3 or type(width) is not int or width <= 0
            or not isinstance(columns, list) or len(columns) != width
            or len(set(columns)) != width or any(not isinstance(x, str) or not x for x in columns)
            or "people_unknown_item" not in columns or core._hash(columns) != manifest["columns_sha256"]
            or core._hash(parameters) != manifest["parameters_sha256"]
            or manifest["dtype"] != "<f4" or manifest["shape_per_origin"] != [vocab, width]
            or manifest["origin_inventory"].get("origin_count") != len(origins)):
        raise ValueError("Bank schema/parameters/origin count mismatch")
    if (manifest["fit_role"] not in core.ROLE_CAPS
            or manifest["role_cap_exclusive"] != core.ROLE_CAPS[manifest["fit_role"]]
            or core._day(manifest["fit_cutoff_exclusive"]) > core._day(manifest["role_cap_exclusive"])
            or any(x > core._day(manifest["fit_cutoff_exclusive"]) for x in origins)):
        raise ValueError("Bank effective cutoff or role cap mismatch")
    for value in (manifest["mapping_sha256"], manifest["index"]["manifest_sha256"],
            manifest["index"]["npz_sha256"], manifest["index"]["metadata_sha256"],
            manifest["graph"]["map_file_sha256"], manifest["graph"]["canonical_work_map_sha256"],
            manifest["implementation"]["core_sha256"], manifest["implementation"]["adapter_sha256"]):
        if not _sha(value):
            raise ValueError("Invalid bank source pin")
    if manifest["index"]["confirmed_work_map_sha256"] != manifest["graph"]["canonical_work_map_sha256"]:
        raise ValueError("Graph and index work-map mismatch")
    days, paths = manifest.get("days", {}), set()
    if not isinstance(days, dict):
        raise ValueError("Invalid days table")
    for key, entry in days.items():
        if (str(entry.get("origin_ordinal")) != key or entry["origin_ordinal"] not in origins
                or entry.get("origin_exclusive") != date.fromordinal(entry["origin_ordinal"]).isoformat()
                or entry.get("state") != "completed" or entry.get("shape") != [vocab, width]
                or entry.get("dtype") != "<f4" or not _sha(entry.get("sha256"))
                or type(entry.get("size_bytes")) is not int or entry["size_bytes"] <= vocab * width * 4
                or entry.get("payload_bytes") != vocab * width * 4
                or entry.get("all_finite") is not True or entry.get("special_rows_zero") is not True):
            raise ValueError("Bank day receipt mismatch")
        if entry.get("identity_sha256") != manifest["identity_sha256"]:
            raise ValueError("Day receipt identity mismatch")
        path = _local(root or Path.cwd(), entry.get("path"))
        normalized = os.path.normcase(str(path))
        if normalized in paths:
            raise ValueError("Duplicate bank path")
        paths.add(normalized)
        _local(root or Path.cwd(), entry.get("parity_receipt"))
    if require_complete and (manifest.get("state") != "completed"
            or set(days) != set(map(str, origins)) or manifest.get("all_origins_completed") is not True):
        raise ValueError("Incomplete exact origin bank inventory")
    return manifest


def make_plan(index_dir, graph_map, origin_inventory, *, expected_index_sha256,
              expected_graph_sha256, expected_origin_array_sha256, artifact_review=None):
    """Read/hash-only inventory. No feature calculation, output tree or worker."""
    index_dir = Path(index_dir).resolve()
    index_pin, graph_pin, origin_pin = _pin(index_dir / "manifest.json"), _pin(graph_map), _pin(origin_inventory)
    if index_pin["sha256"] != expected_index_sha256 or graph_pin["sha256"] != expected_graph_sha256:
        raise ValueError("Index/graph planning pin mismatch")
    meta, graph, inventory = read_json(index_pin["path"]), read_json(graph_pin["path"]), read_json(origin_pin["path"])
    origins = _origins(inventory)
    if inventory["origin_array_sha256"] != expected_origin_array_sha256:
        raise ValueError("Expected origin array pin mismatch")
    if meta["contract"] != core.CONTRACT or core._file_hash(index_dir / "index.npz") != meta["index_npz_sha256"]:
        raise ValueError("Index payload pin mismatch")
    mapping = {str(k): str(v) for k, v in graph["confirmed_work_ids"].items()}
    work_hash = core._hash(mapping)
    provenance = meta["provenance"]
    if (work_hash != provenance["confirmed_work_map_sha256"]
            or core._hash(meta["mapping"]) != provenance["mapping_sha256"]
            or core._file_hash(Path(core.__file__)) != CORE_SHA256):
        raise ValueError("Graph/mapping/core planning mismatch")
    if provenance.get("pilot_only") is not False:
        raise ValueError("Bank planning requires complete non-pilot index")
    if artifact_review is not None:
        review = read_json(artifact_review)
        if review.get("state") != "passed":
            raise ValueError("Index artifact review not passed")
    plan = {"contract": CONTRACT, "feature_contract": core.CONTRACT, "state": "planned",
        "fit_role": provenance["fit_role"], "fit_cutoff_exclusive": date.fromordinal(meta["fit_cutoff"]).isoformat(),
        "role_cap_exclusive": core.ROLE_CAPS[provenance["fit_role"]], "vocab_size": len(meta["token_attributes"]),
        "feature_count": len(meta["columns"]), "dtype": "<f4", "shape_per_origin": [len(meta["token_attributes"]), len(meta["columns"])],
        "columns": meta["columns"], "columns_sha256": core._hash(meta["columns"]), "groups": meta["groups"],
        "parameters": meta["parameters"], "parameters_sha256": core._hash(meta["parameters"]),
        "mapping_sha256": provenance["mapping_sha256"],
        "index": {"manifest_path": index_pin["path"], "manifest_sha256": index_pin["sha256"],
            "npz_sha256": meta["index_npz_sha256"], "source_pins": provenance["source_pins"],
            "metadata_sha256": provenance["metadata_sha256"], "parser_sha256": provenance["parser_sha256"],
            "pilot_only": provenance["pilot_only"], "count_unit": provenance["count_unit"],
            "confirmed_work_map_sha256": work_hash},
        "graph": {"map_path": graph_pin["path"], "map_file_sha256": graph_pin["sha256"],
            "canonical_work_map_sha256": work_hash, "scope": graph.get("scope", "explicit confirmed work map"),
            "source_pins": graph.get("source_pins", {})},
        "implementation": {"core_sha256": CORE_SHA256, "adapter_sha256": core._file_hash(_sibling("kion_people_dynamics_fast.py")),
            "bank_builder_sha256": core._file_hash(Path(__file__)), "job_helper_sha256": core._file_hash(Path(hm_job.__file__))},
        "origin_inventory": {"path": origin_pin["path"], "file_sha256": origin_pin["sha256"],
            "origin_array_sha256": inventory["origin_array_sha256"], "origin_ordinals": origins, "origin_count": len(origins)},
        "days": {}, "payload_bytes_all_origins": len(origins) * len(meta["token_attributes"]) * len(meta["columns"]) * 4,
        "all_origins_completed": False, "day_banks_in_ram": False}
    plan["identity_sha256"] = identity(plan)
    if artifact_review is not None:
        plan["artifact_review"] = _pin(artifact_review)
    validate_manifest(plan, require_complete=False)
    return plan


def verify_plan_inputs(plan):
    validate_manifest(plan, require_complete=False)
    checks = [(plan["index"]["manifest_path"], plan["index"]["manifest_sha256"]),
        (Path(plan["index"]["manifest_path"]).parent / "index.npz", plan["index"]["npz_sha256"]),
        (plan["graph"]["map_path"], plan["graph"]["map_file_sha256"]),
        (plan["origin_inventory"]["path"], plan["origin_inventory"]["file_sha256"]),
        (Path(core.__file__), plan["implementation"]["core_sha256"]),
        (_sibling("kion_people_dynamics_fast.py"), plan["implementation"]["adapter_sha256"]),
        (Path(__file__), plan["implementation"]["bank_builder_sha256"]),
        (Path(hm_job.__file__), plan["implementation"]["job_helper_sha256"])]
    for path, expected in checks:
        if core._file_hash(path) != expected:
            raise ValueError("Frozen bank input changed: " + str(path))
    if "artifact_review" in plan:
        _verify_pin(plan["artifact_review"])
    core._assert_sources_unchanged(plan["index"]["source_pins"])


class PeopleBankReader:
    def __init__(self, manifest_path, *, expected_identity_sha256, cache_days=1):
        if type(cache_days) is not int or cache_days not in (1, 2) or not _sha(expected_identity_sha256):
            raise ValueError("Require pinned identity and cache_days1or2")
        self.path = Path(manifest_path).resolve()
        self.manifest = validate_manifest(read_json(self.path), expected_identity_sha256=expected_identity_sha256,
            root=self.path.parent)
        self.columns, self.groups = self.manifest["columns"], self.manifest["groups"]
        self.fit_cutoff = min(core._day(self.manifest["fit_cutoff_exclusive"]), core._day(self.manifest["role_cap_exclusive"]))
        self.vocab_size, self.mapping_sha256 = self.manifest["vocab_size"], self.manifest["mapping_sha256"]
        self.parameters = self.manifest["parameters"]
        self.provenance = {"bank_contract": CONTRACT, "identity_sha256": expected_identity_sha256,
            "index": self.manifest["index"], "graph": self.manifest["graph"], "origin_inventory": self.manifest["origin_inventory"],
            "source_pins": self.manifest["index"]["source_pins"], "mapping_sha256": self.mapping_sha256,
            "pilot_only": self.manifest["index"]["pilot_only"], "fit_role": self.manifest["fit_role"],
            "fit_cutoff_exclusive": self.manifest["fit_cutoff_exclusive"]}
        for name in ("items", "interactions", "metadata"):
            if name in self.provenance["source_pins"]:
                self.provenance[name + "_sha256"] = self.provenance["source_pins"][name]["sha256"]
        self.cache_days, self._cache = cache_days, OrderedDict()

    def _bank(self, day):
        if day not in self._cache:
            entry = self.manifest["days"].get(str(day))
            if entry is None:
                raise ValueError("Exact effective bank origin missing: " + str(day))
            path = _local(self.path.parent, entry["path"])
            before = path.stat()
            if before.st_size != entry["size_bytes"] or core._file_hash(path) != entry["sha256"]:
                raise ValueError("Bank NPY size/hash mismatch")
            bank = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if (not isinstance(bank, np.memmap) or bank.shape != tuple(entry["shape"]) or bank.dtype.str != "<f4"
                        or bank.ndim != 2 or not bank.flags.c_contiguous
                        or bank.offset + bank.nbytes != entry["size_bytes"]
                        or np.any(bank[:3] != 0)):
                    raise ValueError("Bank NPY shape/dtype/special-row mismatch")
                for left in range(0, len(bank), 512):
                    if not np.isfinite(bank[left:left + 512]).all():
                        raise ValueError("Nonfinite bank payload")
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Bank mutated while opening")
            except BaseException:
                if isinstance(bank, np.memmap):
                    bank._mmap.close()
                elif hasattr(bank, "close"):
                    bank.close()
                raise
            self._cache[day] = bank
            while len(self._cache) > self.cache_days:
                _, old = self._cache.popitem(last=False)
                old._mmap.close()
        self._cache.move_to_end(day)
        return self._cache[day]

    def features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        tokens = np.asarray(item_tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0):
            raise ValueError("item_tokens must be nonnegative integer vector")
        def dates(values):
            values = np.asarray(values)
            if values.ndim == 0:
                return np.full(len(tokens), core._day(values.item()), np.int64)
            if values.shape != tokens.shape:
                raise ValueError("Dates must be scalar or one per token")
            return np.array([core._day(value) for value in values], np.int64)
        cutoffs = np.minimum(dates(query_dates), self.fit_cutoff)
        if history_cutoffs is not None:
            cutoffs = np.minimum(cutoffs, dates(history_cutoffs))
        if user_ids is not None and np.shape(user_ids) != tokens.shape:
            raise ValueError("user_ids must have one value per token")
        output = np.zeros((len(tokens), len(self.columns)), np.float32)
        output[tokens >= self.vocab_size, self.columns.index("people_unknown_item")] = 1
        known = (tokens >= 3) & (tokens < self.vocab_size)
        for day in np.unique(cutoffs[known]):
            take = np.flatnonzero(known & (cutoffs == day))
            output[take] = self._bank(int(day))[tokens[take]]
        return output

    def close(self):
        for bank in self._cache.values():
            bank._mmap.close()
        self._cache.clear()


def _affinity(mask):
    if type(mask) is not int or mask <= 0:
        raise ValueError("Invalid CPU affinity mask")
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise OSError("Cannot set CPU affinity")
    else:
        os.sched_setaffinity(0, [i for i in range(mask.bit_length()) if mask & (1 << i)])


def _tree_rss(pid):
    """Resident bytes of the owned process tree, including venv launchers."""
    if os.name != "nt":
        pending, total = [pid], 0
        while pending:
            current = pending.pop()
            try:
                total += int(Path(f"/proc/{current}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                pending += list(map(int, Path(f"/proc/{current}/task/{current}/children").read_text().split()))
            except FileNotFoundError:
                pass
        return total
    from ctypes import wintypes as W
    class Entry(ctypes.Structure):
        _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("pid", W.DWORD),
            ("heap", ctypes.c_size_t), ("module", W.DWORD), ("threads", W.DWORD),
            ("parent", W.DWORD), ("priority", W.LONG), ("flags", W.DWORD), ("exe", W.WCHAR * 260)]
    class Counters(ctypes.Structure):
        _fields_ = [("cb", W.DWORD), ("faults", W.DWORD)] + [(x, ctypes.c_size_t) for x in
            ("peak", "rss", "qpp", "qp", "qnp", "qn", "page", "peakpage")]
    # Private wrappers: argtypes on ctypes.windll functions are shared with
    # unrelated concurrent RSS monitors, whose structure classes may differ.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = W.HANDLE
    kernel.CreateToolhelp32Snapshot.argtypes = (W.DWORD, W.DWORD)
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = (W.HANDLE, ctypes.POINTER(Entry))
    kernel.CloseHandle.argtypes = (W.HANDLE,)
    kernel.OpenProcess.argtypes, kernel.OpenProcess.restype = (W.DWORD, W.BOOL, W.DWORD), W.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (W.HANDLE, ctypes.POINTER(Counters), W.DWORD)
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise OSError("Cannot snapshot owned process tree")
    parents = {}
    try:
        entry = Entry(); entry.size = ctypes.sizeof(entry)
        ctypes.set_last_error(0)
        present = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while present:
            parents[int(entry.pid)] = int(entry.parent)
            ctypes.set_last_error(0)
            present = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES is the only normal end.
            raise OSError(ctypes.get_last_error(), "Cannot enumerate owned process tree")
    finally:
        kernel.CloseHandle(snapshot)
    owned = {int(pid)}
    for _ in range(len(parents) + 1):
        added = {child for child, parent in parents.items() if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    total = 0
    for child in owned:
        handle = kernel.OpenProcess(0x410, False, child)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: PID exited after snapshot.
                continue
            raise OSError(error, "Cannot inspect owned process " + str(child))
        try:
            counters = Counters(); counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                error = ctypes.get_last_error()
                code = W.DWORD()
                kernel.GetExitCodeProcess.argtypes = (W.HANDLE, ctypes.POINTER(W.DWORD))
                if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
                    continue  # Confirmed exit after OpenProcess, not an inspection failure.
                raise OSError(error, "Cannot measure owned process memory")
            total += int(counters.rss)
        finally:
            kernel.CloseHandle(handle)
    return total


def _index_arrays(index):
    arrays = {"token_work": index.token_work, **{"events_" + k: v for k, v in index.events.items()}}
    prefixes = {"work": index.work_prefix,
        **{"union_" + k: v for k, v in index.unions.items()},
        **{"correction_" + k: v for k, v in index.corrections.items()}}
    for name, prefix in prefixes.items():
        for field in ("keys", "cumulative", "base"):
            arrays[name + "_" + field] = getattr(prefix, field)
    for role, relation in index.relations.items():
        for field in ("offsets", "works", "owners"):
            arrays[role + "_" + field] = relation[field]
    return arrays


def _bind_index(index, plan):
    """Validate the actual writer, including arrays, against the pinned NPZ."""
    meta_path = Path(plan["index"]["manifest_path"])
    if core._file_hash(meta_path) != plan["index"]["manifest_sha256"]:
        raise ValueError("Writer index manifest changed")
    meta = read_json(meta_path)
    if (index.parameters != plan["parameters"] or index.columns != plan["columns"]
            or index.groups != plan["groups"] or index.vocab_size != plan["vocab_size"]
            or index.fit_cutoff != core._day(plan["fit_cutoff_exclusive"])
            or core._hash(index.mapping) != plan["mapping_sha256"]
            or index.provenance != meta["provenance"] or index.work_groups != meta["work_groups"]
            or index.token_attributes != meta["token_attributes"]
            or any(index.relations[role]["labels"] != meta["relation_labels"][role]
                or index.relations[role]["lookup"] != {name: i for i, name in enumerate(meta["relation_labels"][role])}
                for role in core.ROLES)):
        raise ValueError("Actual writer index does not match frozen plan")
    payload = meta_path.parent / "index.npz"
    if core._file_hash(payload) != plan["index"]["npz_sha256"]:
        raise ValueError("Writer index NPZ changed")
    arrays = _index_arrays(index)
    with np.load(payload, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError("Writer index array schema mismatch")
        for name, values in arrays.items():
            expected = archive[name]
            if values.dtype != expected.dtype or values.shape != expected.shape or not np.array_equal(values, expected):
                raise ValueError("Actual writer index array differs: " + name)
            values.flags.writeable = False
    if core._file_hash(payload) != plan["index"]["npz_sha256"]:
        raise ValueError("Writer index changed while binding")
    derived = {
        "years": np.array([group["year"] or 0 for group in index.work_groups], np.int32),
        "genre_incidence": np.array([[genre in group["genres"] for genre in core.GENRES]
            for group in index.work_groups], np.float64).reshape(len(index.work_groups), len(core.GENRES)),
        "format_incidence": np.array([[group["type"] == kind for kind in ("film", "series")]
            for group in index.work_groups], np.float64).reshape(len(index.work_groups), 2),
        "event_offsets": np.searchsorted(index.events["work"], np.arange(len(index.work_groups) + 1))}
    for name, expected in derived.items():
        if not np.array_equal(getattr(index, name), expected):
            raise ValueError("Derived writer index differs: " + name)
        getattr(index, name).flags.writeable = False
    for name in ("_feature_cache", "_day_cache", "_window_cache"):
        if hasattr(index, name):
            getattr(index, name).clear()


def _space(root, required, reserve):
    if shutil.disk_usage(root).free < required + reserve:
        raise Interrupted("Insufficient disk for declared output plus reserve")


def _deadlines(started, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Deadline budget must be finite and positive")
    hard = started + seconds
    return hard - min(2., seconds * .1), hard


def write_day(index, plan, output, origin, *, check=lambda: None, chunk_rows=128, progress=None):
    """Compute one full token axis; no overwrite and no nearest-day reuse."""
    if origin not in plan["origin_inventory"]["origin_ordinals"] or type(chunk_rows) is not int or not 1 <= chunk_rows <= 512:
        raise ValueError("Unknown origin or invalid chunk size")
    check()
    binding_started = time.monotonic()
    _bind_index(index, plan)
    binding_seconds = time.monotonic() - binding_started
    check()
    output = Path(output).resolve()
    for folder in ("days", "receipts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    when = date.fromordinal(origin).isoformat()
    final = output / "days" / (when + ".npy")
    partial = output / "days" / (when + ".partial.npy")
    receipt_path = output / "receipts" / (when + ".json")
    if any(path.exists() for path in (final, partial, receipt_path)):
        raise FileExistsError("Immutable day output already exists")
    started = time.monotonic()
    bank = np.lib.format.open_memmap(partial, mode="w+", dtype="<f4", shape=tuple(plan["shape_per_origin"]))
    done = 0
    try:
        for left in range(0, len(bank), chunk_rows):
            check()
            right = min(left + chunk_rows, len(bank))
            values = index.features(np.arange(left, right, dtype=np.int64), origin)
            if values.shape != (right - left, plan["feature_count"]) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Nonfinite or malformed feature chunk")
            bank[left:right] = values
            done = right
            if progress:
                progress(done, len(bank), time.monotonic() - started)
        if np.any(bank[:3] != 0):
            raise ValueError("Special rows not zero")
        bank.flush()
    except BaseException:
        bank.flush()
        bank._mmap.close()
        atomic_json(output / "receipts" / (when + ".interrupted.json"),
            {"state": "interrupted", "identity_sha256": plan["identity_sha256"], "origin_ordinal": origin,
                "completed_rows": done, "total_rows": plan["vocab_size"], "seconds": time.monotonic() - started})
        raise
    bank._mmap.close()
    check()
    # Verify an independently calculated original-core sample, including aliases.
    tokens = sorted({0, 1, 2, min(3, plan["vocab_size"] - 1), plan["vocab_size"] - 1})
    groups = [g for g in index.work_groups if len(g["cards"]) > 1][:3]
    for group in groups:
        tokens.extend(index.mapping[item] for item in group["cards"][:2])
    tokens = np.array(sorted(set(tokens)), np.int64)
    original = core.PeopleDynamicsIndex(index.mapping, index.token_work, index.work_groups,
        index.relations, index.work_prefix, index.unions, index.corrections, index.events,
        index.fit_cutoff, index.parameters, index.provenance, index.token_attributes)
    reference = core.PeopleDynamicsIndex.features(original, tokens, origin)
    saved = np.load(partial, mmap_mode="r", allow_pickle=False)
    try:
        np.testing.assert_array_equal(saved[tokens], reference)
    finally:
        saved._mmap.close()
    check()
    entry = {"state": "completed", "identity_sha256": plan["identity_sha256"], "origin_ordinal": origin,
        "origin_exclusive": when, "path": "days/" + when + ".npy", "sha256": core._file_hash(partial),
        "size_bytes": partial.stat().st_size, "payload_bytes": plan["vocab_size"] * plan["feature_count"] * 4,
        "shape": plan["shape_per_origin"], "dtype": "<f4", "all_finite": True, "special_rows_zero": True,
        "parity_receipt": "receipts/" + when + ".json", "parity_tokens": tokens.tolist(),
        "parity_bitwise": True, "input_binding_seconds": binding_seconds,
        "lookup_and_publish_seconds": time.monotonic() - started}
    _replace(partial, final, check)
    atomic_json(receipt_path, entry, check)
    return entry


def reuse_day(plan, source_manifest, output, origin, *, mode="copy", check=lambda: None):
    """Explicit copy/hardlink into a new tree; source probe bytes stay immutable."""
    if mode not in ("copy", "hardlink"):
        raise ValueError("Reuse mode must be copy or hardlink")
    source_manifest, output = Path(source_manifest).resolve(), Path(output).resolve()
    previous = validate_manifest(read_json(source_manifest), expected_identity_sha256=plan["identity_sha256"],
        require_complete=False, root=source_manifest.parent)
    if str(origin) not in previous["days"]:
        raise ValueError("Reusable completed origin missing")
    entry = dict(previous["days"][str(origin)])
    source = _local(source_manifest.parent, entry["path"])
    if source.stat().st_size != entry["size_bytes"] or core._file_hash(source) != entry["sha256"]:
        raise ValueError("Reusable probe file hash mismatch")
    for folder in ("days", "receipts"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    destination = _local(output, entry["path"])
    temporary = destination.with_name(destination.name + ".reuse.tmp")
    receipt = _local(output, entry["parity_receipt"])
    if any(path.exists() for path in (destination, temporary, receipt)):
        raise FileExistsError("Immutable reuse destination already exists")
    check()
    if mode == "hardlink":
        os.link(source, temporary)
    else:
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                check()
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    if core._file_hash(temporary) != entry["sha256"]:
        raise ValueError("Copied probe bytes differ")
    _replace(temporary, destination, check)
    entry["reused_from_manifest"] = _pin(source_manifest)
    entry["reuse_mode"] = mode
    atomic_json(receipt, entry, check)
    return entry


class _OwnedWorker:
    def __init__(self, arguments, log):
        self.job = hm_job._new_job() if os.name == "nt" else None
        self.process = None
        self.log = None
        try:
            self.log = Path(log).open("wb")
            options = {"cwd": hm_job.ROOT, "stdout": self.log, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                options["creationflags"] = 4 | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            self.process = subprocess.Popen([sys.executable, *arguments], **options)
            if os.name == "nt":
                if not hm_job._assign_job(self.job, int(self.process._handle)):
                    raise ctypes.WinError(ctypes.get_last_error())
                hm_job._resume_initial_thread(self.process.pid)
        except BaseException:
            self.close()
            raise

    def close(self):
        try:
            if self.process is not None:
                try:
                    hm_job._cleanup(self.process, self.job)
                finally:
                    self.process, self.job = None, None
            elif self.job is not None:
                hm_job._close(self.job)
                self.job = None
        finally:
            if self.log is not None:
                self.log.close()


def _worker(plan_path, output, origins, affinity_mask, worker_id, *, seconds_per_day=900, chunk_rows=128):
    _affinity(affinity_mask)
    plan, output = read_json(plan_path), Path(output)
    verify_plan_inputs(plan)
    if not __package__:
        raise RuntimeError("Owned builder requires package invocation")
    fast = importlib.import_module(".kion_people_dynamics_fast", __package__)
    index = fast.FastPeopleDynamicsIndex.load(Path(plan["index"]["manifest_path"]).parent,
        expected_manifest_sha256=plan["index"]["manifest_sha256"], expected_mapping_sha256=plan["mapping_sha256"])
    progress_path = output / (f"worker-{worker_id}.json")
    started, peak = time.monotonic(), [0]
    if not 0 < seconds_per_day <= 900:
        raise ValueError("Worker date budget must be at most900seconds")
    soft, hard = _deadlines(started, seconds_per_day)
    deadline, hard_deadline, stop = [soft], [hard], threading.Event()
    def check():
        if (output / "STOP").exists():
            raise Interrupted("Cooperative stop requested")
        if time.monotonic() >= deadline[0]:
            raise Interrupted("Per-date deadline reached")
    def watch():
        while not stop.wait(min(.25, max(.001, hard_deadline[0] - time.monotonic()))):
            try:
                peak[0] = max(peak[0], _tree_rss(os.getpid()))
                if peak[0] > 2 * GIB or time.monotonic() >= hard_deadline[0]:
                    os._exit(124)
            except BaseException:
                os._exit(125)
    threading.Thread(target=watch, daemon=True).start()
    try:
        for origin in origins:
            day_started = time.monotonic()
            deadline[0], hard_deadline[0] = _deadlines(day_started, seconds_per_day)
            resource_request = read_json(output / "request.json")
            _space(output, plan["vocab_size"] * plan["feature_count"] * 4 * len(resource_request["affinity_masks"]) + 1024,
                resource_request["reserve_bytes"])
            def progress(done, total, seconds):
                atomic_json(progress_path, {"state": "working", "origin_ordinal": origin,
                    "day_started_monotonic": day_started, "rows_done": done, "rows_total": total,
                    "lookup_seconds": seconds, "peak_rss_bytes": peak[0]}, check)
            progress(0, plan["vocab_size"], 0)
            entry = write_day(index, plan, output, origin, check=check, chunk_rows=chunk_rows, progress=progress)
            atomic_json(progress_path, {"state": "day_completed", "origin_ordinal": origin,
                "day_started_monotonic": day_started, "seconds": time.monotonic() - day_started,
                "peak_rss_bytes": peak[0]}, check)
            print(json.dumps({"origin": origin, "seconds": entry["lookup_and_publish_seconds"]}), flush=True)
        verify_plan_inputs(plan)
        atomic_json(progress_path, {"state": "completed", "origins": origins,
            "seconds": time.monotonic() - started, "peak_rss_bytes": peak[0]})
    finally:
        stop.set()


def run_plan(plan_path, output, *, origins, affinity_masks=(3,), execute=False, allow_full=False,
             hard_seconds=4 * 3600, seconds_per_day=900, reserve_bytes=20_000_000_000, chunk_rows=128):
    """Explicit bounded execution; the default is a read-only cost/inventory plan."""
    started = time.monotonic()
    plan = read_json(plan_path)
    validate_manifest(plan, require_complete=False)
    selected = [core._day(value) for value in origins]
    inventory = plan["origin_inventory"]["origin_ordinals"]
    if (not selected or len(selected) != len(set(selected)) or not set(selected).issubset(inventory)
            or len(selected) > 3 and not allow_full):
        raise ValueError("Explicit unique origins required; full run needs allow_full")
    if (not 1 <= len(affinity_masks) <= 3 or any(type(x) is not int or x <= 0 or x.bit_count() != 2 for x in affinity_masks)
            or any(a & b for i, a in enumerate(affinity_masks) for b in affinity_masks[i + 1:])
            or not math.isfinite(hard_seconds) or not 0 < hard_seconds <= 4 * 3600
            or not math.isfinite(seconds_per_day) or not 0 < seconds_per_day <= 900
            or type(reserve_bytes) is not int or reserve_bytes < 0):
        raise ValueError("Invalid finite CPU/time/disk resource plan")
    request = {"state": "plan_only", "origins": selected, "payload_bytes": len(selected) * plan["vocab_size"] * plan["feature_count"] * 4,
        "affinity_masks": list(affinity_masks), "rss_worker_bytes": 2 * GIB, "rss_total_bytes": 8 * GIB,
        "reserve_bytes": reserve_bytes, "hard_seconds": hard_seconds, "seconds_per_day": seconds_per_day,
        "identity_sha256": plan["identity_sha256"]}
    if not execute:
        return request
    verify_plan_inputs(plan)
    output = Path(output).resolve()
    _space(output.parent, request["payload_bytes"] + len(selected) * 1024, reserve_bytes)
    output.mkdir(parents=False, exist_ok=False)
    for folder in ("days", "receipts"):
        (output / folder).mkdir()
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "request.json", request)
    workers, peak = [], 0
    soft_deadline, hard_deadline = _deadlines(started, hard_seconds)
    phase_starts = {}
    _affinity(sum(affinity_masks))
    try:
        for number, mask in enumerate(affinity_masks):
            assigned = selected[number::len(affinity_masks)]
            if not assigned:
                continue
            args = ["-m", "scripts.kion_people_banks", "worker", "--plan", str(output / "plan.json"),
                "--output", str(output), "--origins", ",".join(map(str, assigned)), "--affinity", str(mask),
                "--worker-id", str(number), "--seconds-per-day", str(seconds_per_day), "--chunk-rows", str(chunk_rows)]
            worker = _OwnedWorker(args, output / f"worker-{number}.log")
            workers.append((number, worker))
            phase_starts[number] = started
        def check():
            if (output / "STOP").exists() or time.monotonic() >= soft_deadline:
                raise Interrupted("Parent stop/deadline")
        while any(worker.process.poll() is None for _, worker in workers):
            check()
            peak = max(peak, _tree_rss(os.getpid()))
            if peak > 8 * GIB:
                raise Interrupted("Total owned RSS budget exceeded")
            _space(output, 0, reserve_bytes)
            for number, worker in workers:
                code = worker.process.poll()
                if code is not None and code != 0:
                    raise RuntimeError(f"Owned worker {number} failed with exit {code}")
                if _tree_rss(worker.process.pid) > 2 * GIB:
                    raise Interrupted("Worker RSS budget exceeded")
                progress = output / f"worker-{number}.json"
                if progress.exists():
                    state = read_json(progress)
                    phase_starts[number] = state.get("day_started_monotonic", phase_starts[number])
                if code is None and time.monotonic() - phase_starts[number] >= seconds_per_day:
                    raise Interrupted("Worker per-date deadline exceeded")
            atomic_json(output / "progress.json", {"state": "running", "seconds": time.monotonic() - started,
                "completed_days": len(list((output / "receipts").glob("????-??-??.json"))),
                "peak_tree_rss_bytes": peak, "identity_sha256": plan["identity_sha256"]}, check)
            time.sleep(min(.25, max(0, soft_deadline - time.monotonic())))
        if any(worker.process.returncode != 0 for _, worker in workers):
            raise RuntimeError("Owned worker failed")
        verify_plan_inputs(plan)
        manifest = dict(plan)
        manifest["days"] = {str(day): read_json(output / "receipts" / (date.fromordinal(day).isoformat() + ".json")) for day in selected}
        manifest["all_origins_completed"] = set(selected) == set(inventory)
        manifest["state"] = "completed" if manifest["all_origins_completed"] else "partial"
        validate_manifest(manifest, require_complete=manifest["all_origins_completed"], root=output)
        atomic_json(output / "manifest.json", manifest, check)
        report = {**request, "state": manifest["state"], "seconds": time.monotonic() - started,
            "peak_tree_rss_bytes": peak, "manifest_sha256": core._file_hash(output / "manifest.json"),
            "days": {key: {"seconds": value["lookup_and_publish_seconds"], "sha256": value["sha256"],
                "size_bytes": value["size_bytes"]} for key, value in manifest["days"].items()}}
        atomic_json(output / "report.json", report, check)
        return report
    except BaseException as error:
        (output / "STOP").write_text(type(error).__name__ + ": " + str(error), encoding="utf-8", newline="\n")
        time.sleep(min(.5, max(0, hard_deadline - time.monotonic())))
        atomic_json(output / "failure.json", {"state": "interrupted_or_failed", "error": str(error),
            "seconds": time.monotonic() - started, "identity_sha256": plan["identity_sha256"]})
        raise
    finally:
        cleanup_errors = []
        for _, worker in workers:
            try:
                worker.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise RuntimeError("Owned worker cleanup failed") from cleanup_errors[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("worker", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origins", required=True)
    parser.add_argument("--affinity", default="3")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--seconds-per-day", type=float, default=900)
    parser.add_argument("--hard-seconds", type=float, default=4 * 3600)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args(argv)
    origins = args.origins.split(",")
    origins = [int(x) if x.isdecimal() else x for x in origins]
    if args.mode == "worker":
        _worker(args.plan, args.output, origins, int(args.affinity), args.worker_id,
            seconds_per_day=args.seconds_per_day, chunk_rows=args.chunk_rows)
    else:
        print(json.dumps(run_plan(args.plan, args.output, origins=origins,
            affinity_masks=tuple(map(int, args.affinity.split(","))), execute=args.execute,
            allow_full=args.allow_full, hard_seconds=args.hard_seconds,
            seconds_per_day=args.seconds_per_day, chunk_rows=args.chunk_rows)), flush=True)


if __name__ == "__main__":
    main()
