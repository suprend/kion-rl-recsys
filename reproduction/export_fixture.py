"""Root-only bounded export of a genuine frozen-representation inference fixture.

Import is stdlib-only. No training, provider construction, model execution, input
rewriting or whole-cache array load occurs here. NumPy/Torch are imported only
after pinned metadata, file identities and the complete byte plan pass. The root
coordinator alone may call run() inside its native resource admission wrapper.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "kion-portable-fixture-v1"
REQUEST_CONTRACT = "kion-export-fixture-request-v1"
CACHE_CONTRACT = "kion-current-C1070-natural1024-H1-v1"
EVALUATION_CONTRACT = "kion-scientific3-evaluation-v1"
SCHEMA = "71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d"
SOURCE_COLUMNS = ["current_detached_gpt_logit", "log1p_strict7d_count"]
ARMS = ("C", "RL-completion", "RL-residual")
POSITIONS = (0, 1, 2, 3)
CORE = {
    "encoded": ("float32", (256, 256)),
    "candidate_ids": ("int32", (256, 1024)),
    "candidate_raw_ids": ("int64", (256, 1024)),
    "base_logits": ("float32", (256, 1024)),
    "raw_features": ("float32", (256, 1024, 1068)),
    "source_values": ("float32", (256, 1024, 2)),
    "item_table": ("float32", (15418, 256)),
}
GPT_INPUTS = {
    "gpt_x": ("x", "int64", (256, 64)),
    "gpt_lengths": ("lengths", "int64", (256,)),
    "gpt_user_features": ("user_features", "float32", (256, 3252)),
}
GPT_CONFIG = {"vocab_size": 15418, "context_length": 64, "hidden_size": 256,
              "layers": 4, "heads": 4, "user_feature_dim": 3252}
HEAD_SHAPES = {
    "pair_projection.weight": (32, 1070),
    "pair_head.0.weight": (256, 800),
    "pair_head.0.bias": (256,),
    "pair_head.2.weight": (1, 256),
    "pair_head.2.bias": (1,),
}
MAX_METADATA = 8 << 20
MAX_TEACHER_CONFIG = 16 << 20
TEACHER_SHA256 = "5e7101857c2dbe100fb4e1587c0369abc95c108303ed149123ee183f62aeac26"
TEACHER_BYTES = 558979727
TEACHER_CONFIG_SHA256 = "e3b4b1730895fceb1b7d4ea6357aea783c046f5e87cacd7d4ff933cf94d45d45"
DTYPES = {"float32": "<f4", "int32": "<i4", "int64": "<i8", "bool": "|b1"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= MAX_METADATA, "Oversized/missing metadata")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    value = Path(path).stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def contained(base, relative):
    """Reject traversal, absolute paths, symlinks escaping a declared source root."""
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Relative file required")
    root = Path(base).resolve()
    result = (root / relative).resolve()
    require(result != root and result.is_relative_to(root), "File escapes declared root")
    return result


class Budget:
    def __init__(self, limits, check=None, deadline_utc=None):
        self.started = time.monotonic()
        self.seconds = limits["maximum_seconds"]
        self.hash_limit = limits["maximum_hashed_source_bytes"]
        self.output_limit = limits["maximum_output_bytes"]
        self.hash_bytes = 0
        self.callback = check
        self.deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00")) if deadline_utc else None
        require(type(self.seconds) is int and 1 <= self.seconds <= 600, "Bounded seconds required")
        require(type(self.hash_limit) is int and 0 <= self.hash_limit <= 128 << 20, "Source hashing capped at128MiB")
        require(type(self.output_limit) is int and 40 << 20 <= self.output_limit <= 128 << 20, "Output cap40..128MiB required")

    def check(self, stage):
        require(time.monotonic() - self.started < self.seconds, "Fixture export deadline exceeded")
        require(self.deadline is None or datetime.now(timezone.utc) < self.deadline, "Fixture export UTC deadline exceeded")
        if self.callback:
            self.callback(stage=stage, hashed_source_bytes=self.hash_bytes)

    def hash(self, path):
        self.check("source-hash")
        size = Path(path).stat().st_size
        require(self.hash_bytes + size <= self.hash_limit, "Source hash budget exceeded; accepted stat-bound pin required")
        self.hash_bytes += size
        result = sha(path)
        self.check("source-hash-completed")
        return result


def verified(entry, budget, reuse=None):
    """Reuse only a pinned root acceptance receipt containing an exact full stat."""
    path = Path(entry["path"]).resolve()
    require(path.is_file() and identity(path)["size"] == entry["bytes"], "Source size differs")
    require(isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64, "SHA256 required")
    before = identity(path)
    accepted = next((value for key, value in (reuse or {}).items() if Path(key).resolve() == path), None)
    if accepted is not None:
        require(accepted.get("hashes_verified") is True and accepted.get("sha256") == entry["sha256"]
                and accepted.get("stat") == before, "Accepted file hash/stat pin differs")
        method = "accepted-full-hash-with-unchanged-stat"
    else:
        require(budget.hash(path) == entry["sha256"], "Source hash differs: " + path.name)
        method = "fresh-full-hash"
    require(identity(path) == before, "Source changed during verification")
    return path, {"sha256": entry["sha256"], "stat": before, "verification": method}


def pin_metadata(entry, budget):
    path = Path(entry["path"]).resolve()
    require(path.is_file() and path.stat().st_size <= MAX_METADATA, "Invalid metadata size")
    full = {**entry, "bytes": path.stat().st_size}
    verified(full, budget)
    return path, read_json(path)


def accepted_pins(entry, budget):
    if entry is None:
        return {}
    _, receipt = pin_metadata(entry, budget)
    require(receipt.get("contract") == "kion-root-accepted-file-stats-v1"
            and receipt.get("state") == "accepted", "Root accepted-stat receipt required")
    result = receipt.get("files")
    require(isinstance(result, dict), "Accepted-stat files mapping required")
    return result


def validate_trusted_teacher(entry, cache_teacher):
    """The ONE root-authorised local legacy pickle; no caller-chosen substitutes."""
    expected = (ROOT.parent / "night-20261023-r1/sft-recovery/run-recovery-r1/worker/model/latest.pt").resolve()
    path = Path(entry["path"]).resolve()
    require(path == expected and entry.get("sha256") == TEACHER_SHA256
            and entry.get("bytes") == TEACHER_BYTES, "Teacher is not the one authorised local frozen artifact")
    require(Path(cache_teacher["path"]).resolve() == expected and cache_teacher.get("sha256") == TEACHER_SHA256,
            "Frozen cache teacher differs from authorised artifact")
    return path


def pin_teacher_config(entry, cache_teacher, budget):
    """This known8.95MB config is pinned, never parsed; ordinary JSON cap stays8MiB."""
    path = Path(entry["path"]).resolve()
    require(path == Path(cache_teacher["config_path"]).resolve()
            and entry.get("sha256") == cache_teacher.get("config_sha256") == TEACHER_CONFIG_SHA256,
            "Wrong teacher config pin/path")
    require(path.is_file() and 0 < path.stat().st_size <= MAX_TEACHER_CONFIG, "Teacher config exceeds16MiB")
    verified({**entry, "bytes": path.stat().st_size}, budget)
    return path


def npy_header(path):
    """Read at most64KiB; reject object, structured, Fortran and trailing payload."""
    path = Path(path)
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Not an NPY file")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
        length_bytes = stream.read(2 if version == b"\x01\x00" else 4)
        require(len(length_bytes) == (2 if version == b"\x01\x00" else 4), "Truncated NPY header")
        length = struct.unpack("<H" if version == b"\x01\x00" else "<I", length_bytes)[0]
        require(0 < length <= 65536, "NPY header exceeds64KiB")
        raw = stream.read(length)
        require(len(raw) == length, "Truncated NPY header")
        header = ast.literal_eval(raw.decode("utf-8" if version == b"\x03\x00" else "latin1").strip())
        offset = stream.tell()
    require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}, "Bad NPY header fields")
    require(header["fortran_order"] is False and header["descr"] in DTYPES.values(), "Unsupported NPY dtype/layout")
    shape = header["shape"]
    require(isinstance(shape, tuple) and shape and all(type(n) is int and n > 0 for n in shape), "Invalid NPY shape")
    itemsize = 1 if header["descr"] == "|b1" else int(header["descr"][-1])
    require(path.stat().st_size == offset + math.prod(shape) * itemsize, "NPY payload size differs")
    return {"dtype": next(k for k, v in DTYPES.items() if v == header["descr"]), "shape": list(shape), "offset": offset}


def source_array(base, descriptor, expected, budget, reuse):
    path = contained(base, descriptor["path"])
    dtype, shape = expected
    require(descriptor.get("dtype") == dtype and descriptor.get("shape") == list(shape), "Source descriptor schema differs")
    _, evidence = verified({**descriptor, "path": path.as_posix()}, budget, reuse)
    header = npy_header(path)
    require(header["dtype"] == dtype and header["shape"] == list(shape), "Source NPY schema differs")
    return {"path": path.as_posix(), "descriptor": descriptor, "identity": evidence, "payload_bytes": path.stat().st_size-header["offset"]}


def validate_request(request):
    require(request.get("contract") == REQUEST_CONTRACT, "Wrong fixture request contract")
    require(request.get("training_updates") == 0 and request.get("provider_opened") is False, "No training/provider construction allowed")
    require(request.get("positions") == list(POSITIONS), "Only fixed first4 source rows; no outcome-dependent selection")
    require(request.get("cpu_threads") in (1, 2, 3, 4), "CPU thread cap1..4 required")
    require(request.get("native_admission") == "root-owned", "Root native resource admission required")
    deadline = datetime.fromisoformat(request["deadline_utc"].replace("Z", "+00:00"))
    require(deadline.utcoffset() is not None and datetime.now(timezone.utc) < deadline, "Aware future deadline required")
    output = Path(request["output_dir"]).resolve()
    private = Path(request["private_provenance_output"]).resolve()
    reproduction_root, research_root = (ROOT / "reproduction").resolve(), (ROOT / "research").resolve()
    require(output.is_relative_to(reproduction_root) and output != reproduction_root and not output.exists(), "Fresh reproduction output required")
    require(private.is_relative_to(research_root) and not private.exists(), "Fresh private research provenance required")
    require(output.parent.is_dir() and private.parent.is_dir(), "Output parents must exist")
    return output, private


def plan(request, check=None):
    """Stdlib-only preflight. A passing plan authorizes no numerical execution."""
    output, private = validate_request(request)
    budget = Budget(request["limits"], check, request["deadline_utc"])
    reuse = accepted_pins(request.get("accepted_stat_receipt"), budget)
    cache_path, cache = pin_metadata(request["cache_manifest"], budget)
    eval_path, evaluation = pin_metadata(request["evaluation_manifest"], budget)
    require(cache.get("contract") == CACHE_CONTRACT and cache.get("state") == "completed"
            and cache.get("cohort") == "screen" and cache.get("split") == "val"
            and cache.get("rows") == 256 and cache.get("candidate_width") == 1024
            and cache.get("raw_width") == 1068 and cache.get("precision") == "cuda-fp16"
            and cache.get("feature_schema_sha256") == SCHEMA and cache.get("source_columns") == SOURCE_COLUMNS
            and cache.get("no_target_injection") is True and cache.get("no_seen_filter") is True, "Wrong frozen screen cache")
    require(evaluation.get("contract") == EVALUATION_CONTRACT and evaluation.get("state") == "completed"
            and evaluation.get("stage") == "screen" and evaluation.get("rows") == 256
            and evaluation.get("precision") == "cuda-fp16", "Wrong frozen scientific screen oracle")
    require(evaluation.get("cache", {}).get("sha256") == request["cache_manifest"]["sha256"], "Historical oracle/cache binding differs")
    arrays = {name: source_array(cache_path.parent, cache["arrays"][name], spec, budget, reuse) for name, spec in CORE.items()}
    arrays["origins"] = source_array(cache_path.parent, cache["arrays"]["origins"], ("int32", (256,)), budget, reuse)
    for name, (source_name, dtype, shape) in GPT_INPUTS.items():
        arrays[name] = source_array(cache_path.parent, cache["arrays"][source_name], (dtype, shape), budget, reuse)
    oracles = {arm: source_array(eval_path.parent, evaluation["arrays"][arm+"_scores"], ("float32", (256, 1024)), budget, reuse) for arm in ARMS}
    bindings = {"candidate_ids": source_array(eval_path.parent, evaluation["arrays"]["pool_ids"], CORE["candidate_ids"], budget, reuse),
                "candidate_raw_ids": source_array(eval_path.parent, evaluation["arrays"]["candidate_raw_ids"], CORE["candidate_raw_ids"], budget, reuse)}
    checkpoints, pins = {}, {}
    require(set(request["checkpoints"]) == set(ARMS), "Exactly three head checkpoints required")
    for arm in ARMS:
        entry = request["checkpoints"][arm]
        path, evidence = verified(entry, budget, reuse)
        require(entry["sha256"] == evaluation["checkpoints"][arm]["sha256"], "Historical oracle checkpoint differs")
        checkpoints[arm] = {**entry, "path": path.as_posix()}
        pins[arm] = evidence
    require(checkpoints["C"]["sha256"] == cache["C_checkpoint_sha256"], "C cache ancestor differs")
    validate_trusted_teacher(request["teacher_checkpoint"], cache["teacher"])
    teacher, teacher_evidence = verified(request["teacher_checkpoint"], budget, reuse)
    require(teacher_evidence["verification"] == "accepted-full-hash-with-unchanged-stat", "Trusted teacher requires root accepted full-hash/stat receipt")
    config_path = pin_teacher_config(request["teacher_config"], cache["teacher"], budget)
    helper_path = Path(request["gpt_helper"]["path"]).resolve()
    require(helper_path == (ROOT / "reproduction/kion_replay/gpt.py").resolve(), "Only owned pinned GPT helper allowed")
    verified({**request["gpt_helper"], "bytes": helper_path.stat().st_size}, budget)
    # No real source-user/row/target identifiers are opened. Full16 producer
    # model-token prefixes ARE opened/exported, under explicit root authority.
    payload = sum(value["payload_bytes"] if name == "item_table" else value["payload_bytes"] * 4 // 256
                  for name, value in arrays.items() if name != "origins" and name not in GPT_INPUTS)
    peer_payload = sum(arrays[name]["payload_bytes"] * 16 // 256 for name in GPT_INPUTS)
    require(payload + peer_payload <= 64 << 20, "Selected array payload exceeds64MiB")
    # Input memmap slicing + table, three small secure checkpoint payloads, oracle slices.
    working = 8 * payload + sum(entry["bytes"] for entry in checkpoints.values()) + 2 * request["teacher_checkpoint"]["bytes"] + (128 << 20)
    require(working <= request["limits"]["maximum_working_bytes"] <= 2 << 30, "Working-set plan exceeds2GiB")
    planned_output = payload + peer_payload + 31821824 + 3 * sum(math.prod(shape)*4 for shape in HEAD_SHAPES.values()) + (2 << 20)
    require(planned_output <= budget.output_limit, "Output plan exceeds cap")
    free = shutil.disk_usage(output.parent).free
    reserve = request["limits"]["minimum_disk_free_bytes"]
    require(type(reserve) is int and reserve >= 15 << 30 and free >= reserve + planned_output, "Disk reserve15GiB required")
    return {"output": output, "private": private, "budget": budget, "cache": cache, "evaluation": evaluation,
            "cache_path": cache_path, "evaluation_path": eval_path, "arrays": arrays, "oracles": oracles,
            "bindings": bindings, "checkpoints": checkpoints, "checkpoint_pins": pins,
            "teacher": {**request["teacher_checkpoint"], "path": teacher.as_posix()}, "teacher_pin": teacher_evidence,
            "gpt_helper": helper_path, "teacher_config_path": config_path,
            "planned_source_slice_bytes": payload+peer_payload, "planned_output_bytes": planned_output, "planned_working_bytes": working}


def descriptor(path, root, value=None, **extra):
    result = {"path": Path(path).relative_to(root).as_posix(), "sha256": sha(path), "bytes": Path(path).stat().st_size, **extra}
    if value is not None:
        result.update(dtype=str(value.dtype), shape=list(value.shape))
    return result


def assert_unchanged(planned):
    for name in ("arrays", "oracles", "bindings"):
        for item in planned[name].values():
            require(identity(item["path"]) == item["identity"]["stat"], "Source changed after plan")
    for arm, item in planned["checkpoints"].items():
        require(identity(item["path"]) == planned["checkpoint_pins"][arm]["stat"], "Checkpoint changed after plan")
    require(identity(planned["teacher"]["path"]) == planned["teacher_pin"]["stat"], "Teacher changed after plan")


def extract_head(torch, payload, arm, c_sha):
    expected_contract = {"C": "kion-three-head-262k-head-v1", "RL-completion": CACHE_CONTRACT,
                         "RL-residual": "kion-current-C1070-residual-H1-v1"}[arm]
    require(isinstance(payload, dict) and payload.get("contract") == expected_contract, "Wrong checkpoint contract")
    if arm == "C":
        require(payload.get("name") == "C" and payload.get("complete") is True
                and payload.get("raw_indices") == list(range(1068)), "Wrong/incomplete supervised C")
        receipt = payload.get("receipt", {})
        require(receipt.get("epochs_completed") == 3 and receipt.get("steps_per_head") == 6144, "C budget/lineage differs")
    else:
        require(payload.get("step") == 1000 and payload.get("lineage", {}).get("C_checkpoint_sha256") == c_sha,
                "Only original-C latest1000 lineage allowed")
    state = payload.get("state_dict")
    require(isinstance(state, dict) and set(state) == set(HEAD_SHAPES), "Unexpected head tensor keys")
    result = {}
    for name, shape in HEAD_SHAPES.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == torch.float32
                and bool(torch.isfinite(value).all()), "Bad head tensor: " + name)
        result[name] = value.detach().cpu().contiguous().clone()
    return result, expected_contract


def run(request, check=None):
    """Export slices and inference-only heads; root must admit/limit this process."""
    planned = plan(request, check)
    budget = planned["budget"]
    budget.check("before-numerical-imports")
    import numpy as np
    import torch
    torch.set_num_threads(request["cpu_threads"])
    output = planned["output"]
    output.mkdir()
    (output / "weights").mkdir()
    (output / "oracle").mkdir()
    arrays, expected, weights, opened = {}, {}, {}, []
    try:
        def load(item):
            budget.check("memmap-open")
            value = np.load(item["path"], mmap_mode="r", allow_pickle=False)
            opened.append(value)
            return value
        origins = load(planned["arrays"]["origins"])
        require(bool(np.all(origins == origins[0])), "Screen producer groups must share one frozen origin")
        layout = [{"size": 16, "slot": pos} for pos in POSITIONS]
        selected = {}
        for name in CORE:
            source = load(planned["arrays"][name])
            # Basic slicing precedes np.array; never materialize all raw1068 rows.
            value = np.array(source if name == "item_table" else source[:4], copy=True)
            require(bool(np.isfinite(value).all()), "Nonfinite fixture values")
            if name in planned["bindings"]:
                oracle_value = load(planned["bindings"][name])
                require(bool(np.array_equal(value, oracle_value[:4])), "Historical oracle candidate alignment differs")
            path = output / (name + ".npy")
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
            arrays[name] = descriptor(path, output, value)
            if name in ("candidate_ids", "candidate_raw_ids", "base_logits", "source_values"):
                selected[name] = value
            del value
        require(bool(np.array_equal(selected["source_values"][..., 0], selected["base_logits"])), "Source12/base mismatch")
        require(bool(((selected["candidate_ids"] >= 3) & (selected["candidate_ids"] < 15418)).all())
                and all(len(set(row)) == 1024 for row in selected["candidate_ids"].tolist()), "Invalid natural candidate pool")
        for arm in ARMS:
            budget.check("export-head")
            pin = planned["checkpoints"][arm]
            # Secure root-pinned artifacts only; optimizer/scaler/receipt excluded.
            payload = torch.load(pin["path"], map_location="cpu", weights_only=True)
            state, source_contract = extract_head(torch, payload, arm, planned["checkpoints"]["C"]["sha256"])
            path = output / "weights" / (arm + ".pt")
            with path.open("xb") as stream:
                torch.save(state, stream)
            weights[arm] = descriptor(path, output, format="torch-state-dict-v1", source_checkpoint_sha256=pin["sha256"],
                                      source_contract=source_contract, step=None if arm == "C" else 1000)
            del payload, state
            scores = np.array(load(planned["oracles"][arm])[:4], copy=True)
            require(bool(np.isfinite(scores).all()), "Nonfinite historical oracle")
            path = output / "oracle" / (arm + "_scores.npy")
            with path.open("xb") as stream:
                np.save(stream, scores, allow_pickle=False)
            expected[arm] = descriptor(path, output, scores, role="validation-only")
        for name in GPT_INPUTS:
            source = load(planned["arrays"][name])
            value = np.array(source[:16], copy=True)
            require(bool(np.isfinite(value).all()), "Nonfinite GPT producer input")
            path = output / (name+".npy")
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
            arrays[name] = descriptor(path, output, value)
            if name == "gpt_lengths":
                require(bool(((value >= 1) & (value <= 64)).all()), "Invalid GPT producer lengths")
                trimmed_length = int(value.max())
        budget.check("secure-teacher-load")
        assert_unchanged(planned)
        validate_trusted_teacher(planned["teacher"], planned["cache"]["teacher"])
        # Explicit root approval: ONE previously trained, exact SHA+path+stat
        # bound local teacher has NumPy RNG state. Restricted loading rejects
        # that legacy payload. This exception NEVER exists in portable runtime.
        payload = torch.load(planned["teacher"]["path"], map_location="cpu", weights_only=False)
        helper_path = planned["gpt_helper"]
        require(sha(helper_path) == request["gpt_helper"]["sha256"], "GPT helper changed after plan")
        spec = importlib.util.spec_from_file_location("_portable_fixture_gpt_helper", helper_path)
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        state = helper.extract_encoder_state(torch, payload)
        del payload
        path = output / "weights/GPT.pt"
        with path.open("xb") as stream:
            torch.save(state, stream)
        weights["GPT"] = descriptor(path, output, format="torch-state-dict-v1",
                                    source_checkpoint_sha256=planned["teacher"]["sha256"],
                                    source_contract="kion-complete-gpt-v1", step=None)
        del state
        assert_unchanged(planned)
        budget.check("before-manifest")
        record = {"contract": CONTRACT, "state": "completed", "rows": 4, "candidate_width": 1024,
                  "raw_width": 1068, "feature_schema_sha256": SCHEMA, "source_columns": SOURCE_COLUMNS,
                  "precision": "cuda-fp16", "head_batch_max": 16, "candidate_chunk": 64,
                  "producer_layout": layout, "arrays": arrays, "weights": weights, "expected": expected,
                  "gpt": {"config": GPT_CONFIG, "producer_batch_size": 16, "row_slots": list(POSITIONS),
                          "trimmed_length": trimmed_length},
                  "context_ids": ["context-001", "context-002", "context-003", "context-004"],
                  "no_target_injection": True, "no_seen_filter": True, "tie_break": "score-desc/rawID-asc",
                  "temperature": 1.0, "target_labels_included": False,
                  "selection": "first4-frozen-screen-cache-positions-outcome-independent",
                  "parity": {"mode": "exact", "atol": 0.0, "rtol": 0.0, "top10_exact": True,
                             "historical_source_tolerance": {"atol": 0.003, "rtol": 0.001}},
                  "capabilities": {"cached_representation_head_inference": True, "GPT_encoder_inference": True,
                                   "RM_inference": False, "training": False},
                  "privacy": {"source_user_ids_included": False, "source_histories_included": True,
                              "histories_representation": "full16-producer-peer-public-KION-model-token-prefixes; no source-user-ID mapping",
                              "derived_representations_can_retain_information": True,
                              "candidate_ids": "public-KION-catalog-token/raw-item-IDs"},
                  "provenance": {"cache_manifest_sha256": request["cache_manifest"]["sha256"],
                                 "historical_oracle_manifest_sha256": request["evaluation_manifest"]["sha256"],
                                 "logical_source": "frozen-screen256/natural1024",
                                 "teacher_export_boundary": "one root-authorised local trusted legacy load; portable pure tensor runtime weights_only=True",
                                 "reference_scores_role": "oracle-only; fresh inference must use head weights and inputs"},
                  "limits": {"planned_source_slice_bytes": planned["planned_source_slice_bytes"],
                             "planned_working_bytes": planned["planned_working_bytes"],
                             "hashed_source_bytes": budget.hash_bytes, "maximum_output_bytes": budget.output_limit},
                  "verification": {"export_only": True, "fresh_inference_executed": False}}
        actual_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        require(actual_bytes + len(json.dumps(record).encode("utf-8")) + 4096 <= budget.output_limit, "Output cap exceeded")
        # Exact original paths/cache positions remain outside the portable fixture.
        private = {"contract": "kion-private-fixture-provenance-v1", "state": "completed", "positions": list(POSITIONS),
                   "cache_manifest": request["cache_manifest"], "evaluation_manifest": request["evaluation_manifest"],
                   "source_arrays": planned["arrays"], "source_oracles": planned["oracles"],
                   "source_bindings": planned["bindings"], "source_checkpoints": planned["checkpoints"],
                   "source_teacher": planned["teacher"], "source_teacher_identity": planned["teacher_pin"],
                   "output_dir": output.as_posix(), "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest(),
                   "include_in_delivery": False}
        write_json(planned["private"], private)
        write_json(output / "manifest.json", record)
        require(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) <= budget.output_limit, "Final output cap exceeded")
        return record
    finally:
        for value in opened:
            if getattr(value, "_mmap", None) is not None:
                value._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    record = run(read_json(args.request))
    print(json.dumps({"contract": record["contract"], "state": record["state"], "rows": record["rows"]}))


if __name__ == "__main__":
    main()
