"""Standard-library integrity checks, not a numerical replay claim."""
from __future__ import annotations

import ast
from pathlib import Path

from .io import MAX_TOTAL, read_json, require, sha256, verify_array, verify_descriptor, verify_torch_archive
from .rm import validate_metadata

ARMS = ("C", "RL-completion", "RL-residual")
SCHEMA = "71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d"
SOURCE = ["current_detached_gpt_logit", "log1p_strict7d_count"]
GPT_CONFIG = {"vocab_size": 15418, "context_length": 64, "hidden_size": 256,
              "layers": 4, "heads": 4, "user_feature_dim": 3252}


def verify_runtime(bundle):
    bundle = Path(bundle).resolve(strict=True)
    root = bundle / "reproduction" / "kion_replay"
    manifest_path = bundle / "reproduction" / "runtime-manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("contract") == "kion-portable-runtime-v1", "Wrong runtime manifest contract")
    entries = manifest.get("files")
    require(type(entries) is list and 0 < len(entries) <= 100, "Invalid runtime file list")
    declared = set()
    for entry in entries:
        path = verify_descriptor(bundle, entry)
        require(path.resolve().is_relative_to(root.resolve()) and path.suffix == ".py", "Runtime manifest must contain only package Python sources")
        require(entry["path"] not in declared, "Duplicate runtime path")
        declared.add(entry["path"])
        ast.parse(path.read_text(encoding="utf-8"), filename=entry["path"])
    observed = {path.relative_to(bundle).as_posix() for path in root.rglob("*.py")}
    require(observed == declared, "Runtime source closure differs (missing or extra Python file)")
    imported_root = Path(__file__).resolve().parent
    require(imported_root == root.resolve(), "Executed runtime is outside the selected bundle")
    return {"manifest_sha256": sha256(manifest_path), "files": entries, "python_ast_parsed": len(entries)}


def _shape(manifest, name, dtype, shape):
    entry = manifest["arrays"].get(name)
    require(entry is not None and entry.get("dtype") == dtype and entry.get("shape") == shape,
            "Fixture shape/dtype differs: " + name)


def validate_fixture_semantics(manifest):
    """This named scientific replay is the fixed first4 fixture, not arbitrary inputs."""
    require(manifest.get("contract") == "kion-portable-fixture-v1" and manifest.get("state") == "completed",
            "Wrong/incomplete fixture contract")
    require(type(manifest.get("rows")) is int and manifest["rows"] == 4, "Exactly first4 frozen rows required")
    require(manifest.get("raw_width") == 1068 and manifest.get("candidate_width") == 1024
            and manifest.get("feature_schema_sha256") == SCHEMA and manifest.get("source_columns") == SOURCE,
            "Fixture scientific schema differs")
    require(manifest.get("no_target_injection") is True and manifest.get("no_seen_filter") is True
            and manifest.get("target_labels_included") is False and manifest.get("tie_break") == "score-desc/rawID-asc"
            and manifest.get("temperature") == 1.0 and manifest.get("precision") == "cuda-fp16",
            "Frozen scientific metadata differs")
    require(manifest.get("selection") == "first4-frozen-screen-cache-positions-outcome-independent",
            "Outcome-independent first4 selection required")
    require(manifest.get("head_batch_max") == 16 and manifest.get("candidate_chunk") == 64
            and manifest.get("producer_layout") == [{"size": 16, "slot": slot} for slot in range(4)],
            "Original B16 slots0..3 and chunk64 required")
    table = manifest.get("arrays", {}).get("item_table", {})
    require(table.get("dtype") == "float32" and table.get("shape") == [15418, 256], "Exact frozen item-table dimensions required")
    gpt = manifest.get("gpt", {})
    require(gpt.get("config") == GPT_CONFIG and gpt.get("producer_batch_size") == 16
            and gpt.get("row_slots") == list(range(4)), "GPT and head producer layout/config differ")


def verify_fixture(fixture_root):
    fixture_root = Path(fixture_root).resolve(strict=True)
    manifest_path = fixture_root / "manifest.json"
    manifest = read_json(manifest_path)
    validate_fixture_semantics(manifest)
    rows = manifest["rows"]
    arrays = manifest.get("arrays")
    require(type(arrays) is dict and len(arrays) <= 40, "Invalid fixture arrays")
    for name, dtype, shape in (
        ("encoded", "float32", [rows, 256]), ("candidate_ids", "int32", [rows, 1024]),
        ("candidate_raw_ids", "int64", [rows, 1024]), ("base_logits", "float32", [rows, 1024]),
        ("raw_features", "float32", [rows, 1024, 1068]), ("source_values", "float32", [rows, 1024, 2])):
        _shape(manifest, name, dtype, shape)
    weights = manifest.get("weights")
    require(type(weights) is dict and all(name in weights for name in ARMS), "All three scientific frozen heads required")
    entries = []
    for name, entry in arrays.items():
        verify_array(fixture_root, entry)
        entries.append(entry)
    for name, entry in weights.items():
        path = verify_descriptor(fixture_root, entry)
        if entry.get("format") == "torch-state-dict-v1":
            verify_torch_archive(path)
        elif name not in ("RM", "rm"):
            raise ValueError("Only inference-only torch state weights are supported")
        entries.append(entry)
    expected = manifest.get("expected", {})
    require(type(expected) is dict and set(expected).issubset(set(ARMS) | {"encoded", "base_logits", "RM"}), "Unknown validation-only oracle")
    for name, entry in expected.items():
        require(entry.get("role") in ("validation-only", "validation-only-never-computational-input"), "Oracle must be marked validation-only")
        verify_array(fixture_root, entry)
        entries.append(entry)
    require(sum(entry["bytes"] for entry in entries) <= MAX_TOTAL, "Combined fixture byte budget exceeded")
    paths = [entry["path"] for entry in entries]
    require(len(paths) == len(set(paths)), "One fixture file cannot have multiple roles")
    if "gpt" in manifest:
        gpt = manifest["gpt"]
        batch = gpt.get("producer_batch_size")
        require(type(batch) is int and 1 <= batch <= 16 and "GPT" in weights, "GPT producer batch/weights required")
        slots = gpt.get("row_slots")
        require(type(slots) is list and len(slots) == rows and len(set(slots)) == rows
                and all(type(slot) is int and 0 <= slot < batch for slot in slots), "Invalid GPT selected row slots")
        _shape(manifest, "gpt_x", "int64", [batch, 64])
        _shape(manifest, "gpt_lengths", "int64", [batch])
        _shape(manifest, "gpt_user_features", "float32", [batch, 3252])
    return manifest, {"manifest_sha256": sha256(manifest_path), "files": len(entries),
                      "bytes": sum(entry["bytes"] for entry in entries), "rows": rows,
                      "numerical_execution": False}


def verify_rm_fixture(fixture_root):
    """RM has its own feature schema and numerical environment, not raw1068."""
    fixture_root = Path(fixture_root).resolve(strict=True)
    manifest_path = fixture_root / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("contract") == "kion-portable-rm-fixture-v1"
            and manifest.get("state") == "completed", "Wrong/incomplete frozen RM fixture")
    validate_metadata(manifest.get("rm"))
    arrays = manifest.get("arrays")
    require(type(arrays) is dict, "RM arrays required")
    spec = {"candidate_raw_ids": ("int64", [4, 1024]),
            "rm_context_numeric": ("float32", [4, 2079]),
            "rm_candidate_numeric": ("float32", [4, 1024, 2008]),
            "rm_fit_counts": ("int64", [4, 1024]), "rm_item_prior": ("float64", [4, 1024]),
            "rm_reference_reward": ("float32", [4, 1024]),
            "rm_reference_reward64": ("float64", [4, 1024]),
            "rm_reference_support": ("bool", [4, 1024])}
    require(set(arrays) == set(spec), "Unexpected/missing RM arrays")
    entries = []
    for name, (dtype, shape) in spec.items():
        _shape(manifest, name, dtype, shape)
        verify_array(fixture_root, arrays[name])
        entries.append(arrays[name])
    require(type(manifest.get("weights")) is dict and set(manifest["weights"]) == {"RM"}, "One frozen RM model required")
    model = manifest["weights"]["RM"]
    require(model.get("sha256") == "b6ed4c6427129afcddec017e2f64d9c3fb59fb31fc300e5c9d074964a21ddd62"
            and model.get("bytes") == 163368, "Frozen CatBoost identity differs")
    verify_descriptor(fixture_root, model)
    entries.append(model)
    require(sum(entry["bytes"] for entry in entries) <= MAX_TOTAL, "RM fixture byte budget exceeded")
    require(len({entry["path"] for entry in entries}) == len(entries), "RM file cannot have multiple roles")
    return manifest, {"manifest_sha256": sha256(manifest_path), "files": len(entries),
                      "bytes": sum(entry["bytes"] for entry in entries), "rows": 4, "numerical_execution": False}


def verify(bundle, fixture_root=None):
    result = {"contract": "kion-portable-verify-v1", "runtime": verify_runtime(bundle),
              "stdlib_only": True, "training_updates": 0, "numerical_execution": False}
    if fixture_root is not None:
        contract = read_json(Path(fixture_root) / "manifest.json").get("contract")
        validator = verify_rm_fixture if contract == "kion-portable-rm-fixture-v1" else verify_fixture
        _, result["fixture"] = validator(fixture_root)
    return result
