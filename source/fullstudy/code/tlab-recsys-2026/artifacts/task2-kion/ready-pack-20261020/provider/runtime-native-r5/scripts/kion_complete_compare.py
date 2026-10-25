"""Pinned CPU comparison of CompleteGPT selected exports on the fixed development rows.

The default command records readiness only. --execute consumes an explicitly
pinned request after the selected export exists. No model modules, weights,
raw data, test split, network, fit or inference are used.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np


CONTRACT = "kion-complete-comparison-v1"
REQUEST_CONTRACT = "kion-complete-comparison-request-v1"
FULL_VALIDATION_ROWS = 158760
ROW_KEYS = ("user_id", "target_item_id", "target_date")
PREPARED_FIELDS = ("x", "lengths", "y", *ROW_KEYS)
EXPORT_FILES = ("predictions.npz", "config.json", "metrics.json", "completion.json", "selected.json")
METRICS = ("map@10", "recall@10", "ndcg@10")
CUTOFF = date(2021, 8, 9).toordinal()
ATOL = 2e-7
HELPER_SHA256 = "49cb08603ec40ce21d11e2b08190d750fbe49ec10d1bad56fb22916362c0ce2c"
REVIEWED_COMPLETE_SOURCES = {
    "kion_complete_train.py": "0cda8e6341011c19ec013901b940d3aaeae394e4d439cc067582e439b1f34a87",
    "kion_complete_model.py": "8d1be2352cd268481f4a1f7b764943f745e910695a1a270fd3d0b1c4e7ec24e7",
}
COMPLETE_SOURCES = (*REVIEWED_COMPLETE_SOURCES, "kion_train.py", "kion_full_train.py",
                    "kion_conditioned_train.py", "kion_people_embeddings.py", "kion_pack_people.py")
LEGACY_CONDITIONED = {"1d6fed9ce10f64d399e3c9d3d2c27f21ffc288a63b8eb1d0af5b462353bba13f",
                      "b6c2c93bb8ac1475d00a1a20cd8d278da898468ae38a001cc5be4ec218478caf"}
PACK_EXPORTER = "0f11d3c1bcc5165dfd9f35a7728b9400b679cd51c8bb444af04360643711546a"
FAMILIES = {"local_base", "local_dynamic", "semantic_extensions", "text3fields", "people_embeddings",
            "personal_candidates", "audience", "people_dynamics", "franchise_progress", "external_facts",
            "external_interest", "entity_links"}
EVALUATION_OBJECTIVE = "unforced exact retrieval topK then pair rerank; all unknown targets are misses; no seen mask"
PROTOCOL = {"query": "one validation row, one next-film positive; MAP@10 equals truncated MRR@10",
    "validation": "2021-08-09..2021-08-15", "history": "BOS+H63; frozen before 2021-08-09",
    "catalog": "tokens 3..vocab_size-1", "seen_mask": False, "forced_evaluation_target": False,
    "unknown_targets": "zero misses in every full-row and user denominator",
    "bootstrap": {"replicates": 1000, "seed": 2026, "unit": "paired user cluster",
                  "percentiles": [0.025, 0.975], "quantile_method": "linear"}}
CRITERIA = {"practical_delta_ndcg10": 0.001, "exploratory_delta_ndcg10": 0.0005,
            "recall_guard_delta": -0.001, "role": "descriptive compute priority; no automatic promotion"}
LIMITS = ("Repeated development week and adaptive checkpoint/feature selection; CI is conditional on these selected models. "
          "User bootstrap does not measure training-seed variance. New architecture, many features, auxiliary objective and "
          "changed SGD order/tail weights prevent a feature-only causal interpretation. ODS 0.0910 is a different protocol "
          "and is not a comparable baseline. No confirmatory superiority, test or online claim.")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _loaded_source_identity(module_code):
    """Reject stale timestamp-pyc code before attributing current source bytes."""
    path = Path(__file__).resolve()
    payload = path.read_bytes()
    compiled = compile(payload, module_code.co_filename, "exec", dont_inherit=True,
                       optimize=sys.flags.optimize)
    if compiled != module_code:
        raise ValueError("Comparator executing code differs from source bytes; stale import cache")
    return path, hashlib.sha256(payload).hexdigest()


# The executing module's code is compared with a fresh compile of the very bytes
# being hashed. This covers stale pyc imports as well as later import->edit drift.
LOADED_SOURCE_PATH, LOADED_SOURCE_SHA256 = _loaded_source_identity(sys._getframe().f_code)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(value):
    value = np.asarray(value)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True,
                        separators=(",", ":")).encode("ascii")
    result = hashlib.sha256(header + b"\n")
    result.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return result.hexdigest()


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": sha256(path)}


def read_json(path):
    payload = Path(path).read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def verify_pin(value, consumed, name=None):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"} or
            not isinstance(value.get("path"), str) or not digest_ok(value.get("sha256"))):
        raise ValueError("Missing/invalid exact file pin")
    path = Path(value["path"])
    if (not path.is_absolute() or path.resolve().as_posix() != value["path"] or
            (name is not None and path.name != name) or path.suffix.lower() in (".pt", ".pth", ".pkl", ".pickle") or
            path.name.lower() == "test.npz"):
        raise ValueError("Noncanonical/disallowed pinned file path")
    if sha256(path) != value["sha256"]:
        raise ValueError("Pinned file SHA mismatch: " + path.as_posix())
    consumed.append(dict(value))
    return path


def load_helper(value, consumed):
    path = verify_pin(value, consumed, "kion_compare_predictions.py")
    if value["sha256"] != HELPER_SHA256:
        raise ValueError("Unreviewed legacy helper")
    # Compile the bytes whose SHA was checked, never a stale or substituted pyc.
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != HELPER_SHA256:
        raise ValueError("Legacy helper changed before loading")
    module = types.ModuleType("_complete_comparison_reviewed_helper")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    for name in ("load_export", "metadata_contract", "checkpoint_binding", "sha256", "read_json_snapshot"):
        function = getattr(module, name)
        if function.__code__.co_filename != str(path) or function.__globals__ is not module.__dict__:
            raise ValueError("Legacy helper function origin mismatch")
    if module.np is not np or module.Path is not Path or sha256(path) != HELPER_SHA256:
        raise ValueError("Legacy helper dependency/source mismatch")
    return module


def check_request(request):
    if (not isinstance(request, dict) or request.get("contract") != REQUEST_CONTRACT or
            request.get("protocol") != PROTOCOL or request.get("criteria") != CRITERIA or
            type(request.get("expected_rows")) is not int or request["expected_rows"] != FULL_VALIDATION_ROWS):
        raise ValueError("Invalid comparison request/protocol/full row scope")
    if set(request.get("prepared", {})) != {"metadata.json", "train.npz", "val.npz"}:
        raise ValueError("Explicit prepared metadata/train/val pins required")
    baselines = request.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        raise ValueError("At least the current local-pack baseline is required")
    names = [entry.get("name") for entry in baselines]
    if len(set(names)) != len(names) or "local_pack_sft" not in names:
        raise ValueError("Unique baseline names and local_pack_sft required")
    if not next(entry for entry in baselines if entry["name"] == "local_pack_sft").get("required"):
        raise ValueError("Current baseline cannot be optional")


def missing_artifacts(request):
    missing = []
    entries = [request["candidate"], *[b for b in request["baselines"] if b.get("required")]]
    for model in entries:
        directory = Path(model["model_dir"])
        files = model.get("files", {})
        for name in EXPORT_FILES:
            item = files.get(name)
            if not isinstance(item, dict) or not digest_ok(item.get("sha256")) or not Path(item.get("path", "")).is_file():
                missing.append({"path": (directory / name).as_posix(), "reason": "missing_or_unpinned_selected_export"})
    provider = request["candidate"].get("provider_manifest")
    if not isinstance(provider, dict) or not digest_ok(provider.get("sha256")) or not Path(provider.get("path", "")).is_file():
        missing.append({"path": provider.get("path") if isinstance(provider, dict) else None,
                        "reason": "missing_or_unpinned_full_provider_manifest"})
    return missing


def load_prepared(request, helper, consumed):
    prepared = request["prepared"]
    paths = {name: verify_pin(value, consumed, name) for name, value in prepared.items()}
    metadata, meta_digest = read_json(paths["metadata.json"])
    if meta_digest != prepared["metadata.json"]["sha256"]:
        raise ValueError("Metadata changed while loading")
    vocab, mapping, count = helper.metadata_contract(metadata)
    if count != request["expected_rows"] or metadata.get("context") != 64:
        raise ValueError("Wrong fixed validation/H63 scope")
    with np.load(paths["val.npz"], allow_pickle=False) as archive:
        if not set(PREPARED_FIELDS).issubset(archive.files):
            raise ValueError("Missing prepared row/context fields")
        data = {name: archive[name] for name in PREPARED_FIELDS}
    if sha256(paths["val.npz"]) != prepared["val.npz"]["sha256"]:
        raise ValueError("Prepared validation changed while loading")
    for name, values in data.items():
        shape = (count, 64) if name == "x" else (count,)
        if values.dtype != np.int64 or values.shape != shape:
            raise ValueError("Invalid prepared dtype/shape: " + name)
    lengths, x = data["lengths"], data["x"]
    active = np.arange(64)[None, :] < lengths[:, None]
    if (np.any((lengths < 2) | (lengths > 64)) or np.any(x[:, 0] != 1) or
            np.any(x[~active] != 0) or np.any((x[active] < 1) | (x[active] >= vocab)) or
            np.any((x[:, 1:] == 1) & active[:, 1:]) or np.any(~(x[:, 1:] >= 3).any(1))):
        raise ValueError("Invalid BOS/H63 history or padding")
    if any(np.any(data[name] < 0) for name in ROW_KEYS):
        raise ValueError("Negative prepared row keys")
    if np.any((data["target_date"] < CUTOFF) | (data["target_date"] >= CUTOFF + 7)):
        raise ValueError("Prepared rows outside fixed development week")
    tokens = np.fromiter((mapping.get(int(item), 2) for item in data["target_item_id"]), np.int64, count=count)
    if not np.array_equal(tokens, data["y"]):
        raise ValueError("Prepared target/vocabulary mismatch")
    order = np.lexsort(tuple(data[name] for name in reversed(ROW_KEYS)))
    keys = np.column_stack([data[name][order] for name in ROW_KEYS])
    if np.any(np.all(keys[1:] == keys[:-1], axis=1)):
        raise ValueError("Duplicate prepared row keys")
    binding = {"fields": {name: array_hash(values) for name, values in data.items()},
               "provider_row_keys_sha256": canonical({name: array_hash(values) for name, values in data.items()}),
               "validation_origin": "2021-08-09", "origin_keys_sha256": array_hash(np.full(count, CUTOFF, np.int64)),
               "verification": "prepared H63 bytes plus exporter/provider source and config bindings; H63 not independently exported"}
    return metadata, data, binding


def export_files(declaration, consumed):
    directory = Path(declaration["model_dir"])
    if not directory.is_absolute() or directory.resolve().as_posix() != declaration["model_dir"]:
        raise ValueError("Canonical model directory required")
    if set(declaration.get("files", {})) != set(EXPORT_FILES):
        raise ValueError("Exact selected export file pins required")
    for name, value in declaration["files"].items():
        path = verify_pin(value, consumed, name)
        if path.parent != directory:
            raise ValueError("Selected export pin belongs to a different model directory")
    return directory


def validate_sources(declaration, contract, consumed, complete=False):
    declared = contract.get("sources_sha256", {})
    sources = declaration.get("sources", {})
    if not isinstance(declared, dict) or not declared or set(sources) != set(declared):
        raise ValueError("Exact declared exporter source pins required")
    if complete and set(declared) != set(COMPLETE_SOURCES):
        raise ValueError("Wrong complete runner source schema")
    if not complete:
        if declared.get("kion_conditioned_train.py") not in LEGACY_CONDITIONED:
            raise ValueError("Unreviewed legacy evaluator source")
        if "kion_pack_train.py" in declared and declared["kion_pack_train.py"] != PACK_EXPORTER:
            raise ValueError("Unreviewed local-pack exporter source")
    for name, digest in declared.items():
        if not name.endswith(".py") or sources[name].get("sha256") != digest:
            raise ValueError("Source declaration/pin mismatch")
        if complete and name in REVIEWED_COMPLETE_SOURCES and digest != REVIEWED_COMPLETE_SOURCES[name]:
            raise ValueError("Unreviewed complete evaluator/model source")
        verify_pin(sources[name], consumed, name)


def align_export(data, prepared):
    count = len(prepared["y"])
    indices = data.get("row_index")
    if (indices is None or indices.dtype != np.int64 or indices.shape != (count,) or
            not np.array_equal(np.sort(indices), np.arange(count))):
        raise ValueError("Export must cover every validation row_index once")
    order = np.argsort(indices)
    for export_name, prepared_name in ((*[(n, n) for n in ROW_KEYS], ("target_token", "y"))):
        value = data.get(export_name)
        if (value is None or value.dtype != np.int64 or value.shape != (count,) or
                not np.array_equal(value[order], prepared[prepared_name])):
            raise ValueError("Export/prepared row mismatch: " + export_name)
    known = prepared["y"] >= 3
    if (data.get("known_target") is None or data["known_target"].dtype != np.bool_ or
            data["known_target"].shape != (count,) or not np.array_equal(data["known_target"][order], known)):
        raise ValueError("Export/prepared known-target mismatch")
    return {name: values[order] for name, values in data.items()}


def rank_scores(top, targets, vocab):
    k = min(10, vocab - 3)
    if (top.dtype != np.int64 or top.shape != (len(targets), k) or np.any((top < 3) | (top >= vocab)) or
            np.any(np.diff(np.sort(top, axis=1), axis=1) == 0)):
        raise ValueError("Invalid/duplicate ranked known candidate tokens")
    match = (top == targets[:, None]) & (targets >= 3)[:, None]
    return np.column_stack(((match / np.arange(1, k + 1)).sum(1), match.any(1).astype(np.float64),
                            (match / np.log2(np.arange(k) + 2)).sum(1)))


def close_metric(saved, expected, name):
    if (isinstance(saved, bool) or not isinstance(saved, (int, float)) or not np.isfinite(saved) or
            abs(saved - expected) > ATOL):
        raise ValueError("Cached metric mismatch: " + name)


def validate_provider(declaration, contract, request, prepared_binding, consumed):
    path = verify_pin(declaration["provider_manifest"], consumed, "manifest.json")
    manifest, digest = read_json(path)
    if digest != declaration["provider_manifest"]["sha256"]:
        raise ValueError("Provider manifest changed while reading")
    signature = canonical({key: value for key, value in manifest.items() if key != "signature"})
    provider, architecture = contract.get("provider", {}), contract.get("architecture", {})
    if (manifest.get("contract") != "kion-complete-pack-v1" or manifest.get("state") != "completed" or
            manifest.get("integration_mode") != "full" or manifest.get("signature") != signature or
            provider.get("manifest_signature") != signature or provider.get("status") != "admitted_full" or
            architecture.get("source_contract") != provider or
            contract.get("integration") != {"mode": "admitted-full-package", "status": "admitted_full", "full_package_admitted": True}):
        raise ValueError("Complete full-provider admission/config mismatch")
    if manifest.get("data", {}).get("files") != request["prepared"]:
        raise ValueError("Complete provider prepared data pins mismatch")
    acceptance = provider.get("family_acceptance", {})
    if (set(acceptance) != FAMILIES or any(value != "accepted_full" for value in acceptance.values()) or
            not FAMILIES.issubset(provider.get("materialized_families", []))):
        raise ValueError("Missing full feature family admission")
    if (provider.get("fit_cutoff") != "2021-08-09" or provider.get("validation_origin") != "2021-08-09" or
            provider.get("row_keys_sha256", {}).get("val") != prepared_binding["provider_row_keys_sha256"] or
            manifest.get("row_keys_sha256", {}).get("val") != prepared_binding["provider_row_keys_sha256"] or
            provider.get("origin_keys", {}).get("sha256", {}).get("val") != prepared_binding["origin_keys_sha256"] or
            provider.get("origin_keys", {}).get("val") != "2021-08-09"):
        raise ValueError("Complete provider row/user/origin/H63 binding mismatch")
    columns = provider.get("feature_columns", {})
    if (set(columns) != {"item", "user", "candidate"} or columns != architecture.get("feature_columns") or
            provider.get("schema_hash") != canonical(columns) or architecture.get("schema_hash") != canonical(columns)):
        raise ValueError("Complete feature schema mismatch")
    for names in columns.values():
        if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("Invalid complete named feature columns")
    source_pins = manifest.get("source_pins", [])
    if not isinstance(source_pins, list) or not source_pins:
        raise ValueError("Full provider source pins missing")
    provider_sources = []
    for value in source_pins:
        if Path(value.get("path", "")).suffix == ".py":
            verify_pin(value, consumed)
            provider_sources.append(value)
    if not any(Path(p["path"]).name == "kion_complete_pack.py" for p in provider_sources):
        raise ValueError("Physical provider source pin missing")
    return {"manifest": declaration["provider_manifest"], "provider_sources": provider_sources}


def load_complete(declaration, request, metadata, prepared, prepared_binding, consumed):
    directory = export_files(declaration, consumed)
    values = {}
    for name in EXPORT_FILES:
        if name.endswith(".json"):
            values[name], digest = read_json(directory / name)
            if digest != declaration["files"][name]["sha256"]:
                raise ValueError("Complete JSON snapshot differs from pinned bytes")
    config, metrics, completion, selected = [values[n] for n in ("config.json", "metrics.json", "completion.json", "selected.json")]
    contract = config.get("contract", {})
    validate_sources(declaration, contract, consumed, complete=True)
    provider = validate_provider(declaration, contract, request, prepared_binding, consumed)
    architecture, training = contract.get("architecture", {}), contract.get("training", {})
    mode = training.get("scoring_mode")
    width = training.get("retrieval_topk")
    vocab, count = metadata["vocab_size"], len(prepared["y"])
    if (mode not in ("retrieval", "rerank") or type(width) is not int or width < min(10, vocab - 3) or
            architecture.get("vocab_size") != vocab or architecture.get("context_length") != 64 or
            contract.get("prediction_row_keys") != list(ROW_KEYS) or
            contract.get("objective", {}).get("evaluation") != EVALUATION_OBJECTIVE or
            training.get("training_order") not in ("origin-grouped-v1", "legacy-bucketed") or
            contract.get("training_schedule", {}).get("mode") != training.get("training_order")):
        raise ValueError("Complete scoring/candidate-mask/context/training-order contract mismatch")
    width = min(width, vocab - 3)
    saved = metrics.get("validation", {})
    epoch, updates = selected.get("epoch"), metrics.get("state", {}).get("updates")
    filename = f"selected-epoch{epoch:04d}.pt" if type(epoch) is int else None
    hashes = completion.get("sha256", {})
    if (type(epoch) is not int or epoch < 1 or type(updates) is not int or updates < 1 or
            type(metrics.get("state", {}).get("selected_epoch")) is not int or
            type(completion.get("selected_epoch")) is not int or type(completion.get("updates")) is not int or
            metrics.get("state", {}).get("selected_epoch") != epoch or completion.get("selected_epoch") != epoch or
            completion.get("updates") != updates or metrics.get("status") not in ("completed", "stopped") or
            completion.get("state") != metrics.get("status") or metrics.get("test_accessed") is not False or
            completion.get("test_accessed") is not False or completion.get("learned_status") != "optimizer-updated" or
            metrics.get("predictions_status") != "exported-selected-checkpoint" or
            completion.get("predictions_status") != "exported-selected-checkpoint" or
            selected.get("file") != filename or not digest_ok(selected.get("sha256")) or
            hashes.get(filename) != selected.get("sha256") or selected.get("validation") != saved or
            completion.get("validation") != saved or metrics.get("integration") != contract.get("integration") or
            completion.get("integration") != contract.get("integration")):
        raise ValueError("Complete selected checkpoint/receipt/status mismatch")
    for name in ("predictions.npz", "metrics.json", "selected.json"):
        if hashes.get(name) != declaration["files"][name]["sha256"]:
            raise ValueError("Complete completion export/metrics/selected digest mismatch")
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        required = {*ROW_KEYS, "row_index", "target_token", "known_target", "top_items", "retrieval_top_items",
                    "rerank_top_items", "retrieval_covered", "hit_at_10", "ndcg_at_10"}
        if not required.issubset(archive.files):
            raise ValueError("Missing complete prediction members")
        data = {name: archive[name] for name in required}
    if sha256(directory / "predictions.npz") != declaration["files"]["predictions.npz"]["sha256"]:
        raise ValueError("Complete predictions changed while loading")
    data = align_export(data, prepared)
    scores = {name: rank_scores(data[name + "_top_items"], prepared["y"], vocab) for name in ("retrieval", "rerank")}
    covered = data["retrieval_covered"]
    if (covered.dtype != np.bool_ or covered.shape != (count,) or np.any(covered & (prepared["y"] < 3)) or
            np.any(scores["retrieval"][:, 1] > covered) or np.any(scores["rerank"][:, 1] > covered)):
        raise ValueError("Invalid unforced retrieval coverage/known mask")
    if width == min(10, vocab - 3) and (not np.array_equal(covered, scores["retrieval"][:, 1].astype(bool)) or
            not np.array_equal(np.sort(data["retrieval_top_items"], axis=1), np.sort(data["rerank_top_items"], axis=1))):
        raise ValueError("Rerank or coverage outside fully exported candidate set")
    if not np.array_equal(data["top_items"], data[mode + "_top_items"]):
        raise ValueError("Selected scoring-mode ranking mismatch")
    for name, expected in (("hit_at_10", scores[mode][:, 1]), ("ndcg_at_10", scores[mode][:, 2])):
        cached = data[name]
        if (cached.shape != (count,) or cached.dtype.kind not in "bfi" or not np.isfinite(cached).all() or
                (name == "hit_at_10" and np.any((cached != 0) & (cached != 1))) or
                not np.allclose(cached, expected, rtol=0, atol=ATOL)):
            raise ValueError("Complete per-row metric cache mismatch")
    known = int((prepared["y"] >= 3).sum())
    for name, expected in {"examples": count, "known_targets": known, "unknown_target_misses": count - known,
                           "actual_k": min(10, vocab - 3), "retrieval_topk": width}.items():
        if type(saved.get(name)) is not int or saved[name] != expected:
            raise ValueError("Complete aggregate scope mismatch: " + name)
    if saved.get("scoring_mode") != mode:
        raise ValueError("Complete aggregate scoring-mode mismatch")
    for policy in ("retrieval", "rerank"):
        for name, index in (("recall@10", 1), ("ndcg@10", 2)):
            close_metric(saved.get(policy, {}).get(name), float(scores[policy][:, index].mean()), policy + name)
    for name, index in (("recall@10", 1), ("ndcg@10", 2)):
        close_metric(saved.get(name), float(scores[mode][:, index].mean()), name)
    close_metric(saved.get("retrieval_coverage_all_rows"), float(covered.mean()), "coverage_all")
    if known:
        close_metric(saved.get("retrieval_coverage_known_rows"), float(covered.sum() / known), "coverage_known")
        ce = saved.get("cross_entropy_known")
        if isinstance(ce, bool) or not isinstance(ce, (int, float)) or not np.isfinite(ce) or ce < 0:
            raise ValueError("Invalid known-target cross entropy")
    elif saved.get("retrieval_coverage_known_rows") is not None or saved.get("cross_entropy_known") is not None:
        raise ValueError("Known-only metrics must be null without known targets")
    binding = {"files": declaration["files"], "config_contract": contract, "provider": provider,
               "checkpoint": {"file": filename, "sha256": selected["sha256"], "epoch": epoch,
                              "verification": "consistent pinned receipts; model weights not opened or hashed"}}
    coverage = {"topk": width, "all_rows": float(covered.mean()), "known_rows": float(covered.sum() / known) if known else None,
                "covered_rows": int(covered.sum()), "verification": "source/receipt-bound coverage flag; full topK IDs not exported",
                "independent_full_topk_membership_verified": False}
    return scores, binding, coverage


def load_baseline(declaration, request, metadata, prepared, helper, consumed):
    directory = export_files(declaration, consumed)
    data, _, _, binding = helper.load_export(directory / "predictions.npz", metadata,
                                            request["prepared"]["metadata.json"]["sha256"])
    for value in [binding[n] for n in ("predictions", "config", "metrics")] + list(binding["receipts"].values()):
        if value != declaration["files"].get(Path(value["path"]).name):
            raise ValueError("Legacy export snapshot differs from pinned bytes")
    validate_sources(declaration, binding["declared_contract"], consumed)
    expected_inputs = {name: value["sha256"] for name, value in request["prepared"].items()}
    if binding["declared_contract"]["inputs_sha256"] != expected_inputs:
        raise ValueError("Baseline prepared binding mismatch")
    data = align_export(data, prepared)
    return rank_scores(data["top_items"], prepared["y"], metadata["vocab_size"]), binding


def bootstrap(delta, sizes):
    """Uniform user draws shared across all metrics; row ratio and user means."""
    rng = np.random.default_rng(PROTOCOL["bootstrap"]["seed"])
    replicates, users = PROTOCOL["bootstrap"]["replicates"], len(sizes)
    row, user = np.empty((replicates, delta.shape[1])), np.empty((replicates, delta.shape[1]))
    means = delta / sizes[:, None]
    batch = max(1, min(10, 2_000_000 // users))
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draws = rng.integers(users, size=(stop - start, users), dtype=np.int64)
        row[start:stop] = delta[draws].sum(axis=1) / sizes[draws].sum(axis=1)[:, None]
        user[start:stop] = means[draws].mean(axis=1)
    return row, user


def paired_result(baseline, candidate, user_ids):
    users, inverse, sizes = np.unique(user_ids, return_inverse=True, return_counts=True)
    sums = [np.column_stack([np.bincount(inverse, weights=values[:, n]) for n in range(3)])
            for values in (baseline, candidate)]
    row_draws, user_draws = bootstrap(sums[1] - sums[0], sizes)
    result = {}
    for scope, points, draws in (("row_primary", [v.mean(0) for v in (baseline, candidate)], row_draws),
            ("user_macro_companion", [(s / sizes[:, None]).mean(0) for s in sums], user_draws)):
        result[scope] = {name: {"baseline": float(points[0][n]), "candidate": float(points[1][n]),
            "delta_candidate_minus_baseline": float(points[1][n] - points[0][n]),
            "paired_percentile_95_ci": np.quantile(draws[:, n], [.025, .975], method="linear").tolist()}
            for n, name in enumerate(METRICS)}
    row = result["row_primary"]
    result["operational_descriptive_only"] = {
        "practical_ndcg_threshold_met": row["ndcg@10"]["delta_candidate_minus_baseline"] >= CRITERIA["practical_delta_ndcg10"],
        "exploratory_ndcg_threshold_met": row["ndcg@10"]["delta_candidate_minus_baseline"] >= CRITERIA["exploratory_delta_ndcg10"],
        "recall_guard_met": row["recall@10"]["delta_candidate_minus_baseline"] >= CRITERIA["recall_guard_delta"],
        "automatic_model_promotion": False}
    return result, row_draws, user_draws, len(users)


def run(request_path, output, execute=False):
    if sha256(LOADED_SOURCE_PATH) != LOADED_SOURCE_SHA256:
        raise ValueError("Comparator source changed after module load")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Comparison output must not exist")
    request_path = Path(request_path).resolve()
    request, request_digest = read_json(request_path)
    check_request(request)
    consumed = [{"path": request_path.as_posix(), "sha256": request_digest},
                {"path": LOADED_SOURCE_PATH.as_posix(), "sha256": LOADED_SOURCE_SHA256}]
    missing = missing_artifacts(request)
    report = {"contract": CONTRACT, "created_utc": datetime.now(timezone.utc).isoformat(),
              "request": consumed[0], "source": consumed[1], "protocol": PROTOCOL, "criteria": CRITERIA,
              "expected_validation_rows": request["expected_rows"], "limits": LIMITS,
              "safety": {"test_accessed": False, "raw_data_accessed": False, "model_weights_accessed": False,
                         "models_executed": False, "fit_started": False}}
    if missing or not execute:
        report.update(state="waiting_artifact" if missing else "ready_for_comparison", missing=missing,
                      compared=False, candidate_model_dir=request["candidate"]["model_dir"])
        if any(sha256(v["path"]) != v["sha256"] for v in consumed):
            raise ValueError("Readiness input/source changed")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "status.json", report)
        return report
    helper = load_helper(request["legacy_helper"], consumed)
    verify_pin(request["criteria_source"], consumed)
    metadata, prepared, prepared_binding = load_prepared(request, helper, consumed)
    scores, candidate_binding, coverage = load_complete(request["candidate"], request, metadata, prepared, prepared_binding, consumed)
    comparisons, baselines, skipped, draws = {}, {}, [], {}
    for baseline in request["baselines"]:
        absent = [name for name in EXPORT_FILES if not Path(baseline.get("files", {}).get(name, {}).get("path", "")).is_file()]
        if absent and not baseline.get("required"):
            skipped.append({"name": baseline["name"], "reason": "optional real row export unavailable", "missing": absent})
            continue
        baseline_scores, binding = load_baseline(baseline, request, metadata, prepared, helper, consumed)
        baselines[baseline["name"]] = binding
        for mode, candidate_scores in scores.items():
            name = baseline["name"] + "__" + mode
            comparisons[name], row, user, users = paired_result(baseline_scores, candidate_scores, prepared["user_id"])
            draws[name + "__row"], draws[name + "__user"] = row, user
    # Recheck the request, implementation, helper, every export/receipt/source,
    # provider manifest, criteria and prepared bytes after all bootstrap draws.
    for value in consumed:
        if sha256(value["path"]) != value["sha256"]:
            raise ValueError("Comparison input/source changed while computing")
    report.update(state="compared", compared=True, rows=len(prepared["y"]), users=users,
                  known_targets=int((prepared["y"] >= 3).sum()), unknown_target_misses=int((prepared["y"] < 3).sum()),
                  candidate=candidate_binding, baselines=baselines, skipped_optional=skipped,
                  prepared=prepared_binding, prepared_files=request["prepared"], metric_order=list(METRICS),
                  retrieval_coverage=coverage, results=comparisons, consumed_files=consumed)
    report["safety"].update(prepared_validation_arrays_accessed=True, prepared_train_bytes_hash_only=True)
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "bootstrap_deltas.npz"
    np.savez_compressed(artifact, **draws)
    report["bootstrap_artifact"] = pin(artifact)
    write_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Consume final pinned exports and compare; default records readiness only")
    args = parser.parse_args(argv)
    report = run(args.request, args.output, args.execute)
    print(json.dumps({key: report[key] for key in ("state", "compared", "expected_validation_rows")}, indent=2))
    return report


if __name__ == "__main__":
    main()
