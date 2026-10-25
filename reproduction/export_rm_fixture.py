"""Root-only bounded export of four exact natural1024 frozen-RM inputs.

Import is stdlib-only. Native numerical bootstrap and all payload reads occur only
inside run(request, check), under the submission's external process guard.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

CONTRACT = "kion-rm-fixture-export-v1"
BRIDGE_SHA = "097aceb6edd8004cd6d3258955ede50073012918ce6759a73020fcd4e9daa6eb"
CACHE_SHA = "5baf5dae822a7fb812acb6344ac08c0a0f94a2e4af7adf24b1c53c4584096a82"
MODEL_SHA = "b6ed4c6427129afcddec017e2f64d9c3fb59fb31fc300e5c9d074964a21ddd62"
PRIOR_SHA = "d5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4"
MANIFEST_SHA = "dfc61f3ebfb9144be3cb86caf70050b72033d9234119145119dbfe90e0ee8507"
SIGNATURE = "ee14bc4d7375a4d6978a2e1f476cd795369b45ef4250501b2f5992767e9ed261"
SCHEMA = "7776a9d611b04ef8f3ef77fce0a67c1fb045f67142bf4b4bbde24cc469f9bda4"
MAPPING = "b90df0d1fcce007008b62007e486bc1e9cb0b4ac89c97bb5107e84305e744db4"
REWARD_META_SHA = "a4bccf42b9f020e72a191c536df59dce5ac4e21b7e38aee4e21edccab5c05676"
SERVING_RECEIPT_SHA = "992c7a8ce3cf45a0f20605dd60f9ac3a11cb51424d2180f9a4cef3f1a08d7b35"
STATE_FIELDS = ("user_ids", "origins", "x", "lengths", "candidate_raw_ids")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, limit=2 << 20):
    path = Path(path)
    require(path.stat().st_size <= limit, "Oversized metadata")
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def checked_pins(pins):
    require(isinstance(pins, dict) and pins, "Nonempty explicit file pins required")
    for path, expected in pins.items():
        require(isinstance(expected, str) and len(expected) == 64 and sha(path) == expected,
                "Pinned file differs: " + str(path))


def validate_request(request):
    require(request.get("contract") == CONTRACT, "Wrong RM export contract")
    require(request.get("row_positions") == [0, 1, 2, 3], "Only fixed first four states authorized")
    for name in ("supervised_updates", "new_rm_fits", "final_test_reads", "rl_updates"):
        require(type(request.get(name)) is int and request[name] == 0, "No-training counter required: " + name)
    for name, maximum in (("init_seconds", 300), ("slice_seconds", 60), ("maximum_seconds", 480)):
        value = request.get(name)
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum,
                "Invalid bounded duration: " + name)
    output = Path(request["fixture_output"]).resolve()
    owned = Path(__file__).resolve().parent
    require(output.is_relative_to(owned) and output != owned and not output.exists(),
            "New owned reproduction output required")
    require(request.get("cache_sha256") == CACHE_SHA and request.get("bridge_sha256") == BRIDGE_SHA,
            "Exact immutable cache/bridge required")
    return output


def run(request, check=None):
    require(callable(check), "Root-owned external resource guard is required")
    output = validate_request(request)
    start = time.perf_counter()
    check(stage="rm-export-validate")
    require(not any(name == "numpy" or name.startswith("numpy.") or name == "catboost"
                    or name.startswith("catboost.") or name == "torch" or name.startswith("torch.")
                    for name in sys.modules), "Fresh isolated RM process required")
    checked_pins(request["source_hashes"])
    checked_pins(request["input_hashes"])
    require(request["source_hashes"].get(Path(__file__).resolve().as_posix()) == sha(__file__),
            "Exporter source pin required")
    repo = Path(request["repo"]).resolve()
    bridge_path = Path(request["bridge_path"]).resolve()
    require(bridge_path == repo / "artifacts/task2-kion/c-rl-20261024-r1/reward/bridge.py"
            and sha(bridge_path) == BRIDGE_SHA, "Wrong original bridge")
    cache_root = Path(request["cache_dir"]).resolve()
    require(cache_root == repo / "artifacts/task2-kion/c-rl-20261024-r1/policy/screen-cache-r1",
            "Wrong natural1024 source cache")
    cache_path = cache_root / "cache.json"
    require(sha(cache_path) == CACHE_SHA, "Cache pin changed")
    cache = read(cache_path)
    require(cache.get("contract") == "kion-current-C1070-natural1024-H1-v1"
            and cache.get("state") == "completed" and cache.get("rows") == 256
            and cache.get("candidate_width") == 1024 and cache.get("raw_width") == 1068
            and cache.get("no_target_injection") is True and cache.get("no_seen_filter") is True,
            "Exact completed natural1024 cache required")
    scorer_root = repo / "artifacts/task2-kion/local-pack-20261020/rm-r2/scorer"
    reward_root = repo / "artifacts/task2-kion/c-rl-20261024-r1/reward/screen256-r1"
    required = {scorer_root / "model.cbm": MODEL_SHA, scorer_root / "scorer.npz": PRIOR_SHA,
                scorer_root / "manifest.json": MANIFEST_SHA, cache_path: CACHE_SHA,
                bridge_path: BRIDGE_SHA, reward_root / "reward.json": REWARD_META_SHA}
    require(all(request["input_hashes"].get(path.as_posix()) == pin
                or request["source_hashes"].get(path.as_posix()) == pin for path, pin in required.items()),
            "Explicit exact RM/cache/model input pins required")
    serving_receipt = repo / "artifacts/task2-kion/local-pack-20261020/rm-eval-r1/manifest.json"
    require(request["input_hashes"].get(serving_receipt.as_posix()) == SERVING_RECEIPT_SHA
            and sha(serving_receipt) == SERVING_RECEIPT_SHA, "Exact serving receipt request pin required")
    required[serving_receipt] = SERVING_RECEIPT_SHA
    receipt = read(serving_receipt)
    for key in ("eval_source", "freeze", "metadata", "basic_npz", "basic_manifest", "raw_items",
                "raw_users", "raw_interactions", "rm_model_manifest", "rm_bundle_manifest"):
        pin = receipt["inputs"][key]
        require(request["input_hashes"].get(pin["path"]) == pin["sha256"]
                or request["source_hashes"].get(pin["path"]) == pin["sha256"],
                "Explicit serving closure request pin required: " + key)
    for name in STATE_FIELDS:
        entry = cache["arrays"][name]
        path = (cache_root / entry["path"]).resolve()
        require(path.is_relative_to(cache_root) and path.stat().st_size == entry["bytes"]
                and sha(path) == entry["sha256"], "State array differs: " + name)
        require(request["input_hashes"].get(path.as_posix()) == entry["sha256"],
                "Explicit state array request pin required: " + name)
    spec = importlib.util.spec_from_file_location("_submission_frozen_rm_bridge", bridge_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    numerical = module.bootstrap_numpy(repo, request["catboost_site"], request["numpy_pins"])
    backend = module.bootstrap_catboost(repo, request["catboost_site"], request["catboost_pins"])
    np = module.np
    check(stage="rm-export-provider-init")
    init_start = time.perf_counter()
    bridge, provenance = module.load_runtime(repo)
    init_elapsed = time.perf_counter() - init_start
    require(init_elapsed <= request["init_seconds"], "RM provider initialization budget exceeded")
    require(bridge.scorer.signature == SIGNATURE and bridge.scorer.catalog.schema_signature == SCHEMA
            and bridge.scorer.catalog.mapping_sha256 == MAPPING, "Frozen RM identity differs")
    require(bridge.scorer.calibration["slope"] == 1.0148506983671792
            and bridge.scorer.calibration["intercept"] == -0.005568380749024937
            and bridge.scorer.priors.global_mean == 0.47802770002556966,
            "Frozen calibration/fallback differs")
    catalog = bridge.scorer.catalog
    require(not catalog.compatibility_pairs and not catalog.context_categorical_names
            and not catalog.item_categorical_names and bridge.scorer.config.get("history") is True,
            "Unexpected RM tree schema")
    state_arrays = {}
    try:
        for name in STATE_FIELDS:
            entry = cache["arrays"][name]
            array = np.load(cache_root / entry["path"], mmap_mode="r", allow_pickle=False)
            state_arrays[name] = array
            require(list(array.shape) == entry["shape"] and str(array.dtype) == entry["dtype"]
                    and array.dtype.kind in "iu", "State array header differs")
        require(state_arrays["candidate_raw_ids"].shape == (256, 1024)
                and state_arrays["x"].shape == (256, 64), "State/pool width differs")
        ids = np.array(state_arrays["candidate_raw_ids"][:4], dtype=np.int64, copy=True)
        require(bool((ids >= 0).all()) and all(len(np.unique(row)) == 1024 for row in ids),
                "Original-ID pools must be nonnegative and deduplicated")
        context = np.empty((4, 2079), np.float32)
        candidates = np.empty((4, 1024, 2008), np.float32)
        reward64 = np.empty((4, 1024), np.float64)
        support = np.empty((4, 1024), bool)
        slice_start = time.perf_counter()
        parity = feature_parity = 0.
        for row in range(4):
            check(stage="rm-export-state", completed=row, total=4)
            require(time.perf_counter() - slice_start <= request["slice_seconds"], "RM slice budget exceeded")
            user = state_arrays["user_ids"][row]
            origin = state_arrays["origins"][row]
            result = bridge.score_state(user, origin, ids[row], state_arrays["x"][row],
                                        state_arrays["lengths"][row])
            reward64[row], support[row] = result["rewards"], result["support"]
            parity = max(parity, result["parity_max_abs"])
            feature_parity = max(feature_parity, result["candidate_feature_parity_max_abs"])
            users, origins = np.asarray([user], np.int64), np.asarray([origin], np.int64)
            observed_x, lengths, maxima = bridge.raw.history_batch(users, origins)
            require(np.array_equal(observed_x[0], state_arrays["x"][row])
                    and int(lengths[0]) == int(state_arrays["lengths"][row]) and maxima[0] < origin,
                    "Strict-past state binding differs")
            context[row] = bridge.pack.user_features(users, origins, observed_x, lengths,
                                                     history_cutoffs=origins)[0]
            tokens = np.asarray([bridge.mapping[int(i)] for i in ids[row]], np.int64)
            for begin in range(0, 1024, 128):
                size = min(128, 1024 - begin)
                pair_users = np.full(size, user, np.int64)
                pair_origins = np.full(size, origin, np.int64)
                candidate = bridge.pack.candidate_features(tokens[begin:begin + size], pair_origins,
                    history_cutoffs=pair_origins, user_ids=pair_users)
                candidates[row, begin:begin + size] = candidate
                rows = {"user_id": pair_users, "item_id": ids[row, begin:begin + size],
                        "numeric": np.repeat(context[row:row + 1], size, axis=0),
                        "candidate_numeric": candidate, "categorical": np.empty((size, 0), dtype="U1")}
                repeated = bridge.scorer.score_rows(rows)
                require(float(np.max(np.abs(repeated - reward64[row, begin:begin + size]))) <= 1e-12,
                        "Exported U/C differs from original bridge")
        require(time.perf_counter() - slice_start <= request["slice_seconds"], "RM slice budget exceeded")
        require(bool(np.isfinite(context).all()) and bool(np.isfinite(candidates).all()), "Nonfinite feature export")
        indices = catalog.indices(ids.reshape(-1)).reshape(4, 1024)
        counts = np.asarray(bridge.scorer.priors.counts[indices], dtype=np.int64)
        prior = np.asarray(bridge.scorer.priors.item_mean[indices], dtype=np.float64)
        require(np.array_equal(support, counts > 0), "Frozen support binding differs")
        reward = reward64.astype(np.float32)
        # Saved native rewards are comparison targets only, never prediction inputs.
        require(sha(reward_root / "reward.json") == REWARD_META_SHA, "Historical reward manifest changed")
        reward_meta = read(reward_root / "reward.json")
        require(reward_meta.get("ready") is True and reward_meta.get("cache", {}).get("sha256") == CACHE_SHA,
                "Historical reward/cache binding differs")
        historical = {}
        try:
            for key in ("rewards", "support"):
                descriptor = reward_meta["files"][key + ".npy"]
                path = reward_root / (key + ".npy")
                require(request["input_hashes"].get(path.as_posix()) == descriptor["sha256"]
                        and sha(path) == descriptor["sha256"], "Historical reward payload/request pin differs")
                historical[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            require(np.array_equal(reward, historical["rewards"][:4])
                    and np.array_equal(support, historical["support"][:4]), "Historical RM parity failed")
        finally:
            for value in historical.values():
                value._mmap.close()
        check(stage="rm-export-write")
        output.mkdir()
        arrays = {}
        for name, array in (("candidate_raw_ids", ids), ("rm_context_numeric", context),
                            ("rm_candidate_numeric", candidates), ("rm_fit_counts", counts),
                            ("rm_item_prior", prior), ("rm_reference_reward", reward),
                            ("rm_reference_reward64", reward64), ("rm_reference_support", support)):
            path = output / (name + ".npy")
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            arrays[name] = {"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size,
                            "dtype": str(array.dtype), "shape": list(array.shape)}
        model_path = output / "model.cbm"
        with (scorer_root / "model.cbm").open("rb") as source, model_path.open("xb") as target:
            shutil.copyfileobj(source, target)
        require(sha(model_path) == MODEL_SHA, "Copied model bytes differ")
        # Transformed fixture uses row numbers; direct source user IDs and raw
        # histories are excluded. This is not a claim of proven anonymization.
        rm_meta = {"scorer_signature": SIGNATURE, "schema_signature": SCHEMA, "mapping_sha256": MAPPING,
                   "prior_archive_sha256": PRIOR_SHA, "context_width": 2079, "candidate_width": 2008,
                   "numeric_width": 4087, "history": True, "compatibility_pairs": [],
                   "categorical_rule": "item:{original_item_id}", "row_positions": [0, 1, 2, 3],
                   "calibration": {"slope": bridge.scorer.calibration["slope"],
                                   "intercept": bridge.scorer.calibration["intercept"]},
                   "global_mean": bridge.scorer.priors.global_mean,
                   "exporter_source_sha256": sha(__file__), "bridge_source_sha256": BRIDGE_SHA,
                   "source_cache_sha256": CACHE_SHA, "source_model_manifest_sha256": MANIFEST_SHA,
                   "feature_clock": "actual query origin; raw history strictly before origin; transforms fitted<2021-08-02",
                   "metadata_availability": "static_snapshot_unverified",
                   "reward_state": "fixed observed s0; no synthetic prefix append",
                   "prior_semantics": "serialized sampled-fit item shrinkage mean; alpha20; no refit",
                   "residual_semantics": "float64(float32(calibrated_reward))-float64(item_prior), unsupported0"}
        result = {"contract": "kion-portable-rm-fixture-v1", "state": "completed", "arrays": arrays,
                  "weights": {"RM": {"path": "model.cbm", "sha256": MODEL_SHA, "bytes": model_path.stat().st_size}},
                  "rm": rm_meta, "parity": {"historical_float32_exact": True,
                  "native_pair_max_abs": parity, "candidate_singleton_batch_max_abs": feature_parity},
                  "environment": {"python": sys.version, "numpy": numerical["version"], "catboost": backend["version"]},
                  "timings_seconds": {"provider_init": init_elapsed, "total": time.perf_counter() - start},
                  "training_updates": 0, "rm_fits": 0, "public_source_user_ids_included": False}
        require(sum(entry["bytes"] for entry in arrays.values()) + model_path.stat().st_size < 40_000_000,
                "Fixture exceeded40MB admission bound")
        checked_pins(request["source_hashes"])
        checked_pins(request["input_hashes"])
        require(sha(cache_path) == CACHE_SHA and all(sha(path) == pin for path, pin in required.items()),
                "Source inputs changed during export")
        for name in STATE_FIELDS:
            require(sha(cache_root / cache["arrays"][name]["path"]) == cache["arrays"][name]["sha256"],
                    "State inputs changed during export")
        require(time.perf_counter() - start <= request["maximum_seconds"], "RM total export budget exceeded")
        write(output / "manifest.json", result)
        check(stage="rm-export-completed", states=4)
        # Native stage receipt retains source-machine provenance; portable fixture
        # JSON above contains source-user-ID-free transformed input descriptors
        # and public hashes; transformed features are not proven anonymous.
        return {**result, "source_provenance": provenance}
    finally:
        for value in state_arrays.values():
            value._mmap.close()
