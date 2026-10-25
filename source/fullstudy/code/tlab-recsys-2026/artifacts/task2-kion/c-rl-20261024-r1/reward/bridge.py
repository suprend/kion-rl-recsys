"""Bounded arbitrary-raw-candidate adapter for the frozen local RM-r2.

Import is inert. Only main()/load_runtime() open actual model/data payloads.
No fits, policy models, providers, GPU, or training are launched here.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Numerical imports belong to the explicitly pinned, isolated worker bootstrap.
# Synthetic tests inject their small-array NumPy module without opening models.
np = None

CONTRACT = "current-C-local-RM-r2-bridge-v1"
SIGNATURE = "ee14bc4d7375a4d6978a2e1f476cd795369b45ef4250501b2f5992767e9ed261"
SCHEMA = "7776a9d611b04ef8f3ef77fce0a67c1fb045f67142bf4b4bbde24cc469f9bda4"
MAPPING = "b90df0d1fcce007008b62007e486bc1e9cb0b4ac89c97bb5107e84305e744db4"
ORIGIN_MIN = date(2021, 8, 2).toordinal()
ORIGIN_MAX = date(2021, 8, 16).toordinal()
STATE_FIELDS = ("user_ids", "origins", "x", "lengths", "candidate_raw_ids")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def small_json(path, limit=2 * 1024 * 1024):
    if Path(path).stat().st_size > limit:
        raise ValueError("JSON exceeds explicit bounded metadata limit: " + str(path))
    return json.loads(Path(path).read_text(encoding="utf-8"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def loaded_numerical_modules():
    return [name for name, value in sys.modules.items() if value is not None and
            (name == "numpy" or name.startswith("numpy.") or name == "catboost" or name.startswith("catboost."))]


def bootstrap_numpy(repo, site, pins):
    """Select the producer's pinned NumPy before any numerical import."""
    global np
    expected_site = (Path(repo).resolve() / ".venv/Lib/site-packages").resolve()
    site = Path(site).resolve()
    if site != expected_site:
        raise ValueError("NumPy site must be the exact existing repository venv site")
    native = "_core/_multiarray_umath.cp312-win_amd64.pyd"
    if (not isinstance(pins, dict) or set(pins) != {"__init__.py", native} or
            any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values())):
        raise ValueError("Explicit NumPy init/native SHA256 pins required")
    package = site / "numpy"
    checked = {}
    for name, expected in pins.items():
        path = (package / name).resolve()
        if not path.is_relative_to(package) or not path.is_file() or digest(path) != expected:
            raise ValueError("Existing NumPy backend pin mismatch: " + name)
        checked[name] = {"path": path.as_posix(), "sha256": expected}
    if loaded_numerical_modules():
        raise ValueError("Isolated reward bootstrap requires a fresh process before NumPy/CatBoost imports")
    # Local to this dedicated reward worker. Do not reload/remove sys.modules.
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("numpy")
    if spec is None or Path(spec.origin or "").resolve() != package / "__init__.py":
        raise ValueError("NumPy import resolution does not select pinned existing backend")
    candidate = importlib.import_module("numpy")
    if getattr(candidate, "__version__", None) != "2.5.4":
        raise ValueError("Pinned NumPy version must be 2.5.4")
    for name, expected in (("numpy", "__init__.py"), ("numpy._core._multiarray_umath", native)):
        loaded = importlib.import_module(name)
        if Path(getattr(loaded, "__file__", "")).resolve() != package / expected:
            raise ValueError("Imported NumPy module escaped pinned backend: " + name)
    if any(digest(package / name) != pin for name, pin in pins.items()):
        raise ValueError("NumPy backend files changed during import")
    np = candidate
    return {"site": site.as_posix(), "version": candidate.__version__, "files": checked,
        "numpy_file": Path(candidate.__file__).resolve().as_posix()}


def bootstrap_catboost(repo, site, pins):
    """Use the pinned existing backend after isolated producer NumPy selection."""
    expected_site = (Path(repo).resolve() / ".venv/Lib/site-packages").resolve()
    site = Path(site).resolve()
    if site != expected_site:
        raise ValueError("CatBoost site must be the exact existing repository venv site")
    filenames = {"__init__.py", "core.py", "_catboost.pyd"}
    if (not isinstance(pins, dict) or set(pins) != filenames or
            any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in pins.values())):
        raise ValueError("Explicit SHA256 pins for all three CatBoost backend files required")
    package = site / "catboost"
    checked = {}
    # Complete ALL byte/path checks before importlib can execute backend code.
    for name in sorted(filenames):
        path = (package / name).resolve()
        if path.parent != package or not path.is_file() or digest(path) != pins[name]:
            raise ValueError("Existing CatBoost backend pin mismatch: " + name)
        checked[name] = {"path": path.as_posix(), "sha256": pins[name]}
    for name, expected in (("catboost", "__init__.py"), ("catboost.core", "core.py"), ("catboost._catboost", "_catboost.pyd")):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != package / expected:
            raise ValueError("CatBoost module was already loaded from another runtime: " + name)
    if str(site) not in sys.path:
        sys.path.append(str(site))
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("catboost")
    if spec is None or Path(spec.origin or "").resolve() != package / "__init__.py":
        raise ValueError("CatBoost import resolution does not select pinned existing backend")
    backend = importlib.import_module("catboost")
    if getattr(backend, "__version__", None) != "1.2.10":
        raise ValueError("Pinned CatBoost version must be 1.2.10")
    for name, expected in (("catboost", "__init__.py"), ("catboost.core", "core.py"), ("catboost._catboost", "_catboost.pyd")):
        loaded = importlib.import_module(name)
        if Path(getattr(loaded, "__file__", "")).resolve() != package / expected:
            raise ValueError("Imported CatBoost module escaped pinned backend: " + name)
    if any(digest(package / name) != pin for name, pin in pins.items()):
        raise ValueError("CatBoost backend files changed during import")
    return {"site": site.as_posix(), "version": backend.__version__, "files": checked,
        "numpy_version": np.__version__, "numpy_file": Path(np.__file__).resolve().as_posix(),
        "python_version": sys.version, "python_executable": Path(sys.executable).resolve().as_posix(),
        "abi_status": "existing backend import passed; real scoring parity remains a probe gate"}


def load_runtime(repo):
    """Root-only launch: exact earlier serving closure, no RM role matrices.

    Existing native scorer.load reads its known 26MB manifest once; guard and
    exact receipt hash precede this explicit load. No blind metadata inspection.
    """
    if np is None:
        raise RuntimeError("Explicit isolated numerical bootstrap is required before RM loading")
    repo = Path(repo).resolve()
    base = repo / "artifacts/task2-kion/local-pack-20261020"
    receipt = small_json(base / "rm-eval-r1/manifest.json")
    if receipt.get("ready") is not True:
        raise ValueError("Existing observed-outcome serving receipt is not ready")
    inputs = receipt["inputs"]
    def checked(key):
        pin = inputs[key]
        path = Path(pin["path"]).resolve()
        if not path.is_file() or digest(path) != pin["sha256"]:
            raise ValueError("Existing serving closure changed: " + key)
        return path
    # Only closure sources and actual input files are read; no observed label
    # role arrays, prepared train/val arrays, or prediction arrays are loaded.
    helper_path = checked("eval_source")
    helper = module(helper_path, "_C_RM_eval_helper")
    freeze_path = checked("freeze")
    source_bundle = base / "bundle-rm-r2"
    rm, data, pack_module, freeze = helper.frozen_runtime(source_bundle, freeze_path)
    metadata_path, basic_npz = checked("metadata"), checked("basic_npz")
    checked("basic_manifest")
    raw_paths = {key: checked("raw_" + key) for key in ("items", "users", "interactions")}
    artifact = base / "rm-r2/scorer"
    manifest_path = artifact / "manifest.json"
    if manifest_path.stat().st_size > 32 * 1024 * 1024 or checked("rm_model_manifest") != manifest_path.resolve():
        raise ValueError("Unexpected native scorer manifest path/size")
    checked("rm_bundle_manifest")
    mapping = data.checked_mapping(small_json(metadata_path))
    if rm.canonical_hash({str(k): v for k, v in mapping.items()}) != MAPPING:
        raise ValueError("Policy mapping changed")
    scorer = rm.RewardModelScorer.load(artifact, MAPPING)
    if (scorer.signature != SIGNATURE or scorer.catalog.schema_signature != SCHEMA or
            scorer.model is None or scorer.config.get("history") is not True):
        raise ValueError("Wrong local RM-r2 scorer/config/schema")
    # Parent feature sources are frozen and checked by frozen_runtime(). The
    # receipt binds exactly this model manifest, model/scorer bytes and its
    # observed bundle manifest. Native load verifies manifest signature/files.
    for key, expected in scorer.catalog.provenance["source_pins"].items():
        actual = {"metadata": metadata_path, "basic_npz": basic_npz, **raw_paths}.get(key)
        if actual is not None and digest(actual) != expected:
            raise ValueError("Scorer input source differs: " + key)
    pack = pack_module.FeaturePack.build(raw_paths["items"].parent, metadata_path, basic_npz,
                                         fit_cutoff="2021-08-02", include_relative=True)
    if pack.fit_cutoff != "2021-08-02":
        raise ValueError("Fitted transform cutoff changed")
    facade = helper.verify_facade_provenance(pack.provenance, scorer.catalog.provenance["facade"], rm)
    data._check_sources(data._declared_sources(pack.provenance))
    raw = data.RawRows.from_pack(pack, mapping)
    pins = {key: inputs[key] for key in ("eval_source", "freeze", "metadata", "basic_npz", "basic_manifest",
            "raw_items", "raw_users", "raw_interactions", "rm_model_manifest", "rm_bundle_manifest")}
    pins["bridge_source"] = {"path": Path(__file__).resolve().as_posix(), "sha256": digest(__file__)}
    return LocalRMBridge(pack, raw, scorer, mapping), {"closure": pins, "facade_timestamp": facade,
        "signature": scorer.signature, "schema": SCHEMA, "mapping": MAPPING,
        "feature_source_bundle": freeze["parent_bundle"], "fit_cutoff_exclusive": "2021-08-02"}


class LocalRMBridge:
    def __init__(self, pack, raw, scorer, mapping, chunk_size=128):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 256:
            raise ValueError("candidate chunk must be integer [1,256]")
        if (list(pack.user_columns) != list(scorer.catalog.context_names) or
                list(pack.candidate_columns) != list(scorer.catalog.item_names) or
                len(pack.user_columns) != 2079 or len(pack.candidate_columns) != 2008):
            raise ValueError("Exact ordered RM2079+2008 schema required")
        self.pack, self.raw, self.scorer, self.mapping = pack, raw, scorer, mapping
        self.chunk_size = chunk_size

    def score_state(self, user_id, origin, candidate_raw_ids, x, length):
        if (isinstance(user_id, (bool, np.bool_)) or not isinstance(user_id, (int, np.integer)) or user_id < 0 or
                isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)) or
                not ORIGIN_MIN <= origin < ORIGIN_MAX):
            raise ValueError("Integer user and actual Aug2--15 query origin required")
        ids = np.asarray(candidate_raw_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or len(ids) > 1024 or np.any(ids < 0):
            raise ValueError("Bounded raw original candidate IDs required")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Natural pool must be deduplicated")
        # Check original IDs before converting: a policy token is not a raw ID.
        self.scorer.catalog.indices(ids)
        tokens = np.asarray([self.mapping[int(i)] for i in ids], np.int64)
        users, origins = np.asarray([user_id], np.int64), np.asarray([origin], np.int64)
        observed_x, lengths, maxima = self.raw.history_batch(users, origins)
        if (np.asarray(x).shape != (64,) or np.asarray(x).dtype.kind not in "iu" or
                isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)) or
                int(lengths[0]) != length or not np.array_equal(observed_x[0], x) or maxima[0] >= origin):
            raise ValueError("Current-C state differs from exact strict-past raw H63")
        numeric = self.pack.user_features(users, origins, observed_x, lengths, history_cutoffs=origins)
        rewards, support, fallback = [], [], []
        max_error = 0.0
        feature_error = 0.0
        for begin in range(0, len(ids), self.chunk_size):
            chunk = ids[begin:begin + self.chunk_size]
            n = len(chunk)
            pair_users = np.full(n, user_id, np.int64)
            pair_origins = np.full(n, origin, np.int64)
            candidate = self.pack.candidate_features(tokens[begin:begin + n], pair_origins,
                history_cutoffs=pair_origins, user_ids=pair_users)
            if begin == 0:
                # One original pair through singleton native facade versus
                # vectorized arbitrary-candidate assembly, on every state.
                single = self.pack.candidate_features(tokens[:1], origins,
                    history_cutoffs=origins, user_ids=users)
                feature_error = float(np.max(np.abs(single[0] - candidate[0])))
                if not np.isfinite(feature_error) or feature_error > 1e-7:
                    raise ValueError("Singleton/batched candidate feature parity failed")
            rows = {"user_id": pair_users, "item_id": chunk,
                "numeric": np.repeat(numeric, n, axis=0), "candidate_numeric": candidate,
                "categorical": np.empty((n, 0), dtype="U1")}
            # Native score_rows is the independent public-path comparison to
            # score_with_diagnostics, with candidate/rawID bindings preserved.
            native = np.asarray(self.scorer.score_rows(rows), np.float64)
            contexts = [{"user_id": int(user_id), "numeric": numeric[0], "categorical": [],
                "candidate_numeric": candidate[j], "candidate_item_id": int(item),
                "schema_signature": self.scorer.catalog.schema_signature} for j, item in enumerate(chunk)]
            result = self.scorer.score_with_diagnostics(contexts, chunk)
            scores = np.asarray(result["rewards"], np.float64)
            valid = np.asarray(result["support"])
            cold = np.asarray(result["fallback"])
            if (scores.shape != (n,) or valid.shape != (n,) or cold.shape != (n,) or
                    valid.dtype.kind != "b" or cold.dtype.kind != "b" or not np.array_equal(cold, ~valid) or
                    not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
                raise ValueError("Invalid calibrated reward/support/fallback result")
            error = float(np.max(np.abs(native - scores)))
            if not np.isfinite(error) or error > 1e-12:
                raise ValueError("Native pair adapter parity failed")
            expected = self.scorer.priors.counts[self.scorer.catalog.indices(chunk)] > 0
            if not np.array_equal(valid, expected):
                raise ValueError("Candidate fit support differs from exact frozen priors")
            if cold.any() and not np.all(scores[cold] == self.scorer.priors.global_mean):
                raise ValueError("Cold fallback changed")
            max_error = max(max_error, error)
            rewards.append(scores)
            support.append(valid)
            fallback.append(cold)
        return {"rewards": np.concatenate(rewards), "support": np.concatenate(support),
            "fallback": np.concatenate(fallback), "parity_max_abs": max_error,
            "candidate_feature_parity_max_abs": feature_error,
            "history_max_date": int(maxima[0]), "history_length": int(lengths[0]) - 1}


def score_cache(bridge, cache_dir, output, *, max_states=32, seconds=600, provenance=None, check=None):
    resources = []
    try:
        return _score_cache(bridge, cache_dir, output, max_states=max_states, seconds=seconds,
                            provenance=provenance, resources=resources, check=check)
    finally:
        for value in reversed(resources):
            value._mmap.close()


def _score_cache(bridge, cache_dir, output, *, max_states, seconds, provenance, resources, check):
    """Stream one pinned current-C natural1024 cache; keep only reward arrays."""
    if max_states not in (32, 256, 512) or not 0 < seconds <= 3600:
        raise ValueError("Use predeclared 32/256/512 cap and <=3600s score budget")
    cache_dir, output = Path(cache_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Reward output must be new")
    cache_manifest = cache_dir / "cache.json"
    cache_sha = digest(cache_manifest)
    cache_meta = small_json(cache_manifest)
    if (cache_meta.get("contract") != "kion-current-C1070-natural1024-H1-v1" or
            cache_meta.get("state") != "completed" or cache_meta.get("candidate_width") != 1024 or
            cache_meta.get("raw_width") != 1068 or cache_meta.get("no_target_injection") is not True or
            cache_meta.get("no_seen_filter") is not True):
        raise ValueError("Completed current-C natural1024 cache required")
    # Exact file hashes pin state identity even when cache metadata layout evolves.
    paths = {name: cache_dir / (name + ".npy") for name in STATE_FIELDS}
    pins = {name: {"path": path.as_posix(), "sha256": digest(path)} for name, path in paths.items()}
    arrays = {}
    for name, path in paths.items():
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        resources.append(arrays[name])
        descriptor = cache_meta.get("arrays", {}).get(name, {})
        if (descriptor.get("sha256") != pins[name]["sha256"] or
                (cache_dir / descriptor.get("path", "")).resolve() != path or
                descriptor.get("shape") != list(arrays[name].shape) or
                descriptor.get("dtype") != str(arrays[name].dtype)):
            raise ValueError("Cache array differs from completed policy descriptor: " + name)
    source_path = cache_dir / "candidate_sources.npy"
    if source_path.exists():
        source_sha = digest(source_path)
        descriptor = cache_meta.get("arrays", {}).get("candidate_sources", {})
        if descriptor.get("sha256") != source_sha:
            raise ValueError("Changed candidate source array")
        arrays["candidate_sources"] = np.load(source_path, mmap_mode="r", allow_pickle=False)
        resources.append(arrays["candidate_sources"])
    n = len(arrays["user_ids"])
    if n < 1 or n > 512:
        raise ValueError("Cache must contain [1,512] fixed states")
    if (arrays["candidate_raw_ids"].shape != (n, 1024) or arrays["x"].shape != (n, 64) or
            any(arrays[name].shape != (n,) for name in ("user_ids", "origins", "lengths")) or
            any(arrays[name].dtype.kind not in "iu" for name in STATE_FIELDS)):
        raise ValueError("Cache state arrays do not match natural1024 contract")
    if "candidate_sources" in arrays and arrays["candidate_sources"].shape != (n, 1024, 14):
        raise ValueError("Candidate source schema must be exactly14")
    take = min(n, max_states)
    output.mkdir(parents=True)
    out = {}
    for name, dtype in (("rewards", np.float32), ("support", bool), ("fallback", bool)):
        out[name] = np.lib.format.open_memmap(output / (name + ".npy"), mode="w+", dtype=dtype, shape=(take, 1024))
        resources.append(out[name])
    started, parity, history_max, feature_parity = time.perf_counter(), 0., 0, 0.
    for row in range(take):
        if check is not None:
            check(stage="reward-state", completed=row, total=take)
        if time.perf_counter() - started > seconds:
            raise TimeoutError("RM scoring budget exceeded; partial arrays are not ready")
        result = bridge.score_state(arrays["user_ids"][row], arrays["origins"][row],
            arrays["candidate_raw_ids"][row], arrays["x"][row], arrays["lengths"][row])
        for name in out:
            out[name][row] = result[name]
        parity = max(parity, result["parity_max_abs"])
        feature_parity = max(feature_parity, result["candidate_feature_parity_max_abs"])
        history_max = max(history_max, result["history_max_date"])
        print(json.dumps({"state": row + 1, "of": take, "seconds": time.perf_counter() - started}), flush=True)
    for value in out.values():
        value.flush()
    # Catch source state mutation during scoring, never approve partial/stale data.
    if digest(cache_manifest) != cache_sha or any(digest(paths[k]) != p["sha256"] for k, p in pins.items()):
        raise ValueError("Policy cache changed during reward scoring")
    support = out["support"]
    counts = bridge.scorer.priors.counts[bridge.scorer.catalog.indices(arrays["candidate_raw_ids"][:take].reshape(-1))].reshape(take, 1024)
    strata = {}
    def stratum(name, mask):
        count = int(mask.sum())
        strata[name] = {"pairs": count, "support_fraction": float(support[mask].mean()) if count else None,
            "fallback_fraction": float(out["fallback"][mask].mean()) if count else None,
            "mean_proxy": float(out["rewards"][mask].mean()) if count else None}
    for name, mask in (("fit_count0", counts == 0), ("fit_count1_4", (counts >= 1) & (counts <= 4)), ("fit_count5plus", counts >= 5)):
        stratum(name, mask)
    if "candidate_sources" in arrays:
        sources = arrays["candidate_sources"][:take]
        if not np.isfinite(sources).all() or not np.isin(sources[..., 3:6], (0., 1.)).all():
            raise ValueError("Invalid source presence flags")
        for col, name in enumerate(("GPT", "pop7d", "KNN"), 3):
            stratum("source_" + name, sources[..., col] == 1)
        stratum("pop7d_logcount_zero", sources[..., 13] == 0)
        stratum("pop7d_logcount_positive", sources[..., 13] > 0)
        if digest(source_path) != source_sha:
            raise ValueError("Candidate sources changed during scoring")
        pins["candidate_sources"] = {"path": source_path.as_posix(), "sha256": source_sha}
    report = {"contract": CONTRACT, "ready": True, "states": take, "candidates_per_state": 1024,
        "rows": take * 1024, "state_selection": "first rows of fixed input cache; no outcome selection",
        "cache": {"path": cache_manifest.as_posix(), "sha256": cache_sha}, "state_arrays": pins,
        "input_cache_contract": cache_meta.get("contract"), "provenance": provenance or {},
        "query_clock": "actual per-state origin; raw observed history strictly before origin; no date clamp",
        "reward_clock": "fitted priors/transforms <2021-08-02; frozen calibration",
        "native_pair_parity_max_abs": parity, "history_max_date": history_max,
        "candidate_feature_singleton_batch_parity_max_abs": feature_parity,
        "parity_scope": "score_rows vs scoring API on same assembled blocks; singleton vs batched native candidate facade, one pair per state",
        "candidate_strata": strata,
        "support_count": int(support.sum()), "support_fraction": float(support.mean()),
        "cold_count": int((~support).sum()), "fallback_fraction": float((~support).mean()),
        "states_without_supported_actions": int(np.count_nonzero(~np.any(support, axis=1))),
        "fit_support_scope": "item ever observed as finite known target in RM fit; not causal overlap",
        "scoring_seconds": time.perf_counter() - started, "chunk_size": bridge.chunk_size,
        "files": {name + ".npy": {"sha256": digest(output / (name + ".npy")),
            "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in out.items()},
        "observed_outcome_quality": "existing frozen local rm-eval-r1 20k later-development proof; no labels for arbitrary candidates"}
    (output / "reward.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def run(request, check=None):
    """Native control-worker entry point; root owns this real launch."""
    if check is not None:
        check(stage="reward-start")
    chunk_size = request.get("chunk_size", 128)
    maximum = request["max_states"]
    seconds = request["scoring_seconds"]
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 1 <= chunk_size <= 256:
        raise ValueError("candidate chunk must be integer [1,256]")
    if (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum not in (32, 256, 512) or
            isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 3600):
        raise ValueError("Use predeclared 32/256/512 cap and <=3600s score budget")
    output = Path(request["reward_output"])
    if output.exists():
        raise FileExistsError("Reward output must be new")
    repo = Path(request.get("repo", "D:/tlab-recsys-2026"))
    if not {"catboost_site", "catboost_pins", "numpy_pins"}.issubset(request):
        raise ValueError("Explicit isolated NumPy/CatBoost site and file pins are required")
    numerical = bootstrap_numpy(repo, request["catboost_site"], request["numpy_pins"])
    backend = bootstrap_catboost(repo, request["catboost_site"], request["catboost_pins"])
    backend["numpy_backend"] = numerical
    bridge, provenance = load_runtime(repo)
    provenance["execution_environment"] = backend
    bridge.chunk_size = chunk_size
    if check is not None:
        check(stage="reward-runtime-ready")
    report = score_cache(bridge, Path(request["cache_dir"]), output, max_states=maximum,
                         seconds=seconds, provenance=provenance, check=check)
    if check is not None:
        check(stage="reward-completed", states=report["states"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("D:/tlab-recsys-2026"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, choices=(32, 256, 512), default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--backend-config", type=Path, required=True,
                        help="Small JSON with explicit catboost_site, catboost_pins and numpy_pins")
    args = parser.parse_args()
    request = small_json(args.backend_config)
    request.update(repo=args.repo, cache_dir=args.cache, reward_output=args.output,
                   max_states=args.max_states, scoring_seconds=args.seconds, chunk_size=args.chunk_size)
    report = run(request)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
