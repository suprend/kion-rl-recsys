"""CPU-only, bounded Aug 9--15 development evaluation of an already frozen RM.

No label in this window fits a model, prior, calibrator, threshold, or selection
rule. Exact prepared SFT validation targets are reused; this is development
evidence, not an untouched test or counterfactual reward evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# Set these before NumPy initializes a BLAS runtime, and before frozen imports.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "2"

import numpy as np


CONTRACT = "kion-reward-development-eval-v1"
ORIGIN = date(2021, 8, 9).toordinal()
STOP = date(2021, 8, 16).toordinal()
MAX_ROWS = 20000
BASELINES = ("global_mean", "item_shrinkage", "user_mean", "combined")
FEATURE_MODULES = ("kion_local_features", "kion_pack_static", "kion_pack_people", "kion_pack_history",
                   "kion_pack_dynamic", "kion_pack_semantics", "kion_basic_bundle_features", "kion_sex_features")


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def verify_freeze(source_bundle, freeze_path):
    """Validate every frozen source before any frozen implementation is loaded."""
    import hashlib
    root = Path(source_bundle).resolve()
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze.get("state") != "frozen" or Path(freeze.get("bundle", "")).resolve() != root:
        raise ValueError("Frozen source bundle identity mismatch")
    sources = freeze.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing frozen source pins")
    for name, expected in sources.items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Frozen source path is outside bundle or absent")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("Frozen source checksum mismatch: " + name)
    for name in ("kion_reward_boosting", "kion_reward_data", "kion_local_pack"):
        if "scripts/" + name + ".py" not in sources:
            raise ValueError("Frozen source pin missing: " + name)
    return freeze


def frozen_runtime(source_bundle, freeze_path):
    freeze = verify_freeze(source_bundle, freeze_path)
    serving_scripts = (Path(source_bundle) / "scripts").resolve()
    rm = _module(serving_scripts / "kion_reward_boosting.py", "_eval_frozen_rm")
    feature_bundle = Path(source_bundle)
    feature_freeze = freeze
    if freeze.get("changed_sources") == ["scripts/kion_reward_boosting.py"]:
        parent, _ = verified_parent_freeze(freeze, rm)
        feature_bundle = Path(parent["bundle"])
        feature_freeze = parent
    scripts = (feature_bundle / "scripts").resolve()
    # The facade's normal imports must not reuse mutable checkout implementations.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("kion_") and getattr(module, "__file__", None):
            if Path(module.__file__).resolve().parent != scripts:
                raise ValueError("Conflicting nonfrozen KION module already imported: " + name)
    sys.path.insert(0, str(scripts))
    data = _module(scripts / "kion_reward_data.py", "_eval_frozen_data")
    pack = _module(scripts / "kion_local_pack.py", "_eval_frozen_pack")
    # Resolve the facade's lazy imports now, while its producer is first on the
    # path, and validate the actual sys.modules objects used by build().
    verify_feature_modules(scripts, feature_freeze, rm)
    return rm, data, pack, freeze


def verify_feature_modules(scripts, freeze, rm):
    scripts = Path(scripts).resolve()
    for name in FEATURE_MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path != scripts / (name + ".py") or rm.file_hash(path) != freeze["sources"].get("scripts/" + name + ".py"):
            raise ValueError("Feature module escaped exact producer source: " + name)


def provenance_differences(actual, expected, prefix="", limit=8):
    if actual == expected:
        return []
    if isinstance(actual, dict) and isinstance(expected, dict):
        paths = []
        for key in sorted(set(actual) | set(expected)):
            path = prefix + "/" + str(key)
            if key not in actual or key not in expected:
                paths.append(path)
            else:
                paths.extend(provenance_differences(actual[key], expected[key], path, limit - len(paths)))
            if len(paths) >= limit:
                break
        return paths[:limit]
    return [prefix]


def verify_facade_provenance(actual, expected, rm):
    """Exclude one generation timestamp, never source/schema/feature evidence."""
    actual, expected = deepcopy(actual), deepcopy(expected)
    timestamps = {}
    for name, value in (("fresh", actual), ("producer", expected)):
        semantic = value.get("components", {}).get("semantics", {})
        stamp = semantic.get("retrieved_at")
        if not isinstance(stamp, str) or datetime.fromisoformat(stamp).tzinfo is None:
            raise ValueError("Semantic generation timestamp must be a timezone-aware ISO timestamp")
        timestamps[name] = semantic.pop("retrieved_at")
    if rm.canonical_hash(actual) != rm.canonical_hash(expected):
        raise ValueError("Fresh facade provenance differs from frozen RM facade at: " +
                         ", ".join(provenance_differences(actual, expected)))
    return {"excluded_path": "components.semantics.retrieved_at", "timestamps": timestamps,
            "reason": "runtime generation timestamp; source bytes, rules, evidence, schema and every other provenance field remain exact"}


def verify_sft_validation(cache_manifest_path, val_path, metadata_path, rm):
    """Bind the evaluated file to the already completed SFT cache receipt."""
    path = Path(cache_manifest_path)
    initial = rm.file_hash(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("contract") != "kion-all-local-pack-v1" or manifest.get("state") != "completed" or
            not isinstance(manifest.get("validation_rows"), int) or manifest["validation_rows"] < 1):
        raise ValueError("A completed SFT cache receipt with validation row count is required")
    for name, actual in (("val.npz", val_path), ("metadata.json", metadata_path)):
        descriptor = manifest.get("inputs", {}).get(name, {})
        if (not isinstance(descriptor.get("path"), str) or
                Path(descriptor["path"]).resolve() != Path(actual).resolve() or
                descriptor.get("sha256") != rm.file_hash(actual)):
            raise ValueError("Evaluation validation/metadata differs from prior SFT cache receipt: " + name)
    if rm.file_hash(path) != initial:
        raise ValueError("SFT cache receipt changed during validation")
    return manifest, initial


def verify_model_training_bundle(model_manifest, bundle_manifest_sha):
    if model_manifest.get("report", {}).get("bundle_manifest_sha256") != bundle_manifest_sha:
        raise ValueError("Frozen model training-bundle manifest checksum mismatch")


def verified_parent_freeze(freeze, rm):
    key = "scripts/kion_reward_boosting.py"
    if (freeze.get("changed_sources") != [key] or not isinstance(freeze.get("parent_freeze"), str) or
            not isinstance(freeze.get("parent_bundle"), str)):
        raise ValueError("Booster producer/serving difference requires an explicit booster-only parent freeze chain")
    parent_path = Path(freeze["parent_freeze"])
    if rm.file_hash(parent_path) != freeze.get("parent_freeze_sha256"):
        raise ValueError("Booster parent freeze checksum mismatch")
    parent = verify_freeze(freeze["parent_bundle"], parent_path)
    parent_sources = parent["sources"]
    if (set(parent_sources) != set(freeze["sources"]) or
            any(parent_sources[name] != value for name, value in freeze["sources"].items() if name != key)):
        raise ValueError("Booster parent chain changes feature/builder sources")
    return parent, parent_path


def verify_boosting_boundary(source_bundle, freeze, producer_hash, model_manifest, rm):
    """Allow only an explicitly chained booster-only technical serving repair."""
    key = "scripts/kion_reward_boosting.py"
    serving_hash = freeze["sources"][key]
    if (model_manifest.get("source_sha256") != serving_hash or
            rm.file_hash(Path(source_bundle) / key) != serving_hash):
        raise ValueError("Frozen model serving source differs from frozen booster")
    facts = {"dataset_producer_sha256": producer_hash, "serving_source_sha256": serving_hash,
             "relation": "identical", "feature_source_bundle": Path(source_bundle).resolve().as_posix()}
    if producer_hash == serving_hash:
        return facts, {}
    parent, parent_path = verified_parent_freeze(freeze, rm)
    parent_sources = parent["sources"]
    if parent_sources.get(key) != producer_hash:
        raise ValueError("Booster parent chain differs from dataset producer")
    facts.update(relation="explicit_booster_only_technical_repair", parent_bundle=freeze["parent_bundle"],
                 parent_freeze=parent_path.resolve().as_posix(), parent_freeze_sha256=freeze["parent_freeze_sha256"],
                 feature_source_bundle=Path(parent["bundle"]).resolve().as_posix())
    paths = {"producer_freeze": parent_path}
    paths.update({"producer_source/" + name: Path(freeze["parent_bundle"]) / name for name in parent_sources})
    return facts, paths


def select_targets(raw, prepared, mapping, rm, max_rows=MAX_ROWS):
    """Bind exact prepared rows to raw observed pairs before label-blind sampling."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError("max_rows must be an integer in [1,20000]")
    required = {"user_id", "target_item_id", "target_date", "y", "reward", "x", "lengths"}
    if not required.issubset(prepared):
        raise ValueError("Prepared validation fields are missing")
    users = rm.checked_ids(prepared["user_id"], "validation users")
    items = rm.checked_ids(prepared["target_item_id"], "validation items")
    days = np.asarray(prepared["target_date"])
    n = len(users)
    if not n or any(np.asarray(prepared[k]).shape != (n,) for k in required - {"x"}):
        raise ValueError("Prepared validation vectors must align and be nonempty")
    if days.dtype.kind not in "iu" or np.any((days < ORIGIN) | (days >= STOP)):
        raise ValueError("Validation target dates must be Aug 9--15")
    if (np.asarray(prepared["x"]).shape != (n, 64) or
            any(np.asarray(prepared[k]).dtype.kind not in "iu" for k in ("x", "lengths", "y"))):
        raise ValueError("Prepared validation must use BOS/H63")
    keys = np.rec.fromarrays((users, items, days))
    if len(np.unique(keys)) != n:
        raise ValueError("Duplicate prepared validation row")
    wanted = users * raw_data_base(raw) + days
    left, right = np.searchsorted(raw.keys, wanted, "left"), np.searchsorted(raw.keys, wanted, "right")
    indices = np.empty(n, np.int64)
    for row, (lo, hi, item) in enumerate(zip(left, right, items)):
        at = int(lo) + int(np.searchsorted(raw.items[lo:hi], item))
        if at >= hi or raw.items[at] != item:
            raise ValueError("Prepared target has no exact raw user/item/date binding")
        indices[row] = at
    expected_tokens = np.fromiter((mapping.get(int(i), 2) for i in items), np.int64, n)
    if not np.array_equal(prepared["y"], expected_tokens):
        raise ValueError("Prepared target token differs from frozen policy mapping")
    first = np.searchsorted(raw.keys, users * raw_data_base(raw) + ORIGIN, "left")
    if np.any(raw.users[first] != users) or np.any(raw.days[first] != days):
        raise ValueError("Validation must use the first active date per user")
    cutoffs = np.full(n, ORIGIN, np.int64)
    x, lengths, _ = raw.history_batch(users, cutoffs)
    if not np.array_equal(x, prepared["x"]) or not np.array_equal(lengths, prepared["lengths"]):
        raise ValueError("Prepared validation differs from exact frozen Aug9 raw H63")
    valid, known = raw.statuses[indices] == 0, expected_tokens >= 3
    rewards = np.asarray(prepared["reward"])
    if not np.isfinite(rewards[valid]).all() or not np.allclose(rewards[valid], raw.outcomes[indices[valid]], rtol=0, atol=2e-7):
        raise ValueError("Prepared reward differs from its raw observed outcome")
    eligible = np.flatnonzero(valid & known)
    if not len(eligible):
        raise ValueError("No valid observed known validation targets")
    keep = rm.deterministic_sample_indices(users[eligible], items[eligible], days[eligible].astype(np.int64), max_rows)
    chosen = eligible[keep]
    facts = {"prepared_rows": n, "valid_known_before_cap": len(eligible), "evaluated_rows": len(chosen),
             "excluded_missing_labels": int(np.count_nonzero(raw.statuses[indices] == 1)),
             "excluded_invalid_labels": int(np.count_nonzero(raw.statuses[indices] == 2)),
             "excluded_unknown_targets": int(np.count_nonzero(~known)),
             "excluded_by_cap": len(eligible) - len(chosen), "cap": max_rows,
             "sampling": "frozen salted SHA256 original user/item/date rank, without outcomes",
             "population": "exact prepared SFT development targets; first active date per user; tied targets retained before cap",
             "missing_imputed_as_zero": False, "unobserved_pairs_generated": 0}
    return indices[chosen], chosen, facts


def raw_data_base(raw):
    # Same constant as the pinned RawRows composite user/date index.
    return date.max.toordinal() + 1


def feature_rows(pack, raw, indices, dynamic_bank=None):
    """Only fixed s0 and candidate identity reach the feature facade."""
    users = raw.users[indices]
    origins = np.full(len(indices), ORIGIN, np.int64)
    x, lengths, maxima = raw.history_batch(users, origins)
    if np.any(maxima >= ORIGIN) or np.any(raw.days[indices] < ORIGIN):
        raise ValueError("History crosses the fixed development origin")
    target_tokens = raw.tokens[indices]
    if np.any((x == target_tokens[:, None]) & (np.arange(64)[None, :] < lengths[:, None])):
        raise ValueError("Current target leaked into observed prefix")
    kwargs = {} if dynamic_bank is None else {"dynamic_bank": dynamic_bank}
    numeric = pack.user_features(users, origins, x, lengths, history_cutoffs=origins, **kwargs)
    candidate = pack.candidate_features(target_tokens, origins, history_cutoffs=origins, user_ids=users)
    if (np.asarray(numeric).shape != (len(indices), len(pack.user_columns)) or
            np.asarray(candidate).shape != (len(indices), len(pack.candidate_columns)) or
            not np.isfinite(numeric).all() or not np.isfinite(candidate).all()):
        raise ValueError("Invalid development feature blocks")
    return {"user_id": users, "item_id": raw.items[indices], "target_date": raw.days[indices],
            "history_cutoff": origins, "history_max_date": maxima, "history_length": lengths - 1,
            "numeric": np.asarray(numeric, np.float32), "candidate_numeric": np.asarray(candidate, np.float32),
            "categorical": np.empty((len(indices), 0), dtype="U1"),
            "reward": raw.outcomes[indices], "observed": np.ones(len(indices), bool)}, x, lengths


def evaluate_predictions(rows, predictions, scorer, rm, x):
    """All methods share exactly one observed row population and cold fallback."""
    support = scorer.priors.counts[scorer.catalog.indices(rows["item_id"])]
    token_support = np.r_[np.zeros(3, np.int64), scorer.priors.counts]
    active = np.arange(x.shape[1])[None, :] < (rows["history_length"] + 1)[:, None]
    history = active & (x >= 3)
    supported = token_support[x] > 0
    prefix_groups = {"prefix_has_fit_unsupported_item": np.any(history & ~supported, axis=1),
                     "prefix_all_known_items_fit_supported": np.all(~history | supported, axis=1),
                     "prefix_has_unknown_token": np.any(active & (x == 2), axis=1)}
    result = {}
    for name, pred in predictions.items():
        result[name] = {"metrics": rm.regression_metrics(rows["reward"], pred),
                        "segments": rm.segmented_metrics(rows, pred, scorer.catalog, scorer.priors)}
        for segment, mask in prefix_groups.items():
            result[name]["segments"][segment] = rm.regression_metrics(rows["reward"][mask], pred[mask]) if mask.any() else {"n": 0}
        bins = result[name]["metrics"]["calibration_bins"]
        result[name]["metrics"]["calibration_absolute_gap"] = float(sum(
            b["n"] * abs(b["predicted"] - b["observed"]) for b in bins if b["n"]) / len(support))
    coverage = {"rows": len(support), "fit_supported_target_rows": int(np.count_nonzero(support)),
                "fit_unsupported_target_rows": int(np.count_nonzero(support == 0)),
                "fit_supported_catalog_items": int(np.count_nonzero(scorer.priors.counts)),
                "catalog_items": len(scorer.priors.counts), "history_known_tokens": int(history.sum()),
                "history_fit_supported_tokens": int(np.count_nonzero(history & supported)),
                "history_unknown_tokens": int(np.count_nonzero(active & (x == 2))),
                "prefix_rows_with_unsupported_fit_items": int(prefix_groups["prefix_has_fit_unsupported_item"].sum()),
                "history_max_date": int(rows["history_max_date"].max()),
                "current_target_in_prefix": False}
    return result, coverage


def complete_prefix_coverage(raw, users, scorer):
    """Count the complete observed prefix once per distinct evaluated user."""
    users = np.unique(users)
    base = raw_data_base(raw)
    starts = np.searchsorted(raw.keys, users * base, "left")
    ends = np.searchsorted(raw.keys, users * base + ORIGIN, "left")
    token_support = np.r_[np.zeros(3, np.int64), scorer.priors.counts]
    known_count = supported_count = unknown_count = has_unsupported = 0
    for start, end in zip(starts, ends):
        tokens = raw.tokens[start:end]
        known = tokens >= 3
        support = token_support[tokens] > 0
        known_count += int(known.sum())
        supported_count += int(np.count_nonzero(known & support))
        unknown_count += int(np.count_nonzero(tokens == 2))
        has_unsupported += int(np.any(known & ~support))
    return {"population": "complete raw date<Aug9 prefix, once per distinct evaluated user",
            "users": len(users), "raw_rows": int(np.sum(ends - starts)), "known_item_rows": known_count,
            "fit_supported_item_rows": supported_count, "unknown_item_rows": unknown_count,
            "users_with_fit_unsupported_items": has_unsupported}


def full_raw_baselines(raw, rows, scorer, fit_cutoff):
    """Fixed analytic controls from all valid known raw fit pairs, without refits."""
    mask = (raw.days < fit_cutoff) & (raw.statuses == 0) & (raw.tokens >= 3)
    if not mask.any():
        raise ValueError("No full-raw fit population for analytic controls")
    y = raw.outcomes[mask].astype(np.float64)
    tokens, users = raw.tokens[mask], raw.users[mask]
    global_mean = float(y.mean())
    size = len(scorer.priors.counts) + 3
    counts = np.bincount(tokens, minlength=size)
    sums = np.bincount(tokens, weights=y, minlength=size)
    item_mean = (sums + 20. * global_mean) / (counts + 20.)
    starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    unique_users = users[starts]
    user_counts = np.diff(np.r_[starts, len(users)])
    user_means = (np.add.reduceat(y, starts) + 10. * global_mean) / (user_counts + 10.)
    locations = np.searchsorted(unique_users, rows["user_id"])
    found = locations < len(unique_users)
    safe = np.minimum(locations, len(unique_users) - 1)
    found &= unique_users[safe] == rows["user_id"]
    user = np.where(found, user_means[safe], global_mean)
    target_tokens = scorer.catalog.indices(rows["item_id"]) + 3
    item, zero = item_mean[target_tokens], counts[target_tokens] == 0
    if not np.array_equal(counts[3:] > 0, scorer.priors.counts > 0):
        raise ValueError("Full raw and serialized sampled fit support sets differ")
    predictions = {"full_raw_global_mean": np.full(len(item), global_mean),
                   "full_raw_item_shrinkage": item.copy(), "full_raw_user_mean": user.copy(),
                   "full_raw_combined": (item + user) / 2.}
    for value in predictions.values():
        value[zero] = global_mean
    facts = {"fit_cutoff_exclusive": "2021-08-02", "fit_rows": len(y), "global_mean": global_mean,
             "supported_items": int(np.count_nonzero(counts[3:])), "zero_support_items": int(np.count_nonzero(counts[3:] == 0)),
             "item_alpha": 20., "user_alpha": 10., "parameters_selected_on_development_labels": False,
             "target_full_raw_fit_count": counts[target_tokens],
             "purpose": "strong analytic controls; serialized baselines separately preserve sampled RM-fit scorer parity"}
    return predictions, facts


def paired_user_bootstrap(rows, predictions, replicates=200, seed=20261020):
    """Fixed descriptive paired intervals; resample users, preserving tied rows."""
    users, inverse = np.unique(rows["user_id"], return_inverse=True)
    y = np.asarray(rows["reward"], np.float64)
    names = list(predictions)
    n = len(users)
    counts = np.bincount(inverse, minlength=n)
    sum_y = np.bincount(inverse, weights=y, minlength=n)
    sum_y2 = np.bincount(inverse, weights=y * y, minlength=n)
    square = np.column_stack([np.bincount(inverse, weights=(predictions[name] - y) ** 2, minlength=n) for name in names])
    absolute = np.column_stack([np.bincount(inverse, weights=np.abs(predictions[name] - y), minlength=n) for name in names])
    rng = np.random.default_rng(seed)
    rmses, maes, r2s = [], [], []
    for _ in range(replicates):
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n)
        total = float(weights @ counts)
        squared_error = weights @ square
        variance = float(weights @ sum_y2) - float(weights @ sum_y) ** 2 / total
        rmses.append(np.sqrt(squared_error / total))
        maes.append((weights @ absolute) / total)
        r2s.append(1. - squared_error / variance if variance > 1e-12 else np.full(len(names), np.nan))
    rmses, maes, r2s = map(np.asarray, (rmses, maes, r2s))
    def interval(values):
        valid = values[np.isfinite(values)]
        return [float(v) for v in np.quantile(valid, [.025, .975])] if len(valid) else None
    intervals = {name: {"rmse": interval(rmses[:, j]), "mae": interval(maes[:, j]), "r2": interval(r2s[:, j])}
                 for j, name in enumerate(names)}
    paired = {}
    if "rm_calibrated" in names:
        reference = names.index("rm_calibrated")
        paired = {name: interval(rmses[:, reference] - rmses[:, j]) for j, name in enumerate(names) if j != reference}
    return {"method": "paired user-cluster percentile bootstrap", "replicates": replicates, "seed": seed,
            "confidence": .95, "users": n, "intervals": intervals,
            "calibrated_rm_minus_other_rmse": paired,
            "limitations": "descriptive sampling uncertainty on reused development users; excludes model/metadata/label uncertainty and selection bias"}


def process_memory():
    """Read this CPU process's OS high-water mark without a monitoring daemon."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                "PeakPagefileUsage", "PrivateUsage")]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"source": "Windows GetProcessMemoryInfo", "peak_working_set_bytes": counters.PeakWorkingSetSize,
                "working_set_bytes": counters.WorkingSetSize, "peak_private_commit_bytes": counters.PeakPagefileUsage,
                "private_commit_bytes": counters.PrivateUsage}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"source": "getrusage", "peak_working_set_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}


def evaluate_reward(raw_dir, metadata_path, basic_npz, val_path, rm_bundle, artifact,
                    source_bundle, freeze_path, output, *, max_rows=MAX_ROWS,
                    sft_cache_manifest=None, batch_size=256, eval_seconds=600, init_seconds=240, progress=None):
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 1024 or
            not 0 < eval_seconds <= 600 or not 0 < init_seconds <= 240):
        raise ValueError("Bounded batch/init/evaluation budgets are required")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError("max_rows must be an integer in [1,20000]")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output must be a new directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    started = time.perf_counter()
    rm, data, pack_module, freeze = frozen_runtime(source_bundle, freeze_path)
    raw_dir, metadata_path, basic_npz, val_path, rm_bundle, artifact = map(Path,
        (raw_dir, metadata_path, basic_npz, val_path, rm_bundle, artifact))
    if sft_cache_manifest is None:
        raise ValueError("The prior SFT cache manifest must explicitly bind validation input")
    sft_receipt, sft_receipt_sha = verify_sft_validation(sft_cache_manifest, val_path, metadata_path, rm)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = data.checked_mapping(metadata)
    bundle_manifest_sha = rm.file_hash(rm_bundle / "manifest.json")
    model_manifest_sha = rm.file_hash(artifact / "manifest.json")
    bundle_roles, catalog, bundle_manifest = rm.load_bundle(rm_bundle)
    scorer = rm.RewardModelScorer.load(artifact, rm.canonical_hash({str(k): v for k, v in mapping.items()}))
    model_manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    verify_model_training_bundle(model_manifest, bundle_manifest_sha)
    booster_boundary, producer_paths = verify_boosting_boundary(source_bundle, freeze,
        catalog.provenance["source_pins"]["boosting"], model_manifest, rm)
    if model_manifest.get("backend") != "catboost_cpu":
        raise ValueError("Development quality evaluation requires a learned CatBoost artifact")
    if scorer.catalog.describe() != catalog.describe():
        raise ValueError("Scorer and observed RM bundle schemas/provenance differ")
    if (scorer.calibration.get("fit_role") != "calibration" or
            scorer.calibration.get("kind") != "bounded_affine"):
        raise ValueError("Scorer must have its already frozen calibration-user calibrator")
    paths = {"eval_source": Path(__file__), "freeze": Path(freeze_path), "metadata": metadata_path,
             "basic_npz": basic_npz, "basic_manifest": basic_npz.with_name("manifest.json"),
             "prepared_val": val_path, "rm_bundle_manifest": rm_bundle / "manifest.json",
             "rm_model_manifest": artifact / "manifest.json", "sft_cache_manifest": Path(sft_cache_manifest)}
    paths.update({"raw_" + name: raw_dir / (name + ".csv") for name in ("items", "users", "interactions")})
    paths.update({"source/" + name: Path(source_bundle) / name for name in freeze["sources"]})
    paths.update(producer_paths)
    for name, expected in {**bundle_manifest["files"], **bundle_manifest.get("sidecars", {})}.items():
        paths["rm_bundle/" + name] = rm_bundle / name
    for name in model_manifest["files"]:
        paths["rm_model/" + name] = artifact / name
    before = {key: rm.file_hash(path) for key, path in paths.items()}
    if before["rm_bundle_manifest"] != bundle_manifest_sha or before["rm_model_manifest"] != model_manifest_sha:
        raise ValueError("Manifest changed while loading frozen RM")
    if before["sft_cache_manifest"] != sft_receipt_sha:
        raise ValueError("Prior SFT cache receipt changed during loading")
    if verify_boosting_boundary(source_bundle, freeze, catalog.provenance["source_pins"]["boosting"], model_manifest, rm)[0] != booster_boundary:
        raise ValueError("Booster producer/serving chain changed during loading")
    for name, expected in {**bundle_manifest["files"], **bundle_manifest.get("sidecars", {})}.items():
        if before["rm_bundle/" + name] != expected:
            raise ValueError("RM bundle checksum changed while loading: " + name)
    for name, expected in model_manifest["files"].items():
        if before["rm_model/" + name] != expected:
            raise ValueError("RM model checksum changed while loading: " + name)
    for name, expected in freeze["sources"].items():
        if before["source/" + name] != expected:
            raise ValueError("Frozen source changed while loading: " + name)
    expected_sources = catalog.provenance["source_pins"]
    for key, direct in (("metadata", "metadata"), ("basic_npz", "basic_npz"),
                        ("interactions", "raw_interactions"), ("users", "raw_users"), ("items", "raw_items")):
        if expected_sources.get(key) != before[direct]:
            raise ValueError("Evaluation input differs from frozen RM source: " + key)
    for name in freeze["sources"]:
        short = Path(name).stem
        # Booster producer/serving versions are separately checked against their
        # exact freeze chain; the data builder and every feature remain exact.
        key = "builder" if short == "kion_reward_data" else "component/" + short
        if key in expected_sources and expected_sources[key] != before["source/" + name]:
            raise ValueError("RM source implementation differs from source freeze: " + name)
    # Free the role feature memmaps; they are validation inputs, never fitted here.
    del bundle_roles
    pack_start = time.perf_counter()
    pack = pack_module.FeaturePack.build(raw_dir, metadata_path, basic_npz,
                                        fit_cutoff="2021-08-02", include_relative=True)
    init_time = time.perf_counter() - pack_start
    if init_time > init_seconds:
        raise TimeoutError("FeaturePack initialization exceeded its CPU time budget")
    if pack.fit_cutoff != "2021-08-02" or pack.user_columns != catalog.context_names or pack.candidate_columns != catalog.item_names:
        raise ValueError("FeaturePack cutoff or ordered feature schema differs from frozen RM")
    producer = booster_boundary["feature_source_bundle"]
    feature_freeze = freeze if Path(producer).resolve() == Path(source_bundle).resolve() else verify_freeze(producer, freeze["parent_freeze"])
    verify_feature_modules(Path(producer) / "scripts", feature_freeze, rm)
    facade_timestamp = verify_facade_provenance(pack.provenance, catalog.provenance["facade"], rm)
    declared_sources = data._declared_sources(pack.provenance)
    data._check_sources(declared_sources)
    evaluation_start = time.perf_counter()
    raw = data.RawRows.from_pack(pack, mapping)
    with np.load(val_path, allow_pickle=False) as source:
        prepared = {k: source[k] for k in ("user_id", "target_item_id", "target_date", "y", "reward", "x", "lengths")}
    if len(prepared["user_id"]) != sft_receipt["validation_rows"]:
        raise ValueError("Validation row count differs from prior SFT cache receipt")
    indices, prepared_indices, facts = select_targets(raw, prepared, mapping, rm, max_rows)
    del prepared
    blocks, sequences, lengths_all = [], [], []
    prediction_blocks = {name: [] for name in ("rm_calibrated", "rm_uncalibrated", *BASELINES)}
    dynamic = getattr(pack, "dynamic", None)
    bank = dynamic.day_bank(ORIGIN) if dynamic is not None else None
    for begin in range(0, len(indices), batch_size):
        if time.perf_counter() - evaluation_start > eval_seconds:
            raise TimeoutError("Development evaluation exceeded its CPU time budget")
        rows, x, lengths = feature_rows(pack, raw, indices[begin:begin + batch_size], bank)
        catalog.validate_context(rows)
        for name in prediction_blocks:
            pred = (scorer.score_rows(rows, calibrated=name == "rm_calibrated") if name.startswith("rm_")
                    else scorer.priors.predict(rows, catalog, name))
            prediction_blocks[name].append(pred)
        blocks.append(rows)
        sequences.append(x)
        lengths_all.append(lengths)
        if progress is not None:
            progress({"rows": min(begin + batch_size, len(indices)), "total": len(indices),
                      "eval_seconds": round(time.perf_counter() - evaluation_start, 3)})
    rows = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    del blocks
    x, lengths = np.concatenate(sequences), np.concatenate(lengths_all)
    predictions = {name: np.concatenate(parts) for name, parts in prediction_blocks.items()}
    full_predictions, full_prior_facts = full_raw_baselines(raw, rows, scorer, rm.FIT_CUTOFF)
    full_target_support = full_prior_facts.pop("target_full_raw_fit_count")
    predictions.update(full_predictions)
    metrics, coverage = evaluate_predictions(rows, predictions, scorer, rm, x)
    uncertainty = paired_user_bootstrap(rows, predictions)
    coverage["complete_prefix"] = complete_prefix_coverage(raw, rows["user_id"], scorer)
    # Recheck all bindings after scoring, before a ready receipt can be published.
    if any(rm.file_hash(path) != before[key] for key, path in paths.items()):
        raise ValueError("Pinned input/model/source changed during evaluation")
    data._check_sources(declared_sources)
    if time.perf_counter() - evaluation_start > eval_seconds:
        raise TimeoutError("Development evaluation exceeded its CPU time budget")
    report = {"contract": CONTRACT, "state": "completed", "population": facts, "metrics": metrics,
              "coverage": coverage, "uncertainty": uncertainty, "full_raw_fit_controls": full_prior_facts,
              "boosting_version_boundary": booster_boundary,
              "facade_provenance_timestamp": facade_timestamp,
              "segment_support_definition": "serialized sampled RM-fit target counts; full/raw zero support sets checked equal",
              "selected_model": scorer.config, "calibration": scorer.calibration,
              "scorer_signature": scorer.signature, "schema_signature": catalog.schema_signature,
              "mapping_sha256": catalog.mapping_sha256, "cpu_threads": 2, "GPU_used": False,
              "development_labels_used_for_fitting_or_selection": False,
              "clocks": {"targets": "2021-08-09..2021-08-15", "query_origin": "2021-08-09",
                         "history": "BOS plus last63 raw rows strictly before 2021-08-09",
                         "candidate_dynamic_and_feature_priors": "strictly before 2021-08-02",
                         "past_history_outcomes": "raw observations before Aug9; strict last-watch-date availability assumption"},
              "timings_seconds": {"pack_init": init_time, "evaluation": time.perf_counter() - evaluation_start,
                                  "total": time.perf_counter() - started},
              "budgets_seconds": {"pack_init": init_seconds, "evaluation": eval_seconds},
              "feature_hashes": {key: rm.array_hash(rows[key]) for key in ("numeric", "candidate_numeric")},
              "limitations": ["SFT development labels are reused; this is not an untouched test",
                              "observed user-item last-watch pairs and aggregated watched_pct, not causal or impression reward",
                              "retrospective static metadata has unverified historical availability",
                              "raw same-day order is arbitrary and no same-day target enters the prefix",
                              "uncalibrated scorer outputs are clipped to [0,1]; cold items always use frozen fit global mean",
                              "only the already selected saved model is evaluated; discarded candidates are unavailable"]}
    output.mkdir(parents=True)
    retained = {key: rows[key] for key in ("user_id", "item_id", "target_date", "history_cutoff", "history_max_date", "history_length", "reward")}
    np.savez_compressed(output / "predictions.npz", **retained, **predictions, x=x, lengths=lengths,
                        raw_row_index=indices, prepared_row_index=prepared_indices,
                        fit_target_support=scorer.priors.counts[catalog.indices(rows["item_id"])],
                        full_raw_fit_target_support=full_target_support)
    report["timings_seconds"].update(evaluation=time.perf_counter() - evaluation_start, total=time.perf_counter() - started)
    report["process_memory"] = process_memory()
    if report["timings_seconds"]["evaluation"] > eval_seconds:
        raise TimeoutError("Development evaluation serialization exceeded its CPU time budget")
    rm.write_json(output / "report.json", report)
    manifest = {"contract": CONTRACT, "ready": True, "report": "report.json",
                "files": {name: rm.file_hash(output / name) for name in ("report.json", "predictions.npz")},
                "inputs": {key: {"path": paths[key].resolve().as_posix(), "sha256": value} for key, value in before.items()},
                "scorer_signature": scorer.signature, "row_binding": rm.canonical_hash({key: rm.array_hash(value) for key, value in retained.items()}),
                "policy_mapping_sha256": catalog.mapping_sha256, "schema_signature": catalog.schema_signature}
    rm.write_json(output / "manifest.json", manifest)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "metadata", "basic-npz", "val", "rm-bundle", "artifact", "source-bundle", "freeze", "output", "sft-cache-manifest"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-seconds", type=float, default=600)
    parser.add_argument("--init-seconds", type=float, default=240)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        report = evaluate_reward(args.raw, args.metadata, args.basic_npz, args.val, args.rm_bundle, args.artifact,
                                 args.source_bundle, args.freeze, args.output, max_rows=args.max_rows,
                                 sft_cache_manifest=args.sft_cache_manifest,
                                 batch_size=args.batch_size, eval_seconds=args.eval_seconds, init_seconds=args.init_seconds,
                                 progress=lambda p: print(json.dumps(p), flush=True))
    except Exception as error:
        failure = {"state": "failed", "ready": False, "seconds": time.perf_counter() - started,
                   "error_type": type(error).__name__, "error": str(error)}
        try:
            failure["process_memory"] = process_memory()
        except Exception as telemetry_error:
            failure["telemetry_error"] = str(telemetry_error)
        print(json.dumps(failure, allow_nan=False), file=sys.stderr, flush=True)
        raise
    print(json.dumps({"state": "completed", "rows": report["population"]["evaluated_rows"],
                      "metrics": {k: {m: v["metrics"][m] for m in ("rmse", "mae", "r2")} for k, v in report["metrics"].items()}}), flush=True)


if __name__ == "__main__":
    main()
