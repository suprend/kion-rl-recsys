"""Portable frozen local RM-r2 prediction; imports numerical packages only in run."""
from __future__ import annotations

import math
from pathlib import Path
import sys

SIGNATURE = "ee14bc4d7375a4d6978a2e1f476cd795369b45ef4250501b2f5992767e9ed261"
SCHEMA = "7776a9d611b04ef8f3ef77fce0a67c1fb045f67142bf4b4bbde24cc469f9bda4"
MAPPING = "b90df0d1fcce007008b62007e486bc1e9cb0b4ac89c97bb5107e84305e744db4"
MODEL_SHA = "b6ed4c6427129afcddec017e2f64d9c3fb59fb31fc300e5c9d074964a21ddd62"
PRIOR_SHA = "d5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4"
SLOPE = 1.0148506983671792
INTERCEPT = -0.005568380749024937
GLOBAL_MEAN = 0.47802770002556966
ARRAY_NAMES = ("candidate_raw_ids", "rm_context_numeric", "rm_candidate_numeric",
               "rm_fit_counts", "rm_item_prior", "rm_reference_reward",
               "rm_reference_reward64", "rm_reference_support")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calibrate_scalar(prediction, count, *, slope=SLOPE, intercept=INTERCEPT, global_mean=GLOBAL_MEAN):
    """Small stdlib seam for testing native clip-before-affine/cold semantics."""
    require(type(count) is int and count >= 0, "Nonnegative fit count required")
    require(all(math.isfinite(v) for v in (prediction, slope, intercept, global_mean)),
            "Nonfinite RM input")
    require(0 <= global_mean <= 1, "Global mean outside [0,1]")
    if count == 0:
        return global_mean
    raw = min(1., max(0., prediction))
    return min(1., max(0., slope * raw + intercept))


def validate_metadata(meta):
    require(isinstance(meta, dict), "Missing RM metadata")
    require(meta.get("scorer_signature") == SIGNATURE and meta.get("schema_signature") == SCHEMA
            and meta.get("mapping_sha256") == MAPPING and meta.get("prior_archive_sha256") == PRIOR_SHA,
            "Frozen RM provenance differs")
    require(meta.get("context_width") == 2079 and meta.get("candidate_width") == 2008
            and meta.get("numeric_width") == 4087 and meta.get("categorical_rule") == "item:{original_item_id}"
            and meta.get("compatibility_pairs") == [] and meta.get("history") is True,
            "Exact RM2079+2008 schema required")
    require(meta.get("calibration") == {"slope": SLOPE, "intercept": INTERCEPT}
            and meta.get("global_mean") == GLOBAL_MEAN, "Frozen calibration/fallback differs")
    require(meta.get("row_positions") == [0, 1, 2, 3], "Fixed four-state selection required")


def predict_blocks(np, model, FeaturesData, Pool, context, candidates, ids, counts):
    """Recompute trees in native-shaped128 chunks; no saved rewards are used."""
    n, k = ids.shape
    result = np.empty((n, k), dtype=np.float64)
    for row in range(n):
        for start in range(0, k, 128):
            stop = min(start + 128, k)
            numeric = np.ascontiguousarray(np.concatenate((
                np.repeat(context[row:row + 1], stop - start, axis=0),
                candidates[row, start:stop]), axis=1), dtype=np.float32)
            cats = np.empty((stop - start, 1), dtype=object)
            cats[:, 0] = [f"item:{int(v)}".encode("utf-8") for v in ids[row, start:stop]]
            pool = Pool(FeaturesData(num_feature_data=numeric, cat_feature_data=cats), thread_count=2)
            pred = np.asarray(model.predict(pool, thread_count=2), dtype=np.float64)
            require(pred.shape == (stop - start,) and bool(np.isfinite(pred).all()), "Invalid tree prediction")
            zero = counts[row, start:stop] == 0
            pred[zero] = GLOBAL_MEAN
            pred = np.clip(pred, 0., 1.)
            pred = np.clip(SLOPE * pred + INTERCEPT, 0., 1.)
            pred[zero] = GLOBAL_MEAN
            result[row, start:stop] = pred
    return result


def run(fixture_root, manifest, output):
    """CLI-owned fresh output directory; execute in the separate RM environment."""
    from .io import close_arrays, load_arrays, verify_descriptor

    require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules),
            "RM replay needs a separate fresh process before actor imports")
    validate_metadata(manifest.get("rm"))
    weight = manifest.get("weights", {}).get("RM", {})
    require(weight.get("sha256") == MODEL_SHA and weight.get("bytes") == 163368,
            "Exact frozen CBM descriptor required")
    model_path = verify_descriptor(fixture_root, weight, max_bytes=1 << 20)
    import numpy as np
    import catboost
    from catboost import CatBoostRegressor, FeaturesData, Pool

    require(np.__version__ == "2.5.4" and catboost.__version__ == "1.2.10",
            "RM requires its pinned NumPy2.5.4/CatBoost1.2.10 environment")
    arrays = load_arrays(np, fixture_root, manifest, names=ARRAY_NAMES)
    try:
        return _run_loaded(np, catboost, CatBoostRegressor, FeaturesData, Pool, model_path, arrays, output)
    finally:
        close_arrays(arrays)


def _run_loaded(np, catboost, CatBoostRegressor, FeaturesData, Pool, model_path, arrays, output):
    ids, context, candidates, counts, prior = (arrays[name] for name in ARRAY_NAMES[:5])
    require(ids.shape == (4, 1024) and ids.dtype.kind in "iu" and bool((ids >= 0).all()),
            "Four original-ID natural1024 pools required")
    require(all(len(np.unique(row)) == 1024 for row in ids), "Duplicate candidate ID")
    require(context.shape == (4, 2079) and context.dtype == np.dtype("float32")
            and candidates.shape == (4, 1024, 2008) and candidates.dtype == np.dtype("float32"),
            "Exact float32 RM context/candidate blocks required")
    require(counts.shape == ids.shape and counts.dtype.kind in "iu" and bool((counts >= 0).all()),
            "Invalid frozen fit counts")
    require(prior.shape == ids.shape and prior.dtype == np.dtype("float64")
            and bool(((prior >= 0) & (prior <= 1)).all()), "Invalid frozen item prior")
    # The same public item has the same fitted prior wherever it occurs.
    lookup = {}
    for item, count, mean in zip(ids.flat, counts.flat, prior.flat):
        record = (int(count), float(mean))
        require(int(item) not in lookup or lookup[int(item)] == record, "Inconsistent item-prior binding")
        lookup[int(item)] = record
    model = CatBoostRegressor(task_type="CPU", thread_count=2)
    model.load_model(str(model_path))
    reward64 = predict_blocks(np, model, FeaturesData, Pool, context, candidates, ids, counts)
    support = counts > 0
    reward = reward64.astype(np.float32)
    utility = np.where(support, reward.astype(np.float64) - prior, 0.)
    reference = arrays["rm_reference_reward"]
    reference64 = arrays["rm_reference_reward64"]
    reference_support = arrays["rm_reference_support"]
    require(reference.shape == ids.shape and reference.dtype == np.dtype("float32"), "Invalid float32 reference")
    require(reference64.shape == ids.shape and reference64.dtype == np.dtype("float64"), "Invalid float64 reference")
    require(reference_support.shape == ids.shape and reference_support.dtype == np.dtype(bool), "Invalid support reference")
    error64 = float(np.max(np.abs(reward64 - reference64)))
    require(error64 <= 1e-12, "Native float64 RM parity failed")
    require(bool(np.array_equal(reward, reference)), "Historical float32 RM parity failed")
    require(bool(np.array_equal(support, reference_support)), "Frozen support parity failed")
    require(bool(np.isfinite(utility).all()) and bool(((utility >= -1) & (utility <= 1)).all()), "Residual range failed")
    output = Path(output)
    require(output.is_dir(), "CLI must create its fresh output directory")
    outputs = {}
    from .io import sha256
    for name, value in (("rm_reward64", reward64), ("rm_reward", reward), ("rm_support", support),
                        ("rm_prior", prior), ("rm_utility", utility)):
        path = output / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
        outputs[name] = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size,
                         "dtype": str(value.dtype), "shape": list(value.shape)}
    return {"contract": "kion-portable-rm-replay-v1", "states": 4, "candidates_per_state": 1024,
            "actual_model_prediction": True, "training_updates": 0, "rm_fits": 0,
            "native_float64_max_abs": error64, "historical_float32_exact": True,
            "support_exact": True, "supported_pairs": int(support.sum()),
            "numpy": np.__version__, "catboost": catboost.__version__, "python": sys.version,
            "scorer_signature": SIGNATURE, "outputs": outputs}
