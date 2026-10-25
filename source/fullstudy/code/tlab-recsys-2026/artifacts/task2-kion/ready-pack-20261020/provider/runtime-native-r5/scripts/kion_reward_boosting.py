"""CPU tabular KION reward model, observed-pair temporal data contract and scorer.

This module never reads development/test data or starts a GPU job. The label is
finite observed watched_pct / 100; missing labels and unobserved pairs are not
negative examples. Static metadata remains a retrospective-snapshot experiment.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import numpy as np


CONTRACT = "kion-reward-boosting-v1"
FIT_CUTOFF = date(2021, 8, 2).toordinal()
WINDOW_STOP = date(2021, 8, 9).toordinal()
PARTITION = {"algorithm": "sha256", "salt": "kion-rm-select-calibration-v1-20261020",
             "canonical_key": "salt + ':' + decimal original user_id, UTF-8",
             "select": "first unsigned big-endian 64 bits < 2**63", "boundary": 2**63}
DEFAULT_CONFIGS = ({"name": "no_history_depth4", "history": False, "depth": 4, "iterations": 160},
                   {"name": "history_depth4", "history": True, "depth": 4, "iterations": 160})
FORBIDDEN = {"reward", "watched_pct", "total_dur", "target_item_id", "target_date", "y", "label"}
ROW_FIELDS = ("user_id", "item_id", "target_date", "history_cutoff", "history_max_date",
              "history_length", "numeric", "categorical", "reward", "observed")
PAIR_ROW_FIELDS = (*ROW_FIELDS, "candidate_numeric")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                                      sort_keys=True).encode("ascii") + b"\n")
    if a.size:
        digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def checked_ids(values, name):
    result = np.asarray(values)
    if result.ndim == 1 and result.size == 0:
        return np.empty(0, dtype=np.int64)
    if result.dtype.kind not in "iu" or result.ndim != 1 or np.any(result < 0):
        raise ValueError(f"{name} must be one-dimensional nonnegative integer ORIGINAL IDs")
    if result.dtype.kind == "u" and np.any(result > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceed int64")
    return result.astype(np.int64, copy=False)


def cohort(user_ids, partition=None):
    spec = PARTITION if partition is None else partition
    if spec != PARTITION:
        raise ValueError("RM cohort partition must equal the pinned canonical partition")
    users = checked_ids(user_ids, "user_ids")
    unique, inverse = np.unique(users, return_inverse=True)
    buckets = [int.from_bytes(hashlib.sha256(f"{spec['salt']}:{int(u)}".encode("utf-8")).digest()[:8],
                              "big") < spec["boundary"] for u in unique]
    return np.asarray(buckets, dtype=bool)[inverse]


def _names(names):
    if not isinstance(names, (list, tuple)) or len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    for name in names:
        if (not isinstance(name, str) or not name or name in FORBIDDEN or
                name.startswith("target_") or ("total_dur" in name and not name.startswith("history_")) or
                ("watched_pct" in name and not name.startswith("history_"))):
            raise ValueError(f"Outcome/target or invalid feature name: {name}")
    return list(names)


class FeatureCatalog:
    """Candidate matrix in policy-token order, with explicit ORIGINAL item IDs.

    Context vectors are supplied by the common local-feature facade. Compatibility
    pairs declare (context numeric index, candidate numeric index); their product
    and absolute difference are added to the tree matrix. No raw user ID is a
    model feature. The item categorical ID is prefixed and never a policy token.
    """
    def __init__(self, item_ids, item_numeric, context_names, item_names,
                 context_categorical_names=(), item_categorical=None, item_categorical_names=(),
                 history_mask=None, compatibility_pairs=(), provenance=None):
        self.item_ids = checked_ids(item_ids, "catalog item_ids").copy()
        if not len(self.item_ids) or len(np.unique(self.item_ids)) != len(self.item_ids):
            raise ValueError("Catalog IDs must be nonempty and unique")
        self.item_numeric = np.asarray(item_numeric, dtype=np.float32).copy()
        self.context_names, self.item_names = _names(context_names), _names(item_names)
        self.context_categorical_names = _names(context_categorical_names)
        self.item_categorical_names = _names(item_categorical_names)
        if self.item_numeric.shape != (len(self.item_ids), len(self.item_names)) or np.isinf(self.item_numeric).any():
            raise ValueError("Invalid candidate numeric matrix")
        if item_categorical is None:
            item_categorical = np.empty((len(self.item_ids), 0), dtype="U1")
        a = np.asarray(item_categorical)
        if a.dtype.kind not in "US" or a.shape != (len(self.item_ids), len(self.item_categorical_names)):
            raise ValueError("Candidate categoricals must be a string matrix")
        self.item_categorical = a.astype(str).copy()
        self.history_mask = np.asarray(history_mask if history_mask is not None else
                                       [True] * len(self.context_names), dtype=bool)
        if self.history_mask.shape != (len(self.context_names),):
            raise ValueError("Invalid context history mask")
        self.compatibility_pairs = [tuple(pair) for pair in compatibility_pairs]
        for pair in self.compatibility_pairs:
            if len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, int) for i in pair):
                raise ValueError("Compatibility pair indices must be integers")
            if not (0 <= pair[0] < len(self.context_names) and 0 <= pair[1] < len(self.item_names)):
                raise ValueError("Compatibility index out of bounds")
        self.provenance = dict(provenance or {})
        if self.provenance.get("fit_cutoff_exclusive") != "2021-08-02":
            raise ValueError("RM feature transforms must be fitted strictly before Aug2")
        if self.provenance.get("metadata_availability") != "static_snapshot_unverified":
            raise ValueError("Declare retrospective static metadata availability")
        pins = self.provenance.get("source_pins")
        if not isinstance(pins, dict) or not pins or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in pins.values()):
            raise ValueError("Feature provenance needs nonempty SHA256 source pins")
        self._lookup = {int(i): n for n, i in enumerate(self.item_ids)}
        self.mapping = {str(int(i)): n + 3 for n, i in enumerate(self.item_ids)}
        self.mapping_sha256 = canonical_hash(self.mapping)
        for a in (self.item_ids, self.item_numeric, self.item_categorical, self.history_mask):
            a.flags.writeable = False
        self._schema_signature = canonical_hash(self.describe())

    def describe(self):
        return {"context_names": self.context_names, "item_names": self.item_names,
                "context_categorical_names": self.context_categorical_names,
                "item_categorical_names": self.item_categorical_names,
                "history_mask": self.history_mask.tolist(), "compatibility_pairs": self.compatibility_pairs,
                "provenance": self.provenance, "mapping_sha256": self.mapping_sha256,
                "id_space": "original_item_id", "candidate_arrays": {
                    name: array_hash(getattr(self, name)) for name in ("item_ids", "item_numeric", "item_categorical")}}

    @property
    def schema_signature(self):
        return self._schema_signature

    def indices(self, item_ids):
        values = checked_ids(item_ids, "candidate_ids")
        try:
            return np.fromiter((self._lookup[int(i)] for i in values), dtype=np.int64, count=len(values))
        except KeyError as error:
            raise ValueError("Candidate ORIGINAL item ID is absent from frozen policy mapping") from error

    def validate_context(self, rows):
        numeric, categorical = np.asarray(rows["numeric"]), np.asarray(rows["categorical"])
        n = len(rows["item_id"])
        if numeric.shape != (n, len(self.context_names)) or np.isinf(numeric).any():
            raise ValueError("Context numeric schema mismatch or infinity")
        if categorical.shape != (n, len(self.context_categorical_names)) or categorical.dtype.kind not in "US":
            raise ValueError("Context categorical schema mismatch")
        candidate = rows.get("candidate_numeric")
        if candidate is None:
            if self.provenance.get("candidate_features_dynamic", False):
                raise ValueError("Dynamic RM candidate features require bound per-pair candidate_numeric")
        else:
            a = np.asarray(candidate)
            if a.shape != (n, len(self.item_names)) or a.dtype.kind != "f" or np.isinf(a).any():
                raise ValueError("Candidate numeric schema mismatch or infinity")
        return numeric, categorical

    def blocks(self, rows, history=True):
        numeric, categorical = self.validate_context(rows)
        indices = self.indices(rows["item_id"])
        mask = np.ones(len(self.context_names), bool) if history else ~self.history_mask
        candidate = np.asarray(rows["candidate_numeric"]) if "candidate_numeric" in rows else self.item_numeric[indices]
        pieces = [numeric[:, mask], candidate]
        pairs = [p for p in self.compatibility_pairs if history or not self.history_mask[p[0]]]
        for ci, ii in pairs:
            a, b = numeric[:, ci:ci + 1], candidate[:, ii:ii + 1]
            pieces.extend((a * b, np.abs(a - b)))
        values = np.ascontiguousarray(np.concatenate(pieces, axis=1), dtype=np.float32)
        cats = np.column_stack((categorical, self.item_categorical[indices],
                                np.asarray([f"item:{int(i)}" for i in rows["item_id"]])))
        return values, cats

    def matrix(self, rows, history=True):
        """Small inspection matrix; training uses float32 FeaturesData blocks."""
        values, cats = self.blocks(rows, history)
        n = len(values)
        matrix = np.empty((n, values.shape[1] + cats.shape[1]), dtype=object)
        matrix[:, :values.shape[1]], matrix[:, values.shape[1]:] = values, cats
        return matrix, list(range(values.shape[1], matrix.shape[1]))


def validate_rows(rows, role, catalog, partition=None):
    if role not in ("fit", "select", "calibration") or set(rows) not in (set(ROW_FIELDS), set(PAIR_ROW_FIELDS)):
        raise ValueError("Unknown role or incomplete/extra RM row fields")
    n = len(rows["item_id"])
    if n == 0:
        raise ValueError(f"Empty RM {role} rows")
    for name in ("user_id", "item_id", "target_date", "history_cutoff", "history_max_date", "history_length"):
        a = np.asarray(rows[name])
        if a.shape != (n,) or a.dtype != np.dtype("int64") or np.any(a < 0):
            raise ValueError(f"Invalid row field {name}")
    observed = np.asarray(rows["observed"])
    if observed.dtype != np.dtype(bool) or observed.shape != (n,) or not observed.all():
        raise ValueError("Every RM label must be an observed pair; no sampled negatives")
    rewards = np.asarray(rows["reward"])
    if rewards.shape != (n,) or rewards.dtype.kind != "f" or not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1)):
        raise ValueError("RM eligible rows require finite normalized observed rewards")
    days, cutoff, maxima = (np.asarray(rows[k]) for k in ("target_date", "history_cutoff", "history_max_date"))
    if role == "fit":
        if not np.all((days > 0) & (days < FIT_CUTOFF)) or not np.array_equal(cutoff, days):
            raise ValueError("RM fit requires strict-past target-specific histories before Aug2")
    else:
        if not np.all((days >= FIT_CUTOFF) & (days < WINDOW_STOP)) or not np.all(cutoff == FIT_CUTOFF):
            raise ValueError("RM select/calibration require Aug2–8 targets and frozen histories before Aug2")
        expected = cohort(rows["user_id"], partition)
        if not np.all(expected if role == "select" else ~expected):
            raise ValueError("RM select/calibration user cohort mismatch")
    if not np.all(maxima < cutoff) or np.any((rows["history_length"] == 0) != (maxima == 0)):
        raise ValueError("History evidence is not strictly before its cutoff or inconsistent with empty history")
    order = np.lexsort((rows["target_date"], rows["item_id"], rows["user_id"]))
    if n > 1:
        l, r = order[:-1], order[1:]
        if np.any((rows["target_date"][l] == rows["target_date"][r]) &
                  (rows["user_id"][l] == rows["user_id"][r]) & (rows["item_id"][l] == rows["item_id"][r])):
            raise ValueError("Duplicate observed target row")
    catalog.validate_context(rows)
    catalog.indices(rows["item_id"])
    return n


def deterministic_sample_indices(user_ids, item_ids, days, maximum, preserve_items=False):
    """Outcome-independent SHA row sample, optionally retaining every fit item.

    Keeping one hashed representative per fit item prevents sampling from silently
    creating additional unsupported item categories. Statistics/support may still
    be fitted only on this declared sampled RM-fit population.
    """
    users, items = checked_ids(user_ids, "user_ids"), checked_ids(item_ids, "item_ids")
    days = np.asarray(days)
    if days.shape != users.shape or items.shape != users.shape or days.dtype != np.dtype("int64"):
        raise ValueError("Invalid sampling row keys")
    if maximum is None or maximum >= len(users):
        return np.arange(len(users), dtype=np.int64)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("Sampling maximum must be a positive integer")
    rank = np.fromiter((int.from_bytes(hashlib.sha256(f"kion-rm-row-sample-v1:{int(u)}:{int(i)}:{int(d)}".encode("ascii")).digest()[:8], "big")
                        for u, i, d in zip(users, items, days)), dtype=np.uint64, count=len(users))
    order = np.argsort(rank, kind="stable")
    mandatory = np.empty(0, dtype=np.int64)
    if preserve_items:
        _, first = np.unique(items[order], return_index=True)
        mandatory = order[first]
        if len(mandatory) > maximum:
            raise ValueError("Fit sample cap is too small to preserve every supported item")
    keep = np.zeros(len(users), bool)
    keep[mandatory] = True
    extra = order[~keep[order]][:maximum - len(mandatory)]
    keep[extra] = True
    return np.flatnonzero(keep).astype(np.int64)


def build_temporal_rows(user_ids, item_ids, target_dates, watched_pct, catalog, context_provider,
                        row_caps=None):
    """Prepare all finite-known observed targets; never invent candidate labels.

    context_provider(users, cutoffs) must return numeric, categorical,
    history_max_date, history_length arrays. It is called with strict dates for fit
    and Aug2 for every select/calibration row, including tied dates. Its pinned
    implementation, raw inputs and fit statistics belong in catalog provenance.
    """
    users, items = checked_ids(user_ids, "user_ids"), checked_ids(item_ids, "item_ids")
    days = np.asarray(target_dates)
    if days.dtype != np.dtype("int64") or days.shape != users.shape or items.shape != users.shape or np.any(days <= 0):
        raise ValueError("Invalid observed row dates/lengths")
    pct = np.asarray(watched_pct, dtype=np.float64)
    if pct.shape != users.shape:
        raise ValueError("watched_pct shape mismatch")
    finite = np.isfinite(pct)
    if np.any((pct[finite] < 0) | (pct[finite] > 100)):
        raise ValueError("Finite watched_pct must be in [0,100]")
    known = np.isin(items, catalog.item_ids)
    eligible = known & finite & (days < WINDOW_STOP)
    caps = {"fit": 200000, "select": 40000, "calibration": 40000} if row_caps is None else dict(row_caps)
    if set(caps) != {"fit", "select", "calibration"}:
        raise ValueError("Declare fit/select/calibration row caps separately")
    select_users = cohort(users)
    masks = {"fit": eligible & (days < FIT_CUTOFF),
             "select": eligible & (days >= FIT_CUTOFF) & select_users,
             "calibration": eligible & (days >= FIT_CUTOFF) & ~select_users}
    role_indices = {}
    for role, mask in masks.items():
        indices = np.flatnonzero(mask)
        sampled = deterministic_sample_indices(users[indices], items[indices], days[indices], caps[role],
                                                preserve_items=role == "fit")
        role_indices[role] = indices[sampled]
    selected = np.sort(np.concatenate(list(role_indices.values())))
    cutoffs = np.minimum(days[selected], FIT_CUTOFF)
    context = context_provider(users[selected].copy(), cutoffs.copy())
    if set(context) != {"numeric", "categorical", "history_max_date", "history_length"}:
        raise ValueError("Context provider must return features and strict-past evidence only")
    all_rows = {"user_id": users[selected], "item_id": items[selected], "target_date": days[selected],
                "history_cutoff": cutoffs, "reward": (pct[selected] / 100).astype(np.float32),
                "observed": np.ones(len(selected), bool), **{k: np.asarray(v) for k, v in context.items()}}
    pre = all_rows["target_date"] < FIT_CUTOFF
    selected_users = cohort(all_rows["user_id"])
    final_masks = {"fit": pre, "select": ~pre & selected_users, "calibration": ~pre & ~selected_users}
    roles = {role: {k: v[mask] for k, v in all_rows.items()} for role, mask in final_masks.items()}
    for role, rows in roles.items():
        validate_rows(rows, role, catalog)
    excluded = {"input_observed_pairs": len(users), "missing_or_nonfinite_labels": int((~finite).sum()),
                "unknown_item_pairs": int((~known).sum()), "outside_rm_window": int((days >= WINDOW_STOP).sum()),
                "eligible_observed_pairs_before_sampling": int(eligible.sum()),
                "sampled_observed_pairs": len(selected), "missing_imputed_as_zero": False,
                "unobserved_pairs_generated": 0, "row_caps": caps,
                "role_rows_before_sampling": {k: int(v.sum()) for k, v in masks.items()},
                "sampling": "outcome-independent salted SHA256 row rank; all supported fit item IDs retained"}
    return roles, excluded


def save_bundle(output, roles, catalog, exclusions=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("RM bundle output must be new")
    for role in ("fit", "select", "calibration"):
        validate_rows(roles[role], role, catalog)
    if np.intersect1d(roles["select"]["user_id"], roles["calibration"]["user_id"]).size:
        raise ValueError("Selection and calibration users overlap")
    output.mkdir(parents=True)
    files = {}
    for role, rows in roles.items():
        path = output / f"{role}.npz"
        np.savez_compressed(path, **rows)
        files[path.name] = file_hash(path)
    path = output / "catalog.npz"
    np.savez_compressed(path, item_ids=catalog.item_ids, item_numeric=catalog.item_numeric,
                        item_categorical=catalog.item_categorical)
    files[path.name] = file_hash(path)
    manifest = {"contract": CONTRACT, "ready": True, "partition": PARTITION,
                "split": {"fit": "<2021-08-02", "select_calibration": "2021-08-02..2021-08-08",
                          "heldout_history": "frozen<2021-08-02"},
                "catalog": catalog.describe(), "files": files, "exclusions": exclusions or {},
                "builder_source_sha256": file_hash(__file__)}
    write_json(output / "manifest.json", manifest)
    return manifest


def _catalog(description, arrays):
    result = FeatureCatalog(arrays["item_ids"], arrays["item_numeric"], description["context_names"],
                            description["item_names"], description["context_categorical_names"],
                            arrays["item_categorical"], description["item_categorical_names"],
                            description["history_mask"], description["compatibility_pairs"], description["provenance"])
    if canonical_hash(result.describe()) != canonical_hash(description):
        raise ValueError("Catalog schema/mapping/array identity mismatch")
    return result


def _read_arrays(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Artifact checksum mismatch: {Path(path).name}")
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def load_bundle(path):
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT or manifest.get("ready") is not True or manifest.get("partition") != PARTITION:
        raise ValueError("Not a ready canonical RM bundle")
    expected_split = {"fit": "<2021-08-02", "select_calibration": "2021-08-02..2021-08-08",
                      "heldout_history": "frozen<2021-08-02"}
    if manifest.get("split") != expected_split:
        raise ValueError("Wrong RM split protocol")
    files = manifest["files"]
    layout = manifest.get("layout", "npz-v1")
    if layout == "npz-v1":
        required = {"catalog.npz", "fit.npz", "select.npz", "calibration.npz"}
    elif layout == "npy-directory-v1":
        fields = PAIR_ROW_FIELDS if manifest["catalog"]["provenance"].get("candidate_features_dynamic", False) else ROW_FIELDS
        required = {"catalog.npz", *(f"{role}.{key}.npy" for role in ("fit", "select", "calibration") for key in fields)}
    else:
        raise ValueError("Unsupported RM bundle layout")
    if set(files) not in (required, required | {"full_target_support.npz"}):
        raise ValueError("RM bundle file allowlist mismatch")
    catalog = _catalog(manifest["catalog"], _read_arrays(root / "catalog.npz", files["catalog.npz"]))
    if layout == "npz-v1":
        roles = {role: _read_arrays(root / f"{role}.npz", files[f"{role}.npz"]) for role in ("fit", "select", "calibration")}
    else:
        roles = {}
        for role in ("fit", "select", "calibration"):
            roles[role] = {}
            for key in fields:
                relative = f"{role}.{key}.npy"
                if file_hash(root / relative) != files[relative]:
                    raise ValueError(f"Artifact checksum mismatch: {relative}")
                roles[role][key] = np.load(root / relative, allow_pickle=False, mmap_mode="r")
    if "full_target_support.npz" in files:
        # Diagnostic only: never inserted into fit row features or learned priors.
        _read_arrays(root / "full_target_support.npz", files["full_target_support.npz"])
    sidecars = manifest.get("sidecars", {})
    allowed_sidecars = {"full_target_support.npz", *(f"{role}.{key}.npy" for role in ("fit", "select", "calibration") for key in ("x", "lengths"))}
    if not isinstance(sidecars, dict) or not set(sidecars).issubset(allowed_sidecars) or set(sidecars) & set(files):
        raise ValueError("RM bundle sidecar allowlist mismatch")
    for relative, expected in sidecars.items():
        if file_hash(root / relative) != expected:
            raise ValueError(f"Artifact checksum mismatch: {relative}")
    for role, rows in roles.items():
        validate_rows(rows, role, catalog, manifest["partition"])
    return roles, catalog, manifest


class FitPriors:
    """Observed-label priors fitted on RM fit only; no per-row feature leakage."""
    def __init__(self, rows, catalog, item_alpha=20., user_alpha=10.):
        y = rows["reward"].astype(np.float64)
        self.global_mean = float(y.mean())
        self.item_alpha, self.user_alpha = float(item_alpha), float(user_alpha)
        ids = catalog.indices(rows["item_id"])
        self.counts = np.bincount(ids, minlength=len(catalog.item_ids)).astype(np.int64)
        sums = np.bincount(ids, weights=y, minlength=len(catalog.item_ids))
        self.item_mean = (sums + item_alpha * self.global_mean) / (self.counts + item_alpha)
        self.user_ids, inverse = np.unique(rows["user_id"], return_inverse=True)
        self.user_count = np.bincount(inverse).astype(np.int64)
        totals = np.bincount(inverse, weights=y)
        self.user_mean = (totals + user_alpha * self.global_mean) / (self.user_count + user_alpha)
        self._users = {int(u): n for n, u in enumerate(self.user_ids)}

    def predict(self, rows, catalog, name="combined"):
        n = len(rows["item_id"])
        global_pred = np.full(n, self.global_mean, np.float64)
        item = self.item_mean[catalog.indices(rows["item_id"])]
        user = np.asarray([self.user_mean[self._users[int(u)]] if int(u) in self._users else self.global_mean for u in rows["user_id"]])
        if name == "global_mean":
            pred = global_pred
        elif name == "item_shrinkage":
            pred = item
        elif name == "user_mean":
            pred = user
        elif name == "combined":
            pred = (item + user) / 2
        else:
            raise ValueError("Unknown baseline")
        pred[self.counts[catalog.indices(rows["item_id"])] == 0] = self.global_mean
        return pred

    def describe(self):
        return {"global_mean": self.global_mean, "item_alpha": self.item_alpha, "user_alpha": self.user_alpha,
                "fit_only": True, "zero_support_fallback": "global_mean", "support_definition": "finite observed fit targets"}


def regression_metrics(y, pred):
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metric inputs must be nonempty finite paired vectors")
    error, variance = pred - y, float(np.sum((y - y.mean()) ** 2))
    mse = float(np.mean(error ** 2))
    bins = []
    bucket = np.minimum((np.clip(pred, 0, 1) * 10).astype(int), 9)
    for b in range(10):
        mask = bucket == b
        bins.append({"bin": b, "n": int(mask.sum()), "predicted": float(pred[mask].mean()) if mask.any() else None,
                     "observed": float(y[mask].mean()) if mask.any() else None})
    return {"n": len(y), "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.abs(error).mean()),
            "r2": 1 - float(np.sum(error ** 2)) / variance if variance > 0 else None,
            "mean_prediction": float(pred.mean()), "mean_observed": float(y.mean()),
            "calibration_bins": bins}


def segmented_metrics(rows, pred, catalog, priors):
    counts = priors.counts[catalog.indices(rows["item_id"])]
    h = rows["history_length"]
    groups = {"item_zero_support": counts == 0, "item_support_1_4": (counts > 0) & (counts < 5),
              "item_support_ge5": counts >= 5, "history_empty": h == 0,
              "history_1_5": (h > 0) & (h <= 5), "history_6_20": (h > 5) & (h <= 20),
              "history_gt20": h > 20, "user_absent_from_fit": ~np.isin(rows["user_id"], priors.user_ids),
              "context_numeric_missing": np.isnan(rows["numeric"]).any(axis=1)}
    return {name: regression_metrics(rows["reward"][mask], np.asarray(pred)[mask]) if mask.any() else {"n": 0}
            for name, mask in groups.items()}


def fit_calibration(y, pred, support=None):
    """Supported-only affine fit, gated against explicit clipped identity.

    Cold scores are immutable global fallbacks, so their labels must not adjust
    the affine map for supported candidates. The two fixed candidates are scored
    after clipping using only calibration labels; a numerical tie chooses identity.
    """
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    if (y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isfinite(y).all() or
            not np.isfinite(pred).all() or np.any((y < 0) | (y > 1))):
        raise ValueError("Calibration requires finite normalized paired labels/scores")
    mask = np.ones(len(y), bool) if support is None else np.asarray(support)
    if mask.dtype != np.dtype(bool) or mask.shape != y.shape:
        raise ValueError("Calibration support must be an aligned boolean mask")
    result = {"kind": "bounded_affine", "selection_protocol": "supported_affine_identity_gate",
              "fit_role": "calibration",
              "n": int(mask.sum()), "total_rows": len(y), "cold_rows_excluded": int((~mask).sum()),
              "slope_bounds": [0., 2.], "gate_metric": "supported calibration MSE after clipping",
              "identity": {"slope": 1., "intercept": 0.}, "slope": 1., "intercept": 0.,
              "selected": "identity", "tie_policy": "identity"}
    if not mask.any():
        result.update(affine_candidate=None, identity_mse=None, affine_mse=None,
                      selected_mse=None, reason="no_supported_calibration_rows")
        return result
    warm_y, warm_pred = y[mask], pred[mask]
    centered = warm_pred - warm_pred.mean()
    denom = float(centered @ centered)
    slope = float(np.clip((centered @ (warm_y - warm_y.mean())) / denom, 0., 2.)) if denom > 1e-12 else 0.
    intercept = float(warm_y.mean() - slope * warm_pred.mean())
    identity_mse = float(np.mean((np.clip(warm_pred, 0., 1.) - warm_y) ** 2))
    affine_mse = float(np.mean((np.clip(slope * warm_pred + intercept, 0., 1.) - warm_y) ** 2))
    result.update(affine_candidate={"slope": slope, "intercept": intercept},
                  identity_mse=identity_mse, affine_mse=affine_mse,
                  selected_mse=identity_mse, reason="identity_has_no_higher_supported_mse")
    if affine_mse < identity_mse:
        result.update(slope=slope, intercept=intercept, selected="affine", selected_mse=affine_mse,
                      reason="affine_has_lower_supported_mse")
    return result


class RewardModelScorer:
    """Frozen CPU scorer. Public candidate IDs are ORIGINAL IDs, never tokens."""
    def __init__(self, catalog, priors, model=None, config=None, calibration=None, signature=None):
        self.catalog, self.priors, self.model = catalog, priors, model
        self.config = dict(config or {"name": "combined", "history": True})
        self.calibration = dict(calibration or {"slope": 1., "intercept": 0.})
        self._signature = signature

    @property
    def signature(self):
        if self._signature is None:
            raise RuntimeError("Save and reload scorer before sharing it with policy/RL")
        return self._signature

    def _raw(self, rows):
        self.catalog.validate_context(rows)
        if self.model is None:
            pred = self.priors.predict(rows, self.catalog, self.config["name"])
        else:
            pool = catboost_pool(self.catalog, rows, history=self.config["history"])
            pred = np.asarray(self.model.predict(pool, thread_count=2), dtype=np.float64)
        # A never-target-trained item ID cannot get a confident CatBoost estimate.
        zero = self.priors.counts[self.catalog.indices(rows["item_id"])] == 0
        pred[zero] = self.priors.global_mean
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite RM prediction")
        return np.clip(pred, 0., 1.)

    def score_rows(self, rows, calibrated=True):
        pred = self._raw(rows)
        if calibrated:
            pred = np.clip(self.calibration["slope"] * pred + self.calibration["intercept"], 0., 1.)
        # Keep cold fallback interpretable and fit-only; calibration must not turn
        # unsupported actions into user/window-label-specific reward estimates.
        zero = self.priors.counts[self.catalog.indices(rows["item_id"])] == 0
        pred[zero] = self.priors.global_mean
        return pred

    def _context_rows(self, contexts, candidate_ids):
        ids = checked_ids(candidate_ids, "candidate_ids")
        contexts = list(contexts)
        if len(contexts) != len(ids):
            raise ValueError("One observed s0 context is required per candidate")
        if not len(ids):
            rows = {"user_id": np.empty(0, np.int64), "item_id": ids,
                    "numeric": np.empty((0, len(self.catalog.context_names)), np.float32),
                    "categorical": np.empty((0, len(self.catalog.context_categorical_names)), dtype="U1")}
            if self.catalog.provenance.get("candidate_features_dynamic", False):
                rows["candidate_numeric"] = np.empty((0, len(self.catalog.item_names)), np.float32)
            return rows
        pair_features = any("candidate_numeric" in c for c in contexts)
        for context, item_id in zip(contexts, ids):
            if not {"user_id", "numeric", "categorical"}.issubset(context) or any(k in context for k in FORBIDDEN):
                raise ValueError("Scoring context requires user_id/numeric/categorical and excludes outcomes")
            checked_ids(np.asarray([context["user_id"]]), "context user_id")
            if "schema_signature" in context and context["schema_signature"] != self.catalog.schema_signature:
                raise ValueError("Scoring context schema signature mismatch")
            if pair_features:
                if "candidate_numeric" not in context or "candidate_item_id" not in context:
                    raise ValueError("Per-pair candidate features need an explicit candidate_item_id binding")
                bound = checked_ids(np.asarray([context["candidate_item_id"]]), "candidate_item_id")
                if bound[0] != item_id:
                    raise ValueError("Per-pair candidate feature/item ID binding mismatch")
        rows = {"user_id": checked_ids([c["user_id"] for c in contexts], "user_id"), "item_id": ids,
                "numeric": np.asarray([c["numeric"] for c in contexts], np.float32),
                "categorical": np.asarray([c["categorical"] for c in contexts], dtype=str).reshape(len(ids), -1)}
        if pair_features:
            rows["candidate_numeric"] = np.asarray([c["candidate_numeric"] for c in contexts], np.float32)
        self.catalog.validate_context(rows)
        return rows

    def score_batch(self, contexts, candidate_ids):
        rows = self._context_rows(contexts, candidate_ids)
        return self.score_rows(rows) if len(rows["item_id"]) else np.empty(0, np.float64)

    def predict(self, user_history_features, candidate_ids):
        ids = checked_ids(candidate_ids, "candidate_ids")
        return self.score_batch([user_history_features] * len(ids), ids)

    def score_with_diagnostics(self, contexts, candidate_ids):
        rows = self._context_rows(contexts, candidate_ids)
        support = self.priors.counts[self.catalog.indices(rows["item_id"])]
        return {"rewards": self.score_rows(rows) if len(support) else np.empty(0, np.float64),
                "support": support > 0, "fallback": support == 0,
                "fit_target_support": support.copy(), "uncertainty": None}

    def save(self, output, report=None, model_snapshot=None):
        output = Path(output)
        if output.exists():
            raise FileExistsError("Scorer output must be a new directory")
        output.mkdir(parents=True)
        values = {"item_ids": self.catalog.item_ids, "item_numeric": self.catalog.item_numeric,
                  "item_categorical": self.catalog.item_categorical, "counts": self.priors.counts,
                  "item_mean": self.priors.item_mean, "user_ids": self.priors.user_ids,
                  "user_count": self.priors.user_count, "user_mean": self.priors.user_mean}
        np.savez_compressed(output / "scorer.npz", **values)
        files = {"scorer.npz": file_hash(output / "scorer.npz")}
        if model_snapshot is not None and (self.model is None or not isinstance(model_snapshot, bytes)):
            raise ValueError("A model snapshot requires an existing model and exact bytes")
        if self.model is not None:
            if model_snapshot is None:
                self.model.save_model(str(output / "model.cbm"))
            else:
                (output / "model.cbm").write_bytes(model_snapshot)
            files["model.cbm"] = file_hash(output / "model.cbm")
        manifest = {"contract": CONTRACT, "backend": "catboost_cpu" if self.model is not None else "declared_baseline",
                    "catalog": self.catalog.describe(), "priors": self.priors.describe(),
                    "config": self.config, "calibration": self.calibration, "files": files,
                    "source_sha256": file_hash(__file__), "score_id_space": "original_item_id",
                    "score_context": "fixed observed s0; synthetic prefix not appended", "report": report or {}}
        manifest["signature"] = canonical_hash(manifest)
        write_json(output / "manifest.json", manifest)
        self._signature = manifest["signature"]
        return manifest

    @classmethod
    def load(cls, path, expected_mapping_sha256=None):
        root = Path(path)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        signature = manifest.pop("signature", None)
        if (manifest.get("contract") != CONTRACT or canonical_hash(manifest) != signature or
                manifest.get("score_id_space") != "original_item_id"):
            raise ValueError("Frozen scorer manifest/signature mismatch")
        if manifest.get("source_sha256") != file_hash(__file__):
            raise ValueError("Scorer implementation changed after freeze")
        if expected_mapping_sha256 is not None and manifest["catalog"]["mapping_sha256"] != expected_mapping_sha256:
            raise ValueError("Policy/RM mapping identity mismatch")
        files = manifest["files"]
        allowed = {"scorer.npz", "model.cbm"} if manifest["backend"] == "catboost_cpu" else {"scorer.npz"}
        if set(files) != allowed or manifest["backend"] not in ("catboost_cpu", "declared_baseline"):
            raise ValueError("Scorer artifact allowlist/backend mismatch")
        values = _read_arrays(root / "scorer.npz", files["scorer.npz"])
        catalog = _catalog(manifest["catalog"], values)
        priors = FitPriors.__new__(FitPriors)
        for key in ("global_mean", "item_alpha", "user_alpha"):
            setattr(priors, key, manifest["priors"][key])
        for key in ("counts", "item_mean", "user_ids", "user_count", "user_mean"):
            setattr(priors, key, values[key])
        if (priors.counts.shape != catalog.item_ids.shape or priors.item_mean.shape != catalog.item_ids.shape or
                not np.isfinite(priors.item_mean).all() or np.any(priors.counts < 0) or
                priors.user_ids.shape != priors.user_mean.shape or priors.user_ids.shape != priors.user_count.shape or
                not np.isfinite(priors.user_mean).all()):
            raise ValueError("Invalid serialized fit priors")
        priors._users = {int(u): n for n, u in enumerate(priors.user_ids)}
        model = None
        if manifest["backend"] == "catboost_cpu":
            if file_hash(root / "model.cbm") != files["model.cbm"]:
                raise ValueError("CatBoost model checksum mismatch")
            try:
                from catboost import CatBoostRegressor
            except ImportError as error:
                raise RuntimeError("CatBoost backend requested but CatBoost is not installed; no automatic sklearn fallback") from error
            model = CatBoostRegressor(task_type="CPU", thread_count=2)
            model.load_model(str(root / "model.cbm"))
        return cls(catalog, priors, model, manifest["config"], manifest["calibration"], signature)


def catboost_pool(catalog, rows, history=True, labels=False):
    """Keep large numeric blocks float32; avoid a Python float object per cell."""
    try:
        from catboost import FeaturesData, Pool
    except ImportError as error:
        raise RuntimeError("CatBoost backend is unavailable; no automatic fallback") from error
    numeric, categorical = catalog.blocks(rows, history)
    cats = np.empty(categorical.shape, dtype=object)
    for col in range(categorical.shape[1]):
        cats[:, col] = [value.encode("utf-8") for value in categorical[:, col]]
    features = FeaturesData(num_feature_data=numeric, cat_feature_data=cats)
    return Pool(features, label=rows["reward"] if labels else None, thread_count=2)


def fit_reward(roles, catalog, backend="catboost", configs=None, seed=2026):
    """Bounded CPU fit. Selection labels select; calibration labels calibrate only."""
    for role in ("fit", "select", "calibration"):
        validate_rows(roles[role], role, catalog)
    priors = FitPriors(roles["fit"], catalog)
    report = {"contract": CONTRACT, "split": {"fit": "<2021-08-02", "heldout": "2021-08-02..2021-08-08"},
              "partition": PARTITION, "rows": {k: len(v["reward"]) for k, v in roles.items()},
              "mapping_sha256": catalog.mapping_sha256, "fit_only_priors": priors.describe(),
              "baselines": {}, "candidates": [], "cpu_threads": 2,
              "calibration_evaluation": "in-sample diagnostic on separate calibration users, not independent test",
              "limitations": ["observed aggregated outcomes, not impression response or causal feedback",
                              "static metadata has unverified historical availability",
                              "zero-support candidates use fit global mean with explicit support flag"]}
    for name in ("global_mean", "item_shrinkage", "user_mean", "combined"):
        report["baselines"][name] = {role: regression_metrics(rows["reward"], priors.predict(rows, catalog, name))
                                    for role, rows in roles.items() if role != "fit"}
    if backend == "baselines":
        # Explicit diagnostic mode; never silently stands in for boosted RM.
        name = min(report["baselines"], key=lambda k: report["baselines"][k]["select"]["mse"])
        scorer = RewardModelScorer(catalog, priors, config={"name": name, "history": True})
        report["backend"] = "declared_baseline"
    elif backend == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as error:
            raise RuntimeError("CatBoost is required for boosted RM; use --backend baselines explicitly for diagnostics") from error
        configs = list(DEFAULT_CONFIGS if configs is None else configs)
        if not 1 <= len(configs) <= 2:
            raise ValueError("At most two prespecified CPU configurations are permitted")
        best = None
        for config in configs:
            if (set(config) != {"name", "history", "depth", "iterations"} or not isinstance(config["history"], bool) or
                    not isinstance(config["iterations"], int) or not 1 <= config["iterations"] <= 400 or
                    not isinstance(config["depth"], int) or not 2 <= config["depth"] <= 6):
                raise ValueError("Unsupported/unbounded CatBoost configuration")
            train = catboost_pool(catalog, roles["fit"], history=config["history"], labels=True)
            select = catboost_pool(catalog, roles["select"], history=config["history"], labels=True)
            model = CatBoostRegressor(loss_function="RMSE", iterations=config["iterations"], depth=config["depth"],
                                      learning_rate=.05, l2_leaf_reg=10., random_seed=seed, task_type="CPU",
                                      thread_count=2, allow_writing_files=False, verbose=False,
                                      border_count=64, od_type="Iter", od_wait=30)
            model.fit(train, eval_set=select, use_best_model=True)
            current = RewardModelScorer(catalog, priors, model=model, config=config)
            pred = current.score_rows(roles["select"], calibrated=False)
            metric = regression_metrics(roles["select"]["reward"], pred)
            report["candidates"].append({"config": config, "trees": model.tree_count_, "select": metric})
            if best is None or metric["mse"] < best[0]:
                best = metric["mse"], current
        scorer = best[1]
        report["backend"] = "catboost_cpu"
    else:
        raise ValueError("Choose catboost or explicitly declared baselines; no automatic backend fallback")
    report["selected_config"] = scorer.config
    calibration_pred = scorer.score_rows(roles["calibration"], calibrated=False)
    support = priors.counts[catalog.indices(roles["calibration"]["item_id"])] > 0
    scorer.calibration = fit_calibration(roles["calibration"]["reward"], calibration_pred, support)
    report["calibration"] = scorer.calibration
    report["selected"] = {}
    for role in ("select", "calibration"):
        rows = roles[role]
        pred = scorer.score_rows(rows)
        report["selected"][role] = {"metrics": regression_metrics(rows["reward"], pred),
                                     "segments": segmented_metrics(rows, pred, catalog, priors)}
    return scorer, report


def _calibration_metrics(rows, pred, support):
    masks = {"all": np.ones(len(pred), bool), "supported": support, "cold": ~support}
    return {name: regression_metrics(rows["reward"][mask], pred[mask]) if mask.any() else {"n": 0}
            for name, mask in masks.items()}


def repair_calibration(old_artifact, legacy_source, bundle, output, expected_old_signature,
                       expected_mapping_sha256):
    """Upgrade calibration only, preserving old artifacts and exact model bytes.

    The old source is used only to verify/load its frozen scorer. New serving code
    and unchanged RM-data feature producers have separate explicit provenance.
    Selection labels are used only to report the fixed result, never to fit or gate
    the new calibration. CatBoost fitting is not called by this path.
    """
    old_artifact, legacy_source, bundle, output = map(lambda p: Path(p).resolve(),
                                                       (old_artifact, legacy_source, bundle, output))
    if output.exists():
        raise FileExistsError("Calibration repair output must be new")
    if any(output == p or p in output.parents for p in (old_artifact, bundle, legacy_source.parent)):
        raise ValueError("Calibration repair must not write inside immutable inputs")
    snapshots = {path: file_hash(path) for path in (old_artifact / "manifest.json", old_artifact / "model.cbm",
                  old_artifact / "scorer.npz", legacy_source, bundle / "manifest.json", Path(__file__).resolve())}
    old_manifest = json.loads((old_artifact / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("signature") != expected_old_signature:
        raise ValueError("Legacy scorer signature differs from admitted repair input")
    if old_manifest.get("backend") != "catboost_cpu" or snapshots[legacy_source] != old_manifest.get("source_sha256"):
        raise ValueError("Repair requires the exact frozen legacy CatBoost source/model")
    spec = importlib.util.spec_from_file_location("_kion_legacy_reward_" + snapshots[legacy_source][:12], legacy_source)
    legacy = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(legacy)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    old = legacy.RewardModelScorer.load(old_artifact, expected_mapping_sha256)
    roles, catalog, data_manifest = load_bundle(bundle)
    # Read-only memmaps still reflect an external writer. Bind every admitted
    # artifact, not only the JSON manifest, and recheck before/after publication.
    for relative, expected in {**data_manifest["files"], **data_manifest.get("sidecars", {})}.items():
        path = bundle / relative
        if file_hash(path) != expected:
            raise ValueError("Immutable repair dataset changed after validation: " + relative)
        snapshots[path] = expected
    if (old.catalog.schema_signature != catalog.schema_signature or
            old_manifest["report"].get("bundle_manifest_sha256") != snapshots[bundle / "manifest.json"]):
        raise ValueError("Repair data/schema is not the original immutable RM dataset")
    counts = np.bincount(catalog.indices(roles["fit"]["item_id"]), minlength=len(catalog.item_ids)).astype(np.int64)
    if not np.array_equal(counts, old.priors.counts):
        raise ValueError("Repair fit support differs from saved fit priors")
    model_snapshot = (old_artifact / "model.cbm").read_bytes()
    model_sha256 = hashlib.sha256(model_snapshot).hexdigest()
    if model_sha256 != old_manifest["files"]["model.cbm"]:
        raise ValueError("Legacy model changed before snapshot")
    scorer = RewardModelScorer(catalog, old.priors, old.model, old.config)
    rows = roles["calibration"]
    raw = scorer.score_rows(rows, calibrated=False)
    support = counts[catalog.indices(rows["item_id"])] > 0
    scorer.calibration = fit_calibration(rows["reward"], raw, support)
    calibrated = scorer.score_rows(rows)
    identity_metrics = _calibration_metrics(rows, raw, support)
    repaired_metrics = _calibration_metrics(rows, calibrated, support)
    if repaired_metrics["all"]["mse"] > identity_metrics["all"]["mse"]:
        # Explicit identity is also the final all-row guard. Cold predictions are
        # unchanged, but differing reduction orders can expose a rounding tie.
        scorer.calibration.update(slope=1., intercept=0., selected="identity",
                                  selected_mse=scorer.calibration["identity_mse"],
                                  reason="final_all_row_mse_identity_guard")
        calibrated = scorer.score_rows(rows)
        repaired_metrics = _calibration_metrics(rows, calibrated, support)
    if repaired_metrics["all"]["mse"] > identity_metrics["all"]["mse"]:
        raise RuntimeError("Calibration repair is worse than identity")
    if not np.array_equal(calibrated[~support], raw[~support]):
        raise RuntimeError("Calibration repair changed immutable cold fallback scores")
    report = json.loads(json.dumps(old_manifest["report"], allow_nan=False))
    report["calibration"] = scorer.calibration
    report["selected"] = {}
    for role in ("select", "calibration"):
        pred = scorer.score_rows(roles[role])
        report["selected"][role] = {"metrics": regression_metrics(roles[role]["reward"], pred),
                                     "segments": segmented_metrics(roles[role], pred, catalog, old.priors)}
    producer_sources = {"rm_data_boosting_sha256": catalog.provenance["source_pins"].get("boosting"),
                        "rm_data_builder_sha256": data_manifest["builder_source_sha256"]}
    repair = {"kind": "supported_calibration_only_repair", "legacy_signature": old.signature,
              "legacy_artifact": old_artifact.as_posix(),
              "legacy_manifest_sha256": snapshots[old_artifact / "manifest.json"],
              "legacy_source_sha256": snapshots[legacy_source], "serving_source_sha256": file_hash(__file__),
              "producer_sources": producer_sources, "model_cbm_sha256": model_sha256,
              "model_weights_copied_byte_for_byte": True, "catboost_fitted": False,
              "model_selection_changed": False, "selection_labels_used_for_repair": False,
              "calibration_support_only": True, "legacy_calibration": old.calibration,
              "identity_gate_metrics": identity_metrics, "repaired_metrics": repaired_metrics,
              "legacy_calibrated_metrics": _calibration_metrics(rows, old.score_rows(rows), support),
              "calibration_evaluation": "fit/gate diagnostics, not independent heldout performance"}
    report["calibration_repair"] = repair
    if any(file_hash(path) != value for path, value in snapshots.items()):
        raise ValueError("Immutable repair inputs/source changed during computation")
    manifest = scorer.save(output, report, model_snapshot=model_snapshot)
    loaded = RewardModelScorer.load(output, expected_mapping_sha256)
    if file_hash(output / "model.cbm") != model_sha256:
        raise ValueError("Repair failed to preserve exact CatBoost model bytes")
    np.testing.assert_array_equal(loaded.score_rows(rows), calibrated)
    if any(file_hash(path) != value for path, value in snapshots.items()):
        raise ValueError("Immutable repair inputs/source changed during publication")
    return loaded, manifest


def serve(scorer, input_stream, output_stream, max_rows=2048, max_line_bytes=16 * 1024 * 1024):
    """Bounded local JSONL protocol; one CPU scorer owns its process/model."""
    if not 1 <= max_rows <= 8192 or not 1024 <= max_line_bytes <= 64 * 1024 * 1024:
        raise ValueError("Invalid reward-server resource bounds")
    protocol = "kion-reward-stdio-v1"
    def emit(value):
        output_stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        output_stream.flush()
    emit({"protocol": protocol, "signature": scorer.signature,
          "schema_signature": scorer.catalog.schema_signature,
          "context_numeric_dim": len(scorer.catalog.context_names),
          "candidate_numeric_dim": len(scorer.catalog.item_names)})
    while True:
        payload = input_stream.readline(max_line_bytes + 1)
        if not payload:
            return
        if len(payload) > max_line_bytes:
            emit({"request_id": None, "error": "Request exceeds byte limit"})
            return
        request_id = None
        try:
            request = json.loads(payload)
            if not isinstance(request, dict):
                raise ValueError("Reward request must be an object")
            request_id = request.get("request_id")
            if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
                raise ValueError("request_id must be a nonnegative integer")
            if set(request) != {"protocol", "request_id", "contexts", "candidate_ids"} or request["protocol"] != protocol:
                raise ValueError("Unsupported reward request protocol/fields")
            if not isinstance(request["contexts"], list) or not isinstance(request["candidate_ids"], list):
                raise ValueError("contexts/candidate_ids must be lists")
            if len(request["contexts"]) > max_rows or len(request["candidate_ids"]) > max_rows:
                raise ValueError("Request exceeds row limit")
            result = scorer.score_with_diagnostics(request["contexts"], request["candidate_ids"])
            emit({"request_id": request_id, **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}})
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            emit({"request_id": request_id, "error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit", help="Explicit CPU fit on ready canonical RM bundle")
    fit_parser.add_argument("--bundle", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--backend", choices=("catboost", "baselines"), default="catboost")
    fit_parser.add_argument("--seed", type=int, default=2026)
    serve_parser = commands.add_parser("serve", help="Local bounded CPU scorer JSONL service for Torch RL")
    serve_parser.add_argument("--artifact", type=Path, required=True)
    serve_parser.add_argument("--expected-mapping-sha256", required=True)
    serve_parser.add_argument("--max-rows", type=int, default=2048)
    serve_parser.add_argument("--max-line-bytes", type=int, default=16 * 1024 * 1024)
    repair_parser = commands.add_parser("repair-calibration", help="Calibrate supported rows using frozen model bytes; no CatBoost fit")
    repair_parser.add_argument("--old-artifact", type=Path, required=True)
    repair_parser.add_argument("--legacy-source", type=Path, required=True)
    repair_parser.add_argument("--bundle", type=Path, required=True)
    repair_parser.add_argument("--output", type=Path, required=True)
    repair_parser.add_argument("--expected-old-signature", required=True)
    repair_parser.add_argument("--expected-mapping-sha256", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        scorer = RewardModelScorer.load(args.artifact, args.expected_mapping_sha256)
        serve(scorer, sys.stdin.buffer, sys.stdout, args.max_rows, args.max_line_bytes)
        return
    if args.command == "repair-calibration":
        scorer, manifest = repair_calibration(args.old_artifact, args.legacy_source, args.bundle, args.output,
                                              args.expected_old_signature, args.expected_mapping_sha256)
        print(json.dumps({"output": args.output.as_posix(), "signature": scorer.signature,
                          "calibration": manifest["calibration"],
                          "model_cbm_sha256": manifest["files"]["model.cbm"]}, indent=2))
        return
    roles, catalog, bundle = load_bundle(args.bundle)
    if args.output.exists():
        raise FileExistsError("Scorer output must be new")
    scorer, report = fit_reward(roles, catalog, args.backend, seed=args.seed)
    report["bundle_manifest_sha256"] = file_hash(args.bundle / "manifest.json")
    report["bundle_builder_source_sha256"] = bundle["builder_source_sha256"]
    manifest = scorer.save(args.output, report)
    frozen = RewardModelScorer.load(args.output, catalog.mapping_sha256)
    print(json.dumps({"output": args.output.resolve().as_posix(), "signature": frozen.signature,
                      "backend": manifest["backend"], "selected_config": report["selected_config"]}, indent=2))


if __name__ == "__main__":
    main()
