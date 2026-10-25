"""Build bounded observed-pair RM rows with strict-date raw H63 contexts.

The SFT vocabulary is a retrospective frozen action catalogue. Every feature
query is strictly before min(last_watch_dt, Aug 2); Aug 2--8 states are frozen.
No synthetic negative, current outcome or rolling SFT heldout context is used.
Large feature blocks are written as float32 NPY memmaps for the CPU RM loader.
"""
from __future__ import annotations

import argparse
from datetime import date
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


FIT_CUTOFF = date(2021, 8, 2).toordinal()
WINDOW_STOP = date(2021, 8, 9).toordinal()
MAX_ORDINAL = date.max.toordinal() + 1
ROLES = ("fit", "select", "calibration")
DEFAULT_CAPS = {"fit": 100000, "select": 20000, "calibration": 20000}
CONTEXT = 64


def _module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("_reward_data_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rm = _module("kion_reward_boosting")


def _ids(values, name):
    value = rm.checked_ids(values, name)
    if value.size and value.max() > np.iinfo(np.int64).max // MAX_ORDINAL - 1:
        raise ValueError(name + " exceeds exact user/day index capacity")
    return value


def checked_mapping(metadata):
    if metadata.get("special_tokens") != {"PAD": 0, "BOS": 1, "UNK": 2}:
        raise ValueError("Frozen policy special tokens differ")
    raw = metadata.get("item_id_to_token")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Missing frozen policy mapping")
    result = {}
    for key, value in raw.items():
        if (not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key
                or isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("Noncanonical frozen policy mapping")
        result[int(key)] = value
    _ids(np.asarray(list(result), np.int64), "mapping item IDs")
    if set(result.values()) != set(range(3, len(result) + 3)):
        raise ValueError("Frozen policy mapping must be contiguous and one-to-one")
    if metadata.get("vocab_size", len(result) + 3) != len(result) + 3:
        raise ValueError("Frozen policy vocabulary size mismatch")
    return result


class RawRows:
    """Compact immutable view over the common facade's checked raw index."""
    def __init__(self, users, items, days, outcomes, statuses, mapping):
        self.users, self.items = _ids(users, "raw users"), _ids(items, "raw items")
        self.days = np.asarray(days)
        self.outcomes, self.statuses = np.asarray(outcomes), np.asarray(statuses)
        n = len(self.users)
        if any(v.shape != (n,) for v in (self.items, self.days, self.outcomes, self.statuses)) or not n:
            raise ValueError("Raw arrays must be nonempty aligned vectors")
        if self.days.dtype.kind not in "iu" or np.any((self.days < 1) | (self.days >= MAX_ORDINAL)):
            raise ValueError("Invalid raw date")
        if not np.isin(self.statuses, (0, 1, 2)).all():
            raise ValueError("Invalid raw outcome status")
        valid = self.statuses == 0
        if not np.isfinite(self.outcomes[valid]).all() or np.any((self.outcomes[valid] < 0) | (self.outcomes[valid] > 1)):
            raise ValueError("Valid raw outcomes must be normalized in [0,1]")
        # Sorting is by date then item within each user; item order on a tied day
        # is only a deterministic storage convention, never a session sequence.
        order = np.lexsort((self.items, self.days, self.users))
        if not np.array_equal(order, np.arange(n)):
            raise ValueError("Raw index must be sorted by user/date/item")
        pair_order = np.lexsort((self.items, self.users))
        if n > 1 and np.any((self.users[pair_order[1:]] == self.users[pair_order[:-1]]) &
                            (self.items[pair_order[1:]] == self.items[pair_order[:-1]])):
            raise ValueError("Duplicate raw user/item pair")
        self.tokens = np.fromiter((mapping.get(int(i), 2) for i in self.items), np.int64, count=n)
        self.keys = self.users * MAX_ORDINAL + self.days
        self.unique_users, self.starts = np.unique(self.users, return_index=True)
        self.stops = np.r_[self.starts[1:], n]

    @classmethod
    def from_pack(cls, pack, mapping):
        index = pack.history
        if hasattr(index, "mapping") and {int(k): int(v) for k, v in index.mapping.items()} != mapping:
            raise ValueError("Raw history mapping differs from the policy catalogue")
        return cls(index.users, index.items, index.days, index.outcomes, index.statuses, mapping)

    def history_batch(self, users, cutoffs):
        users = _ids(users, "query users")
        cutoffs = np.asarray(cutoffs)
        if cutoffs.dtype.kind not in "iu" or cutoffs.shape != users.shape or np.any((cutoffs < 1) | (cutoffs >= MAX_ORDINAL)):
            raise ValueError("History cutoffs must be aligned ordinal dates")
        positions = np.searchsorted(self.unique_users, users)
        found = positions < len(self.unique_users)
        found &= self.unique_users[np.minimum(positions, len(self.unique_users) - 1)] == users
        ends = np.searchsorted(self.keys, users * MAX_ORDINAL + cutoffs, side="left")
        starts = ends.copy()
        starts[found] = self.starts[positions[found]]
        count = np.minimum(ends - starts, CONTEXT - 1)
        if np.any(count < 0):
            raise ValueError("Invalid raw history bounds")
        offsets = np.arange(CONTEXT - 1)[None, :]
        indices = ends[:, None] - count[:, None] + offsets
        mask = offsets < count[:, None]
        safe = np.minimum(indices, len(self.users) - 1)
        x = np.zeros((len(users), CONTEXT), np.int64)
        x[:, 0] = 1
        x[:, 1:] = np.where(mask, self.tokens[safe], 0)
        maxima = np.where(count > 0, self.days[np.maximum(ends - 1, 0)], 0).astype(np.int64)
        if np.any(maxima >= cutoffs):
            raise ValueError("History crosses strict date cutoff")
        return x, (count + 1).astype(np.int64), maxima


def select_rows(raw, row_caps=None):
    """Label-blind bottom-hash sampling; support uses the complete raw prefix."""
    caps = dict(DEFAULT_CAPS if row_caps is None else row_caps)
    if set(caps) != set(ROLES) or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in caps.values()):
        raise ValueError("Positive explicit fit/select/calibration caps are required")
    known, valid = raw.tokens >= 3, raw.statuses == 0
    eligible = known & valid & (raw.days < WINDOW_STOP)
    selection = rm.cohort(raw.users)
    masks = {"fit": eligible & (raw.days < FIT_CUTOFF),
             "select": eligible & (raw.days >= FIT_CUTOFF) & selection,
             "calibration": eligible & (raw.days >= FIT_CUTOFF) & ~selection}
    selected = {}
    for role, mask in masks.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError("No eligible observed rows for " + role)
        keep = rm.deterministic_sample_indices(raw.users[indices], raw.items[indices],
                                                raw.days[indices].astype(np.int64), caps[role],
                                                preserve_items=role == "fit")
        selected[role] = indices[keep]
    vocab = int(raw.tokens.max()) + 1
    full_counts = np.bincount(raw.tokens[masks["fit"]], minlength=vocab).astype(np.int64)
    sampled_counts = np.bincount(raw.tokens[selected["fit"]], minlength=vocab).astype(np.int64)
    if not np.array_equal(full_counts > 0, sampled_counts > 0):
        raise ValueError("Fit sampling changed the supported action set")
    facts = {"input_observed_pairs": len(raw.users), "unobserved_pairs_generated": 0,
             "missing_imputed_as_zero": False, "missing_labels": int(np.count_nonzero(raw.statuses == 1)),
             "invalid_labels": int(np.count_nonzero(raw.statuses == 2)),
             "unknown_item_pairs": int(np.count_nonzero(~known)),
             "outside_rm_window": int(np.count_nonzero(raw.days >= WINDOW_STOP)),
             "role_rows_before_sampling": {k: int(v.sum()) for k, v in masks.items()},
             "role_rows": {k: len(v) for k, v in selected.items()}, "row_caps": caps,
             "sampling": "outcome-independent salted SHA256 row rank; one row per supported fit item retained",
             "fit_population": "all valid observed raw pairs, including BOS-only contexts; differs from old SFT prior-history eligibility"}
    return selected, full_counts, sampled_counts, facts


def _pin_sources(pack, paths):
    pins = {str(key): rm.file_hash(path) for key, path in paths.items()}
    def visit(value, prefix):
        if isinstance(value, dict):
            if isinstance(value.get("sha256"), str):
                pins[prefix] = value["sha256"]
            else:
                for key, child in value.items():
                    visit(child, prefix + "/" + str(key))
    visit(pack.provenance, "feature_pack")
    return pins


def _declared_sources(provenance):
    """Gather facade source descriptors without trusting copied hash claims."""
    sources = {}
    def visit(value):
        if not isinstance(value, dict):
            return
        if isinstance(value.get("sha256"), str) and isinstance(value.get("path"), str):
            path = Path(value["path"]).resolve()
            digest = value["sha256"]
            if path in sources and sources[path] != digest:
                raise ValueError("Conflicting facade source pins: " + str(path))
            sources[path] = digest
        for child in value.values():
            visit(child)
    visit(provenance)
    return sources


def _check_sources(sources):
    for path, expected in sources.items():
        if not path.is_file() or rm.file_hash(path) != expected:
            raise ValueError("RM source changed during data build: " + str(path))


def _feature_batches(pack, raw, indices, batch_size):
    """Reuse a date bank while yielding original output row positions."""
    origins = np.minimum(raw.days[indices], FIT_CUTOFF).astype(np.int64)
    order = np.argsort(origins, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(origins[order])) + 1, len(order)]
    dynamic = getattr(pack, "dynamic", None)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        origin = int(origins[order[left]])
        bank = dynamic.day_bank(origin) if dynamic is not None else None
        for begin in range(int(left), int(right), batch_size):
            positions = order[begin:min(int(right), begin + batch_size)]
            yield positions, indices[positions], origins[positions], bank
        del bank


def build_reward_data(raw_dir, metadata_path, basic_npz, output, *, row_caps=None,
                      batch_size=512, pack=None, progress=None):
    """Write one new checksummed mmap bundle; fitting remains a separate action."""
    raw_dir, metadata_path, basic_npz, output = map(Path, (raw_dir, metadata_path, basic_npz, output))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 8192:
        raise ValueError("batch_size must be an integer in [1,8192]")
    if output.exists():
        raise FileExistsError("RM output must be a new directory")
    paths = {"metadata": metadata_path, "basic_npz": basic_npz, "builder": Path(__file__),
             "boosting": Path(__file__).with_name("kion_reward_boosting.py")}
    paths.update({name: raw_dir / (name + ".csv") for name in ("interactions", "users", "items")})
    sibling_manifest = basic_npz.with_name("manifest.json")
    if sibling_manifest.is_file():
        paths["basic_manifest"] = sibling_manifest
    for name in ("kion_local_pack", "kion_pack_static", "kion_pack_people", "kion_pack_semantics",
                 "kion_pack_history", "kion_pack_dynamic", "kion_local_features",
                 "kion_basic_bundle_features", "kion_sex_features"):
        path = Path(__file__).with_name(name + ".py")
        paths["component/" + name] = path
    before = {key: rm.file_hash(path) for key, path in paths.items()}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = checked_mapping(metadata)
    if pack is None:
        pack = _module("kion_local_pack").FeaturePack.build(raw_dir, metadata_path, basic_npz,
                                                           fit_cutoff="2021-08-02", include_relative=True)
    if pack.fit_cutoff != "2021-08-02":
        raise ValueError("RM facade must be fit strictly before Aug2")
    declared_sources = _declared_sources(pack.provenance)
    _check_sources(declared_sources)
    raw = RawRows.from_pack(pack, mapping)
    selected, full_counts, sampled_counts, exclusions = select_rows(raw, row_caps)
    item_ids = np.asarray(sorted(mapping, key=mapping.get), np.int64)
    provenance = {"fit_cutoff_exclusive": "2021-08-02", "metadata_availability": "static_snapshot_unverified",
                  "source_pins": _pin_sources(pack, paths), "candidate_features_dynamic": True,
                  "static_candidate_fallback": False, "facade": pack.provenance,
                  "query_origin": "min(target_date,2021-08-02); heldout target day is never a feature",
                  "history": "BOS plus last63 raw rows strictly before origin; includes missing-label and OOV history"}
    # Row candidate vectors always override this catalogue template. Static
    # metadata occupies its correct columns; temporal columns are zero here.
    template = np.zeros((len(item_ids), len(pack.candidate_columns)), np.float32)
    static_values = getattr(pack, "item_features", None)
    if static_values is not None:
        template[:, :len(pack.item_columns)] = static_values[3:]
    history_mask = [not str(c).startswith("user.") for c in pack.user_columns]
    catalog = rm.FeatureCatalog(item_ids, template, pack.user_columns, pack.candidate_columns,
                                history_mask=history_mask, provenance=provenance)
    if catalog.mapping != {str(k): v for k, v in mapping.items()}:
        raise ValueError("RM and policy action mapping differ")
    output.mkdir(parents=True)
    started, files, sidecars, bindings = time.perf_counter(), {}, {}, {}
    for role, indices in selected.items():
        n = len(indices)
        columns = {"numeric": (np.float32, (n, len(pack.user_columns))),
                   "candidate_numeric": (np.float32, (n, len(pack.candidate_columns))),
                   "categorical": ("U1", (n, 0)), "observed": (bool, (n,)), "reward": (np.float32, (n,))}
        for key in ("user_id", "item_id", "target_date", "history_cutoff", "history_max_date", "history_length"):
            columns[key] = (np.int64, (n,))
        rows = {key: np.lib.format.open_memmap(output / f"{role}.{key}.npy", mode="w+", dtype=dtype, shape=shape)
                for key, (dtype, shape) in columns.items()}
        xs = np.lib.format.open_memmap(output / f"{role}.x.npy", mode="w+", dtype=np.int64, shape=(n, CONTEXT))
        lengths = np.lib.format.open_memmap(output / f"{role}.lengths.npy", mode="w+", dtype=np.int64, shape=(n,))
        written = 0
        for positions, idx, origins, bank in _feature_batches(pack, raw, indices, batch_size):
            batch_count = len(positions)
            users = raw.users[idx].astype(np.int64, copy=False)
            x, lens, maxima = raw.history_batch(users, origins)
            # Only s0 and candidate identity enter the facade. Target rewards and
            # target-day timestamps have no route to either feature call.
            bank_argument = {} if bank is None else {"dynamic_bank": bank}
            context = pack.user_features(users, origins, x, lens, history_cutoffs=origins, **bank_argument)
            candidates = pack.candidate_features(raw.tokens[idx], origins, history_cutoffs=origins, user_ids=users)
            if (np.asarray(context).shape != (batch_count, len(pack.user_columns)) or
                    np.asarray(candidates).shape != (batch_count, len(pack.candidate_columns)) or
                    not np.isfinite(context).all() or not np.isfinite(candidates).all()):
                raise ValueError("Facade returned invalid feature blocks")
            sl = positions
            rows["numeric"][sl], rows["candidate_numeric"][sl] = context, candidates
            rows["user_id"][sl], rows["item_id"][sl] = users, raw.items[idx]
            rows["target_date"][sl], rows["history_cutoff"][sl] = raw.days[idx], origins
            rows["history_max_date"][sl], rows["history_length"][sl] = maxima, lens - 1
            rows["reward"][sl], rows["observed"][sl] = raw.outcomes[idx], True
            xs[sl], lengths[sl] = x, lens
            written += batch_count
            if progress is not None:
                progress({"role": role, "rows": written, "total": n, "origin": int(origins[0]),
                          "seconds": round(time.perf_counter() - started, 3)})
        for value in (*rows.values(), xs, lengths):
            value.flush()
        rm.validate_rows(rows, role, catalog)
        bindings[role] = rm.canonical_hash({key: rm.array_hash(rows[key]) for key in
                                            ("user_id", "item_id", "target_date", "history_cutoff", "reward")})
        for key in rows:
            name = f"{role}.{key}.npy"
            files[name] = rm.file_hash(output / name)
        for key in ("x", "lengths"):
            name = f"{role}.{key}.npy"
            sidecars[name] = rm.file_hash(output / name)
        del rows, xs, lengths
    np.savez_compressed(output / "catalog.npz", item_ids=catalog.item_ids,
                        item_numeric=catalog.item_numeric, item_categorical=catalog.item_categorical)
    files["catalog.npz"] = rm.file_hash(output / "catalog.npz")
    count = len(item_ids) + 3
    full_counts = np.pad(full_counts, (0, max(0, count - len(full_counts))))[:count]
    sampled_counts = np.pad(sampled_counts, (0, max(0, count - len(sampled_counts))))[:count]
    np.savez_compressed(output / "full_target_support.npz", item_id=item_ids, token=np.arange(3, count, dtype=np.int64),
                        full_raw_fit_count=full_counts[3:], sampled_fit_count=sampled_counts[3:])
    sidecars["full_target_support.npz"] = rm.file_hash(output / "full_target_support.npz")
    exclusions["target_support"] = {"full_raw_fit_rows": int(full_counts.sum()),
                                     "sampled_fit_rows": int(sampled_counts.sum()),
                                     "zero_support_items": int(np.count_nonzero(full_counts[3:] == 0)),
                                     "zero_support_set_preserved": bool(np.array_equal(full_counts > 0, sampled_counts > 0)),
                                     "prior_fit_population": "sampled RM fit; full raw support retained as a separate diagnostic"}
    if any(rm.file_hash(path) != before[key] for key, path in paths.items()):
        raise ValueError("RM source changed during data build")
    _check_sources(declared_sources)
    manifest = {"contract": rm.CONTRACT, "layout": "npy-directory-v1", "ready": True,
                "partition": rm.PARTITION, "split": {"fit": "<2021-08-02", "select_calibration": "2021-08-02..2021-08-08",
                "heldout_history": "frozen<2021-08-02"}, "catalog": catalog.describe(), "files": files,
                "sidecars": sidecars, "row_bindings": bindings, "exclusions": exclusions,
                "builder_source_sha256": before["builder"], "feature_dtype": "float32",
                "seconds": time.perf_counter() - started,
                "limitations": ["conditional observed aggregated watch outcome; not counterfactual reward",
                                "metadata is a retrospective snapshot", "past aggregate outcome availability assumes last_watch_dt"]}
    rm.write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--basic-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fit", type=int, default=DEFAULT_CAPS["fit"])
    parser.add_argument("--max-select", type=int, default=DEFAULT_CAPS["select"])
    parser.add_argument("--max-calibration", type=int, default=DEFAULT_CAPS["calibration"])
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)
    manifest = build_reward_data(args.raw, args.metadata, args.basic_npz, args.output,
                                row_caps={"fit": args.max_fit, "select": args.max_select, "calibration": args.max_calibration},
                                batch_size=args.batch_size, progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"output": args.output.resolve().as_posix(), "rows": manifest["exclusions"]["role_rows"],
                      "support": manifest["exclusions"]["target_support"]}, indent=2))


if __name__ == "__main__":
    main()
