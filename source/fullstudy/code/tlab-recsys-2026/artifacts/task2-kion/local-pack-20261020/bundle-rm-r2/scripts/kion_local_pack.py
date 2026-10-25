"""Common local KION feature facade and bounded, row-bound SFT cache builder.

Metadata is a retrospective snapshot. Dynamic outcomes are available only under
the declared last-watch-date assumption and always strictly before the origin.
This module never opens the old test split or adds candidate outcomes to inputs.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_local_features as util
import kion_pack_static as static
import kion_pack_people as people
import kion_pack_history as history

CONTRACT = "kion-all-local-pack-v1"
ROW_KEYS = ("x", "lengths", "y", "user_id", "target_item_id", "target_date")
ROW_ALGORITHM = "canonical-array-hashes:x,lengths,y,user_id,target_item_id,target_date:v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def row_keys_hash(data):
    return util.canonical_hash({key: util.array_hash(data[key]) for key in ROW_KEYS})


def item_transform(values, preserve=197):
    """Keep original197 exact; signed-log new nonbinary catalog coordinates.

    The mask is fit on the fixed train-known catalog. No response labels, query
    rows or validation distributions choose the transform. Specials stay as-is.
    """
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] < preserve or not np.isfinite(result).all():
        raise ValueError("Invalid combined item feature matrix")
    mask = np.zeros(result.shape[1], bool)
    block = result[3:, preserve:]
    mask[preserve:] = np.any((block != 0) & (block != 1), axis=0)
    result[:, mask] = np.sign(result[:, mask]) * np.log1p(np.abs(result[:, mask]))
    return result, mask


class FeaturePack:
    @classmethod
    def build(cls, raw_dir, metadata_path, basic_npz, *, fit_cutoff="2021-08-09", include_relative=True):
        # Required modules fail explicitly while their implementation/review is pending.
        import kion_pack_dynamic as dynamic
        import kion_pack_semantics as semantics
        raw_dir, metadata_path, basic_npz = map(Path, (raw_dir, metadata_path, basic_npz))
        source_paths = [raw_dir / name for name in ("items.csv", "users.csv", "interactions.csv")]
        source_paths += [metadata_path, basic_npz, basic_npz.with_name("manifest.json")]
        source_paths += [Path(__file__)]
        source_paths += [Path(module.__file__) for module in (util, static, people, history, dynamic, semantics)]
        source_paths += [Path(sys.modules[name].__file__) for name in ("kion_basic_bundle_features", "kion_sex_features")]
        before = {path.resolve().as_posix(): util.file_hash(path) for path in source_paths}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        basic_manifest = json.loads(basic_npz.with_name("manifest.json").read_text(encoding="utf-8"))
        base = static.build(raw_dir, metadata_path, basic_npz)
        persons = people.build(raw_dir, metadata)
        semantic = semantics.build(raw_dir, metadata)
        with np.load(basic_npz, allow_pickle=False) as source:
            original = source["item_features"]
        original_columns = [""] * original.shape[1]
        for name, group in basic_manifest["groups"].items():
            if group["scope"] == "item":
                original_columns[group["start"]:group["stop"]] = ["basic." + name + "." + str(c) for c in group["columns"]]
        if not all(original_columns):
            raise ValueError("Original item schema is incomplete")
        matrices = [original, base["item_features"], persons["item_features"], semantic["item_features"]]
        columns = [original_columns, ["static." + c for c in base["item_columns"]],
                   ["people." + c for c in persons["columns"]], ["semantics." + c for c in semantic["columns"]]]
        obj = cls()
        obj.item_columns = sum(columns, [])
        if len(set(obj.item_columns)) != len(obj.item_columns):
            raise ValueError("Duplicate combined item column")
        obj.item_features, transform = item_transform(np.concatenate(matrices, axis=1))
        obj.user_lookup = base["user_lookup"]
        obj.history = history.build_history_index(raw_dir, metadata, fit_cutoff=fit_cutoff, include_relative=include_relative,
                                                 schema_item_ids=metadata["item_id_to_token"].keys())
        obj.dynamic = dynamic.build_index(raw_dir, metadata, fit_cutoff=fit_cutoff)
        obj.user_columns = (["user." + c for c in base["user_columns"]]
                            + ["history." + c for c in obj.history.columns]
                            + ["dynamic_history." + c for c in obj.dynamic.history_columns])
        obj.candidate_columns = obj.item_columns + ["candidate_dynamic." + c for c in obj.dynamic.columns]
        obj.metadata = metadata
        obj.fit_cutoff = fit_cutoff
        obj.provenance = {"contract": CONTRACT, "fit_cutoff": fit_cutoff,
                          "metadata_schema_fit": "fixed_SFT_train_known_catalog_shared_across_roles; unsupervised_retrospective_metadata",
                          "label_dependent_fit": "only_strict_past_before_role_fit_cutoff",
                          "metadata_availability": "retrospective_catalog_snapshot_not_historical_asof",
                          "outcome_availability": "strict_last_watch_date_assumption_for_aggregated_pairs",
                          "source_pins": {p: {"path": p, "sha256": h} for p, h in before.items()},
                          "item_transform": {"name": "signed_log1p_new_nonbinary_train_catalog_columns", "preserved_prefix": 197,
                                             "column_indices": np.flatnonzero(transform).tolist()},
                          "components": {"static": base["provenance"], "people": persons["provenance"],
                                         "semantics": semantic["provenance"], "history": obj.history.provenance,
                                         "dynamic": obj.dynamic.provenance},
                          "coverage": {"people": persons["coverage"], "semantics": semantic["coverage"]}}
        for path, digest in before.items():
            if util.file_hash(path) != digest:
                raise ValueError("Input changed during feature build: " + path)
        obj.schema_hash = util.canonical_hash({"item": obj.item_columns, "user": obj.user_columns, "candidate": obj.candidate_columns})
        return obj

    def user_features(self, user_ids, query_dates, x, lengths, history_cutoffs=None, dynamic_bank=None):
        dynamic = (self.dynamic.history_features(x, lengths, query_dates, history_cutoffs=history_cutoffs)
                   if dynamic_bank is None else self.dynamic.history_features_from_bank(x, lengths, dynamic_bank))
        blocks = [self.user_lookup(user_ids), self.history.features(user_ids, query_dates, x=x, lengths=lengths,
                  history_cutoffs=history_cutoffs), dynamic]
        result = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        if result.shape != (len(user_ids), len(self.user_columns)) or not np.isfinite(result).all():
            raise ValueError("Invalid user feature batch")
        return result

    def candidate_features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        tokens = np.asarray(item_tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= len(self.item_features)):
            raise ValueError("Invalid candidate token array")
        dynamic = self.dynamic.features(tokens, query_dates, history_cutoffs=history_cutoffs, user_ids=user_ids)
        result = np.concatenate((self.item_features[tokens], dynamic), axis=1).astype(np.float32, copy=False)
        if result.shape != (len(tokens), len(self.candidate_columns)) or not np.isfinite(result).all():
            raise ValueError("Invalid candidate feature batch")
        return result


def user_rows_deduplicated(pack, user_ids, origins, x, lengths, dynamic_bank=None):
    """Targets on one user/date share an identical strictly-past state.

    Check rather than assume the repeated state, then compute once per batch.
    Target IDs/labels do not participate in this safe feature-only reuse.
    """
    pairs = np.column_stack((user_ids, origins))
    _, first, inverse = np.unique(pairs, axis=0, return_index=True, return_inverse=True)
    if not np.array_equal(lengths, lengths[first][inverse]) or not np.array_equal(x, x[first][inverse]):
        raise ValueError("Repeated user/origin rows disagree on observed history")
    kwargs = {} if dynamic_bank is None else {"dynamic_bank": dynamic_bank}
    return pack.user_features(user_ids[first], origins[first], x[first], lengths[first], **kwargs)[inverse]


def _output(path, array):
    return {"path": Path(path).resolve().as_posix(), "sha256": util.file_hash(path), "shape": list(array.shape), "dtype": str(array.dtype)}


def build_cache(raw_dir, data_dir, basic_npz, output, *, batch_size=1024, progress_every=50000):
    data_dir, output = Path(data_dir), Path(output)
    if batch_size < 1 or progress_every < 1 or output.exists():
        raise ValueError("Positive batch size and a new output directory are required")
    output.mkdir(parents=True)
    started = time.perf_counter()
    metadata_path = data_dir / "metadata.json"
    inputs = {name: {"path": (data_dir / name).resolve().as_posix(), "sha256": util.file_hash(data_dir / name)}
              for name in ("metadata.json", "train.npz", "val.npz")}
    pack = FeaturePack.build(raw_dir, metadata_path, basic_npz)
    outputs, keys, timings, row_counts = {}, {}, {}, {}
    item_path = output / "item_features.npy"
    np.save(item_path, pack.item_features, allow_pickle=False)
    outputs[item_path.name] = _output(item_path, pack.item_features)
    for name, array in (("user_lookup_ids.npy", pack.user_lookup.user_ids), ("user_lookup_codes.npy", pack.user_lookup.codes)):
        np.save(output / name, array, allow_pickle=False)
        outputs[name] = _output(output / name, array)
    for split in ("train", "val"):
        clock = time.perf_counter()
        with np.load(data_dir / (split + ".npz"), allow_pickle=False) as source:
            data = {key: source[key] for key in ROW_KEYS}
        keys[split] = row_keys_hash(data)
        count = len(data["x"])
        row_counts[split] = count
        path = output / (split + "_user_features.npy")
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(count, len(pack.user_columns)))
        # One small dynamic item bank per date; write back into original row order.
        origins = data["target_date"] if split == "train" else np.full(count, date(2021, 8, 9).toordinal(), np.int64)
        order = np.argsort(origins, kind="stable")
        sorted_origins = origins[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_origins)) + 1, count]
        written = 0
        try:
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                origin = int(sorted_origins[left])
                bank = pack.dynamic.day_bank(origin)
                for begin in range(int(left), int(right), batch_size):
                    indices = order[begin:min(int(right), begin + batch_size)]
                    batch_origins = origins[indices]
                    block = user_rows_deduplicated(pack, data["user_id"][indices], batch_origins,
                                                  data["x"][indices], data["lengths"][indices], dynamic_bank=bank)
                    if np.any(np.abs(block) > np.finfo(np.float16).max):
                        raise ValueError("Float16 cache would overflow")
                    values[indices] = block
                    previous, written = written, written + len(indices)
                    if previous == 0 or written == count or previous // progress_every != written // progress_every:
                        values.flush()
                        write_json(output / "progress.json", {"state": "building", "split": split, "rows": written,
                                   "total_rows": count, "origin": origin, "elapsed_seconds": time.perf_counter() - started,
                                   "updated_at": datetime.now(timezone.utc).isoformat()})
                        print(json.dumps({"split": split, "rows": written, "total": count, "seconds": round(time.perf_counter() - clock, 2)}), flush=True)
                del bank
            values.flush()
            outputs[path.name] = _output(path, values)
        finally:
            values._mmap.close()
        del values, data, origins, order, sorted_origins
        gc.collect()
        timings[split] = time.perf_counter() - clock
        if util.file_hash(data_dir / (split + ".npz")) != inputs[split + ".npz"]["sha256"]:
            raise ValueError("Source split changed during cache build")
    feature_columns = {"item": pack.item_columns, "user": pack.user_columns}
    manifest = {"contract": CONTRACT, "state": "completed", "feature_columns": feature_columns,
                "schema_hash": util.canonical_hash(feature_columns), "candidate_schema_hash": pack.schema_hash,
                "inputs": inputs, "outputs": outputs, "row_keys_sha256": keys, "rowkeys_algorithm": ROW_ALGORITHM,
                "provenance": pack.provenance, "timings_seconds": timings, "total_seconds": time.perf_counter() - started,
                "validation_origin": "2021-08-09", "cache_dtype": "float16", "batch_transfer_dtype": "float32",
                "training_rows": row_counts["train"], "validation_rows": row_counts["val"],
                "rounding": "history/user features float32->float16 once; original item197 remains float32 exact"}
    write_json(output / "manifest.json", manifest)
    write_json(output / "progress.json", {"state": "completed", "elapsed_seconds": manifest["total_seconds"]})
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--basic-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args(argv)
    manifest = build_cache(args.raw, args.data, args.basic_npz, args.output, batch_size=args.batch_size)
    print(json.dumps({"state": "completed", "schema_hash": manifest["schema_hash"], "seconds": manifest["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
