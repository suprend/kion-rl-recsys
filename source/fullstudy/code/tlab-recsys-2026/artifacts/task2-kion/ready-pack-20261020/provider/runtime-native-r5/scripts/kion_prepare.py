"""Prepare a capped, strictly temporal KION sequence experiment (stdlib + NumPy).

KION rows are user/item last-watch dates with aggregated outcomes, not sessions.
Ordering items within a date is a deterministic storage convention only. Every
target sees strictly earlier dates; evaluation histories are frozen at cutoff.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import csv
from datetime import date
import json
from pathlib import Path

import numpy as np

VAL_START = date(2021, 8, 9).toordinal()
TEST_START = date(2021, 8, 16).toordinal()
TEST_END = date(2021, 8, 23).toordinal()
DEMOGRAPHICS = ("age", "income", "sex", "kids_flg")
UNKNOWN = "__UNKNOWN__"


def read_table(path: Path, key: str):
    rows = {}
    duplicates = 0
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        schema = reader.fieldnames
        if not schema or key not in schema:
            raise ValueError(f"Missing {key} in {path}")
        for row in reader:
            count += 1
            ident = int(row[key])
            duplicates += ident in rows
            rows[ident] = row
    if duplicates:
        raise ValueError(f"{path}: {duplicates} duplicate {key} values")
    return rows, {"rows": count, "duplicate_ids": duplicates, "columns": schema}


def load_interactions(path: Path):
    users, items, days, rewards = array("q"), array("q"), array("i"), array("f")
    day_cache = {}
    missing = invalid = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        schema = reader.fieldnames
        required = {"user_id", "item_id", "last_watch_dt", "watched_pct"}
        if not schema or not required.issubset(schema):
            raise ValueError(f"Missing interaction columns: {required - set(schema or [])}")
        for row in reader:
            users.append(int(row["user_id"]))
            items.append(int(row["item_id"]))
            raw_day = row["last_watch_dt"]
            if raw_day not in day_cache:
                day_cache[raw_day] = date.fromisoformat(raw_day).toordinal()
            days.append(day_cache[raw_day])
            raw = row["watched_pct"].strip()
            if raw.lower() in ("", "nan", "none", "null"):
                reward = float("nan")
                missing += 1
            else:
                try:
                    pct = float(raw)
                except ValueError:
                    pct = float("nan")
                if not np.isfinite(pct) or not 0 <= pct <= 100:
                    invalid += 1
                    reward = float("nan")
                else:
                    reward = pct / 100
            rewards.append(reward)
    if not users:
        raise ValueError("Empty interactions")
    u = np.frombuffer(users, dtype=np.int64).copy()
    i = np.frombuffer(items, dtype=np.int64).copy()
    d = np.frombuffer(days, dtype=np.int32).copy()
    r = np.frombuffer(rewards, dtype=np.float32).copy()
    order = np.lexsort((i, d, u))
    return (u[order], i[order], d[order], r[order]), schema, missing, invalid


def categories(values):
    return [UNKNOWN] + sorted({v for v in values if v and v != UNKNOWN})


def metadata_features(users, items, train_users, item_ids):
    user_categories = {
        field: categories(users.get(int(uid), {}).get(field, "").strip() for uid in train_users)
        for field in DEMOGRAPHICS
    }
    user_maps = {}
    offset = 0
    for field, values in user_categories.items():
        user_maps[field] = {value: offset + n for n, value in enumerate(values)}
        offset += len(values)
    item_categories = {
        "content_type": categories(items.get(int(iid), {}).get("content_type", "").strip() for iid in item_ids),
        "genres": categories(g.strip() for iid in item_ids for g in items.get(int(iid), {}).get("genres", "").split(",")),
    }
    item_maps = {}
    item_dim = 0
    for field, values in item_categories.items():
        item_maps[field] = {value: item_dim + n for n, value in enumerate(values)}
        item_dim += len(values)
    features = np.zeros((len(item_ids) + 3, item_dim), np.float32)
    for token, iid in enumerate(item_ids, 3):
        row = items.get(int(iid), {})
        for field in item_categories:
            values = [x.strip() for x in row.get(field, "").split(",") if x.strip()] or [UNKNOWN]
            for value in values:
                features[token, item_maps[field].get(value, item_maps[field][UNKNOWN])] = 1
    # PAD and BOS carry no content; UNK represents missing item metadata.
    for field in item_maps:
        features[2, item_maps[field][UNKNOWN]] = 1
    return user_categories, user_maps, offset, item_categories, features


def prepare(input_dir: Path, output_dir: Path, max_train=150000, max_eval=3000, context=32, seed=2026):
    if context < 2 or min(max_train, max_eval) < 1:
        raise ValueError("context must be >=2 and caps must be positive")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    users, users_info = read_table(input_dir / "users.csv", "user_id")
    items, items_info = read_table(input_dir / "items.csv", "item_id")
    (u, i, d, reward), schema, missing, invalid = load_interactions(input_dir / "interactions.csv")
    train_mask = d < VAL_START
    item_ids = np.unique(i[train_mask])
    train_users = np.unique(u[train_mask])
    if not len(item_ids):
        raise ValueError("No training items before 2021-08-09")
    mapping = {int(iid): token for token, iid in enumerate(item_ids, 3)}
    tokens = np.fromiter((mapping.get(int(iid), 2) for iid in i), np.int64, count=len(i))
    user_categories, user_maps, user_dim, item_categories, item_features = metadata_features(users, items, train_users, item_ids)
    bounds = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1, len(u)]
    candidates = {name: [] for name in ("train", "val", "test")}
    split_stats = {name: Counter() for name in candidates}
    # The compact candidate triples are target index, history start, history end.
    # Build no contexts until after uniformly sampling candidates.
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        user_days = d[lo:hi]
        train_end = lo + np.searchsorted(user_days, VAL_START)
        if train_end > lo:
            idx = np.arange(lo, train_end)
            ends = lo + np.searchsorted(user_days, d[idx], side="left")
            keep = ends > lo
            split_stats["train"]["excluded_no_prior_history"] += int((~keep).sum())
            idx, ends = idx[keep], ends[keep]
            if len(idx):
                candidates["train"].append(np.column_stack((idx, np.maximum(lo, ends - context + 1), ends)))
        for name, start, stop in (("val", VAL_START, TEST_START), ("test", TEST_START, TEST_END)):
            history_end = lo + np.searchsorted(user_days, start)
            window_end = lo + np.searchsorted(user_days, stop)
            if history_end == window_end:
                continue
            first_end = lo + np.searchsorted(user_days, d[history_end], side="right")
            idx = np.arange(history_end, first_end)
            split_stats[name]["window_rows"] += int(window_end - history_end)
            split_stats[name]["excluded_later_target_dates"] += int(window_end - first_end)
            split_stats[name]["first_date_targets"] += len(idx)
            if not np.any(tokens[lo:history_end] >= 3):
                split_stats[name]["excluded_no_known_history"] += len(idx)
                continue
            split_stats[name]["unseen_target_count_before_cap"] += int(np.count_nonzero(tokens[idx] == 2))
            if len(idx):
                candidates[name].append(np.column_stack((idx, np.full(len(idx), max(lo, history_end - context + 1)), np.full(len(idx), history_end))))
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_num, name in enumerate(candidates):
        rows = np.concatenate(candidates[name]) if candidates[name] else np.empty((0, 3), np.int64)
        eligible = len(rows)
        cap = max_train if name == "train" else max_eval
        if len(rows) > cap:
            rng = np.random.default_rng(np.random.SeedSequence([seed, split_num]))
            rows = rows[np.sort(rng.choice(len(rows), cap, replace=False))]
        x = np.zeros((len(rows), context), np.int64)
        lengths = np.zeros(len(rows), np.int64)
        user_features = np.zeros((len(rows), user_dim), np.float32)
        for n, (target, begin, end) in enumerate(rows):
            lengths[n] = 1 + end - begin
            x[n, 0] = 1
            x[n, 1:lengths[n]] = tokens[begin:end]
            row = users.get(int(u[target]), {})
            for field, values in user_maps.items():
                value = row.get(field, "").strip() or UNKNOWN
                user_features[n, values.get(value, values[UNKNOWN])] = 1
        targets = rows[:, 0]
        np.savez_compressed(output_dir / f"{name}.npz", x=x, lengths=lengths,
                            y=tokens[targets], reward=reward[targets], user_features=user_features,
                            user_id=u[targets], target_item_id=i[targets], target_date=d[targets].astype(np.int64))
        split_stats[name].update(eligible_before_cap=eligible, saved=len(rows), excluded_by_cap=eligible - len(rows), cap=cap)
        split_stats[name]["saved_unseen_targets"] = int(np.count_nonzero(tokens[targets] == 2))
    np.save(output_dir / "item_features.npy", item_features)
    unique_users, history_lengths = np.unique(u, return_counts=True)
    unique_items = np.unique(i)
    valid = reward[np.isfinite(reward)].astype(np.float64) * 100
    quantiles = [0, .1, .25, .5, .75, .9, .99, 1]
    pair_order = np.lexsort((i, u))
    pu, pi = u[pair_order], i[pair_order]
    duplicate_pairs = int(np.count_nonzero((pu[1:] == pu[:-1]) & (pi[1:] == pi[:-1])))
    same_day = (u[1:] == u[:-1]) & (d[1:] == d[:-1])
    eda = {
        "interactions": len(u), "schema": {"interactions": schema, "users": users_info, "items": items_info},
        "users_in_interactions": len(unique_users), "items_in_interactions": len(unique_items),
        "date_min": date.fromordinal(int(d.min())).isoformat(), "date_max": date.fromordinal(int(d.max())).isoformat(),
        "watched_pct_missing": missing, "watched_pct_invalid": invalid,
        "watched_pct_quantiles": {str(q): float(np.quantile(valid, q)) for q in quantiles} if len(valid) else {},
        "watched_pct_mean": float(valid.mean()) if len(valid) else None,
        "watched_pct_zero_count": int(np.count_nonzero(reward == 0)), "watched_pct_100_count": int(np.count_nonzero(reward == 1)),
        "history_length_quantiles": {str(q): float(np.quantile(history_lengths, q)) for q in quantiles},
        "users_orphan_ids": sum(int(uid) not in users for uid in unique_users),
        "items_orphan_ids": sum(int(iid) not in items for iid in unique_items),
        "duplicate_user_item_rows": duplicate_pairs, "same_user_day_extra_rows": int(same_day.sum()),
    }
    metadata = {
        "format_version": 1, "seed": seed, "context": context, "vocab_size": len(item_ids) + 3,
        "user_feature_dim": user_dim, "item_feature_dim": item_features.shape[1],
        "special_tokens": {"PAD": 0, "BOS": 1, "UNK": 2},
        "item_id_to_token": {str(k): v for k, v in mapping.items()},
        "user_categories": user_categories, "item_categories": item_categories,
        "feature_fit_scope": "users/items observed before 2021-08-09; outcome columns excluded",
        "splits": {name: dict(counts) for name, counts in split_stats.items()},
        "split_dates": {"train": "<2021-08-09", "val": "2021-08-09..2021-08-15", "test": "2021-08-16..2021-08-22"},
        "evaluation": "first target date per user/window; all tied targets; histories frozen strictly before split start; unseen targets retained as UNK and count as retrieval misses",
        "train_sampling": "uniform without replacement over eligible rows, independent of reward",
        "context_order": "date then item_id; same-day order arbitrary, never included for a target on that date",
        "target_date_encoding": "Python date.toordinal (days since 0001-01-01, starting at 1)",
        "limitations": ["Rows are user-item last-watch dates with aggregated outcomes, not event sessions.",
                        "watched_pct is an aggregated proxy, not causal reward or duration.",
                        "No true within-day ordering; total_dur is never used to infer content duration.",
                        "Test histories may use validation-window observations; weights and vocab remain frozen."],
    }
    for filename, data in (("metadata.json", metadata), ("eda.json", eda)):
        (output_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/kion/full"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/kion-smoke"))
    parser.add_argument("--max-train", type=int, default=150000)
    parser.add_argument("--max-eval", type=int, default=3000)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    metadata = prepare(args.input, args.output, args.max_train, args.max_eval, args.context, args.seed)
    print(json.dumps(metadata["splits"], indent=2))


if __name__ == "__main__":
    main()
