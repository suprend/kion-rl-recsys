"""Future-only bounded query preparation for the cloned CompleteProvider.

No ML imports, provider loading, installation, worker or runtime activation.
Requires the immutable PersonalIndex.prepare_queries contract reviewed here.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np

MAX_ROWS = 16
NUMERIC_SCRATCH_ESTIMATE_CAP = 64 << 20
PROFILE_FIELDS = frozenset(("owner", "owner_schema_hash", "n", "origins", "user_ids",
    "query_cutoffs", "tokens", "mask", "genre_has_unseen", "interests", "keys",
    "known_n", "distinct_n", "text", "user_features"))


def _row_profile(profile, row, count):
    if set(vars(profile)) != PROFILE_FIELDS:
        raise ValueError("Unsupported prepared query fields; use sequential path")

    def take(value):
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or len(value) != count:
                raise ValueError("Prepared query array row dimension differs")
            return value[row:row + 1].copy()
        if isinstance(value, dict):
            return {key: take(item) for key, item in value.items()}
        if isinstance(value, list) and len(value) == count:
            return [deepcopy(value[row])]
        raise ValueError("Unsupported prepared query field type")

    return type(profile)(**{name: (getattr(profile, name) if name in ("owner", "owner_schema_hash")
        else take(getattr(profile, name))) for name in PROFILE_FIELDS})


def _chunk_rows(personal):
    # Conservative admission estimate for z[tokens], np.where and float64 row
    # profiles, all text fields, plus history/interest working storage. This is
    # not an RSS guarantee: Python/native library overhead needs the owned guard.
    text_per_row = sum(2 * 63 * matrix.shape[1] * max(8, matrix.dtype.itemsize)
        + 4 * matrix.shape[1] * 8 for matrix in personal.text_matrices.values())
    interest_per_row = personal.genre_incidence.shape[1] * 512
    estimate_per_row = (256 << 10) + text_per_row + interest_per_row
    return min(MAX_ROWS, NUMERIC_SCRATCH_ESTIMATE_CAP // max(1, estimate_per_row))


def query_users(provider, rows, split):
    """Match sequential _query cache semantics while preparing miss chunks.

    Shadow keys model *each access*, including duplicate misses after eviction.
    Actual LRU mutation stays in the original ordered _query calls. No provider
    object/cache is changed during speculative preparation. On an ordinary
    preparation/slicing failure, replay the same chunk sequentially, preserving
    the original successful cache prefix and first original exception.
    """
    if provider.personal is None or len(rows) < 2:
        return [provider._query(split, row)["user"] for row in rows]
    limit = _chunk_rows(provider.personal)
    if limit < 2:
        return [provider._query(split, row)["user"] for row in rows]
    data = provider.splits[split]
    result = []
    for begin in range(0, len(rows), limit):
        current = rows[begin:begin + limit]
        shadow = OrderedDict.fromkeys(provider._profiles)
        missing = []
        for pos, row in enumerate(current):
            origin = provider._origin(split, row)
            key = (int(data["user_id"][row]), origin, int(data["lengths"][row]), data["x"][row].tobytes())
            if key in shadow:
                shadow.move_to_end(key)
            else:
                missing.append((pos, row, origin))
                shadow[key] = None
                while len(shadow) > provider.profile_cache_rows:
                    shadow.popitem(last=False)
        if not missing:
            result.extend(provider._query(split, row)["user"] for row in current)
            continue
        packed = None
        prepared = None
        try:
            take = np.array([row for _, row, _ in missing], np.int64)
            origins = [origin for _, _, origin in missing]
            prepared = provider.personal.prepare_queries(data["user_id"][take], origins,
                x=data["x"][take], lengths=data["lengths"][take], history_cutoffs=origins)
            packed = {pos: _row_profile(prepared, index, len(missing))
                for index, (pos, _, _) in enumerate(missing)}
        except Exception:
            # Pure immutable preparation has no LRU effects. The original row
            # path determines the exception and partial cache state, if any.
            packed = None
        finally:
            del prepared
        if packed is None:
            result.extend(provider._query(split, row)["user"] for row in current)
        else:
            for pos, row in enumerate(current):
                result.append(provider._query(split, row, prepared_profile=packed.get(pos))["user"])
        del packed
    return result
