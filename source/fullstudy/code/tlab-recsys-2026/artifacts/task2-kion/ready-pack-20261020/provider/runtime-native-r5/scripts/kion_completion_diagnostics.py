"""Additive, persisted audit sidecars for the five relative-completion families.

This module does not change frozen user/model coordinates. Baselines are strict
other-user item/type/global prefixes, evaluated at each historical row's own day.
Unavailable/raw-missing/invalid outcomes never become observed neutral zeros.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_pack_history as history


CONTRACT = "kion-completion-diagnostics-v1"
FAMILIES = ("personal.relative_completion", "personal.genre_relative", "personal.type_relative",
            "personal.country_relative", "personal.decade_relative")
ROW_STATUS = {"padding": 0, "residual": 1, "missing": 2, "invalid": 3, "unavailable": 4}
FALLBACK = {"item_observed": 0, "type_prior": 1, "global_prior": 2, "fixed_c0": 3}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(value):
    a = np.asarray(value)
    if a.dtype.kind not in "biufSU":
        raise ValueError("Sidecar bindings require canonical non-object arrays")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError("Nonfinite sidecar binding/diagnostic array")
    return canonical_hash({"shape": list(a.shape), "dtype": a.dtype.str,
        "bytes_sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()})


class _Prefix:
    """Sparse key/day count+sum prefix, with per-entity reset by subtraction."""
    def __init__(self, entities, days, values, size):
        keys = np.asarray(entities, np.int64) * history.MAX_ORDINAL + np.asarray(days, np.int64)
        order = np.argsort(keys, kind="stable")
        keys, values = keys[order], np.asarray(values, np.float64)[order]
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1] if len(keys) else np.array([], np.int64)
        self.keys = keys[starts]
        counts = np.diff(np.r_[starts, len(keys)]) if len(starts) else np.array([], np.int64)
        sums = np.add.reduceat(values, starts) if len(starts) else np.array([], np.float64)
        self.counts = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        self.sums = np.r_[0., np.cumsum(sums, dtype=np.float64)]
        self.base = np.searchsorted(self.keys, np.arange(size, dtype=np.int64) * history.MAX_ORDINAL)

    def before(self, entities, days):
        positions = np.searchsorted(self.keys, entities * history.MAX_ORDINAL + days, side="left")
        return self.counts[positions] - self.counts[self.base[entities]], self.sums[positions] - self.sums[self.base[entities]]


class CompletionDiagnostics:
    """Reuse a validated HistoryIndex; never mutate its features or source state."""
    def __init__(self, history_index, *, genre_columns=None):
        self.history = h = history_index
        self.categories = {field: list(categories) for field, categories in h.categories.items()}
        self.genre_labels = h.labels["genres"]
        if genre_columns is not None:
            if (not isinstance(genre_columns, list) or len(set(genre_columns)) != len(genre_columns)
                    or any(not isinstance(name, str) or not name for name in genre_columns)
                    or not set(h.categories["genres"]).issubset(genre_columns)):
                raise ValueError("Diagnostic genre vocabulary must contain the frozen history vocabulary")
            self.categories["genres"] = list(genre_columns)
            lookup = {name: i for i, name in enumerate(genre_columns)}
            remap = np.array([lookup[name] for name in h.categories["genres"]], np.int16)
            self.genre_labels = np.where(h.labels["genres"] >= 0,
                remap[np.maximum(h.labels["genres"], 0)] if len(remap) else -1, -1)
        valid = (h.statuses == 0) & (h.days < h.fit_cutoff)
        self.items = np.unique(h.items)
        item_slots = np.searchsorted(self.items, h.items[valid])
        self.item_prefix = _Prefix(item_slots, h.days[valid], h.outcomes[valid], len(self.items))
        self.global_prefix = _Prefix(np.zeros(valid.sum(), np.int64), h.days[valid], h.outcomes[valid], 1)
        types = h.types[h.tokens]
        chosen = valid & (types >= 0)
        self.type_prefix = _Prefix(types[chosen], h.days[chosen], h.outcomes[chosen], 2)
        # Exactly the frozen baseline's whole-user subtraction, including raw UNK.
        self.own_count = np.r_[0, np.cumsum(h.statuses == 0, dtype=np.int64)]
        self.own_sum = np.r_[0., np.cumsum(np.where(h.statuses == 0, h.outcomes, 0.), dtype=np.float64)]
        self.own_type_count, self.own_type_sum = {}, {}
        for kind in (0, 1):
            mask = (h.statuses == 0) & (types == kind)
            self.own_type_count[kind] = np.r_[0, np.cumsum(mask, dtype=np.int64)]
            self.own_type_sum[kind] = np.r_[0., np.cumsum(np.where(mask, h.outcomes, 0.), dtype=np.float64)]
        self.manifest = {"contract": CONTRACT, "families": list(FAMILIES), "diagnostic_only": True,
            "frozen_model_tensor_changed": False, "history_contract": history.CONTRACT,
            "source_pins": h.source_pins, "mapping_sha256": canonical_hash(h.mapping),
            "raw_arrays_sha256": canonical_hash({name: array_hash(value) for name, value in
                (("users", h.users), ("items", h.items), ("days", h.days), ("outcomes", h.outcomes), ("statuses", h.statuses))}),
            "effective_metadata_sha256": canonical_hash({
                **{f"labels.{field}": array_hash(value) for field, value in h.labels.items()},
                **{f"known.{field}": array_hash(value) for field, value in h.known.items()},
                **{f"unseen.{field}": array_hash(value) for field, value in h.unseen.items()},
                "diagnostic_genre_labels": array_hash(self.genre_labels), "types": array_hash(h.types),
                "year_valid": array_hash(h.year_valid), "year_outside": array_hash(h.year_outside)}),
            "fit_cutoff_exclusive": h.fit_cutoff, "categories": self.categories,
            "genre_eligibility": [name not in history.EXCLUDED_GENRES for name in self.categories["genres"]],
            "alpha": history.BASELINE_ALPHA, "beta": history.BASELINE_BETA, "c0": history.BASELINE_C0,
            "lambda": history.RELATIVE_LAMBDA, "row_status": ROW_STATUS, "fallback_level": FALLBACK,
            "clock": "other users d < min(history row day,role fit cap); query history d < explicit cutoff <= origin",
            "outcome_availability": "assumed at earlier last_watch_dt; optional query-own availability mask has priority",
            "null_encoding": "observed item/type/group means: zero fill plus missing/empty mask; global_mean has declared c0 fallback",
            "invalid_policy": "Decimal [0,100] parser inherited; invalid excluded, never clipped",
            "metadata_availability": "retrospective snapshot; UNK type/labels never restored by raw ID",
            "scope": "H63 audit; generic baseline_rows is diagnostic, not a new candidate model feature",
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "history_module_sha256": hashlib.sha256(Path(history.__file__).read_bytes()).hexdigest()}
        self.signature = canonical_hash(self.manifest)

    def _user_prefix(self, users, bounds):
        h = self.history
        positions = np.searchsorted(h.unique_users, users)
        found = positions < len(h.unique_users)
        if len(h.unique_users):
            found &= h.unique_users[np.minimum(positions, len(h.unique_users) - 1)] == users
        starts = np.zeros(len(users), np.int64)
        starts[found] = h.user_starts[positions[found]]
        ends = np.searchsorted(h.keys, users * history.MAX_ORDINAL + bounds, side="left")
        ends[~found] = starts[~found]
        return starts, ends

    def baseline_rows(self, user_ids, raw_item_ids, dates):
        """Exact diagnostic baseline/support for each requested user/raw-item/day.

        This API also permits a candidate oracle on an explicit day, but is not
        silently appended to a candidate head. All own rows are excluded in all
        three tiers, including a candidate's already observed own pair.
        """
        h = self.history
        users, items = history._ids(user_ids, "baseline user IDs"), history._ids(raw_item_ids, "baseline raw item IDs")
        days = np.asarray([history._day(value) for value in dates], np.int64)
        if users.shape != items.shape or days.shape != users.shape:
            raise ValueError("Baseline user/item/date arrays must align")
        n = len(users)
        bounds = np.minimum(days, h.fit_cutoff)
        starts, ends = self._user_prefix(users, bounds)
        ng, sg = self.global_prefix.before(np.zeros(n, np.int64), bounds)
        ng -= self.own_count[ends] - self.own_count[starts]
        sg -= self.own_sum[ends] - self.own_sum[starts]
        slots = np.searchsorted(self.items, items)
        found = slots < len(self.items)
        if len(self.items):
            found &= self.items[np.minimum(slots, len(self.items) - 1)] == items
        ni, si = np.zeros(n, np.int64), np.zeros(n, np.float64)
        if np.any(found):
            ni[found], si[found] = self.item_prefix.before(slots[found], bounds[found])
        # Unique raw pair can precede generic candidate day, unlike its own row day.
        for row, (start, end) in enumerate(zip(starts, ends)):
            chosen = (h.items[start:end] == items[row]) & (h.statuses[start:end] == 0)
            ni[row] -= chosen.sum()
            si[row] -= h.outcomes[start:end][chosen].sum(dtype=np.float64)
        tokens = np.fromiter((h.mapping.get(int(item), 2) for item in items), np.int32, count=n)
        kinds = h.types[tokens]
        nt, st = np.zeros(n, np.int64), np.zeros(n, np.float64)
        for kind in (0, 1):
            chosen = kinds == kind
            if chosen.any():
                nt[chosen], st[chosen] = self.type_prefix.before(np.full(chosen.sum(), kind, np.int64), bounds[chosen])
                nt[chosen] -= self.own_type_count[kind][ends[chosen]] - self.own_type_count[kind][starts[chosen]]
                st[chosen] -= self.own_type_sum[kind][ends[chosen]] - self.own_type_sum[kind][starts[chosen]]
        if np.any(ni < 0) or np.any(nt < 0) or np.any(ng < 0):
            raise ValueError("Negative baseline support; invalid raw prefix index")
        global_mean = np.divide(sg, ng, out=np.full(n, history.BASELINE_C0), where=ng > 0)
        type_prior = (st + history.BASELINE_BETA * global_mean) / (nt + history.BASELINE_BETA)
        prior = np.where(kinds >= 0, type_prior, global_mean)
        baseline = (si + history.BASELINE_ALPHA * prior) / (ni + history.BASELINE_ALPHA)
        fallback = np.where(ni > 0, 0, np.where((kinds >= 0) & (nt > 0), 1, np.where(ng > 0, 2, 3)))
        output = {"user_id": users, "raw_item_id": items, "date": days, "baseline_cutoff": bounds,
            "token": tokens, "type": kinds, "n_item": ni, "n_type": nt, "n_global": ng,
            "sum_item": si, "sum_type": st, "sum_global": sg,
            "item_mean": np.divide(si, ni, out=np.zeros(n), where=ni > 0), "item_mean_missing": ni == 0,
            "type_mean": np.divide(st, nt, out=np.zeros(n), where=nt > 0), "type_mean_missing": nt == 0,
            "global_mean": global_mean, "global_mean_missing": ng == 0, "type_prior": type_prior,
            "prior": prior, "baseline": baseline, "baseline_missing": np.zeros(n, bool),
            "no_item_support": ni == 0, "no_type_support": nt == 0, "no_global_support": ng == 0,
            "prior_only": ni == 0, "type_unknown": kinds < 0, "fallback_level": fallback.astype(np.int8),
            "used_c0": ng == 0, "c0": np.full(n, history.BASELINE_C0), "scarce": ni < 5,
            "support_bin": np.select((ni == 0, ni == 1, ni <= 5, ni <= 20), (0, 1, 2, 3), default=4).astype(np.int8)}
        if any(not np.isfinite(value).all() for value in output.values()):
            raise ValueError("Nonfinite baseline diagnostics")
        if np.any((baseline < 0) | (baseline > 1)):
            raise ValueError("Baseline range violation; values are never clipped")
        return output

    def history_sidecar(self, user_ids, query_dates, *, x=None, lengths=None, history_cutoffs=None,
                        row_keys=None, outcome_available=None):
        h = self.history
        idx, mask, tokens, n, origins = h._history(user_ids, query_dates, history_cutoffs, x, lengths)
        b, context = mask.shape
        users = history._ids(user_ids, "query user IDs")
        raw_items = h.items[idx] if len(h.items) else np.zeros_like(idx)
        days = h.days[idx] if len(h.days) else np.zeros_like(idx)
        available = mask.copy() if outcome_available is None else np.asarray(outcome_available)
        if available.shape != mask.shape or available.dtype.kind != "b" or np.any(available & ~mask):
            raise ValueError("Outcome availability must be a boolean exact-H63 mask with zero padding")
        selected = np.flatnonzero(mask.ravel())
        baselines = self.baseline_rows(np.broadcast_to(users[:, None], mask.shape).ravel()[selected],
            raw_items.ravel()[selected], days.ravel()[selected])
        arrays = {"history.mask": mask, "history.raw_row_index": np.where(mask, idx, -1),
                  "history.outcome_available": available}
        for name, values in baselines.items():
            a = np.zeros(mask.shape, values.dtype)
            a.ravel()[selected] = values
            arrays["history." + name] = a
        outcome = h.outcomes[idx] if len(h.outcomes) else np.zeros_like(idx, np.float32)
        status = h.statuses[idx] if len(h.statuses) else np.zeros_like(idx, np.uint8)
        valid = mask & available & (status == 0)
        residual = np.where(valid, outcome - arrays["history.baseline"], 0.)
        arrays.update({"history.outcome": np.where(mask, outcome, 0), "history.parser_status": np.where(mask, status, 0),
            "history.residual": residual, "history.residual_valid": valid,
            "history.row_status": np.where(~mask, 0, np.where(~available, 4, np.where(status == 1, 2,
                np.where(status == 2, 3, 1)))).astype(np.uint8)})
        nr = valid.sum(axis=1)
        overall = np.divide(residual.sum(axis=1), nr, out=np.zeros(b), where=nr > 0)
        for name, value in {"n63": n, "n_valid_outcome": (mask & available & (status == 0)).sum(axis=1),
            "n_missing": (mask & available & (status == 1)).sum(axis=1),
            "n_invalid": (mask & available & (status == 2)).sum(axis=1), "n_unavailable": (mask & ~available).sum(axis=1),
            "n_no_baseline": np.zeros(b, np.int32), "n_residual": nr, "relative_mean": overall,
            "relative_empty": nr == 0, "n_prior_only": (valid & arrays["history.prior_only"]).sum(axis=1),
            "n_type_unknown": (valid & arrays["history.type_unknown"]).sum(axis=1)}.items():
            arrays["query." + name] = value
        arrays["query.prior_only_share_fill"] = np.divide(arrays["query.n_prior_only"], nr,
            out=np.zeros(b), where=nr > 0)
        for field in ("genres", "content_type", "countries", "decade"):
            known = h.known[field][tokens] & mask
            unseen = h.unseen[field][tokens] & mask
            field_valid = known & valid
            ng = field_valid.sum(axis=1)
            for name, value in {"metadata_known_h63": known.sum(axis=1), "metadata_known_residual": ng,
                "metadata_unknown_h63": (mask & ~known).sum(axis=1), "metadata_unknown_residual": (valid & ~known).sum(axis=1),
                "unseen_h63": unseen.sum(axis=1), "unseen_residual": (unseen & valid).sum(axis=1),
                "partial_unknown_h63": (known & unseen).sum(axis=1),
                "partial_unknown_residual": (known & unseen & valid).sum(axis=1),
                "has_unseen": unseen.any(axis=1), "residual_coverage": np.divide(ng, nr, out=np.zeros(b), where=nr > 0)}.items():
                arrays[f"field.{field}.{name}"] = value
            if field == "decade":
                for suffix, chosen in (("h63", mask), ("residual", valid)):
                    arrays[f"field.{field}.valid_year_{suffix}"] = (h.year_valid[tokens] & chosen).sum(axis=1)
                    arrays[f"field.{field}.outside_{suffix}"] = (h.year_outside[tokens] & chosen).sum(axis=1)
            labels = (self.genre_labels if field == "genres" else h.labels[field])[tokens]
            # Each label receives the row once, including multi-label overlaps.
            width = len(self.categories[field])
            group_n, group_sum, group_prior = (np.zeros((b, width), np.float64) for _ in range(3))
            for category in range(width):
                chosen = np.any(labels == category, axis=2) & valid
                group_n[:, category] = chosen.sum(axis=1)
                group_sum[:, category] = np.where(chosen, residual, 0).sum(axis=1)
                group_prior[:, category] = (chosen & arrays["history.prior_only"]).sum(axis=1)
            mu = np.divide(group_sum, group_n, out=np.zeros_like(group_sum), where=group_n > 0)
            delta = np.where(group_n > 0, mu - overall[:, None], 0)
            z = group_n / (group_n + history.RELATIVE_LAMBDA) * delta
            for name, value in {"n": group_n.astype(np.int32), "raw_mu_fill": mu, "raw_delta_fill": delta,
                "z": z, "empty": group_n == 0, "prior_only_n": group_prior.astype(np.int32),
                "prior_only_share_fill": np.divide(group_prior, group_n, out=np.zeros_like(group_sum), where=group_n > 0)}.items():
                arrays[f"group.{field}.{name}"] = value
            arrays[f"history.{field}.known"] = known
            arrays[f"history.{field}.has_unseen"] = unseen
            arrays[f"history.{field}.labels"] = np.where(mask[:, :, None], labels, -1)
        expected = arrays["query.n_missing"] + arrays["query.n_invalid"] + arrays["query.n_unavailable"] + nr
        if not np.array_equal(n, expected):
            raise AssertionError("Outcome row-status partition is not exhaustive")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite diagnostic sidecar")
        cutoffs = origins if history_cutoffs is None else np.asarray([history._day(v) for v in history_cutoffs], np.int64)
        bindings = {"user_id": users, "query_origin": origins, "history_cutoff": cutoffs,
                    "x": np.column_stack((np.ones(b, np.int32), tokens)) if x is None else np.asarray(x).copy(),
                    "lengths": n + 1 if lengths is None else np.asarray(lengths).copy(),
                    "outcome_available": available}
        for name, value in (row_keys or {}).items():
            if name in bindings or not isinstance(name, str) or not name or np.asarray(value).shape[:1] != (b,):
                raise ValueError("Additional row keys must have unique names and one entry per query")
            bindings[name] = np.asarray(value)
        for name, value in bindings.items():
            arrays["binding." + name] = value
        binding_hash = canonical_hash({name: array_hash(value) for name, value in bindings.items()})
        receipt = {"contract": CONTRACT, "signature": self.signature, "row_binding_sha256": binding_hash,
            "array_hashes": {name: array_hash(value) for name, value in arrays.items()},
            "row_count": b, "history_rows": int(mask.sum()), "manifest": self.manifest}
        return arrays, receipt

    def save_sidecar(self, output, user_ids, query_dates, **kwargs):
        """Create NPZ+receipt in a fresh directory, never overwrite."""
        output = Path(output)
        if output.exists():
            raise ValueError("Diagnostic output must be a new directory")
        arrays, receipt = self.history_sidecar(user_ids, query_dates, **kwargs)
        output.mkdir(parents=True)
        np.savez_compressed(output / "sidecar.npz", **arrays)
        receipt["sidecar_sha256"] = hashlib.sha256((output / "sidecar.npz").read_bytes()).hexdigest()
        (output / "manifest.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        return receipt


def load_sidecar(output, *, expected_signature=None, expected_row_binding_sha256=None):
    """Verify physical saved arrays; refuse stale source/row keys and corruption."""
    output = Path(output)
    receipt = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if receipt.get("contract") != CONTRACT or canonical_hash(receipt["manifest"]) != receipt.get("signature"):
        raise ValueError("Diagnostic manifest contract/signature mismatch")
    if expected_signature is not None and expected_signature != receipt["signature"]:
        raise ValueError("Diagnostic source/policy signature mismatch")
    if expected_row_binding_sha256 is not None and expected_row_binding_sha256 != receipt["row_binding_sha256"]:
        raise ValueError("Diagnostic row/H63 binding mismatch")
    manifest = receipt["manifest"]
    if (file_hash(__file__) != manifest["module_sha256"]
            or file_hash(history.__file__) != manifest["history_module_sha256"]):
        raise ValueError("Diagnostic implementation source changed")
    for pin in manifest["source_pins"].values():
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("Diagnostic raw/source pin mismatch")
    path = output / "sidecar.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sidecar_sha256"]:
        raise ValueError("Diagnostic NPZ hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(receipt["array_hashes"]):
            raise ValueError("Diagnostic array schema mismatch")
        arrays = {name: archive[name] for name in archive.files}
    if any(array_hash(value) != receipt["array_hashes"][name] for name, value in arrays.items()):
        raise ValueError("Diagnostic array hash mismatch")
    bindings = {name[8:]: value for name, value in arrays.items() if name.startswith("binding.")}
    if canonical_hash({name: array_hash(value) for name, value in bindings.items()}) != receipt["row_binding_sha256"]:
        raise ValueError("Diagnostic physical row binding mismatch")
    return arrays, receipt
