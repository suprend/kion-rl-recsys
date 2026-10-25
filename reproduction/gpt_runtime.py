"""Scoped runtime I/O ports. Explicit caller-admitted backends; no ML on import."""
from array import array
import csv
from datetime import date
from gpt_provider import logical_child

FIT_CUTOFF = "2021-08-09"
SERIES_SHA = "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6"


def init_scoped_runtime(provider, config, *, io, runtime):
    """Compose only active scoped branches after independent scoped admission.

    runtime.load(name) supplies admitted modules; logical_source(module) returns
    their original logical source identity. Provider owns failure cleanup.
    """
    from gpt_related import load_scoped_related, load_scoped_franchise
    self = provider
    if not self.scoped or self.partial:
        raise ValueError("Runtime adapter requires admitted scoped production")
    for name in ("history", "personal", "audience", "audience_core", "people_dynamics",
                 "dynamic", "related", "graph", "prepared_graph"):
        setattr(self, name, None)
    self.runtime_provenance = {}
    if any(config.get(k) for k in ("audience", "people_dynamics", "entity_links")):
        raise ValueError("Deferred runtime branches are outside scoped support")
    if not any(config.get(k) for k in ("personal_candidates", "local_dynamic", "related_interest", "franchise_progress")):
        return
    raw_dir, annotation = prepare_scoped_inputs(io, self.root, self.metadata, config)

    def admitted(name):
        module = runtime.load(name)
        self._require_sources([module])
        return module

    history = admitted("kion_pack_history")
    source = runtime.logical_source(history)
    self.history = build_history(io, raw_dir, self.metadata, backend=history,
        builder_source={"path": source, "sha256": io.bindings[source].sha256})
    self.runtime_provenance["history"] = self.history.provenance
    if config.get("personal_candidates"):
        self.personal = build_personal(io, raw_dir, self.metadata,
            backend=admitted("kion_personal_candidates"), history_index=self.history,
            annotation_logical=annotation, text_matrices=self.text_features,
            text_missing=self.text_missing,
            representation_manifest={k: v["signature"] for k, v in self.text_manifests.items()})
        self._columns["user"] += ["personal." + c for c in self.personal.user_columns]
        for field, values in sorted(self.text_features.items()):
            self._columns["user"] += [f"text_profile.{field}.{i}" for i in range(values.shape[1])]
            self._columns["user"] += [f"text_profile.{field}.{c}" for c in ("known_n", "mean_norm", "valid", "zero_norm")]
        self._columns["candidate"] += ["personal." + c for c in self.personal.columns]
        self.runtime_provenance["personal_candidates"] = self.personal.provenance
        self.materialized_families.add("personal_candidates")
    if config.get("local_dynamic"):
        self.dynamic = build_dynamic(io, raw_dir, self.metadata,
            backend=admitted("kion_pack_dynamic"), annotation_logical=annotation)
        self._columns["candidate"] += ["local_dynamic." + c for c in self.dynamic.columns]
        self.runtime_provenance["local_dynamic"] = self.dynamic.provenance
        self.materialized_families.add("local_dynamic")
    if config.get("related_interest"):
        related = admitted("kion_related_features")
        self.related, provenance = load_scoped_related(io, self.root, config["related_interest"],
            self.metadata, raw_root=raw_dir,
            data_root=logical_child(self.root, self.manifest["data"]["path"]),
            np=related.np, related=related)
        self._columns["candidate"] += list(self.related.columns)
        self.runtime_provenance.update(provenance)
        self.materialized_families.add("related_interest")
    if config.get("franchise_progress"):
        prepared, mode, columns, provenance = load_scoped_franchise(io, self.root,
            config["franchise_progress"], self.metadata, franchise=admitted("kion_franchise_progress"))
        self.prepared_graph, self.graph, self.graph_mode = prepared, prepared.graph, mode
        self.graph_legacy_prefix = False
        self.graph_order_kinds = provenance["order_kinds"]
        self.graph_scope_policies = provenance["scope_policies"]
        self._columns["candidate"] += columns
        self.runtime_provenance["franchise_progress"] = provenance
        self.materialized_families.add("franchise_progress")
    for pin in config.get("source_pins", []):
        logical = logical_child(self.root, pin["path"])
        io.pin(self.root, pin, io.bindings[logical].role)


def prepare_scoped_inputs(io, root, metadata, config):
    """Validate scoped raw pins before builds; admission remains caller-owned.

    Returns logical raw directory and annotation. Related/franchise readers are
    composed elsewhere. Dynamic annotation must be explicit, never searched.
    """
    if any(config.get(k) for k in ("audience", "people_dynamics", "entity_links")):
        raise ValueError("Deferred runtime branches are outside scoped support")
    if config.get("fit_cutoff", FIT_CUTOFF) != FIT_CUTOFF:
        raise ValueError("Runtime SFT fit cutoff mismatch")
    raw = logical_child(root, config["raw_dir"])
    required = {logical_child(raw, name) for name in ("items.csv", "interactions.csv")}
    annotation = metadata.get("series_annotation_path")
    if annotation is not None:
        annotation = logical_child(root, annotation)
    if config.get("personal_candidates"):
        receipt = config.get("series_annotation")
        if not receipt or receipt.get("sha256") != SERIES_SHA:
            raise ValueError("Scoped personal candidates require reviewed annotation")
        explicit = logical_child(root, receipt["path"])
        if annotation is not None and annotation != explicit:
            raise ValueError("Scoped series annotation differs from prepared metadata")
        io.pin(root, receipt, "metadata")
        annotation = explicit
    if config.get("local_dynamic") and annotation is None:
        raise ValueError("Dynamic annotation must be explicitly bound; no path search")
    if annotation is not None:
        required.add(annotation)
    pinned = set()
    for pin in config.get("source_pins", []):
        logical = logical_child(root, pin["path"])
        io.pin(root, pin, io.bindings[logical].role)
        pinned.add(logical)
    if not required.issubset(pinned):
        raise ValueError("Runtime raw source pins missing")
    return raw, annotation


def csv_rows(io, logical, fields):
    path = io.verify(logical, "payload")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or len(set(header)) != len(header) or not set(fields).issubset(header):
            raise ValueError("Missing/duplicate CSV columns: " + logical)
        columns = [header.index(field) for field in fields]
        for line, row in enumerate(reader, 2):
            io.check()
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV row {line}: {logical}")
            yield tuple(row[index] for index in columns)
    io.verify(logical, "payload")


def _hash(io, logical):
    binding = io.bindings[logical]
    io.verify(logical, binding.role)
    return binding.sha256


def build_history(io, raw_dir, metadata, *, backend, builder_source):
    """Retained HistoryIndex constructor with identically parsed arrays/cards."""
    b, np = backend, backend.np
    paths = {"interactions": logical_child(raw_dir, "interactions.csv"),
             "items": logical_child(raw_dir, "items.csv"),
             "history_builder": builder_source["path"]}
    io.verify(paths["history_builder"], "source", builder_source["sha256"])
    pins = {name: {"path": path, "sha256": _hash(io, path)} for name, path in paths.items()}
    users, item_ids, days = array("q"), array("q"), array("i")
    outcomes, statuses, bits = array("f"), array("B"), array("B")
    date_cache = {}
    for raw_user, raw_item, raw_date, raw_pct in csv_rows(io, paths["interactions"],
            ("user_id", "item_id", "last_watch_dt", "watched_pct")):
        try:
            uid, iid = int(raw_user), int(raw_item)
        except ValueError as error:
            raise ValueError("Invalid interaction ID") from error
        if str(uid) != raw_user or str(iid) != raw_item or min(uid, iid) < 0:
            raise ValueError("Noncanonical interaction ID")
        users.append(uid)
        item_ids.append(iid)
        if raw_date not in date_cache:
            date_cache[raw_date] = b._day(raw_date)
        days.append(date_cache[raw_date])
        outcome, status, threshold_bits = b.parse_outcome(raw_pct)
        outcomes.append(outcome)
        statuses.append(status)
        bits.append(threshold_bits)
    fields = ("item_id", "genres", "content_type", "release_year", "countries", "age_rating")
    cards = {}
    for iid, *values in csv_rows(io, paths["items"], fields):
        raw_id = int(iid)
        if str(raw_id) != iid or raw_id < 0 or raw_id in cards:
            raise ValueError("Invalid/duplicate metadata item ID")
        cards[raw_id] = dict(zip(fields[1:], values))
    io.check()
    result = b.HistoryIndex(np.asarray(users, np.int64), np.asarray(item_ids, np.int64),
        np.asarray(days, np.int32), np.asarray(outcomes, np.float32),
        np.asarray(statuses, np.uint8), np.asarray(bits, np.uint8), cards,
        metadata["item_id_to_token"], fit_cutoff=FIT_CUTOFF, include_relative=False,
        schema_item_ids=metadata["item_id_to_token"].keys())
    for path in paths.values():
        _hash(io, path)
    result.source_pins = pins
    result.provenance["source_pins"] = pins
    return result


def build_personal(io, raw_dir, metadata, *, backend, history_index,
                   annotation_logical, text_matrices=None, text_missing=None,
                   genre_matrix=None, genre_columns=None, representation_manifest=None):
    b, np = backend, backend.np
    meta, index = metadata, history_index
    if index.mapping != {int(key): value for key, value in meta["item_id_to_token"].items()}:
        raise ValueError("History and metadata mappings differ")
    if genre_matrix is None and "item_categories" in meta:
        genre_columns = list(meta["item_categories"]["genres"])
        genre_matrix = np.zeros((index.vocab_size, len(genre_columns)), np.float64)
        lookup = {name: i for i, name in enumerate(genre_columns)}
        for old, name in enumerate(index.categories["genres"]):
            if name not in lookup:
                raise ValueError("History genre schema exceeds the fixed prepared vocabulary")
            genre_matrix[:, lookup[name]] = np.any(index.labels["genres"] == old, axis=1)
    token_keys = {role: [() for _ in range(index.vocab_size)] for role in b.ROLES}
    for iid, actors, directors, studios in csv_rows(io, logical_child(raw_dir, "items.csv"), ("item_id", "actors", "directors", "studios")):
        token = index.mapping.get(int(iid))
        if token is not None:
            for role, raw in zip(b.ROLES[:3], (actors, directors, studios)):
                token_keys[role][token] = b.exact_keys(raw)
    annotation = io.read_json(annotation_logical)
    seen_ids = set()
    for series in annotation["series"]:
        if series.get("status") != "confirmed":
            continue
        key = series["id"]
        if not isinstance(key, str) or not key or key in seen_ids:
            raise ValueError("Invalid/duplicate confirmed franchise key")
        seen_ids.add(key)
        for member in series["members"]:
            if not isinstance(member, dict) or member.get("status") != "confirmed":
                raise ValueError("Only explicitly confirmed franchise memberships are permitted")
            iid = member["item_id"] if isinstance(member, dict) else member
            token = index.mapping.get(int(iid))
            if token is not None:
                token_keys["franchise"][token] = tuple(sorted(set(token_keys["franchise"][token]) | {key}))
    matrices = {}
    for role, rows in token_keys.items():
        columns = sorted({key for row in rows for key in row})
        mapping = {key: pos for pos, key in enumerate(columns)}
        matrices[role] = {"offsets": np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64),
            "entities": np.array([mapping[key] for row in rows for key in row], np.int64), "key_columns": columns}
    manifest = dict(representation_manifest or {})
    manifest.update(series_annotation_sha256=_hash(io, annotation_logical),
                    items_sha256=_hash(io, logical_child(raw_dir, "items.csv")))
    return b.PersonalCandidateIndex(index, key_matrices=matrices, text_matrices=text_matrices,
        text_missing=text_missing, genre_matrix=genre_matrix, genre_columns=genre_columns,
        representation_manifest=manifest)


def _interaction_table(io, path, mapping, size, fit_cutoff, b):
    np = b.np
    tokens, days, outcomes, states, all_pairs = (array("i"), array("i"), array("d"), array("b"), array("q"))
    date_cache = {}
    source_rows = excluded = 0
    for user_text, item_id, day_text, outcome_text in csv_rows(io, path,
            ("user_id", "item_id", "last_watch_dt", "watched_pct")):
        source_rows += 1
        if day_text not in date_cache:
            date_cache[day_text] = b._day(day_text)
        day = date_cache[day_text]
        token = mapping.get(item_id)
        if token is None:
            excluded += 1
            continue
        try:
            user = int(user_text)
        except ValueError as error:
            raise ValueError("Invalid user_id") from error
        if user < 0 or user > (np.iinfo(np.int64).max - token) // max(size, 1):
            raise ValueError("Invalid/overflow user_id")
        all_pairs.append(user * size + token)
        if day >= fit_cutoff:
            continue
        value, state = b._parse_outcome(outcome_text)
        tokens.append(token)
        days.append(day)
        outcomes.append(value)
        states.append(state)
    pair_keys = np.frombuffer(all_pairs, dtype=np.int64)
    pair_keys.sort()
    if len(pair_keys) > 1 and np.any(pair_keys[1:] == pair_keys[:-1]):
        raise ValueError("Duplicate user/item retained pair in interactions.csv")
    del pair_keys, all_pairs
    tokens = np.frombuffer(tokens, dtype=np.int32)
    days = np.frombuffer(days, dtype=np.int32)
    values = np.frombuffer(outcomes, dtype=np.float64)
    states = np.frombuffer(states, dtype=np.int8)
    keys = tokens.astype(np.int64) * b.DAY_BASE + days
    order = np.argsort(keys, kind="stable")
    keys, days, values, states = keys[order], days[order], values[order], states[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1] if len(keys) else np.array([], dtype=int)
    grouped = np.zeros((len(starts), b.N_STATS), dtype=np.float64)
    if len(starts):
        valid = states == 0
        grouped[:, b.PAIR] = np.diff(np.r_[starts, len(keys)])
        grouped[:, b.OBS] = np.add.reduceat(valid.astype(np.int64), starts)
        grouped[:, b.SUM] = np.add.reduceat(values, starts)
        grouped[:, b.MISSING] = np.add.reduceat((states == 1).astype(np.int64), starts)
        grouped[:, b.INVALID] = np.add.reduceat((states == 2).astype(np.int64), starts)
        for offset, selected in enumerate((values <= .10, values >= .20, values >= .50,
                                           values >= .80, values >= .95), 5):
            grouped[:, offset] = np.add.reduceat((valid & selected).astype(np.int64), starts)
        grouped[:, b.ACTIVE] = 1
        grouped[np.arange(len(starts)), b.WEEKDAY + (days[starts] - 1) % 7] = grouped[:, b.PAIR]
    return keys[starts], grouped, {"source_rows": source_rows, "excluded_unmapped_rows": excluded,
                                  "indexed_rows": int(grouped[:, b.PAIR].sum())}


def _relations(io, raw_dir, annotation_logical, mapping, size, b):
    np = b.np
    by_token = {role: [()] * size for role in b.ROLES}
    seen = set()
    for item_id, actors, directors, studios in csv_rows(io, logical_child(raw_dir, "items.csv"),
                                                       ("item_id", "actors", "directors", "studios")):
        if not item_id or item_id in seen:
            raise ValueError("Missing/duplicate item_id in items.csv")
        seen.add(item_id)
        token = mapping.get(item_id)
        if token is not None:
            for role, value in zip(b.ROLES, (actors, directors, studios)):
                by_token[role][token] = b._tags(value)
    path = annotation_logical
    if path is not None:
        annotation = io.read_json(path)
        groups = annotation.get("series")
        if not isinstance(groups, list):
            raise ValueError("Malformed confirmed series annotation")
        group_ids = set()
        for group in groups:
            if group.get("status") != "confirmed":
                continue
            group_id = group.get("id")
            if not isinstance(group_id, str) or not group_id or group_id in group_ids:
                raise ValueError("Missing/duplicate confirmed series ID")
            group_ids.add(group_id)
            for member in group.get("members", []):
                if member.get("status") == "confirmed":
                    token = mapping.get(str(member.get("item_id")))
                    if token is not None:
                        by_token["series"][token] = tuple(sorted(set(by_token["series"][token]) | {group_id}))
    tables = {}
    for role in b.ROLES:
        labels = sorted({label for values in by_token[role] for label in values})
        lookup = {label: i for i, label in enumerate(labels)}
        offsets = np.r_[0, np.cumsum([len(values) for values in by_token[role]])].astype(np.int64)
        entities = np.array([lookup[label] for values in by_token[role] for label in values], dtype=np.int32)
        card_counts = np.bincount(entities, minlength=len(labels))
        tables[role] = {"offsets": offsets, "entities": entities, "card_counts": card_counts,
                        "size": len(labels)}
    return tables, path


def build_dynamic(io, raw_dir, metadata, *, backend, annotation_logical):
    """Read local CSVs once and build sparse prefixes, capped strictly at fit_cutoff."""
    if not isinstance(annotation_logical, str) or not annotation_logical:
        raise ValueError("Dynamic annotation must be explicitly bound; no path search")
    b, np = backend, backend.np
    fit_cutoff = FIT_CUTOFF
    meta, mapping = b._metadata(metadata)
    cutoff, size = b._day(fit_cutoff), meta["vocab_size"]
    relations, series_path = _relations(io, raw_dir, annotation_logical, mapping, size, b)
    keys, values, diagnostics = _interaction_table(io, logical_child(raw_dir, "interactions.csv"), mapping, size, cutoff, b)
    item = b._Prefix(keys, values, size)
    global_prefix = b._Prefix(keys % b.DAY_BASE, values, 1)
    prefixes = {role: b._entity_prefix(keys, values[:, b.PAIR], relation) for role, relation in relations.items()}
    provenance = {"contract": b.CONTRACT, "fit_cutoff_exclusive": date.fromordinal(cutoff).isoformat(),
        "effective_cutoff": "min(query_date, supplied_history_cutoff, fit_cutoff); strict last_watch_dt < cutoff",
        "source_unit": "retained aggregated user-item pair; not event history or exposure",
        "temporal_mode": "retrospective metadata; reconstructed strict last-pair prefixes",
        "mapping_sha256": b._canonical_hash(mapping), "interactions_sha256": _hash(io, logical_child(raw_dir, "interactions.csv")),
        "items_sha256": _hash(io, logical_child(raw_dir, "items.csv")), "series_sha256": _hash(io, series_path) if series_path else None,
        "parameters": {"completion_alpha": b.ALPHA, "weekday_alpha": b.ALPHA, "weekday_beta": b.WEEKDAY_BETA,
                       "pop_alpha": b.POP_ALPHA, "trend_lambda": b.TREND_LAMBDA, "trend_clip": b.TREND_CLIP,
                       "completion_constants": [.5, .1, .8, .5, .2, .05]},
        "source_calendar_start": "2021-03-13; declared dataset interval, not proof of complete logging",
        "related_self_exclusion": "candidate raw item only; no verified edition equivalences supplied",
        "related_distinct_users": "unavailable; all related support explicitly counts retained pairs",
        "metadata_identity": "NFKC/casefold exact names, deduplicated within role; unresolved aliases not merged",
        "related_entity_data_fraction": "fraction of listed entities with strict-prefix observed other-card retained pairs",
        "priors": "window-specific strict prefix; empty window -> same-cutoff cumulative -> fixed constants",
        "thresholds_pct": {"lower": [10], "upper": [20, 50, 80, 95]}, **diagnostics}
    provenance["index_bytes"] = item.nbytes + global_prefix.nbytes + sum(prefix.nbytes for prefix in prefixes.values())
    provenance["sparse_item_days"] = len(item.keys)
    provenance["sparse_entity_days"] = {role: len(prefix.keys) for role, prefix in prefixes.items()}
    return b.DynamicIndex(item, global_prefix, relations, prefixes, size, cutoff, provenance)
