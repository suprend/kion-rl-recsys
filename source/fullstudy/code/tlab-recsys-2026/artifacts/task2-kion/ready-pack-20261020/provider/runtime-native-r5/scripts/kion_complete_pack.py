"""Strict, read-only composition of the complete KION training feature package.

The immutable local baseline is reused, never rewritten. Expensive personal
profiles are live and bounded; no row x text-dimension sidecar is produced.
Full admission requires reviewed full-scope receipts for every agreed family.
Partial mode is an explicit integration experiment and cannot train full.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_local_pack as local
import kion_pack_train as baseline
import kion_pack_history as history_module
import kion_text_embeddings as text_module

CONTRACT = "kion-complete-pack-v1"
REQUIRED_FAMILIES = ("local_base", "local_dynamic", "semantic_extensions", "text3fields",
    "people_embeddings", "personal_candidates", "audience", "people_dynamics",
    "franchise_progress", "external_facts", "external_interest", "entity_links")
FIT_CUTOFF = date(2021, 8, 9).toordinal()
FRANCHISE_ORDER_KINDS = ("mainline_release", "catalog_release_precedes")
SCOPED_FAMILIES = ("local_base", "local_dynamic", "semantic_extensions", "text3fields", "people_embeddings",
    "personal_candidates", "franchise_progress", "external_interest")
SCOPED_READY_COMPONENTS = ("professions", "geography_time", "sports_settings", "story_format", "narrative_relations",
    "static_replacements", "related_interest", "completion_diagnostics", "user_diagnostics")
TEXT_FIELDS = text_module.FIELDS
NUMERIC_TRANSFORMS = {"contract": "kion-complete-numeric-v4", "ordinal_epoch": "2020-01-01",
    "ordinal_divisor": 365.25, "ordinal_columns": ["audience.global_cutoff_ordinal", "audience.query_state_cutoff_ordinal", "related.cutoff_ordinal"],
    "missing_sentinel": "zero remains zero; availability masks remain unchanged",
    "existing_log_normalized_cosine": "identity; never apply a second log transform",
    "nonnegative_counts_elapsed": "log1p; exact named columns from numeric_log_columns; reject negatives",
    "nonnegative_physical": "log1p; declared duration minutes, career calendar years, credits per year; reject negatives; preserve zero and masks",
    "model_range": "all transformed item/user/candidate values finite and abs<=65504; otherwise reject by exact column",
    "original_item_prefix": "first 197 coordinates are identity",
    "fit": "fixed constants; no train/validation statistics"}
SOURCE_FILES = ("kion_complete_pack.py", "kion_local_pack.py", "kion_local_features.py", "kion_pack_train.py",
    "kion_pack_history.py", "kion_conditioned_train.py", "kion_full_train.py", "kion_text_embeddings.py",
    "kion_personal_candidates.py", "kion_audience_features.py", "kion_people_dynamics.py",
    "kion_pack_dynamic.py", "kion_franchise_progress.py", "kion_people_embeddings.py", "kion_semantic_extensions.py",
    "kion_external_facts.py", "kion_external_interest.py", "kion_entity_links.py", "kion_audience_fast.py",
    "kion_related_features.py", "kion_profession_relations.py", "kion_geography_time.py", "kion_sports_settings.py",
    "kion_story_format.py", "kion_complete_manifest.py", "kion_completion_event_store.py", "kion_narrative_relations.py", "kion_people_banks.py", "kion_user_diagnostics.py",
    "kion_people_dynamics_fast.py", "hm_job.py", "kion_completion_diagnostics.py")
COMPONENT_ADAPTERS = ("local_base", "local_dynamic", "semantic_extensions", "text3fields", "people_embeddings",
    "personal_candidates", "audience", "people_dynamics", "franchise_progress", "external_facts", "external_interest",
    "entity_links", "candidate_model", "related_interest", "professions", "geography_time", "sports_settings",
    "story_format", "static_replacements", "completion_diagnostics", "narrative_relations", "user_diagnostics")
ASSEMBLY_PAYLOAD_FIELDS = ("item_extensions", "item_replacements", "text_caches", "people_catalog", "runtime", "date_features", "entity_links", "diagnostics")
STATIC_MODULES = {"semantic_extensions": "kion_semantic_extensions.py", "professions": "kion_profession_relations.py",
    "geography_time": "kion_geography_time.py", "sports_settings": "kion_sports_settings.py", "story_format": "kion_story_format.py", "narrative_relations": "kion_narrative_relations.py"}


def numeric_log_columns():
    """Declarative source-semantic allowlist, independent of observed magnitudes.

    In particular, director popularity and person-window popularity are already
    logged by their source; actor popularity is raw. Support counts for both
    roles remain raw. Masks, growth, probabilities, and categorical codes never
    enter this allowlist. The expanded intersection is persisted in contract().
    """
    names = set()
    def add(prefix, fields):
        names.update(prefix + field for field in fields)
    add("audience.", ("item_n", "active_users_n", "global_pairs_n"))
    for field in ("age", "income", "sex", "kids_flg"):
        add("demography_" + field + ".", ("known_n", "unknown_field_n", "unseen_category_n", "no_user_row_n", "global_known_n"))
    add("taste.", ("informative_viewers_n", "audience_n", "other_known_items_mean", "other_known_items_min", "other_known_items_max",
        "query_history_n", "query_known_n", "query_genre_empty_n", "query_genre_missing_n"))
    add("coview.", ("history_n", "supported_neighbors_n", "history_degree_mean"))
    for relation in ("studios", "franchises"):
        prefix = "related." + relation + "."
        add(prefix, ("listed_entities", "other_cards", "other_work_units", "confirmed_other_works"))
        for window in (1, 7, 30):
            add(prefix + f"union_{window}d.", ("pairs", "distinct_viewers", "previous_pairs", "previous_distinct_viewers", "observed_works",
                "global_pairs", "global_distinct_viewers", "previous_global_pairs", "previous_global_distinct_viewers", "observed_days"))
        for window in (7, 30, 60, 90):
            add(prefix + "release.", (f"{field}_{window}d" for field in ("count_lower", "count_upper", "observed_count_upper", "observed_exact_day_count", "count_exact")))
        add(prefix + "release.", ("known_date_works", "unknown_date_works", "exact_date_works", "interval_date_works", "recency_lower_days",
            "recency_upper_days", "days_since_latest", "observed_recency_lower_days", "observed_recency_upper_days"))
    popularity = ("catalog_work_count", "other_work_count", "observed_work_count", "observation_count", "audience_union", "work_popularity_mean",
        "work_popularity_max", "work_popularity_median", "work_popularity_std", "work_popularity_q25", "work_popularity_q75", "work_popularity_q90", "other_work_mean_viewers")
    raw_metrics = {"experience_" + field for field in ("credits_before_work", "films_before_work", "series_before_work", "credits_prev_3y", "credits_prev_5y",
        "years_since_first_known_credit", "years_since_previous_known_credit", "distinct_prior_release_years", "same_year_other_credits", "unknown_year_credits")}
    for mode in ("catalog", "prior_release"):
        raw_metrics.update(f"quality_{mode}_{field}" for field in ("observed_work_count", "valid_user_count", "missing_outcome_count", "invalid_outcome_count"))
        for kind in ("genre", "format"):
            raw_metrics.update(f"specialization_{mode}_{kind}_{field}" for field in ("n_all", "n_valid", "n_unknown"))
    raw_metrics.update(f"related_{window}d_observed_days" for window in (7, 30))
    # Metric support names do not require importing or executing the producer.
    # Source schema builds available_count for every metric; allowlist patterns
    # are expanded against exact schema names in numeric_plan below.
    for role in ("actors", "directors"):
        add(f"people_{role}_", ("listed_count",))
        for metric in raw_metrics | (set(popularity) if role == "actors" else set()):
            add(f"people_{role}_{metric}_", ("mean", "max", "median", "std", "q25", "q75", "q90", "top3_mean"))
    for tier in ("low_evidence", "ordinary", "high", "very_high"):
        add("people_actors_tier_" + tier + "_", ("count",))
    add("people_actors_", ("threshold_pool_count",))
    for group in ("profile_genres", "profile_countries", "profile_type", "profile_decade", "profile_age_rating", "completion", "completion_film", "completion_series", "completion_7d", "completion_30d", "relative"):
        add("history." + group + ".", ("n", "n_known", "n_valid", "n_missing", "n_invalid"))
    add("history.temporal.", ("cards_1d", "dates_1d", "cards_7d", "dates_7d", "cards_30d", "dates_30d", "n_gaps"))
    add("personal.history_pop.", ("n", "valid_n", "positive_n", "global_empty_n"))
    for field in TEXT_FIELDS:
        add("text_profile." + field + ".", ("known_n",))
        add("personal.text_" + field + ".", ("history_n", "known_n"))
    for role in ("studio", "actor", "director", "franchise"):
        add("personal.familiarity_" + role + ".", ("mean_count", "max_count", "candidate_key_count", "matched_key_count", "n_known", "distinct_history_keys"))
    add("personal.familiarity.", ("n",))
    add("personal.director_novelty.", ("history_n", "history_known_n", "candidate_key_n"))
    for k in (1, 2, 3):
        add(f"personal.interest_k{k}.", ("history_n", "known_n", "min_cluster_support"))
    for prefix in ("franchise.", *("franchise." + kind + "." for kind in FRANCHISE_ORDER_KINDS)):
        add(prefix, ("n63", "known_raw_rows", "unmapped_rows", "n_branch_seen", "n_prior", "n_prior_seen", "n_prior_missing", "skip_observed_priors", "recency_days", "graph_hop_distance"))
    external = ("runtime_support", "runtime_episode_sample_n", "work_runtime_proxy_assertion_n", "work_runtime_proxy_distinct_value_n",
        "award_accepted_claim_support", "award_unresolved_attribution_support", "tv_released_episode_count_asof", "tv_started_season_count_asof", "tv_days_since_last_released_episode_asof", "tv_support")
    add("external_facts.", external)
    for language in ("original", "spoken", "edition_audio", "subtitles"):
        add("external_facts.language_" + language + ":", ("support",))
    for role, job in (("actor", "acting"), ("director", "directing")):
        add("external_facts." + role + "_", ("accepted_person_n", "raw_name_n", "filmography_available_person_n"))
        for award_scope in ("person", job):
            add(f"external_facts.award_{role}_{award_scope}_", ("won_count", "nominated_count"))
        for metric in ("observed_work_count", "credits_before_work", "films_before_work", "series_before_work", "distinct_prior_release_years",
                       "observed_career_span_years", "observed_credits_per_span_year", "years_since_first_observed_credit", "confirmed_outside_kion_work_count", "membership_unknown_count"):
            add(f"external_facts.{role}_external_{metric}_", ("mean", "max"))
    add("external_interest.", ("movielens_count", "views_7d", "views_30d", "observed_days", "zero_assumed_days"))
    for language in ("ru", "en"):
        add(f"external_interest.wiki_work_{language}_", ("views_7d", "views_30d"))
        for role in ("actor", "director"):
            prefix = f"external_interest.wiki_{role}_{language}_"
            add(prefix, ("n_people_total", "n_people_mapped", "n_people_with_valid_window_7d", "n_people_with_valid_window_30d"))
            for window in (7, 30):
                add(prefix + f"views_{window}d_", ("mean", "max", "median", "top3"))
    return names


NUMERIC_UNITS = {"raw_count": "log1p_nonnegative", "nonnegative_elapsed_days": "log1p_nonnegative",
    "nonnegative_duration_minutes": "log1p_nonnegative", "nonnegative_elapsed_years": "log1p_nonnegative",
    "nonnegative_credits_per_year": "log1p_nonnegative",
    "already_log": "identity", "signed_growth": "identity", "signed_log_mean_growth": "identity", "probability": "identity", "mask": "identity",
    "rating": "identity", "rating_0.5_to_5": "identity", "cosine": "identity"}


def numeric_plan(feature_columns, declared_units=None):
    allowed = numeric_log_columns()
    plan = {scope: {} for scope in feature_columns}
    for scope, columns in feature_columns.items():
        for index, name in enumerate(columns):
            if scope == "item" and index < 197:
                continue
            if name in NUMERIC_TRANSFORMS["ordinal_columns"]:
                plan[scope][name] = "ordinal_years_since_2020"
            elif name in allowed or (name.startswith(("people_actors_", "people_directors_")) and name.endswith("_available_count")):
                plan[scope][name] = "log1p_nonnegative"
    for scope, columns in (declared_units or {}).items():
        for name, unit in columns.items():
            if unit not in NUMERIC_UNITS or name not in feature_columns[scope]:
                raise ValueError("Unknown numeric unit/column: " + name)
            if scope == "item" and feature_columns[scope].index(name) < 197:
                raise ValueError("Original197 cannot receive a declared numeric transform")
            if NUMERIC_UNITS[unit] != "identity":
                plan[scope][name] = NUMERIC_UNITS[unit]
            else:
                plan[scope].pop(name, None)
    return plan


def canonical_hash(value):
    return text_module.canonical_hash(value)


def file_pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": text_module.file_hash(path)}


def _path(root, value):
    value = Path(value)
    return value if value.is_absolute() else root / value


def _pin(root, declaration):
    if not isinstance(declaration, dict) or set(("path", "sha256")) - set(declaration):
        raise ValueError("Every source/artifact requires path and sha256")
    path = _path(root, declaration["path"])
    if text_module.file_hash(path) != declaration["sha256"]:
        raise ValueError("Source/artifact SHA mismatch: " + path.as_posix())
    return path


def _names(names, name):
    if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
            or len(names) != len(set(names))):
        raise ValueError("Invalid/duplicate " + name + " columns")
    return names


def _provenance(root, declaration):
    provenance = declaration.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source_pins"):
        raise ValueError("Derived matrix requires nonempty provenance source_pins")
    for pin in provenance["source_pins"]:
        _pin(root, pin)


def _positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} requires an integer in 1..{maximum}")
    return value


def _matrix(root, declaration, vocab, metadata_hash, vocabulary_hash):
    if (declaration.get("metadata_sha256") != metadata_hash
            or declaration.get("vocabulary_sha256") != vocabulary_hash):
        raise ValueError("Feature matrix metadata/vocabulary binding mismatch")
    path = _pin(root, declaration)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (values.ndim != 2 or len(values) != vocab or values.shape[1] < 1 or values.dtype.kind != "f"
            or list(values.shape) != declaration.get("shape") or str(values.dtype) != declaration.get("dtype")
            or np.any(values[:3] != 0)):
        raise ValueError("Feature matrix vocabulary/dtype/shape/special rows mismatch")
    for begin in range(0, vocab, 4096):
        if not np.isfinite(values[begin:begin + 4096]).all():
            raise ValueError("Nonfinite feature matrix")
    _names(declaration["columns"], "matrix")
    if values.shape[1] != len(declaration["columns"]):
        raise ValueError("Feature matrix named width mismatch")
    return values


def _related_historical_builder(root, recorded, receipt):
    expected = "98da6c6b939ff3dbe5c8464a826a775437d43fd5b6257efafab33caff34ec739"
    if (not isinstance(receipt, dict) or recorded.get("sha256") != expected or receipt.get("sha256") != expected
            or Path(recorded.get("path", "")).name != "kion_audience_features.py"):
        raise ValueError("Related historical audience builder requires an explicit exact source alias")
    _pin(root, receipt)
    return {"recorded_builder": copy.deepcopy(recorded), "historical_source": copy.deepcopy(receipt)}


def _scoped_base_data(data_dir, feature_dir):
    """Exact baseline validation through bounded reads before opening cold mmaps."""
    data_dir, feature_dir = Path(data_dir), Path(feature_dir)
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("metadata.json", "train.npz", "val.npz"):
        if baseline.sha256(data_dir / name) != manifest["inputs"][name]["sha256"]:
            raise ValueError("Feature source SHA mismatch: " + name)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    vocab, context = int(metadata["vocab_size"]), int(metadata["context"])
    if vocab <= 3:
        raise ValueError("Empty known catalog")
    matrices = {}
    try:
        for name in ("item_features.npy", "train_user_features.npy", "val_user_features.npy"):
            declaration, path = manifest["outputs"][name], feature_dir / name
            if baseline.sha256(path) != declaration["sha256"]:
                raise ValueError("Feature output SHA mismatch: " + name)
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                if (len(shape) != 2 or list(shape) != declaration["shape"] or dtype.kind != "f"
                        or str(dtype) != declaration["dtype"] or fortran):
                    raise ValueError("Feature dtype/shape mismatch: " + name)
                remaining = int(np.prod(shape)) * dtype.itemsize
                while remaining:
                    block = stream.read(min(16 << 20, remaining))
                    if not block or len(block) % dtype.itemsize or not np.isfinite(np.frombuffer(block, dtype=dtype)).all():
                        raise ValueError("Truncated/nonfinite feature payload: " + name)
                    remaining -= len(block)
                if stream.read(1):
                    raise ValueError("Trailing feature payload: " + name)
            matrices[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        items = matrices["item_features.npy"]
        if len(items) != vocab or np.any(items[:3] != 0):
            raise ValueError("Item features require matching vocabulary and zero special rows")
        train, val = (baseline.FeatureData(baseline.shared.load_data(data_dir / (split + ".npz"), vocab, context,
            require_keys=True, unique_keys=split == "val")) for split in ("train", "val"))
        user_dim = matrices["train_user_features.npy"].shape[1]
        baseline._columns(manifest["feature_columns"], items.shape[1], user_dim)
        if (baseline.canonical_hash(manifest["feature_columns"]) != manifest["schema_hash"]
                or manifest.get("rowkeys_algorithm") != baseline.ROWKEYS_ALGORITHM):
            raise ValueError("Feature schema/row algorithm mismatch")
        for split, data in (("train", train), ("val", val)):
            user = matrices[split + "_user_features.npy"]
            if user.shape != (len(data["y"]), user_dim) or manifest["row_keys_sha256"][split] != baseline.make_row_keys_hash(data):
                raise ValueError("Feature row identity/width mismatch: " + split)
            data["user_features"] = user
            for value in data.values():
                value.setflags(write=False)
        return metadata, train, val, items, manifest
    except Exception:
        for value in matrices.values():
            value._mmap.close()
        raise


def write_manifest(output, *, data_dir, base_cache, families, item_extensions=(),
                   text_caches=None, people_catalog=None, runtime=None, date_features=(),
                   source_paths=(), integration_mode="full", limits=None, entity_links=None,
                   item_replacements=(), diagnostics=None, assembly=None, scope_decision=None):
    """Create a receipt only; never build raw indices, encode text, or mutate base.

    All payload declarations are explicit inputs, including reviewed acceptance.
    This writer does not promote pilot evidence or infer a positive review.
    """
    output, data_dir, base_cache = Path(output), Path(data_dir), Path(base_cache)
    if output.exists():
        raise ValueError("A new composite directory is required")
    data = {"path": data_dir.resolve().as_posix(), "files": {
        name: file_pin(data_dir / name) for name in ("metadata.json", "train.npz", "val.npz")}}
    row_hashes = {}
    for split in ("train", "val"):
        with np.load(data_dir / (split + ".npz"), allow_pickle=False) as archive:
            row_hashes[split] = local.row_keys_hash({key: archive[key] for key in local.ROW_KEYS})
    source_paths = list(source_paths) + [Path(__file__).with_name(name) for name in SOURCE_FILES if Path(__file__).with_name(name).is_file()]
    pins = {Path(path).resolve().as_posix(): file_pin(path) for path in source_paths}
    manifest = {"contract": CONTRACT, "state": "completed", "integration_mode": integration_mode,
        "data": data, "base_cache": {"path": base_cache.resolve().as_posix(),
            "manifest_sha256": text_module.file_hash(base_cache / "manifest.json")},
        "row_keys_sha256": row_hashes, "rowkeys_algorithm": local.ROW_ALGORITHM,
        "families": copy.deepcopy(families), "required_families": list(REQUIRED_FAMILIES),
        "item_extensions": list(item_extensions), "text_caches": text_caches or {},
        "people_catalog": people_catalog, "runtime": runtime or {}, "date_features": list(date_features),
        "entity_links": entity_links,
        "item_replacements": list(item_replacements), "diagnostics": diagnostics or {}, "assembly": assembly,
        "source_pins": list(pins.values()), "limits": limits or {},
        "numeric_transforms": copy.deepcopy(NUMERIC_TRANSFORMS),
        "metadata_availability": "retrospective_catalog_snapshot_not_historical_asof",
        "outcome_availability": "last_watch_date_assumption_strict_before_whole_day",
        "fit_cutoff": "2021-08-09", "validation_origin": "2021-08-09"}
    if integration_mode == "scoped_production":
        import kion_complete_manifest as assembler
        decision = assembler.scoped_decision(scope_decision, output)
        manifest.update(scope_decision=scope_decision, scope_id=decision["scope_id"],
            active_families=decision["active_families"], additional_ready_components=decision["additional_ready_components"],
            required_families=list(SCOPED_FAMILIES), status="admitted_scoped", full_row_fit_admitted=True, full_package_admitted=False)
    elif scope_decision is not None:
        raise ValueError("scope_decision requires scoped_production mode")
    manifest["signature"] = canonical_hash(manifest)
    output.mkdir(parents=True)
    text_module.write_json(output / "manifest.json", manifest)
    return manifest


class CompleteProvider:
    """Admitted cheap catalog branches plus bounded query/candidate feature APIs."""

    @classmethod
    def load(cls, data_dir, feature_dir, *, allow_partial_integration=False, scope_decision=None):
        obj = cls()
        obj.root = Path(feature_dir).resolve()
        obj.manifest = json.loads((obj.root / "manifest.json").read_text(encoding="utf-8"))
        obj._maps, obj._profiles = [], OrderedDict()
        obj.requested_scope_decision = scope_decision
        try:
            obj._load(Path(data_dir).resolve(), allow_partial_integration)
        except Exception:
            obj.close()
            raise
        return obj

    def _admission(self, allow_partial):
        m = self.manifest
        if m.get("contract") != CONTRACT or m.get("state") != "completed":
            raise ValueError("Unfinished/unknown complete package")
        if canonical_hash({k: v for k, v in m.items() if k != "signature"}) != m.get("signature"):
            raise ValueError("Complete manifest signature mismatch")
        if m.get("integration_mode") not in ("full", "partial", "scoped_production"):
            raise ValueError("Explicit full/partial integration_mode required")
        if m["integration_mode"] == "partial" and not allow_partial:
            raise ValueError("Partial package requires explicit allow_partial_integration=True; cannot_train_full")
        self.partial = m["integration_mode"] == "partial"
        self.scoped = m["integration_mode"] == "scoped_production"
        requested_scope = getattr(self, "requested_scope_decision", None)
        self.required_families = SCOPED_FAMILIES if self.scoped else REQUIRED_FAMILIES
        if self.scoped:
            import kion_complete_manifest as assembler
            if requested_scope is None:
                raise ValueError("Scoped production requires explicit scope_decision")
            if Path(requested_scope).resolve() != _path(self.root, m["scope_decision"]["path"]).resolve():
                raise ValueError("Requested scope decision path differs from scoped manifest")
            self.scope = assembler.scoped_decision(m.get("scope_decision"), self.root)
            if (m.get("scope_id") != self.scope["scope_id"] or m.get("active_families") != list(SCOPED_FAMILIES)
                    or m.get("status") != "admitted_scoped" or m.get("full_row_fit_admitted") is not True
                    or m.get("full_package_admitted") is not False
                    or m.get("additional_ready_components") != list(SCOPED_READY_COMPONENTS)
                    or set(m.get("families", {})) != set(SCOPED_FAMILIES)
                    or any(m.get("runtime", {}).get(key) for key in ("people_dynamics", "audience"))
                    or m.get("entity_links") is not None
                    or any(d.get("family") != "external_interest" for d in m.get("date_features", []))):
                raise ValueError("Scoped production payload/families differ from the approved ready scope")
        elif requested_scope is not None:
            raise ValueError("scope_decision cannot admit full or partial mode")
        if m.get("required_families") != list(self.required_families):
            raise ValueError("Required family census differs from the complete contract")
        self.acceptance = {}
        for family in self.required_families:
            receipt = m.get("families", {}).get(family, {})
            accepted = (receipt.get("status") == "accepted" and receipt.get("reviewed") is True
                        and receipt.get("scope") == ("approved_ready_scope" if self.scoped else "full") and bool(receipt.get("source_pins")))
            self.acceptance[family] = ("accepted_scoped" if self.scoped else "accepted_full") if accepted else "missing_unreviewed_or_pilot"
            if not self.partial and not accepted:
                raise ValueError("Full package rejects missing/unreviewed/pilot family: " + family)
            for pin in receipt.get("source_pins", []):
                _pin(self.root, pin)
        if m.get("fit_cutoff") != "2021-08-09" or m.get("validation_origin") != "2021-08-09":
            raise ValueError("SFT role cap/frozen validation origin mismatch")
        if m.get("numeric_transforms") != NUMERIC_TRANSFORMS:
            raise ValueError("Numeric transform registry differs from the complete contract")
        if m.get("rowkeys_algorithm") != local.ROW_ALGORITHM:
            raise ValueError("Unknown complete row binding algorithm")
        pins = m.get("source_pins", [])
        if not pins or not any(_path(self.root, p["path"]).resolve() == Path(__file__).resolve() for p in pins):
            raise ValueError("Complete builder source pin is required")
        for pin in pins:
            _pin(self.root, pin)
        if not self.partial:
            import kion_complete_manifest as assembler
            self._require_sources([assembler])
            assessment_path = self.root / "assessment.json"
            assessment = json.loads(assessment_path.read_text(encoding="utf-8")) if assessment_path.is_file() else {}
            assembly = m.get("assembly") or {}
            rows = assessment.get("agreements", [])
            approved = {key for key in assembler.AGREEMENT_COMPONENTS if key not in assembler.DEFERRED_PROPOSALS}
            if self.scoped:
                if (assembly.get("status") != "scoped_data_ready_untrained"
                        or assembly.get("assessment_sha256") != canonical_hash(assessment)
                        or assessment.get("contract") != "kion-complete-scoped-assessment-v1"
                        or assessment.get("status") != "scoped_data_ready_untrained" or assessment.get("blockers")
                        or assessment.get("scope_decision") != m["scope_decision"]
                        or assessment.get("active_families") != list(SCOPED_FAMILIES)
                        or set(assessment.get("components", {})) != set(assembler.SCOPED_COMPONENTS)
                        or any(not component.get("ready") for component in assessment.get("components", {}).values())
                        or assessment.get("provider_source", {}).get("sha256") != text_module.file_hash(__file__)
                        or assessment.get("base_manifest", {}).get("sha256") != m["base_cache"]["manifest_sha256"]
                        or assessment.get("metadata", {}).get("sha256") != m["data"]["files"]["metadata.json"]["sha256"]):
                    raise ValueError("Scoped production requires its bound retained-component assessment")
            elif (assembly.get("status") != "data_ready_untrained"
                    or assembly.get("assessment_sha256") != canonical_hash(assessment)
                    or assessment.get("status") != "data_ready_untrained" or assessment.get("blockers")
                    or assessment.get("inventory_count") != 82 or assessment.get("required_count") != len(approved)
                    or assessment.get("agreement_mapping_sha256") != canonical_hash(assembler.AGREEMENT_COMPONENTS)
                    or {row["id"] for row in rows if row.get("required_for_full")} != approved
                    or any(row.get("state") != "data_ready" for row in rows if row.get("required_for_full"))
                    or assessment.get("provider_source", {}).get("sha256") != text_module.file_hash(__file__)
                    or assessment.get("base_manifest", {}).get("sha256") != m["base_cache"]["manifest_sha256"]
                    or assessment.get("metadata", {}).get("sha256") != m["data"]["files"]["metadata.json"]["sha256"]):
                raise ValueError("Full assembly requires the bound assessment of all approved agreement IDs")
            expected, pending = assembler.compose_payload(assessment, {"COMPONENT_ADAPTERS": COMPONENT_ADAPTERS})
            if pending or any(m.get(key) != value for key, value in expected.items()):
                raise ValueError("Full manifest payload differs from its reviewed approved-ID assessment")
            for component in assessment.get("components", {}).values():
                _pin(self.root, component["evidence"])
                _pin(self.root, component["review"])

    def _require_sources(self, modules):
        declared = {_path(self.root, p["path"]).resolve() for p in self.manifest["source_pins"]}
        for module in modules:
            if Path(module.__file__).resolve() not in declared:
                raise ValueError("Runtime module source pin missing: " + module.__name__)

    def _load(self, data_dir, allow_partial):
        self._admission(allow_partial)
        m = self.manifest
        self._require_sources([baseline, local, text_module])
        if _path(self.root, m["data"]["path"]).resolve() != data_dir:
            raise ValueError("Prepared data directory differs from composite binding")
        for name in ("metadata.json", "train.npz", "val.npz"):
            if _pin(self.root, m["data"]["files"][name]).resolve() != data_dir / name:
                raise ValueError("Prepared data file path binding mismatch")
        base_dir = _path(self.root, m["base_cache"]["path"])
        if text_module.file_hash(base_dir / "manifest.json") != m["base_cache"]["manifest_sha256"]:
            raise ValueError("Base manifest SHA mismatch")
        loader = _scoped_base_data if self.scoped else baseline.load_feature_data
        metadata, train, val, items, self.base_manifest = loader(data_dir, base_dir)
        self.metadata, self.splits = metadata, {"train": train, "val": val}
        self.metadata_hash = m["data"]["files"]["metadata.json"]["sha256"]
        self.vocabulary_hash = canonical_hash(metadata["item_id_to_token"])
        self._raw_item_ids_by_token = MappingProxyType({token: raw for raw, token in metadata["item_id_to_token"].items()})
        self._maps.extend((items, train["user_features"], val["user_features"]))
        if self.partial and any(len(data["y"]) > 4096 for data in self.splits.values()):
            raise ValueError("Partial integration is limited to tiny smoke splits (maximum 4096 rows each)")
        if self.scoped and (len(train["y"]) != self.scope["validation"]["train_rows"]
                or len(val["y"]) != self.scope["validation"]["validation_rows"]):
            raise ValueError("Scoped production requires the exact approved full prepared rows")
        if metadata["context"] != 64:
            raise ValueError("Complete package requires BOS+H63 context64")
        for split, data in self.splits.items():
            if local.row_keys_hash(data) != m["row_keys_sha256"].get(split):
                raise ValueError("Composite row hash mismatch: " + split)
            times = data["target_date"]
            lower, upper = (1, FIT_CUTOFF) if split == "train" else (FIT_CUTOFF, FIT_CUTOFF + 7)
            if np.any(times < lower) or np.any(times >= upper):
                raise ValueError("Target row outside SFT role date interval")
        self._origins = {}
        for split, data in self.splits.items():
            values = np.minimum(data["target_date"], FIT_CUTOFF).astype(np.int64, copy=True) if split == "train" else np.full(len(data["y"]), FIT_CUTOFF, np.int64)
            # Immutable bytes own the backing memory: a caller cannot re-enable
            # writes through setflags and alter the resume-bound schedule.
            self._origins[split] = np.frombuffer(values.tobytes(), dtype=np.int64)
        limits = m.get("limits", {})
        self.profile_cache_rows = _positive(limits.get("profile_cache_rows", 128), "profile_cache_rows", 4096)
        self.max_batch_rows = _positive(limits.get("max_batch_rows", 1024), "max_batch_rows", 4096)
        self.max_pair_cells = _positive(limits.get("max_pair_cells", 65536), "max_pair_cells", 262144)
        self.max_pair_bytes = _positive(limits.get("max_pair_bytes", 64 << 20), "max_pair_bytes", 256 << 20)
        self.bank_cache_days = _positive(limits.get("bank_cache_days", 2), "bank_cache_days", 4)
        self._banks = OrderedDict()
        self._columns = {"item": list(self.base_manifest["feature_columns"]["item"]),
                         "user": list(self.base_manifest["feature_columns"]["user"]), "candidate": []}
        matrices = [items]
        extension_declarations = []
        self.materialized_families = {"local_base"}
        for declaration in m.get("item_extensions", []):
            family = declaration["family"]
            if family not in STATIC_MODULES:
                raise ValueError("Unsupported static extension family/temporal external interest")
            source = Path(__file__).with_name(STATIC_MODULES[family]).resolve()
            if source not in {_path(self.root, p["path"]).resolve() for p in m["source_pins"]}:
                raise ValueError("Static family builder source pin missing: " + family)
            namespace = declaration.get("namespace", family)
            if not isinstance(namespace, str) or not (namespace == family or namespace.startswith(family + ".")):
                raise ValueError("Static extension requires an unambiguous family namespace")
            if declaration.get("temporal_mode") != "static_retrospective":
                raise ValueError("Static extensions require explicit retrospective metadata mode")
            _provenance(self.root, declaration)
            values = self._matrix(declaration)
            self._maps.append(values)
            transform = declaration.get("transform")
            if transform == "signed_log1p":
                indices = declaration.get("transform_columns")
                if (not isinstance(indices, list) or len(set(indices)) != len(indices)
                        or any(type(i) is not int or not 0 <= i < values.shape[1] for i in indices)):
                    raise ValueError("Explicit signed-log column indices required")
                values = np.asarray(values, np.float32).copy()
                values[:, indices] = np.sign(values[:, indices]) * np.log1p(np.abs(values[:, indices]))
            elif transform != "identity":
                raise ValueError("Unknown extension transform")
            matrices.append(values)
            extension_declarations.append(declaration)
            self._columns["item"] += [namespace + "." + c for c in declaration["columns"]]
            self.materialized_families.add(family)
        self.item_features = np.concatenate(matrices, axis=1) if len(matrices) > 1 else items
        self.replacements = []
        replaced = set()
        for declaration in m.get("item_replacements", []):
            if declaration.get("family") not in STATIC_MODULES or declaration.get("mode") != "replace_named_columns":
                raise ValueError("Unknown typed named replacement family/mode")
            source = Path(__file__).with_name(STATIC_MODULES[declaration["family"]]).resolve()
            if source not in {_path(self.root, p["path"]).resolve() for p in m["source_pins"]}:
                raise ValueError("Replacement builder source pin missing")
            if (declaration.get("base_manifest_sha256") != m["base_cache"]["manifest_sha256"]
                    or declaration.get("base_item_features_sha256") != self.base_manifest["outputs"]["item_features.npy"]["sha256"]
                    or declaration.get("base_schema_hash") != self.base_manifest["schema_hash"]
                    or declaration.get("transform") != "identity"):
                raise ValueError("Replacement does not bind exact immutable base/schema/units")
            _provenance(self.root, declaration)
            import kion_complete_manifest as assembler
            self._require_sources([assembler])
            assembler._replacement(declaration, {"sha256": m["base_cache"]["manifest_sha256"]},
                self.base_manifest, extension_declarations, self.root)
            values = self._matrix(declaration)
            self._maps.append(values)
            indices = []
            for name in declaration["columns"]:
                if name not in self._columns["item"] or name in replaced:
                    raise ValueError("Replacement target absent or duplicated")
                index = self._columns["item"].index(name)
                if index < 197:
                    raise ValueError("Replacement cannot touch original197")
                indices.append(index)
                replaced.add(name)
            if not self.replacements:
                self.item_features = np.asarray(self.item_features, np.float32).copy()
            before_hash = text_module.array_hash(self.item_features[:, indices])
            if declaration.get("replaced_values_sha256", before_hash) != before_hash:
                raise ValueError("Replacement preimage values mismatch")
            self.item_features[:, indices] = values
            self.replacements.append({"family": declaration["family"], "columns": declaration["columns"],
                "matrix_sha256": declaration["sha256"], "replaced_values_sha256": before_hash,
                "target_extension": declaration.get("target_extension")})
            self.materialized_families.add(declaration["family"])
        self.item_features.flags.writeable = False
        self.text_features, self.text_missing, self.text_manifests = {}, {}, {}
        if set(m.get("text_caches", {})) - set(TEXT_FIELDS):
            raise ValueError("Unknown text cache field")
        vocab_hash = text_module.canonical_hash({"ordered_item_ids": text_module._mapping(metadata),
                                                "special_tokens": text_module.SPECIALS})
        for field, receipt in m.get("text_caches", {}).items():
            directory = _path(self.root, receipt["path"])
            if text_module.file_hash(directory / "manifest.json") != receipt["manifest_sha256"]:
                raise ValueError("Text manifest SHA mismatch")
            cache = text_module.load_item_cache(directory, expected_vocabulary_sha256=vocab_hash,
                                                allow_test_encoder=self.partial)
            if cache.manifest["field"] != field:
                raise ValueError("Text cache field binding mismatch")
            expected_meta = m["data"]["files"]["metadata.json"]["sha256"]
            if cache.manifest["source_pins"]["metadata"]["sha256"] != expected_meta:
                raise ValueError("Text metadata source pin mismatch")
            for pin in cache.manifest["source_pins"].values():
                _pin(self.root, pin)
            self.text_features[field], self.text_missing[field] = cache.embeddings, cache.missing
            self.text_manifests[field] = cache.manifest
            self._maps.extend((cache.embeddings, cache.missing, cache.token_lengths, cache.truncated))
        if set(self.text_features) == set(TEXT_FIELDS):
            self.materialized_families.add("text3fields")
        self.people_catalog = None
        if m.get("people_catalog"):
            import kion_people_embeddings as people
            path = _pin(self.root, m["people_catalog"])
            self.people_catalog = people.load_catalog(path)
            if (self.people_catalog["vocab_size"] != metadata["vocab_size"]
                    or self.people_catalog["item_id_to_token"] != metadata["item_id_to_token"]):
                raise ValueError("People catalog vocabulary differs from prepared data")
            identity = self.people_catalog["source_identity"]
            if identity.get("metadata", {}).get("sha256") != m["data"]["files"]["metadata.json"]["sha256"]:
                raise ValueError("People catalog metadata source binding mismatch")
            for pin in identity.values():
                _pin(self.root, pin)
            self.materialized_families.add("people_embeddings")
            self._require_sources([people])
        self._init_runtime(m.get("runtime", {}))
        self.date_features = []
        declared_units = {"candidate": {}}
        for declaration in m.get("date_features", []):
            if (declaration.get("family") not in ("external_facts", "external_interest")
                    or declaration.get("clock") != "strict_prefix"
                    or declaration.get("fit_cutoff") != "2021-08-09"):
                raise ValueError("External date bank clock/family/role cap mismatch")
            _names(declaration["columns"], "date-bank")
            units = declaration.get("numeric_units")
            if units is None and not self.partial:
                raise ValueError("Full external date banks require exhaustive named numeric_units")
            if units is not None:
                if set(units) != set(declaration["columns"]) or any(unit not in NUMERIC_UNITS for unit in units.values()):
                    raise ValueError("External numeric_units must classify every exact column")
                declared_units["candidate"].update({declaration["family"] + "." + name: unit for name, unit in units.items()})
            if not declaration.get("banks"):
                raise ValueError("Empty external date banks")
            for day, bank in declaration["banks"].items():
                if str(history_module._day(int(day))) != day or int(day) > FIT_CUTOFF:
                    raise ValueError("External date bank after role cap")
                if (bank.get("cutoff_exclusive") != int(day) or bank.get("columns") != declaration["columns"]
                        or bank.get("numeric_units") != units
                        or not isinstance(bank.get("provenance"), dict)):
                    raise ValueError("External date bank cutoff/schema/provenance missing")
                _provenance(self.root, bank)
                values = self._matrix(bank)
                # Validate every admitted bank now, while retaining no date x V
                # mappings. Full admission must not defer corruption to training.
                if isinstance(values, np.memmap):
                    values._mmap.close()
            required_days = set(map(int, np.unique(np.minimum(self.splits["train"]["target_date"], FIT_CUTOFF)))) | {FIT_CUTOFF}
            if not required_days.issubset(set(map(int, declaration["banks"]))):
                raise ValueError("Exact strict-prefix external banks do not cover all train/val origins")
            self.date_features.append(declaration)
            self._columns["candidate"] += [declaration["family"] + "." + c for c in declaration["columns"]]
            self.materialized_families.add(declaration["family"])
        # Entity links are a nonnumeric dependency but still need actual admitted bytes.
        if m.get("entity_links"):
            import kion_entity_links as links
            self._require_sources([links])
            path = _pin(self.root, m["entity_links"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("contract") != links.CONTRACT or not isinstance(payload.get("works"), list)
                    or not isinstance(payload.get("people"), list)):
                raise ValueError("Unknown entity-link export schema")
            if not self.partial and payload.get("coverage", {}).get("selected_items") != len(metadata["item_id_to_token"]):
                raise ValueError("Entity-link export is a pilot rather than the full selected catalog")
            self.materialized_families.add("entity_links")
        for scope, columns in self._columns.items():
            _names(columns, scope)
        if not self.partial:
            missing = set(self.required_families) - self.materialized_families
            if missing:
                raise ValueError("Accepted receipts have no actual materialization: " + ",".join(sorted(missing)))
        self.feature_columns = copy.deepcopy(self._columns)
        self.diagnostics = self._load_diagnostics(m.get("diagnostics", {}))
        plan = numeric_plan(self.feature_columns, declared_units)
        self.numeric_transforms = {**copy.deepcopy(NUMERIC_TRANSFORMS), "applied_columns": plan,
            "declared_units": declared_units, "expanded_mapping_sha256": canonical_hash(plan), "allowlist_sha256": canonical_hash(sorted(numeric_log_columns()))}
        self._numeric_indices = {scope: [(self.feature_columns[scope].index(name), operation)
            for name, operation in columns.items()] for scope, columns in plan.items()}
        self.item_numeric_audit = self._numeric_audit(self.item_features, "item")
        if self._numeric_indices["item"]:
            self.item_features = np.array(self.item_features, dtype=np.float32, copy=True)
            self._transform_numeric(self.item_features, "item")
        self.item_numeric_audit["after"] = self._numeric_audit(self.item_features, "item")["before"]
        self._validate_model_range(self.item_features, "item")
        self._contract = {"contract": CONTRACT, "manifest_signature": m["signature"],
            "status": "cannot_train_full" if self.partial else "admitted_full",
            "family_acceptance": self.acceptance, "materialized_families": sorted(self.materialized_families),
            "schema_hash": canonical_hash(self.feature_columns), "feature_columns": self.feature_columns,
            "row_keys_sha256": m["row_keys_sha256"], "base_manifest_sha256": m["base_cache"]["manifest_sha256"],
            "text_signatures": {k: v["signature"] for k, v in self.text_manifests.items()},
            "runtime_provenance": self.runtime_provenance, "limits": {"profile_cache_rows": self.profile_cache_rows,
                "bank_cache_days": self.bank_cache_days, "max_batch_rows": self.max_batch_rows,
                "max_pair_cells": self.max_pair_cells, "max_pair_bytes": self.max_pair_bytes}, "fit_cutoff": "2021-08-09",
            "validation_origin": "2021-08-09", "preserved_original_item_prefix": min(197, items.shape[1]),
            "text_transform": "identity_unit_L2; live_H63_mean_then_L2; no_log1p"}
        self._contract["numeric_transforms"] = copy.deepcopy(self.numeric_transforms)
        self._contract["origin_keys"] = {"contract": "kion-sft-origin-keys-v1", "dtype": "int64", "train": "min(target_day,2021-08-09)", "val": "2021-08-09",
            "sha256": {split: text_module.array_hash(values) for split, values in self._origins.items()}}
        self._contract["item_replacements"] = copy.deepcopy(self.replacements)
        self._contract["diagnostics"] = copy.deepcopy(self.diagnostics)
        if self.scoped:
            materialized = set(self.materialized_families)
            if self.replacements:
                materialized.add("static_replacements")
            for store in self._diagnostic_stores.values():
                contract = store.manifest.get("contract")
                if contract == "kion-completion-event-store-v1":
                    materialized.add("completion_diagnostics")
                if contract == "kion-user-diagnostics-v1":
                    materialized.add("user_diagnostics")
            if not set(SCOPED_READY_COMPONENTS) <= materialized:
                raise ValueError("Scoped production lacks a retained ready component")
            self._contract.update(integration_mode="scoped_production", status="admitted_scoped",
                scope_id=self.scope["scope_id"], scope_decision=m["scope_decision"], active_families=list(SCOPED_FAMILIES),
                additional_ready_components=list(SCOPED_READY_COMPONENTS), materialized_families=list(SCOPED_FAMILIES),
                materialized_components=list(SCOPED_READY_COMPONENTS), full_package_admitted=False, full_row_fit_admitted=True)

    def _init_runtime(self, config):
        self.history = self.personal = self.audience = self.audience_core = self.people_dynamics = self.dynamic = self.related = self.graph = self.prepared_graph = None
        self.runtime_provenance = {}
        active = any(config.get(k) for k in ("personal_candidates", "audience", "people_dynamics", "local_dynamic", "related_interest", "franchise_progress"))
        if not active:
            return
        raw_dir = _path(self.root, config["raw_dir"])
        if config.get("fit_cutoff", "2021-08-09") != "2021-08-09":
            raise ValueError("Runtime SFT fit cutoff mismatch")
        required = {raw_dir / name for name in ("items.csv", "interactions.csv")}
        if config.get("audience"):
            required.add(raw_dir / "users.csv")
        if config.get("personal_candidates"):
            annotation = self.metadata.get("series_annotation_path")
            if self.scoped:
                receipt = config.get("series_annotation")
                if not receipt or receipt.get("sha256") != "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6":
                    raise ValueError("Scoped personal candidates require their reviewed series annotation")
                explicit = _pin(self.root, receipt).resolve()
                if annotation is not None and Path(annotation).resolve() != explicit:
                    raise ValueError("Scoped series annotation differs from prepared metadata")
                annotation = explicit.as_posix()
            if not annotation:
                raise ValueError("Explicit pinned series_annotation_path required for personal candidates")
            required.add(Path(annotation))
        pinned = {_pin(self.root, p).resolve() for p in config.get("source_pins", [])}
        if not {p.resolve() for p in required}.issubset(pinned):
            raise ValueError("Runtime raw source pins missing")
        self._require_sources([history_module])
        self.history = history_module.build_history_index(raw_dir, self.metadata, fit_cutoff="2021-08-09",
            schema_item_ids=self.metadata["item_id_to_token"].keys())
        self.runtime_provenance["history"] = self.history.provenance
        if config.get("personal_candidates"):
            import kion_personal_candidates as personal
            self._require_sources([personal])
            personal_metadata = dict(self.metadata, series_annotation_path=annotation) if self.scoped else self.metadata
            self.personal = personal.build_index(raw_dir, personal_metadata, history_index=self.history,
                text_matrices=self.text_features, text_missing=self.text_missing,
                representation_manifest={k: v["signature"] for k, v in self.text_manifests.items()})
            self._columns["user"] += ["personal." + c for c in self.personal.user_columns]
            for field, values in sorted(self.text_features.items()):
                self._columns["user"] += [f"text_profile.{field}.{i}" for i in range(values.shape[1])]
                self._columns["user"] += [f"text_profile.{field}.{c}" for c in ("known_n", "mean_norm", "valid", "zero_norm")]
            self._columns["candidate"] += ["personal." + c for c in self.personal.columns]
            self.runtime_provenance["personal_candidates"] = self.personal.provenance
            self.materialized_families.add("personal_candidates")
        if config.get("audience"):
            import kion_audience_features as audience
            self._require_sources([audience])
            kwargs = dict(config["audience"])
            backend = kwargs.pop("backend", "core")
            fast_options = kwargs.pop("fast", {})
            fast_release = kwargs.pop("fast_release", None)
            source_alias = kwargs.pop("runtime_source_alias", None)
            if backend not in ("core", "fast-exact") or backend == "core" and (fast_options or fast_release):
                raise ValueError("Explicit audience backend/options mismatch")
            if "index_dir" in kwargs:
                directory = _path(self.root, kwargs.pop("index_dir"))
                path = _pin(self.root, kwargs.pop("manifest"))
                accepted_builder = kwargs.pop("accepted_builder_sha256", None)
                builder_source = kwargs.pop("builder_source", None)
                if kwargs or path.resolve() != (directory / "manifest.json").resolve():
                    raise ValueError("Persisted audience manifest directory binding mismatch")
                self.audience = audience.AudienceIndex.load(directory, mmap_mode="r", verify_hashes=True)
                builder = self.audience.source_pins.get("builder", {})
                if (not isinstance(accepted_builder, str) or len(accepted_builder) != 64
                        or any(c not in "0123456789abcdef" for c in accepted_builder)
                        or builder.get("sha256") != accepted_builder):
                    raise ValueError("Persisted audience historical builder is not explicitly admitted")
                current_loader = file_pin(audience.__file__)
                recorded_path = _path(self.root, builder.get("path", "")).resolve()
                if recorded_path != Path(audience.__file__).resolve() or source_alias is not None:
                    if (not isinstance(source_alias, dict)
                            or set(source_alias) - {"recorded_builder_path", "canonical_loader", "runtime_loader", "fast_loader"}
                            or {"recorded_builder_path", "canonical_loader", "runtime_loader"} - set(source_alias)
                            or "fast_loader" in source_alias and backend != "fast-exact"):
                        raise ValueError("Audience loader relocation requires an explicit runtime_source_alias")
                    canonical_source, runtime_source = source_alias["canonical_loader"], source_alias["runtime_loader"]
                    if (_path(self.root, source_alias["recorded_builder_path"]).resolve() != recorded_path
                            or _pin(self.root, canonical_source).resolve() != recorded_path
                            or _pin(self.root, runtime_source).resolve() != Path(audience.__file__).resolve()
                            or canonical_source["sha256"] != current_loader["sha256"]
                            or runtime_source["sha256"] != current_loader["sha256"]):
                        raise ValueError("Audience runtime_source_alias canonical/runtime source identity mismatch")
                if accepted_builder != current_loader["sha256"]:
                    if not builder_source or builder_source.get("sha256") != accepted_builder:
                        raise ValueError("Historical audience builder requires an exact frozen source receipt")
                    _pin(self.root, builder_source)
                elif builder_source is not None:
                    if builder_source.get("sha256") != accepted_builder:
                        raise ValueError("Frozen audience builder SHA differs from accepted builder")
                    _pin(self.root, builder_source)
                self.runtime_provenance["audience_storage"] = {"manifest_sha256": self.manifest["runtime"]["audience"]["manifest"]["sha256"],
                    "accepted_historical_builder": copy.deepcopy(builder),
                    "builder_source": copy.deepcopy(builder_source), "current_loader": current_loader,
                    "runtime_source_alias": copy.deepcopy(source_alias)}
                if not self.partial and self.audience.provenance.get("source_scope") != "full_source":
                    raise ValueError("Full package rejects persisted audience pilot index")
                for name, source in (("items", raw_dir / "items.csv"), ("users", raw_dir / "users.csv"),
                                     ("interactions", raw_dir / "interactions.csv"), ("metadata", _path(self.root, self.manifest["data"]["path"]) / "metadata.json")):
                    pin = self.audience.source_pins.get(name, {})
                    if _pin(self.root, pin).resolve() != source.resolve():
                        raise ValueError("Persisted audience raw source binding mismatch: " + name)
            else:
                if source_alias is not None:
                    raise ValueError("Audience source relocation applies only to an admitted persisted index")
                if not self.partial and kwargs.get("max_source_rows") is not None:
                    raise ValueError("Full audience index cannot use a source-row pilot")
                self.audience = audience.AudienceIndex.from_csv(raw_dir, self.metadata, fit_cutoff="2021-08-09", **kwargs)
            if self.audience.fit_cutoff != FIT_CUTOFF:
                raise ValueError("Audience SFT role cap mismatch")
            self.audience_core = self.audience
            if backend == "fast-exact":
                import kion_audience_fast as fast
                self._require_sources([fast])
                if not self.partial and not fast_release:
                    raise ValueError("Full fast audience needs the independently accepted release receipt")
                if source_alias and "fast_loader" in source_alias and not fast_release:
                    raise ValueError("Fast loader relocation requires the exact accepted fast release")
                if fast_release:
                    release = json.loads(_pin(self.root, fast_release).read_text(encoding="utf-8"))
                    if release.get("contract") != fast.CONTRACT or release.get("columns") != self.audience.columns:
                        raise ValueError("Fast audience release contract/schema mismatch")
                    for name in ("fast", "core", "index_manifest", "benchmark", "historical_build_snapshot"):
                        _pin(self.root, release["source_pins"][name])
                    for name, actual in (("fast", file_pin(fast.__file__)), ("core", file_pin(audience.__file__))):
                        recorded = release["source_pins"][name]
                        pair = None if not source_alias else (source_alias.get("fast_loader") if name == "fast" else {
                            "canonical": source_alias["canonical_loader"], "runtime": source_alias["runtime_loader"]})
                        relocated = _path(self.root, recorded["path"]).resolve() != Path(actual["path"])
                        if recorded["sha256"] != actual["sha256"]:
                            raise ValueError("Fast release does not match the actual backend/core identity")
                        if relocated or pair is not None:
                            if (not isinstance(pair, dict) or set(pair) != {"canonical", "runtime"}
                                    or _pin(self.root, pair["canonical"]).resolve() != _path(self.root, recorded["path"]).resolve()
                                    or _pin(self.root, pair["runtime"]).resolve() != Path(actual["path"])
                                    or pair["canonical"]["sha256"] != actual["sha256"] or pair["runtime"]["sha256"] != actual["sha256"]):
                                raise ValueError("Fast release runtime alias does not bind the actual backend/core identity")
                    chosen_index = config["audience"].get("manifest")
                    recorded = release["source_pins"]["index_manifest"]
                    if (not chosen_index or recorded["sha256"] != chosen_index["sha256"]
                            or _path(self.root, recorded["path"]).resolve() != _path(self.root, chosen_index["path"]).resolve()):
                        raise ValueError("Fast release does not match the selected audience index")
                if "pair_mode" not in fast_options:
                    raise ValueError("Fast audience requires explicit pair_mode; no fallback")
                for name, maximum in (("max_pair_cache", 250000), ("max_gram_cells", 2000000), ("max_pair_work", 50000000)):
                    if name in fast_options:
                        _positive(fast_options[name], "fast." + name, maximum)
                self.audience = fast.FastAudienceIndex(self.audience_core, **fast_options)
                self.runtime_provenance["audience_fast"] = {**self.audience.provenance, "release": fast_release}
            self._columns["candidate"] += list(self.audience.columns)
            self.runtime_provenance["audience"] = self.audience.provenance
            self.materialized_families.add("audience")
        if config.get("people_dynamics"):
            import kion_people_dynamics as people
            self._require_sources([people])
            kwargs = dict(config["people_dynamics"])
            backend = kwargs.pop("backend", "live-index")
            if backend not in ("live-index", "date-bank"):
                raise ValueError("Explicit supported people backend required")
            persisted = "index_dir" in kwargs or backend == "date-bank"
            if backend == "date-bank":
                import kion_people_banks as banks
                import hm_job
                self._require_sources([banks, hm_job])
                fast_source = Path(__file__).with_name("kion_people_dynamics_fast.py").resolve()
                if fast_source not in {_path(self.root, p["path"]).resolve() for p in self.manifest["source_pins"]}:
                    raise ValueError("People bank fast builder source pin missing")
                path = _pin(self.root, kwargs.pop("manifest"))
                identity = kwargs.pop("identity_sha256")
                cache_days = _positive(kwargs.pop("cache_days", 1), "people.bank_cache_days", 2)
                if kwargs:
                    raise ValueError("Unknown people date-bank options")
                self.people_dynamics = banks.PeopleBankReader(path, expected_identity_sha256=identity, cache_days=cache_days)
                banks.verify_plan_inputs(self.people_dynamics.manifest)
                native = self.people_dynamics.manifest
                origins = set(map(int, native["origin_inventory"]["origin_ordinals"]))
                required_origins = set(map(int, self._origins["train"])) | set(map(int, self._origins["val"]))
                if (native["fit_role"] != "sft" or native["fit_cutoff_exclusive"] != "2021-08-09"
                        or native["role_cap_exclusive"] != "2021-08-09"
                        or native["index"]["metadata_sha256"] != self.metadata_hash
                        or self.people_dynamics.vocab_size != self.metadata["vocab_size"]
                        or self.people_dynamics.mapping_sha256 != self.vocabulary_hash
                        or not required_origins.issubset(origins)):
                    raise ValueError("People bank role/metadata/vocabulary/exact-origin binding mismatch")
                self.runtime_provenance["people_bank"] = {"manifest": config["people_dynamics"]["manifest"],
                    "identity_sha256": identity, "cache_days": cache_days}
            elif "index_dir" in kwargs:
                directory = _path(self.root, kwargs.pop("index_dir"))
                path = _pin(self.root, kwargs.pop("manifest"))
                if path.resolve() != (directory / "manifest.json").resolve():
                    raise ValueError("Persisted people manifest directory binding mismatch")
                if kwargs:
                    raise ValueError("Unknown persisted people index options")
                self.people_dynamics = people.PeopleDynamicsIndex.load(directory)
            else:
                if not self.partial and kwargs.get("max_rows") is not None:
                    raise ValueError("Full people index cannot use a source-row pilot")
                self.people_dynamics = people.build(raw_dir, self.metadata, role="sft", fit_cutoff="2021-08-09", **kwargs)
            if (self.people_dynamics.fit_cutoff != FIT_CUTOFF
                    or backend != "date-bank" and self.people_dynamics.mapping != self.metadata["item_id_to_token"]):
                raise ValueError("People dynamics mapping/role cap mismatch")
            if not self.partial and self.people_dynamics.provenance.get("pilot_only") is not False:
                raise ValueError("Full package rejects persisted people pilot index")
            for name in ("items", "interactions"):
                expected = next(p["sha256"] for p in config["source_pins"]
                                if _path(self.root, p["path"]).resolve() == (raw_dir / (name + ".csv")).resolve())
                actual = self.people_dynamics.provenance.get(name + "_sha256")
                if (name == "items" or persisted) and actual != expected:
                    raise ValueError("Persisted people raw source binding mismatch: " + name)
            self._columns["candidate"] += list(self.people_dynamics.columns)
            self.runtime_provenance["people_dynamics"] = self.people_dynamics.provenance
            self.materialized_families.add("people_dynamics")
        if config.get("local_dynamic"):
            import kion_pack_dynamic as dynamic
            self._require_sources([dynamic])
            self.dynamic = dynamic.build_index(raw_dir, self.metadata, fit_cutoff="2021-08-09")
            self._columns["candidate"] += ["local_dynamic." + c for c in self.dynamic.columns]
            self.runtime_provenance["local_dynamic"] = self.dynamic.provenance
            self.materialized_families.add("local_dynamic")
        if config.get("related_interest"):
            import kion_related_features as related
            self._require_sources([related])
            params = dict(config["related_interest"])
            directory = _path(self.root, params.pop("index_dir"))
            receipt = params.pop("manifest")
            historical_builder = params.pop("audience_builder_source", None) if self.scoped else None
            path = _pin(self.root, receipt)
            if params or path.resolve() != (directory / "manifest.json").resolve():
                raise ValueError("Related index manifest directory binding mismatch")
            self.related = related.RelatedInterestIndex.load(directory, verify_hashes=True, mmap_mode="r")
            if (self.related.mapping != self.metadata["item_id_to_token"] or self.related.fit_cutoff != FIT_CUTOFF
                    or self.related.role != "sft"):
                raise ValueError("Related index mapping/typed SFT role cap mismatch")
            if not self.partial and (self.related.provenance.get("pilot_only") is not False
                    or self.related.provenance.get("source_scope") != "full_source"):
                raise ValueError("Full package rejects a related-interest source pilot")
            for collection in ("source_pins", "audience_source_pins"):
                for name, source in self.related.provenance.get(collection, {}).items():
                    if collection == "audience_source_pins" and name == "builder" and historical_builder is not None:
                        self.runtime_provenance["related_historical_builder"] = _related_historical_builder(self.root, source, historical_builder)
                    else:
                        _pin(self.root, source)
            for collection, name, actual in (("source_pins", "items", raw_dir / "items.csv"),
                    ("audience_source_pins", "items", raw_dir / "items.csv"),
                    ("audience_source_pins", "interactions", raw_dir / "interactions.csv"),
                    ("audience_source_pins", "metadata", _path(self.root, self.manifest["data"]["path"]) / "metadata.json")):
                source = self.related.provenance.get(collection, {}).get(name)
                if source is None:
                    if not self.partial:
                        raise ValueError("Related runtime corpus source receipt missing: " + name)
                    continue
                if _path(self.root, source["path"]).resolve() != actual.resolve() or source["sha256"] != text_module.file_hash(actual):
                    raise ValueError("Related index belongs to a different runtime corpus: " + name)
            self._columns["candidate"] += list(self.related.columns)
            self.runtime_provenance["related_interest"] = self.related.provenance
            self.materialized_families.add("related_interest")
        if config.get("franchise_progress"):
            import kion_franchise_progress as franchise
            self._require_sources([franchise])
            params = config["franchise_progress"]
            if set(params) - {"graph", "temporal_mode", "order_kinds", "scope_policies", "order_kind", "scope_policy"}:
                raise ValueError("Unknown franchise progress options")
            path = _pin(self.root, params["graph"])
            self.graph = json.loads(path.read_text(encoding="utf-8"))
            self.graph_mode = params.get("temporal_mode")
            if self.graph_mode not in ("historical_as_of", "retrospective_current"):
                raise ValueError("Explicit franchise temporal mode required")
            self.graph_legacy_prefix = "order_kinds" not in params
            if self.graph_legacy_prefix:
                if not self.partial or "scope_policies" in params:
                    raise ValueError("Full franchise progress requires both explicit typed order_kinds")
                kind = params.get("order_kind")
                if kind is not None and kind not in FRANCHISE_ORDER_KINDS:
                    raise ValueError("Unsupported franchise order kind")
                self.graph_order_kinds = [kind]
                self.graph_scope_policies = {kind: params.get("scope_policy")}
            else:
                kinds = params["order_kinds"]
                if (not isinstance(kinds, list) or not kinds or any(not isinstance(kind, str) for kind in kinds) or len(set(kinds)) != len(kinds)
                        or any(kind not in FRANCHISE_ORDER_KINDS for kind in kinds)
                        or "order_kind" in params or "scope_policy" in params
                        or not self.partial and set(kinds) != set(FRANCHISE_ORDER_KINDS)):
                    raise ValueError("Franchise order_kinds must be distinct supported typed branches")
                policies = params.get("scope_policies", {kind: None for kind in kinds})
                if (not isinstance(policies, dict) or set(policies) != set(kinds)
                        or any(policy is not None and not isinstance(policy, dict) for policy in policies.values())):
                    raise ValueError("Franchise scope_policies must bind each typed branch separately")
                self.graph_order_kinds, self.graph_scope_policies = list(kinds), policies
            for kind, policy in self.graph_scope_policies.items():
                if policy is not None and not isinstance(policy, dict):
                    raise ValueError("Franchise scope policy must be a work-to-scope mapping")
                for work, scope_id in (policy or {}).items():
                    scope = self.graph["scopes"].get(scope_id) if isinstance(scope_id, str) else None
                    if (work not in self.graph["works"] or not isinstance(scope, dict)
                            or work not in scope["work_ids"] or kind is not None and scope["order_kind"] != kind):
                        raise ValueError("Franchise scope policy must bind an existing work and a containing scope of the same order kind")
            self.prepared_graph = franchise.prepare_graph(self.graph, self.metadata,
                order_kinds=self.graph_order_kinds, scope_policies=self.graph_scope_policies)
            self.graph = self.prepared_graph.graph
            prefixes = ["franchise." if self.graph_legacy_prefix else "franchise." + kind + "." for kind in self.graph_order_kinds]
            for prefix in prefixes:
                self._columns["candidate"] += [prefix + c for c in franchise.COLUMNS]
            self.runtime_provenance["franchise_progress"] = {**params, "order_kinds": self.graph_order_kinds,
                "scope_policies": {("legacy_default" if kind is None else kind): policy for kind, policy in self.graph_scope_policies.items()}, "feature_namespaces": prefixes,
                "typed_orders_merged": False}
            self.materialized_families.add("franchise_progress")
        # Check raw inputs again after every build, so a race cannot be receipted.
        for pin in config.get("source_pins", []):
            _pin(self.root, pin)

    def contract(self):
        return copy.deepcopy(self._contract)

    def origin_keys(self, split):
        """Stable immutable exact origins used by the resumable epoch scheduler."""
        if split not in ("train", "val"):
            raise ValueError("Only train/val origins are admitted")
        return self._origins[split]

    def _load_diagnostics(self, declarations):
        result = {}
        self._diagnostic_stores = {}
        for name, receipt in declarations.items():
            path = _pin(self.root, receipt)
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value.get("contract") in ("kion-user-diagnostics-v1", "kion-completion-event-store-v1")
                    and path.resolve() != (path.parent / "manifest.json").resolve()):
                raise ValueError("Diagnostic receipt must pin the actually opened native manifest.json")
            if value.get("contract") == "kion-user-diagnostics-v1":
                import kion_user_diagnostics as diagnostics
                self._require_sources([diagnostics])
                store = diagnostics.load(path.parent, expected_metadata_sha256=self.metadata_hash,
                    expected_prepared_pins={split + ".npz": self.manifest["data"]["files"][split + ".npz"] for split in ("train", "val")},
                    expected_row_keys_sha256=self.manifest["row_keys_sha256"], verify_hashes=True)
                self._diagnostic_stores[name] = store
                required = {"personal.activity", "personal.typed_completion", "personal.window_completion", "personal.age_rating_profile"}
                if (value.get("clock") != "strict_h63_prior" or value.get("fit_cutoff") != "2021-08-09"
                        or value.get("storage") != "normalized_event_facts_exact_query_mapping"
                        or set(value.get("families", [])) != required
                        or not self.partial and value.get("coverage") != "full_rows"):
                    raise ValueError("User diagnostics require exact four-family normalized full query coverage")
                result[name] = {"manifest": receipt, "coverage": value["coverage"], "model_features": False,
                    "storage": value["storage"], "families": sorted(required)}
                continue
            if value.get("contract") == "kion-completion-event-store-v1":
                import kion_completion_event_store as events
                self._require_sources([events])
                store = events.load(path.parent, expected_metadata_sha256=self.metadata_hash,
                    expected_prepared_pins={split + ".npz": self.manifest["data"]["files"][split + ".npz"] for split in ("train", "val")},
                    expected_row_keys_sha256=self.manifest["row_keys_sha256"], verify_hashes=True)
                self._diagnostic_stores[name] = store
                coverage = store.manifest.get("coverage", {})
                if not self.partial and (coverage.get("raw_events") != "complete_before_role_cap"
                        or coverage.get("query_mapping") != "complete_all_prepared_rows"):
                    raise ValueError("Normalized completion event store lacks full corpus/query join proof")
                result[name] = {"manifest": receipt, "coverage": coverage, "model_features": False,
                    "storage": "normalized_events_plus_deterministic_H63_join"}
                continue
            if (value.get("contract") != "kion-completion-diagnostics-v1"
                    or value.get("fit_cutoff") != "2021-08-09" or value.get("clock") != "strict_h63_prior_loo"
                    or value.get("row_keys_sha256") != self.manifest["row_keys_sha256"]
                    or value.get("metadata_sha256") != self.metadata_hash):
                raise ValueError("Completion diagnostics row/source/clock binding mismatch")
            if not self.partial and value.get("coverage") != "full_rows":
                raise ValueError("Full completion diagnostics require explicit full row coverage")
            for source in value.get("source_pins", []):
                _pin(self.root, source)
            if not value.get("source_pins"):
                raise ValueError("Completion diagnostics source pins missing")
            _names(value.get("columns"), "diagnostic")
            if not value["columns"]:
                raise ValueError("Completion diagnostics need named diagnostic-only fields")
            for split in ("train", "val"):
                declaration = value["splits"][split]
                array = np.load(_pin(self.root, declaration), mmap_mode="r", allow_pickle=False)
                try:
                    if (list(array.shape) != declaration.get("shape") or array.dtype.kind not in "fiu"
                            or str(array.dtype) != declaration.get("dtype") or array.ndim != 2
                            or array.shape[1] != len(value["columns"])
                            or value.get("coverage") == "full_rows" and len(array) != len(self.splits[split]["y"])):
                        raise ValueError("Completion diagnostics shape/row coverage mismatch")
                    for begin in range(0, len(array), 4096):
                        if not np.isfinite(array[begin:begin + 4096]).all():
                            raise ValueError("Nonfinite completion diagnostic fields")
                finally:
                    if isinstance(array, np.memmap):
                        array._mmap.close()
            result[name] = {"manifest": receipt, "coverage": value["coverage"], "columns": value["columns"], "model_features": False}
        return result

    def _matrix(self, declaration):
        return _matrix(self.root, declaration, self.metadata["vocab_size"], self.metadata_hash, self.vocabulary_hash)

    def _rows(self, rows, split):
        if split not in ("train", "val"):
            raise ValueError("Only train/val are admitted")
        rows = np.asarray(rows)
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
                or np.any(rows >= len(self.splits[split]["y"])) or len(rows) > self.max_batch_rows):
            raise ValueError("Invalid/budget-exceeding provider row indices")
        return rows.astype(np.int64, copy=False)

    def _origin(self, split, row):
        return int(self._origins[split][row])

    def _query(self, split, row):
        data, origin = self.splits[split], self._origin(split, row)
        x, length, uid = data["x"][row], int(data["lengths"][row]), int(data["user_id"][row])
        key = (uid, origin, length, x.tobytes())
        if key in self._profiles:
            self._profiles.move_to_end(key)
            return self._profiles[key]
        state = {"origin": origin, "uid": uid, "x": x[None], "lengths": np.array([length])}
        blocks = []
        if self.history is not None:
            # Also prevents a prepared same-day/future token from entering any cross.
            idx, mask, _, _, _ = self.history._history(np.array([uid]), [origin], [origin], x[None], np.array([length]))
            state["raw_history"] = [{"item_id": int(self.history.items[i]),
                "date": date.fromordinal(int(self.history.days[i])).isoformat(), "token": int(self.history.tokens[i])}
                for i in idx[0, mask[0]]]
        if self.personal is not None:
            profile = self.personal.prepare_queries(np.array([uid]), [origin], x=x[None], lengths=np.array([length]),
                                                    history_cutoffs=[origin])
            state["personal"] = profile
            blocks.append(profile.user_features[0])
            for field in sorted(self.text_features):
                text = profile.text[field]
                blocks += [text["profile"][0], np.array([text[c][0] for c in ("n_known", "mean_norm", "valid", "zero_norm")])]
        state["user"] = np.concatenate(blocks).astype(np.float32) if blocks else np.zeros(0, np.float32)
        self._profiles[key] = state
        while len(self._profiles) > self.profile_cache_rows:
            self._profiles.popitem(last=False)
        return state

    def batch(self, rows, split):
        rows = self._rows(rows, split)
        data = self.splits[split]
        extra = [self._query(split, row)["user"] for row in rows]
        width = len(self.feature_columns["user"]) - data["user_features"].shape[1]
        tail = np.stack(extra) if len(rows) else np.zeros((0, width), np.float32)
        user = np.concatenate((data["user_features"][rows].astype(np.float32), tail), axis=1)
        if not np.isfinite(user).all():
            raise ValueError("Nonfinite composed user features")
        self.last_user_numeric_audit = self._numeric_audit(user, "user")
        self._transform_numeric(user, "user")
        self.last_user_numeric_audit["after"] = self._numeric_audit(user, "user")["before"]
        self._validate_model_range(user, "user")
        return data["x"][rows], data["lengths"][rows], data["y"][rows], user

    def _external_bank(self, declaration, origin):
        key = (declaration["family"], tuple(declaration["columns"]), origin)
        if key not in self._banks:
            receipt = declaration["banks"].get(str(origin))
            if receipt is None:
                raise ValueError("Exact strict-prefix external bank missing for origin: " + str(origin))
            self._banks[key] = self._matrix(receipt)
            while len(self._banks) > self.bank_cache_days * max(1, len(self.date_features)):
                _, old = self._banks.popitem(last=False)
                if isinstance(old, np.memmap):
                    old._mmap.close()
        self._banks.move_to_end(key)
        return self._banks[key]

    def _fill_personal_crosses(self, queries, output):
        if self.personal is None or not output.size:
            return
        profiles = self.personal.concatenate_profiles([state["personal"] for _, state, _, _ in queries])
        width = max(len(candidates) for _, _, candidates, _ in queries)
        # Row-local unique candidates padded by a real token of that same row;
        # the inverse below discards padding. Never form batch x candidate-union.
        candidates = np.empty((len(queries), width), np.int64)
        for row, (_, _, unique, _) in enumerate(queries):
            candidates[row, :len(unique)] = unique
            candidates[row, len(unique):] = unique[0]
        values = self.personal.candidate_features(profiles, candidates)
        if values.shape != (*candidates.shape, len(self.personal.columns)):
            raise ValueError("Batched personal feature shape mismatch")
        target = [self.feature_columns["candidate"].index("personal." + name) for name in self.personal.columns]
        for row, (pos, _, _, inverse) in enumerate(queries):
            output[pos][:, target] = values[row, inverse]

    def _fill_batched_item_crosses(self, queries, output):
        """Native bounded point batches, preserving row-local dedup/scatter."""
        indices = ((self.dynamic, "local_dynamic."), (self.related, ""))
        if not output.size or not any(index is not None for index, _ in indices):
            return
        counts = [len(candidates) for _, _, candidates, _ in queries]
        tokens = np.concatenate([candidates for _, _, candidates, _ in queries])
        origins = np.repeat([state["origin"] for _, state, _, _ in queries], counts)
        users = np.repeat([state["uid"] for _, state, _, _ in queries], counts)
        inverse = np.empty(output.shape[:2], np.int64)
        offset = 0
        for pos, _, candidates, row_inverse in queries:
            inverse[pos] = row_inverse + offset
            offset += len(candidates)
        groups = [np.flatnonzero(origins == origin) for origin in np.unique(origins)]
        columns = {name: i for i, name in enumerate(self.feature_columns["candidate"])}
        for index, prefix in indices:
            if index is None:
                continue
            width = len(index.columns)
            # Related's native API admits at most 4096 rows. The same bound also
            # keeps dynamic scratch arrays independent of the full vocabulary.
            chunk = min(4096, self.max_pair_cells, self.max_pair_bytes // (4 * width))
            values = np.empty((len(tokens), width), np.float32)
            for group in groups:
                for begin in range(0, len(group), chunk):
                    take = group[begin:begin + chunk]
                    block = index.features(tokens[take], origins[take], history_cutoffs=origins[take], user_ids=users[take])
                    if block.shape != (len(take), width):
                        raise ValueError("Batched item-cross feature shape mismatch")
                    values[take] = block
            target = [columns[prefix + name] for name in index.columns]
            output.reshape(-1, output.shape[-1])[:, target] = values[inverse.ravel()]

    def candidate_features(self, rows, ids, split):
        rows = self._rows(rows, split)
        ids = np.asarray(ids)
        if (ids.ndim != 2 or len(ids) != len(rows) or ids.dtype.kind not in "iu"
                or ids.shape[1] > 4096 or ids.size > self.max_pair_cells
                or ids.size * len(self.feature_columns["candidate"]) * 4 > self.max_pair_bytes
                or np.any(ids < 3) or np.any(ids >= self.metadata["vocab_size"])):
            raise ValueError("Candidates require bounded B x C real mapped integer tokens")
        output = np.zeros((*ids.shape, len(self.feature_columns["candidate"])), np.float32)
        # Audience owns one date snapshot; group rows within this call to avoid
        # rebuilding that full snapshot once per alternating-origin row.
        order = sorted(range(len(rows)), key=lambda pos: self._origin(split, rows[pos]))
        queries = []
        for pos in order:
            row = rows[pos]
            state = self._query(split, row)
            candidates, inverse = np.unique(ids[pos], return_inverse=True)
            queries.append((pos, state, candidates, inverse))
            origin, uid = state["origin"], state["uid"]
            n = len(candidates)
            blocks = []
            if self.personal is not None:
                blocks.append(np.zeros((n, len(self.personal.columns)), np.float32))
            if self.audience is not None:
                # Fill this block in one flattened date-grouped call below.
                blocks.append(np.zeros((n, len(self.audience.columns)), np.float32))
            for index in (self.people_dynamics, self.dynamic, self.related):
                if index is not None:
                    if index is self.dynamic or index is self.related:
                        blocks.append(np.zeros((n, len(index.columns)), np.float32))
                    else:
                        blocks.append(index.features(candidates, np.full(n, origin), history_cutoffs=np.full(n, origin),
                                                     user_ids=np.full(n, uid)))
            if self.graph is not None:
                raw_candidates = [self._raw_item_ids_by_token[int(t)] for t in candidates]
                when = date.fromordinal(origin).isoformat()
                for kind in self.graph_order_kinds:
                    blocks.append(self.prepared_graph.candidate_progress(state["raw_history"], raw_candidates,
                        when, when, temporal_mode=self.graph_mode, order_kind=kind)["values"])
            for declaration in self.date_features:
                blocks.append(self._external_bank(declaration, origin)[candidates])
            if blocks:
                output[pos] = np.concatenate(blocks, axis=1)[inverse]
        self._fill_personal_crosses(queries, output)
        self._fill_batched_item_crosses(queries, output)
        if self.audience is not None and ids.size:
            width = ids.shape[1]
            row_positions = np.repeat(np.arange(len(rows)), width)
            origins = self._origins[split][rows][row_positions]
            tokens = ids.ravel()
            audience_columns = [self.feature_columns["candidate"].index(name) for name in self.audience.columns]
            flat = output.reshape(-1, output.shape[-1])
            data = self.splits[split]
            for origin in np.unique(origins):
                selected = np.flatnonzero(origins == origin)
                for begin in range(0, len(selected), 4096):
                    take = selected[begin:begin + 4096]
                    query_rows = rows[row_positions[take]]
                    block = self.audience.features(tokens[take], np.full(len(take), origin),
                        user_ids=data["user_id"][query_rows], role="sft_train" if split == "train" else "sft_val",
                        x=data["x"][query_rows], lengths=data["lengths"][query_rows], history_cutoffs=np.full(len(take), origin))
                    flat[np.ix_(take, audience_columns)] = block
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite composed candidate features")
        self.last_numeric_audit = self._numeric_audit(output, "candidate")
        self._transform_numeric(output, "candidate")
        self.last_numeric_audit["after"] = self._numeric_audit(output, "candidate")["before"]
        self._validate_model_range(output, "candidate")
        return output

    def _numeric_audit(self, values, scope):
        maxima = np.max(np.abs(values), axis=tuple(range(values.ndim - 1))) if values.size else np.zeros(values.shape[-1])
        order = np.argsort(-maxima, kind="stable")[:20]
        return {"before": [{"column": self.feature_columns[scope][int(i)], "max_abs": float(maxima[i])} for i in order]}

    def _transform_numeric(self, values, scope):
        epoch = date.fromisoformat(NUMERIC_TRANSFORMS["ordinal_epoch"]).toordinal()
        for column, operation in self._numeric_indices[scope]:
            raw = values[..., column]
            if operation == "log1p_nonnegative":
                if np.any(raw < 0):
                    raise ValueError("Negative value in declared nonnegative feature: " + self.feature_columns[scope][column])
                values[..., column] = np.log1p(raw)
            else:
                # A zero denotes unavailable, independently from explicit masks.
                values[..., column] = np.where(raw != 0, (raw - epoch) / NUMERIC_TRANSFORMS["ordinal_divisor"], 0)

    def _validate_model_range(self, values, scope):
        for begin in range(0, len(values), 4096):
            block = values[begin:begin + 4096]
            invalid = ~np.isfinite(block) | (np.abs(block) > 65504)
            if invalid.any():
                index = np.argwhere(invalid)[0, -1]
                raise ValueError("Feature is outside finite FP16 model range: " + self.feature_columns[scope][index])

    def close(self):
        for name in ("audience_core", "related", "people_dynamics"):
            value = getattr(self, name, None)
            if value is not None and hasattr(value, "close"):
                value.close()
        for value in getattr(self, "_diagnostic_stores", {}).values():
            value.close()
        for values in getattr(self, "_maps", []) + list(getattr(self, "_banks", {}).values()):
            if isinstance(values, np.memmap) and not values._mmap.closed:
                values._mmap.close()
        self._maps = []
        self._banks = OrderedDict()
        self._profiles = OrderedDict()


def load_training_provider(args):
    return CompleteProvider.load(args.data, args.feature_dir,
        allow_partial_integration=getattr(args, "allow_partial_integration", False),
        scope_decision=getattr(args, "scope_decision", None))
