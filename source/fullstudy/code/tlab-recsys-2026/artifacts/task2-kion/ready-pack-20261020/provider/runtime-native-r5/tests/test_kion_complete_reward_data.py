"""Tiny native RM components; no production source build, fit or GPU."""
import copy
import csv
from datetime import date
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location("complete_reward_tested", Path(__file__).parents[1] / "scripts/kion_complete_reward_data.py")
complete = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(complete)
import kion_pack_history as history
import kion_pack_dynamic as dynamic
import kion_text_embeddings as text
import kion_people_embeddings as people
import kion_franchise_progress as franchise
import kion_audience_features as audience_module
import kion_completion_diagnostics as diagnostics
import kion_completion_event_store as events
import kion_people_dynamics as people_core
import kion_user_diagnostics as user_diagnostics_module
import kion_related_features as related_module


def day(value):
    return date.fromisoformat(value).toordinal()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Encoder:
    spec = {"model_id": "fixture-only", "revision": "unit-v1", "dimension": 384,
            "production_encoder": False, "compute_dtype": "float32"}

    def encode(self, inputs):
        values = np.zeros((len(inputs), 384), np.float32)
        for row, value in enumerate(inputs):
            values[row, sum(map(ord, value)) % 7] = 1
        return values, np.full(len(inputs), 5, np.int32)


class TinyLocal:
    """Replace only expensive static assembly; temporal algorithms are native."""
    def __init__(self, raw_dir, metadata_path, basic_npz, *, fit_cutoff, include_relative):
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.fit_cutoff = fit_cutoff
        self.history = history.build_history_index(raw_dir, metadata, fit_cutoff=fit_cutoff,
            include_relative=True, schema_item_ids=metadata["item_id_to_token"].keys())
        self.dynamic = dynamic.build_index(raw_dir, metadata, fit_cutoff=fit_cutoff)
        self.item_features = np.zeros((metadata["vocab_size"], 199), np.float32)
        self.item_features[3:, :197] = np.arange(197, dtype=np.float32) / 200
        self.item_features[3:, 197:] = 1
        self.item_columns = [f"original.{i}" for i in range(197)] + ["legacy.story_flag", "legacy.format_flag"]
        self.user_columns = ["user.fixture_demography"] + ["history." + c for c in self.history.columns]
        self.user_columns += ["dynamic_history." + c for c in self.dynamic.history_columns]
        self.provenance = {"source_pins": {name: complete.pin(Path(raw_dir) / (name + ".csv"))
            for name in ("items", "users", "interactions")}, "history": self.history.provenance,
            "dynamic": self.dynamic.provenance}

    def user_features(self, users, origins, x, lengths, history_cutoffs=None, dynamic_bank=None):
        state = self.history.features(users, origins, x=x, lengths=lengths, history_cutoffs=history_cutoffs)
        dyn = self.dynamic.history_features(x, lengths, origins, history_cutoffs=history_cutoffs) if dynamic_bank is None else self.dynamic.history_features_from_bank(x, lengths, dynamic_bank)
        return np.concatenate(((np.asarray(users) % 2).astype(np.float32)[:, None], state, dyn), axis=1)


def write_native_basic_fixture(raw, metadata_path, path, metadata):
    """Small hash-bound seven-group cache with the real original197 layout."""
    import kion_pack_static as static
    fixed = {"age": static.local.AGE_CATEGORIES, "sex": static.sex.SEX_CATEGORIES,
        "countries": [static.UNKNOWN], "content_type": static.basic.CONTENT_TYPE_CATEGORIES,
        "release_year": static.basic.YEAR_COLUMNS, "age_rating": static.basic.AGE_RATING_CATEGORIES}
    n_genres = 197 - sum(len(fixed[n]) for n in ("countries", "content_type", "release_year", "age_rating"))
    fixed["genres"] = [static.UNKNOWN] + [f"genre-{i:03d}" for i in range(n_genres - 1)]
    groups, offset = {}, {"user": 0, "item": 0}
    values = np.zeros((metadata["vocab_size"], 197), np.float32)
    for name in static.basic.GROUP_NAMES:
        scope = "user" if name in ("age", "sex") else "item"
        start, columns = offset[scope], list(fixed[name])
        groups[name] = dict(scope=scope, start=start, stop=start + len(columns), columns=columns)
        if scope == "item":
            values[3:, start + (1 if name == "release_year" else 0)] = 1
        offset[scope] += len(columns)
    np.savez(path, item_features=values)
    complete.rm.write_json(path.with_name("manifest.json"), {"builder_contract": static.basic.CONTRACT,
        "features": {"sha256": complete.rm.file_hash(path), "arrays": {"item_features": {"sha256": static.local.array_hash(values)}}},
        "groups": groups, "vocabulary": {"sha256": static.local.canonical_hash(metadata["item_id_to_token"])},
        "inputs": {name: complete.pin(source) for name, source in (("metadata", metadata_path),
            ("items", raw / "items.csv"), ("users", raw / "users.csv"))}})


def fixture(root, *, mutations=None, explicit_series=True, native_local=False):
    root.mkdir(parents=True, exist_ok=True)
    raw, data, artifacts = root / "raw", root / "data", root / "artifacts"
    for path in (raw, data, artifacts):
        path.mkdir()
    annotation = artifacts / "series.json"
    complete.rm.write_json(annotation, {"series": []})
    metadata = {"context": 64, "vocab_size": 8, "special_tokens": {"PAD": 0, "BOS": 1, "UNK": 2},
        "item_id_to_token": {str(i): n + 3 for n, i in enumerate((10, 20, 30, 40, 50))},
        "series_annotation_path": annotation.as_posix(),
        "user_categories": {key: ["__UNKNOWN__", "A", "B"] for key in ("age", "income", "sex", "kids_flg")}}
    if native_local:
        import kion_pack_static as static
        annotation = root / "configs/datasets/kion-series-v1.json"
        annotation.parent.mkdir(parents=True)
        metadata.update(format_version=1, user_feature_dim=20,
            split_dates={"train": "<2021-08-09", "val": "2021-08-09..2021-08-15"},
            series_annotation_path=annotation.as_posix(), user_categories={
                "age": static.local.AGE_CATEGORIES,
                "income": [static.UNKNOWN, "income_0_20", "income_150_inf", "income_20_40", "income_40_60", "income_60_90", "income_90_150"],
                "sex": static.sex.PREPARED_SEX_CATEGORIES, "kids_flg": [static.UNKNOWN, "0", "1"]})
    if not explicit_series:
        metadata.pop("series_annotation_path")
    metadata_path = data / "metadata.json"
    complete.rm.write_json(metadata_path, metadata)
    users = np.arange(100, 200, dtype=np.int64)
    selected = complete.rm.cohort(users)
    sel, cal = int(users[selected][0]), int(users[~selected][0])
    records = [(1, 10, "2021-07-01", "0"), (1, 20, "2021-07-02", "100"), (1, 30, "2021-08-01", "80"),
        (2, 20, "2021-07-01", "20"), (2, 30, "2021-07-02", "40"), (2, 10, "2021-08-01", "50"),
        (3, 30, "2021-07-01", ""), (3, 40, "2021-07-02", "70"),
        (5, 20, "2021-07-02", "101"), (5, 40, "2021-08-01", "50"),
        (sel, 10, "2021-07-01", "50"), (sel, 40, "2021-08-02", "40"), (sel, 50, "2021-08-08", "90"),
        (cal, 20, "2021-07-02", "60"), (cal, 50, "2021-08-02", "80"), (cal, 30, "2021-08-08", "30"),
        (99, 50, "2021-08-09", "100")]
    if mutations is not None:
        records = [mutations(row) for row in records]
    write_csv(raw / "interactions.csv", [dict(user_id=u, item_id=i, last_watch_dt=d, watched_pct=y) for u, i, d, y in records])
    cards = {str(i): dict(item_id=str(i), title="Title " + str(i), title_orig="" if i == 40 else "Original " + str(i),
        description="" if i == 30 else "Description " + str(i), content_type="film" if i % 20 else "series",
        release_year="2000" if i < 40 else "2010", genres="A" if i < 30 else "B", countries="X",
        age_rating="16.0", actors="" if i == 30 else "Actor A, Actor B", directors="Director A",
        studios="Studio A", for_kids="0.0", keywords="") for i in (10, 20, 30, 40, 50)}
    write_csv(raw / "items.csv", list(cards.values()))
    profile = dict(age="age_18_24", income="income_20_40", sex="М", kids_flg="0") if native_local else dict(age="A", income="B", sex="A", kids_flg="B")
    write_csv(raw / "users.csv", [dict(user_id=u, **profile)
        for u in sorted({r[0] for r in records})])
    basic_path = artifacts / "basic.npz"
    np.savez(basic_path, item_features=np.zeros((8, 1), np.float32))
    complete.rm.write_json(artifacts / "manifest.json", {"contract": "fixture-only-basic"})
    if native_local:
        write_native_basic_fixture(raw, metadata_path, basic_path, metadata)
        complete.rm.write_json(annotation, {"format_version": 1, "annotation_version": "kion-series-v1",
            "provenance": {"inputs": [{"path": name, "sha256": complete.rm.file_hash(path)} for name, path in
                (("items.csv", raw / "items.csv"), ("metadata.json", metadata_path))]},
            "coverage": {"confirmed_groups": 80, "confirmed_member_items": 80, "train_known_confirmed_member_items": 0},
            "series": [{"id": f"group-{i:03d}", "status": "confirmed", "curator_rationale": "fixture",
                "members": [{"item_id": 1000 + i, "status": "confirmed", "train_known": False,
                             "evidence": {"title": "fixture"}}]} for i in range(80)]})
    people_path = artifacts / "people.json"
    people.write_catalog(people_path, people.build(raw, metadata_path, rare_threshold=2))
    graph_path = artifacts / "graph.json"
    graph = {"contract": franchise.CONTRACT, "temporal_mode": "retrospective_current", "works": {}, "scopes": {},
             "item_work_links": [], "memberships": [], "edges": [], "evidence": {}}
    complete.rm.write_json(graph_path, graph)
    links_path = artifacts / "links.json"
    complete.rm.write_json(links_path, {"works": [], "people": []})
    text_caches = {}
    for field in complete.TEXT_FIELDS:
        cache = text.build_item_cache(raw / "items.csv", metadata_path, field, artifacts / ("text-" + field), Encoder())
        path = artifacts / ("text-" + field) / cache.manifest["signature"] / "manifest.json"
        text_caches[field] = complete.pin(path)
    mapping_sha = complete.rm.canonical_hash(metadata["item_id_to_token"])
    def matrix(name, family, origin=None):
        path = artifacts / (name + ".npy")
        values = np.zeros((8, 2), np.float32)
        values[3:, 0] = np.arange(5) / 10
        np.save(path, values, allow_pickle=False)
        result = {**complete.pin(path), "family": family, "columns": ["observed_value", "missing"],
            "metadata_sha256": complete.rm.file_hash(metadata_path), "mapping_sha256": mapping_sha,
            "temporal_mode": "static_retrospective", "transform": "identity"}
        if origin is not None:
            result.update(role="rm", cutoff_exclusive=origin, whole_day_exclusive=True,
                max_event_day=origin - 1, max_known_at_day=origin - 1, label_fit_max_day=0)
        return result
    extensions = []
    for slot in complete.EXTENSION_SLOTS:
        family = "semantic_extensions" if slot == "semantic_extensions.franchise" else slot
        entry = matrix(slot, family)
        entry.update(slot=slot, namespace=complete.STATIC_NAMESPACES[slot])
        if slot == "semantic_extensions.franchise":
            entry["columns"] = ["manual_series:observed_value", "manual_series:missing"]
        extensions.append(entry)
    if native_local:
        import kion_local_pack as local_pack
        static_copy = root / "scripts/kion_pack_static.py"
        static_copy.parent.mkdir()
        static_copy.write_bytes(Path(static.__file__).read_bytes())
        with mock.patch.object(static, "__file__", static_copy.as_posix()), mock.patch.dict(
                static.build.__kwdefaults__, {"series_sha256": complete.rm.file_hash(annotation)}):
            tiny_local = local_pack.FeaturePack.build(raw, metadata_path, basic_path, fit_cutoff=complete.CAP, include_relative=True)
    else:
        tiny_local = TinyLocal(raw, metadata_path, basic_path, fit_cutoff=complete.CAP, include_relative=True)
    baseline_dir = artifacts / "local-static"
    baseline_dir.mkdir()
    np.save(baseline_dir / "item_features.npy", tiny_local.item_features, allow_pickle=False)
    baseline_columns = {"item": tiny_local.item_columns, "user": tiny_local.user_columns}
    baseline_manifest = {"feature_columns": baseline_columns, "schema_hash": complete.rm.canonical_hash(baseline_columns),
        "outputs": {"item_features.npy": complete.pin(baseline_dir / "item_features.npy")}}
    complete.rm.write_json(baseline_dir / "manifest.json", baseline_manifest)
    replacement = matrix("story-format-replacement", "story_format")
    replacement.update(columns=tiny_local.item_columns[197:199], mode="replace_named_columns",
        base_manifest_sha256=complete.rm.file_hash(baseline_dir / "manifest.json"),
        base_schema_hash=baseline_manifest["schema_hash"],
        base_item_features_sha256=baseline_manifest["outputs"]["item_features.npy"]["sha256"],
        replaced_values_sha256=text.array_hash(tiny_local.item_features[:, 197:199]))
    semantic = extensions[0]
    narrative = matrix("narrative-replacement", "narrative_relations")
    changed = np.load(narrative["path"])
    changed[3:] += .25
    np.save(narrative["path"], changed, allow_pickle=False)
    narrative.update(sha256=complete.rm.file_hash(narrative["path"]),
        columns=["semantic_extensions." + c for c in semantic["columns"]], mode="replace_named_columns",
        base_manifest_sha256=replacement["base_manifest_sha256"], base_schema_hash=replacement["base_schema_hash"],
        base_item_features_sha256=replacement["base_item_features_sha256"],
        replaced_values_sha256=text.array_hash(np.load(semantic["path"])),
        target_extension={"path": semantic["path"], "sha256": semantic["sha256"], "family": "semantic_extensions",
            "namespace": "semantic_extensions", "shape": [8, 2], "dtype": "float32",
            "columns_sha256": complete.rm.canonical_hash(semantic["columns"]),
            "metadata_sha256": complete.rm.file_hash(metadata_path), "vocabulary_sha256": mapping_sha, "transform": "identity"})
    date_features = []
    origins = [day(v) for v in ("2021-07-01", "2021-07-02", "2021-08-01", "2021-08-02")]
    for family in ("external_facts", "external_interest"):
        banks = {str(origin): matrix(family + "-" + str(origin), family, origin) for origin in origins}
        date_features.append({"family": family, "role": "rm", "fit_cutoff_exclusive": complete.CAP,
            "clock": "strict_prefix", "columns": ["observed_value", "missing"], "banks": banks})
    runtime = {name: {"role": "rm", "fit_cutoff_exclusive": complete.CAP} for name in
               ("local", "personal", "audience", "people_dynamics", "related_interest")}
    runtime["audience"]["parameters"] = {"max_retained_pairs": 100, "max_intersection_work": 100000}
    runtime["people_dynamics"]["parameters"] = {"top_actors": 2, "top_directors": 2, "min_works": 1,
                                               "min_outcome_users": 1}
    runtime["related_interest"]["parameters"] = {"min_pairs": 1, "min_viewers": 1}
    event_history = history.build_history_index(raw, metadata, fit_cutoff="2021-08-09", include_relative=True,
        schema_item_ids=metadata["item_id_to_token"].keys())
    event_audit = diagnostics.CompletionDiagnostics(event_history)
    event_path = artifacts / "event-store"
    events.write_probe_store(event_audit, event_path, metadata_path,
        np.flatnonzero(event_history.days < day("2021-08-09")))
    threshold_dir = artifacts / "threshold-facts"
    event_store = events.load(event_path, allow_probe=True)
    try:
        threshold = user_diagnostics_module.build_threshold_facts(event_store, threshold_dir)
    finally:
        event_store.close()
    user_diagnostics = artifacts / "user-diagnostics.json"
    complete.rm.write_json(user_diagnostics, {"contract": "kion-user-diagnostics-v1", "diagnostic_only": True,
        "model_input_columns_added": [], "columns": ["diagnostic." + str(i) for i in range(99)],
        "event_store": complete.pin(event_path / "manifest.json"), "threshold_facts": complete.pin(threshold_dir / "manifest.json"),
        "threshold_signature": threshold["signature"]})
    manifest_path = root / "provider.json"
    declaration = {"state": "fixture_only", "metadata": complete.pin(metadata_path), "mapping_sha256": mapping_sha,
        "raw_sources": {name: complete.pin(raw / (name + ".csv")) for name in ("items", "users", "interactions")},
        "basic_npz": complete.pin(basic_path), "basic_manifest": complete.pin(artifacts / "manifest.json"),
        "metadata_availability": "retrospective_catalog_snapshot_not_historical_asof", "runtime": runtime,
        "families": {name: {"status": "fixture_only"} for name in complete.COMPONENTS},
        "static_extensions": extensions, "text_caches": text_caches, "people_catalog": complete.pin(people_path),
        "static_replacements": [replacement, narrative], "local_static_baseline": {"manifest": complete.pin(baseline_dir / "manifest.json"),
            "scope": "item_only_outcome_free_rebuilt_metadata", "array_sha256": text.array_hash(tiny_local.item_features)},
        "franchise_graph": complete.pin(graph_path), "franchise_temporal_mode": "retrospective_current",
        "franchise_order_kinds": list(complete.FRANCHISE_ORDER_KINDS),
        "franchise_scope_policies": {kind: None for kind in complete.FRANCHISE_ORDER_KINDS},
        "entity_links": complete.pin(links_path), "date_features": date_features,
        "completion_event_store": complete.pin(event_path / "manifest.json"),
        "user_diagnostics": complete.pin(user_diagnostics),
        "limits": {"batch_rows": 32, "profile_cache_rows": 16, "bank_cache_dates": 2}}
    runtime["people_dynamics"]["backend"] = "fast-exact"
    complete.seal_manifest(manifest_path, declaration)
    return manifest_path, records, sel, cal


def ready_manifest(full_path):
    """Tiny dedicated RM prefix, then exact pinned human-scope wrapper."""
    value = complete._read(full_path)
    root = Path(full_path).parent
    raw = Path(value["raw_sources"]["items"]["path"]).parent
    metadata = complete._read(value["metadata"]["path"])
    native = history.build_history_index(raw, metadata, fit_cutoff=complete.CAP, include_relative=True,
        schema_item_ids=metadata["item_id_to_token"].keys())
    with (raw / "items.csv").open(encoding="utf-8", newline="") as stream:
        cards = {row["item_id"]: row for row in csv.DictReader(stream)}
    related = related_module.RelatedInterestIndex(metadata, cards, native.users, native.tokens, native.days,
        role="rm", fit_cutoff=complete.CAP, min_pairs=1, min_viewers=1,
        provenance={"source_pins": {**value["raw_sources"], "metadata": value["metadata"]},
                    "source_scope": "full_source", "pilot_only": False})
    directory = root / "dedicated-related-rm"
    related.save(directory)
    decision_path = Path(__file__).parents[1] / "artifacts/task2-kion/ready-pack-20261020/scope-decision-r1.json"
    decision = complete._read(decision_path)
    value.update(integration_mode="scoped_production", scope_id=complete.READY_SCOPE_ID,
        scope_decision=complete.pin(decision_path), active_families=decision["active_families"],
        additional_ready_components=decision["additional_ready_components"])
    value["runtime"] = {name: value["runtime"][name] for name in ("local", "personal", "related_interest")}
    value["runtime"]["personal"]["series_annotation"] = complete.pin(metadata.get("series_annotation_path", root / "artifacts/series.json"))
    value["runtime"]["related_interest"] = {"role": "rm", "fit_cutoff_exclusive": complete.CAP,
                                            "manifest": complete.pin(directory / "manifest.json")}
    value["date_features"] = [d for d in value["date_features"] if d["family"] == "external_interest"]
    value["families"] = {name: {"status": "fixture_only"} for name in decision["active_families"] + decision["additional_ready_components"]}
    value.pop("entity_links")
    value.pop("signature")
    path = root / "provider-ready.json"
    complete.seal_manifest(path, value)
    return path, decision_path


def retrospective_ready_manifest(path):
    value = complete._read(path)
    root = Path(path).parent / "fixed403-fixture"
    root.mkdir()
    sources = {}
    for name in complete.READY_EXTERNAL_PINS:
        source = root / (name + ".json")
        complete.rm.write_json(source, {"results": [{"receipt": {"retrieved_at": "2026-10-20T12:00:00Z"}}]}
                               if name == "collection" else {"fixture_only": True})
        sources[name] = complete.pin(source)
    declaration = value["date_features"][0]
    declaration.update(temporal_regime=complete.READY_EXTERNAL_REGIME, retrospective_sources=sources,
        strict_asof_claim=False, columns=["wiki_work_ru_temporal_mask", "missing"])
    for key, bank in declaration["banks"].items():
        origin = int(key)
        values = np.load(bank["path"])
        values[:, 0] = 0
        np.save(bank["path"], values, allow_pickle=False)
        bank.update(sha256=complete.rm.file_hash(bank["path"]), columns=declaration["columns"],
            numeric_max_event_day=origin - 3, max_event_day=origin - 3,
            modeled_lag_hours=24, modeled_numeric_available_at=date.fromordinal(origin - 1).isoformat() + "T00:00:00Z",
            max_known_at_day=day("2026-10-20"), actual_response_retrieved_at="2026-10-20T12:00:00Z",
            numeric_prefix_valid=True, identity_known_at=None, retrospective_identity=True,
            strict_asof_claim=False, label_fit_performed=False, label_fit_max_day=0)
    value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
    complete.rm.write_json(path, value)
    return value


class ReadyRuntimeClosureTests(unittest.TestCase):
    def test_complete_provider_uses_real_factory_without_injected_local(self):
        import kion_pack_static as static
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "native"
            full, _, _, _ = fixture(root, native_local=True)
            path, decision = ready_manifest(full)
            annotation = root / "configs/datasets/kion-series-v1.json"
            with mock.patch.object(static, "__file__", (root / "scripts/kion_pack_static.py").as_posix()), mock.patch.dict(
                    static.build.__kwdefaults__, {"series_sha256": complete.rm.file_hash(annotation)}):
                provider = complete.CompleteRewardProvider.load(path, fixture_mode=True, scope_decision=decision)
                try:
                    users = np.array([1, 1], np.int64)
                    origins = np.array([day("2021-07-01"), complete.CAP_DAY], np.int64)
                    x, lengths, _ = provider.raw.history_batch(users, origins)
                    self.assertEqual(lengths[0], 1)
                    self.assertGreater(lengths[1], 1)
                    actual = provider.user_features(users, origins, x, lengths)
                    candidate = provider.candidate_features(np.array([3, 4]), origins, user_ids=users)
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertEqual(candidate.dtype, np.float32)
                    self.assertTrue(np.isfinite(actual).all() and np.isfinite(candidate).all())
                    self.assertEqual(provider.local.provenance["components"]["static"]["inputs"]["series"]["sha256"], complete.rm.file_hash(annotation))
                    self.assertEqual(provider.dynamic.provenance["series_sha256"], complete.rm.file_hash(annotation))
                    self.assertEqual(provider.personal.provenance["representation_manifest"]["series_annotation_sha256"], complete.rm.file_hash(annotation))
                finally:
                    provider.close()

    def test_three_reader_paths_and_native_consumption_are_bound(self):
        import kion_pack_static as static
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied_static = root / "scripts/kion_pack_static.py"
            copied_static.parent.mkdir()
            copied_static.write_bytes(Path(static.__file__).read_bytes())
            source = Path(__file__).parents[1] / "configs/datasets/kion-series-v1.json"
            frozen = root / "configs/datasets/kion-series-v1.json"
            frozen.parent.mkdir(parents=True)
            frozen.write_bytes(source.read_bytes())
            raw = root / "data/raw"
            raw.mkdir(parents=True)
            (raw / "items.csv").write_text("item_id\n10\n", encoding="utf-8")
            value = {"runtime": {"local": {"series_annotation": complete.pin(frozen)},
                                "personal": {"series_annotation": complete.pin(source)}},
                     "source_pins": [complete.pin(frozen)],
                     "raw_sources": {"items": complete.pin(raw / "items.csv")}}
            consumed = SimpleNamespace(provenance={"components": {"static": {"inputs": {
                "series": complete.pin(frozen)}}}}, dynamic=SimpleNamespace(provenance={
                    "series_sha256": complete.READY_SERIES_SHA256}))
            with mock.patch.object(static, "__file__", copied_static.as_posix()):
                complete._ready_series_config(root, value, {}, local=consumed)
                for mutation, error in (
                    (lambda v: v["runtime"]["local"].pop("series_annotation"), "static series"),
                    (lambda v: v.update(source_pins=[]), "missing from source pins"),
                    (lambda v: v["runtime"]["personal"]["series_annotation"].update(sha256="0" * 64), "personal series")):
                    changed = copy.deepcopy(value)
                    mutation(changed)
                    with self.assertRaisesRegex(ValueError, error):
                        complete._ready_series_config(root, changed, {})
                other = root / "other.json"
                other.write_text('{"series":[]}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "dynamic series"):
                    complete._ready_series_config(root, value, {"series_annotation_path": other.as_posix()})
                consumed.dynamic.provenance["series_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "consumed another"):
                    complete._ready_series_config(root, value, {}, local=consumed)
                frozen.unlink()
                with self.assertRaises(FileNotFoundError):
                    complete._ready_series_config(root, value, {})

    def test_real_feature_pack_constructor_requires_and_consumes_frozen_config(self):
        """All five native builders execute; only tiny fixture hash/path differ."""
        import kion_local_pack as local_pack
        import kion_pack_static as static
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, prepared = root / "raw", root / "prepared"
            raw.mkdir(); prepared.mkdir()
            frozen_static = root / "scripts/kion_pack_static.py"
            frozen_static.parent.mkdir()
            frozen_static.write_bytes(Path(static.__file__).read_bytes())
            annotation = root / "configs/datasets/kion-series-v1.json"
            annotation.parent.mkdir(parents=True)
            meta = {"format_version": 1, "context": 64, "vocab_size": 5, "user_feature_dim": 20,
                "split_dates": {"train": "<2021-08-09", "val": "2021-08-09..2021-08-15"},
                "special_tokens": {"PAD": 0, "BOS": 1, "UNK": 2},
                "item_id_to_token": {"10": 3, "20": 4}, "user_categories": {
                    "age": static.local.AGE_CATEGORIES,
                    "income": [static.UNKNOWN, "income_0_20", "income_150_inf", "income_20_40", "income_40_60", "income_60_90", "income_90_150"],
                    "sex": static.sex.PREPARED_SEX_CATEGORIES, "kids_flg": [static.UNKNOWN, "0", "1"]}}
            metadata_path = prepared / "metadata.json"
            complete.rm.write_json(metadata_path, meta)
            write_csv(raw / "items.csv", [dict(item_id=i, genres="драмы", content_type="film", release_year="2000",
                countries="Россия", age_rating="16.0", actors="Actor A", directors="Director A", studios="Studio A",
                for_kids="0.0", title="Title", title_orig="Title", description="Description", keywords="") for i in (10, 20)])
            write_csv(raw / "users.csv", [dict(user_id=1, age="age_18_24", income="income_20_40", sex="М", kids_flg="0")])
            write_csv(raw / "interactions.csv", [dict(user_id=1, item_id=10, last_watch_dt="2021-07-01", watched_pct="50"),
                dict(user_id=1, item_id=20, last_watch_dt="2021-08-02", watched_pct="100")])
            groups, offset = {}, {"user": 0, "item": 0}
            fixed = {"age": static.local.AGE_CATEGORIES, "sex": static.sex.SEX_CATEGORIES,
                "countries": [static.UNKNOWN], "content_type": static.basic.CONTENT_TYPE_CATEGORIES,
                "release_year": static.basic.YEAR_COLUMNS, "age_rating": static.basic.AGE_RATING_CATEGORIES}
            n_genres = 197 - sum(len(fixed[n]) for n in ("countries", "content_type", "release_year", "age_rating"))
            fixed["genres"] = [static.UNKNOWN] + [f"genre-{i:03d}" for i in range(n_genres - 1)]
            values = np.zeros((5, 197), np.float32)
            for name in static.basic.GROUP_NAMES:
                scope = "user" if name in ("age", "sex") else "item"
                start, columns = offset[scope], list(fixed[name])
                groups[name] = dict(scope=scope, start=start, stop=start + len(columns), columns=columns)
                if scope == "item":
                    values[3:, start + (1 if name == "release_year" else 0)] = 1
                offset[scope] += len(columns)
            basic = root / "basic.npz"
            np.savez(basic, item_features=values)
            complete.rm.write_json(root / "manifest.json", {"builder_contract": static.basic.CONTRACT,
                "features": {"sha256": complete.rm.file_hash(basic), "arrays": {"item_features": {"sha256": static.local.array_hash(values)}}},
                "groups": groups, "vocabulary": {"sha256": static.local.canonical_hash(meta["item_id_to_token"])},
                "inputs": {name: complete.pin(path) for name, path in (("metadata", metadata_path),
                    ("items", raw / "items.csv"), ("users", raw / "users.csv"))}})
            series = {"format_version": 1, "annotation_version": "kion-series-v1", "provenance": {"inputs": [
                {"path": name, "sha256": complete.rm.file_hash(path)} for name, path in
                (("items.csv", raw / "items.csv"), ("metadata.json", metadata_path))]},
                "coverage": {"confirmed_groups": 80, "confirmed_member_items": 80, "train_known_confirmed_member_items": 0},
                "series": [{"id": f"group-{i:03d}", "status": "confirmed", "curator_rationale": "fixture",
                    "members": [{"item_id": 1000 + i, "status": "confirmed", "train_known": False,
                                 "evidence": {"title": "fixture"}}]} for i in range(80)]}
            complete.rm.write_json(annotation, series)
            expected = complete.rm.file_hash(annotation)
            # Fixture annotation binds tiny raw sources; all native build logic,
            # default path resolution and post-load hashes remain unmodified.
            with mock.patch.object(static, "__file__", frozen_static.as_posix()), mock.patch.dict(
                    static.build.__kwdefaults__, {"series_sha256": expected}):
                pack = local_pack.FeaturePack.build(raw, metadata_path, basic, fit_cutoff=complete.CAP, include_relative=True)
                self.assertEqual(pack.history.fit_cutoff, complete.CAP_DAY)
                self.assertEqual(pack.dynamic.fit_cutoff, complete.CAP_DAY)
                self.assertEqual(pack.dynamic.provenance["indexed_rows"], 1)
                self.assertEqual(Path(pack.provenance["components"]["static"]["inputs"]["series"]["path"]), annotation.resolve())
                self.assertEqual(pack.dynamic.provenance["series_sha256"], expected)
                x, lengths = np.zeros((2, 64), np.int64), np.array([1, 2], np.int64)
                x[:, 0] = 1
                x[1, 1] = 3
                origins = np.array([day("2021-07-01"), complete.CAP_DAY])
                self.assertTrue(np.isfinite(pack.user_features(np.array([1, 1]), origins, x, lengths)).all())
                self.assertTrue(np.isfinite(pack.candidate_features(np.array([3, 4]), origins)).all())
                annotation.unlink()
                with self.assertRaises(FileNotFoundError):
                    local_pack.FeaturePack.build(raw, metadata_path, basic, fit_cutoff=complete.CAP, include_relative=True)


class CompleteRewardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.records, self.sel, self.cal = fixture(self.root / "fixture")

    def load(self, path=None):
        value = complete.CompleteRewardProvider.load(path or self.path, fixture_mode=True, local_factory=TinyLocal)
        self.addCleanup(value.close)
        return value

    def alter(self, callback):
        value = complete._read(self.path)
        callback(value)
        value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
        complete.rm.write_json(self.path, value)

    def test_ready_missing_runtime_config_fails_before_repeated_input_hashing(self):
        path, decision = ready_manifest(self.path)
        value = complete._read(path)
        value["state"] = "scoped_production"
        value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
        with mock.patch.object(complete, "_walk_pins", side_effect=AssertionError("expensive source hashing")) as walk:
            with self.assertRaisesRegex(ValueError, "static series config closure"):
                complete._admit(path.parent, value, False, decision)
        walk.assert_not_called()

    def test_native_role_caps_complete_schema_and_no_sft_weights(self):
        value = self.load()
        self.assertEqual(value.contract()["fit_cutoff_exclusive"], complete.CAP)
        for index in (value.history, value.dynamic, value.audience, value.people_dynamics, value.related):
            self.assertEqual(index.fit_cutoff, complete.CAP_DAY)
        self.assertEqual(value.people_dynamics.provenance["fit_role"], "rm")
        self.assertEqual(value.related.role, "rm")
        self.assertEqual(len(value.contract()["agreement_components"]), 81)
        self.assertEqual(value.event_store.manifest["fit_cutoff"], "2021-08-09")
        self.assertNotEqual(len(value.user_columns) + len(value.candidate_columns), 4087)
        for prefix in ("e5.title.", "e5.title_orig.", "e5.description.", "raw_people.", "personal.",
                       "demography_", "people_", "related_interest.", "local_dynamic.", "franchise.",
                       "external_facts.", "external_interest."):
            self.assertTrue(any(c.startswith(prefix) for c in value.candidate_columns), prefix)
        self.assertEqual(len(value.item_categorical_names), 6)
        self.assertTrue(any("Actor".casefold() in c.casefold() for c in value.item_categorical[3]))
        np.testing.assert_array_equal(value.item_features[:, :197], value.local.item_features[:, :197])
        self.assertFalse(np.array_equal(value.item_features[:, 197:199], value.local.item_features[:, 197:199]))

    def test_ready_scope_constructs_only_retained_backends_and_preserves_values(self):
        full = self.load()
        path, decision = ready_manifest(self.path)
        with mock.patch.object(complete, "_people_fast_module", side_effect=AssertionError("deferred people factory")), \
                mock.patch.object(audience_module.AudienceIndex, "from_csv", side_effect=AssertionError("deferred audience factory")):
            value = self.load(path)
        self.assertIsNone(value.people_dynamics)
        self.assertIsNone(value.audience)
        contract = value.contract()
        self.assertFalse(contract["all81_complete"])
        self.assertFalse(contract["full_package_admitted"])
        self.assertFalse(contract["full_row_fit_admitted"])
        self.assertEqual(contract["family_acceptance"].keys(), dict.fromkeys(contract["active_families"]).keys())
        self.assertEqual(contract["scope_decision"]["sha256"], complete.READY_SCOPE_SHA256)
        self.assertEqual(value.user_columns, full.user_columns)
        retained = [full.candidate_columns.index(name) for name in value.candidate_columns]
        users, origins = np.array([1, self.sel]), np.array([day("2021-08-01"), complete.CAP_DAY])
        x, lengths, _ = value.raw.history_batch(users, origins)
        np.testing.assert_array_equal(value.user_features(users, origins, x, lengths), full.user_features(users, origins, x, lengths))
        expected = full.candidate_features(np.array([5, 6]), origins, user_ids=users)[:, retained]
        np.testing.assert_array_equal(value.candidate_features(np.array([5, 6]), origins, user_ids=users), expected)
        complete._feature_contract(contract, fixture_mode=True)

    def test_ready_scope_rejects_decision_family_backend_and_clock_substitution(self):
        path, decision = ready_manifest(self.path)
        original = complete._read(path)
        def alter(callback):
            value = copy.deepcopy(original)
            callback(value)
            value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
            complete.rm.write_json(path, value)
        for callback, message in ((lambda m: m["scope_decision"].update(sha256="0" * 64), "hash mismatch"),
                (lambda m: m["active_families"].append("audience"), "census differs"),
                (lambda m: m["runtime"].update(people_dynamics={"role": "rm", "fit_cutoff_exclusive": complete.CAP}), "deferred feature backends"),
                (lambda m: m["runtime"]["related_interest"].update(fit_cutoff_exclusive="2021-08-09"), "native RM/Aug2")):
            alter(callback)
            with self.assertRaisesRegex(ValueError, message):
                self.load(path)
        alter(lambda m: m.update(state="scoped_production"))
        with self.assertRaisesRegex(ValueError, "explicit scope_decision"):
            complete.CompleteRewardProvider.load(path)

    def test_ready_observed_bundle_roundtrip_and_missing_origin_never_drops_BOS(self):
        path, decision = ready_manifest(self.path)
        output = self.root / "ready-dataset"
        manifest = complete.build_complete_reward_data(path, output, fixture_mode=True, local_factory=TinyLocal,
            row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)
        roles, catalog, loaded = complete.load_complete_reward_bundle(output, fixture_mode=True)
        self.assertEqual(loaded["feature_pack"]["integration_mode"], "scoped_production")
        self.assertIn(0., roles["fit"]["reward"])
        self.assertIn(1., roles["fit"]["reward"])
        self.assertIn(0, roles["fit"]["history_length"])
        self.assertTrue(manifest["exclusions"]["target_support"]["zero_support_set_preserved"])
        provider = complete.load_reward_provider_from_bundle(output, fixture_mode=True, local_factory=TinyLocal)
        self.addCleanup(provider.close)
        for rows in roles.values():
            for array in rows.values():
                complete._close_map(array)
        value = complete._read(path)
        value["date_features"][0]["banks"].pop(str(day("2021-07-01")))
        value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
        complete.rm.write_json(path, value)
        with self.assertRaisesRegex(ValueError, "bank coverage is incomplete"):
            complete.build_complete_reward_data(path, self.root / "missing-origin", fixture_mode=True, local_factory=TinyLocal,
                row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)

    def test_ready_series_pin_overrides_nearer_ambient_config_without_mutating_metadata(self):
        full, *_ = fixture(self.root / "no-series-in-metadata", explicit_series=False)
        path, _ = ready_manifest(full)
        manifest = complete._read(path)
        raw = Path(manifest["raw_sources"]["items"]["path"]).parent
        ambient = raw / "configs/datasets/kion-series-v1.json"
        ambient.parent.mkdir(parents=True)
        complete.rm.write_json(ambient, {"series": [{"id": "wrong-nearer-franchise", "status": "confirmed",
            "members": [{"item_id": 10, "status": "confirmed"}]}]})
        import kion_personal_candidates as personal_module
        with mock.patch.object(personal_module, "build_index", wraps=personal_module.build_index) as build:
            value = self.load(path)
        consumed = build.call_args.args[1]
        self.assertIsNot(consumed, value.metadata)
        self.assertNotIn("series_annotation_path", value.metadata)
        self.assertEqual(consumed["series_annotation_path"], manifest["runtime"]["personal"]["series_annotation"]["path"])
        self.assertEqual(value.personal.provenance["representation_manifest"]["series_annotation_sha256"],
                         manifest["runtime"]["personal"]["series_annotation"]["sha256"])
        self.assertEqual(len(value.personal.key_rows["franchise"][3]), 0)

    def test_ready_series_pin_missing_hash_and_metadata_path_substitution_fail(self):
        path, _ = ready_manifest(self.path)
        original = complete._read(path)
        for mutation, message in ((lambda m: m["runtime"]["personal"].pop("series_annotation"), "Explicit pinned RM"),
                (lambda m: m["runtime"]["personal"]["series_annotation"].update(sha256="0" * 64), "hash mismatch")):
            value = copy.deepcopy(original)
            mutation(value)
            value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
            complete.rm.write_json(path, value)
            with self.assertRaisesRegex(ValueError, message):
                self.load(path)
        copy_path = self.root / "same-bytes-other-path.json"
        copy_path.write_bytes(Path(original["runtime"]["personal"]["series_annotation"]["path"]).read_bytes())
        original["runtime"]["personal"]["series_annotation"] = complete.pin(copy_path)
        original["signature"] = complete.rm.canonical_hash({k: v for k, v in original.items() if k != "signature"})
        complete.rm.write_json(path, original)
        with self.assertRaisesRegex(ValueError, "differs from prepared metadata"):
            self.load(path)

    def test_ready_series_consumed_native_hash_is_verified(self):
        path, _ = ready_manifest(self.path)
        import kion_personal_candidates as personal_module
        native_build = personal_module.build_index
        def wrong_consumption(*args, **kwargs):
            result = native_build(*args, **kwargs)
            result.provenance["representation_manifest"]["series_annotation_sha256"] = "0" * 64
            return result
        with mock.patch.object(personal_module, "build_index", side_effect=wrong_consumption):
            with self.assertRaisesRegex(ValueError, "consumed a different series"):
                self.load(path)

    def test_ready_retrospective_source_keeps_actual2026_knowledge_and_past_numeric_clock(self):
        path, _ = ready_manifest(self.path)
        manifest = retrospective_ready_manifest(path)
        value = self.load(path)
        clock = value.contract()["external_clocks"]["external_interest"]
        self.assertEqual(clock["actual_max_known_at_day"], day("2026-10-20"))
        self.assertEqual(clock["actual_response_retrieved_at"], "2026-10-20T12:00:00Z")
        self.assertFalse(clock["strict_asof_claim"])
        self.assertFalse(clock["label_fit_performed"])
        features = value.candidate_features(np.array([3]), np.array([complete.CAP_DAY]), user_ids=np.array([1]))
        self.assertEqual(features[0, value.candidate_columns.index("external_interest.wiki_work_ru_temporal_mask")], 0)
        declaration = manifest["date_features"][0]
        value.fixture_mode = False
        with self.assertRaisesRegex(ValueError, "fixed403 source substitution"):
            value._external_clock(declaration)
        value.fixture_mode = True
        value.scope = None
        with self.assertRaisesRegex(ValueError, "restricted to ready"):
            value._external_clock(declaration)

    def test_ready_retrospective_future_event_availability_and_false_known_time_fail(self):
        path, _ = ready_manifest(self.path)
        original = retrospective_ready_manifest(path)
        def mutate(callback):
            value = copy.deepcopy(original)
            bank = next(iter(value["date_features"][0]["banks"].values()))
            callback(bank)
            value["signature"] = complete.rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
            complete.rm.write_json(path, value)
        for callback, message in (
                (lambda b: b.update(numeric_max_event_day=b["cutoff_exclusive"]), "numeric event/modeled availability"),
                (lambda b: b.update(modeled_numeric_available_at=date.fromordinal(b["cutoff_exclusive"]).isoformat()+"T00:00:00Z"), "numeric event/modeled availability"),
                (lambda b: b.update(max_known_at_day=b["cutoff_exclusive"] - 1), "actual source knowledge"),
                (lambda b: b.update(label_fit_performed=True), "actual source knowledge"),
                (lambda b: b.update(strict_asof_claim=True), "actual source knowledge")):
            mutate(callback)
            with self.assertRaisesRegex(ValueError, message):
                self.load(path)

    def test_ready_retrospective_strict_mask_and_legacy_strict_clock_cannot_be_promoted(self):
        path, _ = ready_manifest(self.path)
        original = retrospective_ready_manifest(path)
        bank = next(iter(original["date_features"][0]["banks"].values()))
        values = np.load(bank["path"])
        values[3:, 0] = 1
        np.save(bank["path"], values, allow_pickle=False)
        bank["sha256"] = complete.rm.file_hash(bank["path"])
        original["signature"] = complete.rm.canonical_hash({k: v for k, v in original.items() if k != "signature"})
        complete.rm.write_json(path, original)
        with self.assertRaisesRegex(ValueError, "strict temporal masks must remain zero"):
            self.load(path)
        fresh_path, *_ = fixture(self.root / "fresh-full-strict")
        strict = self.load(fresh_path)
        declaration = copy.deepcopy(strict.date_features[0])
        declaration.update(temporal_regime=complete.READY_EXTERNAL_REGIME, strict_asof_claim=False)
        with self.assertRaisesRegex(ValueError, "restricted to ready"):
            strict._external_clock(declaration)

    def test_dedicated_fast_people_uses_exact_core_and_aug2_clock(self):
        value = self.load()
        adapter = value.people_dynamics
        self.assertTrue(hasattr(adapter, "_prepare_layout"))
        self.assertEqual(type(adapter).__name__, "FastPeopleDynamicsIndex")
        self.assertEqual(value.contract()["people_backend"], complete.PEOPLE_BACKEND)
        core = people_core.build(value.raw_dir, value.metadata_path, role="rm", fit_cutoff=complete.CAP,
                                 **value.manifest["runtime"]["people_dynamics"]["parameters"])
        tokens = np.arange(3, value.metadata["vocab_size"], dtype=np.int64)
        for origin in (day("2021-07-01"), day("2021-07-02"), day("2021-08-01"), complete.CAP_DAY):
            origins = np.full(len(tokens), origin, np.int64)
            np.testing.assert_array_equal(adapter.features(tokens, origins), core.features(tokens, origins))
        np.testing.assert_array_equal(adapter.features(tokens, "2021-08-09"), core.features(tokens, complete.CAP))
        # A real candidate call must dispatch to the adapter. An adapter failure
        # propagates; it must never silently switch to the slower core path.
        with mock.patch.object(type(adapter), "features", side_effect=RuntimeError("forced-fast-dispatch")) as dispatched:
            with self.assertRaisesRegex(RuntimeError, "forced-fast-dispatch"):
                value.candidate_features(np.array([3]), np.array([complete.CAP_DAY]), user_ids=np.array([1]))
            dispatched.assert_called_once()

    def test_persisted_aug2_people_still_loads_fast_and_pins_adapter(self):
        original = self.load()
        directory = self.root / "rm-people"
        original.people_dynamics.save(directory)
        self.alter(lambda m: m["runtime"].update(people_dynamics={"role": "rm", "fit_cutoff_exclusive": complete.CAP,
            "backend": "fast-exact", "manifest": complete.pin(directory / "manifest.json")}))
        value = self.load()
        self.assertEqual(type(value.people_dynamics).__name__, "FastPeopleDynamicsIndex")
        self.assertEqual(value.people_dynamics.provenance["fit_role"], "rm")
        self.assertEqual(value.people_dynamics.fit_cutoff, complete.CAP_DAY)
        closure = {entry["path"]: entry["sha256"] for entry in value.contract()["input_pins"]}
        adapter = Path(complete.__file__).with_name("kion_people_dynamics_fast.py").resolve()
        self.assertEqual(closure[adapter.as_posix()], complete.PEOPLE_BACKEND["adapter_sha256"])
        tokens = np.arange(3, value.metadata["vocab_size"], dtype=np.int64)
        np.testing.assert_array_equal(value.people_dynamics.features(tokens, complete.CAP),
                                      original.people_dynamics.features(tokens, complete.CAP))
        self.alter(lambda m: m["runtime"]["people_dynamics"].update(backend="core"))
        with self.assertRaisesRegex(ValueError, "no fallback"):
            self.load()

    def test_fast_loader_preserves_ambient_package_and_rejects_missing_source_pin(self):
        package = sys.modules.get("scripts")
        before = None if package is None else dict(vars(package))
        module = complete._people_fast_module()
        self.assertIs(module.core, people_core)
        self.assertIs(sys.modules.get("scripts"), package)
        if package is not None:
            self.assertEqual(vars(package), before)
        self.alter(lambda m: m.update(source_pins=[p for p in m["source_pins"]
            if Path(p["path"]).name != "kion_people_dynamics_fast.py"]))
        with self.assertRaisesRegex(ValueError, "Missing complete RM/runtime source pins"):
            self.load()

    def test_declared_aug2_fast_cannot_admit_persisted_sft_people_index(self):
        manifest = complete._read(self.path)
        raw_dir = Path(manifest["raw_sources"]["items"]["path"]).parent
        index = people_core.build(raw_dir, Path(manifest["metadata"]["path"]), role="sft", fit_cutoff="2021-08-09",
                                 **manifest["runtime"]["people_dynamics"]["parameters"])
        directory = self.root / "sft-people"
        index.save(directory)
        self.alter(lambda m: m["runtime"].update(people_dynamics={"role": "rm", "fit_cutoff_exclusive": complete.CAP,
            "backend": "fast-exact", "manifest": complete.pin(directory / "manifest.json")}))
        with self.assertRaisesRegex(ValueError, "Native index cap is not Aug2: people_dynamics"):
            self.load()

    def test_late_static_values_and_two_independent_preimages(self):
        manifest = complete._read(self.path)
        profession = next(e for e in manifest["static_extensions"] if e["family"] == "professions")
        values = np.load(profession["path"])
        values[3:, 0] = -np.arange(1, 6)
        np.save(profession["path"], values, allow_pickle=False)
        self.alter(lambda m: next(e for e in m["static_extensions"] if e["family"] == "professions").update(
            sha256=complete.rm.file_hash(profession["path"]), transform="signed_log1p", transform_columns=[0]))
        value = self.load()
        manifest = complete._read(self.path)
        blocks, columns = [value.local.item_features], list(value.local.item_columns)
        for entry in manifest["static_extensions"]:
            block = np.load(entry["path"])
            if entry["transform"] == "signed_log1p":
                idx = entry["transform_columns"]
                block[:, idx] = np.sign(block[:, idx]) * np.log1p(np.abs(block[:, idx]))
            blocks.append(block)
            columns += [entry["namespace"] + "." + c for c in entry["columns"]]
        expected = np.concatenate(blocks, axis=1)
        for entry in manifest["static_replacements"]:
            idx = [columns.index(name) for name in entry["columns"]]
            self.assertEqual(text.array_hash(expected[:, idx]), entry["replaced_values_sha256"])
            expected[:, idx] = np.load(entry["path"])
        np.testing.assert_array_equal(value.item_features[:, :len(columns)], expected)
        np.testing.assert_array_equal(value.item_features[:, :197], value.local.item_features[:, :197])
        for family in complete.COMPONENTS:
            complete._family_payload(manifest, family)
        self.assertEqual(value.contract()["user_diagnostics"]["model_columns"], [])
        self.assertFalse(any(name.startswith("diagnostic.") for name in value.user_columns + value.candidate_columns))
        self.assertIn("semantic_extensions.manual_series:observed_value", value.item_columns)
        self.assertFalse(any(name.startswith("semantic_extensions.franchise.") for name in value.item_columns))

    def test_late_missing_slot_narrative_target_and_diagnostic_input_are_rejected(self):
        original = complete._read(self.path)
        self.alter(lambda m: m.update(static_extensions=[e for e in m["static_extensions"] if e["namespace"] != "story_format"]))
        with self.assertRaisesRegex(ValueError, "extension slot"):
            self.load()
        complete.rm.write_json(self.path, original)
        self.alter(lambda m: m["static_replacements"][1]["target_extension"].update(columns_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "exact appended semantic328"):
            self.load()
        complete.rm.write_json(self.path, original)
        path = Path(original["user_diagnostics"]["path"])
        descriptor = complete._read(path)
        descriptor["model_input_columns_added"] = ["diagnostic.0"]
        complete.rm.write_json(path, descriptor)
        self.alter(lambda m: m.update(user_diagnostics=complete.pin(path)))
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            self.load()

    def test_manual_franchise_slot_cannot_rename_native_namespace(self):
        self.alter(lambda m: next(e for e in m["static_extensions"] if e["slot"] == "semantic_extensions.franchise").update(
            namespace="semantic_extensions.franchise"))
        with self.assertRaisesRegex(ValueError, "namespace/family/clock"):
            self.load()

    def test_two_native_franchise_branches_preserve_distinct_semantics(self):
        scopes = {"main": {"work_ids": ["a", "b", "c"], "order_kind": "mainline_release", "scope_complete": True,
            "known_prior_complete": True, "adjacency_complete": True, "source_known_at": "2020-01-01"}}
        scopes["catalog"] = {**scopes["main"], "order_kind": "catalog_release_precedes", "scope_complete": False,
                             "known_prior_complete": False, "adjacency_complete": False}
        graph = {"contract": franchise.CONTRACT, "evidence": {"e": {"source": "fixture"}},
            "works": {w: {"release_upper": "2000-12-31", "source_known_at": "2020-01-01"} for w in ("a", "b", "c")},
            "scopes": scopes, "item_work_links": [{"item_id": raw, "work_id": work, "status": "verified",
                "source_known_at": "2020-01-01", "evidence_ids": ["e"]} for raw, work in ((10, "a"), (20, "b"), (30, "c"))],
            "memberships": [{"work_id": w, "scope_id": sid, "status": "verified", "source_known_at": "2020-01-01",
                "evidence_ids": ["e"]} for sid in scopes for w in ("a", "b", "c")],
            "edges": [{"from_work": "a", "to_work": "b", "scope_id": sid, "status": "verified",
                "order_kind": scope["order_kind"], "direct_next": sid == "main", "source_known_at": "2020-01-01",
                "evidence_ids": ["e"]} for sid, scope in scopes.items()]}
        path = self.root / "typed-graph.json"
        complete.rm.write_json(path, graph)
        self.alter(lambda m: m.update(franchise_graph=complete.pin(path)))
        value = self.load()
        result = value.candidate_features(np.array([4]), np.array([day("2021-07-02")]), user_ids=np.array([1]))[0]
        columns = value.candidate_columns
        self.assertEqual(sum(c.startswith("franchise.") for c in columns), 50)
        self.assertEqual(result[columns.index("franchise.mainline_release.direct_next")], 1)
        self.assertEqual(result[columns.index("franchise.catalog_release_precedes.frontier")], 1)
        self.assertEqual(result[columns.index("franchise.catalog_release_precedes.direct_next")], 0)
        ready_path, decision = ready_manifest(self.path)
        with mock.patch.object(franchise, "prepare_graph", wraps=franchise.prepare_graph) as prepare:
            ready = complete.CompleteRewardProvider.load(ready_path, fixture_mode=True,
                local_factory=TinyLocal, scope_decision=decision)
        self.addCleanup(ready.close)
        self.assertEqual(prepare.call_count, 1)
        indices = [i for i, name in enumerate(columns) if name.startswith("franchise.")]
        ready_indices = [i for i, name in enumerate(ready.candidate_columns) if name.startswith("franchise.")]
        with mock.patch.object(franchise, "candidate_progress", side_effect=AssertionError("public hot path")), \
                mock.patch.object(franchise, "validate_graph", side_effect=AssertionError("revalidation")), \
                mock.patch.object(franchise, "canonical_hash", side_effect=AssertionError("graph rehash")):
            for origin in ("2021-07-02", "2021-08-01", "2021-08-02"):
                got = ready.candidate_features(np.array([4]), np.array([day(origin)]), user_ids=np.array([1]))[0]
                self.assertEqual(sum(c.startswith("franchise.") for c in ready.candidate_columns), 50)
                if origin == "2021-07-02":
                    np.testing.assert_array_equal(got[ready_indices], result[indices])
        for origin in ("2021-07-02", "2021-08-01", "2021-08-02"):
            got = ready.candidate_features(np.array([4]), np.array([day(origin)]), user_ids=np.array([1]))[0]
            expected = value.candidate_features(np.array([4]), np.array([day(origin)]), user_ids=np.array([1]))[0]
            np.testing.assert_array_equal(got[ready_indices], expected[indices])
        self.alter(lambda m: m["franchise_scope_policies"].update(catalog_release_precedes={"b": "main"}))
        with self.assertRaisesRegex(ValueError, "same order kind"):
            self.load()
        self.alter(lambda m: m.update(franchise_order_kinds=["mainline_release"]))
        with self.assertRaisesRegex(ValueError, "Both explicit typed"):
            self.load()

    def test_own_outcome_and_future_invariance_across_all_native_families(self):
        original = self.load()
        query_users = np.array([1, self.sel, self.cal], np.int64)
        origins = np.array([day("2021-08-01"), complete.CAP_DAY, complete.CAP_DAY], np.int64)
        x, lengths, _ = original.raw.history_batch(query_users, origins)
        user = original.user_features(query_users, origins, x, lengths)
        candidates = original.candidate_features(np.array([5, 7, 7]), origins, user_ids=query_users)
        path, _, _, _ = fixture(self.root / "mutant", mutations=lambda r: (*r[:3], "1") if
            (r[2] >= "2021-08-02" or (r[0] == 1 and r[1] == 30)) else r)
        mutant = self.load(path)
        new_user = mutant.user_features(query_users, origins, x, lengths)
        new_candidates = mutant.candidate_features(np.array([5, 7, 7]), origins, user_ids=query_users)
        # The altered own target is on the query day and therefore unavailable.
        np.testing.assert_array_equal(user[0], new_user[0])
        np.testing.assert_array_equal(candidates[0], new_candidates[0])
        # Mutating that fit outcome may legally affect Aug2 heldout features;
        # a future-only mutation cannot affect any query.
        path, _, _, _ = fixture(self.root / "future", mutations=lambda r: (*r[:3], "1") if r[2] >= "2021-08-02" else r)
        future = self.load(path)
        np.testing.assert_array_equal(user, future.user_features(query_users, origins, x, lengths))
        np.testing.assert_array_equal(candidates, future.candidate_features(np.array([5, 7, 7]), origins, user_ids=query_users))

    def test_native_query_h63_masks_and_candidate_binding(self):
        value = self.load()
        users = np.array([3, 1], np.int64)
        origins = np.array([day("2021-07-02"), complete.CAP_DAY], np.int64)
        x, lengths, _ = value.raw.history_batch(users, origins)
        user = value.user_features(users, origins, x, lengths, history_cutoffs=origins)
        self.assertEqual(user[0, value.user_columns.index("text_profile.description.known_n")], 0)
        self.assertEqual(user[0, value.user_columns.index("text_profile.description.valid")], 0)
        candidate = value.candidate_features(np.array([5, 6]), origins, user_ids=users)
        self.assertEqual(candidate[0, value.candidate_columns.index("e5.description.missing")], 1)
        self.assertEqual(candidate[1, value.candidate_columns.index("e5.title_orig.missing")], 1)
        swapped = value.candidate_features(np.array([6, 5]), origins, user_ids=users)
        self.assertFalse(np.array_equal(candidate, swapped))
        bad = x.copy()
        bad[0, 1] = 3
        with self.assertRaisesRegex(ValueError, "H63"):
            value.user_features(users, origins, bad, lengths)

    def test_reject_aug9_component_and_query_clocks(self):
        self.alter(lambda m: m["runtime"]["audience"].update(fit_cutoff_exclusive="2021-08-09"))
        with self.assertRaisesRegex(ValueError, "native RM/Aug2"):
            self.load()
        self.alter(lambda m: m["runtime"]["audience"].update(fit_cutoff_exclusive=complete.CAP))
        value = self.load()
        with self.assertRaisesRegex(ValueError, "<=Aug2"):
            value.candidate_features(np.array([3]), np.array([day("2021-08-09")]), user_ids=np.array([1]))
        with self.assertRaisesRegex(ValueError, "cutoff"):
            value.candidate_features(np.array([3]), np.array([complete.CAP_DAY]), user_ids=np.array([1]),
                                     history_cutoffs=np.array([complete.CAP_DAY - 1]))

    def test_reject_outcome_learned_weights_external_future_and_missing_family(self):
        self.alter(lambda m: m["representations"].update(people_embeddings="learned_sft_checkpoint_aug9"))
        with self.assertRaisesRegex(ValueError, "learned SFT"):
            self.load()
        self.alter(lambda m: m.update(representations=copy.deepcopy(complete.REPRESENTATIONS)))
        self.alter(lambda m: m["date_features"][0]["banks"][str(complete.CAP_DAY)].update(label_fit_max_day=complete.CAP_DAY))
        with self.assertRaisesRegex(ValueError, "future/own-day"):
            self.load()
        self.alter(lambda m: m["date_features"][0]["banks"][str(complete.CAP_DAY)].update(label_fit_max_day=0))
        self.alter(lambda m: m["families"].pop("related_interest"))
        with self.assertRaisesRegex(ValueError, "family census"):
            self.load()

    def test_mapping_source_and_production_fixture_guards(self):
        value = complete._read(self.path)
        matrix = Path(value["static_extensions"][2]["path"])
        with matrix.open("ab") as stream:
            stream.write(b"mutation")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()
        self.alter(lambda m: m["static_extensions"][2].update(sha256=complete.rm.file_hash(matrix), mapping_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "mapping binding"):
            self.load()
        with self.assertRaisesRegex(ValueError, "fixture-only"):
            complete.CompleteRewardProvider.load(self.path, local_factory=TinyLocal)

    def test_observed_dataset_roundtrip_freeze_support_and_raw_people_categories(self):
        output = self.root / "dataset"
        manifest = complete.build_complete_reward_data(self.path, output, fixture_mode=True, local_factory=TinyLocal,
            row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)
        roles, catalog, loaded = complete.load_complete_reward_bundle(output, fixture_mode=True)
        self.assertEqual(len(catalog.item_categorical_names), 6)
        self.assertEqual(manifest["exclusions"]["invalid_labels"], 1)
        self.assertEqual(manifest["exclusions"]["missing_labels"], 1)
        self.assertEqual(manifest["exclusions"]["unobserved_pairs_generated"], 0)
        self.assertEqual(manifest["exclusions"]["target_support"]["zero_support_items"], 1)
        self.assertTrue(manifest["exclusions"]["target_support"]["zero_support_set_preserved"])
        for role in ("select", "calibration"):
            self.assertTrue((roles[role]["history_cutoff"] == complete.CAP_DAY).all())
            np.testing.assert_array_equal(roles[role]["numeric"][0], roles[role]["numeric"][1])
            self.assertEqual(int(roles[role]["history_max_date"].max()) < complete.CAP_DAY, True)
            ids = np.load(output / f"{role}.event_ids.npy")
            self.assertEqual(ids.shape, (len(roles[role]["user_id"]), 63))
        self.assertIn(0., roles["fit"]["reward"])
        self.assertIn(1., roles["fit"]["reward"])
        provider = complete.load_reward_provider_from_bundle(output, fixture_mode=True, local_factory=TinyLocal)
        self.addCleanup(provider.close)
        self.assertEqual(provider.user_columns, catalog.context_names)
        with self.assertRaisesRegex(ValueError, "acceptance mismatch"):
            complete.load_complete_reward_bundle(output)
        old = copy.deepcopy(loaded)
        old.pop("feature_pack")
        complete.rm.write_json(output / "manifest.json", old)
        with self.assertRaisesRegex(ValueError, "old RM"):
            complete.load_complete_reward_bundle(output)

    def test_manifest_replacement_during_stream_never_marks_ready(self):
        output = self.root / "mutating"
        changed = False
        def mutate(_):
            nonlocal changed
            if not changed:
                self.alter(lambda m: m["runtime"]["audience"]["parameters"].update(coview_lambda=1000))
                changed = True
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            complete.build_complete_reward_data(self.path, output, fixture_mode=True, local_factory=TinyLocal,
                row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8, progress=mutate)
        self.assertTrue(changed)
        self.assertFalse((output / "manifest.json").exists())

    def test_nested_event_and_text_payload_mutations_never_mark_ready(self):
        for family in ("event", "text"):
            with self.subTest(family=family):
                path, _, _, _ = fixture(self.root / family)
                value = complete._read(path)
                native = Path(value["completion_event_store"]["path"] if family == "event"
                              else value["text_caches"]["title"]["path"])
                payload = native.parent / ("n_global.npy" if family == "event" else "truncated.npy")
                output = self.root / (family + "-mutating")
                changed = []
                def mutate(_):
                    if not changed:
                        array = np.load(payload, mmap_mode="r+", allow_pickle=False)
                        array[0] = not array[0] if family == "text" else array[0] + 1
                        array.flush()
                        array._mmap.close()
                        changed.append(True)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    complete.build_complete_reward_data(path, output, fixture_mode=True, local_factory=TinyLocal,
                        row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8, progress=mutate)
                self.assertTrue(changed)
                self.assertFalse((output / "manifest.json").exists())

    def test_user_diagnostic_threshold_payload_closure_blocks_mutation(self):
        for field in ("threshold_bits", "source_csv_data_row"):
            with self.subTest(field=field):
                path, _, _, _ = fixture(self.root / field)
                manifest = complete._read(path)
                diagnostic = complete._read(manifest["user_diagnostics"]["path"])
                native = complete._read(diagnostic["threshold_facts"]["path"])
                payload = Path(native["arrays"][field]["path"])
                provider = self.load(path)
                self.assertIn(complete.pin(payload), provider.contract()["input_pins"])
                provider.close()
                output = self.root / (field + "-mutating")
                changed = []
                def mutate(_):
                    if not changed:
                        array = np.load(payload, mmap_mode="r+", allow_pickle=False)
                        array[0] += 1
                        array.flush()
                        array._mmap.close()
                        changed.append(True)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    complete.build_complete_reward_data(path, output, fixture_mode=True, local_factory=TinyLocal,
                        row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8, progress=mutate)
                self.assertTrue(changed)
                self.assertFalse((output / "manifest.json").exists())
        output = self.root / "successful-diagnostics"
        complete.build_complete_reward_data(self.path, output, fixture_mode=True, local_factory=TinyLocal,
            row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)
        descriptor = complete._read(complete._read(self.path)["user_diagnostics"]["path"])
        payload = complete._read(descriptor["threshold_facts"]["path"])["arrays"]["threshold_bits"]["path"]
        array = np.load(payload, mmap_mode="r+", allow_pickle=False)
        array[0] += 1
        array.flush()
        array._mmap.close()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            complete.load_complete_reward_bundle(output, fixture_mode=True)

    def test_checked_input_closure_includes_persisted_native_payloads(self):
        original = self.load()
        directory = self.root / "audience"
        descriptor = original.audience.save(directory,
            source_pins=audience_module.source_pins(original.raw_dir, original.metadata_path))
        self.alter(lambda m: m["runtime"].update(audience={"role": "rm", "fit_cutoff_exclusive": complete.CAP,
            "manifest": complete.pin(directory / "manifest.json")}))
        provider = self.load()
        closure = {entry["path"]: entry["sha256"] for entry in provider.contract()["input_pins"]}
        for spec in descriptor["arrays"].values():
            self.assertEqual(closure[(directory / spec["file"]).resolve().as_posix()], spec["sha256"])
        event = Path(provider.manifest["completion_event_store"]["path"])
        for spec in provider.event_store.manifest["arrays"].values():
            self.assertEqual(closure[(event.parent / spec["path"]).resolve().as_posix()], spec["sha256"])

    def test_failed_complete_load_closes_role_maps_with_retained_traceback(self):
        output = self.root / "dataset"
        complete.build_complete_reward_data(self.path, output, fixture_mode=True, local_factory=TinyLocal,
            row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)
        with (output / "select.event_ids.npy").open("ab") as stream:
            stream.write(b"tamper")
        acquired, errors = [], []
        native_load = complete.rm.load_bundle
        def capture(*args, **kwargs):
            result = native_load(*args, **kwargs)
            acquired.append(result[0])
            return result
        with mock.patch.object(complete.rm, "load_bundle", side_effect=capture):
            try:
                complete.load_complete_reward_bundle(output, fixture_mode=True)
            except ValueError as error:
                errors.append(error)
        self.assertEqual(len(errors), 1)
        self.assertIn("event sidecar checksum", str(errors[0]))
        self.assertTrue(acquired)
        for rows in acquired[0].values():
            for array in rows.values():
                self.assertTrue(array._mmap.closed)
        source = output / "fit.numeric.npy"
        renamed = source.with_suffix(".closed")
        source.rename(renamed)
        renamed.rename(source)

    def test_reject_promoted_fixture_status_and_catalogue_receipt_swap(self):
        output = self.root / "dataset"
        complete.build_complete_reward_data(self.path, output, fixture_mode=True, local_factory=TinyLocal,
            row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)
        value = complete._read(output / "manifest.json")
        original = copy.deepcopy(value)
        value["feature_pack"]["status"] = "admitted_full"
        value["feature_pack"]["family_acceptance"] = {k: "accepted_full" for k in complete.COMPONENTS}
        complete.rm.write_json(output / "manifest.json", value)
        with self.assertRaisesRegex(ValueError, "Fixture admission"):
            complete.load_complete_reward_bundle(output)
        original["feature_pack"]["provider_signature"] = "1" * 64
        complete.rm.write_json(output / "manifest.json", original)
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            complete.load_complete_reward_bundle(output, fixture_mode=True)

    def test_persisted_native_index_from_different_raw_cannot_add_future_rows(self):
        original = self.load()
        directory = self.root / "old-audience"
        original.audience.save(directory, source_pins=audience_module.source_pins(original.raw_dir, original.metadata_path))
        current_path, _, _, _ = fixture(self.root / "changed-authority", mutations=lambda r:
            (r[0], r[1], "2021-08-05", r[3]) if r[0] == 2 and r[1] == 30 else r)
        current = complete._read(current_path)
        current["runtime"]["audience"] = {"role": "rm", "fit_cutoff_exclusive": complete.CAP,
                                           "manifest": complete.pin(directory / "manifest.json")}
        current["signature"] = complete.rm.canonical_hash({k: v for k, v in current.items() if k != "signature"})
        complete.rm.write_json(current_path, current)
        with self.assertRaisesRegex(ValueError, "raw source identity mismatch: audience/interactions"):
            self.load(current_path)

    def test_dynamic_evidence_payload_binds_actual_raw_and_runtime_code(self):
        value = complete._read(self.path)
        payload = complete._family_payload(value, "audience")
        self.assertEqual(payload["raw_sources"], value["raw_sources"])
        self.assertEqual(payload["metadata"], value["metadata"])
        self.assertEqual(payload["source_pins"], value["source_pins"])
        value["raw_sources"]["interactions"]["sha256"] = "2" * 64
        self.assertNotEqual(complete._family_payload(value, "audience"), payload)

    def test_fixture_full_source_guard_precedes_any_native_factory(self):
        value = complete._read(self.path)
        path = Path(value["raw_sources"]["interactions"]["path"])
        write_csv(path, [dict(user_id=i, item_id=10, last_watch_dt="2021-07-01", watched_pct=50) for i in range(4097)])
        self.alter(lambda m: m["raw_sources"].update(interactions=complete.pin(path)))
        factory = mock.Mock(side_effect=AssertionError("native source build must not start"))
        with self.assertRaisesRegex(ValueError, "full source"):
            complete.CompleteRewardProvider.load(self.path, fixture_mode=True, local_factory=factory)
        factory.assert_not_called()

    def test_same_width_mapping_permutation_is_rejected_before_build(self):
        value = complete._read(self.path)
        path = Path(value["metadata"]["path"])
        metadata = complete._read(path)
        metadata["item_id_to_token"]["10"], metadata["item_id_to_token"]["20"] = 4, 3
        complete.rm.write_json(path, metadata)
        self.alter(lambda m: m.update(metadata=complete.pin(path)))
        with self.assertRaisesRegex(ValueError, "metadata/mapping identity"):
            self.load()

    def test_event_store_c0_clock_and_row_join_are_bound(self):
        value = self.load()
        users = np.array([1, self.sel, self.cal, 3], np.int64)
        origins = np.repeat(complete.CAP_DAY, 4).astype(np.int64)
        x, lengths, _ = value.raw.history_batch(users, origins)
        ids = value.query_event_ids(users, origins, x=x, lengths=lengths)
        selected = ids[ids >= 0]
        self.assertTrue((value.event_store.arrays["date"][selected] < complete.CAP_DAY).all())
        self.assertEqual(value.event_store.manifest["normalization"], "immutable_per_event_no_global_fit")
        output = self.root / "events-dataset"
        complete.build_complete_reward_data(self.path, output, fixture_mode=True, local_factory=TinyLocal,
            row_caps={"fit": 100, "select": 100, "calibration": 100}, batch_size=8)
        manifest = complete._read(output / "manifest.json")
        target = output / "select.event_ids.npy"
        with target.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "event sidecar checksum"):
            complete.load_complete_reward_bundle(output, fixture_mode=True)
        descriptor = complete._read(self.path)
        store_path = Path(descriptor["completion_event_store"]["path"])
        store_manifest = complete._read(store_path)
        store_manifest["c0"] = .9
        store_manifest["signature"] = events.canonical_hash({k: v for k, v in store_manifest.items() if k != "signature"})
        complete.rm.write_json(store_path, store_manifest)
        self.alter(lambda m: m.update(completion_event_store=complete.pin(store_path)))
        with self.assertRaisesRegex(ValueError, "fixed c0"):
            self.load()

    def test_named_replacements_reject_wrong_preimage_and_original197(self):
        self.alter(lambda m: m["static_replacements"][0].update(replaced_values_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "preimage"):
            self.load()
        self.alter(lambda m: m["static_replacements"][0].update(columns=["original.1", "legacy.story_flag"]))
        with self.assertRaisesRegex(ValueError, "original197"):
            self.load()

    def test_declared_aug2_cannot_admit_native_aug9_index(self):
        def wrong_factory(raw_dir, metadata, basic, **kwargs):
            return TinyLocal(raw_dir, metadata, basic, fit_cutoff="2021-08-09", include_relative=True)
        with self.assertRaisesRegex(ValueError, "returned an SFT/Aug9 cap"):
            complete.CompleteRewardProvider.load(self.path, fixture_mode=True, local_factory=wrong_factory)


if __name__ == "__main__":
    unittest.main()
