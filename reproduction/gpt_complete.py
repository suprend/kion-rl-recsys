"""Scoped complete-provider composition with explicit bound filesystem seams."""
from collections import OrderedDict
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
from types import MappingProxyType
from gpt_inputs import identity
from gpt_provider import BoundIO, canonical_hash, logical_child, read_matrix, load_scoped_base_data
from gpt_admission import AdmissionBackend, admit_scoped
from gpt_text_people import load_item_cache, load_catalog
from gpt_diagnostics import load_diagnostics


class _Composition:
    @classmethod
    def load(cls, *args, **kwargs):
        raise TypeError("Use gpt_complete.CompleteProvider.load context manager")

    def _admission(self, *args):
        raise RuntimeError("Admission belongs to the explicit composition entrypoint")

    def _live(self):
        if self._closed:
            raise RuntimeError("Provider is closed")
        self.io.check()

    def batch(self, rows, split):
        self._live()
        return self._original.CompleteProvider.batch(self, rows, split)

    def candidate_features(self, rows, ids, split):
        self._live()
        return self._original.CompleteProvider.candidate_features(self, rows, ids, split)

    def origin_keys(self, split):
        self._live()
        return self._original.CompleteProvider.origin_keys(self, split)

    def contract(self):
        self._live()
        return self._original.CompleteProvider.contract(self)

    def _external_bank(self, declaration, origin):
        self._live()
        # Retained LRU implementation opens only through overridden _matrix.
        return self._original.CompleteProvider._external_bank(self, declaration, origin)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        resources = [getattr(self, n, None) for n in ("audience_core", "related", "people_dynamics")]
        resources += list(getattr(self, "_diagnostic_stores", {}).values())
        resources += list(getattr(self, "_maps", [])) + list(getattr(self, "_banks", {}).values())
        seen = set()
        for value in resources:
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            try:
                mapping = getattr(value, "_mmap", None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
                elif hasattr(value, "close"):
                    value.close()
            except BaseException as error:
                errors.append(error)
        self._maps, self._banks, self._profiles = [], OrderedDict(), OrderedDict()
        if errors:
            raise errors[0]

    def _source(self, name):
        return self.runtime.logical_source_name(name)

    def _pin(self, root, receipt):
        logical = logical_child(root, receipt["path"])
        self.io.pin(root, receipt, self.io.bindings[logical].role)
        return logical

    def _require_sources(self, modules):
        declared = {logical_child(self.root, p["path"]): p["sha256"] for p in self.manifest["source_pins"]}
        for module in modules:
            logical = self.runtime.logical_source(module)
            if logical not in declared:
                raise ValueError("Runtime module source pin missing: " + logical)
            self.io.verify(logical, "source", declared[logical])

    def _matrix(self, declaration):
        self.io.check()
        return read_matrix(self.io, self.root, declaration, self.metadata["vocab_size"],
            self.metadata_hash, self.vocabulary_hash, np=self._original.np)

    def _init_runtime(self, config):
        from gpt_runtime import init_scoped_runtime
        return init_scoped_runtime(self, config, io=self.io, runtime=self.runtime)

    def _load_diagnostics(self, declarations):
        result, stores = load_diagnostics(self.io, self.root, declarations,
            metadata_hash=self.metadata_hash, manifest=self.manifest, np=self._original.np,
            events=self.runtime.load("kion_completion_event_store"),
            diagnostics=self.runtime.load("kion_user_diagnostics"), require_sources=self._require_sources)
        self._diagnostic_stores = stores
        return result

    def _replacement(self, declaration, extensions):
        self._require_sources([self.runtime.load("kion_complete_manifest")])
        columns = self.base_manifest["feature_columns"]["item"]
        targets = declaration.get("columns")
        if not isinstance(targets, list) or not targets or any(not isinstance(n, str) or not n for n in targets) or len(set(targets)) != len(targets):
            raise ValueError("Nonempty unique replacement columns required")
        target = declaration.get("target_extension")
        if target is None:
            if any(n not in columns or columns.index(n) < 197 for n in targets):
                raise ValueError("Replacement target absent or protected original197")
            return
        if not isinstance(target, dict):
            raise ValueError("Replacement extension binding must be an object")
        def binding(d):
            return {"path": logical_child(self.root, d["path"]), "sha256": d["sha256"], "family": d["family"],
                "namespace": d.get("namespace", d["family"]), "columns_sha256": canonical_hash(d["columns"]),
                **{k: d[k] for k in ("shape", "dtype", "metadata_sha256", "vocabulary_sha256", "transform")}}
        matches = [d for d in extensions if binding(d) == target]
        preimage = declaration.get("replaced_values_sha256", "")
        if len(matches) != 1 or target.get("transform") != "identity" or not isinstance(preimage, str) or len(preimage) != 64 or any(c not in "0123456789abcdef" for c in preimage):
            raise ValueError("Replacement extension identity/schema/preimage binding mismatch")
        extension = matches[0]
        named = {target["namespace"] + "." + name for name in extension["columns"]}
        if not set(targets).issubset(named):
            raise ValueError("Replacement target absent from bound extension")
        selected = [extension["columns"].index(n[len(target["namespace"]) + 1:]) for n in targets]
        source = self._matrix(extension)
        try:
            np = self._original.np
            if list(source.shape) != target["shape"] or str(source.dtype) != target["dtype"] or self._original.text_module.array_hash(np.asarray(source[:, selected], dtype=np.float32)) != preimage:
                raise ValueError("Extension preimage values mismatch")
        finally:
            mapping = getattr(source, "_mmap", None)
            if mapping is not None:
                mapping.close()


    def _load(self, data_dir, allow_partial):
        if allow_partial:
            raise ValueError("Scoped-only provider")
        original = self._original
        np, baseline, local, text_module = original.np, original.baseline, original.local, original.text_module
        history_module = original.history_module
        FIT_CUTOFF, STATIC_MODULES, TEXT_FIELDS = original.FIT_CUTOFF, original.STATIC_MODULES, original.TEXT_FIELDS
        NUMERIC_UNITS, NUMERIC_TRANSFORMS = original.NUMERIC_UNITS, original.NUMERIC_TRANSFORMS
        CONTRACT, SCOPED_READY_COMPONENTS, SCOPED_FAMILIES = original.CONTRACT, original.SCOPED_READY_COMPONENTS, original.SCOPED_FAMILIES
        _names, _positive = original._names, original._positive
        numeric_plan, numeric_log_columns = original.numeric_plan, original.numeric_log_columns
        _path, _pin = logical_child, self._pin
        _provenance = self.io.provenance
        m = self.manifest
        self._require_sources([baseline, local, text_module])
        if _path(self.root, m["data"]["path"]) != data_dir:
            raise ValueError("Prepared data directory differs from composite binding")
        for name in ("metadata.json", "train.npz", "val.npz"):
            if _pin(self.root, m["data"]["files"][name]) != logical_child(data_dir, name):
                raise ValueError("Prepared data file path binding mismatch")
        base_dir = _path(self.root, m["base_cache"]["path"])
        metadata, train, val, items, self.base_manifest = load_scoped_base_data(
            self.io, data_dir, base_dir, manifest_sha256=m["base_cache"]["manifest_sha256"], np=np, baseline=baseline)
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
            source = self._source(STATIC_MODULES[family][:-3])
            if source not in {_path(self.root, p["path"]) for p in m["source_pins"]}:
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
            source = self._source(STATIC_MODULES[declaration["family"]][:-3])
            if source not in {_path(self.root, p["path"]) for p in m["source_pins"]}:
                raise ValueError("Replacement builder source pin missing")
            if (declaration.get("base_manifest_sha256") != m["base_cache"]["manifest_sha256"]
                    or declaration.get("base_item_features_sha256") != self.base_manifest["outputs"]["item_features.npy"]["sha256"]
                    or declaration.get("base_schema_hash") != self.base_manifest["schema_hash"]
                    or declaration.get("transform") != "identity"):
                raise ValueError("Replacement does not bind exact immutable base/schema/units")
            _provenance(self.root, declaration)
            self._replacement(declaration, extension_declarations)
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
            cache = load_item_cache(self.io, directory, source_logical=self._source("kion_text_embeddings"),
                np=np, expected_manifest_sha256=receipt["manifest_sha256"], expected_vocabulary_sha256=vocab_hash,
                allow_test_encoder=False)
            self._maps.extend((cache.embeddings, cache.missing, cache.token_lengths, cache.truncated))
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
            people = self.runtime.load("kion_people_embeddings")
            path = _pin(self.root, m["people_catalog"])
            self.people_catalog = load_catalog(self.io, path,
                source_bindings={name + ".py": self._source(name) for name in ("kion_people_embeddings", "kion_pack_people")},
                expected_manifest_sha256=m["people_catalog"]["sha256"])
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
        if m.get("entity_links") is not None:
            raise ValueError("Entity links are outside the approved scope")
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



def _compose(io, runtime, *, data_dir, feature_dir, scope_decision, manifest_sha256):
    original = runtime.load("kion_complete_pack")
    backend = AdmissionBackend.from_sources(io,
        provider_source=runtime.logical_source(original),
        assembler_source=runtime.logical_source(runtime.load("kion_complete_manifest")),
        local_source=runtime.logical_source(runtime.load("kion_local_pack")))
    admission = admit_scoped(io, feature_dir, requested_scope_decision=scope_decision,
                             backend=backend, manifest_sha256=manifest_sha256)
    cls = type("ScopedCompleteProvider", (_Composition, original.CompleteProvider), {})
    obj = cls()
    obj.io, obj.runtime, obj._original = io, runtime, original
    obj._closed = False
    obj.root, obj.requested_scope_decision = feature_dir, scope_decision
    obj.manifest, obj.scope = admission.manifest, admission.scope
    obj.acceptance = dict(admission.acceptance)
    obj.required_families = admission.required_families
    obj.scoped, obj.partial = True, False
    obj._maps, obj._profiles = [], OrderedDict()
    try:
        obj._load(data_dir, False)
        obj.source_receipt = runtime.receipt()
        obj.source_receipt["adapters"] = {
            name: {"sha256": hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()}
            for name in ("gpt_complete.py", "gpt_provider.py", "gpt_admission.py", "gpt_runtime.py",
                         "gpt_text_people.py", "gpt_related.py", "gpt_diagnostics.py", "gpt_sources.py",
                         "gpt_inputs.py")}
        io.check()
        return obj
    except BaseException:
        obj.close()
        raise


class CompleteProvider:
    @staticmethod
    @contextmanager
    def load(data_dir, feature_dir, *, bindings, check, scope_decision,
             manifest_sha256, external=("numpy", "torch")):
        """Execute only in an authorized future process; owns all resource lifetimes.

        All paths are ORIGINAL logical identities. Bindings alone relocate bytes.
        The mandatory manifest digest identifies original manifest bytes; its own
        canonical signature and exact human scope are independently admitted.
        `check` must raise on caller cancellation/deadline. No model is built.
        """
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha256):
            raise ValueError("Exact original manifest SHA256 required")
        data_dir, feature_dir, scope_decision = map(identity, (data_dir, feature_dir, scope_decision))
        io = BoundIO(bindings, check)
        from gpt_sources import pinned_runtime
        with pinned_runtime(io, external=external) as runtime:
            obj = _compose(io, runtime, data_dir=data_dir, feature_dir=feature_dir,
                           scope_decision=scope_decision, manifest_sha256=manifest_sha256)
            try:
                yield obj
            finally:
                obj.close()


def load_training_provider(args, *, bindings, check, manifest_sha256,
                           external=("numpy", "torch")):
    """Return a context manager matching the original training argument names."""
    if getattr(args, "allow_partial_integration", False):
        raise ValueError("Partial integration is outside this scoped entrypoint")
    return CompleteProvider.load(args.data, args.feature_dir, bindings=bindings, check=check,
        manifest_sha256=manifest_sha256, scope_decision=args.scope_decision, external=external)
