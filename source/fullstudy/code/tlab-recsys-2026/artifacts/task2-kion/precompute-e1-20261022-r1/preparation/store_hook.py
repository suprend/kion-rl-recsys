"""Explicit prepared-profile hook; original factory admission remains intact.

No automatic install or provider import at module import. Root pins this file,
all dependencies and the complete source/input bindings in its worker request.
"""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


EXPECTED_ROWS = {"train": 3477437, "val": 158760}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def checked_pin(pin):
    path = Path(pin["path"]).resolve()
    if file_sha(path) != pin["sha256"]:
        raise ValueError("Source/receipt pin changed: " + path.as_posix())
    return path


def load_pinned(pin, name):
    path = checked_pin(pin)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def query_key(provider, split, row):
    data = provider.splits[split]
    return (int(data["user_id"][row]), int(provider._origin(split, row)),
        int(data["lengths"][row]), data["x"][row].tobytes())


def prepared_query_method(original, pin):
    """Compile only reviewed clone's _query injection seam, never its factory."""
    path = checked_pin(pin)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CompleteProvider")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_query"]
    if len(methods) != 1 or [node.arg for node in methods[0].args.kwonlyargs] != ["prepared_profile"]:
        raise ValueError("Reviewed prepared-profile _query seam changed")
    namespace = dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), path.as_posix(), "exec"), namespace)
    return namespace["_query"]


def reader_query(original_method, prepared_method, reader, profile_type, *, allow_missing=False, counters=None):
    counters = counters if counters is not None else {}
    for name in ("lru_hits", "store_hits", "store_misses", "fallback_prepares", "tail_checks"):
        counters.setdefault(name, 0)
    counters.setdefault("store_read_seconds", 0.)

    def query(self, split, row):
        key = query_key(self, split, row)
        if key in self._profiles:
            counters["lru_hits"] += 1
            return original_method(self, split, row)
        before = time.perf_counter()
        value = reader.get(key, personal=self.personal, profile_type=profile_type)
        counters["store_read_seconds"] += time.perf_counter() - before
        if value is None:
            counters["store_misses"] += 1
            if not allow_missing:
                raise KeyError("Prepared query store lacks exact required key")
            counters["fallback_prepares"] += 1
            return original_method(self, split, row)
        profile, tail = value
        counters["store_hits"] += 1
        state = prepared_method(self, split, row, prepared_profile=profile)
        actual = state["user"]
        if actual.dtype != tail.dtype or actual.shape != tail.shape or actual.tobytes() != tail.tobytes():
            # Original _query inserts only after raw history/user construction;
            # remove the just-inserted corrupt record before failing closed.
            self._profiles.pop(key, None)
            raise ValueError("Prepared profile reconstructed user tail differs")
        counters["tail_checks"] += 1
        return state
    return query


def check_coverage(options):
    path = checked_pin(options["coverage_receipt"])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("state") != "completed" or receipt.get("mode") != "full"
            or receipt.get("split_rows") != EXPECTED_ROWS
            or Path(receipt.get("store_path", "")).resolve() != Path(options["store_path"]).resolve()
            or receipt.get("store_source_pins_sha256") != canonical_sha(options["store_source_pins"])):
        raise ValueError("Full training requires exact completed train/val coverage")
    if not receipt.get("store_sha256") or file_sha(options["store_path"]) != receipt["store_sha256"]:
        raise ValueError("Completed query database differs from full coverage receipt")
    return receipt


class HookHandle:
    def __init__(self, module, options, codec, prepared):
        self.module, self.options = module, dict(options)
        self.codec, self.prepared = codec, prepared
        self.original_factory = module.load_training_provider
        self.instances = []
        self.closed_evidence = []
        self.factory_calls = 0
        self._installed = False

    def factory(self, args):
        # All original load/admission/source/raw input checks run first.
        provider = self.original_factory(args)
        reader = None
        try:
            original_cls = type(provider)
            counts = {}
            if self.options["enabled"]:
                if provider.personal is None:
                    raise ValueError("Prepared query hook requires admitted personal features")
                for split, count in EXPECTED_ROWS.items():
                    if len(provider.splits[split]["y"]) != count:
                        raise ValueError("Unexpected corrected E1 split rows")
                reader = self.codec.QueryStoreReader(self.options["store_path"],
                    expected_schema_hash=provider.personal.schema_hash,
                    expected_source_pins=self.options["store_source_pins"])
                profile_type = importlib.import_module("kion_personal_candidates").QueryProfiles
                query = reader_query(original_cls._query, self.prepared, reader, profile_type,
                    allow_missing=self.options.get("allow_missing_fallback", False), counters=counts)
            else:
                query = original_cls._query
            handle = self
            record = {"provider": provider, "reader": reader, "counts": counts, "closed": False}

            def close(instance):
                if record["closed"]:
                    return
                record["closed"] = True
                snapshot = {"query": dict(counts)}
                try:
                    if reader is not None:
                        snapshot["reader"] = reader.stats()
                        reader.close()
                finally:
                    original_cls.close(instance)
                    handle.closed_evidence.append(snapshot)

            provider.__class__ = type("PreparedQueryProvider", (original_cls,), {"_query": query, "close": close})
            self.instances.append(record)
            self.factory_calls += 1
            return provider
        except BaseException:
            if reader is not None:
                reader.close()
            provider.close()
            raise

    def evidence(self):
        active = [{"query": dict(row["counts"]),
            **({"reader": row["reader"].stats()} if row["reader"] is not None else {})}
            for row in self.instances if not row["closed"]]
        return {"contract": "prepared-query-provider-hook-v1", "enabled": self.options["enabled"],
            "allow_missing_fallback": self.options.get("allow_missing_fallback", False),
            "factory_calls": self.factory_calls, "active": active, "closed": list(self.closed_evidence)}

    def uninstall(self):
        if self._installed:
            if self.module.load_training_provider != self.factory:
                raise RuntimeError("Provider factory ownership changed before uninstall")
            self.module.load_training_provider = self.original_factory
            self._installed = False
        for row in self.instances:
            if not row["closed"]:
                row["provider"].close()


def install(train, options):
    """Training worker API. DynamicIndex cache is owned by its original caller."""
    if type(options.get("enabled")) is not bool:
        raise ValueError("Hook requires explicit boolean enabled")
    allow = options.get("allow_missing_fallback", False)
    if type(allow) is not bool or allow and options.get("purpose") != "pilot-shortbench":
        raise ValueError("Missing-query fallback is restricted to explicit pilot-shortbench")
    if options["enabled"] and not allow:
        check_coverage(options)
    original = importlib.import_module("kion_complete_pack")
    if Path(original.__file__).resolve() != checked_pin(options["provider_source"]):
        raise ValueError("Loaded provider source path differs")
    if getattr(original.load_training_provider, "__module__", None) != original.__name__:
        raise ValueError("Install prepared hook before any other provider factory wrapper")
    codec = load_pinned(options["codec_source"], "_prepared_query_codec")
    prepared = prepared_query_method(original, options["query_method_source"])
    handle = HookHandle(original, options, codec, prepared)
    original.load_training_provider = handle.factory
    handle._installed = True
    return handle
