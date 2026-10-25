"""Portable, explicit read seams for the retained complete feature provider.

Importing this module is stdlib-only. Component reads are execution, not input
planning: payload hashing and numerical loading occur only on explicit calls.
This module does not yet compose a CompleteProvider or grant training admission.
Original signed dictionaries and logical source identities are never rewritten.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import stat
from types import MappingProxyType

from gpt_inputs import identity, sha, json_bytes
from input_inventory import checked_path, relative_parts

ORIGINAL_PROVIDER_SHA256 = "7cbe284dfbd9d133bd98e2d363a9ee8b8d8530ffe7b977cbe13c6146d0f4fe5e"
METADATA_LIMIT = 16 * 1024 * 1024
SCOPED_BASE_MANIFEST_LIMIT = 32 * 1024 * 1024
SOURCE_LIMIT = 1024 * 1024


def _metadata_limit(schema):
    if schema is None:
        return METADATA_LIMIT
    if schema == "scoped-base-manifest":
        return SCOPED_BASE_MANIFEST_LIMIT
    raise ValueError("unknown bounded metadata schema")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def logical_child(root, value):
    """Join identities without consulting the current machine's filesystem."""
    identity(root)
    if isinstance(value, str) and PureWindowsPath(value).is_absolute():
        return identity(value)
    return identity(root.rstrip("/") + "/" + "/".join(relative_parts(value)))


@dataclass(frozen=True)
class FileBinding:
    logical: str
    physical: str
    sha256: str
    role: str


class BoundIO:
    """Exact-file resolver with role enforcement; no prefix or implicit fallback.

    ``check`` is the owner's real cancellation/deadline callback and is called
    before reads, every hash block, and every numerical validation block.
    Construction only validates/stat-checks paths, and never reads payloads.
    """
    def __init__(self, bindings, check):
        if not callable(check):
            raise TypeError("a real cancellation/deadline callback is required")
        rows, physical = {}, set()
        for binding in bindings:
            if not isinstance(binding, FileBinding):
                raise TypeError("expected FileBinding")
            logical = identity(binding.logical)
            sha(binding.sha256)
            if binding.role not in ("metadata", "source", "payload"):
                raise ValueError("unknown binding role")
            local = checked_path(Path(binding.physical))
            key = str(local).casefold()
            if logical in rows or key in physical:
                raise ValueError("duplicate logical or ambiguous physical binding")
            physical.add(key)
            rows[logical] = FileBinding(logical, local.as_posix(), binding.sha256, binding.role)
        if not rows or len(rows) > 8192:
            raise ValueError("finite binding census exceeded")
        self.bindings = MappingProxyType(rows)
        self.check = check

    def resolve(self, logical, role):
        self.check()
        row = self.bindings[identity(logical)]
        if row.role != role:
            raise ValueError("binding role mismatch: " + logical)
        path = checked_path(Path(row.physical))
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("bound input must be a regular file")
        return path

    def verify(self, logical, role, expected=None, *, metadata_schema=None):
        if metadata_schema is not None and role != "metadata":
            raise ValueError("metadata schema requires metadata role")
        metadata_limit = _metadata_limit(metadata_schema)
        row = self.bindings[identity(logical)]
        if expected is not None and sha(expected) != row.sha256:
            raise ValueError("declaration differs from bound pin: " + logical)
        path = self.resolve(logical, role)
        before = path.stat()
        limit = metadata_limit if role == "metadata" else SOURCE_LIMIT if role == "source" else None
        if limit is not None and before.st_size > limit:
            raise ValueError("bounded metadata/source size exceeded")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                self.check()
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        after = path.stat()
        if ((before.st_size, before.st_mtime_ns, before.st_ino) !=
                (after.st_size, after.st_mtime_ns, after.st_ino)):
            raise ValueError("bound input changed during read")
        if digest.hexdigest() != row.sha256:
            raise ValueError("bound input SHA mismatch: " + logical)
        return path

    def read_json(self, logical, *, expected=None, signed=False, metadata_schema=None):
        # Read once, then validate bytes BEFORE interpreting signed paths.
        limit = _metadata_limit(metadata_schema)
        row = self.bindings[identity(logical)]
        if expected is not None and sha(expected) != row.sha256:
            raise ValueError("declaration differs from bound metadata pin")
        path = self.resolve(logical, "metadata")
        if path.stat().st_size > limit:
            raise ValueError("bounded metadata size exceeded")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        self.check()
        if len(data) > limit or hashlib.sha256(data).hexdigest() != row.sha256:
            raise ValueError("original metadata SHA mismatch")
        result = json_bytes(data)
        if signed and (not isinstance(result, dict) or canonical_hash(
                {k: v for k, v in result.items() if k != "signature"}) != result.get("signature")):
            raise ValueError("original canonical signature mismatch")
        return result

    def pin(self, root, declaration, role):
        if not isinstance(declaration, dict) or {"path", "sha256"} - set(declaration):
            raise ValueError("Every source/artifact requires path and sha256")
        logical = logical_child(root, declaration["path"])
        return self.verify(logical, role, declaration["sha256"])

    def provenance(self, root, declaration):
        provenance = declaration.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("source_pins"):
            raise ValueError("Derived matrix requires nonempty provenance source_pins")
        for pin in provenance["source_pins"]:
            logical = logical_child(root, pin["path"])
            # Provenance may legitimately reference metadata or raw payloads.
            self.pin(root, pin, self.bindings[logical].role)

    def original_provider_source(self, logical):
        return self.verify(logical, "source", ORIGINAL_PROVIDER_SHA256)


def numpy_backend():
    """Explicit optional dependency activation; never called by planning/import."""
    import numpy
    return numpy


def _close_array(value):
    mapping = getattr(value, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _names(names):
    if (not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
            or len(names) != len(set(names))):
        raise ValueError("Invalid/duplicate matrix columns")


def read_matrix(io, root, declaration, vocab, metadata_hash, vocabulary_hash, *, np):
    """Port of CompleteProvider's _matrix; caller owns successful mmap lifetime.

    No transform/replacement is performed here. The declaration is unchanged.
    The injected numerical backend must implement the NumPy operations used by
    the retained reader; numpy_backend() supplies it at execution time.
    """
    if (declaration.get("metadata_sha256") != metadata_hash
            or declaration.get("vocabulary_sha256") != vocabulary_hash):
        raise ValueError("Feature matrix metadata/vocabulary binding mismatch")
    path = io.pin(root, declaration, "payload")
    io.check()
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if (values.ndim != 2 or len(values) != vocab or values.shape[1] < 1 or values.dtype.kind != "f"
                or list(values.shape) != declaration.get("shape") or str(values.dtype) != declaration.get("dtype")
                or np.any(values[:3] != 0)):
            raise ValueError("Feature matrix vocabulary/dtype/shape/special rows mismatch")
        for begin in range(0, vocab, 4096):
            io.check()
            if not np.isfinite(values[begin:begin + 4096]).all():
                raise ValueError("Nonfinite feature matrix")
        _names(declaration["columns"])
        if values.shape[1] != len(declaration["columns"]):
            raise ValueError("Feature matrix named width mismatch")
        return values
    except BaseException:
        _close_array(values)
        raise


def load_scoped_base_data(io, data_root, feature_root, *, manifest_sha256, np, baseline):
    """Explicit-I/O port of retained _scoped_base_data, with original predicates.

    ``baseline`` supplies retained FeatureData/shared.load_data/_columns/
    ROWKEYS_ALGORITHM/make_row_keys_hash. Its source admission/import is the
    composer's responsibility; no implicit import or sys.path modification here.
    This returns components, NOT an admitted complete provider.
    """
    manifest = io.read_json(logical_child(feature_root, "manifest.json"), expected=manifest_sha256,
                            metadata_schema="scoped-base-manifest")
    for name in ("metadata.json", "train.npz", "val.npz"):
        io.verify(logical_child(data_root, name), "metadata" if name == "metadata.json" else "payload",
                  manifest["inputs"][name]["sha256"])
    metadata = io.read_json(logical_child(data_root, "metadata.json"))
    vocab, context = int(metadata["vocab_size"]), int(metadata["context"])
    if vocab <= 3:
        raise ValueError("Empty known catalog")
    matrices = {}
    try:
        for name in ("item_features.npy", "train_user_features.npy", "val_user_features.npy"):
            declaration = manifest["outputs"][name]
            path = io.verify(logical_child(feature_root, name), "payload", declaration["sha256"])
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                if (len(shape) != 2 or list(shape) != declaration["shape"] or dtype.kind != "f"
                        or str(dtype) != declaration["dtype"] or fortran):
                    raise ValueError("Feature dtype/shape mismatch: " + name)
                remaining = int(np.prod(shape)) * dtype.itemsize
                while remaining:
                    io.check()
                    block = stream.read(min(16 << 20, remaining))
                    if not block or len(block) % dtype.itemsize or not np.isfinite(np.frombuffer(block, dtype=dtype)).all():
                        raise ValueError("Truncated/nonfinite feature payload: " + name)
                    remaining -= len(block)
                if stream.read(1):
                    raise ValueError("Trailing feature payload: " + name)
            io.check()
            matrices[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        items = matrices["item_features.npy"]
        if len(items) != vocab or np.any(items[:3] != 0):
            raise ValueError("Item features require matching vocabulary and zero special rows")
        splits = []
        for split in ("train", "val"):
            io.check()
            path = io.resolve(logical_child(data_root, split + ".npz"), "payload")
            splits.append(baseline.FeatureData(baseline.shared.load_data(
                path, vocab, context, require_keys=True, unique_keys=split == "val")))
        train, val = splits
        user_dim = matrices["train_user_features.npy"].shape[1]
        baseline._columns(manifest["feature_columns"], items.shape[1], user_dim)
        if (canonical_hash(manifest["feature_columns"]) != manifest["schema_hash"]
                or manifest.get("rowkeys_algorithm") != baseline.ROWKEYS_ALGORITHM):
            raise ValueError("Feature schema/row algorithm mismatch")
        for split, data in (("train", train), ("val", val)):
            io.check()
            user = matrices[split + "_user_features.npy"]
            if user.shape != (len(data["y"]), user_dim) or manifest["row_keys_sha256"][split] != baseline.make_row_keys_hash(data):
                raise ValueError("Feature row identity/width mismatch: " + split)
            data["user_features"] = user
            for value in data.values():
                value.setflags(write=False)
        return metadata, train, val, items, manifest
    except BaseException:
        for value in matrices.values():
            _close_array(value)
        raise
