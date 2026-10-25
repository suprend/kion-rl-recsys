"""Exact one-row QueryProfiles codec and bounded SQLite preparation store.

No provider/ML imports. Callers own provider admission, input/source hashes,
runtime/RAM/wall guards and training integration. SQLite stores are derived data.
"""
from collections import Counter, OrderedDict
import hashlib
import json
import math
import operator
from pathlib import Path
import shutil
import sqlite3
import struct
import zlib

import numpy as np

MAX_RECORD_BYTES = 64 << 20
MAX_METADATA_BYTES = 1 << 20
MAX_NODES = 50000
MAGIC = b"KQSP01\0\0"
HEADER = struct.Struct("<8sII")
CONTRACT = "exact-query-profile-sqlite-v1"
PROFILE_FIELDS = frozenset(("owner", "owner_schema_hash", "n", "origins", "user_ids",
    "query_cutoffs", "tokens", "mask", "genre_has_unseen", "interests", "keys",
    "known_n", "distinct_n", "text", "user_features"))
ROLES = ("actor", "director", "studio", "franchise")
INTEREST_FIELDS = frozenset(("centers", "incidence", "support", "mass", "coherence",
    "valid", "k_eff", "n_known", "reason", "objective", "gain", "max_pair_cosine"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def codec_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _limit(limit):
    require(type(limit) is int and 1024 <= limit <= MAX_RECORD_BYTES, "Record cap must be 1KiB..64MiB")
    return limit


def _dtype(value):
    dtype = np.dtype(value)
    require(not dtype.hasobject and dtype.fields is None and dtype.subdtype is None and dtype.metadata is None
        and dtype.kind in "biufc" and 0 < dtype.itemsize <= 16, "Unsupported numeric dtype")
    return dtype


def encode_value(value, *, max_record_bytes=MAX_RECORD_BYTES):
    """Typed tree metadata plus raw bytes; preserves Python/NumPy scalar bits."""
    limit = _limit(max_record_bytes)
    raw = bytearray()
    active, nodes, metadata_bound = set(), 0, 0

    def block(data):
        start = len(raw)
        require(start + len(data) <= limit, "Record binary budget exceeded")
        raw.extend(data)
        return [start, len(data)]

    def visit(item, depth=0):
        nonlocal nodes, metadata_bound
        nodes += 1
        require(depth <= 32 and nodes <= MAX_NODES, "Record tree budget exceeded")
        metadata_bound += ((256 if isinstance(item, np.ndarray) else 64) + (6 * len(item) if type(item) is str else 0)
            + ((item.bit_length() * 30103 // 100000 + 3) if type(item) is int else 0))
        require(metadata_bound <= MAX_METADATA_BYTES, "Record metadata budget exceeded")
        if isinstance(item, np.ndarray):
            dtype = _dtype(item.dtype)
            require(item.ndim <= 8 and item.nbytes <= limit - len(raw), "Array budget exceeded")
            return ["array", dtype.str, list(item.shape), block(item.tobytes(order="C"))]
        if isinstance(item, np.generic):
            dtype = _dtype(item.dtype)
            return ["numpy", dtype.str, block(item.tobytes())]
        if item is None:
            return ["none"]
        if type(item) is bool:
            return ["bool", item]
        if type(item) is int:
            require(item.bit_length() <= 4096, "Integer budget exceeded")
            return ["int", str(item)]
        if type(item) is float:
            return ["float", block(struct.pack("<d", item))]
        if type(item) is str:
            require(len(item) <= MAX_METADATA_BYTES, "String budget exceeded")
            return ["str", item]
        if type(item) is bytes:
            return ["bytes", block(item)]
        kind = ("counter" if type(item) is Counter else "ordered" if type(item) is OrderedDict
                else "dict" if type(item) is dict else "list" if type(item) is list
                else "tuple" if type(item) is tuple else None)
        require(kind is not None, "Unsupported record value type: " + type(item).__name__)
        require(id(item) not in active, "Cyclic record")
        active.add(id(item))
        try:
            children = ([[visit(k, depth + 1), visit(v, depth + 1)] for k, v in item.items()]
                if kind in ("counter", "ordered", "dict") else [visit(v, depth + 1) for v in item])
            return [kind, children]
        finally:
            active.remove(id(item))
    metadata = json.dumps(visit(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    require(len(metadata) <= MAX_METADATA_BYTES, "Record metadata budget exceeded")
    prefix = HEADER.pack(MAGIC, len(metadata), len(raw))
    require(len(prefix) + len(metadata) + len(raw) + 32 <= limit, "Record total budget exceeded")
    body = prefix + metadata + raw
    return body + hashlib.sha256(body).digest()


def decode_value(payload, *, max_record_bytes=MAX_RECORD_BYTES):
    """Fresh owned arrays/containers, with strict bounds and envelope digest."""
    limit = _limit(max_record_bytes)
    require(type(payload) is bytes and HEADER.size + 32 <= len(payload) <= limit, "Invalid record size")
    require(hashlib.sha256(payload[:-32]).digest() == payload[-32:], "Record checksum mismatch")
    magic, meta_size, raw_size = HEADER.unpack_from(payload)
    require(magic == MAGIC and meta_size <= MAX_METADATA_BYTES
        and HEADER.size + meta_size + raw_size + 32 == len(payload), "Invalid record envelope")
    try:
        root = json.loads(payload[HEADER.size:HEADER.size + meta_size].decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Malformed record metadata") from error
    raw = memoryview(payload)[HEADER.size + meta_size:-32]
    nodes, cursor = 0, 0

    def take(declaration):
        nonlocal cursor
        require(type(declaration) is list and len(declaration) == 2, "Invalid binary declaration")
        start, count = declaration
        require(type(start) is type(count) is int and start == cursor and count >= 0
            and start + count <= len(raw), "Invalid binary offset/length")
        cursor += count
        return raw[start:start + count]

    def visit(node, depth=0):
        nonlocal nodes
        nodes += 1
        require(depth <= 32 and nodes <= MAX_NODES and type(node) is list and node
            and type(node[0]) is str, "Invalid typed tree")
        tag = node[0]
        if tag == "none":
            require(len(node) == 1, "Invalid None")
            return None
        if tag in ("bool", "str", "int"):
            require(len(node) == 2, "Invalid scalar")
            if tag == "bool":
                require(type(node[1]) is bool, "Invalid bool")
                return node[1]
            require(type(node[1]) is str and len(node[1]) <= MAX_METADATA_BYTES, "Invalid scalar string")
            if tag == "str":
                return node[1]
            require(len(node[1]) <= 1235, "Integer budget exceeded")
            value = int(node[1])
            require(value.bit_length() <= 4096 and str(value) == node[1], "Noncanonical integer")
            return value
        if tag in ("float", "bytes"):
            require(len(node) == 2, "Invalid raw scalar")
            data = take(node[1])
            if tag == "bytes":
                return bytes(data)
            require(len(data) == 8, "Invalid float bytes")
            return struct.unpack("<d", data)[0]
        if tag in ("array", "numpy"):
            require(len(node) == (4 if tag == "array" else 3), "Invalid numeric declaration")
            dtype = _dtype(node[1])
            shape = node[2] if tag == "array" else []
            require(type(shape) is list and len(shape) <= 8 and all(type(n) is int and n >= 0 for n in shape),
                "Invalid array shape")
            size = math.prod(shape)
            data = take(node[-1])
            require(size * dtype.itemsize == len(data), "Array byte count mismatch")
            values = np.frombuffer(data, dtype=dtype, count=size).copy()
            return values.reshape(tuple(shape)) if tag == "array" else values[0]
        require(tag in ("dict", "ordered", "counter", "list", "tuple")
            and len(node) == 2 and type(node[1]) is list, "Unknown record node")
        if tag in ("list", "tuple"):
            values = [visit(v, depth + 1) for v in node[1]]
            return values if tag == "list" else tuple(values)
        result = {"dict": dict, "ordered": OrderedDict, "counter": Counter}[tag]()
        for pair in node[1]:
            require(type(pair) is list and len(pair) == 2, "Invalid mapping entry")
            key, value = visit(pair[0], depth + 1), visit(pair[1], depth + 1)
            try:
                require(key not in result, "Duplicate mapping key")
                result[key] = value
            except TypeError as error:
                raise ValueError("Unhashable mapping key") from error
        return result
    result = visit(root)
    require(cursor == len(raw), "Unreferenced binary bytes")
    return result


def normalize_key(key):
    require(type(key) is tuple and len(key) == 4, "Exact query key tuple required")
    values = []
    for item in key[:3]:
        require(not isinstance(item, (bool, np.bool_)), "Boolean query key")
        try:
            number = operator.index(item)
        except TypeError as error:
            raise ValueError("Integer query key required") from error
        require(0 <= number < (1 << 63), "Query key outside SQLite integer range")
        values.append(number)
    require(values[1] > 0 and 1 <= values[2] <= 64 and type(key[3]) is bytes
        and len(key[3]) in (64 * 4, 64 * 8), "Invalid H63 query key")
    return (*values, key[3])


def validate_profile(profile, user_tail, *, schema_hash, personal=None, key=None):
    """Validate frozen one-row layout without importing the provider module."""
    require(set(vars(profile)) == PROFILE_FIELDS and profile.owner_schema_hash == schema_hash,
        "Profile fields/schema mismatch")
    if personal is not None:
        require(personal.schema_hash == schema_hash and profile.owner is personal._owner, "Profile owner mismatch")
        if hasattr(personal, "_check_profiles"):
            personal._check_profiles(profile)
    def array(value, shape, kinds):
        require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kinds
            and not value.dtype.hasobject, "Profile array shape/dtype mismatch")
    for name in ("n", "origins", "user_ids", "query_cutoffs"):
        array(getattr(profile, name), (1,), "iu")
    array(profile.tokens, (1, 63), "iu")
    array(profile.mask, (1, 63), "b")
    array(profile.genre_has_unseen, (1,), "b")
    require(isinstance(profile.user_features, np.ndarray) and profile.user_features.ndim == 2
        and profile.user_features.shape[0] == 1, "Invalid personal user feature layout")
    array(profile.user_features, profile.user_features.shape, "f")
    n = int(profile.n[0])
    require(0 <= n <= 63 and int(profile.user_ids[0]) >= 0
        and int(profile.query_cutoffs[0]) <= int(profile.origins[0])
        and np.array_equal(profile.mask, np.arange(63)[None] < n)
        and np.all(profile.tokens[~profile.mask] == 0)
        and np.all(profile.tokens[profile.mask] >= 2), "Profile history/clock mismatch")
    if personal is not None and hasattr(personal, "vocab_size"):
        require(np.all(profile.tokens < personal.vocab_size), "Profile token outside vocabulary")
    for name in ("keys", "known_n", "distinct_n"):
        require(type(getattr(profile, name)) is dict and set(getattr(profile, name)) == set(ROLES), "Profile roles mismatch")
    for role in ROLES:
        require(type(profile.keys[role]) is list and len(profile.keys[role]) == 1
            and type(profile.keys[role][0]) is Counter
            and all(type(k) is int and type(v) is int and v > 0 for k, v in profile.keys[role][0].items()),
            "Profile Counter mismatch")
        array(profile.known_n[role], (1,), "iu")
        array(profile.distinct_n[role], (1,), "iu")
    require(type(profile.interests) is dict and set(profile.interests) == {1, 2, 3}, "Interest modes mismatch")
    genre_width = None
    if personal is not None and hasattr(personal, "genre_incidence"):
        genre_width = personal.genre_incidence.shape[1]
    for states in profile.interests.values():
        require(type(states) is list and len(states) == 1 and type(states[0]) is dict
            and set(states[0]) == INTEREST_FIELDS, "Interest fields mismatch")
        state = states[0]
        centers = state["centers"]
        require(isinstance(centers, np.ndarray) and centers.ndim == 2 and centers.shape[0] == 3, "Interest centers mismatch")
        width = centers.shape[1] if genre_width is None else genre_width
        for name in ("centers", "incidence"):
            array(state[name], (3, width), "f")
        for name in ("support", "mass", "coherence"):
            array(state[name], (3,), "iuf")
        array(state["valid"], (3,), "b")
        for name in INTEREST_FIELDS - {"centers", "incidence", "support", "mass", "coherence", "valid"}:
            require(np.shape(state[name]) == (), "Interest scalar mismatch")
    require(type(profile.text) is dict, "Profile text mismatch")
    if personal is not None and hasattr(personal, "text_matrices"):
        require(set(profile.text) == set(personal.text_matrices), "Text field mismatch")
    blocks = [profile.user_features[0]]
    for field in sorted(profile.text):
        value = profile.text[field]
        require(type(value) is dict and set(value) == {"profile", "n_known", "mean_norm", "valid", "zero_norm"},
            "Text profile fields mismatch")
        require(isinstance(value["profile"], np.ndarray) and value["profile"].ndim == 2
            and value["profile"].shape[0] == 1, "Text profile layout mismatch")
        width = value["profile"].shape[1]
        if personal is not None and hasattr(personal, "text_matrices"):
            require(width == personal.text_matrices[field].shape[1], "Text width mismatch")
        array(value["profile"], (1, width), "f")
        array(value["n_known"], (1,), "iu")
        array(value["mean_norm"], (1,), "f")
        for name in ("valid", "zero_norm"):
            array(value[name], (1,), "b")
        blocks += [value["profile"][0], np.array([value[name][0] for name in ("n_known", "mean_norm", "valid", "zero_norm")])]
    expected = np.concatenate(blocks).astype(np.float32)
    array(user_tail, expected.shape, "f")
    require(user_tail.dtype == np.dtype(np.float32) and user_tail.tobytes() == expected.tobytes(),
        "Stored user tail differs from original pre-transform assembly")
    if key is not None:
        uid, origin, length, x = normalize_key(key)
        require(uid == int(profile.user_ids[0]) and origin == int(profile.origins[0])
            and length == n + 1 and int(profile.query_cutoffs[0]) == origin, "Query key/profile mismatch")
        tokens = np.concatenate((np.array([1]), profile.tokens[0]))
        require(tokens.astype("<i4" if len(x) == 256 else "<i8").tobytes() == x, "Query x/profile mismatch")


def encode_record(profile, user_tail, *, schema_hash, key=None, max_record_bytes=MAX_RECORD_BYTES):
    validate_profile(profile, user_tail, schema_hash=schema_hash, key=key)
    fields = {name: value for name, value in vars(profile).items() if name != "owner"}
    return encode_value({"profile": fields, "user_tail": user_tail}, max_record_bytes=max_record_bytes)


def decode_record(payload, *, personal, profile_type, schema_hash, key=None, max_record_bytes=MAX_RECORD_BYTES):
    require(personal.schema_hash == schema_hash, "Personal schema mismatch")
    value = decode_value(payload, max_record_bytes=max_record_bytes)
    require(type(value) is dict and set(value) == {"profile", "user_tail"}
        and type(value["profile"]) is dict and set(value["profile"]) == PROFILE_FIELDS - {"owner"},
        "Stored profile record layout mismatch")
    profile = profile_type(owner=personal._owner, **value["profile"])
    validate_profile(profile, value["user_tail"], schema_hash=schema_hash, personal=personal, key=key)
    return profile, value["user_tail"]


def _binding(schema_hash, source_pins, max_record_bytes, compress, max_total_bytes, reserve):
    require(type(schema_hash) is str and schema_hash and type(source_pins) is dict and source_pins, "Explicit schema/source binding required")
    require(all(type(k) is str and type(v) is str and len(v) == 64
        and all(c in "0123456789abcdef" for c in v) for k, v in source_pins.items()), "Invalid source pins")
    _limit(max_record_bytes)
    require(type(compress) is bool and type(max_total_bytes) is int and max_total_bytes >= 1 << 20
        and type(reserve) is int and reserve >= 0, "Explicit disk policy required")
    return {"contract": CONTRACT, "schema_hash": schema_hash, "source_pins": dict(source_pins),
        "codec_sha256": codec_sha256(), "max_record_bytes": max_record_bytes, "compress": compress,
        "max_total_bytes": max_total_bytes, "free_space_reserve_bytes": reserve}


def _configure(connection):
    connection.execute("PRAGMA cache_size=-2048")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA busy_timeout=0")


def _read_metadata(connection):
    require(connection.execute("SELECT count(*) FROM metadata").fetchone()[0] <= 4,
        "Too many metadata rows")
    for name_type, name_size, value_type, value_size in connection.execute(
            "SELECT typeof(name),length(CAST(name AS BLOB)),typeof(value),length(CAST(value AS BLOB)) FROM metadata"):
        require(name_type == value_type == "text" and type(name_size) is type(value_size) is int
            and 0 < name_size <= 128 and 0 <= value_size <= MAX_METADATA_BYTES,
            "Metadata size/type budget exceeded")
    return dict(connection.execute("SELECT name,value FROM metadata"))


class QueryStoreWriter:
    """Single owner; explicit checkpoint/resume. Completed stores are immutable."""
    def __init__(self, path, *, schema_hash, source_pins, max_total_bytes,
                 free_space_reserve_bytes, max_record_bytes=MAX_RECORD_BYTES, compress=True, resume=False):
        self.path = Path(path).resolve()
        self.binding = _binding(schema_hash, source_pins, max_record_bytes, compress, max_total_bytes, free_space_reserve_bytes)
        self.closed = False
        require(shutil.disk_usage(self.path.parent).free >= free_space_reserve_bytes + 65536,
            "Store initial free-space reserve exhausted")
        if not resume:
            with self.path.open("xb"):
                pass
        require(self.path.is_file(), "Missing resumable store")
        self.connection = sqlite3.connect(self.path, isolation_level="DEFERRED")
        try:
            _configure(self.connection)
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            page = self.connection.execute("PRAGMA page_size").fetchone()[0]
            # A rollback journal can approach the database size. Reserve a full
            # second database plus1MiB, instead of calling payload bytes a cap.
            pages = (max_total_bytes - (1 << 20)) // (2 * page)
            require(pages >= 4, "Disk cap too small for SQLite plus rollback reserve")
            self.connection.execute("PRAGMA max_page_count=" + str(pages))
            if resume:
                meta = _read_metadata(self.connection)
                require(json.loads(meta["binding"]) == self.binding and meta["state"] == "writing",
                    "Resume binding/state mismatch")
            else:
                self.connection.execute("CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL)")
                self.connection.execute("CREATE TABLE profiles(uid INTEGER NOT NULL,origin INTEGER NOT NULL,query_len INTEGER NOT NULL,x BLOB NOT NULL,payload BLOB NOT NULL,digest TEXT NOT NULL,raw_bytes INTEGER NOT NULL,PRIMARY KEY(uid,origin,query_len,x)) WITHOUT ROWID")
                self.connection.executemany("INSERT INTO metadata VALUES(?,?)", [
                    ("binding", json.dumps(self.binding, sort_keys=True, separators=(",", ":"))), ("state", "writing")])
                self.connection.commit()
            self._refresh_counts()
            self._disk_check(0)
        except BaseException:
            self.connection.close()
            self.closed = True
            raise

    def _open(self):
        require(not self.closed, "Store writer closed")

    def _refresh_counts(self):
        self._count, self._stored, self._raw = self.connection.execute(
            "SELECT count(*),coalesce(sum(length(payload)),0),coalesce(sum(raw_bytes),0) FROM profiles").fetchone()

    def _disk_check(self, pending):
        size = sum(p.stat().st_size for p in (self.path, Path(str(self.path) + "-journal")) if p.exists())
        require(size + pending <= self.binding["max_total_bytes"], "Store total disk cap exceeded")
        require(shutil.disk_usage(self.path.parent).free >= self.binding["free_space_reserve_bytes"] + pending,
            "Store free-space reserve exhausted")

    def contains(self, key):
        self._open()
        return self.connection.execute("SELECT 1 FROM profiles WHERE uid=? AND origin=? AND query_len=? AND x=?",
            normalize_key(key)).fetchone() is not None

    def put(self, key, profile, user_tail):
        self._open()
        key = normalize_key(key)
        raw = encode_record(profile, user_tail, schema_hash=self.binding["schema_hash"], key=key,
            max_record_bytes=self.binding["max_record_bytes"])
        payload = zlib.compress(raw, 3) if self.binding["compress"] else raw
        require(len(payload) <= self.binding["max_record_bytes"], "Stored record exceeds cap")
        digest = hashlib.sha256(payload).hexdigest()
        previous = self.connection.execute("SELECT digest,raw_bytes FROM profiles WHERE uid=? AND origin=? AND query_len=? AND x=?", key).fetchone()
        if previous is not None:
            require(previous == (digest, len(raw)), "Duplicate query has different exact payload")
            return False
        self._disk_check(2 * (len(payload) + len(key[3]) + 16384))
        try:
            self.connection.execute("INSERT INTO profiles VALUES(?,?,?,?,?,?,?)", (*key, payload, digest, len(raw)))
            self._disk_check(0)
        except BaseException:
            # A failed transaction is not allowed to publish a completed store.
            self.connection.rollback()
            self._refresh_counts()
            raise
        self._count += 1
        self._stored += len(payload)
        self._raw += len(raw)
        return True

    def stats(self):
        self._open()
        page = self.connection.execute("PRAGMA page_size").fetchone()[0]
        return {"unique_count": self._count, "stored_payload_bytes": self._stored, "raw_payload_bytes": self._raw,
            "database_bytes": self.path.stat().st_size,
            "database_allocated_cap_bytes": page * self.connection.execute("PRAGMA max_page_count").fetchone()[0],
            "max_total_bytes": self.binding["max_total_bytes"], "codec_sha256": self.binding["codec_sha256"]}

    def checkpoint(self):
        self._open()
        self._disk_check(0)
        self.connection.commit()
        self._disk_check(0)
        return self.stats()

    def finish(self):
        self._open()
        self.checkpoint()
        result = self.stats()
        self._disk_check(32768)
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES('counts',?)",
            (json.dumps({k: result[k] for k in ("unique_count", "stored_payload_bytes", "raw_payload_bytes")}),))
        self.connection.execute("UPDATE metadata SET value='completed' WHERE name='state'")
        self.connection.commit()
        self._disk_check(0)
        self.close()
        return result

    def close(self):
        if not self.closed:
            self.connection.rollback()
            self.connection.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class QueryStoreReader:
    """Read-only exact-key lookup; no decoded cache or column-wide load."""
    def __init__(self, path, *, expected_schema_hash, expected_source_pins, max_record_bytes=MAX_RECORD_BYTES):
        self.path = Path(path).resolve()
        require(self.path.is_file(), "Missing query store")
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.closed = False
        try:
            _configure(self.connection)
            self.connection.execute("PRAGMA query_only=ON")
            self.connection.execute("BEGIN")
            meta = _read_metadata(self.connection)
            self.binding = json.loads(meta["binding"])
            binding = self.binding
            require(binding == _binding(binding["schema_hash"], binding["source_pins"],
                binding["max_record_bytes"], binding["compress"], binding["max_total_bytes"],
                binding["free_space_reserve_bytes"]), "Malformed store binding")
            require(meta["state"] == "completed" and self.binding["contract"] == CONTRACT
                and self.binding["codec_sha256"] == codec_sha256()
                and self.binding["schema_hash"] == expected_schema_hash
                and self.binding["source_pins"] == expected_source_pins, "Store completion/source/schema binding mismatch")
            require(self.binding["max_record_bytes"] <= _limit(max_record_bytes), "Reader record budget below store budget")
            self.max_record_bytes = self.binding["max_record_bytes"]
            self.counts = json.loads(meta["counts"])
            require(type(self.counts) is dict and set(self.counts) == {
                "unique_count", "stored_payload_bytes", "raw_payload_bytes"}
                and all(type(n) is int and n >= 0 for n in self.counts.values()), "Malformed store counts")
            require(self.connection.execute("SELECT count(*) FROM profiles").fetchone()[0]
                == self.counts["unique_count"], "Store row count mismatch")
        except BaseException:
            self.connection.close()
            self.closed = True
            raise

    def get(self, key, *, personal, profile_type):
        require(not self.closed, "Store reader closed")
        key = normalize_key(key)
        declaration = self.connection.execute("SELECT typeof(payload),length(payload),typeof(digest),length(digest),raw_bytes FROM profiles WHERE uid=? AND origin=? AND query_len=? AND x=?", key).fetchone()
        if declaration is None:
            return None
        require(declaration[0] == "blob" and declaration[2] == "text" and declaration[3] == 64
            and all(type(n) is int and 0 <= n <= self.max_record_bytes for n in (declaration[1], declaration[4])),
            "Stored record length/type exceeds cap")
        payload, digest = self.connection.execute("SELECT payload,digest FROM profiles WHERE uid=? AND origin=? AND query_len=? AND x=?", key).fetchone()
        require(all(c in "0123456789abcdef" for c in digest), "Malformed stored record digest")
        require(hashlib.sha256(payload).hexdigest() == digest, "Stored record checksum mismatch")
        if self.binding["compress"]:
            unpacker = zlib.decompressobj()
            try:
                raw = unpacker.decompress(payload, self.max_record_bytes + 1)
            except zlib.error as error:
                raise ValueError("Corrupt compressed record") from error
            require(len(raw) <= self.max_record_bytes and unpacker.eof
                and not unpacker.unused_data and not unpacker.unconsumed_tail, "Compressed record exceeds budget or has trailing data")
        else:
            raw = payload
        require(len(raw) == declaration[4], "Uncompressed record length mismatch")
        return decode_record(raw, personal=personal, profile_type=profile_type,
            schema_hash=self.binding["schema_hash"], key=key, max_record_bytes=self.max_record_bytes)

    def get_for_row(self, provider, split, row, *, profile_type):
        data = provider.splits[split]
        key = (int(data["user_id"][row]), provider._origin(split, row),
            int(data["lengths"][row]), data["x"][row].tobytes())
        return self.get(key, personal=provider.personal, profile_type=profile_type)

    def stats(self):
        require(not self.closed, "Store reader closed")
        return {**self.counts, "database_bytes": self.path.stat().st_size,
            "codec_sha256": self.binding["codec_sha256"]}

    def close(self):
        if not self.closed:
            self.connection.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
