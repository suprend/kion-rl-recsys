"""Bounded portable input validation; numerical imports are deliberately lazy."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import zipfile

MAX_FILE = 128 << 20
MAX_TOTAL = 1 << 30
DTYPES = {"float32": ("<f4", 4), "float64": ("<f8", 8),
          "int32": ("<i4", 4), "int64": ("<i8", 8), "bool": ("|b1", 1)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, max_bytes=8 << 20):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= max_bytes, "Missing or oversized JSON: " + path.name)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=invalid_constant)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def checked_path(root, relative):
    require(type(relative) is str and bool(relative), "Nonempty relative POSIX path required")
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and "\\" not in relative and ":" not in relative
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Unsafe portable path: " + relative)
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*pure.parts)
    require(path.resolve(strict=True).is_relative_to(root), "Path escapes input root: " + relative)
    require(path.is_file(), "Input is not a regular file: " + relative)
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "Symlinks are forbidden in portable inputs")
        cursor = cursor.parent
    return path


def verify_descriptor(root, entry, max_bytes=MAX_FILE):
    require(type(entry) is dict, "File descriptor must be an object")
    require(type(entry.get("bytes")) is int and 0 < entry["bytes"] <= max_bytes, "Invalid file byte budget")
    require(type(entry.get("sha256")) is str and re.fullmatch("[0-9a-f]{64}", entry["sha256"]),
            "Invalid SHA256 descriptor")
    path = checked_path(root, entry.get("path"))
    require(path.stat().st_size == entry["bytes"], "File size changed: " + entry["path"])
    require(sha256(path) == entry["sha256"], "File hash changed: " + entry["path"])
    return path


def npy_header(path):
    """Parse a safe NPY header using only stdlib, never pickle or eval."""
    with Path(path).open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "Not an NPY file")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
        count = 2 if version == b"\x01\x00" else 4
        length_bytes = stream.read(count)
        require(len(length_bytes) == count, "Truncated NPY length")
        length = struct.unpack("<H" if count == 2 else "<I", length_bytes)[0]
        require(0 < length <= 65536, "Oversized NPY header")
        header_bytes = stream.read(length)
        require(len(header_bytes) == length and header_bytes.endswith(b"\n"), "Truncated NPY header")
        header = ast.literal_eval(header_bytes.decode("utf-8" if version == b"\x03\x00" else "latin1"))
        require(type(header) is dict and set(header) == {"descr", "fortran_order", "shape"}, "Invalid NPY header fields")
        require(header["fortran_order"] is False, "Only C-order NPY arrays are supported")
        shape = header["shape"]
        require(type(shape) is tuple and 1 <= len(shape) <= 4
                and all(type(value) is int and 0 < value <= 1000000 for value in shape), "Invalid NPY shape")
        matches = [(name, size) for name, (descriptor, size) in DTYPES.items() if descriptor == header["descr"]]
        require(len(matches) == 1, "Unsupported NPY dtype (object/structured/big-endian arrays forbidden)")
        dtype, size = matches[0]
        elements = math.prod(shape)
        require(elements <= MAX_FILE // size, "NPY element budget exceeded")
        offset = stream.tell()
    require(Path(path).stat().st_size == offset + elements * size, "NPY payload length mismatch")
    return {"dtype": dtype, "shape": list(shape), "offset": offset}


def verify_array(root, entry):
    path = verify_descriptor(root, entry)
    header = npy_header(path)
    require(header["dtype"] == entry.get("dtype") and header["shape"] == entry.get("shape"),
            "NPY descriptor/header mismatch: " + entry["path"])
    return path


def verify_torch_archive(path):
    """Reject archives that could exceed the inference-only tensor budget."""
    require(zipfile.is_zipfile(path), "Only current ZIP-format torch state dictionaries are supported")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(0 < len(members) <= 512, "Torch archive member budget exceeded")
        names = set()
        total = 0
        roots = set()
        for info in members:
            pure = PurePosixPath(info.filename)
            require(not pure.is_absolute() and "\\" not in info.filename and ":" not in info.filename
                    and all(part not in ("", ".", "..") for part in info.filename.split("/")), "Unsafe torch archive member")
            require(info.filename not in names and not info.is_dir(), "Duplicate/directory torch archive member")
            names.add(info.filename)
            roots.add(pure.parts[0])
            require(len(pure.parts) >= 2 and info.compress_type == zipfile.ZIP_STORED,
                    "Only uncompressed torch state archives are supported")
            relative = "/".join(pure.parts[1:])
            require(relative in {"data.pkl", "byteorder", "version", ".data/serialization_id", ".format_version", ".storage_alignment"}
                    or re.fullmatch(r"data/[0-9]+", relative), "Unexpected torch archive member")
            require(0 <= info.file_size <= MAX_FILE, "Torch member byte budget exceeded")
            total += info.file_size
        require(len(roots) == 1 and total <= MAX_FILE, "Torch archive payload budget exceeded")
        require(any(name.endswith("/data.pkl") for name in names), "Torch tensor metadata missing")


def safe_torch_state(torch, root, descriptor):
    """Load only a hashed, bounded, dense float32 tensor mapping, on CPU."""
    require(descriptor.get("format") == "torch-state-dict-v1", "Inference-only torch state format required")
    path = verify_descriptor(root, descriptor)
    verify_torch_archive(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(type(state) is dict and 0 < len(state) <= 256, "Pure tensor state dictionary required")
    total = 0
    for key, tensor in state.items():
        require(type(key) is str and isinstance(tensor, torch.Tensor), "Only string/tensor state entries are supported")
        require(tensor.layout == torch.strided and tensor.device.type == "cpu" and tensor.dtype == torch.float32,
                "State tensors must be dense CPU float32")
        require(bool(torch.isfinite(tensor).all()), "Nonfinite state tensor")
        total += tensor.numel()
    require(total <= MAX_FILE // 4, "State tensor element budget exceeded")
    return state


def load_arrays(np, fixture_root, manifest, names=None):
    arrays = {}
    try:
        entries = manifest["arrays"]
        for name in entries if names is None else names:
            path = verify_array(fixture_root, entries[name])
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            require(str(value.dtype) == entries[name]["dtype"] and list(value.shape) == entries[name]["shape"], "Array changed after validation")
            if value.dtype.kind == "f":
                require(bool(np.isfinite(value).all()), "Nonfinite fixture array: " + name)
            arrays[name] = value
        return arrays
    except BaseException:
        close_arrays(arrays)
        raise


def close_arrays(arrays):
    for value in arrays.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()
