#!/usr/bin/env python3
"""Bounded, explicit, stdlib-only submission packaging. Never executes payload code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import unicodedata
from urllib.parse import unquote, urlsplit
import zipfile
import zlib

FORMAT = "kion-handin-package-v1"
INTERNAL_MANIFEST = "PACKAGE_MANIFEST.json"
DEFAULT_CAP = 300 * 1024 * 1024
HARD_CAP = 300 * 1024 * 1024
MANIFEST_CAP = 8 * 1024 * 1024
MAX_FILES = 10000
CHUNK = 1024 * 1024
FIXED_DATE = (1980, 1, 1, 0, 0, 0)
DENIED_PARTS = {".git", ".svn", ".venv", "venv", "node_modules", "__pycache__",
                ".pytest_cache", ".mypy_cache", "cache", "caches", "raw", "raw_data",
                "raw-data", "history", "histories", "credentials", "secrets"}
DENIED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db", ".jsonl", ".log",
                   ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}
NUMERICAL_SUFFIXES = {".npy", ".npz"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors", ".bin", ".ckpt"}
TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".csv", ".tsv", ".yaml", ".yml",
                 ".toml", ".ini", ".cfg", ".ps1", ".sh", ".bat", ".cmd", ".html"}
TIERS = {"passed", "not_run", "requires_external_inputs", "failed"}
SECRET_PATTERNS = [re.compile(p) for p in (
    r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----",
    r"\b(?:ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{16,}",
    r"\bAKIA[A-Z0-9]{16}\b",
    r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}",
)]


class PackageError(ValueError):
    """Invalid, stale, or unsafe package input."""


def fail(message):
    raise PackageError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path, cap=MANIFEST_CAP):
    path = Path(path).absolute()
    ensure_no_links(path)
    if path.stat().st_size > cap:
        fail(f"JSON exceeds {cap} bytes: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Invalid UTF-8 JSON: {path}: {exc}")


def ensure_no_links(path, missing_leaf=False):
    """Reject symlinks and Windows reparse points on every existing ancestor."""
    path = Path(path).absolute()
    chain = list(reversed(path.parents)) + [path]
    for part in chain:
        try:
            info = part.lstat()
        except FileNotFoundError:
            if missing_leaf and part == path:
                return
            fail(f"Missing path or parent: {part}")
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            fail(f"Symlink/junction/reparse point is forbidden: {part}")


def safe_name(value):
    if not isinstance(value, str) or not value or len(value) > 240:
        fail(f"Invalid relative path: {value!r}")
    if value != unicodedata.normalize("NFC", value):
        fail(f"Path must use NFC Unicode: {value!r}")
    if "\\" in value or value.startswith("/") or any(ord(c) < 32 for c in value):
        fail(f"Unsafe path: {value!r}")
    parts = value.split("/")
    for part in parts:
        stem = part.split(".")[0].upper()
        if (part in {"", ".", ".."} or ":" in part or part.endswith((".", " "))
                or any(c in part for c in '<>"|?*')
                or stem in {"CON", "PRN", "AUX", "NUL"}
                or re.fullmatch(r"(?:COM|LPT)[1-9]", stem)):
            fail(f"Unsafe portable path: {value!r}")
    return value


def collision_key(name):
    return unicodedata.normalize("NFC", name).casefold()


def check_names(names):
    seen = set()
    for name in names:
        safe_name(name)
        key = collision_key(name)
        if key in seen:
            fail(f"Duplicate/case-colliding member: {name}")
        seen.add(key)
    for name in seen:
        parts = name.split("/")
        if any("/".join(parts[:i]) in seen for i in range(1, len(parts))):
            fail(f"File/directory member collision: {name}")


def check_payload_path(name, admission=None, reason=None):
    safe_name(name)
    parts = PurePosixPath(name).parts
    if any(p.casefold() in DENIED_PARTS for p in parts):
        fail(f"Secret/raw/history/cache path is forbidden: {name}")
    base = parts[-1].casefold()
    if base == ".env" or base.startswith(".env.") or base in {"id_rsa", "id_ed25519", "credentials.json", "secrets.json"}:
        fail(f"Secret file is forbidden: {name}")
    suffix = PurePosixPath(name).suffix.casefold()
    if suffix in DENIED_SUFFIXES:
        fail(f"Raw history, cache/database, or secret suffix is forbidden: {name}")
    if admission not in {None, "derived_fixture", "frozen_weights"}:
        fail(f"Unknown admission: {admission}")
    if suffix in NUMERICAL_SUFFIXES and admission != "derived_fixture":
        fail(f"Numerical arrays need explicit derived_fixture admission: {name}")
    if suffix in WEIGHT_SUFFIXES and admission != "frozen_weights":
        fail(f"Weights need explicit frozen_weights admission: {name}")
    if admission is not None and (not isinstance(reason, str) or not reason.strip()):
        fail(f"Admission requires an audit reason: {name}")


def source_path(roots, root_id, relative):
    if root_id not in roots:
        fail(f"Unknown source root alias: {root_id}")
    safe_name(relative)
    root = Path(roots[root_id]).absolute()
    ensure_no_links(root)
    if not root.is_dir():
        fail(f"Source root is not a directory: {root}")
    path = root.joinpath(*PurePosixPath(relative).parts)
    ensure_no_links(path)
    if not path.is_file():
        fail(f"Explicit source is not a regular file: {path}")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        fail(f"Source escapes its root: {relative}")
    return path


def digest_file(path, cap=HARD_CAP):
    ensure_no_links(path)
    before = Path(path).stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
        fail(f"Source is not a bounded regular file: {path}")
    sha, crc, size = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as stream:
        while block := stream.read(CHUNK):
            size += len(block)
            if size > cap:
                fail(f"Source grew beyond cap while reading: {path}")
            sha.update(block)
            crc = zlib.crc32(block, crc)
    after = Path(path).stat()
    ensure_no_links(path)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or size != after.st_size):
        fail(f"Source changed while reading: {path}")
    return {"size": size, "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}


def scan_text(path, member):
    suffix = PurePosixPath(member).suffix.casefold()
    if suffix not in TEXT_SUFFIXES:
        return None
    if Path(path).stat().st_size > MANIFEST_CAP:
        fail(f"Text exceeds the bounded text-audit cap: {member}")
    try:
        value = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeError:
        fail(f"Text payload is not UTF-8: {member}")
    if any(pattern.search(value) for pattern in SECRET_PATTERNS):
        fail(f"Possible credential found in selected text (value suppressed): {member}")
    return value


def _link_target(raw):
    raw = raw.strip()
    if raw.startswith("<"):
        end = raw.find(">")
        if end < 0:
            return raw
        return raw[1:end]
    return re.split(r'\s+[\"\']', raw, maxsplit=1)[0].strip()


def markdown_link_errors(texts, members):
    """Check ordinary Markdown/reference/HTML/wiki local targets in archive space."""
    # Exact casing matters on the recipient's case-sensitive filesystem.
    allowed = set(members)
    errors = []
    for member, body in texts.items():
        if not member.casefold().endswith(".md"):
            continue
        body = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", body)
        body = re.sub(r"`+[^`\n]*`+", "", body)
        definitions = {m.group(1).strip().casefold(): _link_target(m.group(2))
                       for m in re.finditer(r"(?m)^\s{0,3}\[([^\]\n]+)\]:\s*(.+)$", body)}
        targets = [_link_target(m.group(1)) for m in re.finditer(r"!?\[[^\]\n]*\]\(([^\n]*?)\)", body)]
        for match in re.finditer(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]", body):
            ref = (match.group(2) or match.group(1)).strip().casefold()
            if ref not in definitions:
                errors.append(f"{member}: undefined Markdown reference [{ref}]")
            else:
                targets.append(definitions[ref])
        targets.extend(definitions.values())
        targets.extend(m.group(1) for m in re.finditer(r"(?:href|src)\s*=\s*[\"\']([^\"\']+)[\"\']", body, re.I))
        targets.extend(m.group(1).split("|", 1)[0] for m in re.finditer(r"\[\[([^\]]+)\]\]", body))
        for target in targets:
            target = unquote(target)
            parsed = urlsplit(target)
            if parsed.scheme.casefold() in {"http", "https", "mailto"}:
                continue
            if parsed.scheme or parsed.netloc or "\\" in target or target.startswith("/"):
                errors.append(f"{member}: nonportable local link {target!r}")
                continue
            if not parsed.path:
                continue
            parts = list(PurePosixPath(member).parent.parts)
            escaped = False
            for part in parsed.path.split("/"):
                if part in {"", "."}:
                    continue
                if part == "..":
                    if not parts:
                        escaped = True
                        break
                    parts.pop()
                else:
                    parts.append(part)
            resolved = "/".join(parts)
            if escaped or resolved not in allowed:
                errors.append(f"{member}: unclosed local link {target!r} -> {resolved!r}")
    return sorted(set(errors))


def validate_tiers(tiers, members):
    if not isinstance(tiers, list) or not tiers:
        fail("acceptance_tiers must explicitly report actual passed/not_run/external/failed tiers")
    ids = set()
    for tier in tiers:
        if not isinstance(tier, dict) or not isinstance(tier.get("id"), str) or not tier["id"].strip():
            fail("Every acceptance tier needs an id")
        if tier["id"] in ids:
            fail(f"Duplicate acceptance tier: {tier['id']}")
        ids.add(tier["id"])
        if tier.get("status") not in TIERS or not isinstance(tier.get("basis"), str) or not tier["basis"].strip():
            fail(f"Tier needs an actual status and factual basis: {tier['id']}")
        evidence = tier.get("evidence_members", [])
        if not isinstance(evidence, list) or any(not isinstance(e, str) or e not in members for e in evidence):
            fail(f"Tier evidence is absent from explicit package selection: {tier['id']}")
        if tier["status"] == "passed" and not evidence:
            fail(f"Passed tier needs included evidence, not a planned claim: {tier['id']}")


def prepare_selection(selection_path):
    selection_path = Path(selection_path).absolute()
    selection = read_json(selection_path)
    if not isinstance(selection, dict) or selection.get("version") != 1:
        fail("Selection must be a version 1 object")
    roots = selection.get("roots")
    if not isinstance(roots, dict) or not roots or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", k) for k in roots):
        fail("roots must be a nonempty alias-to-absolute-directory mapping")
    for alias, value in roots.items():
        if not isinstance(value, str) or not Path(value).is_absolute():
            fail(f"Root must be absolute: {alias}")
    cap = selection.get("max_total_bytes", DEFAULT_CAP)
    if type(cap) is not int or not 0 < cap <= HARD_CAP:
        fail(f"max_total_bytes must be a positive integer <= {HARD_CAP}")
    files = selection.get("files")
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        fail(f"files must explicitly list 1..{MAX_FILES} regular files")
    check_names([INTERNAL_MANIFEST] + [f.get("member") if isinstance(f, dict) else None for f in files])
    members = [f["member"] for f in files]
    validate_tiers(selection.get("acceptance_tiers"), members)
    records, texts, seen_sources, total = [], {}, set(), 0
    for item in sorted(files, key=lambda f: f["member"]):
        if not isinstance(item.get("purpose"), str) or not item["purpose"].strip():
            fail(f"Every selected file needs a purpose: {item['member']}")
        for value in (item.get("source"), item["member"]):
            check_payload_path(value, item.get("admission"), item.get("admission_reason"))
        path = source_path(roots, item.get("root"), item["source"])
        info = path.stat()
        identity = (info.st_dev, info.st_ino) if info.st_ino else collision_key(path.as_posix())
        if identity in seen_sources:
            fail(f"Same source selected more than once: {item['source']}")
        seen_sources.add(identity)
        digest = digest_file(path, cap)
        total += digest["size"]
        if total > cap:
            fail(f"Explicit payload exceeds cap: {total} > {cap}")
        text = scan_text(path, item["member"])
        if text is not None:
            texts[item["member"]] = text
        record = {k: item[k] for k in ("root", "source", "member", "purpose", "admission", "admission_reason") if k in item}
        record.update(digest)
        if "origin" in item:
            origin = item["origin"]
            if not isinstance(origin, dict):
                fail(f"Invalid origin: {item['member']}")
            original = source_path(roots, origin.get("root"), origin.get("source"))
            record["origin"] = {"root": origin["root"], "source": origin["source"], **digest_file(original, cap)}
            if origin.get("must_match", False) and record["origin"]["sha256"] != record["sha256"]:
                fail(f"Selected copy does not match declared original: {item['member']}")
            record["origin"]["must_match"] = bool(origin.get("must_match", False))
        records.append(record)
    link_errors = markdown_link_errors(texts, members + [INTERNAL_MANIFEST])
    if link_errors:
        fail("Unclosed/nonportable Markdown links:\n" + "\n".join(link_errors))
    public = {"format": FORMAT, "max_total_bytes": cap, "payload_bytes": total,
              "operation": "Packaging only; this tool does not execute training or payload code.",
              "acceptance_tiers": selection["acceptance_tiers"], "files": records}
    if len(json_bytes(public)) > MANIFEST_CAP or total + len(json_bytes(public)) > cap:
        fail("Payload plus embedded manifest exceeds package cap")
    return {"format": FORMAT, "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_path": selection_path.as_posix(), "selection_sha256": digest_file(selection_path, MANIFEST_CAP)["sha256"],
            "roots": {key: Path(value).absolute().as_posix() for key, value in roots.items()}, "package": public}


def write_new_json(path, data):
    path = Path(path).absolute()
    ensure_no_links(path, missing_leaf=True)
    if path.exists():
        fail(f"Refusing to overwrite: {path}")
    payload = json_bytes(data)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def load_frozen(path, fresh=True):
    frozen = read_json(path)
    if not isinstance(frozen, dict) or frozen.get("format") != FORMAT:
        fail("Wrong frozen-manifest format")
    if fresh:
        current = prepare_selection(frozen["selection_path"])
        for key in ("selection_path", "selection_sha256", "roots", "package"):
            if current[key] != frozen[key]:
                fail(f"Frozen input freshness mismatch: {key}; preview and freeze again")
    return frozen


def approve(manifest, approval, reviewer):
    if not reviewer.strip():
        fail("An explicit reviewer identifier is required")
    frozen = load_frozen(manifest)
    write_new_json(approval, {"format": FORMAT, "manifest_sha256": digest_file(manifest, MANIFEST_CAP)["sha256"],
                             "reviewer": reviewer, "approved_utc": datetime.now(timezone.utc).isoformat(),
                             "payload_bytes": frozen["package"]["payload_bytes"],
                             "scope": "Exact manifest bytes, explicit payload, admissions, links and declared evidence tiers reviewed."})


def zip_info(name):
    info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _parse_embedded(data):
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"Invalid embedded manifest: {exc}")
    if not isinstance(result, dict) or result.get("format") != FORMAT:
        fail("Wrong embedded manifest format")
    cap = result.get("max_total_bytes")
    if type(cap) is not int or not 0 < cap <= HARD_CAP:
        fail("Invalid embedded byte cap")
    files = result.get("files")
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        fail("Invalid embedded explicit file list")
    for item in files:
        if not isinstance(item, dict):
            fail("Invalid embedded file record")
        check_payload_path(item.get("member"), item.get("admission"), item.get("admission_reason"))
        if type(item.get("size")) is not int or item["size"] < 0 or not re.fullmatch(r"[a-f0-9]{64}", str(item.get("sha256"))) or not re.fullmatch(r"[a-f0-9]{8}", str(item.get("crc32"))):
            fail("Invalid embedded size/SHA-256/CRC32")
    check_names([INTERNAL_MANIFEST] + [f["member"] for f in files])
    validate_tiers(result.get("acceptance_tiers"), [f["member"] for f in files])
    if sum(f["size"] for f in files) != result.get("payload_bytes") or result["payload_bytes"] + len(data) > cap:
        fail("Embedded payload size/cap mismatch")
    return result


def verify(archive, manifest=None, extract_to=None):
    archive = Path(archive).absolute()
    ensure_no_links(archive)
    if archive.stat().st_size > HARD_CAP + MANIFEST_CAP:
        fail("Archive exceeds bounded archive cap")
    expected = load_frozen(manifest, fresh=False)["package"] if manifest is not None else None
    with zipfile.ZipFile(archive, "r") as bundle:
        infos = bundle.infolist()
        if not 1 < len(infos) <= MAX_FILES + 1:
            fail("ZIP has an invalid bounded member count")
        check_names([i.filename for i in infos])
        index = {i.filename: i for i in infos}
        if INTERNAL_MANIFEST not in index or index[INTERNAL_MANIFEST].file_size > MANIFEST_CAP:
            fail("Missing or oversized embedded manifest")
        for info in infos:
            mode = info.external_attr >> 16
            if info.is_dir() or info.flag_bits & 1 or info.compress_type != zipfile.ZIP_DEFLATED or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
                fail(f"Nonregular/encrypted/unsupported ZIP member: {info.filename}")
            if info.date_time != FIXED_DATE or info.create_system != 3 or info.external_attr != 0o100644 << 16:
                fail(f"Noncanonical ZIP metadata: {info.filename}")
        embedded_bytes = bundle.read(INTERNAL_MANIFEST)
        embedded = _parse_embedded(embedded_bytes)
        if embedded_bytes != json_bytes(embedded):
            fail("Embedded manifest does not have canonical JSON bytes")
        if expected is not None and embedded != expected:
            fail("ZIP manifest differs from exact frozen manifest")
        allowlist = {f["member"]: f for f in embedded["files"]}
        if set(index) != set(allowlist) | {INTERNAL_MANIFEST}:
            fail("ZIP member allowlist differs from embedded manifest")
        if [i.filename for i in infos] != sorted(index):
            fail("ZIP member order is not deterministic")
        if sum(i.file_size for i in infos) > embedded["max_total_bytes"]:
            fail("ZIP exceeds uncompressed byte cap")
        texts = {}
        for name, record in allowlist.items():
            info = index[name]
            if info.file_size != record["size"] or f"{info.CRC:08x}" != record["crc32"]:
                fail(f"ZIP declared size/CRC differs: {name}")
            sha, crc, size, chunks = hashlib.sha256(), 0, 0, []
            with bundle.open(info, "r") as stream:
                while block := stream.read(CHUNK):
                    size += len(block)
                    if size > record["size"]:
                        fail(f"ZIP inflated beyond declared size: {name}")
                    sha.update(block)
                    crc = zlib.crc32(block, crc)
                    if name.casefold().endswith(".md"):
                        if size > MANIFEST_CAP:
                            fail(f"Markdown exceeds text cap: {name}")
                        chunks.append(block)
            if {"size": size, "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"} != {k: record[k] for k in ("size", "sha256", "crc32")}:
                fail(f"ZIP full-readback hash mismatch: {name}")
            if chunks:
                try:
                    texts[name] = b"".join(chunks).decode("utf-8-sig")
                except UnicodeError:
                    fail(f"Markdown is not UTF-8: {name}")
        errors = markdown_link_errors(texts, list(index))
        if errors:
            fail("ZIP has unclosed Markdown links:\n" + "\n".join(errors))
        result = {"format": FORMAT, "archive": archive.name, **digest_file(archive, HARD_CAP + MANIFEST_CAP),
                  "member_count": len(infos), "payload_bytes": embedded["payload_bytes"],
                  "embedded_manifest_sha256": hashlib.sha256(embedded_bytes).hexdigest(),
                  "acceptance_tiers": embedded["acceptance_tiers"], "full_readback": True}
        if extract_to is not None:
            destination = Path(extract_to).absolute()
            ensure_no_links(destination, missing_leaf=True)
            if destination.exists():
                fail("Extraction destination must not exist (no overwrite, even an empty directory)")
            destination.mkdir()
            for info in infos:
                target = destination.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                ensure_no_links(target, missing_leaf=True)
                with bundle.open(info) as source, target.open("xb") as output:
                    while block := source.read(CHUNK):
                        output.write(block)
                if info.filename != INTERNAL_MANIFEST and digest_file(target) != {k: allowlist[info.filename][k] for k in ("size", "sha256", "crc32")}:
                    fail(f"Extracted file readback mismatch: {info.filename}")
            if digest_file(destination / INTERNAL_MANIFEST)["sha256"] != result["embedded_manifest_sha256"]:
                fail("Extracted embedded manifest mismatch")
            result["extracted_to"] = destination.as_posix()
    return result


def build(manifest, approval, output, receipt):
    manifest_sha256 = digest_file(manifest, MANIFEST_CAP)["sha256"]
    approval_sha256 = digest_file(approval, MANIFEST_CAP)["sha256"]
    frozen = load_frozen(manifest)
    approval_data = read_json(approval)
    if (approval_data.get("format") != FORMAT
            or approval_data.get("manifest_sha256") != manifest_sha256
            or not isinstance(approval_data.get("reviewer"), str) or not approval_data["reviewer"].strip()):
        fail("Missing exact-manifest approval; review and approve this freeze")
    for path in (Path(output).absolute(), Path(receipt).absolute()):
        ensure_no_links(path, missing_leaf=True)
        if path.exists():
            fail(f"Refusing to overwrite output/receipt: {path}")
    if Path(output).absolute() == Path(receipt).absolute():
        fail("Archive and receipt must have different paths")
    package = frozen["package"]
    records = {f["member"]: f for f in package["files"]}
    with Path(output).open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=False) as bundle:
            for name in sorted([INTERNAL_MANIFEST] + list(records)):
                if name == INTERNAL_MANIFEST:
                    bundle.writestr(zip_info(name), json_bytes(package), compresslevel=9)
                    continue
                record = records[name]
                source = source_path(frozen["roots"], record["root"], record["source"])
                sha, crc, size = hashlib.sha256(), 0, 0
                with source.open("rb") as incoming, bundle.open(zip_info(name), "w") as outgoing:
                    while block := incoming.read(CHUNK):
                        size += len(block)
                        if size > record["size"]:
                            fail(f"Source grew during build: {name}")
                        sha.update(block)
                        crc = zlib.crc32(block, crc)
                        outgoing.write(block)
                if {"size": size, "sha256": sha.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"} != {k: record[k] for k in ("size", "sha256", "crc32")}:
                    fail(f"Source changed during build: {name}")
        raw.flush()
        os.fsync(raw.fileno())
    # Recheck original sources after writing; a stale build never receives a receipt.
    load_frozen(manifest)
    checked = verify(output, manifest=manifest)
    # Readback can take time: bind the final receipt to the same exact approved
    # control bytes and source versions, including changes during readback.
    load_frozen(manifest)
    if (digest_file(manifest, MANIFEST_CAP)["sha256"] != manifest_sha256
            or digest_file(approval, MANIFEST_CAP)["sha256"] != approval_sha256):
        fail("Manifest/approval bytes changed during build/readback; no receipt issued")
    checked["manifest_sha256"] = manifest_sha256
    checked["approval_sha256"] = approval_sha256
    checked["receipt_created_utc"] = datetime.now(timezone.utc).isoformat()
    write_new_json(receipt, checked)
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("preview", help="Read explicit files, hashes, admissions, links and factual tier declarations")
    preview.add_argument("--selection", required=True)
    freeze = sub.add_parser("freeze", help="Create a fresh local manifest; never overwrite")
    freeze.add_argument("--selection", required=True)
    freeze.add_argument("--manifest", required=True)
    approval = sub.add_parser("approve", help="Bind a review record to exact fresh manifest bytes")
    approval.add_argument("--manifest", required=True)
    approval.add_argument("--approval", required=True)
    approval.add_argument("--reviewer", required=True)
    packing = sub.add_parser("build", help="Require fresh approved inputs, write and fully reread one deterministic ZIP")
    for arg in ("manifest", "approval", "output", "receipt"):
        packing.add_argument(f"--{arg}", required=True)
    checking = sub.add_parser("verify", help="Bounded full ZIP readback, optionally safe extraction into a nonexistent directory")
    checking.add_argument("--archive", required=True)
    checking.add_argument("--manifest")
    checking.add_argument("--extract-to")
    args = parser.parse_args(argv)
    try:
        if args.command == "preview":
            value = prepare_selection(args.selection)
            public = value["package"]
            print(json_bytes({"payload_bytes": public["payload_bytes"], "max_total_bytes": public["max_total_bytes"],
                              "members": [f["member"] for f in public["files"]], "local_markdown_links": "closed",
                              "acceptance_tiers": public["acceptance_tiers"],
                              "note": "Tier results are factual declarations backed by the selected evidence; packaging does not execute them."}).decode(), end="")
        elif args.command == "freeze":
            write_new_json(args.manifest, prepare_selection(args.selection))
            print(f"Created fresh manifest: {args.manifest}")
        elif args.command == "approve":
            approve(args.manifest, args.approval, args.reviewer)
            print(f"Created exact-manifest approval: {args.approval}")
        elif args.command == "build":
            print(json_bytes(build(args.manifest, args.approval, args.output, args.receipt)).decode(), end="")
        elif args.command == "verify":
            print(json_bytes(verify(args.archive, args.manifest, args.extract_to)).decode(), end="")
        return 0
    except (PackageError, OSError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
