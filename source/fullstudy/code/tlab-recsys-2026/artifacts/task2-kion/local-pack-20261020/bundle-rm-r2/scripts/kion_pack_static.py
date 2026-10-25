"""Deterministic local static extension to the frozen KION seven-group cache.

No interactions, outcomes, splits, external encoders or network are read. Item
rows use the existing train vocabulary; user profiles remain a compact lookup.
The caller keeps the original 197 item columns as a separate, unchanged block.
Metadata availability is a retrospective snapshot, not a historical as-of claim.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_basic_bundle_features as basic
import kion_local_features as local
import kion_sex_features as sex


CONTRACT = "kion-pack-static-v1"
UNKNOWN = "__UNKNOWN__"
KIDS_COLUMNS = ["kids_no", "kids_yes", "kids_unknown"]
INCOME_COLUMNS = ["income_0_20", "income_20_40", "income_40_60", "income_60_90",
                  "income_90_150", "income_150_inf", "income_unknown"]
FOR_KIDS_COLUMNS = ["yes", "no", "unknown"]
TEXT_FIELDS = ("title", "title_orig", "description", "keywords")
ITEM_FIELDS = ("studios", "for_kids", *TEXT_FIELDS)
APPROVED_SERIES_SHA256 = "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6"
EDITION_FLAGS = ("edition.is_4k", "accessibility.sign_interpretation",
                 "accessibility.audio_description", "subtitles.present")
USER_COLUMNS = (["age:" + value for value in local.AGE_CATEGORIES]
                + ["sex:" + value for value in sex.SEX_CATEGORIES]
                + KIDS_COLUMNS + INCOME_COLUMNS)
_WORD = re.compile(r"\w+", re.UNICODE)
_SUFFIX = re.compile(r"[\(\[]([^()\[\]]+)[\)\]]?\s*$")


def _metadata(value, manifest):
    """Resolve even a supplied dict against the hash-bound basic source file."""
    source = Path(value) if isinstance(value, (str, Path)) else Path(manifest["inputs"]["metadata"]["path"])
    metadata, mapping = sex.read_metadata(source)
    if not isinstance(value, (str, Path)) and local.canonical_hash(value) != local.canonical_hash(metadata):
        raise ValueError("Supplied metadata disagrees with the basic metadata source")
    if local.file_hash(source) != manifest["inputs"]["metadata"]["sha256"]:
        raise ValueError("Basic metadata source hash mismatch")
    expected = {"age": local.AGE_CATEGORIES,
                "income": [UNKNOWN, "income_0_20", "income_150_inf", "income_20_40",
                           "income_40_60", "income_60_90", "income_90_150"],
                "sex": sex.PREPARED_SEX_CATEGORIES, "kids_flg": [UNKNOWN, "0", "1"]}
    if metadata["user_categories"] != expected:
        raise ValueError("Unexpected prepared profile categories")
    return metadata, mapping, source


def _basic_cache(path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("builder_contract") != basic.CONTRACT:
        raise ValueError("Expected the frozen seven-group basic cache")
    if local.file_hash(path) != manifest["features"]["sha256"]:
        raise ValueError("Basic cache hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        items = source["item_features"]
    if local.array_hash(items) != manifest["features"]["arrays"]["item_features"]["sha256"]:
        raise ValueError("Basic item matrix hash mismatch")
    # Validate all original item columns and the original age/sex named layout,
    # without inflating the 3.5-million-row original user matrix.
    dummy = np.zeros((1, 10), np.float32)
    dummy[0, [0, 9]] = 1
    basic.validate_bundle_layout(dummy, items, manifest["groups"])
    return manifest, items


class UserLookup:
    """Sorted unique raw profiles, four uint8 codes, and named one-hot queries.

    A missing UID maps independently to age/sex/kids/income unknown. No user ID
    becomes a model feature. Query order and duplicate UIDs are preserved.
    """

    columns = USER_COLUMNS

    def __init__(self, user_ids, codes, coverage):
        ids, categories = np.asarray(user_ids), np.asarray(codes)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids < 0).any()
                or (ids > np.iinfo(np.int64).max).any()
                or (len(ids) > 1 and not (ids[1:] > ids[:-1]).all())):
            raise ValueError("Compact user IDs must be sorted, unique nonnegative int64 values")
        if (categories.shape != (len(ids), 4) or categories.dtype.kind not in "iu"
                or (categories < 0).any() or (categories >= np.array([7, 3, 3, 7])).any()):
            raise ValueError("Compact user profile codes have invalid shape, dtype or category")
        self.user_ids = ids.astype(np.int64, copy=False)
        self.codes = categories.astype(np.uint8, copy=False)
        self.coverage = coverage
        self.user_ids.setflags(write=False)
        self.codes.setflags(write=False)

    def __call__(self, user_ids):
        values = np.asarray(user_ids)
        if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
            raise ValueError("User lookup requires a one-dimensional nonnegative integer array")
        if values.dtype.kind == "u" and (values > np.iinfo(np.int64).max).any():
            raise ValueError("User ID exceeds int64")
        values = values.astype(np.int64, copy=False)
        positions = np.searchsorted(self.user_ids, values)
        codes = np.tile(np.array([0, 2, 2, 6], np.uint8), (len(values), 1))
        in_range = positions < len(self.user_ids)
        hit = np.zeros(len(values), bool)
        hit[in_range] = self.user_ids[positions[in_range]] == values[in_range]
        codes[hit] = self.codes[positions[hit]]
        result = np.zeros((len(values), 20), np.float32)
        rows = np.arange(len(values))
        for block, start in enumerate((0, 7, 10, 13)):
            result[rows, start + codes[:, block]] = 1
        return result


def _user_lookup(path):
    lookups = [{value: index for index, value in enumerate(local.AGE_CATEGORIES)},
               {"М": 0, "Ж": 1}, {"0": 0, "1": 1},
               {value: index for index, value in enumerate(INCOME_COLUMNS[:-1])}]
    defaults = [0, 2, 2, 6]
    fields = ("age", "sex", "kids_flg", "income")
    ids, codes, seen = [], [], set()
    coverage = {field: Counter() for field in fields}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"user_id", *fields}.issubset(reader.fieldnames)):
            raise ValueError("Missing or duplicate user profile headers")
        for row in reader:
            if None in row or any(row.get(field) is None for field in ("user_id", *fields)):
                raise ValueError("Malformed user profile CSV row")
            uid = local.strict_id(row["user_id"], "user_id")
            if uid > np.iinfo(np.int64).max or uid in seen:
                raise ValueError("Duplicate or out-of-range user_id")
            seen.add(uid)
            ids.append(uid)
            encoded = []
            for field, lookup, default in zip(fields, lookups, defaults):
                raw = row[field].strip()
                encoded.append(lookup.get(raw, default))
                reason = "known" if raw in lookup else ("empty" if not raw else "invalid_nonblank")
                coverage[field][reason] += 1
            codes.append(encoded)
    order = np.argsort(np.asarray(ids, dtype=np.int64))
    encoded = np.asarray(codes, dtype=np.uint8).reshape(-1, 4)
    return UserLookup(np.asarray(ids, dtype=np.int64)[order], encoded[order],
                      {key: dict(value) for key, value in coverage.items()})


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _normal_offsets(text):
    """Normalize marker characters while retaining their literal source spans."""
    characters, starts, ends = [], [], []
    whitespace_start, whitespace_end = None, None
    for index, original in enumerate(text):
        for character in unicodedata.normalize("NFKC", original).casefold():
            if character.isspace():
                if whitespace_start is None:
                    whitespace_start = index
                whitespace_end = index + 1
                continue
            if whitespace_start is not None and characters:
                characters.append(" ")
                starts.append(whitespace_start)
                ends.append(whitespace_end)
            whitespace_start = whitespace_end = None
            characters.append(character)
            starts.append(index)
            ends.append(index + 1)
    return "".join(characters), starts, ends


def _read_items(path):
    """Keep original text for source evidence; strip categorical fields only."""
    result = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not {"item_id", *ITEM_FIELDS}.issubset(reader.fieldnames)):
            raise ValueError("Missing or duplicate item CSV headers")
        for row in reader:
            if None in row or any(row.get(field) is None for field in ("item_id", *ITEM_FIELDS)):
                raise ValueError("Malformed item CSV row")
            iid = local.strict_id(row["item_id"], "item_id")
            if iid in result:
                raise ValueError("Duplicate item_id")
            result[iid] = tuple(row[field] if field in TEXT_FIELDS else row[field].strip() for field in ITEM_FIELDS)
    return result


def _subtitle_marker(marker):
    if marker in ("субтитры", "сурдоперевод + субтитры"):
        return True
    qualifier = r"[а-яёa-z]+(?:-[а-яёa-z]+)*"
    subtitle_part = rf"с (?:{qualifier} ){{0,2}}субтитрами"
    return bool(re.fullmatch(rf"(?:на (?:{qualifier} ){{1,3}}языке )?{subtitle_part}", marker)
                or re.fullmatch(rf"на языке жестов {subtitle_part}", marker)
                or re.fullmatch(r"(?:with )?(?:[a-z]+ )?subtitles?", marker))


def _current_title(text):
    text = re.sub(r"\[\s*4[кk]\s*\]", "", _normal(text)).strip()
    while match := _SUFFIX.search(text):
        marker = match.group(1).strip()
        if not (_subtitle_marker(marker) or marker in ("жестовым языком", "sign language",
                                                       "с тифлокомментарием", "with audio description")):
            break
        text = text[:match.start()].strip()
    return text


def _edition(row, raw_id):
    """Accept documented edition markers and narrowly scoped current-card CTAs.

    Ordinary plot/keyword mentions and broad format aliases remain unknown.
    This intentionally bounded parser never equates no match with false.
    """
    evidence, positives = [], set()

    def record(flag, field, text, starts, ends, start, end, rule):
        raw_start, raw_end = starts[start], ends[end - 1]
        evidence.append({"item_id": raw_id, "flag": EDITION_FLAGS[flag], "field": field,
                         "quote": text[raw_start:raw_end], "start": raw_start, "end": raw_end, "rule_id": rule})

    for field in ("title", "title_orig"):
        text = row[ITEM_FIELDS.index(field)]
        normalized, starts, ends = _normal_offsets(text)
        # A bracketed 4K edition label can precede the title or follow it.
        # Parenthesized 4K/UHD and unbracketed production mentions stay candidates.
        for match in re.finditer(r"\[\s*4[кk]\s*\]", normalized):
            positives.add(0)
            record(0, field, text, starts, ends, match.start(), match.end(), "documented-bracket-4k-marker-v2")
        rest = normalized
        while match := _SUFFIX.search(rest):
            marker = _normal(match.group(1))
            flags = set()
            if marker in ("жестовым языком", "sign language", "сурдоперевод + субтитры"):
                flags.add(1)
            if marker in ("с тифлокомментарием", "with audio description"):
                flags.add(2)
            if _subtitle_marker(marker):
                flags.add(3)
            for flag in flags:
                record(flag, field, text, starts, ends, match.start(), match.end(), "documented-title-edition-suffix-v2")
            positives.update(flags)
            rest = rest[:match.start()].rstrip()
    description = row[ITEM_FIELDS.index("description")]
    normalized, starts, ends = _normal_offsets(description)
    for match in re.finditer(r"(?<!\w)смотрите\s+в\s+4[кk]\s*!", normalized):
        positives.add(0)
        record(0, "description", description, starts, ends, match.start(), match.end(), "documented-current-card-4k-cta-v2")
    direct_patterns = {
        1: (r"(?:в (?:ленту|фильм) (?:добавлена|внедрена) дорожка с жестовым языком|"
            r"(?:этот )?фильм содержит сурдоперевод|(?:эта )?(?:версия|лента) представлена с сурдопереводом|"
            r"(?:этот )?фильм представлен с сурдопереводом)\b"),
        2: (r"(?:в (?:ленту|фильм) (?:добавлен|внедр[её]н) тифлокомментарий|"
            r"(?:этот )?фильм представлен с тифлокомментариями|(?:эта )?версия с тифлокомментариями)\b"),
    }
    current_titles = {_current_title(row[ITEM_FIELDS.index(field)]) for field in ("title", "title_orig")}
    current_titles.discard("")
    watch = (r"(?<!\w)(?:(?:рекомендуем|предлагаем) (?:вам )?(?:смотреть|посмотреть) онлайн|смотрите(?: онлайн)?)"
             r"[^.!?]{0,250}?\bс (?:[а-яёa-z]+(?:-[а-яёa-z]+)* ){0,2}субтитрами\b")
    for sentence in re.finditer(r"[^.!?]+(?:[.!?]|$)", normalized):
        content = sentence.group(0).lstrip()
        for flag, pattern in direct_patterns.items():
            # The documented subject starts the sentence; plot/filmography
            # mentions such as 'его предыдущий фильм содержит...' do not match.
            if re.match(pattern, content):
                positives.add(flag)
                record(flag, "description", description, starts, ends, sentence.start(), sentence.end(),
                       "documented-current-card-accessibility-statement-v2")
        named_current = any(re.search(r"[«\"']" + re.escape(title) + r"[»\"']", content) for title in current_titles)
        explicit_current = bool(re.match(r"(?:этот|данный) (?:фильм|сериал)\b|смотрите (?:этот|данный) (?:фильм|сериал)\b", content))
        cta = re.search(watch, content)
        negated_cta = cta is not None and bool(re.search(r"\b(?:не|никогда)\s*$", content[:cta.start()]))
        quoted_targets = (re.findall(r"«([^»]+)»|\"([^\"]+)\"", cta.group(0)) if cta else [])
        other_target = any(next(part for part in quoted if part).strip() not in current_titles for quoted in quoted_targets)
        if (named_current or explicit_current) and cta and not negated_cta and not other_target:
            positives.add(3)
            record(3, "description", description, starts, ends, sentence.start(), sentence.end(),
                   "documented-current-card-subtitle-watch-cta-v2")
    conflicts = set()
    # Explicitly quarantined in the versioned local contract. Never OR its two
    # incompatible accessibility claims into two positive model inputs.
    if raw_id == 10807:
        conflicts.update((1, 2))
        evidence.append({"item_id": raw_id, "flags": [EDITION_FLAGS[1], EDITION_FLAGS[2]],
                         "status": "conflict", "rule_id": "documented-viy-quarantine-v1",
                         "field": "description", "quote": description, "start": 0, "end": len(description)})
    # Negation only blocks an already accepted suffix; it never becomes an
    # inferred negative for an unrelated plot/keyword mention.
    negative_patterns = (r"4\s*[кk]\s+недоступн", r"не содержит сурдоперевод",
                         r"без тифлокомментар", r"без субтитр")
    for flag in positives:
        match = re.search(negative_patterns[flag], _normal(description))
        if match:
            conflicts.add(flag)
            evidence.append({"item_id": raw_id, "flag": EDITION_FLAGS[flag], "status": "conflict",
                             "field": "description", "quote": description, "start": 0,
                             "end": len(description), "rule_id": "suffix-description-negation-quarantine-v1"})
    values = []
    for flag in range(4):
        positive = flag in positives and flag not in conflicts
        values.extend((int(positive), int(not positive), int(flag in conflicts)))
    return values, evidence


def _text_features(items, mapping, size):
    columns = [field + ":" + metric for field in TEXT_FIELDS
               for metric in ("present", "missing", "log_chars_z", "log_words_z")]
    columns += [name + ":" + metric for name in EDITION_FLAGS for metric in ("positive", "missing", "conflict")]
    values = np.zeros((size, len(columns)), np.float32)
    fit, evidence = {}, []
    real_tokens = np.array(list(mapping.values()), dtype=np.int64)
    for index, field in enumerate(TEXT_FIELDS):
        field_index = ITEM_FIELDS.index(field)
        texts = [items.get(int(iid), ("",) * len(ITEM_FIELDS))[field_index].strip() for iid in mapping]
        present = np.array([bool(text) for text in texts])
        offset = index * 4
        values[real_tokens, offset] = present
        values[real_tokens, offset + 1] = ~present
        stats = {}
        for target, counts, metric in ((offset + 2, [len(t) for t in texts], "log_chars"),
                                       (offset + 3, [len(_WORD.findall(t)) for t in texts], "log_words")):
            raw = np.log1p(np.asarray(counts, np.float64))
            known = raw[present]
            mean, std = (float(known.mean()), float(known.std())) if len(known) else (0.0, 0.0)
            scale = std if std > 0 else 1.0
            values[real_tokens[present], target] = ((raw[present] - mean) / scale).astype(np.float32)
            stats[metric] = {"mean": mean, "scale": scale, "fit_items": int(present.sum())}
        fit[field] = stats
    for iid, token in mapping.items():
        row = items.get(int(iid), ("",) * len(ITEM_FIELDS))
        encoded, facts = _edition(row, int(iid))
        values[token, 16:] = encoded
        evidence.extend(facts)
    return values, columns, fit, evidence


def _series(path, expected_hash, mapping, size, source_hashes):
    if local.file_hash(path) != expected_hash:
        raise ValueError("Approved series annotation hash mismatch")
    annotation = json.loads(path.read_text(encoding="utf-8"))
    if annotation.get("format_version") != 1 or annotation.get("annotation_version") != "kion-series-v1":
        raise ValueError("Unsupported series annotation version")
    for suffix, source_hash in source_hashes.items():
        inputs = [entry for entry in annotation["provenance"]["inputs"] if entry["path"].endswith(suffix)]
        if len(inputs) != 1 or inputs[0]["sha256"] != source_hash:
            raise ValueError("Series source snapshot hash mismatch: " + suffix)
    groups = annotation.get("series", [])
    ids = [entry.get("id") for entry in groups]
    if (len(ids) != 80 or any(not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value)
                              for value in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Series requires exactly 80 unique canonical ASCII IDs")
    columns = ["series:" + value for value in sorted(ids)] + ["series_unknown"]
    lookup = {name: index for index, name in enumerate(columns)}
    values = np.zeros((size, 81), np.float32)
    seen, raw_only = set(), []
    for group in groups:
        if group.get("status") != "confirmed" or not group.get("curator_rationale"):
            raise ValueError("Unconfirmed or unexplained series group")
        members = group.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError("Series group requires confirmed members")
        for member in members:
            iid = member.get("item_id")
            if isinstance(iid, bool) or not isinstance(iid, int) or iid < 0 or iid in seen:
                raise ValueError("Duplicate or invalid series member ID")
            seen.add(iid)
            if member.get("status") != "confirmed" or not member.get("evidence"):
                raise ValueError("Unconfirmed series member")
            if member.get("train_known") is not (str(iid) in mapping):
                raise ValueError("Series train_known flag disagrees with train mapping")
            if str(iid) not in mapping:
                raw_only.append(iid)
                continue
            values[mapping[str(iid)], lookup["series:" + group["id"]]] = 1
    tokens = np.array(list(mapping.values()), np.int64)
    values[tokens, -1] = values[tokens, :-1].sum(axis=1) == 0
    coverage = {"groups": 80, "confirmed_raw_members": len(seen), "projected_members": len(seen) - len(raw_only),
                "raw_only_member_ids": sorted(raw_only), "unknown_items": int(values[:, -1].sum())}
    for key, actual in (("confirmed_groups", 80), ("confirmed_member_items", len(seen)),
                        ("train_known_confirmed_member_items", len(seen) - len(raw_only))):
        if annotation.get("coverage", {}).get(key) != actual:
            raise ValueError("Series coverage disagrees with annotation: " + key)
    return values, columns, coverage


def build(raw_dir, metadata, basic_npz, *, series_path=None,
          series_sha256=APPROVED_SERIES_SHA256, basic_manifest=None):
    """Return new item blocks and a compact user lookup; write no input/output.

    ``metadata`` may be a path or the loaded JSON dict. The basic manifest must
    bind its raw source files and metadata. A custom series snapshot requires an
    explicit expected hash; the default pins the accepted 80-series annotation.
    The user lookup first ten columns use the exact original age/sex encodings.
    """
    raw_dir, basic_npz = Path(raw_dir).resolve(), Path(basic_npz).resolve()
    manifest_path = Path(basic_manifest).resolve() if basic_manifest else basic_npz.with_name("manifest.json")
    base, original_items = _basic_cache(basic_npz, manifest_path)
    metadata, mapping, metadata_path = _metadata(metadata, base)
    if original_items.shape[0] != metadata["vocab_size"] or base["vocabulary"]["sha256"] != local.canonical_hash(mapping):
        raise ValueError("Basic cache vocabulary mismatch")
    series_path = Path(series_path).resolve() if series_path else Path(__file__).resolve().parents[1] / "configs/datasets/kion-series-v1.json"
    paths = {"users": raw_dir / "users.csv", "items": raw_dir / "items.csv", "metadata": metadata_path,
             "basic": basic_npz, "basic_manifest": manifest_path, "series": series_path,
             "builder": Path(__file__).resolve(), "basic_helper": Path(basic.__file__).resolve(),
             "local_helper": Path(local.__file__).resolve(), "sex_helper": Path(sex.__file__).resolve()}
    hashes = {name: local.file_hash(path) for name, path in paths.items()}
    for name in ("users", "items"):
        if hashes[name] != base["inputs"][name]["sha256"]:
            raise ValueError("Raw source differs from basic cache: " + name)
    items = _read_items(paths["items"])
    users = _user_lookup(paths["users"])
    size = metadata["vocab_size"]
    groups, blocks, item_columns = {}, [], []
    for name, start, stop, columns in (("age", 0, 7, local.AGE_CATEGORIES), ("sex", 7, 10, sex.SEX_CATEGORIES),
                                        ("kids", 10, 13, KIDS_COLUMNS), ("income", 13, 20, INCOME_COLUMNS)):
        groups[name] = {"scope": "user", "start": start, "stop": stop, "columns": list(columns)}

    def add(name, values, columns):
        start = len(item_columns)
        groups[name] = {"scope": "item", "start": start, "stop": start + len(columns), "columns": list(columns)}
        item_columns.extend(name + ":" + column for column in columns)
        blocks.append(values)

    studio_columns = basic.fit_tag_categories(items, mapping, 0)
    studios, studio_coverage = basic.tag_features(items, mapping, size, 0, studio_columns)
    add("studios", studios, studio_columns)
    kids = np.zeros((size, 3), np.float32)
    kids_coverage = Counter()
    for iid, token in mapping.items():
        raw = items.get(int(iid), ("",) * len(ITEM_FIELDS))[1]
        if raw not in ("", "1.0", "0.0"):
            raise ValueError("Invalid nonblank for_kids annotation for train-known item " + iid)
        index = {"1.0": 0, "0.0": 1, "": 2}[raw]
        kids[token, index] = 1
        kids_coverage[FOR_KIDS_COLUMNS[index]] += 1
    add("for_kids", kids, FOR_KIDS_COLUMNS)
    series, series_columns, series_coverage = _series(series_path, series_sha256, mapping, size,
                                                     {"items.csv": hashes["items"], "metadata.json": hashes["metadata"]})
    add("approved_series", series, series_columns)
    text, text_columns, text_fit, evidence = _text_features(items, mapping, size)
    add("text_local", text, text_columns)
    values = np.concatenate(blocks, axis=1)
    if values.dtype != np.float32 or not np.isfinite(values).all() or np.any(values[:3]):
        raise ValueError("Invalid static item feature matrix")
    if any(local.file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Static input/source changed while building")
    provenance = {"contract": CONTRACT, "metadata_availability_mode": "static_retrospective_catalog_snapshot",
                  "inputs": {name: {"path": path.as_posix(), "sha256": hashes[name]} for name, path in paths.items()},
                  "mapping_sha256": local.canonical_hash(mapping),
                  "original_basic_item_sha256": local.array_hash(original_items),
                  "original_basic_item_dim": original_items.shape[1], "original_basic_user_prefix_dim": 10,
                  "schema_sha256": local.canonical_hash({"groups": groups, "item_columns": item_columns, "user_columns": USER_COLUMNS}),
                  "fit": {"scope": "unique train-known item catalog only", "cutoff_exclusive": local.TRAIN_CUTOFF,
                          "item_count": len(mapping), "item_ids_sha256": local.canonical_hash(sorted(map(int, mapping))),
                          "text": text_fit, "studios_categories_sha256": local.canonical_hash(studio_columns)},
                  "coverage": {"raw_users": len(users.user_ids), "raw_items": len(items), "train_known_items": len(mapping),
                               "missing_item_rows": sum(int(iid) not in items for iid in mapping),
                               "studios": studio_coverage, "for_kids": dict(kids_coverage), "series": series_coverage,
                               "users": users.coverage,
                               "edition_positive": {name: int(text[:, 16 + index * 3].sum()) for index, name in enumerate(EDITION_FLAGS)},
                               "edition_evidence_records": len(evidence)},
                  "edition_parser_version": "conservative-documented-edition-rules-v2",
                  "edition_evidence": evidence,
                  "unknown_semantics": "Unobserved semantic flags remain missing; no curated negative is inferred.",
                  "not_implemented": ["broad format aliases require current-work evidence review; lexical candidates are not accepted labels"],
                  "safety": {"interactions_read": False, "splits_read": [], "outcomes_read": False,
                             "external_models_or_network": False, "special_item_rows": "PAD/BOS/UNK all zero",
                             "user_identity": "account metadata; viewer identity unverified"}}
    return {"item_features": values, "item_columns": item_columns, "user_lookup": users,
            "user_ids": users.user_ids, "user_codes": users.codes, "user_columns": list(USER_COLUMNS),
            "groups": groups, "provenance": provenance}
