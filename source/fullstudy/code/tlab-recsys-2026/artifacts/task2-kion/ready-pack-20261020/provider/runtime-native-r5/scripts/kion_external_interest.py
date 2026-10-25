"""Historical MovieLens support and Wikimedia direct-title interest.

Network access is explicit. This module consumes the shared identity ledger; it
never resolves a title, calls Wikidata, guesses a person or fits on evaluation.
MovieLens ingestion streams the ZIP, retaining selected works in disk SQLite.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
from contextlib import closing
import csv
from datetime import date, datetime, timedelta, timezone
import email.utils
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import statistics
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


CONTRACT = "kion-external-interest-v1"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MOVIELENS_BYTES = 261978986
MOVIELENS_MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"
MOVIELENS_SHA256 = "8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa"
# GroupLens says released December 2019, without a precise public release day.
MOVIELENS_KNOWN_AT = "2020-01-01T00:00:00Z"
UTC = timezone.utc
ROLES = {"work", "actor", "director"}
ML_COLUMNS = ["movielens_mean", "movielens_count", "movielens_shrunk_mean", "movielens_missing",
              "movielens_low_support", "movielens_numeric_history_mask", "movielens_identity_known_before_boundary",
              "movielens_value_available_mask", "movielens_temporal_mask", "movielens_work_proxy",
              "external_work_link_accepted"]
WIKI_WINDOW_FIELDS = ["views_7d", "views_30d", "growth_7d", "growth_30d",
                      "views_7d_missing", "views_30d_missing", "growth_7d_missing", "growth_30d_missing",
                      "article_coverage", "window_coverage_7d", "window_coverage_30d", "previous_window_coverage_30d",
                      "historical_title_unresolved", "temporal_mask", "literal_title_series"]
PEOPLE_METRICS = [metric + "_" + aggregate for metric in ("views_7d", "views_30d", "growth_7d", "growth_30d")
                  for aggregate in ("mean", "max", "median", "top3")]
PEOPLE_SUPPORT = ["n_people_total", "n_people_mapped", "n_people_with_valid_window_7d",
                  "n_people_with_valid_window_30d", "coverage_7d", "coverage_30d",
                  "roster_observed", "roster_size_missing",
                  "views_7d_missing", "views_30d_missing", "growth_7d_missing", "growth_30d_missing",
                  "historical_title_unresolved", "temporal_mask"]
COLUMNS = (ML_COLUMNS + ["wiki_work_" + language + "_" + field for language in ("ru", "en") for field in WIKI_WINDOW_FIELDS]
           + ["wiki_" + role + "_" + language + "_" + field for role in ("actor", "director")
              for language in ("ru", "en") for field in PEOPLE_METRICS + PEOPLE_SUPPORT])


def column_schema():
    """Literal numerical units; callers must never infer transforms from names."""
    output = []
    for name in COLUMNS:
        if name in {"movielens_mean", "movielens_shrunk_mean"}:
            unit = "rating"
        elif "growth_" in name and not name.endswith("_missing"):
            unit = "signed_growth"
        elif "coverage" in name and not name.endswith("article_coverage"):
            unit = "probability"
        elif (name == "movielens_count" or "n_people_" in name
              or ("views_" in name and not name.endswith("_missing"))):
            unit = "raw_count"
        else:
            unit = "mask"
        output.append({"name": name, "dtype": "float32", "unit": unit,
                       "neural_transform": "log1p" if unit == "raw_count" else "identity",
                       "gbm_transform": "identity", "missing_numeric_fill": 0})
    return output


def blank_population_matrix(metadata):
    """Full vocabulary rows, zero special tokens, explicit numeric missing masks."""
    import numpy as np
    mapping = metadata["item_id_to_token"]
    vocab = int(metadata["vocab_size"])
    tokens = [int(value) for value in mapping.values()]
    if (len(set(tokens)) != len(tokens) or set(tokens) != set(range(3, vocab))):
        raise ValueError("Expected exact item vocabulary with three zero special rows")
    values = np.zeros((vocab, len(COLUMNS)), dtype=np.float32)
    for i, name in enumerate(COLUMNS):
        if name.endswith("_missing") or name.endswith("historical_title_unresolved"):
            values[3:, i] = 1
    return values


def population_movielens_matrix(metadata, ledger, receipt_path, *, expected_receipt_sha256, origins,
                                 expected_archive_sha256=MOVIELENS_SHA256):
    """All reviewed links/full U, one exact static bank aliased across proven dates."""
    origins = [instant(date.fromordinal(int(day))) for day in origins]
    if not origins or origins != sorted(set(origins)):
        raise ValueError("Origins must be sorted and unique")
    stats, links, source, prior = load_corpus_aggregates(receipt_path,
        expected_receipt_sha256=expected_receipt_sha256, cutoff=origins[0],
        expected_archive_sha256=expected_archive_sha256)
    if (source["max_rating_timestamp"] >= origins[0].timestamp()
            or instant(source["dataset_known_at"]) >= origins[0]
            or any(instant(prior["source_known_at"]) >= day for day in origins)):
        raise ValueError("One static source bank cannot cover these origins")
    normalized = shared_ledger_inputs(ledger, work_proxy_policy="accepted_work_identity_with_unverified_edition")
    accepted, seen = [], {}
    for row in normalized["works"]:
        item = str(row["item_id"])
        if item not in metadata["item_id_to_token"] or row["match_status"] != "accepted":
            continue
        if item in seen:
            if seen[item] != row["work_id"]:
                raise ValueError("Conflicting accepted work identities for item")
            continue
        seen[item] = row["work_id"]
        if row.get("known_at") is not None:
            first_identity_mask = instant(row["known_at"]) < origins[0]
            if any((instant(row["known_at"]) < day) != first_identity_mask for day in origins):
                raise ValueError("Static bank alias would change identity-known mask across origins")
        accepted.append(row)
    features = rating_features(accepted, stats, links, source, boundary=origins[0],
                               prior=prior, temporal_mode="retrospective_identity")
    values = blank_population_matrix(metadata)
    index = {name: i for i, name in enumerate(COLUMNS)}
    values[3:, index["movielens_numeric_history_mask"]] = 1
    for link, feature in zip(accepted, features):
        token = metadata["item_id_to_token"][str(feature["item_id"])]
        for name in ML_COLUMNS:
            if name == "external_work_link_accepted":
                value = 1
            elif name == "movielens_work_proxy":
                value = int(link["is_work_proxy"])
            else:
                value = feature.get(name)
            values[token, index[name]] = value if value is not None else 0
    supported = [row for row in features if not row["movielens_missing"]]
    return {"values": values, "features": features, "normalized_links": normalized,
            "columns": list(COLUMNS), "numeric_units": {row["name"]: row["unit"] for row in column_schema()},
            "coverage": {"model_vocabulary_rows": len(values), "real_items": len(metadata["item_id_to_token"]),
                "accepted_work_items": len(accepted), "ratings_observed_items": len(supported),
                "distinct_observed_works": len({row["work_id"] for row in supported}),
                "rating_missing_items": len(metadata["item_id_to_token"]) - len(supported)},
            "clock_proof": {"alias_all_origins": True, "first_origin": iso(origins[0]),
                "last_origin": iso(origins[-1]), "source_max_rating_timestamp": source["max_rating_timestamp"],
                "dataset_known_at": source["dataset_known_at"], "all_numeric_events_before_first_origin": True,
                "retrospective_identity": True, "strict_asof_claim": False},
            "source_receipt_sha256": expected_receipt_sha256, "prior": prior}


def instant(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time(), UTC)
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso(value):
    return instant(value).isoformat().replace("+00:00", "Z")


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Acquisitions and versions are immutable; do not silently replace evidence.
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(path)
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def feature_boundary(role, target_time):
    target = instant(target_time)
    if role == "sft_train":
        return target
    if role in {"sft_validation", "initial_test"}:
        return instant("2021-08-09")
    if role in {"rm_fit", "rm_select", "rm_calibration", "rm_score"}:
        return min(target, instant("2021-08-02"))
    raise ValueError("Unknown feature role: " + str(role))


def last_available_day(boundary, lag_hours=24):
    """Require day-end + chosen lag < boundary (strict even at midnight)."""
    if not math.isfinite(lag_hours) or lag_hours < 0:
        raise ValueError("Availability lag must be finite and nonnegative")
    available_before = instant(boundary) - timedelta(hours=lag_hours, microseconds=1)
    return available_before.date() - timedelta(days=1)


def mapping_allowed(mapping, boundary, temporal_mode="strict"):
    """Identity and historical availability are distinct gates."""
    if mapping.get("match_status") not in {"matched", "accepted"}:
        return False, mapping.get("match_status", "missing")
    if not mapping.get("evidence") and not mapping.get("match_evidence"):
        return False, "missing_identity_evidence"
    if temporal_mode not in {"strict", "retrospective_identity"}:
        raise ValueError("Unknown temporal mode")
    if temporal_mode == "strict":
        known = mapping.get("known_at")
        if known is None or instant(known) >= instant(boundary):
            return False, "identity_not_known_before_boundary"
    return True, "accepted"


def _zip_members(archive):
    seen = set()
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        parts = PurePosixPath(name)
        if (parts.is_absolute() or ".." in parts.parts or ":" in name
                or ((member.external_attr >> 16) & 0o170000) == 0o120000):
            raise ValueError("Unsafe ZIP member: " + name)
        if name in seen:
            raise ValueError("Duplicate ZIP member")
        seen.add(name)
    required = {"links.csv", "movies.csv", "ratings.csv", "README.txt"}
    found = {}
    for name in seen:
        base = PurePosixPath(name).name
        if base in required:
            if base in found:
                raise ValueError("Ambiguous ZIP layout")
            found[base] = name
    if set(found) != required:
        raise ValueError("Missing MovieLens source files")
    return found


def _csv(archive, name, columns):
    stream = io.TextIOWrapper(archive.open(name), encoding="utf-8", errors="strict", newline="")
    reader = csv.DictReader(stream)
    if reader.fieldnames != columns:
        stream.close()
        raise ValueError("Unexpected CSV schema for " + name)
    return stream, reader


def _positive(value):
    parsed = int(value)
    if str(parsed) != str(value) or parsed <= 0:
        raise ValueError("Expected positive integer")
    return parsed


def imdb_id(value):
    value = str(value)
    if value.startswith("imdb:"):
        value = value[5:]
    if value.startswith("tt"):
        value = value[2:]
    if not re.fullmatch(r"[0-9]{1,10}", value) or int(value) <= 0:
        raise ValueError("Invalid IMDb ID")
    return "tt" + value.zfill(7)


def ingest_movielens(archive_path, database_path, accepted_imdb_ids, *,
                     expected_md5=MOVIELENS_MD5, expected_bytes=MOVIELENS_BYTES,
                     progress_callback=None):
    """Scan every row, store only linked movies; no full extraction or pandas load.

    SQLite is a disposable derivative, atomically published after successful QA.
    Duplicate users are resolved at query cutoff, not at ingestion. A future
    replacement rating therefore cannot erase an eligible earlier rating.
    """
    source, destination = Path(archive_path), Path(database_path)
    if destination.exists():
        raise FileExistsError(destination)
    if source.stat().st_size != expected_bytes or digest(source, "md5") != expected_md5:
        raise ValueError("MovieLens archive size/checksum mismatch")
    accepted = {imdb_id(x) for x in accepted_imdb_ids}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    connection = sqlite3.connect(partial)
    connection.execute("PRAGMA threads=2")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("CREATE TABLE ratings(movie_id INTEGER,user_id INTEGER,ts INTEGER,rating REAL)")
    connection.execute("CREATE TABLE links(movie_id INTEGER PRIMARY KEY,imdb_id TEXT,tmdb_id TEXT)")
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
    try:
        with zipfile.ZipFile(source) as archive:
            members = _zip_members(archive)
            # Reading required files verifies their CRC; tag/genome are never read.
            readme = archive.read(members["README.txt"]).decode("utf-8")
            movies = set()
            stream, reader = _csv(archive, members["movies.csv"], ["movieId", "title", "genres"])
            with stream:
                for row in reader:
                    movie = _positive(row["movieId"])
                    if movie in movies or None in row or not row["title"]:
                        raise ValueError("Duplicate/invalid MovieLens movie")
                    movies.add(movie)
            selected, links, reverse = set(), set(), defaultdict(list)
            stream, reader = _csv(archive, members["links.csv"], ["movieId", "imdbId", "tmdbId"])
            with stream:
                for row in reader:
                    movie = _positive(row["movieId"])
                    if movie in links or movie not in movies or None in row:
                        raise ValueError("MovieLens links key/referential integrity failure")
                    links.add(movie)
                    external = imdb_id(row["imdbId"]) if row["imdbId"] else None
                    if row["tmdbId"]:
                        _positive(row["tmdbId"])
                    connection.execute("INSERT INTO links VALUES(?,?,?)", (movie, external, row["tmdbId"] or None))
                    if external:
                        reverse[external].append(movie)
                    if external in accepted:
                        selected.add(movie)
            if movies != links:
                raise ValueError("MovieLens movies/links universes differ")
            count, retained, minimum, maximum, batch = 0, 0, None, None, []
            generated_before = instant("2019-11-22").timestamp()
            stream, reader = _csv(archive, members["ratings.csv"], ["userId", "movieId", "rating", "timestamp"])
            with stream:
                for row in reader:
                    user, movie = _positive(row["userId"]), _positive(row["movieId"])
                    rating, stamp = float(row["rating"]), int(row["timestamp"])
                    if (None in row or not math.isfinite(rating) or rating < .5 or rating > 5
                            or rating * 2 != int(rating * 2) or stamp < 0
                            or stamp >= generated_before
                            or str(stamp) != row["timestamp"] or movie not in links):
                        raise ValueError("Invalid MovieLens rating row")
                    count += 1
                    if progress_callback is not None and count % 250000 == 0:
                        progress_callback(count)
                    minimum = stamp if minimum is None else min(minimum, stamp)
                    maximum = stamp if maximum is None else max(maximum, stamp)
                    if movie in selected:
                        batch.append((movie, user, stamp, rating))
                        retained += 1
                    if len(batch) >= 10000:
                        connection.executemany("INSERT INTO ratings VALUES(?,?,?,?)", batch)
                        batch.clear()
            if not count:
                raise ValueError("Empty ratings source")
            if batch:
                connection.executemany("INSERT INTO ratings VALUES(?,?,?,?)", batch)
            if expected_md5 == MOVIELENS_MD5 and (count != 25000095 or len(movies) != 62423):
                raise ValueError("Official MovieLens row count mismatch")
            connection.execute("CREATE INDEX ratings_query ON ratings(movie_id,user_id,ts)")
            receipt = {"contract": CONTRACT, "source_url": MOVIELENS_URL,
                       "archive_sha256": digest(source), "archive_md5": expected_md5,
                       "archive_bytes": source.stat().st_size, "rows_scanned": count,
                       "rows_retained": retained, "movies": len(movies),
                       "selected_movie_ids": sorted(selected), "accepted_imdb_ids": sorted(accepted),
                       "ambiguous_imdb_ids": sorted(k for k, v in reverse.items() if len(v) > 1),
                       "min_rating_timestamp": minimum, "max_rating_timestamp": maximum,
                       "dataset_known_at": MOVIELENS_KNOWN_AT,
                       "readme_sha256": hashlib.sha256(readme.encode()).hexdigest(),
                       "license": "GroupLens research terms; attribution; separate redistribution/commercial permission",
                       "retrieved_at": iso(datetime.now(UTC))}
            connection.execute("INSERT INTO metadata VALUES('receipt',?)", (json.dumps(receipt, sort_keys=True),))
            connection.commit()
        connection.close()
        partial.replace(destination)
        return receipt
    except BaseException:
        connection.close()
        # Keep a failed artifact for diagnosis; it is never a usable published DB.
        raise


def aggregate_ratings(database_path, cutoff):
    """Unique raters, last eligible event; tied conflicting latest values fail QA."""
    limit = instant(cutoff).timestamp()
    with closing(sqlite3.connect("file:" + Path(database_path).as_posix() + "?mode=ro", uri=True)) as db:
        receipt = json.loads(db.execute("SELECT value FROM metadata WHERE key='receipt'").fetchone()[0])
        latest = """WITH latest AS (SELECT movie_id,user_id,MAX(ts) ts FROM ratings
                    WHERE ts<? GROUP BY movie_id,user_id), chosen AS
                    (SELECT r.movie_id,r.user_id,r.rating FROM ratings r JOIN latest l
                    ON r.movie_id=l.movie_id AND r.user_id=l.user_id AND r.ts=l.ts) """
        conflict = db.execute(latest + "SELECT movie_id,user_id FROM chosen GROUP BY movie_id,user_id HAVING MIN(rating)<>MAX(rating) LIMIT 1", (limit,)).fetchone()
        if conflict:
            raise ValueError("Conflicting ratings at identical latest timestamp")
        rows = db.execute(latest + "SELECT movie_id,COUNT(*),AVG(rating) FROM (SELECT DISTINCT movie_id,user_id,rating FROM chosen) GROUP BY movie_id", (limit,))
        stats = {int(movie): {"count": int(n), "mean": float(mean), "cutoff": iso(cutoff)} for movie, n, mean in rows}
        links = defaultdict(list)
        for movie, external in db.execute("SELECT movie_id,imdb_id FROM links WHERE imdb_id IS NOT NULL"):
            links[external].append(movie)
    return stats, dict(links), receipt


def aggregate_movielens_corpus(archive_path, output_dir, *, cutoff="2021-08-02",
                               expected_md5=MOVIELENS_MD5, expected_sha256=MOVIELENS_SHA256,
                               expected_bytes=MOVIELENS_BYTES, progress_callback=None):
    """One bounded full-source pass for the predeclared external-work prior.

    The official export is ordered by (userId,movieId). Requiring strict order
    proves uniqueness using constant pair state. Duplicates or disorder stop the
    pass; no unjustified row-count-as-rater-count approximation is accepted.
    Counts and half-star sums use two uint64 arrays, one slot per source movie.
    """
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if (archive_path.stat().st_size != expected_bytes or digest(archive_path, "md5") != expected_md5
            or digest(archive_path) != expected_sha256):
        raise ValueError("MovieLens corpus archive integrity mismatch")
    limit = instant(cutoff).timestamp()
    if instant(cutoff) > instant("2021-08-02"):
        raise ValueError("Complete-source RM prior fit cannot exceed August2")
    if instant(MOVIELENS_KNOWN_AT) >= instant(cutoff):
        raise ValueError("Full source corpus not released before requested cutoff")
    with zipfile.ZipFile(archive_path) as archive:
        members = _zip_members(archive)
        ids, externals, slots, seen_external = [], [], {}, set()
        stream, reader = _csv(archive, members["links.csv"], ["movieId", "imdbId", "tmdbId"])
        with stream:
            for row in reader:
                movie = _positive(row["movieId"])
                if None in row or movie in slots:
                    raise ValueError("Duplicate/invalid corpus movie ID")
                external = imdb_id(row["imdbId"]) if row["imdbId"] else None
                if external is None or external in seen_external:
                    raise ValueError("Corpus work identity missing or colliding IMDb ID")
                seen_external.add(external)
                slots[movie] = len(ids)
                ids.append(movie)
                externals.append(external)
        counts, sums = array("Q", [0]) * len(ids), array("Q", [0]) * len(ids)
        rows = eligible = users = 0
        previous = None
        minimum = maximum = min_units = max_units = None
        generated_before = instant("2019-11-22").timestamp()
        stream, reader = _csv(archive, members["ratings.csv"], ["userId", "movieId", "rating", "timestamp"])
        with stream:
            for row in reader:
                user, movie = _positive(row["userId"]), _positive(row["movieId"])
                rating, stamp = float(row["rating"]), int(row["timestamp"])
                if (None in row or not math.isfinite(rating) or not .5 <= rating <= 5
                        or rating * 2 != int(rating * 2) or stamp < 0 or stamp >= generated_before
                        or str(stamp) != row["timestamp"] or movie not in slots):
                    raise ValueError("Invalid corpus rating/time/reference")
                key = (user, movie)
                if previous is not None and key <= previous:
                    raise ValueError("Corpus user/movie duplicate or unsorted source; aggregation stopped")
                users += int(previous is None or previous[0] != user)
                previous = key
                units = int(rating * 2)
                rows += 1
                minimum = stamp if minimum is None else min(minimum, stamp)
                maximum = stamp if maximum is None else max(maximum, stamp)
                min_units = units if min_units is None else min(min_units, units)
                max_units = units if max_units is None else max(max_units, units)
                if stamp < limit:
                    slot = slots[movie]
                    counts[slot] += 1
                    sums[slot] += units
                    eligible += 1
                if progress_callback is not None and rows % 250000 == 0:
                    progress_callback(rows)
        if not rows or not eligible:
            raise ValueError("Empty source or no eligible source ratings")
        if expected_md5 == MOVIELENS_MD5 and (rows != 25000095 or users != 162541 or len(ids) != 62423):
            raise ValueError("Official corpus row/user/work count mismatch")
    rated = [slot for slot, n in enumerate(counts) if n]
    mu = math.fsum(sums[slot] / (2 * counts[slot]) for slot in rated) / len(rated)
    output_dir.mkdir(parents=True)
    aggregate_path = output_dir / "per-work.csv"
    with aggregate_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["movie_id", "imdb_id", "count", "half_star_sum", "mean"])
        for slot in sorted(range(len(ids)), key=ids.__getitem__):
            n = counts[slot]
            writer.writerow([ids[slot], externals[slot], n, sums[slot], sums[slot] / (2 * n) if n else ""])
    prior = {"mu": mu, "m": 20., "fit_role": "rm_fit", "fit_boundary": iso(cutoff),
             "weighting": "equal_unique_source_imdb_work_means_positive_support_only",
             "temporal_mode": "retrospective_identity", "temporal_mask": 0,
             "numeric_history_mask": 1, "fit_corpus": "full_public_movielens25m_2019",
             "candidate_source_work_count": len(ids), "fitted_source_work_count": len(rated),
             "zero_rating_source_work_count": len(ids) - len(rated),
             "source_rating_rows_scanned": rows, "eligible_rating_events": eligible,
             "source_unique_raters": users, "source_known_at": MOVIELENS_KNOWN_AT,
             "source_max_event_timestamp": maximum, "archive_sha256": expected_sha256,
             "per_work_path": aggregate_path.as_posix(), "per_work_sha256": digest(aggregate_path),
             "m_selection": "fixed20_before_pilot; no development tuning",
             "kion_outcomes_used": False, "kion_pilot_selection_used": False,
             "prior_corpus_policy": "Root-authorized fixed complete public source corpus; independent of accepted KION link coverage"}
    write_json(output_dir / "prior.json", prior)
    receipt = {"contract": CONTRACT, "source_url": MOVIELENS_URL, "archive_sha256": expected_sha256,
               "archive_md5": expected_md5, "archive_bytes": expected_bytes,
               "rows_scanned": rows, "eligible_rating_events": eligible, "unique_users": users,
               "source_works": len(ids), "unique_imdb_ids": len(seen_external), "imdb_collisions": 0,
               "rated_works": len(rated), "zero_rating_works": len(ids) - len(rated),
               "min_timestamp": minimum, "max_timestamp": maximum,
               "min_rating": min_units / 2, "max_rating": max_units / 2,
               "cutoff_exclusive": iso(cutoff), "dataset_known_at": MOVIELENS_KNOWN_AT,
               "pair_order_validation": "strict lexicographic userId,movieId for every source row",
               "duplicate_user_movie_pairs": 0, "dedup_method": "strict_order_proof; duplicates_or_disorder_abort",
               "array_slots": len(ids), "count_array_type": counts.typecode, "sum_array_type": sums.typecode,
               "array_bytes": counts.itemsize * len(counts) + sums.itemsize * len(sums),
               "aggregate_path": aggregate_path.as_posix(), "aggregate_sha256": digest(aggregate_path),
               "prior_path": (output_dir / "prior.json").as_posix(), "prior_sha256": digest(output_dir / "prior.json"),
               "prior": prior, "processed_at": iso(datetime.now(UTC)), "network_requests": 0}
    write_json(output_dir / "receipt.json", receipt)
    return receipt


def fit_rating_prior(work_observations, *, fit_work_ids, fit_role, fit_boundary, m,
                     temporal_mode="strict"):
    """Equal weight per accepted work; duplicates/editions never reweight mu."""
    if fit_role not in {"sft_train", "rm_fit"} or not math.isfinite(m) or m <= 0:
        raise ValueError("Prior requires an explicit train role and positive m")
    if temporal_mode not in {"strict", "retrospective_identity"}:
        raise ValueError("Unknown prior temporal mode")
    boundary = instant(fit_boundary)
    if fit_role == "rm_fit" and boundary > instant("2021-08-02"):
        raise ValueError("RM prior fit extends beyond August 2")
    fit_ids, works = set(map(str, fit_work_ids)), {}
    for row in work_observations:
        if str(row["work_id"]) not in fit_ids or row.get("movielens_missing", 1):
            continue
        if (row.get("temporal_mode") != temporal_mode
                or (temporal_mode == "strict" and not row.get("movielens_temporal_mask"))):
            raise ValueError("Prior observation temporal provenance mismatch")
        if instant(row["cutoff"]) > boundary:
            raise ValueError("Prior observation exceeds fit boundary")
        key, value = str(row["work_id"]), float(row["movielens_mean"])
        if not math.isfinite(value) or not .5 <= value <= 5:
            raise ValueError("Invalid observed rating for prior")
        if key in works and works[key] != value:
            raise ValueError("Conflicting work observations for prior")
        works[key] = value
    if not works:
        raise ValueError("No eligible works for train prior")
    return {"mu": statistics.mean(works.values()), "m": float(m), "fit_role": fit_role,
            "fit_boundary": iso(boundary), "fit_work_ids": sorted(works), "weighting": "equal_unique_work",
            "temporal_mode": temporal_mode, "temporal_mask": int(temporal_mode == "strict")}


def load_corpus_aggregates(receipt_path, *, expected_receipt_sha256, cutoff,
                           expected_archive_sha256=MOVIELENS_SHA256):
    """Reuse the frozen complete-source aggregates for later accepted links.

    No raw rating rescan is needed. Earlier cutoffs are permitted only when all
    source events already precede them; otherwise the aggregate is insufficient.
    """
    receipt_path = Path(receipt_path)
    if digest(receipt_path) != expected_receipt_sha256:
        raise ValueError("Frozen corpus receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("contract") != CONTRACT or receipt.get("archive_sha256") != expected_archive_sha256:
        raise ValueError("Frozen corpus source/contract mismatch")
    if instant(receipt["dataset_known_at"]) >= instant(cutoff):
        raise ValueError("Source release not available at requested cutoff")
    if (instant(receipt["cutoff_exclusive"]) > instant(cutoff)
            and receipt["max_timestamp"] >= instant(cutoff).timestamp()):
        raise ValueError("Frozen work aggregates cannot reconstruct earlier source prefix")
    aggregate_path, prior_path = Path(receipt["aggregate_path"]), Path(receipt["prior_path"])
    if digest(aggregate_path) != receipt["aggregate_sha256"] or digest(prior_path) != receipt["prior_sha256"]:
        raise ValueError("Frozen corpus aggregate/prior hash mismatch")
    stats, links, total, rated = {}, {}, 0, 0
    with aggregate_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["movie_id", "imdb_id", "count", "half_star_sum", "mean"]:
            raise ValueError("Unexpected frozen work aggregate schema")
        for row in reader:
            movie, external = _positive(row["movie_id"]), imdb_id(row["imdb_id"])
            count, units = int(row["count"]), int(row["half_star_sum"])
            if movie in stats or external in links or count < 0 or not count <= units <= 10 * count:
                raise ValueError("Frozen source work identity/count mismatch")
            mean = units / (2 * count) if count else None
            if (mean is None and row["mean"] != "") or (mean is not None and float(row["mean"]) != mean):
                raise ValueError("Frozen work mean differs from exact half-star sum")
            stats[movie] = {"count": count, "mean": mean, "cutoff": receipt["cutoff_exclusive"]}
            links[external] = [movie]
            total += count
            rated += int(count > 0)
    if (len(stats) != receipt["source_works"] or total != receipt["eligible_rating_events"]
            or rated != receipt["rated_works"]):
        raise ValueError("Frozen corpus counters disagree")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior != receipt["prior"]:
        raise ValueError("Frozen corpus prior differs from receipt")
    source = dict(receipt, max_rating_timestamp=receipt["max_timestamp"],
                  frozen_receipt_sha256=expected_receipt_sha256)
    return stats, links, source, prior


def _fixed_source_prior_available(prior, boundary, source_receipt):
    # A predeclared public2019 corpus contains no future2021 fit information;
    # its protocol cutoff is distinct from when its complete corpus was known.
    archive = prior.get("archive_sha256")
    count, eligible = prior.get("source_rating_rows_scanned"), prior.get("eligible_rating_events")
    return (prior.get("fit_corpus") == "full_public_movielens25m_2019"
            and prior.get("kion_outcomes_used") is False and prior.get("kion_pilot_selection_used") is False
            and prior.get("m") == 20
            and isinstance(archive, str) and re.fullmatch(r"[0-9a-f]{64}", archive) is not None
            and archive == source_receipt.get("archive_sha256")
            and type(count) is int and count > 0 and type(eligible) is int and eligible == count
            and count == source_receipt.get("rows_scanned")
            and prior.get("source_known_at") is not None
            and prior["source_known_at"] == source_receipt.get("dataset_known_at")
            and instant(prior["source_known_at"]) < instant(boundary)
            and prior.get("source_max_event_timestamp") == source_receipt.get("max_rating_timestamp")
            and prior.get("source_max_event_timestamp", math.inf) < instant(boundary).timestamp()
            and prior.get("weighting") == "equal_unique_source_imdb_work_means_positive_support_only"
            and prior.get("m_selection") == "fixed20_before_pilot; no development tuning")


def rating_features(work_links, stats, source_links, source_receipt, *, boundary,
                    prior=None, temporal_mode="strict", low_support_threshold=20):
    """Transfer a work rating only through one accepted IMDb relation."""
    output = []
    for link in work_links:
        row = {"item_id": link["item_id"], "work_id": link.get("work_id"),
               "cutoff": iso(boundary), "temporal_mode": temporal_mode,
               "is_work_proxy": bool(link.get("is_work_proxy", False)),
               "movielens_mean": None, "movielens_count": 0,
               "movielens_log1p_count": 0., "movielens_shrunk_mean": None,
               "movielens_missing": 1, "movielens_temporal_mask": 0,
               "movielens_low_support": 0, "movielens_value_available_mask": 0,
               "movielens_numeric_history_mask": int(instant(source_receipt["dataset_known_at"]) < instant(boundary)),
               "movielens_identity_known_before_boundary": int(mapping_allowed(link, boundary, "strict")[0]),
               "numeric_source_known_at": source_receipt["dataset_known_at"],
               "linking_metadata_known_at": link.get("known_at"),
               "scope_assumption": link.get("scope_assumption"),
               "work_proxy_policy": link.get("work_proxy_policy", "explicit_direct_work_relation")}
        allowed, reason = mapping_allowed(link, boundary, temporal_mode)
        if link.get("rating_scope", link.get("scope")) != "work":
            allowed, reason = False, "unsupported_scope"
        if instant(source_receipt["dataset_known_at"]) >= instant(boundary):
            allowed, reason = False, "dataset_not_available_at_boundary"
        ids = source_links.get(imdb_id(link["imdb_id"]), []) if link.get("imdb_id") else []
        if not ids:
            allowed, reason = False, "no_movielens_link"
        elif len(ids) != 1:
            allowed, reason = False, "ambiguous_movielens_link"
        value = stats.get(ids[0]) if len(ids) == 1 else None
        if (value and instant(value["cutoff"]) > instant(boundary)
                and source_receipt["max_rating_timestamp"] >= instant(boundary).timestamp()):
            allowed, reason = False, "aggregation_exceeds_boundary"
            row["movielens_numeric_history_mask"] = 0
        if allowed and value and value["count"] > 0:
            n, mean = value["count"], value["mean"]
            row.update(movielens_mean=mean, movielens_count=n, movielens_log1p_count=math.log1p(n),
                       movielens_missing=0, movielens_temporal_mask=int(temporal_mode == "strict"),
                       movielens_low_support=int(n < low_support_threshold), movielens_value_available_mask=1)
            if prior is not None:
                if (prior.get("fit_role") not in {"sft_train", "rm_fit"}
                        or prior.get("temporal_mode") != temporal_mode
                        or (temporal_mode == "strict" and not prior.get("temporal_mask"))
                        or (instant(prior["fit_boundary"]) > instant(boundary)
                            and not _fixed_source_prior_available(prior, boundary, source_receipt))
                        or prior["m"] <= 0 or not math.isfinite(prior["m"])
                        or not math.isfinite(prior["mu"]) or not .5 <= prior["mu"] <= 5):
                    raise ValueError("Invalid or future-trained rating prior")
                row["movielens_shrunk_mean"] = (n * mean + prior["m"] * prior["mu"]) / (n + prior["m"])
        elif allowed:
            reason = "no_rating_before_cutoff"
        row["reason"] = reason
        output.append(row)
    return output


def pageviews_url(project, title, start, end):
    if project not in {"en.wikipedia.org", "ru.wikipedia.org"} or not title or title == "-":
        raise ValueError("Only explicit RU/EN article titles are supported")
    first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if first > last or first < date(2015, 7, 1):
        raise ValueError("Invalid pageviews interval")
    return ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/" + project
            + "/all-access/user/" + quote(title.replace(" ", "_"), safe="") + "/daily/"
            + first.strftime("%Y%m%d00") + "/" + last.strftime("%Y%m%d00"))


def _http(url, user_agent, max_bytes):
    """Use Schannel on Windows, retaining certificate verification everywhere."""
    if os.name == "nt":
        with tempfile.TemporaryDirectory() as temporary:
            body, headers = Path(temporary) / "body", Path(temporary) / "headers"
            result = subprocess.run(["curl.exe", "--silent", "--show-error", "--max-time", "45",
                                     "--max-filesize", str(max_bytes), "--proto", "=https",
                                     "--user-agent", user_agent, "--dump-header", str(headers),
                                     "--output", str(body), "--write-out", "%{http_code}", url],
                                    capture_output=True, text=True, timeout=50, check=False)
            if result.returncode:
                raise OSError("Verified HTTPS transport failed: " + result.stderr.strip())
            raw = body.read_bytes()
            parsed = {}
            for line in headers.read_text(encoding="iso-8859-1").splitlines():
                if line.startswith("HTTP/"):
                    parsed = {}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.lower()] = v.strip()
            return int(result.stdout), parsed, raw
    try:
        response = urlopen(Request(url, headers={"User-Agent": user_agent}), timeout=45)
    except HTTPError as error:
        response = error
    with response:
        return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read(max_bytes + 1)


class PageviewsClient:
    """Sequential, bounded, receipt-cached client. Retries consume request budget."""
    def __init__(self, cache_dir, *, user_agent, max_requests=20, max_bytes=2_000_000,
                 min_interval=1., max_attempts=3, transport=None, sleeper=time.sleep, clock=time.time):
        if (not re.search(r"\S+/\S+", user_agent) or not re.search(r"https?://|@", user_agent)
                or "example.org" in user_agent or max_requests < 0 or max_bytes <= 0
                or max_attempts < 1 or min_interval < 0):
            raise ValueError("Explicit real-contact UA and positive acquisition limits required")
        self.cache = Path(cache_dir)
        self.user_agent, self.max_requests, self.max_bytes = user_agent, max_requests, max_bytes
        self.min_interval, self.max_attempts = min_interval, max_attempts
        self.transport, self.sleep = transport or _http, sleeper
        self.clock = clock
        self.requests, self.last_request = 0, None

    def _check_throttle(self):
        for path in self.cache.glob("throttle-*.json"):
            blocked = json.loads(path.read_text(encoding="utf-8"))["retry_not_before_epoch"]
            if self.clock() < blocked:
                raise RuntimeError("Pageviews server throttle active; retry after " + iso(datetime.fromtimestamp(blocked, UTC)))

    def fetch(self, project, title, start, end):
        url = pageviews_url(project, title, start, end)
        key = hashlib.sha256(url.encode()).hexdigest()
        receipt_path, body_path = self.cache / (key + ".json"), self.cache / (key + ".body")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("url") != url or digest(body_path) != receipt["body_sha256"]:
                raise ValueError("Pageviews cache identity/hash mismatch")
            raw = body_path.read_bytes()
            return (json.loads(raw) if receipt["status"] == 200 else None), receipt
        history = []
        for attempt in range(self.max_attempts):
            self._check_throttle()
            if self.requests >= self.max_requests:
                raise RuntimeError("Pageviews request budget exhausted")
            if self.last_request is not None:
                self.sleep(max(0., self.min_interval - (time.monotonic() - self.last_request)))
            self.requests += 1
            status, headers, raw = self.transport(url, self.user_agent, self.max_bytes)
            self.last_request = time.monotonic()
            if len(raw) > self.max_bytes:
                raise ValueError("Pageviews response exceeds byte budget")
            history.append({"status": status, "attempt": attempt + 1,
                            "retrieved_at": iso(datetime.now(UTC)),
                            "body_sha256": hashlib.sha256(raw).hexdigest()})
            if status in {429, 503}:
                retry = headers.get("retry-after")
                try:
                    delay = float(retry)
                except (TypeError, ValueError):
                    try:
                        delay = email.utils.parsedate_to_datetime(retry).timestamp() - self.clock()
                    except (TypeError, ValueError):
                        delay = 5. * 2 ** attempt
                delay = max(5., delay)
                throttle = {"url": url, "status": status, "retry_not_before_epoch": self.clock() + delay}
                write_json(self.cache / ("throttle-" + signature(throttle) + ".json"), throttle)
                # Persist across titles/processes, including the final attempt.
                if delay > 60 or attempt + 1 == self.max_attempts:
                    break
                self.sleep(delay)
                continue
            break
        if status == 200:
            payload = json.loads(raw)
            parse_pageviews(payload, project=project, title=title, start=start, end=end)
        else:
            payload = None
        self.cache.mkdir(parents=True, exist_ok=True)
        if status not in {200, 404}:
            # Failed attempts are evidence, never a permanently poisoned data cache.
            failure_key = key + ".failed-" + signature(history)[:16]
            receipt_path, body_path = self.cache / (failure_key + ".json"), self.cache / (failure_key + ".body")
        if body_path.exists():
            raise FileExistsError(body_path)
        with body_path.open("xb") as stream:
            stream.write(raw)
        receipt = {"contract": CONTRACT, "url": url, "status": status, "project": project,
                   "title": title, "start": str(start), "end": str(end),
                   "access": "all-access", "agent": "user", "granularity": "daily",
                   "license": "CC0-1.0", "temporal_mode": "retrieved_historical_counts",
                   "body_sha256": hashlib.sha256(raw).hexdigest(), "body_bytes": len(raw),
                   "retrieved_at": iso(datetime.now(UTC)), "attempts": history,
                   "user_agent": self.user_agent, "response_headers": headers}
        write_json(receipt_path, receipt)
        return payload, receipt


def parse_pageviews(payload, *, project, title, start, end):
    output = {}
    first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("Unexpected pageviews schema")
    for item in payload["items"]:
        stamp, count = str(item.get("timestamp", "")), item.get("views")
        if (not re.fullmatch(r"[0-9]{8}00", stamp) or type(count) is not int or count < 0
                or item.get("project") not in {project, project.removesuffix(".org")} or item.get("access") != "all-access"
                or item.get("agent") != "user" or item.get("granularity") != "daily"
                or item.get("article", "").replace(" ", "_") != title.replace(" ", "_")):
            raise ValueError("Pageviews item identity/filter/value mismatch")
        day = datetime.strptime(stamp, "%Y%m%d%H").date()
        if day < first or day > last or day in output:
            raise ValueError("Pageviews day outside request or duplicated")
        output[day] = count
    return output


def article_day_allowed(mapping, day, boundary, temporal_mode):
    allowed, reason = mapping_allowed(mapping, boundary, temporal_mode)
    if not allowed:
        return False, reason
    if (mapping.get("identity_status") != "historically_verified" or not mapping.get("history_evidence")
            or type(mapping.get("pageid")) is not int or mapping["pageid"] <= 0):
        return False, "historical_article_identity_unresolved"
    start, end = mapping.get("valid_from"), mapping.get("valid_to")
    day_start, day_end = instant(day), instant(day + timedelta(days=1))
    if start is None or instant(start) > day_start or (end and instant(end) < day_end):
        return False, "article_absent_or_title_changed"
    if mapping.get("redirect_policy") != "direct_title" or mapping.get("split_merge_unresolved", False):
        return False, "redirect_or_split_merge_unresolved"
    return True, "accepted"


def article_features(mapping, daily, *, boundary, lag_hours=24, temporal_mode="strict",
                     zero_policy=None, request_status=200, source_known_at=None,
                     title_policy="historically_verified"):
    """Sparse/missing API days remain null unless explicit evidenced zero policy."""
    end = last_available_day(boundary, lag_hours)
    if title_policy not in {"historically_verified", "literal_current_title_retrospective"}:
        raise ValueError("Unknown article title policy")
    if title_policy == "literal_current_title_retrospective" and temporal_mode != "retrospective_identity":
        raise ValueError("Literal title counts require explicit retrospective identity mode")
    row = {"entity_id": mapping["entity_id"], "role": mapping["role"],
           "project": mapping["project"], "pageid": mapping.get("pageid"),
           "title": mapping["title"], "cutoff": iso(boundary), "last_available_day": str(end),
           "lag_hours": lag_hours, "temporal_mode": temporal_mode,
           "article_coverage": 0, "mapping_status": mapping.get("match_status", "missing"),
           "data_status": "request_failed" if request_status not in {200, 404} else "missing_unresolved",
           "temporal_mask": 0, "observed_days": 0, "zero_assumed_days": 0,
           "source_snapshot_known_at": iso(source_known_at) if source_known_at else None,
           "source_snapshot_temporal_mask": int(source_known_at is not None and instant(source_known_at) < instant(boundary)),
           "retrospective_snapshot": int(source_known_at is None or instant(source_known_at) >= instant(boundary))}
    row["title_policy"] = title_policy
    row["literal_title_series"] = int(title_policy == "literal_current_title_retrospective")
    row["historical_title_unresolved"] = int(not all(article_day_allowed(mapping, end - timedelta(days=i), boundary,
                                                                          "retrospective_identity")[0] for i in range(60)))
    if mapping["role"] not in ROLES or mapping["project"] not in {"ru.wikipedia.org", "en.wikipedia.org"}:
        raise ValueError("Invalid article entity role/project")
    eligible = {}
    for offset in range(60):
        day = end - timedelta(days=offset)
        valid, reason = article_day_allowed(mapping, day, boundary, temporal_mode)
        historical_day_valid = valid
        if title_policy == "literal_current_title_retrospective":
            valid, reason = mapping_allowed(mapping, boundary, "retrospective_identity")
            if valid:
                reason = "accepted_current_identity_literal_title_history_unresolved" if row["historical_title_unresolved"] else "accepted_historical_title"
        if day == end:
            row["article_coverage"] = int(valid)
            row["mapping_reason"] = reason
        if temporal_mode == "strict" and not row["source_snapshot_temporal_mask"]:
            valid = False
            row["source_reason"] = "source_availability_unproven"
        if day == end:
            row["temporal_mask"] = int(valid and temporal_mode == "strict" and row["source_snapshot_temporal_mask"])
        if valid and request_status == 200 and day in daily:
            count = daily[day]
            if type(count) is not int or count < 0:
                raise ValueError("Invalid daily count")
            eligible[day] = count
            row["observed_days"] += 1
        elif (valid and historical_day_valid and request_status in {200, 404} and zero_policy
              and zero_policy.get("period_available") is True
              and zero_policy.get("no_known_source_outage") is True
              and zero_policy.get("evidence")
              and zero_policy.get("policy") == "omitted_day_assumed_zero"):
            eligible[day] = 0
            row["zero_assumed_days"] += 1
    for width in (7, 30):
        current = [eligible.get(end - timedelta(days=i)) for i in range(width)]
        previous = [eligible.get(end - timedelta(days=i)) for i in range(width, 2 * width)]
        n_current, n_previous = sum(x is not None for x in current), sum(x is not None for x in previous)
        complete, growth_complete = n_current == width, n_current == width and n_previous == width
        row.update({f"window_coverage_{width}d": n_current / width,
                    f"previous_window_coverage_{width}d": n_previous / width,
                    f"partial_window_{width}d": int(0 < n_current < width),
                    f"views_{width}d": sum(current) if complete else None,
                    f"log1p_views_{width}d": math.log1p(sum(current)) if complete else None,
                    f"growth_{width}d": (math.log1p(sum(current) / width) - math.log1p(sum(previous) / width)) if growth_complete else None,
                    f"views_{width}d_missing": int(not complete),
                    f"growth_{width}d_missing": int(not growth_complete)})
    if row["window_coverage_30d"] == 1:
        row["data_status"] = "zero_assumed_under_policy" if row["zero_assumed_days"] else "observed"
    elif row["observed_days"] or row["zero_assumed_days"]:
        row["data_status"] = "partial_window"
    elif not row["article_coverage"]:
        row["data_status"] = "article_absent_or_unresolved"
    row["zero_policy"] = zero_policy
    return row


def aggregate_people_features(people_ids, article_rows, *, role, project,
                              boundary=None, roster_known_at=None, temporal_mode="strict"):
    """Deduplicate people; retain denominators and omit missing people from stats."""
    if role not in {"actor", "director"}:
        raise ValueError("People roles are separate")
    ids, rows = set(map(str, people_ids)), {}
    if temporal_mode not in {"strict", "retrospective_identity"}:
        raise ValueError("Unknown people temporal mode")
    roster_valid = (boundary is not None and roster_known_at is not None
                    and instant(roster_known_at) < instant(boundary))
    for row in article_rows:
        if row["role"] == role and row["project"] == project and str(row["entity_id"]) in ids:
            if (temporal_mode == "strict" and (not roster_valid or not row.get("temporal_mask"))) or (boundary is not None and instant(row["cutoff"]) != instant(boundary)):
                continue
            key = str(row["entity_id"])
            if key in rows:
                raise ValueError("Duplicate person article; title segments must be combined before aggregation")
            rows[key] = row
    result = {"role": role, "project": project, "n_people_total": len(ids),
              "n_people_mapped": sum(r["article_coverage"] for r in rows.values()),
              "roster_temporal_mask": int(roster_valid), "temporal_mode": temporal_mode,
              "roster_status": "known_before_boundary" if roster_valid else "availability_unproven"}
    for width in (7, 30):
        valid = [r for r in rows.values() if r[f"views_{width}d"] is not None]
        result[f"n_people_with_valid_window_{width}d"] = len(valid)
        result[f"coverage_{width}d"] = len(valid) / len(ids) if ids else 0.
        for metric in (f"views_{width}d", f"growth_{width}d"):
            values = [r[metric] for r in rows.values() if r[metric] is not None]
            result.update({metric + "_mean": statistics.mean(values) if values else None,
                           metric + "_max": max(values) if values else None,
                           metric + "_median": statistics.median(values) if values else None,
                           metric + "_top3": statistics.mean(sorted(values, reverse=True)[:3]) if values else None,
                           metric + "_support": len(values)})
    return result


def population_wikipedia_matrix(base_values, metadata, normalized, series, *, origin, raw_rosters=None):
    """Fill literal-title work/person evidence without claiming old-title identity.

    ``series`` is keyed by (role, entity_id, project), and only contains acquired
    independently accepted entities. Unsampled people remain missing. Operational
    roster denominators deduplicate accepted IDs and exact unresolved name slots;
    they do not claim a complete historical cast or resolve a homonym.
    """
    values = base_values.copy()
    index = {name: i for i, name in enumerate(COLUMNS)}
    boundary = instant(date.fromordinal(int(origin)))
    computed = {}
    for key, record in series.items():
        mapping = record["mapping"]
        if key != (mapping["role"], mapping["entity_id"], mapping["project"]):
            raise ValueError("Acquired series role/entity/project key mismatch")
        computed[key] = article_features(mapping, record["daily"], boundary=boundary, lag_hours=24,
            temporal_mode="retrospective_identity", title_policy="literal_current_title_retrospective",
            request_status=record["receipt"]["status"], source_known_at=record["receipt"]["retrieved_at"])
    raw_rosters = raw_rosters or {}
    by_item = defaultdict(list)
    for person in normalized["people"]:
        if person["match_status"] == "accepted":
            by_item[(str(person["item_id"]), person["role"])].append(person)
    current_pages = {(str(a["item_id"]), a["role"], a["entity_id"], a["project"])
                     for a in normalized["articles"] if a["match_status"] == "accepted"}
    for article in normalized["articles"]:
        item = str(article["item_id"])
        if article["role"] != "work" or item not in metadata["item_id_to_token"] or article["match_status"] != "accepted":
            continue
        token = metadata["item_id_to_token"][item]
        language = article["project"][:2]
        key = ("work", article["entity_id"], article["project"])
        values[token, index["wiki_work_" + language + "_article_coverage"]] = 1
        if key in computed:
            feature = computed[key]
            for name in WIKI_WINDOW_FIELDS:
                value = feature[name]
                values[token, index["wiki_work_" + language + "_" + name]] = value if value is not None else 0
    for (item, role), people in by_item.items():
            if item not in metadata["item_id_to_token"]:
                continue
            token = metadata["item_id_to_token"][item]
            exact_names = {str(p.get("raw_name", "")).strip().casefold(): str(p["person_id"]) for p in people if p.get("raw_name")}
            slots = {exact_names.get(str(name).strip().casefold(), "unresolved-name:" + str(name).strip().casefold())
                     for name in raw_rosters.get((str(item), role), []) if str(name).strip()}
            ids = {str(p["person_id"]) for p in people}
            slots.update(ids)
            for language in ("ru", "en"):
                project = language + ".wikipedia.org"
                rows = [computed[(role, entity, project)] for entity in ids if (role, entity, project) in computed]
                aggregate = aggregate_people_features(slots, rows, role=role, project=project,
                    boundary=boundary, temporal_mode="retrospective_identity")
                aggregate["n_people_mapped"] = sum((str(item), role, entity, project) in current_pages for entity in ids)
                for metric in ("views_7d", "views_30d", "growth_7d", "growth_30d"):
                    aggregate[metric + "_missing"] = int(aggregate[metric + "_mean"] is None)
                aggregate["historical_title_unresolved"] = int(not rows or any(r["historical_title_unresolved"] for r in rows))
                aggregate["temporal_mask"] = 0
                roster_observed = any(str(name).strip() for name in raw_rosters.get((str(item), role), []))
                aggregate["roster_observed"] = int(roster_observed)
                aggregate["roster_size_missing"] = int(not roster_observed)
                if not roster_observed:
                    aggregate["coverage_7d"] = aggregate["coverage_30d"] = 0.
                for name in PEOPLE_METRICS + PEOPLE_SUPPORT:
                    value = aggregate[name]
                    values[token, index["wiki_" + role + "_" + language + "_" + name]] = value if value is not None else 0
    return values, computed


def populate_roster_denominators(values, metadata, raw_rosters):
    """Static operational exact-name slots; never a historical completeness claim."""
    index = {name: i for i, name in enumerate(COLUMNS)}
    for (item, role), names in raw_rosters.items():
        if str(item) not in metadata["item_id_to_token"] or role not in {"actor", "director"}:
            continue
        n = len({str(name).strip().casefold() for name in names if str(name).strip()})
        token = metadata["item_id_to_token"][str(item)]
        for language in ("ru", "en"):
            values[token, index["wiki_" + role + "_" + language + "_n_people_total"]] = n
            values[token, index["wiki_" + role + "_" + language + "_roster_observed"]] = int(n > 0)
            values[token, index["wiki_" + role + "_" + language + "_roster_size_missing"]] = int(n == 0)
    return values


def shared_ledger_inputs(ledger, *, work_proxy_policy="verified_relation_only"):
    """Adapter for the single entity-resolver owner; never promotes its status.

    Current sitelinks remain unresolved historically. Unverified edition/work
    relations cannot supply a rating. Historical article evidence is an explicit
    downstream supplement, never inferred from a QID or contemporary sitelink.
    """
    if ledger.get("contract") != "kion-entity-links-v1":
        raise ValueError("Unsupported entity ledger contract")
    if work_proxy_policy not in {"verified_relation_only", "accepted_work_identity_with_unverified_edition"}:
        raise ValueError("Unknown explicit work proxy policy")
    works, articles, people = [], [], []
    for entity_role, records in (("work", ledger.get("works", [])),
                                 ("person", ledger.get("people", []))):
        for record in records:
            role = record.get("role") if entity_role == "person" else "work"
            if role not in ROLES:
                continue
            candidate = record.get("candidate") or {}
            external = record.get("external_ids") or candidate.get("external_ids") or {}
            evidence = record.get("source_evidence") or candidate.get("source_evidence")
            known = record.get("source_known_at")
            entity_id = record.get("work_id") if role == "work" else record.get("person_id")
            status = record.get("status", "missing")
            if role == "work":
                relation = record.get("edition_relation", "unverified")
                proxy = (relation == "unverified" and work_proxy_policy == "accepted_work_identity_with_unverified_edition"
                         and status == "accepted" and entity_id and evidence)
                accepted = status if relation == "same_work" or proxy else "unsupported_scope"
                assumption = ("Accepted card-to-work identity is independently reviewed; unverified refers only to specific cut; value is a work proxy"
                              if proxy else "Confirmed same-work relation required")
                works.append({"item_id": record["item_id"], "work_id": entity_id,
                              "imdb_id": external.get("imdb"), "match_status": accepted,
                              "rating_scope": "work", "known_at": known, "evidence": evidence,
                              "is_work_proxy": not bool(record.get("edition_specific_identity")),
                              "edition_relation": relation, "ledger_record_sha256": signature(record),
                              "work_proxy_policy": work_proxy_policy, "scope_assumption": assumption})
            else:
                people.append({"item_id": record["item_id"], "person_id": entity_id,
                               "role": role, "match_status": status, "known_at": known,
                               "evidence": evidence, "mention_id": record.get("mention_id"),
                               "raw_name": record.get("raw_name")})
            for site, article in external.get("wikipedia", {}).items():
                if site not in {"ruwiki", "enwiki"}:
                    continue
                articles.append({"item_id": record["item_id"], "entity_id": entity_id,
                                 "role": role, "project": site[:2] + ".wikipedia.org",
                                 "title": article["title"], "pageid": article.get("pageid"),
                                 "match_status": accepted if role == "work" else status, "known_at": known, "evidence": evidence,
                                 "identity_status": article.get("identity_status", "retrospective"),
                                 "history_evidence": article.get("history_evidence", article.get("historical_evidence")),
                                 "valid_from": article.get("valid_from", article.get("validity_from")),
                                 "valid_to": article.get("valid_to", article.get("validity_to")),
                                 "redirect_policy": article.get("redirect_policy", "unresolved"),
                                 "split_merge_unresolved": article.get("split_merge_unresolved", False),
                                 "is_work_proxy": role == "work",
                                 "work_proxy_policy": work_proxy_policy if role == "work" else None,
                                 "scope_assumption": assumption if role == "work" else None,
                                 "ledger_record_sha256": signature(record)})
    return {"works": works, "articles": articles, "people": people,
            "ledger_sha256": signature(ledger), "work_proxy_policy": work_proxy_policy}


def acquire_article(mapping, client, *, boundary, lag_hours=24, temporal_mode="strict"):
    """Never send requests for unresolved historical title identity."""
    end = last_available_day(boundary, lag_hours)
    start = end - timedelta(days=59)
    allowed, reason = article_day_allowed(mapping, end, boundary, temporal_mode)
    if not allowed:
        return article_features(mapping, {}, boundary=boundary, lag_hours=lag_hours,
                                temporal_mode=temporal_mode, request_status=0), {"status": "abstained", "reason": reason}
    payload, receipt = client.fetch(mapping["project"], mapping["title"], start, end)
    daily = parse_pageviews(payload, project=mapping["project"], title=mapping["title"], start=start, end=end) if payload is not None else {}
    row = article_features(mapping, daily, boundary=boundary, lag_hours=lag_hours,
                           temporal_mode=temporal_mode, request_status=receipt["status"],
                           source_known_at=receipt.get("source_known_at", receipt.get("retrieved_at")))
    row["source_receipt_sha256"] = signature(receipt)
    row["mapping_sha256"] = signature(mapping)
    return row, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest-movielens")
    ingest.add_argument("--archive", required=True)
    ingest.add_argument("--database", required=True)
    ingest.add_argument("--work-links", required=True, help="JSON list with accepted shared IMDb links")
    ingest.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "ingest-movielens":
        links = json.loads(Path(args.work_links).read_text(encoding="utf-8"))
        ids = [row["imdb_id"] for row in links if row.get("match_status") in {"accepted", "matched"} and row.get("imdb_id")]
        receipt = ingest_movielens(args.archive, args.database, ids)
        write_json(args.receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
