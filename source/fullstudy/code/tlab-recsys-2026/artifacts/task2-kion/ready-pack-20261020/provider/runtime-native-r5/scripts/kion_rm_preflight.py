"""CPU-only RM input preflight; support counts are not approval to train RM.

Reads metadata.json and only five explicit train.npz members. It never decodes
histories, demographics, validation or test arrays, and never imports a trainer.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np


ROW_KEYS = ("user_id", "target_item_id", "target_date")
ALLOWED_MEMBERS = (*ROW_KEYS, "y", "reward")
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}
RM_FIT_CUTOFF = date(2021, 8, 2).toordinal()
SFT_CUTOFF = date(2021, 8, 9).toordinal()


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def array_hash(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def validate_metadata(metadata):
    if metadata.get("format_version") != 1 or metadata.get("special_tokens") != SPECIAL_TOKENS:
        raise ValueError("Unsupported metadata or special-token contract")
    if metadata.get("split_dates", {}).get("train") != "<2021-08-09":
        raise ValueError("Unexpected SFT train cutoff")
    size, mapping = metadata.get("vocab_size"), metadata.get("item_id_to_token")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 3 or not isinstance(mapping, dict):
        raise ValueError("Invalid policy vocabulary")
    for item, token in mapping.items():
        if (not isinstance(item, str) or not item.isdecimal() or str(int(item)) != item or
                isinstance(token, bool) or not isinstance(token, int)):
            raise ValueError("Invalid original item ID/token")
    if len(mapping) != size - 3 or sorted(mapping.values()) != list(range(3, size)):
        raise ValueError("Policy mapping must be one-to-one and contiguous after specials")
    saved = metadata.get("splits", {}).get("train", {}).get("saved")
    if isinstance(saved, bool) or not isinstance(saved, int) or saved <= 0:
        raise ValueError("Metadata must declare nonempty train row count")
    return size, {int(item): token for item, token in mapping.items()}, saved


def load_train_snapshot(payload, metadata):
    """Decode only the allowlist from the exact hashed compressed NPZ snapshot."""
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        if not set(ALLOWED_MEMBERS).issubset(archive.files):
            raise ValueError("Missing required train member")
        arrays = {name: archive[name] for name in ALLOWED_MEMBERS}
    size, mapping, count = validate_metadata(metadata)
    for name in (*ROW_KEYS, "y"):
        if arrays[name].dtype != np.dtype("int64") or arrays[name].shape != (count,):
            raise ValueError(f"Expected one-dimensional int64 train field: {name}")
    reward = arrays["reward"]
    if reward.dtype != np.dtype("float32") or reward.shape != (count,):
        raise ValueError("Expected one-dimensional float32 reward")
    if any(np.any(arrays[name] < 0) for name in ROW_KEYS):
        raise ValueError("Negative row key")
    days, target = arrays["target_date"], arrays["y"]
    if not np.all((days > 0) & (days < SFT_CUTOFF)):
        raise ValueError("Train target dates outside fixed SFT window")
    if np.any((target < 2) | (target >= size)):
        raise ValueError("PAD/BOS or out-of-vocabulary target token")
    expected = np.fromiter((mapping.get(int(item), 2) for item in arrays["target_item_id"]), np.int64, count=count)
    if not np.array_equal(target, expected):
        raise ValueError("Original item ID/target token mapping mismatch")
    finite = np.isfinite(reward)
    if np.any((reward[finite] < 0) | (reward[finite] > 1)):
        raise ValueError("Finite reward must be in [0,1]; nonfinite labels are excluded")
    order = np.lexsort(tuple(arrays[name] for name in reversed(ROW_KEYS)))
    left, right = order[:-1], order[1:]
    duplicate = np.ones(len(left), dtype=bool)
    for name in ROW_KEYS:
        duplicate &= arrays[name][left] == arrays[name][right]
    if duplicate.any():
        raise ValueError("Duplicate train row keys")
    return arrays, size, mapping


def summarize(arrays, size):
    target, reward, days = arrays["y"], arrays["reward"], arrays["target_date"]
    known, finite = target >= 3, np.isfinite(reward)
    before = days < RM_FIT_CUTOFF
    eligible = before & known & finite
    counts = np.bincount(target[eligible], minlength=size).astype(np.int64)
    item_counts = counts[3:]
    fit = {"target_cutoff_exclusive": "2021-08-02", "target_rows_before_cutoff": int(before.sum()),
           "eligible_finite_known_rows": int(eligible.sum()), "unknown_rows_before_cutoff": int((before & ~known).sum()),
           "nonfinite_reward_rows_before_cutoff": int((before & ~finite).sum()),
           "missing_reward_rows_before_cutoff": int((before & np.isnan(reward)).sum()),
           "infinite_reward_rows_before_cutoff": int((before & np.isinf(reward)).sum()),
           "eligible_zero_reward_rows": int((eligible & (reward == 0)).sum()),
           "eligible_one_reward_rows": int((eligible & (reward == 1)).sum()),
           "policy_items": size - 3, "zero_target_support_items": int((item_counts == 0).sum()),
           "one_to_four_target_support_items": int(((item_counts >= 1) & (item_counts <= 4)).sum()),
           "at_least_five_target_support_items": int((item_counts >= 5).sum()),
           "support_sum": int(item_counts.sum()),
           "support_definition": "eligible observed finite-reward target count; not seen-in-history count",
           "rolling_train_role_filter_permitted": True,
           "history_verification": "not inspected; RM-fit reuse requires compatible strict-past preparer provenance"}
    window = {"start_inclusive": "2021-08-02", "stop_exclusive": "2021-08-09",
              "old_rolling_train_rows": int((~before).sum()),
              "old_rolling_finite_known_rows": int((~before & known & finite).sum()),
              "ready_frozen_examples": False, "exported_as_select_or_calibration": False,
              "reason": "rolling contexts may contain events from Aug2..8; rebuild all histories strictly before Aug2"}
    return counts, fit, window


def require_role_ready(report, role):
    """Fail closed for callers: a valid diagnostic receipt is not a loader lease."""
    if role not in ("rm_fit", "rm_select", "rm_calibration"):
        raise ValueError("Unknown RM role")
    if report.get("readiness", {}).get(role) is not True:
        raise RuntimeError(f"{role} is not ready: preflight does not authorize fitting or evaluation")


def preflight(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preflight output must not already exist")
    if output == data or data in output.parents:
        raise ValueError("Output must be outside read-only prepared inputs")
    paths = {"metadata": data / "metadata.json", "train": data / "train.npz", "source": Path(__file__).resolve()}
    # Hash and parse the same single-read bytes; the final pass detects changes.
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: digest_bytes(payload) for name, payload in snapshots.items()}
    metadata = json.loads(snapshots["metadata"].decode("utf-8"))
    arrays, size, mapping = load_train_snapshot(snapshots["train"], metadata)
    counts, fit, window = summarize(arrays, size)
    inputs = {name: {"path": path.as_posix(), "sha256": hashes[name]} for name, path in paths.items()}
    ordered_ids = [item for item, token in sorted(mapping.items(), key=lambda pair: pair[1])]
    readiness = {"rm_fit": False, "rm_select": False, "rm_calibration": False,
                 "status": "blocked_not_ready_for_rm_fit_or_selection",
                 "scope": "schema/mapping/target-support preflight only; not RM feature freeze or launch approval",
                 "blockers": ["history arrays deliberately not inspected or rebuilt",
                              "Aug2..8 select/calibration require frozen histories strictly before Aug2",
                              "deterministic user-disjoint SHA-256 select/calibration partition not constructed",
                              "trained fallback or explicitly reviewed common action mask required for zero target support",
                              "RM scorer/features/calibrator and selected SFT loader contracts not frozen"]}
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "preflight_valid": True, "inputs": inputs, "readiness": readiness,
              "policy_vocabulary": {"size": size, "special_tokens": SPECIAL_TOKENS,
                                    "mapping_sha256": canonical_hash(metadata["item_id_to_token"]),
                                    "ordered_item_ids_sha256": canonical_hash(ordered_ids),
                                    "fit_scope": "policy structural vocabulary before Aug9; not RM learned fit scope"},
              "train": {"rows": len(arrays["y"]), "unique_users": int(len(np.unique(arrays["user_id"]))),
                        "known_targets": int((arrays["y"] >= 3).sum()), "unknown_targets": int((arrays["y"] == 2).sum()),
                        "missing_rewards": int(np.isnan(arrays["reward"]).sum()),
                        "nonfinite_rewards": int((~np.isfinite(arrays["reward"])).sum()),
                        "infinite_rewards": int(np.isinf(arrays["reward"]).sum()),
                        "allowed_arrays": list(ALLOWED_MEMBERS),
                        "row_keys_sha256": canonical_hash({name: array_hash(arrays[name]) for name in ROW_KEYS})},
              "rm_fit_target_support": fit, "rm_select_calibration_window": window,
              "required_partition": {"algorithm": "SHA-256 of canonical user key with fixed declared salt and bucket boundary",
                                     "salt": None, "bucket_boundary": None, "constructed": False,
                                     "requirements": "choose and pin before fit; all tied targets of a user stay together; select/calibration user sets disjoint; calibration labels never select RM"},
              "fallback": {"implemented": False, "required_for_zero_support": True,
                           "reason": "candidate embedding and candidate UNK may be untrained; seen-in-history does not establish target support",
                           "allowed_future_resolution": "trained metadata/global/item-prior fallback with support flag, or separately reviewed common support mask"},
              "caveats": ["Static metadata availability is unverified; retrospective snapshot is not strict as-of evidence.",
                          "Prepared reward is aggregated watched_pct/100 on observed user-item last-watch pairs, not sessions or impression response.",
                          "Finite normalized reward does not prove the label was historically available at target date.",
                          "Missing reward remains missing; observed zero is valid; unobserved candidates are not zero labels.",
                          "Full-prefix support counts describe the future frozen scorer; do not use them as strict-past features of earlier RM-fit rows."],
              "safety": {"read_files": ["metadata.json", "train.npz", "own source"],
                         "decoded_train_members": list(ALLOWED_MEMBERS), "histories_decoded": False,
                         "demographics_decoded": False, "validation_or_test_opened": False,
                         "models_imported_or_fit": False, "example_rows_exported": False,
                         "metadata_availability": "static_snapshot_unverified"}}
    if any(not path.is_file() or file_hash(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Preflight input/source changed while computing")
    output.mkdir(parents=True, exist_ok=False)
    support_path = output / "target_support.npz"
    np.savez_compressed(support_path, item_id=np.asarray(ordered_ids, np.int64),
                        token=np.arange(3, size, dtype=np.int64), finite_known_target_count=counts[3:])
    report["support_artifact"] = {"file": support_path.name, "sha256": file_hash(support_path),
                                  "role": "diagnostic only; not a trained feature or fallback"}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    manifest = {"format_version": 1, "status": readiness["status"], "inputs": inputs,
                "report": {"file": report_path.name, "sha256": file_hash(report_path)},
                "support_artifact": report["support_artifact"], "rm_training_ready": False}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = preflight(args.data, args.output)
    print(json.dumps({"preflight_valid": report["preflight_valid"], "readiness": report["readiness"],
                      "rm_fit_target_support": report["rm_fit_target_support"],
                      "rm_select_calibration_window": report["rm_select_calibration_window"]}, indent=2))


if __name__ == "__main__":
    main()
