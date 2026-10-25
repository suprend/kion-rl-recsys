"""Read-only standard-library check of the source-only Git projection.

This hashes preserved payloads without importing producers, opening omitted
fixtures, running numerical inference, or making network requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


AUTHORED = {
    "report-latex/rollout-reward256.json",
    "report-latex/rollout256.json", "report-latex/duration-summary.json",
    "reproduction/profiles/head-bank-transfer.json",
    "reproduction/train_gpt.py", "reproduction/test_train_gpt.py", "reproduction/GPT_TRAINING.md",
    "reproduction/execute_head_bank.py", "reproduction/test_execute_head_bank.py",
    "reproduction/head_bank_hooks.py", "reproduction/test_head_bank_hooks.py",
    "reproduction/gpt_admission.py", "reproduction/test_gpt_admission.py",
    "reproduction/gpt_text_people.py", "reproduction/test_gpt_text_people.py",
    "reproduction/gpt_related.py", "reproduction/test_gpt_related.py",
    "reproduction/gpt_diagnostics.py", "reproduction/test_gpt_diagnostics.py",
    "reproduction/gpt_runtime.py", "reproduction/test_gpt_runtime.py",
    "reproduction/gpt_sources.py", "reproduction/test_gpt_sources.py",
    "reproduction/gpt_complete.py", "reproduction/test_gpt_complete.py",
    "reproduction/execute_rl.py", "reproduction/test_execute_rl.py",
    "reproduction/rl_heads.py", "reproduction/test_rl_heads.py",
    "reproduction/transfer_inputs.py", "reproduction/test_transfer_inputs.py", "reproduction/TRANSFER_INPUTS.md",
    "reproduction/gpt_provider.py", "reproduction/test_gpt_provider.py",
    "reproduction/gpt_transitive.py", "reproduction/test_gpt_transitive.py", "reproduction/profiles/gpt-transitive.json",
    "reproduction/rl_bindings.py", "reproduction/test_rl_bindings.py",
    "reproduction/train_rl.py", "reproduction/test_train_rl.py", "reproduction/TRAIN_RL.md",
    "reproduction/gpt_inputs.py", "reproduction/test_gpt_inputs.py",
    "reproduction/GPT_INPUTS.md", "reproduction/profiles/gpt-inputs.json",
    "reproduction/prepare_head_bank.py", "reproduction/test_prepare_head_bank.py",
    "reproduction/PREPARE_HEAD_BANK.md", "reproduction/profiles/head-bank-request-r1.json",
    "reproduction/train_reward_model.py", "reproduction/test_train_reward_model.py",
    "reproduction/TRAIN_REWARD_MODEL.md",
    "reproduction/prepare_reward_data.py", "reproduction/test_prepare_reward_data.py",
    "reproduction/PREPARE_REWARD_DATA.md", "reproduction/profiles/kion-series-v1.json",
    "report-latex/reward1024.json",
    "reproduction/prepare_data.py", "reproduction/test_prepare_data.py", "reproduction/PREPARE_DATA.md",
    "reproduction/input_inventory.py", "reproduction/test_input_inventory.py",
    "reproduction/train_heads.py", "reproduction/test_train_heads.py",
    "reproduction/HEAD_TRAINING.md", "reproduction/profiles/head-bank-inputs.json",
    "reproduction/profiles/roots.example.json",
    "report-latex/evaluation64k.json",
    "reproduction/saved_metrics64k.py", "reproduction/export_saved_metrics64k.py",
    "reproduction/test_saved_metrics64k.py", "reproduction/requirements64k.txt",
    "reproduction/saved-metrics64k/manifest.json", "reproduction/saved-metrics64k/rankings.json",
    "tools/check_document_links.py", "tools/package_submission.py", "tools/test_package_submission.py",
    "report-latex/images/rollout-steps.pdf",
    ".gitattributes", ".gitignore", "README.md", "REPRODUCING.md",
    "REPORT.md", "REPORT.pdf", "PROJECTION.md", "projection.json", "verify_repository.py",
    "make_report.py", "test_make_report.py",
    "report-latex/main.tex", "report-latex/figure.py", "report-latex/feature-schema.json",
    "report-latex/evaluation16k.json",
    "report-latex/images/train-kl-and-map.pdf", "report-latex/images/train-kl-and-map.svg",
    "report_tool.py", "test_report_tool.py", "test_verify_repository.py",
    "reproduction/saved_metrics.py", "reproduction/export_saved_metrics.py",
    "reproduction/test_saved_metrics.py", "reproduction/saved-metrics/manifest.json",
    "reproduction/saved-metrics/rankings.json",
}
REVIEWABLE_COPIED_PATHS = {
    "source/README.md", "reports/REPORT.md", "reports/LIMITATIONS.md",
    "reproduction/README.md", "reproduction/INPUTS.md", "reproduction/ENVIRONMENTS.md",
    "reproduction/HAND_IN.md", "reproduction/kion_replay/verify.py",
    "reproduction/test_rm_portability.py", "reproduction/runtime-manifest.json",
}
OVERRIDE_FIELDS = {"path", "baseline_sha256", "bytes", "sha256", "reason"}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".cbm", ".bin", ".ckpt"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def member(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Invalid relative path")
    portable = PurePosixPath(relative)
    require(not portable.is_absolute() and portable.as_posix() == relative
            and not ({".", ".."} & set(portable.parts))
            and "\\" not in relative and ":" not in relative
            and all(ord(char) >= 32 for char in relative), "Unsafe path: " + relative)
    path = root
    for part in portable.parts:
        path /= part
        require(not path.is_symlink() and not path.is_junction(), "Linked path: " + relative)
    require(path.resolve().is_relative_to(root), "Escaping path: " + relative)
    return path


def reviewed_overrides(manifest: dict) -> dict[str, dict]:
    """Accept only explicitly reviewed copy changes, retaining their base pins."""
    rows = manifest.get("reviewed_copy_overrides", [])
    require(type(rows) is list and len(rows) <= len(REVIEWABLE_COPIED_PATHS),
            "Invalid reviewed-copy override list")
    result = {}
    for row in rows:
        require(type(row) is dict and set(row) == OVERRIDE_FIELDS,
                "Invalid reviewed-copy override record")
        relative = row["path"]
        require(isinstance(relative, str) and relative in REVIEWABLE_COPIED_PATHS,
                "Unapproved reviewed-copy override path: " + str(relative))
        require(relative not in result, "Duplicate reviewed-copy override: " + relative)
        require(type(row["bytes"]) is int and row["bytes"] >= 0,
                "Invalid reviewed-copy override size: " + relative)
        for field in ("baseline_sha256", "sha256"):
            value = row[field]
            require(isinstance(value, str) and len(value) == 64
                    and all(char in "0123456789abcdef" for char in value),
                    "Invalid reviewed-copy override " + field + ": " + relative)
        require(isinstance(row["reason"], str) and bool(row["reason"].strip()),
                "Missing reviewed-copy override reason: " + relative)
        result[relative] = row
    return result


def verify(root: Path) -> dict:
    root = root.resolve(strict=True)
    require(root.is_dir(), "Repository root is not a directory")
    manifest_path = member(root, "projection.json")
    require(manifest_path.stat().st_size <= 2 << 20, "Oversized projection manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("contract") == "kion-source-only-git-projection-v1", "Unknown projection contract")
    require(set(manifest["authored_files_not_hashed"]) == AUTHORED, "Unexpected authored-file allowance")
    rows = manifest["copied_files"]
    require(type(rows) is list and 0 < len(rows) <= 1000, "Invalid copied-file list")
    overrides = reviewed_overrides(manifest)
    seen = set()
    total = 0
    current_total = 0
    unchanged_total = 0
    override_baseline_total = 0
    override_current_total = 0
    for row in rows:
        relative = row["path"]
        require(relative not in seen and relative not in AUTHORED, "Duplicate or authored copied path")
        path = member(root, relative)
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, "Forbidden copied payload: " + relative)
        current = overrides.get(relative, row)
        if relative in overrides:
            require(current["baseline_sha256"] == row["sha256"],
                    "Reviewed-copy baseline SHA256 differs: " + relative)
            override_baseline_total += row["bytes"]
            override_current_total += current["bytes"]
        else:
            unchanged_total += row["bytes"]
        require(path.is_file() and path.stat().st_size == current["bytes"], "Missing or changed size: " + relative)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == current["sha256"], "Changed SHA256: " + relative)
        seen.add(relative)
        total += row["bytes"]
        current_total += current["bytes"]
    require(set(overrides) <= seen, "Reviewed-copy override is not a copied path")
    # These totals describe the frozen upstream selection, including pre-review bytes.
    require(manifest["copied_totals"] == {"files": len(seen), "bytes": total}, "Copied totals differ")
    excluded = set()
    omitted_bytes = 0
    for row in manifest["excluded_files"]:
        relative = row["path"]
        path = member(root, relative)
        require(relative not in excluded and relative not in seen, "Duplicate or overlapping exclusion")
        require(bool(row["reason"]), "Missing exclusion reason")
        if row["disposition"] == "replaced-by-repository-documentation":
            require(relative == "README.md", "Unexpected replacement")
        else:
            require(row["disposition"] == "omitted" and not path.exists(), "Excluded payload is present: " + relative)
        excluded.add(relative)
        omitted_bytes += row["bytes"]
    require(manifest["excluded_totals"] == {"files": len(excluded), "bytes": omitted_bytes}, "Excluded totals differ")
    require(manifest["upstream"]["payload_files"] == len(seen) + len(excluded)
            and manifest["upstream"]["payload_bytes"] == total + omitted_bytes, "Upstream accounting differs")
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if base == root:
            dirs[:] = [name for name in dirs if name != ".git"]
            files = [name for name in files if name != ".git"]
        for name in dirs + files:
            relative = (base / name).relative_to(root).as_posix()
            member(root, relative)
        actual.update((base / name).relative_to(root).as_posix() for name in files)
    require(actual == seen | AUTHORED, "Unexpected or missing repository files: " + str(sorted(actual ^ (seen | AUTHORED))))
    return {"state": "passed", "copied_files": len(seen), "copied_bytes": total,
            "copied_bytes_basis": "historical baseline",
            "unchanged_copied_files": len(seen) - len(overrides),
            "unchanged_copied_bytes": unchanged_total,
            "reviewed_override_files": len(overrides),
            "reviewed_override_baseline_bytes": override_baseline_total,
            "reviewed_override_current_bytes": override_current_total,
            "current_copied_bytes": current_total,
            "excluded_members": len(excluded), "stdlib_only": True,
            "numerical_execution": False, "training_updates": 0,
            "scope": "source-only checkout baseline/reviewed-copy hashes and finite file closure; authored files are not checksummed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root), ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, "repository verification failed: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
