"""Synthetic-only regression tests for the bounded packaging helper."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / "package.py"
SPEC = importlib.util.spec_from_file_location("submission_package", MODULE_PATH)
pkg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pkg)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="package synthetic ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source with spaces"
        self.root.mkdir()
        (self.root / "evidence").mkdir()
        (self.root / "README.md").write_text("# Synthetic hand-in\n\n[Report](REPORT.md)\n", encoding="utf-8")
        (self.root / "REPORT.md").write_text("# Synthetic report\n\n[Evidence](evidence/check.json)\n", encoding="utf-8")
        (self.root / "evidence/check.json").write_text('{"synthetic": true}\n', encoding="utf-8")
        self.selection = self.base / "selection.json"
        self.manifest = self.base / "frozen.json"
        self.approval = self.base / "approval.json"
        self.archive = self.base / "hand in.zip"
        self.receipt = self.base / "receipt.json"
        self.plan = {"version": 1, "roots": {"R": self.root.as_posix()}, "max_total_bytes": 2 * 1024 * 1024,
                     "acceptance_tiers": [{"id": "synthetic-structure", "status": "passed", "basis": "Synthetic test evidence only.", "evidence_members": ["evidence/check.json"]},
                                          {"id": "real-model-replay", "status": "not_run", "basis": "No actual data or models in unit tests.", "evidence_members": []}],
                     "files": [{"root": "R", "source": n, "member": n, "purpose": "Synthetic test file"} for n in ("README.md", "REPORT.md", "evidence/check.json")]}
        self.save_plan()

    def save_plan(self):
        self.selection.write_bytes(pkg.json_bytes(self.plan))

    def freeze_and_approve(self):
        pkg.write_new_json(self.manifest, pkg.prepare_selection(self.selection))
        pkg.approve(self.manifest, self.approval, "synthetic-unit-test")

    def build(self):
        self.freeze_and_approve()
        return pkg.build(self.manifest, self.approval, self.archive, self.receipt)

    def rewrite_zip(self, changes=None, extra=None, mutate_info=None):
        rewritten = self.base / "altered.zip"
        with zipfile.ZipFile(self.archive) as src, zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for old in src.infolist():
                info = pkg.zip_info(old.filename)
                if mutate_info is not None:
                    mutate_info(info)
                data = src.read(old.filename)
                if changes and old.filename in changes:
                    data = changes[old.filename]
                dst.writestr(info, data)
            if extra:
                for name, data in extra:
                    dst.writestr(pkg.zip_info(name), data)
        return rewritten

    def test_lifecycle_determinism_full_readback_and_safe_extract(self):
        result = self.build()
        self.assertTrue(result["full_readback"])
        self.assertEqual(result["sha256"], pkg.read_json(self.receipt)["sha256"])
        second = self.base / "second.zip"
        pkg.build(self.manifest, self.approval, second, self.base / "second-receipt.json")
        self.assertEqual(self.archive.read_bytes(), second.read_bytes())
        extraction = self.base / "fresh extracted directory with spaces"
        checked = pkg.verify(self.archive, self.manifest, extraction)
        self.assertTrue(checked["full_readback"])
        self.assertEqual((extraction / "REPORT.md").read_bytes(), (self.root / "REPORT.md").read_bytes())
        with zipfile.ZipFile(self.archive) as bundle:
            manifest = bundle.read(pkg.INTERNAL_MANIFEST)
            self.assertNotIn(self.root.as_posix().encode(), manifest)
            self.assertEqual(bundle.namelist(), sorted(bundle.namelist()))

    def test_cli_verify_from_extracted_script_and_spaced_cwd(self):
        (self.root / "source").mkdir()
        shutil.copyfile(MODULE_PATH, self.root / "source/package.py")
        self.plan["files"].append({"root": "R", "source": "source/package.py", "member": "source/package.py", "purpose": "Synthetic CLI fixture"})
        self.save_plan()
        self.build()
        extracted = self.base / "clean extraction with spaces"
        pkg.verify(self.archive, self.manifest, extracted)
        result = subprocess.run([sys.executable, str(extracted / "source/package.py"), "verify", "--archive", str(self.archive)], cwd=extracted, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["full_readback"])

    def test_unlisted_files_are_not_discovered_or_read(self):
        (self.root / "raw").mkdir()
        (self.root / "raw/secret.key").write_text("not selected", encoding="utf-8")
        value = pkg.prepare_selection(self.selection)
        self.assertEqual(len(value["package"]["files"]), 3)

    def test_changed_source_prevents_build_and_receipt(self):
        self.freeze_and_approve()
        (self.root / "REPORT.md").write_text("Changed\n", encoding="utf-8")
        with self.assertRaises(pkg.PackageError):
            pkg.build(self.manifest, self.approval, self.archive, self.receipt)
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.receipt.exists())

    def test_changed_origin_prevents_build(self):
        original = self.root / "original.py"
        original.write_text("# old original\n", encoding="utf-8")
        self.plan["files"][0]["origin"] = {"root": "R", "source": "original.py", "must_match": False}
        self.save_plan()
        self.freeze_and_approve()
        original.write_text("# new original\n", encoding="utf-8")
        with self.assertRaises(pkg.PackageError):
            pkg.build(self.manifest, self.approval, self.archive, self.receipt)
        self.assertFalse(self.receipt.exists())

    def test_must_match_origin(self):
        (self.root / "original.py").write_text("# different\n", encoding="utf-8")
        self.plan["files"][0]["origin"] = {"root": "R", "source": "original.py", "must_match": True}
        self.save_plan()
        with self.assertRaisesRegex(pkg.PackageError, "does not match"):
            pkg.prepare_selection(self.selection)

    def test_selection_byte_change_invalidates_freeze(self):
        self.freeze_and_approve()
        self.selection.write_bytes(self.selection.read_bytes() + b"\n")
        with self.assertRaisesRegex(pkg.PackageError, "freshness mismatch"):
            pkg.build(self.manifest, self.approval, self.archive, self.receipt)

    def test_manifest_byte_change_invalidates_approval(self):
        self.freeze_and_approve()
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(pkg.PackageError, "exact-manifest approval"):
            pkg.build(self.manifest, self.approval, self.archive, self.receipt)

    def test_failed_full_readback_never_writes_receipt(self):
        self.freeze_and_approve()
        with mock.patch.object(pkg, "verify", side_effect=pkg.PackageError("synthetic readback failure")):
            with self.assertRaisesRegex(pkg.PackageError, "readback failure"):
                pkg.build(self.manifest, self.approval, self.archive, self.receipt)
        self.assertTrue(self.archive.exists())
        self.assertFalse(self.receipt.exists())

    def test_changed_manifest_approval_or_source_during_readback_never_receipted(self):
        for changed in ("manifest", "approval", "source"):
            with self.subTest(changed=changed):
                self.freeze_and_approve()
                real_verify = pkg.verify
                def change_during_verify(*args, **kwargs):
                    result = real_verify(*args, **kwargs)
                    target = {"manifest": self.manifest, "approval": self.approval,
                              "source": self.root / "REPORT.md"}[changed]
                    target.write_bytes(target.read_bytes() + b"\n")
                    return result
                with mock.patch.object(pkg, "verify", side_effect=change_during_verify):
                    with self.assertRaises(pkg.PackageError):
                        pkg.build(self.manifest, self.approval, self.archive, self.receipt)
                self.assertFalse(self.receipt.exists())
                # Only synthetic temporary fixture controls are reset for next case.
                self.manifest.unlink()
                self.approval.unlink()
                self.archive.unlink()

    def test_existing_output_receipt_and_extraction_refused(self):
        self.build()
        before = self.archive.read_bytes()
        with self.assertRaisesRegex(pkg.PackageError, "overwrite"):
            pkg.build(self.manifest, self.approval, self.archive, self.base / "other-receipt.json")
        self.assertEqual(before, self.archive.read_bytes())
        existing = self.base / "existing empty"
        existing.mkdir()
        with self.assertRaisesRegex(pkg.PackageError, "must not exist"):
            pkg.verify(self.archive, extract_to=existing)
        self.assertEqual(list(existing.iterdir()), [])

    def test_fresh_manifest_and_approval_no_overwrite(self):
        self.freeze_and_approve()
        with self.assertRaisesRegex(pkg.PackageError, "overwrite"):
            pkg.write_new_json(self.manifest, {})
        with self.assertRaisesRegex(pkg.PackageError, "overwrite"):
            pkg.approve(self.manifest, self.approval, "test")

    def test_missing_local_link_blocks_preview(self):
        (self.root / "README.md").write_text("[Missing](old/report.md)\n", encoding="utf-8")
        with self.assertRaisesRegex(pkg.PackageError, "unclosed local link"):
            pkg.prepare_selection(self.selection)

    def test_reference_html_wiki_and_encoded_traversal_link_checks(self):
        text = "[Report][r]\n[r]: REPORT.md\n<img src='REPORT.md'>\n[[REPORT.md]]\n"
        self.assertEqual(pkg.markdown_link_errors({"README.md": text}, ["README.md", "REPORT.md"]), [])
        bad = "[x][missing]\n[x](%2e%2e/secret.md)\n"
        self.assertEqual(len(pkg.markdown_link_errors({"README.md": bad}, ["README.md"])), 2)

    def test_link_casing_must_work_on_case_sensitive_recipient(self):
        self.assertEqual(len(pkg.markdown_link_errors({"README.md": "[report](report.md)\n"}, ["README.md", "REPORT.md"])), 1)

    def test_code_fence_links_do_not_require_payload_files(self):
        text = "```md\n[x](not-a-file.md)\n```\n`[x](another.md)`\n[actual](REPORT.md)\n"
        self.assertEqual(pkg.markdown_link_errors({"README.md": text}, ["README.md", "REPORT.md"]), [])

    def test_members_and_source_traversal_case_collision_duplicate(self):
        for name in ("../outside.py", "/absolute.py", "C:/local.py", "a\\b.py", "a/../b.py", "NUL.py", "a./x.py", "a /x.py"):
            with self.subTest(name=name), self.assertRaises(pkg.PackageError):
                pkg.safe_name(name)
        for names in (["README.md", "readme.MD"], ["a", "a/b.py"], ["a.py", "a.py"]):
            with self.subTest(names=names), self.assertRaises(pkg.PackageError):
                pkg.check_names(names)
        self.plan["files"].append({"root": "R", "source": "README.md", "member": "copy.md", "purpose": "duplicate"})
        self.save_plan()
        with self.assertRaisesRegex(pkg.PackageError, "Same source"):
            pkg.prepare_selection(self.selection)

    def test_unicode_collision_and_reserved_manifest(self):
        with self.assertRaises(pkg.PackageError):
            pkg.check_names(["e\u0301.txt"])
        self.plan["files"][0]["member"] = pkg.INTERNAL_MANIFEST
        self.save_plan()
        with self.assertRaisesRegex(pkg.PackageError, "Duplicate"):
            pkg.prepare_selection(self.selection)

    def test_raw_history_cache_and_secret_paths_denied(self):
        for name in ("raw/x.csv", "cache/model.txt", "histories/x.txt", ".env", "other/.env.production", "source/x.pem", "events.jsonl", "data.sqlite", "full-cache.zip"):
            with self.subTest(name=name), self.assertRaises(pkg.PackageError):
                pkg.check_payload_path(name)

    def test_common_credential_scan(self):
        (self.root / "REPORT.md").write_text("ghp_" + "a" * 30 + "\n", encoding="utf-8")
        with self.assertRaisesRegex(pkg.PackageError, "Possible credential"):
            pkg.prepare_selection(self.selection)

    def test_explicit_numerical_and_weights_admission(self):
        for suffix, admission in (("npz", "derived_fixture"), ("pt", "frozen_weights")):
            with self.subTest(suffix=suffix):
                name = "replay." + suffix
                with self.assertRaises(pkg.PackageError):
                    pkg.check_payload_path(name)
                with self.assertRaises(pkg.PackageError):
                    pkg.check_payload_path(name, admission)
                pkg.check_payload_path(name, admission, "Synthetic reviewed opaque bytes only")

    def test_admitted_opaque_files_are_hashed_without_deserialization(self):
        (self.root / "fixture.npz").write_bytes(b"synthetic opaque fixture bytes, not actual arrays")
        self.plan["files"].append({"root": "R", "source": "fixture.npz", "member": "reproduction/fixture.npz", "purpose": "Synthetic opacity test", "admission": "derived_fixture", "admission_reason": "Synthetic derived-byte admission; no anonymity assertion"})
        self.save_plan()
        self.build()
        self.assertTrue(pkg.verify(self.archive)["full_readback"])

    def test_derived_fixture_enum_replaces_prior_name_without_alias(self):
        prior_name = "anonymous" + "_fixture"
        with self.assertRaisesRegex(pkg.PackageError, "Unknown admission"):
            pkg.check_payload_path("fixture.npz", prior_name, "Obsolete admission name")
        pkg.check_payload_path("fixture.npz", "derived_fixture", "Explicit bounded derivation; anonymity not established")

    def test_report_relative_hierarchy_is_preserved(self):
        (self.root / "reports").mkdir()
        (self.root / "reports/REPORT.md").write_text("# Synthetic report\n\n[Evidence](../evidence/check.json)\n", encoding="utf-8")
        (self.root / "README.md").write_text("# Synthetic hand-in\n\n[Report](reports/REPORT.md)\n", encoding="utf-8")
        self.plan["files"][1].update({"source": "reports/REPORT.md", "member": "reports/REPORT.md"})
        self.save_plan()
        self.build()
        extraction = self.base / "relative hierarchy with spaces"
        pkg.verify(self.archive, self.manifest, extraction)
        self.assertTrue((extraction / "reports/REPORT.md").is_file())
        self.assertFalse((extraction / "REPORT.md").exists())

    def test_actual_tier_basis_and_evidence_required(self):
        for change in ({"status": "planned"}, {"status": "passed", "evidence_members": []}, {"basis": ""}, {"evidence_members": ["absent.json"]}):
            tiers = copy.deepcopy(self.plan["acceptance_tiers"])
            tiers[0].update(change)
            with self.subTest(change=change), self.assertRaises(pkg.PackageError):
                pkg.validate_tiers(tiers, [f["member"] for f in self.plan["files"]])

    def test_byte_budget_and_duplicate_json_keys(self):
        self.plan["max_total_bytes"] = 1
        self.save_plan()
        with self.assertRaises(pkg.PackageError):
            pkg.prepare_selection(self.selection)
        self.selection.write_text('{"version": 1, "version": 1}', encoding="utf-8")
        with self.assertRaisesRegex(pkg.PackageError, "Duplicate JSON key"):
            pkg.read_json(self.selection)

    def test_source_symlink_is_rejected(self):
        linked = self.root / "linked.md"
        try:
            linked.symlink_to(self.root / "REPORT.md")
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows file-symlink privilege unavailable")
            raise
        with self.assertRaisesRegex(pkg.PackageError, "Symlink/junction"):
            pkg.source_path({"R": str(self.root)}, "R", "linked.md")

    def test_mocked_windows_junction_attributes_rejected(self):
        original = Path.lstat
        def attribute_check(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == self.root:
                return type("JunctionStat", (), {"st_mode": result.st_mode, "st_file_attributes": 0x400})()
            return result
        with mock.patch.object(Path, "lstat", attribute_check):
            with self.assertRaisesRegex(pkg.PackageError, "reparse point"):
                pkg.source_path({"R": str(self.root)}, "R", "REPORT.md")

    @unittest.skipUnless(os.name == "nt", "Native Windows junction scenario")
    def test_native_windows_junction_source_and_extraction_parent_rejected(self):
        junction = self.base / "junction with spaces"
        result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(self.root)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with self.assertRaisesRegex(pkg.PackageError, "reparse point"):
            pkg.source_path({"R": str(junction)}, "R", "REPORT.md")
        self.build()
        with self.assertRaisesRegex(pkg.PackageError, "reparse point"):
            pkg.verify(self.archive, extract_to=junction / "fresh")
        self.assertFalse((self.root / "fresh").exists())

    def test_extra_or_duplicate_zip_members_refused_before_extract(self):
        self.build()
        for extra in ([('extra.txt', b"extra")], [('README.md', b"duplicate")]):
            with self.subTest(extra=extra):
                changed = self.rewrite_zip(extra=extra)
                target = self.base / "not created"
                with self.assertRaises(pkg.PackageError):
                    pkg.verify(changed, extract_to=target)
                self.assertFalse(target.exists())
                changed.unlink()

    def test_zip_traversal_symlink_and_noncanonical_metadata_rejected(self):
        self.build()
        changed = self.rewrite_zip(extra=[("../escape.txt", b"no")])
        with self.assertRaises(pkg.PackageError):
            pkg.verify(changed)
        changed.unlink()
        def symlink(info):
            if info.filename == "README.md":
                info.external_attr = 0o120777 << 16
        changed = self.rewrite_zip(mutate_info=symlink)
        with self.assertRaisesRegex(pkg.PackageError, "Nonregular"):
            pkg.verify(changed)
        changed.unlink()
        changed = self.rewrite_zip(mutate_info=lambda info: setattr(info, "date_time", (2021, 1, 1, 0, 0, 0)))
        with self.assertRaisesRegex(pkg.PackageError, "metadata"):
            pkg.verify(changed)

    def test_payload_hash_and_embedded_manifest_tampering_rejected(self):
        self.build()
        changed = self.rewrite_zip(changes={"REPORT.md": b"not original\n"})
        with self.assertRaises(pkg.PackageError):
            pkg.verify(changed)
        changed.unlink()
        with zipfile.ZipFile(self.archive) as src:
            embedded = json.loads(src.read(pkg.INTERNAL_MANIFEST))
        embedded["files"][0]["sha256"] = "0" * 64
        changed = self.rewrite_zip(changes={pkg.INTERNAL_MANIFEST: pkg.json_bytes(embedded)})
        with self.assertRaises(pkg.PackageError):
            pkg.verify(changed, self.manifest)


if __name__ == "__main__":
    unittest.main()
