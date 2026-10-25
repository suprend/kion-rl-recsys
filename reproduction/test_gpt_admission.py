"""Admission unit checks: retained source reads and tiny temporary JSON only."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpt_admission as admission
from gpt_provider import BoundIO, FileBinding, canonical_hash, ORIGINAL_PROVIDER_SHA256

SOURCE = (Path(__file__).resolve().parents[1] / "source/fullstudy/code/tlab-recsys-2026/artifacts/"
          "task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts")
ROOT = "D:/historical/features"


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = []
        for name, digest in (("kion_complete_pack.py", ORIGINAL_PROVIDER_SHA256),
                             ("kion_complete_manifest.py", admission.ASSEMBLER_SHA256),
                             ("kion_local_pack.py", admission.LOCAL_SHA256)):
            self.bindings.append(FileBinding("D:/historical/source/" + name,
                                             str(SOURCE / name), digest, "source"))
        self.backend = admission.AdmissionBackend.from_sources(
            self.io(), provider_source=self.bindings[0].logical,
            assembler_source=self.bindings[1].logical, local_source=self.bindings[2].logical)
        self.c = self.backend.constants()
        self.scope = {"scope_id": "ready-pack-20261020-r1"}

    def io(self):
        return BoundIO(self.bindings, lambda: None)

    def put(self, name, value):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        physical = self.root / name
        physical.write_bytes(raw)
        logical = ROOT + "/" + name
        self.bindings = [b for b in self.bindings if b.logical != logical]
        digest = hashlib.sha256(raw).hexdigest()
        self.bindings.append(FileBinding(logical, str(physical), digest, "metadata"))
        return {"path": logical, "sha256": digest}

    def fixture(self):
        evidence = self.put("evidence.json", {"tiny": "evidence"})
        review = self.put("review.json", {"tiny": "review"})
        scope_receipt = {"path": ROOT + "/scope.json", "sha256": self.c["SCOPED_DECISION_SHA256"]}
        assessment = {
            "contract": "kion-complete-scoped-assessment-v1", "status": "scoped_data_ready_untrained",
            "blockers": [], "scope_decision": scope_receipt, "active_families": self.c["SCOPED_FAMILIES"],
            "provider_source": {"sha256": ORIGINAL_PROVIDER_SHA256},
            "base_manifest": {"sha256": "a" * 64}, "metadata": {"sha256": "b" * 64},
            "components": {name: {"ready": True, "payload": {}, "evidence": evidence, "review": review}
                           for name in self.c["SCOPED_COMPONENTS"]},
            "runtime_metadata": {"series_annotation": {"path": "D:/logical/series.json"},
                                 "related_historical_builder": {"path": "D:/logical/audience.py"}},
        }
        assessment["components"]["related_interest"]["payload"] = {"runtime": {"related_interest": {}}}
        payload, _ = self.backend.compose_payload(assessment)
        manifest = {
            "contract": self.c["CONTRACT"], "state": "completed", "integration_mode": "scoped_production",
            "scope_decision": scope_receipt, "scope_id": self.scope["scope_id"],
            "active_families": self.c["SCOPED_FAMILIES"], "status": "admitted_scoped",
            "full_row_fit_admitted": True, "full_package_admitted": False,
            "additional_ready_components": self.c["SCOPED_READY_COMPONENTS"],
            "required_families": self.c["SCOPED_FAMILIES"],
            "families": {name: {"status": "accepted", "reviewed": True, "scope": "approved_ready_scope",
                                "source_pins": [evidence, review]} for name in self.c["SCOPED_FAMILIES"]},
            "fit_cutoff": "2021-08-09", "validation_origin": "2021-08-09",
            "numeric_transforms": self.c["NUMERIC_TRANSFORMS"], "rowkeys_algorithm": self.c["ROW_ALGORITHM"],
            "source_pins": [{"path": b.logical, "sha256": b.sha256} for b in self.bindings if b.role == "source"],
            "base_cache": {"manifest_sha256": "a" * 64},
            "data": {"files": {"metadata.json": {"sha256": "b" * 64}}},
            "assembly": {"status": "scoped_data_ready_untrained", "assessment_sha256": canonical_hash(assessment)},
            **payload,
        }
        return manifest, assessment

    def run_admission(self, manifest, assessment, requested=None):
        self.put("assessment.json", assessment)
        manifest = copy.deepcopy(manifest)
        manifest.pop("signature", None)
        manifest["signature"] = canonical_hash(manifest)
        self.put("manifest.json", manifest)
        # Scope validation is tested independently; no historical decision or
        # its private bytes are opened to manufacture an end-to-end fixture.
        with patch.object(admission, "_scoped_decision", return_value=self.scope):
            return admission.admit_scoped(self.io(), ROOT, backend=self.backend,
                requested_scope_decision=requested or ROOT + "/scope.json")

    def test_backend_pinned_and_import_stdlib_only(self):
        self.assertEqual(len(self.c["AGREEMENT_COMPONENTS"]), 82)
        self.c["NUMERIC_TRANSFORMS"].clear()
        self.assertTrue(self.backend.constants()["NUMERIC_TRANSFORMS"])
        with self.assertRaises(TypeError):
            admission.AdmissionBackend(None, "x", "y", "z", {}, None)
        bad = list(self.bindings)
        bad[0] = FileBinding(bad[0].logical, bad[0].physical, "0" * 64, "source")
        with self.assertRaises(ValueError):
            admission.AdmissionBackend.from_sources(BoundIO(bad, lambda: None),
                provider_source=bad[0].logical, assembler_source=bad[1].logical, local_source=bad[2].logical)
        run = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import gpt_admission; assert not any(n in sys.modules for n in ('numpy','torch','transformers'))"],
            cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_scoped_success_is_immutable_and_compose_does_not_mutate(self):
        m, a = self.fixture()
        before = copy.deepcopy(a)
        result = self.run_admission(m, a)
        self.assertEqual(a, before)
        self.assertTrue(result.scoped)
        self.assertFalse(result.partial)
        self.assertEqual(set(result.acceptance), set(self.c["SCOPED_FAMILIES"]))
        result.manifest["numeric_transforms"].clear()
        self.assertTrue(result.manifest["numeric_transforms"])
        with self.assertRaises(TypeError):
            result.acceptance["local_base"] = "wrong"

    def test_manifest_policy_rejections(self):
        changes = [
            ("integration_mode", "full"), ("integration_mode", "partial"), ("state", "pending"),
            ("contract", "wrong"), ("scope_id", "wrong"), ("status", "full"),
            ("full_package_admitted", True), ("full_row_fit_admitted", 1),
            ("active_families", []), ("required_families", []), ("additional_ready_components", []),
            ("fit_cutoff", "2021-08-10"), ("validation_origin", "2021-08-10"),
            ("numeric_transforms", {}), ("rowkeys_algorithm", "wrong"), ("source_pins", []),
            ("entity_links", {}), ("runtime", {"audience": {"active": True}}),
            ("date_features", [{"family": "audience"}]), ("item_extensions", [{"changed": True}]),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                m, a = self.fixture()
                m[key] = value
                with self.assertRaises(ValueError):
                    self.run_admission(m, a)
        m, a = self.fixture()
        m["families"]["local_base"]["reviewed"] = False
        with self.assertRaises(ValueError):
            self.run_admission(m, a)
        m, a = self.fixture()
        with self.assertRaises(ValueError):
            self.run_admission(m, a, "D:/different/scope.json")

    def test_assessment_and_evidence_rejections(self):
        for change in (lambda a: a.update(contract="wrong"),
                       lambda a: a.update(blockers=["blocked"]),
                       lambda a: a.update(active_families=[]),
                       lambda a: a.update(provider_source={"sha256": "0" * 64}),
                       lambda a: a["components"]["local_base"].update(ready=False),
                       lambda a: a["components"].pop("local_base"),
                       lambda a: a.update(scope_decision={}),
                       lambda a: a.update(metadata={"sha256": "0" * 64})):
            m, a = self.fixture()
            change(a)
            m["assembly"]["assessment_sha256"] = canonical_hash(a)
            with self.assertRaises(ValueError):
                self.run_admission(m, a)
        m, a = self.fixture()
        m["assembly"]["assessment_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.run_admission(m, a)
        m, a = self.fixture()
        (self.root / "evidence.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_admission(m, a)

    def test_scope_reader_exact_hash_and_fields(self):
        value = {"contract": "kion-user-approved-ready-scope-v1", "scope_id": "ready-pack-20261020-r1",
                 "active_families": self.c["SCOPED_FAMILIES"],
                 "additional_ready_components": self.c["SCOPED_READY_COMPONENTS"]}
        pin = self.put("scope.json", value)
        # The isolated predicate accepts test constants; the public admission
        # API obtains only fixed, hash-verified constants from AdmissionBackend.
        c = dict(self.c, SCOPED_DECISION_SHA256=pin["sha256"])
        self.assertEqual(admission._scoped_decision(self.io(), ROOT, pin, c), value)
        with self.assertRaises(ValueError):
            admission._scoped_decision(self.io(), ROOT, pin, self.c)
        value["scope_id"] = "wrong"
        pin = self.put("scope.json", value)
        c["SCOPED_DECISION_SHA256"] = pin["sha256"]
        with self.assertRaises(ValueError):
            admission._scoped_decision(self.io(), ROOT, pin, c)

    def test_missing_scope_and_bad_signature(self):
        m, a = self.fixture()
        m["signature"] = "0" * 64
        self.put("manifest.json", m)
        with self.assertRaises(ValueError):
            admission.admit_scoped(self.io(), ROOT, backend=self.backend, requested_scope_decision=None)
        m.pop("signature")
        m["signature"] = canonical_hash(m)
        self.put("manifest.json", m)
        with self.assertRaises(ValueError):
            admission.admit_scoped(self.io(), ROOT, backend=self.backend, requested_scope_decision=None)


if __name__ == "__main__":
    unittest.main()
