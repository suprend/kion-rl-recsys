"""Stdlib source tests; --ml is an explicit root-owned tiny CPU fixture."""
import ast
import importlib.util
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot_heads", HERE / "heads.py")
heads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heads)


def metadata_fixture(directory):
    directory = Path(directory)
    bank_dir = directory / "bank"
    bank_dir.mkdir()
    teacher = {"path": "synthetic.pt", "sha256": "a" * 64,
               "config_path": "synthetic.json", "config_sha256": "b" * 64}
    lineage = {"teacher": teacher, "feature_schema_sha256": heads.FEATURE_SCHEMA_SHA256,
               "source_columns": list(heads.SOURCE_COLUMNS)}
    bank = dict(lineage, contract=heads.BANK_CONTRACT, state="completed", split="train", train_rows=heads.TRAIN_ROWS,
                natural_covered_only=False, same_day_positives_excluded=True)
    heads.atomic_json(bank_dir / "bank.json", bank)
    request = dict(lineage, bank_dir=bank_dir.as_posix(), output_dir=directory.as_posix(),
                   bank_manifest_sha256=heads.sha256(bank_dir / "bank.json"), epochs=3)
    receipt = dict(lineage, contract=heads.CONTRACT, status="completed", epochs_requested=3,
                   epochs_completed=3, steps_per_head=6144, head_steps={name: 6144 for name in heads.HEAD_NAMES},
                   rows_per_epoch=heads.TRAIN_ROWS, batch_size=128, seed=20261024, common_batches=True,
                   shuffle=heads.SHUFFLE, raw_gather=heads.RAW_GATHER,
                   epoch_records=[heads.epoch_record(epoch) for epoch in range(3)],
                   trained_from="fresh-common-init-final-zero", bank_manifest_sha256=request["bank_manifest_sha256"])
    return request, receipt


class SourceTests(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        allowed = {"__future__", "contextlib", "datetime", "hashlib", "json", "pathlib", "random", "time"}
        tree = ast.parse((HERE / "heads.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name.split(".")[0] in allowed for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module.split(".")[0], allowed)

    def test_masks_exact(self):
        self.assertEqual(heads.mask_indices("A"), ())
        self.assertEqual(heads.mask_indices("B"), tuple(range(142, 536)) + tuple(range(907, 1068)))
        self.assertEqual(len(set(heads.mask_indices("B"))), 555)
        self.assertEqual(heads.mask_indices("C"), tuple(range(1068)))
        with self.assertRaises(ValueError):
            heads.mask_indices("D")

    def test_fixed_request(self):
        request = {"bank_dir": "cache/train", "output_dir": "heads"}
        heads.validate_request(request)
        heads.validate_request(dict(request, epochs=3))
        for key, bad in (("epochs", 1), ("epochs", 2), ("epochs", 0), ("epochs", 3.0), ("seed", 42), ("batch_size", 64)):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                heads.validate_request(dict(request, **{key: bad}))

    def test_physical_schema(self):
        self.assertEqual(heads.ARRAY_SPEC["raw_features"], ("float32", (262144, 32, 1068)))
        self.assertEqual(heads.ARRAY_SPEC["source_values"], ("float32", (262144, 32, 2)))
        self.assertEqual(heads.STEPS_PER_EPOCH, 2048)
        self.assertEqual(heads.EPOCHS * heads.STEPS_PER_EPOCH, 6144)
        self.assertEqual(heads.CONFIG["mlp_input_width"], 800)
        self.assertEqual(heads.CONFIG["projection_bias"], False)

    def test_json_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            heads.atomic_json(path, {"name": "голова"})
            self.assertEqual(heads._read_json(path), {"name": "голова"})
            self.assertEqual(len(heads.sha256(path)), 64)
            self.assertFalse(path.with_name("receipt.json.tmp").exists())
            self.assertNotIn(b"\r", path.read_bytes())

    def test_run_has_only_lazy_numeric_imports(self):
        tree = ast.parse((HERE / "heads.py").read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        imports = {alias.name for node in ast.walk(run) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {"numpy", "torch"})
        text = (HERE / "heads.py").read_text(encoding="utf-8")
        self.assertNotIn("subprocess", text)
        self.assertNotIn("candidate_features", text)

    def test_preflight_without_numeric_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            request, receipt = metadata_fixture(directory)
            for name in heads.HEAD_NAMES:
                path = Path(directory) / name / "head.pt"
                path.parent.mkdir()
                path.write_bytes(b"synthetic checkpoint hash fixture " + name.encode())
            receipt["checkpoint_sha256"] = {name: heads.sha256(Path(directory) / name / "head.pt")
                                             for name in heads.HEAD_NAMES}
            heads.atomic_json(Path(directory) / "training.json", receipt)
            self.assertEqual(heads.preflight_heads(request), receipt)
            for key, value in (("epochs", 1), ("teacher", {}), ("source_columns", [12, 13]),
                               ("feature_schema_sha256", "c" * 64), ("bank_manifest_sha256", "d" * 64)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    heads.preflight_heads(dict(request, **{key: value}))
            (Path(directory) / "C" / "head.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                heads.preflight_heads(request)

    def test_preflight_rejects_unequal_steps_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            request, receipt = metadata_fixture(directory)
            receipt["head_steps"]["B"] = 6143
            heads.atomic_json(Path(directory) / "training.json", receipt)
            with self.assertRaisesRegex(ValueError, "budget"):
                heads.preflight_heads(request)

    def test_schedule_full_coverage_deterministic_and_exact_hash(self):
        hashes = []
        for epoch in range(3):
            order = heads.epoch_schedule(epoch)
            expected = list(range(262144))
            random.Random(20261024 + epoch).shuffle(expected)
            self.assertEqual(order, expected)
            self.assertEqual(order, heads.epoch_schedule(epoch))
            self.assertEqual(sorted(order), list(range(262144)))
            self.assertEqual(len(order) // 128, 2048)
            # Original global shuffle mixes fixed date-sorted physical blocks
            # within each batch instead of 64 date-local consecutive updates.
            self.assertTrue(all(len({row // 8192 for row in order[start:start + 128]}) > 16
                                for start in range(0, 262144, 128)))
            digest = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
            record = heads.epoch_record(epoch)
            self.assertEqual(record["shuffle_sha256"], digest)
            self.assertEqual(heads.epoch_record(epoch, order), record)
            hashes.append(digest)
        self.assertEqual(len(set(hashes)), 3)
        for invalid in (-1, 3, 1.0, True):
            with self.subTest(epoch=invalid), self.assertRaises(ValueError):
                list(heads.epoch_schedule(invalid))

    def test_preflight_rejects_changed_epoch_lineage_before_decode(self):
        for key, bad in (("shuffle_sha256", "a" * 64), ("rows", 32768),
                         ("updates_per_head", 256), ("epoch", 2)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                request, receipt = metadata_fixture(directory)
                receipt["epoch_records"][0][key] = bad
                heads.atomic_json(Path(directory) / "training.json", receipt)
                with self.assertRaisesRegex(ValueError, "schedule"):
                    heads.preflight_heads(request)

    def test_batch_copy_sorts_io_restores_order_and_closes_mapping(self):
        # Fake list/mmap backend: no NumPy, tensors, real arrays or checkpoints.
        events = []
        class FakeGather:
            def __init__(self, rows):
                self.rows = list(rows)
            def __getitem__(self, indices):
                self_test.assertEqual(events[-1], "close")
                events.append("inverse")
                return [self.rows[index] for index in indices]
        self_test = self
        class FakeRaw:
            _mmap = SimpleNamespace(close=lambda: events.append("close"))
            def __getitem__(self, rows):
                self_test.assertEqual(rows, sorted(rows))
                self_test.assertEqual(len(rows), 128)
                events.append(("gather", rows))
                return FakeGather(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npy"
            path.write_bytes(b"fake identity")
            stat = path.stat()
            identity = {"path": path.as_posix(), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            def load(*args, **kwargs):
                events.append("reopen")
                self.assertEqual(kwargs, {"mmap_mode": "r", "allow_pickle": False})
                return FakeRaw()
            np = SimpleNamespace(load=load)
            arrays = {"raw_features": FakeRaw()}
            for start in (0, 128):
                positions = heads.epoch_schedule(0)[start:start + 128]
                batch = heads._copy_raw_batch(arrays, {"raw_features": identity}, positions, np)
                self.assertEqual(batch, positions)
                self.assertIsNone(arrays["raw_features"])
            self.assertEqual(events.count("close"), 2)
            self.assertEqual(events.count("reopen"), 1)
            path.write_bytes(b"changed fake identity")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                heads._copy_raw_batch(arrays, {"raw_features": identity}, positions, np)
            with self.assertRaisesRegex(ValueError, "B128"):
                heads._copy_raw_batch(arrays, {"raw_features": identity}, positions[:64], np)

    def test_bank_hash_once_then_stat_bound_receipt_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            request, _ = metadata_fixture(directory)
            root = Path(request["bank_dir"])
            manifest = heads._read_json(root / "bank.json")
            entries = {}
            for name, (dtype, shape) in heads.ARRAY_SPEC.items():
                path = root / (name + ".npy")
                path.write_bytes(("fake header identity " + name).encode())
                entries[name] = {"path": path.name, "dtype": dtype,
                    "shape": [4 if size is None else size for size in shape],
                    "bytes": path.stat().st_size, "sha256": heads.sha256(path)}
            manifest["arrays"] = entries
            heads.atomic_json(root / "bank.json", manifest)
            request["bank_manifest_sha256"] = heads.sha256(root / "bank.json")
            def fake_load(path, **kwargs):
                self.assertEqual(kwargs, {"mmap_mode": "r", "allow_pickle": False})
                entry = entries[Path(path).stem]
                return SimpleNamespace(dtype=entry["dtype"], shape=tuple(entry["shape"]))
            np = SimpleNamespace(load=fake_load)
            callbacks = []
            def check(**fields):
                callbacks.append(fields)
            _, receipt, _ = heads._bank(request, np, check)
            self.assertEqual(len(callbacks), len(heads.ARRAY_SPEC))
            receipt_path = root / "verification.json"
            heads.atomic_json(receipt_path, receipt)
            request["bank_verification_receipt"] = receipt_path.as_posix()
            callbacks.clear()
            original_hash = heads.sha256
            with patch.object(heads, "sha256", wraps=original_hash) as hashing:
                _, reused, _ = heads._bank(request, np, check)
                self.assertEqual([Path(call.args[0]).name for call in hashing.call_args_list], ["bank.json"])
            self.assertEqual(receipt, reused)
            self.assertEqual(callbacks, [])
            path = root / "encoded.npy"
            path.write_bytes(b"corrupted-file")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                heads._bank(request, np, check)

    def test_save_captures_cuda_rng_only_for_cuda_models(self):
        # A fake backend makes unintended CUDA access fail without importing Torch.
        for is_cuda in (False, True):
            with self.subTest(is_cuda=is_cuda), tempfile.TemporaryDirectory() as directory:
                calls = []
                captured = []
                def cuda_rng():
                    calls.append("cuda-rng")
                    if not is_cuda:
                        self.fail("CPU save attempted CUDA initialization")
                    return ["fake-cuda-rng"]
                def save(payload, path):
                    captured.append(payload)
                    Path(path).write_bytes(b"fake-checkpoint")
                torch = SimpleNamespace(cuda=SimpleNamespace(get_rng_state_all=cuda_rng),
                                        get_rng_state=lambda: "fake-cpu-rng", save=save)
                models = {name: SimpleNamespace(parameters=lambda: iter([SimpleNamespace(is_cuda=is_cuda)]),
                                                state_dict=lambda: {}) for name in heads.HEAD_NAMES}
                states = {name: SimpleNamespace(state_dict=lambda: {}) for name in heads.HEAD_NAMES}
                heads._save(models, states, states, Path(directory), {}, torch, True)
                self.assertEqual(calls, ["cuda-rng"] if is_cuda else [])
                self.assertEqual(len(captured), 3)
                self.assertTrue(all(payload["cuda_rng_state"] == (["fake-cuda-rng"] if is_cuda else [])
                                    for payload in captured))


def tiny_cpu_fixture():
    """Root calls explicitly; no actual artifacts/data/GPU, tiny seeded tensors."""
    import torch
    torch.set_num_threads(1)
    models = heads.make_heads(torch, seed=20261024, device="cpu")
    astate = models["A"].state_dict()
    for name in ("B", "C"):
        for key, value in astate.items():
            assert torch.equal(value, models[name].state_dict()[key]), (name, key)
            assert value.data_ptr() != models[name].state_dict()[key].data_ptr(), (name, key)
    assert all(not isinstance(layer, torch.nn.Dropout) for model in models.values() for layer in model.modules())
    generator = torch.Generator().manual_seed(89)
    q = torch.randn(2, 256, generator=generator).requires_grad_()
    table = torch.randn(67, 256, generator=generator).requires_grad_()
    ids = torch.tensor([[3, 5, 7, 8], [4, 6, 9, 10]])
    base = torch.randn(2, 4, generator=generator).requires_grad_()
    raw = torch.randn(2, 4, 1068, generator=generator).requires_grad_()
    source = torch.randn(2, 4, 2, generator=generator).requires_grad_()
    values = heads.score_heads(models, q, table, ids, base, raw, source, torch)
    assert all(torch.equal(score, base.detach()) for score in values.values())
    half_values = heads.score_heads(models, q, table, ids, base.detach().half(), raw, source, torch)
    assert all(score.dtype == torch.float32 and torch.equal(score, base.detach().half().float())
               for score in half_values.values())
    # Deliberately open residual output for sensitivity and frozen-gradient checks.
    with torch.no_grad():
        models["A"].pair_head[2].weight.fill_(0.02)
        for name in ("B", "C"):
            models[name].load_state_dict(models["A"].state_dict())
    values = heads.score_heads(models, q, table, ids, base, raw, source, torch)
    changed = raw.detach().clone()
    changed[..., 0] += 5
    changed_values = heads.score_heads(models, q, table, ids, base, changed, source, torch)
    assert torch.equal(values["A"], changed_values["A"])
    assert torch.equal(values["B"], changed_values["B"])
    assert not torch.equal(values["C"], changed_values["C"])
    changed_b = raw.detach().clone()
    changed_b[..., 142] += 5
    b_values = heads.score_heads(models, q, table, ids, base, changed_b, source, torch)
    assert torch.equal(values["A"], b_values["A"])
    assert not torch.equal(values["B"], b_values["B"])
    shifted_source = source.detach() + 3
    shifted_values = heads.score_heads(models, q, table, ids, base, raw, shifted_source, torch)
    assert all(not torch.equal(values[name], shifted_values[name]) for name in heads.HEAD_NAMES)
    # Candidate permutation/chunk invariance checks on every final scorer.
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = heads.score_heads(models, q, table, ids[:, permutation], base[:, permutation],
                                 raw[:, permutation], source[:, permutation], torch)
    for name in heads.HEAD_NAMES:
        assert torch.allclose(values[name][:, permutation], permuted[name], atol=1e-6, rtol=1e-6)
        parts = [heads.score_heads(models, q, table, ids[:, start:start+2], base[:, start:start+2],
                                  raw[:, start:start+2], source[:, start:start+2], torch)[name]
                 for start in (0, 2)]
        assert torch.allclose(values[name], torch.cat(parts, dim=1), atol=1e-6, rtol=1e-6)
    sum(value.sum() for value in values.values()).backward()
    assert all(value.grad is None for value in (q, table, base, raw, source))
    assert all(model.pair_projection.weight.grad is not None for model in models.values())
    # One CPU CE32 update, independent optimizer; other heads remain bitwise unchanged.
    fresh = heads.make_heads(torch, device="cpu")
    frozen_b = {key: value.clone() for key, value in fresh["B"].state_dict().items()}
    optimizer = torch.optim.AdamW(fresh["A"].parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ids32 = torch.arange(3, 67).reshape(2, 32)
    q32 = q.detach()
    raw32 = torch.randn(2, 32, 1068, generator=generator)
    source32 = torch.randn(2, 32, 2, generator=generator)
    base32 = torch.randn(2, 32, generator=generator)
    record = heads._update(fresh["A"], optimizer, scaler,
                           (q32, table.detach()[ids32], base32, raw32, source32), torch, torch.device("cpu"))
    assert record["attempts"] == 1 and record["precision"] == "fp32"
    assert all(torch.equal(value, fresh["B"].state_dict()[key]) for key, value in frozen_b.items())
    assert fresh["A"].pair_head[2].weight.abs().sum() > 0
    # Full state serialization/load parity without invoking train-bank construction.
    with tempfile.TemporaryDirectory() as directory:
        request, receipt = metadata_fixture(directory)
        optimizers = {name: torch.optim.AdamW(fresh[name].parameters()) for name in heads.HEAD_NAMES}
        scalers = {name: torch.amp.GradScaler("cuda", enabled=False) for name in heads.HEAD_NAMES}
        heads._save(fresh, optimizers, scalers, Path(directory), receipt, torch, True)
        published = dict(receipt, checkpoint_sha256={name: heads.sha256(Path(directory) / name / "head.pt")
                                                   for name in heads.HEAD_NAMES})
        heads.atomic_json(Path(directory) / "training.json", published)
        loaded = heads.load_heads(request, torch, "cpu")
        for name in heads.HEAD_NAMES:
            for key, value in fresh[name].state_dict().items():
                assert torch.equal(value, loaded[name].state_dict()[key])
    return {"status": "passed", "fixture": "synthetic-cpu-only", "head_count": 3,
            "checks": ["common-independent-init", "zero-output", "raw-masks", "source2-sensitivity",
                       "permutation", "chunking", "frozen-gradients", "independent-CE32-update", "save-load"]}


if __name__ == "__main__":
    if "--ml" in sys.argv:
        import json
        print(json.dumps(tiny_cpu_fixture()))
    else:
        unittest.main()
