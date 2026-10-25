"""Stdlib preparation tests. Never call retained train/run/evaluate functions."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import train_rl as rl


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.data = b'{"path":"D:/old-host/immutable.npy","teacher":{"path":"E:/old/model.pt"}}\n'
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.logical = 'D:/old-host/manifest.json'

    def file(self, name='manifest.json', data=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.data if data is None else data)
        return path

    def test_two_relocated_roots_preserve_bytes_and_provenance(self):
        for directory in ('clone-one', 'clone-two'):
            path = self.file(directory + '/manifest.json')
            bindings = rl.Bindings()
            bindings.add(self.logical, path, self.digest, 'metadata')
            self.assertEqual(bindings.read_json(self.logical), json.loads(self.data))
            self.assertEqual(bindings.resolve(self.logical), path.resolve())
            self.assertEqual(path.read_bytes(), self.data)
            with self.assertRaisesRegex(ValueError, 'Unknown logical'):
                bindings.resolve('D:/old-host/immutable.npy')

    def test_changed_source_or_metadata_pin_rejected(self):
        path = self.file()
        for expected in ('0' * 64, self.digest):
            if expected == self.digest:
                path.write_bytes(self.data + b' ')
            with self.assertRaisesRegex(ValueError, 'Pinned bytes changed'):
                rl.pinned_bytes(path, expected)

    def test_duplicate_logical_and_physical_bindings_rejected(self):
        path = self.file()
        bindings = rl.Bindings()
        bindings.add(self.logical, path, self.digest, 'metadata')
        with self.assertRaisesRegex(ValueError, 'Duplicate logical'):
            bindings.add(self.logical, path, self.digest, 'metadata')
        with self.assertRaisesRegex(ValueError, 'Duplicate physical'):
            bindings.add('D:/old/other.json', path, self.digest, 'metadata')

    def test_unknown_or_changed_parent_and_support_descriptors(self):
        bindings = rl.Bindings()
        bindings.add(self.logical, self.file(), self.digest, 'metadata')
        for descriptor in ({'path': self.logical, 'sha256': '0' * 64},
                           {'path': 'D:/old/unknown.json', 'sha256': self.digest}):
            with self.assertRaises(ValueError):
                bindings.descriptor(descriptor, self.logical)

    def test_traversal_and_absolute_relative_paths_rejected(self):
        for value in ('../other', '/other', 'D:/other', 'x/../../other', 'x\\other', 'x:stream'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rl.child(self.root, value)

    def test_missing_inputs_and_wrong_sizes_rejected(self):
        for name in ('screen/utility.json', 'A/head.pt', 'B/head.pt', 'bank.json'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Missing/empty'):
                rl.Bindings().add('D:/old/' + name, self.root / name, self.digest, 'input')
        with self.assertRaisesRegex(ValueError, 'Payload size mismatch'):
            rl.Bindings().add(self.logical, self.file(), self.digest, 'input', 1)

    def test_science_has_no_overrides(self):
        for key, value in (('rl_updates', 999), ('seeds', [20261024]),
                           ('utilities', {'completion': .5, 'residual': .1627403}),
                           ('precision', 'float32'), ('action_normalization', 32),
                           ('unknown', True)):
            modified = copy.deepcopy(rl.SCIENCE)
            modified[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Fixed science'):
                rl.validate_science(modified)

    def test_output_overlap_rejected_before_metadata_reads(self):
        artifacts = self.root / 'inputs'
        artifacts.mkdir()
        bank = self.file('bank/bank.json')
        with self.assertRaisesRegex(ValueError, 'Output/input overlap'):
            rl.plan(artifacts, bank, artifacts / 'output', root=self.root)

    def test_existing_output_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Fresh output'):
            rl.plan(self.root, self.file(), self.root, root=self.root)

    def test_plan_destination_cannot_occupy_output_or_inputs(self):
        future = self.root / 'future'
        source = self.file('source/module.py')
        metadata = self.file('inputs/cache.json')
        request = dict(output=future.as_posix(),
                       bindings=[dict(physical=metadata.as_posix())],
                       source_pins={source.as_posix(): self.digest},
                       beta_evidence=dict(path=metadata.as_posix()))
        for destination in (future, future / 'plan.json', self.root,
                            source, source.parent, metadata, metadata.parent):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, 'overlaps'):
                rl.plan_destination(destination, request)
        self.assertFalse(future.exists())
        self.assertEqual(rl.plan_destination(self.root / 'plan.json', request), (self.root / 'plan.json').resolve())

    def test_no_execution_switch(self):
        # Parsing rejects dispatch before inspecting any input.
        with self.assertRaises(SystemExit) as error:
            rl.main(['--artifact-root', 'x', '--bank', 'y', '--output', 'z', '--execute'])
        self.assertEqual(error.exception.code, 2)

    def test_module_imports_only_stdlib(self):
        tree = ast.parse(Path(rl.__file__).read_text(encoding='utf-8'))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module)
        self.assertEqual(names, {'__future__', 'argparse', 'hashlib', 'json', 'pathlib'})

    def test_symlink_escape_rejected(self):
        target = self.file()
        link = self.root / 'link.json'
        try:
            link.symlink_to(target)
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314:
                self.skipTest('Windows symlink privilege unavailable')
            raise
        with self.assertRaisesRegex(ValueError, 'Symlink/junction'):
            rl.safe_path(link)


if __name__ == '__main__':
    unittest.main()
