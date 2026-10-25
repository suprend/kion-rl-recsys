import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest

from gpt_provider import BoundIO, FileBinding, ORIGINAL_PROVIDER_SHA256
from gpt_sources import SourceRuntime, RUNTIME_PINS, pinned_runtime, runtime_source_bindings


class SourcesTests(unittest.TestCase):
    def runtime(self, bodies):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        bindings, pins = [], {}
        for name, body in bodies.items():
            path = Path(temp.name) / (name + '.py')
            source = body.encode()
            path.write_bytes(source)
            logical = 'D:/original/' + path.name
            digest = hashlib.sha256(source).hexdigest()
            pins[name] = (logical, digest)
            bindings.append(FileBinding(logical, path.as_posix(), digest, 'source'))
        return SourceRuntime(BoundIO(bindings, lambda: None), pins)

    def test_import_is_stdlib_only(self):
        code = ('import sys; sys.path.insert(0, ' + repr(str(Path(__file__).parent)) + '); '
                'import gpt_sources; assert not any(n in sys.modules for n in '
                '("numpy", "torch", "transformers", "scipy"))')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dependency_dataclass_identity_and_path_isolation(self):
        runtime = self.runtime({'kion_one': 'import sys\nsys.path.insert(0,"bad")\nimport kion_two\nvalue=kion_two.Value(3)\n',
                                'kion_two': 'from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass Value:\n x: int\n'})
        before = list(sys.path)
        with runtime:
            self.assertEqual(runtime.logical_source_name('kion_two'), 'D:/original/kion_two.py')
            self.assertEqual(runtime.modules, {})
            module = runtime.load('kion_one')
            self.assertEqual(module.value.x, 3)
            self.assertEqual(sys.path, before)
            self.assertEqual(runtime.logical_source(module), 'D:/original/kion_one.py')
            self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), module.__source_sha256__)
            self.assertEqual(runtime.receipt()['loaded'], ['kion_one', 'kion_two'])
        self.assertNotIn('kion_one', sys.modules)
        self.assertNotIn('kion_two', sys.modules)
        with self.assertRaises(RuntimeError):
            runtime.load('kion_one')

    def test_all_sources_verified_before_any_execution(self):
        runtime = self.runtime({'kion_one': 'raise AssertionError("executed")\n', 'kion_two': 'x=2\n'})
        Path(runtime.io.bindings['D:/original/kion_two.py'].physical).write_bytes(b'x=9\n')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            with runtime:
                runtime.load('kion_one')
        self.assertNotIn('kion_one', sys.modules)

    def test_missing_closure(self):
        with self.assertRaisesRegex(ValueError, 'closure'):
            with self.runtime({'kion_one': 'import kion_missing\n'}):
                pass

    def test_collision_preserves_existing_module(self):
        old = ModuleType('kion_one')
        sys.modules['kion_one'] = old
        try:
            with self.assertRaisesRegex(RuntimeError, 'collision'):
                with self.runtime({'kion_one': 'x=1\n'}):
                    pass
            self.assertIs(sys.modules['kion_one'], old)
        finally:
            sys.modules.pop('kion_one')

    def test_failure_cleanup_and_no_unknown_dynamic_import(self):
        for body in ['__import__("kion_unknown")', 'import numpy', 'import importlib.util',
                     'import importlib\nimportlib.import_module("kion_unknown")',
                     'from importlib import util', 'raise ValueError("failed")']:
            with self.assertRaises((ImportError, ValueError)):
                with self.runtime({'kion_one': body}) as runtime:
                    runtime.load('kion_one')
            self.assertNotIn('kion_one', sys.modules)

    def test_nested_runtime_rejected_and_lock_released(self):
        outer = self.runtime({'kion_one': 'x=1'})
        inner = self.runtime({'kion_two': 'x=2'})
        with outer:
            with self.assertRaises(RuntimeError):
                with inner:
                    pass
        with inner:
            self.assertEqual(inner.load('kion_two').x, 2)

    def test_production_pins_and_missing_bindings_fail_closed(self):
        self.assertEqual(RUNTIME_PINS['kion_complete_pack'], ORIGINAL_PROVIDER_SHA256)
        self.assertEqual(len(RUNTIME_PINS), 41)
        tiny = self.runtime({'kion_one': 'x=1'})
        with self.assertRaises(KeyError):
            with pinned_runtime(tiny.io):
                pass

    def test_metadata_import_syntax_without_dynamic_loader(self):
        source = ('import importlib.metadata\n'
                  'from importlib import metadata\n'
                  'from importlib.metadata import PackageNotFoundError\n'
                  'assert importlib.metadata is metadata\n'
                  'assert PackageNotFoundError is metadata.PackageNotFoundError\n'
                  'assert not hasattr(importlib, "util")\n')
        with self.runtime({'kion_one': source}) as runtime:
            runtime.load('kion_one')

    def test_explicit_repository_binding_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = runtime_source_bindings(temp)
            self.assertEqual(len(rows), 41)
            self.assertTrue(all(r.physical.startswith(Path(temp).as_posix() + '/') for r in rows))
            self.assertEqual(runtime_source_bindings(temp, existing=rows), rows)
            bad = FileBinding(rows[0].logical, rows[0].physical + '.other', rows[0].sha256, 'source')
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                runtime_source_bindings(temp, existing=[bad])

    def test_dynamic_import_only_exact_admitted_name(self):
        source = ('import importlib\nfrom importlib import import_module\n'
                  'value=importlib.import_module("kion_two").value\n'
                  'assert import_module("kion_two").value == value\n')
        with self.runtime({'kion_one': source, 'kion_two': 'value=12'}) as runtime:
            self.assertEqual(runtime.load('kion_one').value, 12)
            for name, package in [('numpy', None), ('kion_unknown', None),
                                  ('.kion_two', 'scripts'), ('kion_two', 'scripts')]:
                with self.assertRaises(ImportError):
                    runtime._import_module(name, package)


if __name__ == '__main__':
    unittest.main()
