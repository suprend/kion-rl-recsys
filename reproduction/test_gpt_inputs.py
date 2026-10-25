import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpt_inputs as g


class InputsTests(unittest.TestCase):
    def fixture(self, root):
        payload = root / 'payload.pt'
        payload.write_bytes(b'not a model')
        logical = 'D:/historical/model.pt'
        value = {'pin': {'path': logical, 'sha256': 'a' * 64}}
        data = json.dumps(value).encode()
        metadata = root / 'metadata.json'
        metadata.write_bytes(data)
        profile = {'schema_version': 1, 'documents': [
            {'id': 'fixture', 'logical': 'D:/historical/manifest.json',
             'sha256': hashlib.sha256(data).hexdigest(), 'signature': False}],
            'entries': [{'id': 'payload', 'document': 'fixture', 'selector': ['pin', 'sha256'],
                         'logical': logical, 'sha256': 'a' * 64, 'kind': 'payload'}],
            'blocked_stages': ['fixture is incomplete']}
        bindings = {'D:/historical/manifest.json': metadata.as_posix(), logical: payload.as_posix()}
        return profile, bindings

    def test_relocated_roots_same_identity_no_payload_read(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            p, x = self.fixture(Path(a))
            q, y = self.fixture(Path(b))
            original = copy.deepcopy(p)
            read = g.read_small
            def safe(path, limit=g.LIMIT):
                self.assertNotEqual(Path(path).suffix, '.pt')
                return read(path, limit)
            with patch.object(g, 'read_small', side_effect=safe):
                r, s = g.inspect(p, x), g.inspect(q, y)
            self.assertEqual(r['transfer_recipe'], s['transfer_recipe'])
            self.assertFalse(r['ready'])
            self.assertEqual(p, original)

    def test_missing_and_unbound(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            Path(b['D:/historical/model.pt']).unlink()
            self.assertEqual(g.inspect(p, b)['entries'][-1]['status'], 'blocked_missing')
            self.assertEqual(g.inspect(p, {})['entries'][-1]['status'], 'blocked_metadata')

    def test_metadata_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            Path(b['D:/historical/manifest.json']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): g.inspect(p, b)

    def test_unknown_duplicate_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'undeclared'): g.inspect(p, {**b, 'D:/other': str(Path(tmp)/'other')})
            p['entries'].append(copy.deepcopy(p['entries'][0]))
            with self.assertRaisesRegex(ValueError, 'duplicate'): g.inspect(p, b)
            p['entries'].pop()
            b['D:/historical/model.pt'] = b['D:/historical/manifest.json']
            with self.assertRaisesRegex(ValueError, 'ambiguous'): g.inspect(p, b)

    def test_binding_not_identity_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            p['entries'][0]['logical'] = 'D:/different/model.pt'
            with self.assertRaisesRegex(ValueError, 'logical identity'): g.inspect(p, b)

    def test_pin_and_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            p['entries'][0]['sha256'] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'differs'): g.inspect(p, b)
            p['entries'][0]['sha256'] = 'a' * 64
            p['entries'][0]['kind'] = 'source'
            self.assertEqual(g.inspect(p, b)['entries'][-1]['status'], 'blocked_source_hash')

    def test_json_and_size_limits(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'): g.json_bytes(b'{"x":1,"x":2}')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'large.json'
            path.write_bytes(b'12345')
            with self.assertRaisesRegex(ValueError, 'bounded'): g.read_small(path, 4)

    def test_escape_and_relative_bindings(self):
        for value in ('D:/root/../other', 'D:/root/a\\b', 'relative/path'):
            with self.assertRaises(ValueError): g.identity(value)
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            b['D:/historical/model.pt'] = 'relative.pt'
            with self.assertRaises(ValueError): g.inspect(p, b)

    def test_signature_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp))
            p['documents'][0]['signature'] = True
            with self.assertRaisesRegex(ValueError, 'signature'): g.inspect(p, b)
            p['documents'][0]['signature'] = False
            profile, bindings, output = [Path(tmp)/n for n in ('profile.json', 'bindings.json', 'report.json')]
            profile.write_text(json.dumps(p)); bindings.write_text(json.dumps(b)); output.write_text('keep')
            self.assertEqual(g.main(['--profile', str(profile), '--bindings', str(bindings), '--output', str(output)]), 2)
            self.assertEqual(output.read_text(), 'keep')

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, b = self.fixture(Path(tmp)); link = Path(tmp)/'link'
            try: link.symlink_to(b['D:/historical/model.pt'])
            except OSError as exc:
                if getattr(exc, 'winerror', None) == 1314: self.skipTest('Windows symlink privilege unavailable')
                raise
            b['D:/historical/model.pt'] = str(link)
            with self.assertRaises(ValueError): g.inspect(p, b)


if __name__ == '__main__':
    unittest.main()
