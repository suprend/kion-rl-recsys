import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpt_inputs as g
import gpt_transitive as t
import transfer_inputs


class TransitiveTests(unittest.TestCase):
    def test_derived_assessment_identity_and_canonical_parent_pin(self):
        with tempfile.TemporaryDirectory() as a:
            root = Path(a)
            p,b,raw = self.fixture(root)
            d = g.json_bytes(raw)
            d['documents'][0]['id'] = 'provider'
            d['entries'][0]['document'] = 'provider'
            value = {'contract': 'kion-complete-scoped-assessment-v1', 'components': {}}
            assessment = json.dumps(value).encode()
            source = b'# exact pinned consumer fixture\n'
            source_sha = g.digest(source)
            (root/'provider.py').write_bytes(source)
            (root/'assessment.json').write_bytes(assessment)
            parent = g.json_bytes((root/'root.json').read_bytes())
            parent.update(assembly={'assessment_sha256': t.canonical_hash(value)},
                          consumer={'path': 'D:/old/provider.py', 'sha256': source_sha})
            parent_bytes = json.dumps(parent).encode()
            (root/'root.json').write_bytes(parent_bytes)
            d['documents'][0]['sha256'] = g.digest(parent_bytes)
            d['entries'].append(dict(id='source', document='provider', selector=['consumer','sha256'],
                                    logical='D:/old/provider.py', sha256=source_sha, kind='source'))
            raw = json.dumps(d).encode(); p['direct_profile_sha256'] = g.digest(raw)
            p['documents'].append(dict(id='assessment', logical='D:/old/assessment.json',
                sha256=g.digest(assessment), contract=value['contract'], keys=sorted(value),
                signature_exclude=None, limit=1024, derived_from=dict(document='provider',
                selector=['assembly','assessment_sha256'], consumer_sha256=source_sha)))
            b.update({'D:/old/provider.py':(root/'provider.py').as_posix(),
                      'D:/old/assessment.json':(root/'assessment.json').as_posix()})
            with patch.object(t, 'PROVIDER_SOURCE_SHA', source_sha):
                report = t.inspect(p,b,raw)
                self.assertEqual(report['counts']['verified_metadata'],3)
                p['documents'][-1]['logical'] = 'D:/different/assessment.json'
                with self.assertRaisesRegex(ValueError,'derived metadata identity'): t.inspect(p,b,raw)
                p['documents'][-1]['logical'] = 'D:/old/assessment.json'
                changed = json.dumps(dict(value, components={'unexpected':1})).encode()
                (root/'assessment.json').write_bytes(changed)
                p['documents'][-1]['sha256'] = g.digest(changed)
                with self.assertRaisesRegex(ValueError,'canonical parent pin'): t.inspect(p,b,raw)
                for malformed in (None, 'not-a-sha256'):
                    parent['assembly']['assessment_sha256'] = malformed
                    parent_bytes = json.dumps(parent).encode()
                    (root/'root.json').write_bytes(parent_bytes)
                    d['documents'][0]['sha256'] = g.digest(parent_bytes)
                    raw = json.dumps(d).encode(); p['direct_profile_sha256'] = g.digest(raw)
                    with self.assertRaisesRegex(ValueError,'invalid SHA256'): t.inspect(p,b,raw)

    def test_transfer_adapter_integration_without_copy_or_payload_open(self):
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a))
            report = t.inspect(p,b,d)
            with patch.object(Path, 'open', side_effect=AssertionError('no content reads for transfer plan')):
                converted = transfer_inputs.convert(report, 'gpt-report', b)
                planned = transfer_inputs.plan(converted)
            self.assertEqual(planned['file_count'], 3)
            self.assertFalse(planned['payloads_hashed'])
            self.assertEqual({r['logical']:r['kind'] for r in report['transfer_recipe']},
                             {'D:/old/manifest.json':'metadata', 'D:/old/cache/manifest.json':'metadata', 'D:/old/cache/x.npy':'payload'})

    def test_transfer_rejects_conflicting_kinds(self):
        a = dict(logical='D:/old/x', sha256='a'*64, kind='source')
        with self.assertRaisesRegex(ValueError, 'conflicting transfer kinds'):
            t.transfer_recipe([a, dict(a, kind='payload')])

    def fixture(self, root):
        child = {'contract': 'kion-text-embeddings-v1', 'outputs': {'x.npy': {'sha256': 'a'*64}}}
        data = json.dumps(child).encode()
        (root/'child.json').write_bytes(data)
        original = {'pin': {'path': 'D:/old/cache/manifest.json', 'sha256': g.digest(data)}}
        data_root = json.dumps(original).encode()
        (root/'root.json').write_bytes(data_root)
        (root/'payload.npy').write_bytes(b'not read')
        d = {'schema_version': 1, 'documents': [{'id': 'root', 'logical': 'D:/old/manifest.json', 'sha256': g.digest(data_root), 'signature': False}],
             'entries': [{'id': 'child', 'document': 'root', 'selector': ['pin', 'sha256'], 'logical': original['pin']['path'], 'sha256': original['pin']['sha256'], 'kind': 'payload'}], 'blocked_stages': ['training']}
        direct_bytes = json.dumps(d).encode('utf-8')
        p = {'schema_version': 1, 'direct_profile_sha256': g.digest(direct_bytes),
             'documents': [{'id': 'child', 'logical': original['pin']['path'], 'sha256': original['pin']['sha256'], 'contract': child['contract'], 'keys': sorted(child), 'signature_exclude': None, 'limit': 1024}],
             'entries': [{'document': 'child', 'selector': ['outputs', 'x.npy', 'sha256'], 'path_selector': None, 'implicit_name': 'x.npy', 'original_path': 'x.npy', 'base': 'D:/old/cache', 'logical': 'D:/old/cache/x.npy', 'sha256': 'a'*64}], 'unexpanded': ['opaque other assets']}
        b = {'D:/old/manifest.json': (root/'root.json').as_posix(), 'D:/old/cache/manifest.json': (root/'child.json').as_posix(), 'D:/old/cache/x.npy': (root/'payload.npy').as_posix()}
        return p,b,direct_bytes

    def test_original_direct_profile_bytes_required_and_authenticated_before_io(self):
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a))
            with patch.object(Path, 'open', side_effect=AssertionError('premature I/O')):
                with self.assertRaisesRegex(ValueError, 'original direct-profile bytes'):
                    t.inspect(p,b,g.json_bytes(d))
                with self.assertRaisesRegex(ValueError, 'direct profile pin mismatch'):
                    t.inspect(p,b,d+b' ')
                p['direct_profile_sha256'] = 'b'*64
                with self.assertRaisesRegex(ValueError, 'direct profile pin mismatch'):
                    t.inspect(p,b,d)

    def test_relocation_guard_and_immutability(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            p,x,d = self.fixture(Path(a)); q,y,e = self.fixture(Path(b))
            before = copy.deepcopy((p,x,d))
            original = Path.open
            def safe(path, *args, **kwargs):
                self.assertNotEqual(path.suffix, '.npy', 'payload read forbidden')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', safe):
                r = t.inspect(p,x,d); s = t.inspect(q,y,e)
            self.assertEqual(r['transfer_recipe'], s['transfer_recipe'])
            self.assertEqual(before, (p,x,d))
            self.assertFalse(r['ready']); self.assertFalse(r['closure_complete'])
            self.assertFalse(r['payloads_hashed'])

    def test_tampered_child_bytes(self):
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a)); Path(b['D:/old/cache/manifest.json']).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): t.inspect(p,b,d)

    def test_selector_and_path_tamper(self):
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a)); p['entries'][0]['sha256'] = 'c'*64
            with self.assertRaisesRegex(ValueError, 'pin differs'): t.inspect(p,b,d)
            p['entries'][0]['sha256'] = 'a'*64; p['entries'][0]['base'] = 'D:/different'
            with self.assertRaisesRegex(ValueError, 'identity differs'): t.inspect(p,b,d)

    def test_unknown_schema_or_payload_read_request(self):
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a)); p['documents'][0]['contract'] = 'unknown'
            with self.assertRaisesRegex(ValueError, 'allowlisted'): t.inspect(p,b,d)
            p['documents'][0]['contract'] = 'kion-text-embeddings-v1'; p['documents'][0]['logical'] = 'D:/old/payload.npy'
            with self.assertRaisesRegex(ValueError, 'allowlisted'): t.inspect(p,b,d)

    def test_missing_parent_child_payload(self):
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a))
            self.assertIn('blocked_parent',t.inspect(p,{},d)['counts'])
            b.pop('D:/old/cache/manifest.json')
            self.assertEqual(t.inspect(p,b,d)['entries'][-1]['status'],'blocked_metadata')
            p,b,d = self.fixture(Path(a)); Path(b['D:/old/cache/x.npy']).unlink()
            self.assertEqual(t.inspect(p,b,d)['entries'][-1]['status'],'blocked_missing')

    def test_escaping_ambiguous_and_size(self):
        with self.assertRaises(ValueError): t.logical_path('../escape','D:/old')
        with tempfile.TemporaryDirectory() as a:
            p,b,d = self.fixture(Path(a)); b['D:/old/cache/x.npy'] = b['D:/old/cache/manifest.json']
            with self.assertRaisesRegex(ValueError,'ambiguous'): t.inspect(p,b,d)
            p,b,d = self.fixture(Path(a)); p['documents'][0]['limit'] = 4
            with self.assertRaisesRegex(ValueError,'bounded'): t.inspect(p,b,d)


if __name__ == '__main__':
    unittest.main()
