"""Small temporary-file tests. Never reads historical payloads."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import transfer_inputs as transfer


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'tiny fixture')
        self.row = dict(logical='D:/original/cache/source.bin', physical=self.source.as_posix(),
                        destination='inputs/D/original/cache/source.bin', bytes=12,
                        sha256=hashlib.sha256(b'tiny fixture').hexdigest())
        self.manifest = transfer.envelope([self.row])

    def test_plan_never_reads_payload_or_creates_files(self):
        before = set(self.root.iterdir())
        with patch.object(Path, 'open', side_effect=AssertionError('payload read')):
            result = transfer.plan(self.manifest)
        self.assertFalse(result['payloads_hashed'])
        self.assertFalse(result['closure_complete'])
        self.assertEqual(result['total_bytes'], 12)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_copy_and_receiver_manifest(self):
        output = self.root / 'bundle'
        receipt = transfer.copy_inputs(transfer.plan(self.manifest), output)
        self.assertEqual(receipt['state'], 'completed')
        self.assertFalse(receipt['closure_complete'])
        bindings = json.loads((output / 'bindings.json').read_text())
        self.assertEqual(Path(bindings[self.row['logical']]).read_bytes(), self.source.read_bytes())
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(transfer.plan(manifest)['total_bytes'], 12)
        self.assertTrue((output / 'completed.json').is_file())

    def test_hash_failure_receipt_never_completed(self):
        self.row['sha256'] = '0' * 64
        output = self.root / 'bad'
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            transfer.copy_inputs(transfer.plan(self.manifest), output)
        self.assertFalse((output / 'completed.json').exists())
        self.assertFalse((output / 'bindings.json').exists())
        self.assertEqual(json.loads((output / 'failed.json').read_text())['state'], 'failed')
        self.assertEqual(self.source.read_bytes(), b'tiny fixture')

    def test_interrupted_completion_write_has_no_completed_marker(self):
        original = transfer.json.dump
        def fail_completion(value, handle, **kwargs):
            if value.get('state') == 'completed':
                handle.write('{')
                raise OSError('injected completion write failure')
            return original(value, handle, **kwargs)
        output = self.root / 'interrupted'
        with patch.object(transfer.json, 'dump', side_effect=fail_completion), self.assertRaises(OSError):
            transfer.copy_inputs(transfer.plan(self.manifest), output)
        self.assertFalse((output / 'completed.json').exists())
        self.assertEqual(json.loads((output / 'failed.json').read_text())['state'], 'failed')

    def test_future_receiver_paths_do_not_write_receiver(self):
        output = self.root / 'bundle'
        remote = '/future/receiver/bundle'
        transfer.copy_inputs(transfer.plan(self.manifest), output, remote)
        bindings = json.loads((output / 'bindings.json').read_text())
        self.assertEqual(bindings[self.row['logical']], remote + '/' + self.row['destination'])
        self.assertTrue((output / self.row['destination']).is_file())

    def test_reject_paths_and_aliases(self):
        for destination in ('../outside', 'inputs/../bad', 'inputs/C:/bad', 'inputs/NUL',
                            'inputs/a.', 'inputs/a\\b', '/inputs/a', 'inputs//a', 'inputs/a:stream'):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                transfer.plan(transfer.envelope([{**self.row, 'destination': destination}]))
        for other in ({**self.row, 'logical': 'D:/other.bin', 'destination': 'inputs/other'},
                      {**self.row, 'logical': self.row['logical'].upper()}):
            with self.assertRaises(ValueError):
                transfer.plan(transfer.envelope([self.row, other]))

    def test_reject_links(self):
        link = self.root / 'hardlink'
        os.link(self.source, link)
        with self.assertRaisesRegex(ValueError, 'Hardlinked'):
            transfer.plan(self.manifest)

    def test_reject_junction_or_symlink(self):
        link = self.root / 'link'
        try:
            link.symlink_to(self.source)
        except OSError as exc:
            self.skipTest(str(exc))
        self.row['physical'] = link.as_posix()
        with self.assertRaises(ValueError):
            transfer.plan(self.manifest)

    def test_fresh_external_and_overlap(self):
        planned = transfer.plan(self.manifest)
        for output in (self.root, self.source, self.source / 'child', transfer.ROOT / 'forbidden'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                transfer.copy_inputs(planned, output)

    def test_type_errors_and_unknown_fields(self):
        for row in ({**self.row, 'bytes': True}, {**self.row, 'unexpected': 1},
                    {**self.row, 'bytes': -1}, {**self.row, 'sha256': 'bad'},
                    {**self.row, 'logical': 'relative'}, {**self.row, 'bytes': 99}):
            with self.assertRaises(ValueError):
                transfer.plan(transfer.envelope([row]))

    def test_gpt_and_rl_adapters(self):
        gpt = {'schema_version': 1, 'transfer_recipe': [dict(logical=self.row['logical'],
               sha256=self.row['sha256'], kind='payload', destination=self.row['destination'])]}
        with patch.object(Path, 'open', side_effect=AssertionError('payload read')):
            converted = transfer.convert(gpt, 'gpt-report', {self.row['logical']: self.source.as_posix()})
            self.assertEqual(transfer.plan(converted)['total_bytes'], 12)
        rl = {'contract': 'kion-historical-cache-rl-preparation-v1', 'bindings': [
            {k: v for k, v in self.row.items() if k != 'destination'} | {'role': 'payload:test', 'content_verified': False}]}
        self.assertEqual(transfer.convert(rl, 'rl-report'), self.manifest)
        with self.assertRaises(ValueError):
            transfer.convert(gpt, 'gpt-report', {})

    def test_cli_default_no_files_and_explicit_report(self):
        source = self.root / 'manifest.json'
        source.write_text(json.dumps(self.manifest), encoding='utf-8')
        before = set(self.root.iterdir())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(transfer.main(['--input', str(source)]), 0)
        self.assertEqual(set(self.root.iterdir()), before)
        report = self.root / 'plan.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(transfer.main(['--input', str(source), '--report', str(report)]), 0)
        self.assertEqual(json.loads(report.read_text())['state'], 'planned')

    def test_cli_cannot_overwrite_metadata(self):
        source = self.root / 'manifest.json'
        source.write_text(json.dumps(self.manifest), encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(transfer.main(['--input', str(source), '--report', str(source)]), 2)
            self.assertEqual(transfer.main(['--input', str(source), '--destination', str(self.root / 'out')]), 2)
        self.assertFalse((self.root / 'out').exists())

    def test_source_mutation_during_copy_detected(self):
        original = transfer.snapshot
        count = 0
        def changed(info):
            nonlocal count
            count += 1
            if count == 3:
                self.source.write_bytes(b'changed data')
            return original(info)
        output = self.root / 'mutated'
        with patch.object(transfer, 'snapshot', side_effect=changed), self.assertRaises(ValueError):
            transfer.copy_inputs(transfer.plan(self.manifest), output)
        self.assertFalse((output / 'completed.json').exists())


if __name__ == '__main__':
    unittest.main()
