"""Synthetic admission/guard tests only: no subprocess, native Job, GPU or payload."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common
import launch
import supervise

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=common.CONTROL, prefix='.test-control-')
        self.root = Path(self.temporary.name)/'c-rl-20261024-r1'
        self.control = self.root/'control'
        self.control.mkdir(parents=True)
        self.pas = self.root/'pas'
        self.runtime = self.root/'runtime.py'
        self.stage = self.root/'policy'/'stage.py'
        self.stage.parent.mkdir()
        self.stage.write_text('def run(request, check):\n    check()\n    return {}\n', encoding='utf-8')
        self.patches = [patch.object(common, 'ROOT', self.root), patch.object(common, 'CONTROL', self.control), patch.object(common, 'PAS', self.pas), patch.object(common, 'RUNTIME', self.runtime), patch.object(supervise, 'ROOT', self.root)]
        for item in self.patches:
            item.start()
        sources = [self.stage, self.runtime, *(self.control/name for name in ('common.py', 'launch.py', 'supervise.py', 'worker.py')), self.pas/'scripts'/'portability.py']
        for path in sources:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# pinned synthetic source\n', encoding='utf-8')
        self.input = self.root/'input.bin'
        self.input.write_bytes(b'synthetic input')
        self.request_path = self.root/'request-synthetic.json'
        self.request = {'contract': 'kion-c-rl-native-v1', 'operation': 'synthetic', 'stage': 'train_rl', 'module_path': self.stage.as_posix(), 'function': 'run', 'supervised_updates': 0, 'new_rm_fits': 0, 'final_test_reads': 0, 'rl_updates': 1, 'cpu_threads': 4, 'reserve_gib': 2, 'cap_gib': 4, 'maximum_seconds': 60, 'deadline_utc': (datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat(), 'source_hashes': {path.as_posix(): common.sha(path) for path in sources}, 'input_hashes': {self.input.as_posix(): common.sha(self.input)}, 'owned_output_roots': [(self.root/'outputs').as_posix()], 'output_paths': [(self.root/'outputs'/'out.json').as_posix()], 'required_output_bytes': {self.root.drive.upper(): 0}}
        self.campaign = {'contract': 'kion-C3-reward-RL-execution-v1', 'state': 'running', 'round_limit': 3, 'limits': {'supervised_updates': 0, 'new_RM_fits': 0, 'final_test_reads': 0, 'GPU_jobs': 1, 'heavy_native_jobs': 1, 'external_downloads': False, 'paid_operations': False, 'RL_updates_max': 1000}}
        self.save_request()
        self.disk = patch.object(common.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 100*common.GIB})())
        self.disk.start()

    def tearDown(self):
        self.disk.stop()
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def save_request(self):
        self.request_path.write_text(json.dumps(self.request), encoding='utf-8')
        self.digest = common.sha(self.request_path)
        self.campaign['admitted_requests'] = {'synthetic': {'request_path': self.request_path.as_posix(), 'sha256': self.digest}}
        (self.root/'campaign.json').write_text(json.dumps(self.campaign), encoding='utf-8')

    def validate(self):
        return common.validate(self.request, self.request_path, self.digest)

    def test_valid_synthetic_admission(self):
        self.assertIs(self.validate(), self.request)

    def test_root_admission_digest_cannot_be_changed(self):
        self.campaign['admitted_requests']['synthetic']['sha256'] = '0'*64
        (self.root/'campaign.json').write_text(json.dumps(self.campaign), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'root campaign admission'):
            self.validate()

    def test_exclusions_and_caps(self):
        for key, value in [('supervised_updates', 1), ('new_rm_fits', 1), ('final_test_reads', 1), ('rl_updates', 1001), ('cap_gib', 16), ('cpu_threads', 8), ('maximum_seconds', 3601)]:
            with self.subTest(key=key):
                original = self.request[key]
                self.request[key] = value
                self.save_request()
                with self.assertRaises(ValueError):
                    self.validate()
                self.request[key] = original
        self.save_request()

    def test_extended_cache_probe_budget_but_no_other_extension(self):
        for stage in ('prepare_cache', 'reward_probe'):
            self.request.update(stage=stage, maximum_seconds=7200)
            self.save_request()
            self.validate()

    def test_unknown_or_foreign_module_and_function_rejected(self):
        for key, value in [('stage', 'fit_rm'), ('function', 'exec'), ('module_path', self.input.as_posix())]:
            original = self.request[key]
            self.request[key] = value
            self.save_request()
            with self.assertRaises(ValueError):
                self.validate()
            self.request[key] = original

    def test_input_receipt_stat_change_is_rejected_without_rehash(self):
        path = self.input.as_posix()
        self.request['input_digest_receipts'] = {path: {'sha256': common.sha(self.input), 'stat': common.stat_key(self.input)}}
        self.save_request()
        self.validate()
        self.input.write_bytes(b'changed input')
        with self.assertRaisesRegex(ValueError, 'receipt stale'):
            self.validate()

    def test_source_change_is_rejected_even_with_input_receipts(self):
        self.stage.write_text('# changed\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Pinned content changed'):
            self.validate()

    def test_changed_supervisor_cannot_execute_before_pin_check(self):
        (self.control/'supervise.py').write_text('# changed supervisor\n', encoding='utf-8')
        with patch.object(launch, 'ROOT', self.root), patch.object(launch.sys, 'argv', ['launch.py', self.request_path.name]), patch.object(launch.subprocess, 'run') as spawned:
            with self.assertRaisesRegex(ValueError, 'Pinned source changed'):
                launch.main()
            spawned.assert_not_called()

    def test_stop_and_expired_deadline_reject_admission(self):
        (self.root/'STOP').touch()
        with self.assertRaisesRegex(ValueError, 'stop requested'):
            self.validate()
        (self.root/'STOP').unlink()
        self.request['deadline_utc'] = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        self.save_request()
        with self.assertRaisesRegex(ValueError, 'deadline reached'):
            self.validate()

    def test_owned_space_probe_and_reserve(self):
        common.probe_space(self.request, write_probe=True)
        self.assertEqual(list((self.root/'outputs').iterdir()), [])
        with patch.object(common.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 14*common.GIB})()):
            with self.assertRaisesRegex(ValueError, 'Disk reserve'):
                common.probe_space(self.request)

    def test_lifecycle_stop_reasons(self):
        request = self.request
        self.assertIsNone(supervise.stop_reason(request, 0, 10*common.GIB, 0, 0))
        self.assertEqual(supervise.stop_reason(request, 61, 10*common.GIB, 0, 0), 'time_limit')
        self.assertEqual(supervise.stop_reason(request, 0, common.GIB, 0, 0), 'host_reserve')
        self.assertEqual(supervise.stop_reason(request, 0, 10*common.GIB, 5*common.GIB, 0), 'memory_cap')
        (self.root/'STOP').touch()
        self.assertEqual(supervise.stop_reason(request, 0, 10*common.GIB, 0, 0), 'stop_requested')

    def test_cleanup_failure_blocks_next_native_operation(self):
        previous = self.root/'operations'/'previous'
        previous.mkdir(parents=True)
        (previous/'receipt.json').write_text(json.dumps({'cleanup_quiescent': False}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'quiescence'):
            supervise.previous_quiescent(self.root/'operations'/'next')
        (previous/'receipt.json').write_text(json.dumps({'cleanup_quiescent': True}), encoding='utf-8')
        supervise.previous_quiescent(self.root/'operations'/'next')

if __name__ == '__main__':
    unittest.main()
