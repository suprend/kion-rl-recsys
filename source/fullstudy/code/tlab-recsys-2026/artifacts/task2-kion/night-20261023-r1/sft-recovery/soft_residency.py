"""Opt-in SELF-only advisory residency relief, never a memory-fit guarantee.

Import/option validation perform no native API calls. Native construction and
operations occur only in a later admitted worker at completed-update boundaries.
"""
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes as W
import json
import math
import os
from pathlib import Path
import time

OPTIONS = {
    'contract': 'kion-self-soft-residency-v1',
    'enabled': True,
    'boundary': 'after-successful-train-update-return',
    'trigger_rss_bytes': 12 << 30,
    'minimum_seconds_between_checks': 60,
    'minimum_successful_updates_between_checks': 100,
    'maximum_empty_working_set_calls': 60,
    'maximum_observed_native_api_seconds': 30,
}
PROBE_OPTIONS = {**OPTIONS, 'contract': 'kion-self-soft-residency-probe-v1',
    'boundary': 'after-owned-64MiB-buffer-setup-before-retouch',
    'trigger_rss_bytes': 1, 'minimum_seconds_between_checks': 0,
    'minimum_successful_updates_between_checks': 1,
    'maximum_empty_working_set_calls': 1, 'maximum_observed_native_api_seconds': 5}


def require(ok, message):
    if not ok:
        raise ValueError('SELF residency: ' + message)


def validate_options(options):
    require(type(options) is dict and options == OPTIONS
            and all(type(options[key]) is type(value) for key, value in OPTIONS.items()),
            'only exact independently reviewed opt-in bounds are supported')


def validate_probe_options(options):
    require(type(options) is dict and options == PROBE_OPTIONS
            and all(type(options[key]) is type(value) for key, value in PROBE_OPTIONS.items()),
            'only fixed small standalone probe options are supported')


class SelfNative:
    """No PID, OpenProcess, CloseHandle, setter, child or privilege interface."""
    def __init__(self):
        require(os.name == 'nt', 'native implementation is Windows-only')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.kernel.GetCurrentProcess.argtypes = []
        self.kernel.GetCurrentProcess.restype = W.HANDLE
        self.handle = self.kernel.GetCurrentProcess()
        require(self.handle == ctypes.c_void_p(-1).value, 'GetCurrentProcess must return SELF pseudo-handle')
        self.self_pseudo_handle_verified = True

        class Counters(ctypes.Structure):
            _fields_ = [('cb', W.DWORD), ('PageFaultCount', W.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
        self.Counters = Counters
        self.psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.POINTER(Counters), W.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = W.BOOL
        self.psapi.EmptyWorkingSet.argtypes = [W.HANDLE]
        self.psapi.EmptyWorkingSet.restype = W.BOOL

    def sample(self):
        require(self.handle == ctypes.c_void_p(-1).value, 'SELF pseudo-handle changed')
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'rss_bytes': int(counters.WorkingSetSize), 'private_bytes': int(counters.PrivateUsage),
                'page_faults': int(counters.PageFaultCount)}

    def empty(self):
        require(self.handle == ctypes.c_void_p(-1).value, 'SELF pseudo-handle changed')
        # Return and native last-error are captured before another native call.
        ok = bool(self.psapi.EmptyWorkingSet(self.handle))
        error = int(ctypes.get_last_error()) if not ok else 0
        return {'ok': ok, 'winerror': error}


class Journal:
    """Single writer; every attempt is durable before EmptyWorkingSet is called."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('x', encoding='utf-8', newline='\n')

    def append(self, value):
        self.stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


class Control:
    options_validator = staticmethod(validate_options)

    def __init__(self, options, journal_path, *, initial_global_update, api=None, journal=None, clock=time.monotonic,
                 budget_check=lambda: None):
        self.options_validator(options)
        require(type(initial_global_update) is int and initial_global_update >= 0, 'initial update must be explicit')
        self.options = dict(options)
        self.initial_global_update = initial_global_update
        self.clock = clock
        self.budget_check = budget_check
        self.started = self.last_check_at = self.clock()
        self.last_boundary = self.last_check_update = 0
        self.calls = self.attempts = self.checks = 0
        self.native_api_seconds = 0.0
        self.failed = self.closed = False
        self.last_sample = None
        # Create the local journal before even constructing native access.
        self.journal = journal if journal is not None else Journal(journal_path)
        try:
            self.api = api if api is not None else self._timed(SelfNative)
        except BaseException:
            self.journal.close()
            raise

    def _timed(self, callback):
        self.budget_check()
        started = self.clock()
        try:
            return callback()
        finally:
            elapsed = self.clock() - started
            require(math.isfinite(elapsed) and elapsed >= 0, 'non-monotonic native API clock')
            self.native_api_seconds += elapsed

    def _sample(self):
        value = self._timed(self.api.sample)
        require(type(value) is dict and set(value) == {'rss_bytes', 'private_bytes', 'page_faults'}
                and all(type(x) is int and x >= 0 for x in value.values()), 'SELF memory sample invalid')
        self.last_sample = dict(value)
        return value

    def _event(self, event, completed_updates, **fields):
        return {'event': event, 'scope': 'SELF GetCurrentProcess pseudo-handle',
                'safe_boundary': self.options['boundary'],
                'updated_utc': datetime.now(timezone.utc).isoformat(),
                'completed_updates_in_recovery': completed_updates,
                'global_update': self.initial_global_update + completed_updates,
                'observed_native_api_seconds': self.native_api_seconds, **fields}

    def after_successful_update(self, completed_updates):
        """Caller invokes ONLY after original update fully returned successfully."""
        require(not self.failed and not self.closed, 'failed/closed control cannot run again')
        try:
            require(type(completed_updates) is int and completed_updates == self.last_boundary + 1,
                    'exactly one boundary per successful completed update is required')
            self.last_boundary = completed_updates
            self.budget_check()
            now = self.clock()
            if (now - self.last_check_at < self.options['minimum_seconds_between_checks']
                    or completed_updates - self.last_check_update < self.options['minimum_successful_updates_between_checks']):
                return {'state': 'cooldown', 'empty_working_set_called': False}
            require(self.native_api_seconds < self.options['maximum_observed_native_api_seconds'], 'observed native API budget exhausted')
            self.last_check_at, self.last_check_update = now, completed_updates
            self.checks += 1
            before = self._sample()
            require(self.native_api_seconds < self.options['maximum_observed_native_api_seconds'], 'observed native API budget exhausted after sample')
            if before['rss_bytes'] < self.options['trigger_rss_bytes']:
                return {'state': 'below-trigger', 'empty_working_set_called': False, 'sample': before}
            require(self.calls < self.options['maximum_empty_working_set_calls'], 'EmptyWorkingSet call budget exhausted')
            sequence = self.attempts + 1
            # append() flushes+fsyncs, and any failure prevents the call.
            self.journal.append(self._event('attempt', completed_updates,
                sequence=sequence, before=before, action='advisory EmptyWorkingSet(SELF)'))
            self.attempts += 1
            native_started = self.clock()
            result, after, error = None, None, None
            try:
                def invoke_empty():
                    self.calls += 1
                    return self.api.empty()
                result = self._timed(invoke_empty)
                require(type(result) is dict and set(result) == {'ok', 'winerror'}
                        and type(result['ok']) is bool and type(result['winerror']) is int,
                        'EmptyWorkingSet result invalid')
            except Exception as exception:
                error = type(exception).__name__ + ': ' + str(exception)
            empty_seconds = self.clock() - native_started
            try:
                after = self._sample()
            except Exception as exception:
                error = error or type(exception).__name__ + ': ' + str(exception)
            budget_ok = self.native_api_seconds <= self.options['maximum_observed_native_api_seconds']
            ok = error is None and result is not None and result['ok'] and budget_ok
            self.journal.append(self._event('result', completed_updates, sequence=sequence,
                before=before, after=after, return_value=result, empty_call_seconds=empty_seconds,
                error=error, observed_native_api_budget_ok=budget_ok, state='returned' if ok else 'failed'))
            require(ok, 'advisory call/sample failed or observed API budget exceeded; no retry')
            return {'state': 'advisory-returned', 'empty_working_set_called': True, 'before': before, 'after': after}
        except BaseException:
            self.failed = True
            raise

    def evidence(self):
        return {'contract': self.options['contract'], 'options': self.options,
                'scope': 'SELF-only', 'safe_boundary': self.options['boundary'],
                'self_pseudo_handle_verified': bool(getattr(self.api, 'self_pseudo_handle_verified', False)),
                'completed_update_boundaries': self.last_boundary, 'eligible_memory_checks': self.checks,
                'durable_attempt_records': self.attempts, 'empty_working_set_calls': self.calls,
                'observed_native_api_seconds': self.native_api_seconds,
                'last_sample': self.last_sample, 'failed': self.failed, 'closed': self.closed,
                'single_native_call_internally_cancellable': False,
                'hard_residency_limit': False, 'commit_reduction_claim': False,
                'whole_run_fit_or_speed_claim': False}

    def close(self):
        if not self.closed:
            self.closed = True
            self.journal.close()


class ProbeControl(Control):
    """Separate fixed small-buffer protocol; production Control rejects these bounds."""
    options_validator = staticmethod(validate_probe_options)

    def _event(self, event, completed_updates, **fields):
        value = super()._event(event, completed_updates, **fields)
        value['completed_buffer_boundaries'] = value.pop('completed_updates_in_recovery')
        value.pop('global_update')
        return value

    def evidence(self):
        value = super().evidence()
        value['completed_buffer_boundaries'] = value.pop('completed_update_boundaries')
        value['optimizer_updates'] = 0
        return value

    def after_buffer_setup(self):
        return self.after_successful_update(1)

    def after_retouch_sample(self):
        require(not self.failed and not self.closed and self.calls == 1, 'single successful probe call required first')
        try:
            value = self._sample()
            require(self.native_api_seconds <= self.options['maximum_observed_native_api_seconds'], 'probe API budget exceeded')
            return value
        except BaseException:
            self.failed = True
            raise
