"""Same-handle identity-checked remote WS ceiling. CLI is pinned to this owned SFT."""
import argparse
import ctypes
from ctypes import wintypes as W
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PLAN_SHA = 'f3e0bac22787ae3b1e7e1609a428128842205b18c39293d4fba83170f4bb514f'
BASE_SHA = 'abc1aa0b5a63f745804e88789b2dfe6857582ec19ed36b3b5f9dc6b340212344'
SFT_PID = 33360
SFT_CREATION = 134370094266828798
SFT_PARENT = 100900


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalized(path):
    return os.path.normcase(str(Path(path).resolve()))


class Native:
    def __init__(self):
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'OpenProcess': (W.HANDLE, [W.DWORD, W.BOOL, W.DWORD]),
            'CloseHandle': (W.BOOL, [W.HANDLE]),
            'GetProcessId': (W.DWORD, [W.HANDLE]),
            'GetProcessTimes': (W.BOOL, [W.HANDLE] + [ctypes.POINTER(W.FILETIME)] * 4),
            'QueryFullProcessImageNameW': (W.BOOL, [W.HANDLE, W.DWORD, W.LPWSTR, ctypes.POINTER(W.DWORD)]),
            'GetExitCodeProcess': (W.BOOL, [W.HANDLE, ctypes.POINTER(W.DWORD)]),
            'GetProcessWorkingSetSizeEx': (W.BOOL, [W.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(W.DWORD)]),
            'SetProcessWorkingSetSizeEx': (W.BOOL, [W.HANDLE, ctypes.c_size_t, ctypes.c_size_t, W.DWORD]),
        }
        for name, (result, args) in signatures.items():
            method = getattr(self.k, name)
            method.argtypes, method.restype = args, result

    def checked(self, result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def open(self, pid):
        return self.checked(self.k.OpenProcess(0x1000 | 0x100, False, pid))

    def close(self, handle):
        self.checked(self.k.CloseHandle(handle))

    def identity(self, handle):
        times = [W.FILETIME() for _ in range(4)]
        self.checked(self.k.GetProcessTimes(handle, *(ctypes.byref(x) for x in times)))
        size, buffer = W.DWORD(32768), ctypes.create_unicode_buffer(32768)
        self.checked(self.k.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)))
        exit_code = W.DWORD()
        self.checked(self.k.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
        if exit_code.value != 259:
            raise ValueError('Target exited; refuse any setter')
        return {'pid': self.k.GetProcessId(handle), 'exe': Path(buffer.value).resolve().as_posix(),
            'creation_time_100ns': times[0].dwLowDateTime | (times[0].dwHighDateTime << 32)}

    def policy(self, handle):
        minimum, maximum, flags = ctypes.c_size_t(), ctypes.c_size_t(), W.DWORD()
        self.checked(self.k.GetProcessWorkingSetSizeEx(handle, ctypes.byref(minimum), ctypes.byref(maximum), ctypes.byref(flags)))
        return dict(minimum=minimum.value, maximum=maximum.value, flags=flags.value)

    def set(self, handle, minimum, maximum):
        self.checked(self.k.SetProcessWorkingSetSizeEx(handle, minimum, maximum, 0x2 | 0x4))


def command_identity(pid):
    # Strict integer interpolation only; no supplied paths/command text enter PowerShell source.
    if type(pid) is not int or pid <= 0:
        raise ValueError('Positive integer PID required')
    script = ('[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding; '
              f'Get-CimInstance Win32_Process -Filter "ProcessId = {pid}" | '
              'Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress')
    shell = Path(os.environ['SYSTEMROOT']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    result = subprocess.run([str(shell), '-NoProfile', '-NonInteractive', '-Command', script],
        capture_output=True, text=True, encoding='utf-8', timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Cannot obtain exact command/parent identity')
    info = json.loads(result.stdout)
    shell32 = ctypes.WinDLL('shell32', use_last_error=True)
    parse = shell32.CommandLineToArgvW
    parse.argtypes, parse.restype = [W.LPCWSTR, ctypes.POINTER(ctypes.c_int)], ctypes.POINTER(W.LPWSTR)
    count = ctypes.c_int()
    pointer = parse(info['CommandLine'], ctypes.byref(count))
    if not pointer:
        raise ctypes.WinError(ctypes.get_last_error())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes, kernel.LocalFree.restype = [W.HLOCAL], W.HLOCAL
    try:
        argv = [pointer[i] for i in range(count.value)]
    finally:
        kernel.LocalFree(pointer)
    return {'pid': info['ProcessId'], 'parent_pid': info['ParentProcessId'], 'argv': argv}


def verify_identity(actual, expected):
    if (actual['pid'] != expected['pid'] or actual['creation_time_100ns'] != expected['creation_time_100ns']
            or normalized(actual['exe']) != normalized(expected['exe'])):
        raise ValueError('Same-handle PID/creation/image mismatch')


def apply_verified(expected, maximum, *, api=None, query=command_identity):
    if type(maximum) is not int or not 32 << 20 <= maximum <= 11 << 30:
        raise ValueError('32MiB..11GiB ceiling required')
    api = Native() if api is None else api
    handle = api.open(expected['pid'])
    try:
        verify_identity(api.identity(handle), expected)
        command = query(expected['pid'])
        if (command['pid'] != expected['pid'] or command['parent_pid'] != expected['parent_pid']
                or not command['argv'] or normalized(command['argv'][0]) != normalized(expected['argv'][0])
                or command['argv'][1:] != expected['argv'][1:]):
            raise ValueError('Exact owned command/parent mismatch')
        before = api.policy(handle)
        # Recheck the ORIGINAL handle after CIM; never reopen by PID for the setter.
        verify_identity(api.identity(handle), expected)
        api.set(handle, min(before['minimum'], maximum), maximum)
        after = api.policy(handle)
        if not after['flags'] & 0x4 or after['flags'] & (0x1 | 0x8) or after['maximum'] > maximum:
            raise RuntimeError('Windows did not retain soft-minimum/hard-maximum policy')
        return {'state': 'applied', 'identity': expected, 'before': before, 'after': after,
            'same_handle_identity_rechecked': True, 'elevation_requested': False, 'privileges_adjusted': False}
    finally:
        api.close(handle)


def owned_binding(plan_path, approval_path, state_path):
    plan, approval, state = read(plan_path), read(approval_path), read(state_path)
    if canonical(plan) != PLAN_SHA or approval.get('verdict') != 'approved' or approval.get('plan_sha256') != PLAN_SHA:
        raise ValueError('Only the exact independently approved current SFT plan admitted')
    job = plan['jobs'][0]
    attempts = [a for a in state['attempts'] if a.get('pid') == SFT_PID]
    if len(plan['jobs']) != 1 or len(attempts) != 1 or job['id'] != 'ready-scoped-sft-r1':
        raise ValueError('Only one exact owned current SFT attempt admitted')
    attempt = attempts[0]
    if (attempt['identity']['creation_time_100ns'] != SFT_CREATION
            or attempt['supervisor_pid'] != SFT_PARENT
            or normalized(attempt['identity']['exe']) != normalized(plan['interpreter'])):
        raise ValueError('Exact root-bound SFT creation/image/parent required')
    output = Path(job['output'])
    expanded = [a.replace('{output}', output.as_posix()).replace('{stop_file}', (output / 'stop-request.json').as_posix()) for a in job['argv']]
    expected_argv = [plan['interpreter'], *expanded]
    if (attempt.get('state') != 'running' or attempt['argv'] != expected_argv
            or expected_argv[1:3] != ['scripts/kion_ready_runtime.py', 'sft']
            or hashlib.sha256(Path(plan['interpreter']).read_bytes()).hexdigest() != BASE_SHA):
        raise ValueError('Owned SFT argv/state/base executable differs')
    return {**attempt['identity'], 'parent_pid': attempt['supervisor_pid'], 'argv': expected_argv}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'approval', 'state', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--maximum-gib', type=float, choices=(10., 10.5, 11.), required=True)
    parser.add_argument('--apply-owned-sft', action='store_true', required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    expected = owned_binding(args.plan, args.approval, args.state)
    receipt = apply_verified(expected, int(args.maximum_gib * (1 << 30)))
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
