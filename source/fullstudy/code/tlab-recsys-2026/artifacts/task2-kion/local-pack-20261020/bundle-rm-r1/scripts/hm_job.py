"""Run one Python invocation with a finite deadline and owned child cleanup.

Usage: python -m scripts.hm_job --timeout 1860 -- -m scripts.hm_experiment fit ...
Windows starts suspended, assigns a kill-on-close Job, then resumes. This bounds
wall time and owns descendants, including virtualenv launcher children; it does
not impose an RSS or VRAM limit. POSIX uses a new process session.
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

if os.name == "nt":
    from ctypes import wintypes as W

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    def _api(name, result, *arguments):
        function = getattr(_kernel, name)
        function.restype, function.argtypes = result, arguments
        return function

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                    ("Flags", W.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                    ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", W.DWORD),
                    ("Affinity", ctypes.c_size_t), ("Priority", W.DWORD),
                    ("Scheduling", W.DWORD)]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadOperations", "WriteOperations", "OtherOperations",
                     "ReadBytes", "WriteBytes", "OtherBytes")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("Basic", _BasicLimits), ("Io", _IoCounters),
                    ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                    ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]

    class _Accounting(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in
                    ("UserTime", "KernelTime", "PeriodUserTime", "PeriodKernelTime")] + [
                    (name, W.DWORD) for name in
                    ("PageFaults", "TotalProcesses", "ActiveProcesses", "TerminatedProcesses")]

    class _ThreadEntry(ctypes.Structure):
        _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("tid", W.DWORD),
                    ("pid", W.DWORD), ("basePriority", W.LONG),
                    ("deltaPriority", W.LONG), ("flags", W.DWORD)]

    _close = _api("CloseHandle", W.BOOL, W.HANDLE)
    _create_job = _api("CreateJobObjectW", W.HANDLE, W.LPVOID, W.LPCWSTR)
    _set_job = _api("SetInformationJobObject", W.BOOL, W.HANDLE, ctypes.c_int, W.LPVOID, W.DWORD)
    _assign_job = _api("AssignProcessToJobObject", W.BOOL, W.HANDLE, W.HANDLE)
    _terminate_job = _api("TerminateJobObject", W.BOOL, W.HANDLE, W.UINT)
    _query_job = _api("QueryInformationJobObject", W.BOOL, W.HANDLE, ctypes.c_int,
                      W.LPVOID, W.DWORD, W.LPVOID)
    _snapshot = _api("CreateToolhelp32Snapshot", W.HANDLE, W.DWORD, W.DWORD)
    _thread_first = _api("Thread32First", W.BOOL, W.HANDLE, ctypes.POINTER(_ThreadEntry))
    _thread_next = _api("Thread32Next", W.BOOL, W.HANDLE, ctypes.POINTER(_ThreadEntry))
    _open_thread = _api("OpenThread", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
    _resume_thread = _api("ResumeThread", W.DWORD, W.HANDLE)

    def _new_job():
        job = _create_job(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            _close(job)
            raise error
        return job

    def _resume_initial_thread(pid):
        snapshot = _snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            present = _thread_first(snapshot, ctypes.byref(entry))
            while present:
                if entry.pid == pid:
                    thread = _open_thread(2, False, entry.tid)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if _resume_thread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        _close(thread)
                    return
                present = _thread_next(snapshot, ctypes.byref(entry))
            raise OSError(f"Initial thread of suspended process {pid} not found")
        finally:
            _close(snapshot)


def _cleanup(process, job):
    """Kill descendants even if the Python/virtualenv launcher already exited."""
    try:
        if os.name == "nt":
            if not _terminate_job(job, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + 10
            while True:
                accounting = _Accounting()
                if not _query_job(job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if accounting.ActiveProcesses == 0:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Windows Job cleanup did not finish")
                time.sleep(.01)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        if job is not None:
            _close(job)  # Kill-on-close also applies if explicit cleanup failed.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)


def run(arguments, timeout):
    """Return the child exit status, 124 on timeout, or 130 on interruption."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if not arguments or any(not isinstance(value, str) for value in arguments):
        raise ValueError("Python arguments must be a nonempty list of strings")
    job = _new_job() if os.name == "nt" else None
    process = None
    try:
        kwargs = dict(cwd=ROOT)
        if os.name == "nt":
            kwargs["creationflags"] = 4 | subprocess.CREATE_NEW_PROCESS_GROUP  # CREATE_SUSPENDED
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen([sys.executable, *arguments], **kwargs)
        if os.name == "nt":
            # The worker cannot execute Python until ownership is established.
            if not _assign_job(job, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            _resume_initial_thread(process.pid)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"hm_job: timeout after {timeout:g} seconds", file=sys.stderr, flush=True)
            return 124
        except KeyboardInterrupt:
            return 130
    finally:
        if process is not None:
            _cleanup(process, job)
        elif job is not None:
            _close(job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.arguments or args.arguments[0] != "--" or len(args.arguments) < 2:
        parser.error("Use -- followed by Python arguments, for example -- -m scripts.hm_experiment fit ...")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        return run(args.arguments[1:], args.timeout)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"hm_job: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
