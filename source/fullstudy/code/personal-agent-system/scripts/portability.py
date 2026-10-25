"""Small stdlib-only OS boundary for locks, safe files and owned subprocesses.

``popen_isolated`` owns a bounded invocation and all its children. Always call
``terminate_process_tree`` in a finally block, including after a successful wait.
Use ordinary Popen for daemons intended to outlive their launching process.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import weakref

if os.name == "nt":
    import msvcrt
    from ctypes import wintypes as W

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    def _api(name, result, *args):
        function = getattr(_kernel, name)
        function.restype = result
        function.argtypes = args
        return function

    _close = _api("CloseHandle", W.BOOL, W.HANDLE)
    _INVALID = ctypes.c_void_p(-1).value

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", W.DWORD), ("OffsetHigh", W.DWORD), ("hEvent", W.HANDLE)]

    _lock_file = _api("LockFileEx", W.BOOL, W.HANDLE, W.DWORD, W.DWORD,
                      W.DWORD, W.DWORD, ctypes.POINTER(_Overlapped))
    _unlock_file = _api("UnlockFileEx", W.BOOL, W.HANDLE, W.DWORD,
                        W.DWORD, W.DWORD, ctypes.POINTER(_Overlapped))

    class _FileInfo(ctypes.Structure):
        _fields_ = [("attributes", W.DWORD), ("creation", W.FILETIME),
                    ("access", W.FILETIME), ("write", W.FILETIME),
                    ("volume", W.DWORD), ("sizeHigh", W.DWORD), ("sizeLow", W.DWORD),
                    ("links", W.DWORD), ("indexHigh", W.DWORD), ("indexLow", W.DWORD)]

    _create_file = _api("CreateFileW", W.HANDLE, W.LPCWSTR, W.DWORD, W.DWORD,
                        W.LPVOID, W.DWORD, W.DWORD, W.HANDLE)
    _file_info = _api("GetFileInformationByHandle", W.BOOL, W.HANDLE, ctypes.POINTER(_FileInfo))
    _file_type = _api("GetFileType", W.DWORD, W.HANDLE)

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

    class _JobAccounting(ctypes.Structure):
        _fields_ = [(name, ctypes.c_int64) for name in
                    ("UserTime", "KernelTime", "PeriodUserTime", "PeriodKernelTime")] + [
                    (name, W.DWORD) for name in
                    ("PageFaults", "TotalProcesses", "ActiveProcesses", "TerminatedProcesses")]

    class _ThreadEntry(ctypes.Structure):
        _fields_ = [("size", W.DWORD), ("usage", W.DWORD), ("tid", W.DWORD),
                    ("pid", W.DWORD), ("basePriority", W.LONG),
                    ("deltaPriority", W.LONG), ("flags", W.DWORD)]

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
else:
    import fcntl


class _LockEntry:
    def __init__(self):
        self.lock = threading.RLock()


_locks = weakref.WeakValueDictionary()
_locks_guard = threading.Lock()
_depths = threading.local()


@contextmanager
def exclusive_lock(handle, timeout: float | None = None):
    """Lock byte zero without moving the file position, across threads/processes.

    The file must remain open for the complete context. Reentrant in one thread.
    Finite timeouts cover both the in-process and OS lock wait.
    """
    if timeout is not None and timeout < 0:
        raise ValueError("Lock timeout must be nonnegative")
    descriptor = handle if isinstance(handle, int) else handle.fileno()
    metadata = os.fstat(descriptor)
    key = (metadata.st_dev, metadata.st_ino)
    with _locks_guard:
        entry = _locks.get(key)
        if entry is None:
            entry = _locks[key] = _LockEntry()
    deadline = None if timeout is None else time.monotonic() + timeout
    acquired = entry.lock.acquire() if timeout is None else entry.lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError("Timed out waiting for file lock")
    depths = getattr(_depths, "values", None)
    if depths is None:
        depths = _depths.values = {}
    nested = key in depths
    locked = False
    overlapped = _Overlapped() if os.name == "nt" else None
    try:
        if not nested:
            while True:
                if os.name == "nt":
                    ok = _lock_file(msvcrt.get_osfhandle(descriptor), 2 | (1 if deadline is not None else 0),
                                    0, 1, 0, ctypes.byref(overlapped))
                    if ok:
                        break
                    code = ctypes.get_last_error()
                    if deadline is None or code != 33:  # ERROR_LOCK_VIOLATION
                        raise ctypes.WinError(code)
                else:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | (fcntl.LOCK_NB if deadline is not None else 0))
                        break
                    except OSError as error:
                        if deadline is None or error.errno not in (errno.EACCES, errno.EAGAIN):
                            raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for file lock")
                time.sleep(min(.02, remaining))
            locked = True
        depths[key] = depths.get(key, 0) + 1
        try:
            yield handle
        finally:
            depths[key] -= 1
            if not depths[key]:
                del depths[key]
    finally:
        try:
            if locked:
                if os.name == "nt":
                    if not _unlock_file(msvcrt.get_osfhandle(descriptor), 0, 1, 0, ctypes.byref(overlapped)):
                        raise ctypes.WinError(ctypes.get_last_error())
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            entry.lock.release()


def open_regular_no_follow(path, flags: int, mode: int = 0o600) -> int:
    """Open a single-linked regular file; validate before any truncation/write.

    Rejects a symlink/reparse point in the final path component. Callers must
    separately validate parent directories and control the containing directory.
    """
    safe_flags = flags & ~os.O_TRUNC
    if os.name == "nt":
        access = 0x80000000 if not flags & os.O_WRONLY else 0
        if flags & (os.O_WRONLY | os.O_RDWR):
            access |= 0x40000000
        disposition = (1 if flags & os.O_EXCL else 4) if flags & os.O_CREAT else 3
        raw = _create_file(os.fsdecode(path), access, 7, None, disposition,
                           0x00200000 | 0x02000000, None)  # OPEN_REPARSE_POINT | BACKUP_SEMANTICS
        if raw == _INVALID:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = _FileInfo()
            if not _file_info(raw, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & 0x400:
                raise ValueError(f"File cannot be a symlink/reparse point: {path}")
            if info.attributes & 0x10 or info.links != 1 or _file_type(raw) != 1:
                raise ValueError(f"File must be an unlinked regular file: {path}")
            descriptor = msvcrt.open_osfhandle(raw, safe_flags & ~(os.O_CREAT | os.O_EXCL))
            raw = None  # CRT owns the native handle now.
        finally:
            if raw is not None:
                _close(raw)
    else:
        try:
            descriptor = os.open(path, safe_flags | os.O_NOFOLLOW | os.O_NONBLOCK, mode)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(f"File cannot be a symlink: {path}") from error
            raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"File must be an unlinked regular file: {path}")
        if os.name != "nt":
            current = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            fcntl.fcntl(descriptor, fcntl.F_SETFL, current & ~os.O_NONBLOCK)
        if flags & os.O_TRUNC:
            os.ftruncate(descriptor, 0)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def process_group_kwargs() -> dict:
    """Flags for a separate group, without claiming descendant ownership."""
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}


if os.name == "nt":
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
        if snapshot == _INVALID:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            item = _ThreadEntry()
            item.size = ctypes.sizeof(item)
            present = _thread_first(snapshot, ctypes.byref(item))
            while present:
                if item.pid == pid:
                    thread = _open_thread(2, False, item.tid)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if _resume_thread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        _close(thread)
                    return
                present = _thread_next(snapshot, ctypes.byref(item))
            raise OSError(f"Initial thread of suspended process {pid} was not found")
        finally:
            _close(snapshot)


class _OwnedPopen(subprocess.Popen):
    def __init__(self, *args, **kwargs):
        self._tree_closed = False
        self._tree_lock = threading.Lock()
        self._job = None
        if os.name == "nt":
            if kwargs.get("creationflags", 0) & 0x01000000:
                raise ValueError("Owned processes cannot break away from their Job")
            self._job = _new_job()
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | 4 | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            if kwargs.get("process_group") is not None or kwargs.get("start_new_session") is False:
                raise ValueError("Owned processes require a new session")
            kwargs["start_new_session"] = True
        try:
            super().__init__(*args, **kwargs)
            if os.name == "nt":
                if not _assign_job(self._job, int(self._handle)):
                    raise ctypes.WinError(ctypes.get_last_error())
                _resume_initial_thread(self.pid)
        except BaseException:
            if getattr(self, "_child_created", False):
                self.kill()
                self.wait()
            if self._job is not None:
                _close(self._job)
                self._job = None
            raise

    def close_process_tree(self):
        with self._tree_lock:
            if self._tree_closed:
                return
            if os.name == "nt":
                if self._job is not None:
                    if not _terminate_job(self._job, 1):
                        raise ctypes.WinError(ctypes.get_last_error())
                    deadline = time.monotonic() + 10
                    while True:
                        accounting = _JobAccounting()
                        if not _query_job(self._job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if accounting.ActiveProcesses == 0:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Windows Job did not finish terminating")
                        time.sleep(.01)
                    _close(self._job)
                    self._job = None
            else:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._tree_closed = True
            self.wait(timeout=10)

    def __exit__(self, *args):
        # Kill before Popen waits, so inherited stdout/stderr cannot keep it hung.
        try:
            self.close_process_tree()
        except BaseException:
            super().__exit__(*args)
            raise
        return super().__exit__(*args)

    def __del__(self):
        # Losing an owned Windows object must not leak its Job handle/children.
        if os.name == "nt" and getattr(self, "_job", None) is not None:
            _close(self._job)
            self._job = None
        super().__del__()


def popen_isolated(*args, **kwargs) -> subprocess.Popen:
    """Start a bounded process with child ownership established before execution."""
    return _OwnedPopen(*args, **kwargs)


def process_peak_memory_bytes(process: subprocess.Popen) -> int:
    """Windows Job's largest per-process peak, retained after a leader exits.

    Query before closing the owned tree. POSIX callers should use wait4/rusage.
    """
    if os.name != "nt":
        raise NotImplementedError("Use wait4/rusage for POSIX peak memory")
    with process._tree_lock:
        job = process._job
        if job is None:
            raise OSError("Process Job has already been closed")
        limits = _ExtendedLimits()
        if not _query_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return limits.PeakProcessMemory


def terminate_process_tree(process: subprocess.Popen) -> None:
    """Stop an owned invocation even when its original leader has already exited.

    Unmanaged POSIX callers must have created a new session. An unmanaged Windows
    process is terminated via taskkill only while its leader is still alive;
    use popen_isolated wherever descendant cleanup after leader exit is required.
    """
    close = getattr(process, "close_process_tree", None)
    if close is not None:
        close()
        return
    if os.name == "nt":
        if process.poll() is None:
            result = subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                                    capture_output=True, timeout=15)
            if result.returncode and process.poll() is None:
                raise OSError(f"taskkill failed for process {process.pid}: {result.returncode}")
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


_PROCESS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$items = @(Get-CimInstance Win32_Process | ForEach-Object {
    [pscustomobject]@{pid=[int]$_.ProcessId; ppid=[int]$_.ParentProcessId; name=[string]$_.Name;
        command=[string]$_.CommandLine;
        start=$(if ($null -ne $_.CreationDate) { $_.CreationDate.ToUniversalTime().ToString('o') } else { '' });
        state='running'}
})
ConvertTo-Json -InputObject $items -Compress -Depth 3
"""


def processes_info() -> list[dict]:
    """Snapshot all processes; inaccessible metadata is unknown, never absent."""
    try:
        if os.name == "nt":
            executable = shutil.which("powershell.exe")
            if not executable:
                raise OSError("Windows PowerShell is required to inspect process identity")
            result = subprocess.run([executable, "-NoLogo", "-NoProfile", "-NonInteractive",
                                     "-Command", _PROCESS_SCRIPT], capture_output=True,
                                    encoding="utf-8-sig", errors="strict", timeout=20,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode:
                raise OSError(f"Process inspection failed: {result.stderr.strip()}")
            rows = json.loads(result.stdout)
            if not isinstance(rows, list):
                raise ValueError("Process snapshot must be a JSON array")
            seen = set()
            for row in rows:
                if (not isinstance(row, dict) or
                        any(type(row.get(key)) is not int for key in ("pid", "ppid")) or
                        any(not isinstance(row.get(key), str) for key in ("name", "command", "start", "state")) or
                        row["pid"] < 0 or row["pid"] in seen):
                    raise ValueError("Malformed process snapshot")
                seen.add(row["pid"])
            return rows
        # macOS truncates a non-final comm column to 16 characters (including
        # an executable's path). ucomm supplies its basename, as on Linux and
        # Windows; -ww keeps the final command line independent of display width.
        name_field = "ucomm" if sys.platform == "darwin" else "comm"
        result = subprocess.run(["ps", "-ww", "-A", "-o", f"pid=,ppid=,lstart=,stat=,{name_field}=,args="],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=20, env={**os.environ, "LC_ALL": "C"})
        if result.returncode:
            raise OSError(f"Process inspection failed: {result.stderr.strip()}")
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split(None, 9)
            if len(fields) < 9:
                raise ValueError("Incomplete process snapshot")
            rows.append({"pid": int(fields[0]), "ppid": int(fields[1]),
                         "start": " ".join(fields[2:7]), "state": fields[7],
                         "name": fields[8], "command": fields[9] if len(fields) > 9 else ""})
        return rows
    except (subprocess.SubprocessError, ValueError, UnicodeError) as error:
        raise OSError(f"Could not inspect processes: {error}") from error


def process_info(pid: int) -> dict | None:
    """Return None only for a positively absent PID; unknown identity raises."""
    pid = int(pid)
    if pid <= 0:
        return None
    for item in processes_info():
        if item.get("pid") == pid:
            if not item.get("start") or not item.get("command"):
                raise OSError(f"Identity of process {pid} is inaccessible")
            return item
    return None


def venv_python(venv_dir) -> Path:
    """Path to this platform's interpreter in a virtual environment."""
    root = Path(venv_dir)
    return root / "Scripts" / "python.exe" if os.name == "nt" else root / "bin" / "python"


def is_link(path) -> bool:
    """Recognize symlinks and Windows reparse points (including junctions)."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def iter_files_no_links(root, pattern: str = "*"):
    """Walk regular files without entering symlink/junction directories.

    Propagate filesystem errors rather than claiming an incomplete scan is complete.
    Parent directories above ``root`` remain the caller's responsibility.
    """
    root = Path(root)
    if is_link(root):
        raise ValueError(f"Scan root cannot be a symlink/reparse point: {root}")
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False, onerror=fail):
        base = Path(directory)
        dirs[:] = [name for name in dirs if not is_link(base / name)]
        for name in files:
            path = base / name
            if path.match(pattern) and not is_link(path) and stat.S_ISREG(os.lstat(path).st_mode):
                yield path


def find_executable(name: str, env_var: str | None = None, candidates=()) -> str:
    """Find an executable, respecting a strict explicit environment override.

    Never split command lines or invoke a shell. An override is one executable
    path/name; an invalid override is an error rather than a silent fallback.
    """
    def resolve(value):
        value = os.path.expandvars(os.path.expanduser(os.fspath(value)))
        found = shutil.which(value)
        if found:
            return str(Path(found).resolve())
        return None

    if env_var and os.environ.get(env_var):
        found = resolve(os.environ[env_var])
        if not found:
            raise FileNotFoundError(f"{env_var} does not identify an executable: {os.environ[env_var]}")
        return found
    for value in (name, *candidates):
        if found := resolve(value):
            return found
    raise FileNotFoundError(f"Executable not found: {name}" + (f"; set {env_var}" if env_var else ""))


def resolve_godot() -> str:
    override = next((name for name in ("GODOT_BIN", "GODOT", "GODOT_PATH") if os.environ.get(name)), "GODOT_BIN")
    candidates = ["godot4", "Godot", "/Applications/Godot.app/Contents/MacOS/Godot",
                  "Godot_v4.6-stable_win64.exe", "Godot_v4.5-stable_win64.exe"]
    for root in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles"), "D:/Tools", "D:/Applications"):
        if root:
            base = Path(root)
            candidates.extend((base / "Godot" / "Godot.exe", base / "Godot" / "godot.exe"))
    return find_executable("godot", override, candidates)


def resolve_chrome() -> str:
    override = next((name for name in ("CHROME_BIN", "CHROME_PATH") if os.environ.get(name)), "CHROME_BIN")
    candidates = ["google-chrome", "chromium", "chromium-browser", "msedge"]
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        if root := os.environ.get(variable):
            candidates.extend((Path(root) / "Google/Chrome/Application/chrome.exe",
                               Path(root) / "Microsoft/Edge/Application/msedge.exe"))
    candidates.extend(("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                       "/Applications/Chromium.app/Contents/MacOS/Chromium"))
    return find_executable("chrome", override, candidates)
