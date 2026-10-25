"""Native inspection of one already-owned Job, including nested-job descendants."""
import ctypes
from ctypes import wintypes as W


class Memory(ctypes.Structure):
    _fields_ = [('cb', W.DWORD), ('faults', W.DWORD)] + [(name, ctypes.c_size_t) for name in
        ('peak_ws', 'rss', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]


class JobAPI:
    def __init__(self, job_handle, ws_module):
        if not job_handle:
            raise ValueError('An existing OWNED Job handle is required')
        self.job, self.ws = job_handle, ws_module.Native()
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        definitions = {
            'IsProcessInJob': (W.BOOL, [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)]),
            'QueryInformationJobObject': (W.BOOL, [W.HANDLE, ctypes.c_int, W.LPVOID, W.DWORD, ctypes.POINTER(W.DWORD)]),
            'K32GetProcessMemoryInfo': (W.BOOL, [W.HANDLE, ctypes.POINTER(Memory), W.DWORD]),
        }
        for name, (result, args) in definitions.items():
            api = getattr(self.k, name)
            api.restype, api.argtypes = result, args

    def members(self):
        # No PPID traversal: Job lists include nested members after a leader exits.
        size = 8 + 64 * ctypes.sizeof(ctypes.c_size_t)
        buffer, returned = ctypes.create_string_buffer(size), W.DWORD()
        if not self.k.QueryInformationJobObject(self.job, 3, buffer, size, ctypes.byref(returned)):
            raise ctypes.WinError(ctypes.get_last_error())
        assigned, present = (W.DWORD * 2).from_buffer(buffer)
        if assigned > 64 or present > 64 or assigned != present:
            raise RuntimeError('Owned Job membership overflow; fail closed')
        return list((ctypes.c_size_t * present).from_buffer(buffer, 8))

    def open(self, pid):
        return self.ws.open(pid)

    def close(self, handle):
        self.ws.close(handle)

    def exited(self, handle):
        """Positive exit proof from the ORIGINAL process handle; errors are not proof."""
        code = W.DWORD()
        self.ws.checked(self.ws.k.GetExitCodeProcess(handle, ctypes.byref(code)))
        return code.value != 259

    def identity(self, handle):
        inside = W.BOOL()
        if not self.k.IsProcessInJob(handle, self.job, ctypes.byref(inside)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not inside.value:
            raise ValueError('Process is not in this OWNED outer Job')
        try:
            return self.ws.identity(handle)
        except ValueError as error:
            if str(error) == 'Target exited; refuse any setter':
                raise ProcessLookupError('Owned member exited during sample') from error
            raise

    def memory(self, handle):
        self.identity(handle)
        info = Memory()
        info.cb = ctypes.sizeof(info)
        if not self.k.K32GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'rss': info.rss, 'private': info.private, 'page_faults': info.faults}

    def policy(self, handle):
        self.identity(handle)
        return self.ws.policy(handle)

    def set(self, handle, minimum, maximum):
        self.identity(handle)  # Original setter handle must still be in the OWNED Job.
        return self.ws.set(handle, minimum, maximum)
