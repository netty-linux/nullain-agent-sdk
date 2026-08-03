"""Nullain Agent SDK — Windows Job Object sandbox launcher (standalone, no nullain imports).

Invoked by :class:`~nullain.tools.sandbox.adapters.windows_job.WindowsJobSandbox` as
``python -m nullain.tools.sandbox.adapters._win_launcher <cfg_json>``. It:

1. Builds a restricted token from the current process token with **all privileges
   disabled** (``CreateRestrictedToken(DISABLE_MAX_PRIVILEGES)``). Integrity is
   unchanged (medium), so the child can still write to the workspace the parent
   created — low-integrity confinement (which would also require labeling the
   workspace) is a follow-up.
2. Creates a Job Object with ``KILL_ON_JOB_CLOSE`` so the whole process tree is
   torn down if the launcher (or its parent) dies — process containment.
3. Launches the real command with the restricted token, ``CREATE_SUSPENDED``,
   inheriting the launcher's stdio handles, atomically assigns it to the job,
   and resumes it. This avoids the assign-after-start race that would let the
   child spawn a breakaway process before confinement is applied.
4. Waits for the child and forwards its exit code.

This module imports only the stdlib + ctypes, so it can be shipped and invoked
without pulling the rest of the SDK into the sandboxed child's parent. It only
runs on Windows; it is imported nowhere at module load on other platforms.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from ctypes import wintypes as w

if sys.platform != "win32":  # pragma: no cover - never imported off Windows
    raise RuntimeError("the Windows sandbox launcher only runs on Windows")

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_adv = ctypes.WinDLL("advapi32", use_last_error=True)

# --- token access rights / flags ------------------------------------------------

TOKEN_QUERY = 0x0008
TOKEN_DUPLICATE = 0x0002
TOKEN_ASSIGN_PRIMARY = 0x0001
DISABLE_MAX_PRIVILEGES = 0x1

# --- job object -----------------------------------------------------------------

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9

# --- process creation ------------------------------------------------------------

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_NO_WINDOW = 0x08000000
STARTF_USESTDHANDLES = 0x00000100
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12
INFINITE = 0xFFFFFFFF


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", w.LARGE_INTEGER),
        ("PerJobUserTimeLimit", w.LARGE_INTEGER),
        ("LimitFlags", w.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", w.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", w.DWORD),
        ("SchedulingClass", w.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("lpReserved", w.LPWSTR),
        ("lpDesktop", w.LPWSTR),
        ("lpTitle", w.LPWSTR),
        ("dwX", w.DWORD),
        ("dwY", w.DWORD),
        ("dwXSize", w.DWORD),
        ("dwYSize", w.DWORD),
        ("dwXCountChars", w.DWORD),
        ("dwYCountChars", w.DWORD),
        ("dwFillAttribute", w.DWORD),
        ("dwFlags", w.DWORD),
        ("wShowWindow", w.WORD),
        ("cbReserved2", w.WORD),
        ("lpReserved2", ctypes.POINTER(w.BYTE)),
        ("hStdInput", w.HANDLE),
        ("hStdOutput", w.HANDLE),
        ("hStdError", w.HANDLE),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", w.HANDLE),
        ("hThread", w.HANDLE),
        ("dwProcessId", w.DWORD),
        ("dwThreadId", w.DWORD),
    ]


def _die(code: int, what: str) -> None:
    err = ctypes.get_last_error()
    sys.stderr.write(f"[nullain sandbox launcher] {what} failed (WinError {err})\n")
    sys.exit(code)


def _build_restricted_token() -> w.HANDLE:
    h_token = w.HANDLE()
    _adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    _adv.OpenProcessToken.restype = w.BOOL
    if not _adv.OpenProcessToken(
        _k32.GetCurrentProcess(),
        TOKEN_QUERY | TOKEN_DUPLICATE | TOKEN_ASSIGN_PRIMARY,
        ctypes.byref(h_token),
    ):
        _die(2, "OpenProcessToken")

    h_restricted = w.HANDLE()
    _adv.CreateRestrictedToken.argtypes = [
        w.HANDLE,
        w.DWORD,
        w.DWORD,
        ctypes.c_void_p,
        w.DWORD,
        ctypes.c_void_p,
        w.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(w.HANDLE),
    ]
    _adv.CreateRestrictedToken.restype = w.BOOL
    ok = _adv.CreateRestrictedToken(
        h_token, DISABLE_MAX_PRIVILEGES, 0, None, 0, None, 0, None, ctypes.byref(h_restricted)
    )
    _k32.CloseHandle(h_token)
    if not ok:
        _die(2, "CreateRestrictedToken")
    return h_restricted


def _build_job() -> w.HANDLE:
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    _k32.CreateJobObjectW.restype = w.HANDLE
    job = _k32.CreateJobObjectW(None, None)
    if not job:
        _die(2, "CreateJobObject")

    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    _k32.SetInformationJobObject.argtypes = [
        w.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        w.DWORD,
    ]
    _k32.SetInformationJobObject.restype = w.BOOL
    if not _k32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        _k32.CloseHandle(job)
        _die(2, "SetInformationJobObject")
    return job


def _launch_in_job(cmd: list[str], h_token: w.HANDLE, job: w.HANDLE) -> int:
    si = _StartupInfo()
    si.cb = ctypes.sizeof(si)
    si.dwFlags = STARTF_USESTDHANDLES
    si.hStdInput = _k32.GetStdHandle(STD_INPUT_HANDLE)
    si.hStdOutput = _k32.GetStdHandle(STD_OUTPUT_HANDLE)
    si.hStdError = _k32.GetStdHandle(STD_ERROR_HANDLE)
    pi = _ProcessInformation()

    cmdline = subprocess.list2cmdline(cmd)
    _adv.CreateProcessAsUserW.argtypes = [
        w.HANDLE,
        w.LPCWSTR,
        w.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        w.BOOL,
        w.DWORD,
        ctypes.c_void_p,
        w.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _adv.CreateProcessAsUserW.restype = w.BOOL
    flags = CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW
    ok = _adv.CreateProcessAsUserW(
        h_token,
        None,
        cmdline,
        None,
        None,
        True,
        flags,
        None,
        None,
        ctypes.byref(si),
        ctypes.byref(pi),
    )
    if not ok:
        _die(2, f"CreateProcessAsUserW ({ctypes.get_last_error()})")

    _k32.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    _k32.AssignProcessToJobObject.restype = w.BOOL
    if not _k32.AssignProcessToJobObject(job, pi.hProcess):
        _k32.TerminateProcess(pi.hProcess, 2)
        _die(2, "AssignProcessToJobObject")

    _k32.ResumeThread.argtypes = [w.HANDLE]
    _k32.ResumeThread.restype = w.DWORD
    _k32.ResumeThread(pi.hThread)

    _k32.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    _k32.WaitForSingleObject.restype = w.DWORD
    _k32.WaitForSingleObject(pi.hProcess, INFINITE)

    code = w.DWORD()
    _k32.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    _k32.GetExitCodeProcess.restype = w.BOOL
    _k32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    _k32.CloseHandle(pi.hThread)
    _k32.CloseHandle(pi.hProcess)
    return code.value


def main() -> None:
    cfg = json.loads(sys.argv[1])
    cmd: list[str] = cfg["cmd"]
    h_token = _build_restricted_token()
    try:
        job = _build_job()
        try:
            exit_code = _launch_in_job(cmd, h_token, job)
        finally:
            _k32.CloseHandle(job)
    finally:
        _k32.CloseHandle(h_token)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
