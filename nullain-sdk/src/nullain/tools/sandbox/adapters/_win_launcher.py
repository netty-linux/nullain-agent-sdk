"""Nullain Agent SDK — Windows sandbox launcher (standalone, no nullain
imports).

Invoked by :class:`~nullain.tools.sandbox.adapters.windows_job.WindowsJobSandbox`
as ``python -m nullain.tools.sandbox.adapters._win_launcher <cfg_json>``.
Dispatches on ``deny_network`` to one of two real, distinct confinement
mechanisms — see ``main()`` — rather than one mechanism trying to cover both:

**deny_network=True** (``_main_deny_network`` / ``_launch_in_app_container``):
real AppContainer confinement.

1. Creates (or reuses) an AppContainer profile scoped to this launch —
   ``CreateAppContainerProfile`` — with NO capabilities attached. An
   AppContainer process with no capabilities cannot open a TCP/UDP socket at
   all: Windows' firewall/WFP layer enforces this for every AppContainer
   process unconditionally, independent of anything the process itself does.
   Proven live: a denied child's ``socket.connect()`` raises
   ``WinError 10013`` (WSAEACCES) before ever reaching the network.
2. Grants the AppContainer SID read+write ACL access on the workspace
   directory (and any ``allow_paths``) via ``SetNamedSecurityInfoW`` — an
   AppContainer process's low-box token means *no* file access is implied by
   default, even to files the launching user owns; without an explicit grant
   the child cannot open anything, including its own workspace. Critically,
   this MERGES into the object's existing DACL (fetched via
   ``GetNamedSecurityInfoW`` first) rather than replacing it — an earlier
   version passed ``OldAcl=None`` to ``SetEntriesInAclW``, which silently
   stripped SYSTEM/Administrators/owner access from the granted directory,
   found live by reproducing a write failure for the *launching user itself*
   after granting, no sandboxed child involved at all.
3. Builds a primary token for the AppContainer via ``CreateProcessW``'s
   ``PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES`` extended attribute (the
   standard way to launch a process *into* an AppContainer — there is no
   separate "AppContainer token" to open() the way a restricted token works).
4. Creates a Job Object with ``KILL_ON_JOB_CLOSE``, launches
   ``CREATE_SUSPENDED``, assigns atomically, resumes — same process
   containment as v1, avoiding the assign-after-start breakaway race.
5. Waits for the child, forwards its exit code, releases the ACL grant and
   deletes the AppContainer profile (best-effort; a leaked per-launch
   profile is harmless — Windows namespaces by name).

**deny_network=False** (``_main_allow_network`` / ``_launch_with_restricted_token``):
plain restricted-token launch, no AppContainer. Confirmed live that
AppContainer's capability grant (``INTERNET_CLIENT`` / attaching the
well-known S-1-15-3-1/2 capability SIDs to the token) does NOT actually
restore network access for an ad-hoc AppContainer created at runtime —
Windows' WFP integration only honors network capabilities for a process
belonging to a properly *registered* MSIX/APPX package, which a loose
Python script launched this way is not. Since AppContainer's only
load-bearing property here is "no capabilities ⇒ no socket, guaranteed", and
that guarantee is exactly what's NOT wanted when the caller allowed
network, this path skips AppContainer and reproduces the original v1
mechanism instead: ``CreateRestrictedToken(DISABLE_MAX_PRIVILEGES)`` (no
elevation, but normal filesystem + network access) inside the same Job
Object for process containment.

If setup fails for any reason on either path (AppContainer profile/SID/ACL,
or restricted token/job), the launcher dies non-zero rather than falling
back to an unconfined or partially-confined launch — a misconfigured host
surfaces as a failed run, never silent unsandboxed execution.

This module imports only the stdlib + ctypes, so it can be shipped and
invoked without pulling the rest of the SDK into the sandboxed child's
parent. It only runs on Windows; it is imported nowhere at module load on
other platforms.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import uuid
from ctypes import wintypes as w

if sys.platform != "win32":  # pragma: no cover - never imported off Windows
    raise RuntimeError("the Windows sandbox launcher only runs on Windows")

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_adv = ctypes.WinDLL("advapi32", use_last_error=True)
_userenv = ctypes.WinDLL("userenv", use_last_error=True)

# --- job object -----------------------------------------------------------------

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9

# --- process creation ------------------------------------------------------------

CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_NO_WINDOW = 0x08000000
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
STARTF_USESTDHANDLES = 0x00000100
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12
INFINITE = 0xFFFFFFFF

# --- restricted token (deny_network=False fallback; see _launch_with_restricted_token) --

TOKEN_QUERY = 0x0008
TOKEN_DUPLICATE = 0x0002
TOKEN_ASSIGN_PRIMARY = 0x0001
DISABLE_MAX_PRIVILEGES = 0x1

# --- security / ACL ---------------------------------------------------------------

SE_FILE_OBJECT = 1
DACL_SECURITY_INFORMATION = 0x00000004
GRANT_ACCESS = 1
NO_MULTIPLE_TRUSTEE = 0
TRUSTEE_IS_SID = 0
TRUSTEE_IS_UNKNOWN = 0
SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
FILE_ALL_ACCESS = 0x1F01FF
ERROR_SUCCESS = 0

# HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS) — CreateAppContainerProfile returns
# this when a stale profile survives a prior crashed launcher.
_ALREADY_EXISTS_HRESULT = -2147024713  # 0x800700B7

PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009


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


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _StartupInfo),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", w.HANDLE),
        ("hThread", w.HANDLE),
        ("dwProcessId", w.DWORD),
        ("dwThreadId", w.DWORD),
    ]


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", w.DWORD),
    ]


class _SecurityCapabilities(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.POINTER(_SidAndAttributes)),
        ("CapabilityCount", w.DWORD),
        ("Reserved", w.DWORD),
    ]


class _Trustee(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),  # actually a SID* here (TRUSTEE_IS_SID)
    ]


class _ExplicitAccess(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", w.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", w.DWORD),
        ("Trustee", _Trustee),
    ]


def _die(code: int, what: str) -> None:
    err = ctypes.get_last_error()
    sys.stderr.write(f"[nullain sandbox launcher] {what} failed (WinError {err})\n")
    sys.exit(code)


# --- AppContainer profile + SID ---------------------------------------------------


def _create_app_container(profile_name: str) -> ctypes.c_void_p:
    """Create (or open, if it already exists from a crashed prior run) an
    AppContainer profile and return its SID. The SID is what everything else
    — the security-capabilities struct handed to CreateProcessAsUserW, and
    the ACL grant on the workspace — is keyed on."""
    _userenv.CreateAppContainerProfile.argtypes = [
        w.LPCWSTR,
        w.LPCWSTR,
        w.LPCWSTR,
        ctypes.c_void_p,
        w.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _userenv.CreateAppContainerProfile.restype = ctypes.c_long

    sid_ptr = ctypes.c_void_p()
    hr = _userenv.CreateAppContainerProfile(
        profile_name,
        "Nullain Sandbox",
        "Nullain Agent SDK per-run sandbox container",
        None,
        0,
        ctypes.byref(sid_ptr),
    )
    # A stale profile from a prior crashed launcher. Derive its SID instead
    # of failing; the profile itself carries no state we care about
    # (capabilities are supplied fresh on every launch via
    # SECURITY_CAPABILITIES, not stored on the profile).
    if hr == _ALREADY_EXISTS_HRESULT:
        _userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
            w.LPCWSTR,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
        hr2 = _userenv.DeriveAppContainerSidFromAppContainerName(
            profile_name, ctypes.byref(sid_ptr)
        )
        if hr2 != 0:
            _die(2, f"DeriveAppContainerSidFromAppContainerName (hr={hr2})")
    elif hr != 0:
        _die(2, f"CreateAppContainerProfile (hr={hr})")
    return sid_ptr


def _delete_app_container(profile_name: str) -> None:
    """Best-effort cleanup; failure here must never fail the launch/exit code
    — a leaked profile is harmless (Windows namespaces by name, and the next
    launch with the same name just re-derives the same SID)."""
    _userenv.DeleteAppContainerProfile.argtypes = [w.LPCWSTR]
    _userenv.DeleteAppContainerProfile.restype = ctypes.c_long
    _userenv.DeleteAppContainerProfile(profile_name)


# --- ACL grant on the workspace -----------------------------------------------


def _grant_appcontainer_access(path: str, sid: ctypes.c_void_p) -> None:
    """Grant the AppContainer SID full access to ``path`` (and everything
    beneath it) via SetNamedSecurityInfoW.

    An AppContainer token's default DACL implies NO access to anything the
    launching user owns, including the workspace the parent process created
    for the child to work in — the child cannot open ANY file until it is
    explicitly granted here. Everything not explicitly granted (via this
    function, called once per workspace_root/allow_path) stays inaccessible;
    that default-deny is what makes AppContainer real filesystem isolation
    rather than a convention the tool layer has to also enforce (#41).

    Critically, this MERGES the new ACE into the existing DACL (fetched via
    ``GetNamedSecurityInfoW`` and passed as ``SetEntriesInAclW``'s ``OldAcl``)
    rather than replacing it outright. Passing ``None`` for the old ACL
    builds a DACL with ONLY the new entry — found live: that silently
    stripped SYSTEM/Administrators/the object owner's own access from the
    workspace directory, so the *launching process itself* could no longer
    write there afterward (a regression an escape/allow test pair alone
    would not have caught, since both denied-outside and allowed-inside
    cases still "worked" by accident on a shallow test path — confirmed by
    reproducing the failure with the launching user's own unsandboxed write,
    no child process or AppContainer involved at all).
    """
    _adv.GetNamedSecurityInfoW.argtypes = [
        w.LPWSTR,
        ctypes.c_int,
        w.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _adv.GetNamedSecurityInfoW.restype = w.DWORD
    existing_sd = ctypes.c_void_p()
    existing_dacl = ctypes.c_void_p()
    err = _adv.GetNamedSecurityInfoW(
        path,
        SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(existing_dacl),
        None,
        ctypes.byref(existing_sd),
    )
    if err != ERROR_SUCCESS:
        _die(2, f"GetNamedSecurityInfoW({path}) (err={err})")

    try:
        trustee = _Trustee(
            pMultipleTrustee=None,
            MultipleTrusteeOperation=NO_MULTIPLE_TRUSTEE,
            TrusteeForm=TRUSTEE_IS_SID,
            TrusteeType=TRUSTEE_IS_UNKNOWN,
            ptstrName=sid,
        )
        ea = _ExplicitAccess(
            grfAccessPermissions=FILE_ALL_ACCESS,
            grfAccessMode=GRANT_ACCESS,
            grfInheritance=SUB_CONTAINERS_AND_OBJECTS_INHERIT,
            Trustee=trustee,
        )

        new_dacl = ctypes.c_void_p()
        _adv.SetEntriesInAclW.argtypes = [
            w.ULONG,
            ctypes.POINTER(_ExplicitAccess),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        _adv.SetEntriesInAclW.restype = w.DWORD
        err = _adv.SetEntriesInAclW(1, ctypes.byref(ea), existing_dacl, ctypes.byref(new_dacl))
        if err != ERROR_SUCCESS:
            _die(2, f"SetEntriesInAclW({path}) (err={err})")

        try:
            _adv.SetNamedSecurityInfoW.argtypes = [
                w.LPWSTR,
                ctypes.c_int,
                w.DWORD,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            _adv.SetNamedSecurityInfoW.restype = w.DWORD
            err = _adv.SetNamedSecurityInfoW(
                path,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                new_dacl,
                None,
            )
            if err != ERROR_SUCCESS:
                _die(2, f"SetNamedSecurityInfoW({path}) (err={err})")
        finally:
            _k32.LocalFree(new_dacl)
    finally:
        # existing_dacl points INSIDE existing_sd's buffer (per
        # GetNamedSecurityInfoW's contract) — only the security descriptor
        # itself is freed, never existing_dacl separately.
        _k32.LocalFree(existing_sd)


# --- process launch inside the AppContainer ---------------------------------------


def _launch_in_app_container(
    cmd: list[str],
    ac_sid: ctypes.c_void_p,
    job: w.HANDLE,
) -> int:
    """Launch ``cmd`` inside the AppContainer identified by ``ac_sid``, with
    NO capabilities attached — used only for ``deny_network=True``.

    Capabilities are deliberately never granted here, even though the
    SECURITY_CAPABILITIES struct supports it: confirmed live that Windows'
    firewall/WFP layer only honors network capabilities for a process
    belonging to a properly *registered* MSIX/APPX package. An AppContainer
    created ad-hoc at runtime via CreateAppContainerProfile — which is all
    a loose Python script can do — never gets network access back no matter
    what capability SIDs are attached to its token; every attempt reproduced
    WinError 10013 (WSAEACCES) regardless. Since AppContainer's real,
    load-bearing property here is "no capabilities => no socket, guaranteed",
    and the network-allow case doesn't need AppContainer's confinement at
    all, deny_network=False instead uses the plain restricted-token launch
    (_launch_with_restricted_token) — see main()'s dispatch.
    """
    sec_caps = _SecurityCapabilities(
        AppContainerSid=ac_sid,
        Capabilities=None,
        CapabilityCount=0,
        Reserved=0,
    )

    attr_list = _build_attribute_list(sec_caps)

    si = _StartupInfoEx()
    si.StartupInfo.cb = ctypes.sizeof(si)
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
    si.StartupInfo.hStdInput = _k32.GetStdHandle(STD_INPUT_HANDLE)
    si.StartupInfo.hStdOutput = _k32.GetStdHandle(STD_OUTPUT_HANDLE)
    si.StartupInfo.hStdError = _k32.GetStdHandle(STD_ERROR_HANDLE)
    si.lpAttributeList = attr_list

    pi = _ProcessInformation()
    cmdline = subprocess.list2cmdline(cmd)

    _k32.CreateProcessW.argtypes = [
        w.LPCWSTR,
        w.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        w.BOOL,
        w.DWORD,
        ctypes.c_void_p,
        w.LPCWSTR,
        ctypes.POINTER(_StartupInfoEx),
        ctypes.POINTER(_ProcessInformation),
    ]
    _k32.CreateProcessW.restype = w.BOOL
    flags = (
        CREATE_SUSPENDED
        | CREATE_UNICODE_ENVIRONMENT
        | CREATE_NO_WINDOW
        | EXTENDED_STARTUPINFO_PRESENT
    )
    ok = _k32.CreateProcessW(
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
        _die(2, f"CreateProcessW/AppContainer ({ctypes.get_last_error()})")

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
    _free_attribute_list(attr_list)
    return code.value


# --- process launch with a plain restricted token (deny_network=False) ------------
#
# AppContainer's network-capability grant only works for a process belonging
# to a registered MSIX/APPX package (confirmed live — see
# _launch_in_app_container's docstring), so when the caller does NOT want
# network denied, launching inside an AppContainer would just make the
# process fail to reach the network it was allowed to use, with no way to
# fix that from a loose script. This path skips AppContainer entirely and
# reproduces the original v1 mechanism: a token with ALL privileges disabled
# (CreateRestrictedToken(DISABLE_MAX_PRIVILEGES)) — no elevation, but the
# same filesystem access as the launching user (including normal network
# access, since nothing here restricts it) — still wrapped in the same Job
# Object for process containment.


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


def _launch_with_restricted_token(cmd: list[str], h_token: w.HANDLE, job: w.HANDLE) -> int:
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


def _build_attribute_list(sec_caps: _SecurityCapabilities) -> ctypes.c_void_p:
    _k32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        w.DWORD,
        w.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _k32.InitializeProcThreadAttributeList.restype = w.BOOL

    size = ctypes.c_size_t(0)
    _k32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    ok = _k32.InitializeProcThreadAttributeList(buf, 1, 0, ctypes.byref(size))
    if not ok:
        _die(2, "InitializeProcThreadAttributeList")

    _k32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        w.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _k32.UpdateProcThreadAttribute.restype = w.BOOL
    ok = _k32.UpdateProcThreadAttribute(
        buf,
        0,
        PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
        ctypes.byref(sec_caps),
        ctypes.sizeof(sec_caps),
        None,
        None,
    )
    if not ok:
        _die(2, "UpdateProcThreadAttribute(SECURITY_CAPABILITIES)")
    # Keep the buffer (and the struct/caps arrays it points into) alive for
    # the duration of CreateProcessW by stashing references on the return
    # value; ctypes has no automatic lifetime tracking across this boundary.
    buf._sec_caps_keepalive = sec_caps  # type: ignore[attr-defined]
    return ctypes.cast(buf, ctypes.c_void_p)


def _free_attribute_list(attr_list: ctypes.c_void_p) -> None:
    _k32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    _k32.DeleteProcThreadAttributeList.restype = None
    _k32.DeleteProcThreadAttributeList(attr_list)


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


def _main_deny_network(cmd: list[str], grant_paths: list[str]) -> int:
    """deny_network=True path: real AppContainer confinement — no
    capabilities means the child cannot open a socket, and only the granted
    paths are accessible on disk."""
    # Unique per-launch profile name (max 64 chars) so concurrent sandboxed
    # runs never collide on the same AppContainer / ACL state.
    profile_name = f"nullain-sbx-{uuid.uuid4().hex[:24]}"
    ac_sid = _create_app_container(profile_name)
    try:
        for path in grant_paths:
            _grant_appcontainer_access(path, ac_sid)

        job = _build_job()
        try:
            return _launch_in_app_container(cmd, ac_sid, job)
        finally:
            _k32.CloseHandle(job)
    finally:
        _delete_app_container(profile_name)


def _main_allow_network(cmd: list[str]) -> int:
    """deny_network=False path: plain restricted-token launch (no
    AppContainer) — see _launch_with_restricted_token's module comment for
    why AppContainer isn't used here."""
    h_token = _build_restricted_token()
    try:
        job = _build_job()
        try:
            return _launch_with_restricted_token(cmd, h_token, job)
        finally:
            _k32.CloseHandle(job)
    finally:
        _k32.CloseHandle(h_token)


def main() -> None:
    cfg = json.loads(sys.argv[1])
    cmd: list[str] = cfg["cmd"]
    deny_network: bool = cfg.get("deny_network", True)
    grant_paths: list[str] = cfg.get("grant_paths", [])

    exit_code = _main_deny_network(cmd, grant_paths) if deny_network else _main_allow_network(cmd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
