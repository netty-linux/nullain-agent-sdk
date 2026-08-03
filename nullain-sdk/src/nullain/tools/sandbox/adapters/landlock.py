"""Nullain Agent SDK — Linux Landlock sandbox adapter (pure ctypes, no new deps).

Landlock (Linux >=5.13) restricts a process's filesystem access at the kernel
level: after ``landlock_restrict_self`` the child can only touch paths we
explicitly allowed. Since 6.2, Landlock *scope* also lets us deny network
socket creation. This adapter applies both in a ``preexec_fn`` that runs in the
forked child before ``exec``.

Everything is done via raw syscalls through libc — no compiled extension, no
third-party package. The syscall numbers (444/445/446) come from the asm-generic
table shared by x86_64 and aarch64, the two arches CI runs on.

Honesty: this isolates filesystem (and network on >=6.2) for real — see the
gated escape test. It does NOT filter arbitrary syscalls (seccomp) and does not
restrict compute; that is out of scope for v1.
"""

from __future__ import annotations

import ctypes
import errno
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nullain.tools.sandbox.port import SandboxOptions

# --- libc -------------------------------------------------------------------

# use_errno=True so ctypes.get_errno() reflects the syscall's failure reason.
_libc = ctypes.CDLL(None, use_errno=True)
# syscall() is variadic; leave argtypes unset and pass c_long + per-arg values.
_libc.syscall.restype = ctypes.c_long

# --- landlock syscall numbers (asm-generic; same on x86_64 and aarch64) -------

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

# --- landlock constants -----------------------------------------------------

LANDLOCK_CREATE_RULESET_VERSION = 1

# Filesystem access rights (uapi/linux/landlock.h). The full set a v1/v2/v3
# kernel can *handle*; we restrict to whatever the running kernel advertises.
_ACCESS_FS = {
    "EXECUTE": 1 << 0,
    "WRITE_FILE": 1 << 1,
    "READ_FILE": 1 << 2,
    "READ_DIR": 1 << 3,
    "REMOVE_DIR": 1 << 4,
    "REMOVE_FILE": 1 << 5,
    "MAKE_CHAR": 1 << 6,
    "MAKE_DIR": 1 << 7,
    "MAKE_REG": 1 << 8,
    "MAKE_SOCK": 1 << 9,
    "MAKE_FIFO": 1 << 10,
    "MAKE_BLOCK": 1 << 11,
    "MAKE_SYM": 1 << 12,
    "REFER": 1 << 13,  # v2 (6.2): cross-directory rename
    "TRUNCATE": 1 << 14,  # v2 (6.2): truncate/open-with-trunc
}

# A read+write-everything mask (used for the workspace and allow_paths).
_RW_ALL = (
    _ACCESS_FS["WRITE_FILE"]
    | _ACCESS_FS["READ_FILE"]
    | _ACCESS_FS["READ_DIR"]
    | _ACCESS_FS["REMOVE_DIR"]
    | _ACCESS_FS["REMOVE_FILE"]
    | _ACCESS_FS["MAKE_DIR"]
    | _ACCESS_FS["MAKE_REG"]
    | _ACCESS_FS["MAKE_SYM"]
    | _ACCESS_FS["MAKE_FIFO"]
    | _ACCESS_FS["MAKE_SOCK"]
    | _ACCESS_FS["MAKE_BLOCK"]
    | _ACCESS_FS["MAKE_CHAR"]
    | _ACCESS_FS["EXECUTE"]
)
# Read+execute only (for system dirs the child must read to exec its binary).
_RO_ALL = _ACCESS_FS["READ_FILE"] | _ACCESS_FS["READ_DIR"] | _ACCESS_FS["EXECUTE"]

# Network scope (6.2+): deny TCP/UDP bind+connect by handling the scope.
LANDLOCK_SCOPE_NETWORK = 1 << 0

_PR_SET_NO_NEW_PRIVS = 38

# os.O_PATH / os.O_CLOEXEC are POSIX-only; use getattr so this module imports
# (and pyright type-checks) on Windows too. The runtime path only runs on Linux.
_O_PATH: int = getattr(os, "O_PATH", os.O_RDONLY)
_O_CLOEXEC: int = getattr(os, "O_CLOEXEC", 0)


# --- ctypes structs ---------------------------------------------------------


class _RulesetAttr(ctypes.Structure):
    # struct landlock_ruleset_attr { __u64 handled_access_fs; __u64 handled_access_scope; }
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_scope", ctypes.c_uint64),
    ]


class _PathBeneath(ctypes.Structure):
    # struct landlock_path_beneath_attr { __u64 allowed_access; __s32 parent_fd; }
    # 4 bytes of implicit tail padding make it 16 bytes; spell it out.
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


def _set_errno(result: int, what: str) -> int:
    if result < 0:
        err = ctypes.get_errno() or errno.EPERM
        raise OSError(err, f"{what} failed: {os.strerror(err)}")
    return result


def _probe_abi_version() -> int:
    """Return the Landlock ABI version the running kernel supports (1, 2, 3...),
    or 0 if Landlock is unavailable/disabled. Uses the version-query flag, which
    never requires privileges and is the supported way to detect support."""
    if os.name != "posix":
        return 0
    try:
        result = _libc.syscall(
            ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
            ctypes.c_void_p(0),  # NULL attr
            ctypes.c_size_t(0),
            ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError:
        return 0
    if result < 0:
        return 0
    return int(result)


def _max_fs_mask(abi: int) -> int:
    """The handled_access_fs mask this ABI understands. REFER/TRUNCATE land in
    ABI v2; older kernels reject masks with those bits set."""
    mask = _RW_ALL
    if abi < 2:
        mask &= ~(_ACCESS_FS["REFER"] | _ACCESS_FS["TRUNCATE"])
    return mask


class LandlockSandbox:
    """Linux Landlock sandbox: real kernel-enforced FS isolation (+ network
    scope on >=6.2). Applied in a preexec_fn so restrictions take effect in the
    child just before exec."""

    def __init__(self, required: bool = True) -> None:
        self._required = required

    @property
    def name(self) -> str:
        return "landlock"

    @property
    def required(self) -> bool:
        return self._required

    def available(self) -> bool:
        # Landlock is available iff the kernel advertises at least ABI v1. The
        # runner fail-closes on a required adapter that reports unavailable.
        return _probe_abi_version() >= 1

    def prepare(self, argv: Sequence[str], opts: SandboxOptions) -> dict[str, Any]:
        # Unavailable + not required: the runner already declined to fail-closed
        # (required=False), so the caller opted out of strict isolation. Returning
        # no confinement lets the child run rather than crashing inside preexec.
        abi = _probe_abi_version()
        if abi < 1:
            return {}
        fs_mask = _max_fs_mask(abi)
        deny_net = opts.deny_network and abi >= 2
        scope_mask = LANDLOCK_SCOPE_NETWORK if deny_net else 0
        exe_path = argv[0] if argv else ""

        def _preexec() -> None:
            self._apply(opts.workspace_root, opts.allow_paths, fs_mask, scope_mask, exe_path)

        return {"preexec_fn": _preexec}

    @staticmethod
    def _apply(
        workspace_root: Path,
        allow_paths: Sequence[Path],
        fs_mask: int,
        scope_mask: int,
        exe_path: str,
    ) -> None:
        # 1) Drop new privileges — required before restricting self.
        if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno() or errno.EPERM
            raise OSError(err, "prctl(PR_SET_NO_NEW_PRIVS) failed")

        # 2) Create a ruleset handling the FS accesses (and network scope).
        attr = _RulesetAttr(handled_access_fs=fs_mask, handled_access_scope=scope_mask)
        ruleset_fd = _set_errno(
            _libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
                ctypes.byref(attr),
                ctypes.c_size_t(ctypes.sizeof(attr)),
                ctypes.c_uint32(0),
            ),
            "landlock_create_ruleset",
        )
        try:
            # 3) Allow read+write to the workspace and any configured allow_paths.
            for path in [workspace_root, *allow_paths]:
                LandlockSandbox._add_path_rule(ruleset_fd, path, _RW_ALL & fs_mask, fs_mask)
            # 4) Allow read+execute on the system trees the child needs to exec
            #    its interpreter and load shared libs. Read-only everywhere else
            #    is *denied* by default — only these listed trees are readable.
            for sys_dir in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/dev", "/proc"):
                if os.path.exists(sys_dir):
                    LandlockSandbox._add_path_rule(ruleset_fd, sys_dir, _RO_ALL & fs_mask, fs_mask)
            # 5) Allow the interpreter's own directory tree (RO) so the child can
            #    exec its interpreter and read its venv wherever it lives — CI
            #    toolcache, pyenv, /usr/bin, a uv venv symlink, etc. We must allow
            #    BOTH the literal argv[0] tree (e.g. .venv/ holding pyvenv.cfg that
            #    site.py reads at startup) AND its realpath target tree (the actual
            #    binary + stdlib), because sys.executable is often a symlink.
            #    Without this the child fails to start, which would make the escape
            #    test pass for the wrong reason.
            if exe_path:
                allowed: set[str] = set()
                for cand in (exe_path, os.path.realpath(exe_path)):
                    d = os.path.dirname(cand)
                    for tree in (d, os.path.dirname(d)):
                        if tree and os.path.exists(tree) and tree not in allowed:
                            allowed.add(tree)
                            LandlockSandbox._add_path_rule(
                                ruleset_fd, tree, _RO_ALL & fs_mask, fs_mask
                            )
            # 6) Bind the ruleset to ourselves. After this, the restrictions are
            #    enforced for this process and all exec'd children.
            _set_errno(
                _libc.syscall(
                    ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_uint32(0),
                ),
                "landlock_restrict_self",
            )
        finally:
            os.close(ruleset_fd)

    @staticmethod
    def _add_path_rule(ruleset_fd: int, path: str | Path, allowed: int, fs_mask: int) -> None:
        if not allowed:
            return  # kernel older than the bit we want — skip rather than fail
        parent = os.open(os.fspath(path), _O_PATH | _O_CLOEXEC)
        try:
            beneath = _PathBeneath(allowed_access=allowed, parent_fd=parent)
            _set_errno(
                _libc.syscall(
                    ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
                    ctypes.c_int(ruleset_fd),
                    ctypes.c_int(1),  # LANDLOCK_RULE_PATH_BENEATH
                    ctypes.byref(beneath),
                    ctypes.c_uint32(0),
                ),
                f"landlock_add_rule({path})",
            )
        finally:
            os.close(parent)


__all__ = ["LandlockSandbox"]
