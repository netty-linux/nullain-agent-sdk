"""Temporary diagnostic: isolates which seatbelt profile rule causes the
SIGABRT-with-empty-output failure seen on real macOS CI (issue #42's PR).
Builds a sequence of increasingly-complete profiles and reports each
result in one CI run, instead of costing one full CI round-trip per
hypothesis. Delete once #42's real fix lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _esc(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


@pytest.mark.skipif(sys.platform != "darwin", reason="diagnostic is macOS-only")
@pytest.mark.asyncio
async def test_diagnose_seatbelt_profile_stages(tmp_path: Path) -> None:
    import asyncio

    async def run(profile: str, label: str) -> None:
        script = "print('OK')"
        proc = await asyncio.create_subprocess_exec(
            "/usr/bin/sandbox-exec",
            "-p",
            profile,
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        print(f"\n=== {label} ===")
        print(f"returncode={proc.returncode}")
        print(f"output={stdout.decode('utf-8', errors='replace')!r}")

    venv_bin = f"{sys.prefix}/bin" if hasattr(sys, "base_prefix") else ""
    workspace = str(tmp_path)

    # Stage 0: totally unconfined baseline (sanity: sandbox-exec itself works).
    await run("(version 1)\n(allow default)", "stage0-allow-default")

    # Stage 1: deny default, allow only process-fork/exec/signal (no file
    # rules at all) — expect the interpreter to fail (can't read its own
    # binary), but NOT abort/crash; a clean nonzero exit with a normal
    # "Permission denied"/"Operation not permitted" execvp error.
    await run(
        "\n".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process-fork)",
                "(allow process-exec)",
                "(allow signal)",
            ]
        ),
        "stage1-no-file-rules",
    )

    # Stage 2: add ONLY the workspace read grant.
    await run(
        "\n".join(
            [
                "(version 1)",
                "(deny default)",
                "(allow process-fork)",
                "(allow process-exec)",
                "(allow signal)",
                "(deny file-read*)",
                f'(allow file-read* (subpath "{_esc(workspace)}"))',
            ]
        ),
        "stage2-workspace-read-only",
    )

    # Stage 3: add the venv bin tree read grant (this is argv[0]'s dir).
    if venv_bin:
        await run(
            "\n".join(
                [
                    "(version 1)",
                    "(deny default)",
                    "(allow process-fork)",
                    "(allow process-exec)",
                    "(allow signal)",
                    "(deny file-read*)",
                    f'(allow file-read* (subpath "{_esc(workspace)}"))',
                    f'(allow file-read* (subpath "{_esc(venv_bin)}"))',
                ]
            ),
            "stage3-plus-venv-bin",
        )

    # Stage 4: add the bootstrap system paths one at a time.
    bootstrap = [
        "/usr/lib",
        "/System/Library/Frameworks",
        "/System/Library/PrivateFrameworks",
        "/private/var/db/dyld",
        "/dev",
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal)",
        "(deny file-read*)",
        f'(allow file-read* (subpath "{_esc(workspace)}"))',
    ]
    if venv_bin:
        lines.append(f'(allow file-read* (subpath "{_esc(venv_bin)}"))')
    for path in bootstrap:
        lines.append(f'(allow file-read* (subpath "{_esc(path)}"))')
        await run("\n".join(lines), f"stage4-plus-{path}")

    # Stage 5: full profile via the real _build_profile, for comparison.
    sys.path.insert(0, "nullain-sdk/src")
    from nullain.tools.sandbox.adapters import seatbelt as sb_mod

    full_profile = sb_mod._build_profile(  # type: ignore[reportPrivateUsage]
        workspace,
        [],
        sb_mod._interpreter_read_paths([sys.executable]),  # type: ignore[reportPrivateUsage]
        deny_network=True,
    )
    await run(full_profile, "stage5-real-build-profile")

    # Always "fail" so the diagnostic output prints in CI's failure summary
    # even though pytest normally hides captured stdout on success.
    pytest.fail("diagnostic complete — see captured stdout above for stage results")
