"""Nullain Agent SDK — Permission Policy Engine."""

import re
from enum import StrEnum

from pydantic import BaseModel, Field


class PermissionLevel(StrEnum):
    """Permission evaluation levels for tool actions."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionPolicy(BaseModel):
    """Configurable security permission policy."""

    workspace_root: str
    default_read_level: PermissionLevel = PermissionLevel.ALLOW
    default_write_level: PermissionLevel = PermissionLevel.ASK
    default_exec_level: PermissionLevel = PermissionLevel.ASK

    deny_patterns: list[str] = Field(
        default_factory=lambda: [
            # Destructive filesystem/disk commands.
            r"rm\s+-rf",
            r"\bdd\s+.*of=",
            r"\bmkfs(\.\w+)?\b",
            r">\s*/dev/(sd|nvme|hd)\w*",
            r"\bchmod\s+(-R\s+)?777\b",
            r"\bchown\s+(-R\s+)?.*\s+/\s*$",
            # Remote-code-execution-by-pipe: fetch a script and hand it
            # straight to a shell with no chance to review it first.
            r"(curl|wget)\s+.*\|\s*(sh|bash|zsh|python[0-9.]*)\b",
            # Destructive/history-rewriting git operations.
            r"git\s+push\s+--force",
            r"git\s+reset\s+--hard",
            r"git\s+clean\s+-[a-z]*f",
            # Credential/secret files — .env and common SSH/cloud/package-
            # manager credential locations, not just the two SSH key types
            # this list previously covered. Explicitly excludes the
            # .env.example / .env.sample / .env.template convention (a safe,
            # secret-free documentation file every project with a real .env
            # tends to also have) via a negative lookahead — a plain r"\.env"
            # substring match denied those too, which was never the intent.
            r"\.env(?!\.(example|sample|template))(\.\w+)?$",
            r"id_rsa",
            r"id_ed25519",
            r"id_ecdsa",
            r"\.pem$",
            r"\.pfx$",
            r"\.npmrc$",
            r"\.pypirc$",
            r"\.netrc$",
            r"\.aws[/\\]credentials",
            r"\.ssh[/\\]config",
        ]
    )

    def evaluate_command(self, cmd_args: list[str]) -> PermissionLevel:
        """Evaluate permission level for executing a command."""
        full_cmd = " ".join(cmd_args)
        for pattern in self.deny_patterns:
            if re.search(pattern, full_cmd, re.IGNORECASE):
                return PermissionLevel.DENY
        return self.default_exec_level

    def evaluate_file_access(self, file_path: str, is_write: bool = False) -> PermissionLevel:
        """Evaluate permission level for accessing a file path."""
        for pattern in self.deny_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return PermissionLevel.DENY

        if is_write:
            return self.default_write_level
        return self.default_read_level


__all__ = ["PermissionLevel", "PermissionPolicy"]
