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
            r"rm\s+-rf",
            r"git\s+push\s+--force",
            r"git\s+reset\s+--hard",
            r"\.env",
            r"id_rsa",
            r"id_ed25519",
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
