"""Nullain Agent SDK — LSP response models (M11.2).

LSP server output is untrusted, so it is validated through Pydantic at the
boundary exactly like MCP and LLM output (AGENTS.md rule 3). These models cover
the subset of LSP 3.17 response shapes the four read-only tools consume:
pull-diagnostics, definition, references, and hover. Fields the tools do not
use are omitted; unknown fields are ignored (``extra="ignore"``).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field


class Position(BaseModel):
    """A zero-based line/character position in a document."""

    line: int
    character: int


class Range(BaseModel):
    """A zero-based range between two positions."""

    start: Position
    end: Position


class Location(BaseModel):
    """A location: a document URI plus a range."""

    uri: str
    range: Range


class LocationLink(BaseModel):
    """A location link (definition result variant with a target URI).

    The LSP wire format uses camelCase field names, so the Python fields are
    snake_case with explicit aliases to keep the protocol bytes correct.
    """

    target_uri: str = Field(alias="targetUri")
    target_range: Range = Field(alias="targetRange")


class Diagnostic(BaseModel):
    """A single diagnostic item (severity per the LSP enum)."""

    range: Range
    severity: int | None = None
    message: str
    source: str | None = None


class DiagnosticReport(BaseModel):
    """The pull-diagnostics result (``textDocument/diagnostic``, LSP 3.17)."""

    kind: str | None = None
    items: list[Diagnostic] = Field(default_factory=list[Diagnostic])


class MarkupContent(BaseModel):
    """A hover ``MarkupContent`` (``{kind, value}``)."""

    kind: str
    value: str


class Hover(BaseModel):
    """A hover result: contents as plain text, markup, or a list of either."""

    contents: str | MarkupContent | list[str | MarkupContent] | None = None


#: A definition/references result: a single location/link, a list, or null.
#: ``Sequence`` (covariant) so a ``list[Location]`` from references is accepted.
LocationResult = Location | LocationLink | Sequence[Location | LocationLink] | None


__all__ = [
    "Diagnostic",
    "DiagnosticReport",
    "Hover",
    "Location",
    "LocationLink",
    "LocationResult",
    "MarkupContent",
    "Position",
    "Range",
]
