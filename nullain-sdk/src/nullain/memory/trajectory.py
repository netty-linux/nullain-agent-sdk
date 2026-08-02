"""Nullain Agent SDK — Trajectory Recording and Episodic Memory."""

import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel, Field


class TrajectoryRecord(BaseModel, frozen=True):
    """Immutable record of an agent execution trajectory for episodic learning."""

    session_id: str = Field(description="Unique session identifier")
    intent: str = Field(description="Classified intent name")
    model: str = Field(description="Model used for execution")
    steps_count: int = Field(description="Total execution steps taken")
    success: bool = Field(description="Whether execution met acceptance criteria")
    objective: str = Field(description="Original prompt or goal objective")
    repo_fingerprint: str = Field(default="", description="Optional repository fingerprint")
    user_feedback: str | None = Field(default=None, description="Optional user feedback string")


class EpisodicMemory:
    """SQLite-backed episodic memory storage for past agent trajectories."""

    def __init__(self, db_path: str | Path = "~/.nullain/memory.db") -> None:
        """Initialize EpisodicMemory with database path."""
        self.db_path = Path(db_path).expanduser()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database parent directories and initialize table schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = sqlite3.Row
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                intent TEXT NOT NULL,
                model TEXT NOT NULL,
                steps_count INTEGER NOT NULL,
                success INTEGER NOT NULL,
                objective TEXT NOT NULL,
                repo_fingerprint TEXT NOT NULL,
                user_feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.commit()

    async def record_trajectory(self, record: TrajectoryRecord) -> None:
        """Record a completed trajectory into SQLite storage."""
        if self._db is None:
            await self.initialize()

        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO trajectories (
                session_id, intent, model, steps_count,
                success, objective, repo_fingerprint, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.intent,
                record.model,
                record.steps_count,
                1 if record.success else 0,
                record.objective,
                record.repo_fingerprint,
                record.user_feedback,
            ),
        )
        await self._db.commit()

    async def get_relevant_examples(self, intent: str, limit: int = 2) -> list[TrajectoryRecord]:
        """Query successful trajectories matching intent as few-shot examples."""
        if self._db is None:
            await self.initialize()

        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT session_id, intent, model, steps_count,
                   success, objective, repo_fingerprint, user_feedback
            FROM trajectories
            WHERE intent = ? AND success = 1
            ORDER BY id DESC
            LIMIT ?
            """,
            (intent, limit),
        )
        rows: list[sqlite3.Row] = list(await cursor.fetchall())
        await cursor.close()

        records: list[TrajectoryRecord] = []
        for row in rows:
            row_dict: dict[str, Any] = dict(row)
            uf = row_dict["user_feedback"]
            records.append(
                TrajectoryRecord(
                    session_id=str(row_dict["session_id"]),
                    intent=str(row_dict["intent"]),
                    model=str(row_dict["model"]),
                    steps_count=int(row_dict["steps_count"]),
                    success=bool(row_dict["success"]),
                    objective=str(row_dict["objective"]),
                    repo_fingerprint=str(row_dict["repo_fingerprint"]),
                    user_feedback=str(uf) if uf is not None else None,
                )
            )
        return records

    async def close(self) -> None:
        """Close SQLite database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None


__all__ = ["EpisodicMemory", "TrajectoryRecord"]
