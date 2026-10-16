"""Async SQLite repository for Spark-Foresight."""

from __future__ import annotations

import os
import aiosqlite
from typing import Optional, Any, AsyncIterator
from pathlib import Path
from contextlib import asynccontextmanager

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "foresight.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class DatabaseRepository:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = str(db_path)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open an aiosqlite connection with foreign keys enabled."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            yield db

    async def initialize(self) -> None:
        """Create database schema if not already initialized."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        async with self.connect() as db:
            await db.executescript(schema_sql)
            await db.commit()

    async def create_session(
        self,
        session_id: str,
        title: str = "Untitled Meeting",
        started_at: Optional[int] = None,
    ) -> dict[str, Any]:
        """Create a new recording and transcription session."""
        import time

        if started_at is None:
            started_at = int(time.time() * 1000)

        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO sessions (id, title, started_at, status)
                VALUES (?, ?, ?, 'recording')
                """,
                (session_id, title, started_at),
            )
            await db.commit()

        return {
            "id": session_id,
            "title": title,
            "started_at": started_at,
            "status": "recording",
        }

    async def stop_session(
        self,
        session_id: str,
        ended_at: Optional[int] = None,
    ) -> bool:
        """Mark a session as stopped."""
        import time

        if ended_at is None:
            ended_at = int(time.time() * 1000)

        async with self.connect() as db:
            cursor = await db.execute(
                """
                UPDATE sessions
                SET ended_at = ?, status = 'stopped'
                WHERE id = ?
                """,
                (ended_at, session_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Retrieve session details by ID."""
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions ordered by start time descending."""
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def add_transcript_segment(
        self,
        session_id: str,
        start_ms: int,
        end_ms: int,
        text: str,
        speaker_channel: str = "mixed",
        speaker_label: str = "Speaker",
        confidence: float = 1.0,
    ) -> int:
        """Insert a single timestamped transcription segment."""
        async with self.connect() as db:
            cursor = await db.execute(
                """
                INSERT INTO transcript_segments (
                    session_id, start_ms, end_ms, speaker_channel, speaker_label, text, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    start_ms,
                    end_ms,
                    speaker_channel,
                    speaker_label,
                    text.strip(),
                    confidence,
                ),
            )
            await db.commit()
            return cursor.lastrowid

    async def get_transcript_segments(
        self,
        session_id: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve transcript segments for a session, optionally filtered by time range."""
        query = "SELECT * FROM transcript_segments WHERE session_id = ?"
        params: list[Any] = [session_id]

        if start_ms is not None:
            query += " AND end_ms >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND start_ms <= ?"
            params.append(end_ms)

        query += " ORDER BY start_ms ASC"

        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def get_temporal_window(
        self,
        session_id: str,
        center_ms: int,
        window_before_ms: int = 45000,
        window_after_ms: int = 15000,
    ) -> list[dict[str, Any]]:
        """
        Slices the conversation window [center_ms - 45s, center_ms + 15s].
        Used by the Shorthand Engine to correlate typed notes to what was spoken.
        """
        lower_bound = max(0, center_ms - window_before_ms)
        upper_bound = center_ms + window_after_ms
        return await self.get_transcript_segments(
            session_id=session_id, start_ms=lower_bound, end_ms=upper_bound
        )

    async def save_note(
        self,
        session_id: str,
        note_id: str,
        raw_text: str,
        created_at_ms: int,
        enriched_text: Optional[str] = None,
        status: str = "draft",
    ) -> None:
        """Insert or replace a note for a session."""
        import time

        now_ms = int(time.time() * 1000)
        async with self.connect() as db:
            await db.execute(
                """
                INSERT INTO notes (id, session_id, created_at_ms, updated_at_ms, raw_text, enriched_text, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at_ms = excluded.updated_at_ms,
                    raw_text = excluded.raw_text,
                    enriched_text = excluded.enriched_text,
                    status = excluded.status
                """,
                (
                    note_id,
                    session_id,
                    created_at_ms,
                    now_ms,
                    raw_text,
                    enriched_text,
                    status,
                ),
            )
            await db.commit()

    async def get_notes(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve notes for a session ordered by creation offset."""
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM notes WHERE session_id = ? ORDER BY created_at_ms ASC",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
