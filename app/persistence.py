"""SQLite-backed checkpointing, long-term store setup, and session metadata."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


SESSIONS_TABLE = "sessions"


@dataclass(frozen=True)
class Session:
    thread_id: str
    title: str
    created_at: str


class SessionRepository:
    """Owns session metadata and the checkpoint database connection."""

    def __init__(self, database_path: Path):
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._connection)
        self.checkpointer.setup()
        self._connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def list_sessions(self) -> list[Session]:
        rows = self._connection.execute(
            f"SELECT thread_id, title, created_at FROM {SESSIONS_TABLE} ORDER BY created_at DESC"
        ).fetchall()
        return [Session(*row) for row in rows]

    def create_session(self, thread_id: str, title: str) -> None:
        self._connection.execute(
            f"INSERT INTO {SESSIONS_TABLE} (thread_id, title, created_at) VALUES (?, ?, ?)",
            (thread_id, title, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        self._connection.commit()

    def delete_session(self, thread_id: str) -> None:
        self._connection.execute(f"DELETE FROM {SESSIONS_TABLE} WHERE thread_id = ?", (thread_id,))
        self._connection.commit()
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        self._connection.close()


def create_store(database_path: Path):
    """Return the SqliteStore context manager used for cross-session memory."""
    return SqliteStore.from_conn_string(str(database_path))
