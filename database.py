"""
DatabaseManager — SQLCipher-encrypted local storage for posture telemetry.
=========================================================================
Uses AES-256 on-the-fly decryption via ``PRAGMA key``.

Tables
------
sessions
    One row per monitoring run (start/end times, duration, average CVA).
events
    One row per detected posture-violation event, linked to a session.
"""

import os
import datetime

try:
    import sqlcipher3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    # Fallback: plain sqlite3 (no encryption) — useful for dev / testing
    import sqlite3  # type: ignore[no-redef]

DB_FILENAME = "posture_monitor.db"
DB_KEY = "secret_password"


class DatabaseManager:
    """Encrypted SQLite database for posture-monitoring telemetry."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)
        self._db_path = db_path
        self._conn = sqlite3.connect(self._db_path)
        # Decrypt on the fly with AES-256
        self._conn.execute(f"PRAGMA key='{DB_KEY}';")
        self._create_tables()

    # ── DDL ────────────────────────────────────────────────────────────

    def _create_tables(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time  TEXT    NOT NULL,
                end_time    TEXT    NOT NULL,
                duration    REAL   NOT NULL,
                avg_cva     REAL   NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                start_time      TEXT    NOT NULL,
                end_time        TEXT    NOT NULL,
                duration        REAL   NOT NULL,
                event_type      TEXT    NOT NULL,
                deviation_value REAL   NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────

    def save_session(
        self,
        start_time: str,
        end_time: str,
        duration: float,
        avg_cva: float,
        events: list[dict],
    ) -> int:
        """
        Persist a complete monitoring session with all its events in one
        transaction.

        Parameters
        ----------
        start_time, end_time : str
            ISO-8601 timestamps.
        duration : float
            Session length in seconds.
        avg_cva : float
            Mean craniovertebral angle over the session.
        events : list[dict]
            Each dict must have keys: ``start_time``, ``end_time``,
            ``duration``, ``event_type``, ``deviation_value``.

        Returns
        -------
        int
            The auto-generated ``session_id``.
        """
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN;")
            cur.execute(
                """
                INSERT INTO sessions (start_time, end_time, duration, avg_cva)
                VALUES (?, ?, ?, ?);
                """,
                (start_time, end_time, duration, avg_cva),
            )
            session_id = cur.lastrowid

            if events:
                cur.executemany(
                    """
                    INSERT INTO events
                        (session_id, start_time, end_time, duration, event_type, deviation_value)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            session_id,
                            ev["start_time"],
                            ev["end_time"],
                            ev["duration"],
                            ev["event_type"],
                            ev["deviation_value"],
                        )
                        for ev in events
                    ],
                )
            self._conn.commit()
            return session_id  # type: ignore[return-value]
        except Exception:
            self._conn.rollback()
            raise

    # ── Read (analytics) ──────────────────────────────────────────────

    def get_all_sessions(self) -> list[dict]:
        """Return every session row as a list of dicts."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT session_id, start_time, end_time, duration, avg_cva "
            "FROM sessions ORDER BY session_id DESC;"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_events_for_session(self, session_id: int) -> list[dict]:
        """Return all events belonging to *session_id*."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT event_id, session_id, start_time, end_time, duration, "
            "event_type, deviation_value FROM events WHERE session_id = ?;",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_event_summary(self) -> dict[str, int]:
        """
        Aggregate event counts across **all** sessions, grouped by
        ``event_type``.

        Returns
        -------
        dict
            ``{event_type: count}``, e.g.
            ``{'bad_neck': 5, 'bad_lean': 2, ...}``
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT event_type, COUNT(*) AS cnt FROM events GROUP BY event_type;"
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_posture_time_ratio(self) -> dict[str, float]:
        """
        Return total *good* and *bad* time across all sessions.

        ``bad_time`` is the sum of event durations; ``good_time`` is total
        session duration minus ``bad_time``.

        Returns
        -------
        dict
            ``{'good_time': float, 'bad_time': float}`` in seconds.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COALESCE(SUM(duration), 0) FROM sessions;")
        total_duration = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(duration), 0) FROM events;")
        total_bad = cur.fetchone()[0]

        good = max(0.0, total_duration - total_bad)
        return {"good_time": good, "bad_time": total_bad}

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
