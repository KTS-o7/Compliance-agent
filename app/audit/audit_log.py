from __future__ import annotations
import sqlite3
import json
import os
from app.models.schemas import AuditEntry


class AuditLog:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get("AUDIT_DB_PATH", "data/audit.db")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript_id           TEXT NOT NULL,
                    rules_evaluated         TEXT NOT NULL,
                    verdicts                TEXT NOT NULL,
                    model_used              TEXT NOT NULL,
                    latency_seconds         REAL NOT NULL,
                    timestamp               TEXT NOT NULL,
                    user_role               TEXT NOT NULL,
                    corrective_disagreement INTEGER NOT NULL DEFAULT 0
                )
            """)

    def append(self, entry: AuditEntry) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO audit_log
                   (transcript_id, rules_evaluated, verdicts, model_used,
                    latency_seconds, timestamp, user_role, corrective_disagreement)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.transcript_id,
                    json.dumps(entry.rules_evaluated),
                    json.dumps([v.model_dump() for v in entry.verdicts]),
                    entry.model_used,
                    entry.latency_seconds,
                    entry.timestamp,
                    entry.user_role,
                    int(entry.corrective_disagreement),
                )
            )

    def read_all(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
