import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ParkingEvent:
    plate: str
    event_type: str
    timestamp: datetime
    allowed: bool
    reason: str
    rule_code: str
    vehicle_type: str
    access_category: str


class ParkingStorage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parking_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    rule_code TEXT NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    access_category TEXT NOT NULL
                )
                """
            )

    def log_event(self, event: ParkingEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO parking_events (plate, event_type, timestamp, allowed, reason, rule_code, vehicle_type, access_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.plate,
                    event.event_type,
                    event.timestamp.isoformat(),
                    int(event.allowed),
                    event.reason,
                    event.rule_code,
                    event.vehicle_type,
                    event.access_category,
                ),
            )

    def get_last_entry(self, plate: str):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM parking_events
                WHERE plate = ? AND event_type = 'entry' AND allowed = 1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (plate,),
            ).fetchone()
        return row

    def list_events(self, limit: int = 100):
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM parking_events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return rows
