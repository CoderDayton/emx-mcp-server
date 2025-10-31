"""SQLite-based graph storage for temporal relationships."""

import sqlite3
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class GraphStore:
    """SQLite graph database for event relationships."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "events.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Create tables for events and relationships."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
      CREATE TABLE IF NOT EXISTS events (
      event_id TEXT PRIMARY KEY,
      timestamp REAL,
      token_count INTEGER,
      access_count INTEGER DEFAULT 0,
      metadata TEXT
    )
    """
        )
        cursor.execute(
            """
      CREATE TABLE IF NOT EXISTS relationships (
      from_event TEXT,
      to_event TEXT,
      relationship_type TEXT,
      lag INTEGER,
      FOREIGN KEY (from_event) REFERENCES events(event_id),
      FOREIGN KEY (to_event) REFERENCES events(event_id)
    )
    """
        )
        cursor.execute(
            """
      CREATE INDEX IF NOT EXISTS idx_from_event
      ON relationships(from_event)
    """
        )
        cursor.execute(
            """
      CREATE INDEX IF NOT EXISTS idx_to_event
      ON relationships(to_event)
    """
        )
        self.conn.commit()

    def add_event(
        self,
        event_id: str,
        timestamp: float,
        token_count: int,
        metadata: Optional[str] = None,
    ):
        """Add event node to graph."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO events VALUES (?, ?, ?, 0, ?)",
            (event_id, timestamp, token_count, metadata),
        )
        self.conn.commit()

    def remove_event(self, event_id: str):
        """Remove event and its relationships."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM relationships WHERE from_event = ? OR to_event = ?",
            (event_id, event_id),
        )
        cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
        self.conn.commit()

    def increment_access(self, event_id: str):
        """Increment access count for event."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE events SET access_count = access_count + 1 WHERE event_id = ?",
            (event_id,),
        )
        self.conn.commit()

    def link_events(
        self, from_id: str, to_id: str, relationship: str = "PRECEDES", lag: int = 1
    ):
        """Create temporal relationship between events."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO relationships VALUES (?, ?, ?, ?)",
            (from_id, to_id, relationship, lag),
        )
        self.conn.commit()

    def get_neighbors(
        self, event_id: str, max_distance: int = 3, bidirectional: bool = True
    ) -> List[str]:
        """Get temporally adjacent events."""
        cursor = self.conn.cursor()
        # Increment access count
        self.increment_access(event_id)
        # Forward neighbors
        cursor.execute(
            """
      SELECT to_event FROM relationships
      WHERE from_event = ? AND lag <= ?
      ORDER BY lag
    """,
            (event_id, max_distance),
        )
        forward = [row[0] for row in cursor.fetchall()]
        if not bidirectional:
            return forward
        # Backward neighbors
        cursor.execute(
            """
      SELECT from_event FROM relationships
      WHERE to_event = ? AND lag <= ?
      ORDER BY lag
    """,
            (event_id, max_distance),
        )
        backward = [row[0] for row in cursor.fetchall()]
        return backward + [event_id] + forward

    def get_least_accessed_events(self, limit: int = 100) -> List[str]:
        """Get least accessed events for pruning."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
      SELECT event_id FROM events
      ORDER BY access_count ASC, timestamp ASC
      LIMIT ?
    """,
            (limit,),
        )
        return [row[0] for row in cursor.fetchall()]

    def count_events(self) -> int:
        """Get total event count."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]

    def clear(self):
        """Delete all events and relationships."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM relationships")
        cursor.execute("DELETE FROM events")
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()
