"""
Characterization test for temporal link integrity with UUID-based event IDs.

Validates that:
1. Event IDs follow UUID format (event_<32 hex chars>)
2. Temporal PRECEDES relationships exist between sequential events
3. Graph database contains correct event ID references (no orphaned links)
4. Metadata persistence tracks last_event_id correctly
"""

import sqlite3

from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config


def test_temporal_chain_with_uuid_event_ids(tmp_path):
    """
    Verify temporal linking works with UUID-based event IDs.

    Regression test for bug where event IDs were timestamp-based in
    ProjectMemoryManager but counter-based in HierarchicalMemoryStore,
    causing "Event not found" warnings during retrieval.

    After fix: Both layers use UUID format (event_<uuid4_hex>).
    """
    config = load_config()

    # Initialize manager with persistent storage
    manager = ProjectMemoryManager(
        project_path=str(tmp_path / "project"),
        global_path=str(tmp_path / "global"),
        config=config,
    )

    # Create 5 sequential events
    event_ids = []
    for i in range(5):
        tokens = [f"token_{j}" for j in range(i * 10, (i + 1) * 10)]
        result = manager.add_event(tokens, embeddings=None, metadata={"test_sequence": i})
        event_ids.append(result["event_id"])

    # Flush any buffered events to ensure they're persisted
    manager.flush_events()

    # Validate event ID format (event_<32 hex chars>)
    for event_id in event_ids:
        assert event_id.startswith("event_"), f"Event ID missing prefix: {event_id}"
        uuid_part = event_id.replace("event_", "")
        assert len(uuid_part) == 32, f"UUID part wrong length: {len(uuid_part)} (expected 32)"
        assert all(c in "0123456789abcdef" for c in uuid_part), f"Invalid hex in UUID: {uuid_part}"

    # Check graph database contains correct relationships
    graph_db_path = tmp_path / "project" / ".memories" / "graph_db" / "events.db"
    assert graph_db_path.exists(), "Graph database not created"

    conn = sqlite3.connect(graph_db_path)
    cursor = conn.cursor()

    # Verify 4 PRECEDES relationships (n-1 for n events)
    cursor.execute("SELECT COUNT(*) FROM relationships WHERE relationship_type='PRECEDES'")
    rel_count = cursor.fetchone()[0]
    assert rel_count == 4, f"Expected 4 PRECEDES links, found {rel_count}"

    # Verify temporal chain matches actual event IDs
    cursor.execute(
        """
        SELECT from_event, to_event
        FROM relationships
        WHERE relationship_type='PRECEDES'
        ORDER BY rowid
    """
    )

    temporal_links = cursor.fetchall()
    assert len(temporal_links) == 4, "Missing temporal links"

    for i, (from_id, to_id) in enumerate(temporal_links):
        expected_from = event_ids[i]
        expected_to = event_ids[i + 1]

        assert from_id == expected_from, (
            f"Link {i}: from_event mismatch. Got {from_id}, expected {expected_from}"
        )
        assert to_id == expected_to, (
            f"Link {i}: to_event mismatch. Got {to_id}, expected {expected_to}"
        )

    # Verify all event IDs exist in events table (no orphaned relationships)
    cursor.execute("SELECT event_id FROM events ORDER BY timestamp")
    db_event_ids = [row[0] for row in cursor.fetchall()]

    assert db_event_ids == event_ids, "Event IDs in graph DB don't match created events"

    conn.close()


def test_metadata_persistence_tracks_last_event_id(tmp_path):
    """
    Verify metadata.json correctly persists last_event_id across sessions.

    This enables temporal linking to resume after process restart without
    relying on event counters or timestamps.
    """
    config = load_config()
    project_path = str(tmp_path / "project")
    global_path = str(tmp_path / "global")

    # Session 1: Create 2 events
    manager1 = ProjectMemoryManager(project_path, global_path, config)

    result1 = manager1.add_event(["token_a", "token_b"], embeddings=None, metadata={})
    _ = result1["event_id"]

    result2 = manager1.add_event(["token_c", "token_d"], embeddings=None, metadata={})
    event_id_2 = result2["event_id"]

    # Flush buffered events to ensure metadata is persisted
    manager1.flush_events()

    # Check metadata contains last_event_id
    metadata_path = tmp_path / "project" / ".memories" / "metadata.json"
    assert metadata_path.exists(), "Metadata file not created"

    import json

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "last_event_id" in metadata, "last_event_id not in metadata"
    assert metadata["last_event_id"] == event_id_2, (
        f"last_event_id should be {event_id_2}, got {metadata['last_event_id']}"
    )

    # Session 2: Load existing manager and add event
    manager2 = ProjectMemoryManager(project_path, global_path, config)

    result3 = manager2.add_event(["token_e", "token_f"], embeddings=None, metadata={})
    event_id_3 = result3["event_id"]

    # Flush to persist the event and its temporal link
    manager2.flush_events()

    # Verify temporal link from event_2 → event_3
    graph_db_path = tmp_path / "project" / ".memories" / "graph_db" / "events.db"
    conn = sqlite3.connect(graph_db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT from_event, to_event
        FROM relationships
        WHERE relationship_type='PRECEDES'
        AND from_event = ?
    """,
        (event_id_2,),
    )

    link = cursor.fetchone()
    assert link is not None, f"No temporal link from {event_id_2}"

    from_id, to_id = link
    assert from_id == event_id_2, f"Link from_event mismatch: {from_id}"
    assert to_id == event_id_3, f"Link to_event mismatch: {to_id}"

    conn.close()


def test_no_temporal_links_on_first_event(tmp_path):
    """First event should not create temporal relationship (no predecessor)."""
    config = load_config()

    manager = ProjectMemoryManager(
        project_path=str(tmp_path / "project"),
        global_path=str(tmp_path / "global"),
        config=config,
    )

    # Create single event
    _ = manager.add_event(["first", "event"], embeddings=None, metadata={})

    # Check no relationships exist
    graph_db_path = tmp_path / "project" / ".memories" / "graph_db" / "events.db"
    conn = sqlite3.connect(graph_db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM relationships")
    count = cursor.fetchone()[0]

    assert count == 0, f"First event should not create temporal links, found {count}"

    conn.close()
