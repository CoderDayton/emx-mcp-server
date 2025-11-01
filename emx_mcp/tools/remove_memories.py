"""Tool for removing specific memories from project memory."""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def remove_memories(
    manager,
    event_ids: list[str],
    confirm: bool = False,
) -> dict:
    """
    Remove specific memories (events) from project memory.

    Use this to selectively delete memories you no longer need, freeing up
    storage and improving search relevance. Unlike manage_memory(action="clear")
    which deletes everything, this removes only the specified events.

    The removal is permanent and affects all storage backends:
    - Vector store (FAISS embeddings)
    - Graph store (temporal relationships)
    - Disk/JSON storage (event data)

    Args:
        manager: ProjectMemoryManager instance
        event_ids: List of event IDs to remove (get these from recall_memories)
        confirm: Safety flag - must be True to execute deletion

    Returns:
        Removal status, count of removed events, and updated system stats
    """
    logger.info(f"Remove memories request: {len(event_ids)} events, confirm={confirm}")

    if not confirm:
        return {
            "status": "error",
            "error": "Destructive operation requires confirm=True",
            "example": "Set confirm=True to proceed with deletion",
            "event_ids_to_remove": event_ids,
            "count": len(event_ids),
        }

    if not event_ids:
        return {
            "status": "error",
            "error": "No event_ids provided",
            "example": "Provide a list of event IDs from recall_memories()",
        }

    # Remove events from all storage backends
    try:
        result = manager.project_store.remove_events(event_ids)

        # Get updated stats
        stats = manager.get_stats()
        index_info = manager.get_index_info()

        return {
            "status": "success",
            "removed_count": result["removed_count"],
            "attempted_count": len(event_ids),
            "event_ids": event_ids,
            "remaining_events": stats["project_events"],
            "index_health": {
                "trained": index_info.get("is_trained", False),
                "total_vectors": index_info.get("total_vectors", 0),
                "nlist": index_info.get("nlist", 0),
            },
            "recommendation": (
                "Consider retraining index if you removed many events"
                if result["removed_count"] > 100
                else "Memory updated successfully"
            ),
        }

    except Exception as e:
        logger.error(f"Failed to remove events: {e}")
        return {
            "status": "error",
            "error": str(e),
            "event_ids": event_ids,
        }
