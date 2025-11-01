"""Tool for importing/exporting project memory to portable archives."""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def transfer_memory(
    manager,
    action: str,
    path: str,
    merge: bool = False,
    expected_tokens: Optional[int] = None,
) -> dict:
    """
    Import or export project memory to/from portable archive files.

    Use this to:
    - Share memory between machines or team members
    - Backup memory before risky operations
    - Migrate memory to a new project
    - Merge memories from multiple sources

    Export creates a .tar.gz archive containing all project memory.
    Import loads memory from an archive (optionally merging with existing).
    When importing, provide expected_tokens hint for optimal FAISS nlist.

    Args:
        manager: ProjectMemoryManager instance
        action: "export" or "import"
        path: File path for archive (.tar.gz)
        merge: For import only - merge with existing memory (vs replace)
        expected_tokens: For import - hint for optimal nlist calculation

    Returns:
        Transfer status, file info, event counts, and index health
    """
    logger.info(f"Memory transfer: action={action}, path={path}, merge={merge}")

    if action == "export":
        result = manager.export_memory(path)
        return {
            "status": "success",
            "action": "export",
            "path": result["path"],
            "size_bytes": result["size_bytes"],
            "size_mb": round(result["size_bytes"] / 1024 / 1024, 2),
            "events_exported": manager.get_stats()["project_events"],
        }

    elif action == "import":
        # Update expected vector count before import if provided
        if expected_tokens:
            expected_events = expected_tokens // 30
            expected_vectors = expected_events * 27

            vs = manager.project_store.vector_store
            vs.expected_vector_count = expected_vectors
            min_training = int(expected_vectors * 0.9)
            vs.min_training_size = min_training
            vs.nlist = vs._calculate_optimal_nlist(expected_vectors)

            logger.info(
                f"Pre-import index optimization: expected {expected_vectors} vectors → "
                f"nlist={vs.nlist}, training at {min_training}"
            )

        result = manager.import_memory(path, merge=merge)

        # Get index health after import
        index_info = manager.get_index_info()
        total_vecs = index_info.get("total_vectors", 0)
        current_nlist = index_info.get("nlist", 0)
        optimal_nlist = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
        nlist_ratio = current_nlist / optimal_nlist if optimal_nlist > 0 else 0.0

        return {
            "status": "success",
            "action": "import",
            "path": path,
            "merge": merge,
            "events_imported": result["events"],
            "total_events_now": manager.get_stats()["project_events"],
            "index_health": {
                "nlist": current_nlist,
                "optimal_nlist": optimal_nlist,
                "nlist_ratio": nlist_ratio,
                "status": (
                    "optimal"
                    if nlist_ratio >= 0.85
                    else "acceptable" if nlist_ratio >= 0.5 else "suboptimal"
                ),
                "recommendation": (
                    "Index optimal"
                    if nlist_ratio >= 0.85
                    else f"Consider retraining with expected_tokens={expected_tokens or 'hint'}"
                ),
            },
        }

    else:
        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "available_actions": ["export", "import"],
        }
