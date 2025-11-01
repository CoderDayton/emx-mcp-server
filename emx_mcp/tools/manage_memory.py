"""Tool for memory system maintenance and diagnostics."""

from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


def manage_memory(
    manager,
    action: str,
    options: Optional[dict[Any, Any]] = None,
) -> dict:
    """
    Administrative operations for memory system maintenance and diagnostics.

    Think of this as system maintenance: retraining search indexes, pruning
    old data, checking health status, or clearing everything. These operations
    don't add or retrieve memories - they manage the system itself.

    Available actions:
    - "stats": Get detailed memory statistics and index health with nlist optimization info
    - "retrain": Rebuild IVF search index with optimal nlist (use after adding many events)
    - "optimize": Prune least-accessed events and compress storage
    - "clear": Delete all project memory (requires confirm=True in options)
    - "estimate": Calculate expected vector count from corpus size hint

    Args:
        manager: ProjectMemoryManager instance
        action: Operation to perform (stats, retrain, optimize, clear, estimate)
        options: Action-specific options:
            - "force": true for retrain
            - "confirm": true for clear
            - "expected_tokens": int for estimate/retrain

    Returns:
        Operation results and updated system status
    """
    logger.info(f"Memory management: action={action}")
    options = options or {}

    if action == "stats":
        stats = manager.get_stats()
        index_info = manager.get_index_info()

        # Calculate optimal nlist for comparison
        total_vecs = index_info.get("total_vectors", 0)
        optimal_nlist = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
        current_nlist = index_info.get("nlist", 0)
        nlist_ratio = current_nlist / optimal_nlist if optimal_nlist > 0 else 0.0

        return {
            "status": "success",
            "action": "stats",
            "project_events": stats["project_events"],
            "global_events": stats["global_events"],
            "local_context_tokens": stats["local_context_size"],
            "index_info": {
                "trained": index_info.get("is_trained", False),
                "total_vectors": total_vecs,
                "buffered_vectors": index_info.get("buffered_vectors", 0),
                "nlist": current_nlist,
                "optimal_nlist": optimal_nlist,
                "nlist_ratio": nlist_ratio,
                "nlist_status": (
                    "optimal"
                    if nlist_ratio >= 0.85
                    else (
                        "acceptable"
                        if nlist_ratio >= 0.5
                        else "suboptimal"
                        if nlist_ratio > 0
                        else "not_trained"
                    )
                ),
                "nprobe": index_info.get("nprobe", 8),
            },
            "paths": {
                "project": str(manager.memory_dir),
                "global": str(manager.global_path),
            },
            "recommendation": (
                "Index optimal"
                if nlist_ratio >= 0.85
                else f"Consider retraining with expected_tokens hint (nlist {current_nlist}/{optimal_nlist} = {nlist_ratio:.1%})"
            ),
        }

    elif action == "estimate":
        expected_tokens = options.get("expected_tokens")
        if not expected_tokens:
            return {
                "status": "error",
                "error": "expected_tokens parameter required in options for 'estimate' action",
                "example": '{"expected_tokens": 60000}',
            }

        # Use standard estimation formula
        expected_events = expected_tokens // 30
        expected_vectors = expected_events * 27
        optimal_nlist = int(4 * (expected_vectors**0.5))
        min_training_size = int(expected_vectors * 0.9)

        return {
            "status": "success",
            "action": "estimate",
            "estimation": {
                "input_tokens": expected_tokens,
                "expected_events": expected_events,
                "expected_vectors": expected_vectors,
                "optimal_nlist": optimal_nlist,
                "min_training_size": min_training_size,
                "training_threshold": "90%",
            },
            "recommendation": (
                f"Set EMX_EXPECTED_TOKENS={expected_tokens} or pass expected_tokens={expected_tokens} "
                f"to store_memory() for optimal nlist={optimal_nlist}"
            ),
        }

    elif action == "retrain":
        force = options.get("force", False)
        expected_tokens = options.get("expected_tokens")

        # Calculate expected vector count if hint provided
        expected_vector_count = None
        if expected_tokens:
            expected_events = expected_tokens // 30
            expected_vector_count = expected_events * 27
            logger.info(f"Retraining with expected {expected_vector_count} vectors")

        result = manager.retrain_index(
            force=force, expected_vector_count=expected_vector_count
        )

        # Get updated index info
        index_info = manager.get_index_info()
        total_vecs = index_info.get("total_vectors", 0)
        new_nlist = index_info.get("nlist", 0)
        optimal = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0

        return {
            "status": "success",
            "action": "retrain",
            "force": force,
            "result": {
                "success": result.get("success", False),
                "new_nlist": new_nlist,
                "optimal_nlist": optimal,
                "nlist_ratio": new_nlist / optimal if optimal > 0 else 0.0,
                "vectors_indexed": total_vecs,
            },
        }

    elif action == "optimize":
        prune = options.get("prune_old_events", True)
        compress = options.get("compress_embeddings", False)
        result = manager.optimize_memory(prune, compress)
        return {
            "status": "success",
            "action": "optimize",
            "result": result,
        }

    elif action == "clear":
        if not options.get("confirm", False):
            return {
                "status": "error",
                "error": "Destructive operation requires confirm=True in options",
                "example": '{"confirm": true}',
            }
        manager.clear_memory()
        return {
            "status": "success",
            "action": "clear",
            "project": str(manager.memory_dir),
            "warning": "All project memory cleared (global memory preserved)",
        }

    else:
        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "available_actions": ["stats", "retrain", "optimize", "clear", "estimate"],
        }
