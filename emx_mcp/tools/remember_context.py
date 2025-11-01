"""Tool for storing new information into project memory."""

from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


def remember_context(
    manager,
    content: str,
    metadata: Optional[dict[Any, Any]] = None,
    auto_segment: bool = True,
    gamma: float = 1.0,
    expected_tokens: Optional[int] = None,
) -> dict:
    """
    Store new information into project memory with automatic segmentation.

    Think of this like saving a conversation, document, or experience into
    long-term memory. The content is automatically:
    1. Tokenized (split into words/segments)
    2. Segmented into coherent episodic events (if auto_segment=True)
    3. Embedded using sentence-transformers
    4. Indexed for fast semantic retrieval with optimal nlist

    Use this when you want to remember something for later retrieval.

    Args:
        manager: ProjectMemoryManager instance
        content: Text content to remember (conversation, notes, code, etc)
        metadata: Optional tags/labels (e.g., {"topic": "debugging", "date": "2025-10-31"})
        auto_segment: Automatically split into semantic episodes (recommended: True)
        gamma: Boundary sensitivity for segmentation (1.0=balanced, >1=more segments)
        expected_tokens: Hint for total expected tokens (enables optimal nlist calculation)

    Returns:
        Event IDs created, segmentation info, and index health status
    """
    logger.info(
        f"Remembering content: {len(content)} chars, auto_segment={auto_segment}"
    )

    # Simple tokenization (split on whitespace)
    tokens = content.split()

    if not tokens:
        return {"status": "error", "error": "Empty content provided"}

    # Update expected vector count for optimal nlist if provided
    if expected_tokens:
        # Estimate: ~27 vectors per event, ~30 tokens per event
        expected_events = expected_tokens // 30
        expected_vectors = expected_events * 27
        min_training = int(expected_vectors * 0.9)

        vs = manager.project_store.vector_store
        vs.expected_vector_count = expected_vectors
        vs.min_training_size = min_training
        vs.nlist = vs._calculate_optimal_nlist(expected_vectors)

        logger.info(
            f"Updated index: expected {expected_vectors} vectors → "
            f"nlist={vs.nlist}, training at {min_training}"
        )

    results: dict[str, Any] = {
        "status": "success",
        "tokens_processed": len(tokens),
        "event_ids": [],
        "segmentation_used": auto_segment,
    }

    if auto_segment and len(tokens) > 50:  # Only segment if enough content
        # Segment into episodic boundaries using O(n) linear method
        seg_result = manager.segment_tokens(tokens, gamma)
        boundaries = seg_result["refined_boundaries"]

        results["num_segments"] = seg_result["num_events"]
        results["segmentation_method"] = seg_result["method"]

        # Store each segment as separate event (batched encoding optimization)
        for i in range(len(boundaries) - 1):
            segment_tokens = tokens[boundaries[i] : boundaries[i + 1]]
            event_result = manager.add_event(
                segment_tokens, embeddings=None, metadata=metadata or {}
            )
            results["event_ids"].append(event_result["event_id"])

        # Flush any remaining buffered events
        flush_result = manager.flush_events()
        if flush_result["status"] == "flushed":
            results["batch_encoding_stats"] = {
                "events_flushed": flush_result["num_events"],
                "total_tokens": flush_result["total_tokens"],
                "tokens_per_second": flush_result["tokens_per_second"],
            }

    else:
        # Store as single event without segmentation
        event_result = manager.add_event(
            tokens, embeddings=None, metadata=metadata or {}, force_flush=True
        )
        results["event_ids"].append(event_result["event_id"])
        results["num_segments"] = 1

    # Check index health with enhanced diagnostics
    index_info = manager.get_index_info()
    results["index_health"] = {
        "trained": index_info.get("is_trained", False),
        "total_vectors": index_info.get("total_vectors", 0),
        "buffered_vectors": index_info.get("buffered_vectors", 0),
        "nlist": index_info.get("nlist", 0),
        "optimal_nlist": index_info.get("optimal_nlist", 0),
        "nlist_ratio": index_info.get("nlist_ratio", 0.0),
    }

    # Set status based on nlist ratio
    nlist_ratio = index_info.get("nlist_ratio", 0.0)
    if not index_info.get("is_trained", False):
        results["index_status"] = "buffering"
    elif nlist_ratio < 0.5:
        results["index_status"] = "suboptimal_retrain_recommended"
    elif nlist_ratio < 0.85:
        results["index_status"] = "acceptable"
    else:
        results["index_status"] = "optimal"

    return results
