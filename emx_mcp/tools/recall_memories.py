"""Tool for retrieving relevant memories based on semantic similarity."""

from typing import Any
import logging

logger = logging.getLogger(__name__)


def recall_memories(
    manager,
    query: str,
    scope: str = "project",
    format: str = "detailed",
    k: int = 10,
) -> dict:
    """
    Retrieve relevant memories based on semantic similarity to your query.

    This is like asking your memory system "what do I know about X?". The system:
    1. Encodes your query into a semantic embedding
    2. Searches the FAISS IVF vector index for similar past events
    3. Optionally expands results with temporally adjacent events
    4. Returns formatted results optimized for your use case

    Search quality depends on index optimization - use EMX_EXPECTED_TOKENS or
    pass expected_tokens to remember_context() for optimal nlist configuration.
    Check index health with manage_memory(action="stats").

    Args:
        manager: ProjectMemoryManager instance
        query: Natural language query (e.g., "debugging React hooks", "meeting notes about Q3 goals")
        scope: Search scope - "project" (current codebase), "global" (all projects), "both"
        format: Response format - "concise" (IDs + snippets) or "detailed" (full events)
        k: Number of relevant events to retrieve (default: 10)

    Returns:
        Retrieved memories with relevance scores and index health info
    """
    logger.info(
        f"Recalling memories: query='{query[:50]}...', scope={scope}, format={format}"
    )

    if scope not in ["project", "global", "both"]:
        return {
            "status": "error",
            "error": f"Invalid scope: {scope}. Use: project, global, both",
        }

    if format not in ["concise", "detailed"]:
        return {
            "status": "error",
            "error": f"Invalid format: {format}. Use: concise, detailed",
        }

    # Encode query
    query_embedding = manager.encode_query(query)

    # Warm cache on first query or periodically
    cache_info = manager.get_cache_info()
    if cache_info["cache_size"] == 0:
        logger.debug("Pre-warming retrieval cache for first query...")
        manager.warmup_cache_manual(num_passes=3)

    # Retrieve from project memory (always if scope != global)
    results: dict[str, Any] = {
        "status": "success",
        "query": query,
        "scope": scope,
        "memories": [],
        "cache_info": cache_info,
    }

    if scope in ["project", "both"]:
        project_results = manager.retrieve_memories(
            query_embedding.tolist(),
            k_similarity=k,
            k_contiguity=5,
            use_contiguity=True,
        )

        for event in project_results.get("events", []):
            memory_entry = {
                "event_id": event["event_id"],
                "source": "project",
                "relevance_score": event.get("distance", 0),
            }

            if format == "detailed":
                memory_entry["tokens"] = event.get("tokens", [])
                memory_entry["metadata"] = event.get("metadata", {})
                memory_entry["timestamp"] = event.get("timestamp")
            else:  # concise
                # Return first 20 tokens as snippet
                tokens = event.get("tokens", [])
                memory_entry["snippet"] = " ".join(tokens[:20]) + (
                    "..." if len(tokens) > 20 else ""
                )

            results["memories"].append(memory_entry)

    # TODO: Add global scope retrieval when requested
    if scope == "global":
        results["warning"] = "Global scope not yet implemented, searching project only"

    # Add index health info
    index_info = manager.get_index_info()
    total_vecs = index_info.get("total_vectors", 0)
    current_nlist = index_info.get("nlist", 0)
    optimal_nlist_val = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
    nlist_ratio = current_nlist / optimal_nlist_val if optimal_nlist_val > 0 else 0.0

    results["total_retrieved"] = len(results["memories"])
    results["index_health"] = {
        "nlist": current_nlist,
        "optimal_nlist": optimal_nlist_val,
        "nlist_ratio": nlist_ratio,
        "status": (
            "optimal"
            if nlist_ratio >= 0.85
            else "acceptable"
            if nlist_ratio >= 0.5
            else "suboptimal"
        ),
    }

    return results
