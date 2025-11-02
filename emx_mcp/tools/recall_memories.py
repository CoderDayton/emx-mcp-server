"""Tool for retrieving relevant memories based on semantic similarity."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emx_mcp.memory.project_manager import ProjectMemoryManager

logger = logging.getLogger(__name__)


def recall_memories(
    manager: "ProjectMemoryManager",
    query: str | list[str],
    scope: str = "project",
    format: str = "detailed",
    k: int = 10,
) -> dict[str, Any]:  # sourcery skip: assign-if-exp, extract-duplicate-method
    """
    Retrieve relevant memories based on semantic similarity to your query.

    This is like asking your memory system "what do I know about X?". The system:
    1. Encodes your query into a semantic embedding
    2. Searches the FAISS IVF vector index for similar past events
    3. Optionally expands results with temporally adjacent events
    4. Returns formatted results optimized for your use case

    Automatically uses batch retrieval for multiple queries (3+ queries)
    for 3x speedup via GPU cache + vectorized operations.

    Search quality depends on index optimization - use EMX_EXPECTED_TOKENS or
    pass expected_tokens to remember_context() for optimal nlist configuration.
    Check index health with manage_memory(action="stats").

    Args:
        manager: ProjectMemoryManager instance
        query: Natural language query or list of queries.
            Single: "debugging React hooks"
            Multiple: ["debugging React hooks", "meeting notes Q3", "API design patterns"]
        scope: Search scope - "project" (current codebase), "global" (all projects), "both"
        format: Response format - "concise" (IDs + snippets) or "detailed" (full events)
        k: Number of relevant events to retrieve per query (default: 10)

    Returns:
        Retrieved memories with relevance scores and index health info.
        For batch queries, returns combined results from all queries.
    """
    # Normalize query input
    is_batch = isinstance(query, list)
    if is_batch:
        queries: list[str] = query  # type: ignore[assignment]
    else:
        queries = [query]  # type: ignore[list-item]
    query_display = f"{len(queries)} queries" if is_batch else f"'{queries[0][:50]}...'"

    logger.info(f"Recalling memories: query={query_display}, scope={scope}, format={format}")

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

    # Warm cache on first query or periodically
    cache_info = manager.get_cache_info()
    if cache_info["cache_size"] == 0:
        logger.debug("Pre-warming retrieval cache for first query...")
        manager.retrieval.warmup_cache_manual(num_passes=3)

    # Retrieve from project memory (always if scope != global)
    results: dict[str, Any] = {
        "status": "success",
        "query": query,
        "scope": scope,
        "memories": [],
        "cache_info": cache_info,
        "used_batch_api": False,
    }

    if scope == "global":
        results["warning"] = "Global scope not yet implemented, searching project only"

    elif scope in {"project", "both"}:
        if len(queries) >= 3:
            query_embeddings = manager.encode_queries_batch(queries)
            batch_results = manager.retrieve_memories_batch(
                query_embeddings,
                k_similarity=k,
                k_contiguity=5,
                use_contiguity=True,
            )
            results["used_batch_api"] = True

            # Flatten batch results into single memory list
            seen_event_ids: set[str] = set()
            for query_idx, project_results in enumerate(batch_results):
                for event in project_results.get("events", []):
                    event_id = event["event_id"]
                    # Deduplicate across queries
                    if event_id in seen_event_ids:
                        continue
                    seen_event_ids.add(event_id)

                    memory_entry: dict[str, Any] = {
                        "event_id": event_id,
                        "source": "project",
                        "relevance_score": event.get("distance", 0),
                    }

                    if is_batch:
                        memory_entry["query_index"] = query_idx
                        memory_entry["query_text"] = (
                            queries[query_idx] if format == "detailed" else None
                        )

                    if format == "detailed":
                        memory_entry["tokens"] = event.get("tokens", [])
                        memory_entry["metadata"] = event.get("metadata", {})
                        memory_entry["timestamp"] = event.get("timestamp")
                    else:  # concise
                        tokens = event.get("tokens", [])
                        memory_entry["snippet"] = " ".join(tokens[:20]) + (
                            "..." if len(tokens) > 20 else ""
                        )

                    results["memories"].append(memory_entry)

        else:
            # Single query or small batch: use individual retrieval
            for query_idx, single_query in enumerate(queries):
                query_embedding = manager.encode_query(single_query)
                project_results = manager.retrieve_memories(
                    query_embedding,
                    k_similarity=k,
                    k_contiguity=5,
                    use_contiguity=True,
                )

                for event in project_results.get("events", []):
                    single_memory: dict[str, Any] = {
                        "event_id": event["event_id"],
                        "source": "project",
                        "relevance_score": event.get("distance", 0),
                    }

                    if is_batch:
                        single_memory["query_index"] = query_idx
                        single_memory["query_text"] = single_query if format == "detailed" else None

                    if format == "detailed":
                        single_memory["tokens"] = event.get("tokens", [])
                        single_memory["metadata"] = event.get("metadata", {})
                        single_memory["timestamp"] = event.get("timestamp")
                    else:  # concise
                        tokens = event.get("tokens", [])
                        single_memory["snippet"] = " ".join(tokens[:20]) + (
                            "..." if len(tokens) > 20 else ""
                        )

                    results["memories"].append(single_memory)

    # Add index health info
    index_info = manager.get_index_info()
    total_vecs = index_info.get("total_vectors", 0)
    current_nlist = index_info.get("nlist", 0)
    optimal_nlist_val = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
    nlist_ratio = current_nlist / optimal_nlist_val if optimal_nlist_val > 0 else 0.0

    results["total_retrieved"] = len(results["memories"])
    results["num_queries"] = len(queries)
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
