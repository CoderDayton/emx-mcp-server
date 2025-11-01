"""Main EMX-LLM MCP server with STDIO transport and IVF indexing."""

import os
import sys
from pathlib import Path
from typing import Optional, Any
from fastmcp import FastMCP
from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config
from emx_mcp.utils.logging import setup_logging

mcp = FastMCP("EMX Memory MCP Server", version="1.0.0")

# Initialize configuration and logging
config = load_config()
logger = setup_logging(config)

# Log expected tokens if configured
if config.get("storage", {}).get("expected_total_tokens"):
    expected = config["storage"]["expected_total_tokens"]
    logger.info(f"Using expected token count: {expected} for optimal nlist")

# Detect project path from config (already validated)
project_path = config.get("project_path", os.getcwd())
global_path = config.get(
    "global_path", str(Path.home() / ".emx-mcp" / "global_memories")
)

# Initialize memory manager
manager = ProjectMemoryManager(
    project_path=project_path, global_path=global_path, config=config
)

logger.info(f"Server started - Project: {project_path}")
logger.info(f"Global memory: {global_path}")
logger.info("Using IVF indexing with adaptive nlist calculation")


@mcp.resource("memory://status")
def get_memory_status() -> dict:
    """
    Get comprehensive memory system status with FAISS IVF nlist optimization info.

    Returns project memory stats, IVF index health with nlist ratio, local context size,
    and global memory availability. Use this to understand current memory state and
    index quality before storing or retrieving memories.
    """
    stats = manager.get_stats()
    index_info = manager.get_index_info()

    # Calculate nlist optimization metrics
    total_vecs = index_info.get("total_vectors", 0)
    current_nlist = index_info.get("nlist", 0)
    optimal_nlist = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
    nlist_ratio = current_nlist / optimal_nlist if optimal_nlist > 0 else 0.0

    return {
        "project": {
            "path": project_path,
            "events_stored": stats["project_events"],
            "local_context_tokens": stats["local_context_size"],
        },
        "global": {
            "path": global_path,
            "events_stored": stats["global_events"],
        },
        "index_health": {
            "trained": index_info.get("is_trained", False),
            "total_vectors": total_vecs,
            "nlist": current_nlist,
            "optimal_nlist": optimal_nlist,
            "nlist_ratio": nlist_ratio,
            "nlist_status": (
                "optimal"
                if nlist_ratio >= 0.85
                else (
                    "acceptable"
                    if nlist_ratio >= 0.5
                    else "suboptimal" if nlist_ratio > 0 else "not_trained"
                )
            ),
            "nprobe": index_info.get("nprobe", 8),
            "buffered_vectors": index_info.get("buffered_vectors", 0),
        },
        "recommendation": (
            "Index optimal"
            if nlist_ratio >= 0.85
            else (
                "Set EMX_EXPECTED_TOKENS or use expected_tokens parameter for optimal nlist"
                if nlist_ratio == 0
                else f"Consider retraining with expected_tokens hint (current nlist {current_nlist}/{optimal_nlist} = {nlist_ratio:.1%})"
            )
        ),
    }


@mcp.tool()
def remember_context(
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
        actual_tokens = len(tokens)
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

    results = {
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


@mcp.tool()
def recall_memories(
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

    # Retrieve from project memory (always if scope != global)
    results = {"status": "success", "query": query, "scope": scope, "memories": []}

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
    optimal_nlist = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
    nlist_ratio = current_nlist / optimal_nlist if optimal_nlist > 0 else 0.0

    results["total_retrieved"] = len(results["memories"])
    results["index_health"] = {
        "nlist": current_nlist,
        "optimal_nlist": optimal_nlist,
        "nlist_ratio": nlist_ratio,
        "status": (
            "optimal"
            if nlist_ratio >= 0.85
            else "acceptable" if nlist_ratio >= 0.5 else "suboptimal"
        ),
    }

    return results


@mcp.tool()
def manage_memory(
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
                        else "suboptimal" if nlist_ratio > 0 else "not_trained"
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
                f"to remember_context() for optimal nlist={optimal_nlist}"
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
            "project": project_path,
            "warning": "All project memory cleared (global memory preserved)",
        }

    else:
        return {
            "status": "error",
            "error": f"Unknown action: {action}",
            "available_actions": ["stats", "retrain", "optimize", "clear", "estimate"],
        }


@mcp.tool()
def transfer_memory(
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
            min_training = int(expected_vectors * 0.9)

            vs = manager.project_store.vector_store
            vs.expected_vector_count = expected_vectors
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


@mcp.tool()
def search_memory_batch(
    queries: list[str],
    k: int = 10,
    format: str = "concise",
    force_batch: bool = False,
) -> dict:
    """
    Advanced: Batch similarity search with adaptive CPU/GPU routing.

    Use this for high-throughput retrieval when you have many queries to run
    at once. The system automatically optimizes execution:
    - GPU: Always batches (4-5x speedup via kernel fusion)
    - CPU: Batches only for >=100 queries (avoids overhead for small batches)

    Search quality depends on FAISS IVF nlist optimization. Check index
    health with manage_memory(action="stats") before bulk operations.

    Most users should use recall_memories() instead - this is for specialized
    workloads like bulk analysis, testing, or benchmarking.

    Args:
        queries: List of query strings (max 1000 per call)
        k: Number of results per query
        format: "concise" (IDs only) or "detailed" (full events)
        force_batch: Override adaptive routing (debugging/benchmarking only)

    Returns:
        Results array with per-query matches, performance metadata, and index health
    """
    import numpy as np

    if len(queries) > 1000:
        return {
            "status": "error",
            "error": f"Query limit exceeded: {len(queries)} > 1000. Split into smaller batches.",
        }

    if format not in ["concise", "detailed"]:
        return {
            "status": "error",
            "error": f"Invalid format: {format}. Use: concise, detailed",
        }

    logger.info(f"Batch search: {len(queries)} queries (k={k}, format={format})")

    # Encode all queries to embeddings
    query_embeddings = np.array(
        [manager.encode_query(query).tolist() for query in queries], dtype=np.float32
    )

    vector_store = manager.project_store.vector_store

    # Determine routing decision
    will_use_batch = force_batch or vector_store._should_use_batch(len(queries))
    routing_reason = (
        "forced"
        if force_batch
        else (
            "gpu_enabled"
            if vector_store.gpu_enabled
            else (
                f"cpu_query_count>={100}"
                if will_use_batch
                else f"cpu_query_count<{100}"
            )
        )
    )

    # Execute batch search
    batch_results = vector_store.search_batch(
        query_embeddings, k, force_batch=force_batch
    )

    # Format results per query
    results_per_query = []
    for idx, (event_ids, distances, metadata) in enumerate(batch_results):
        result_entry = {
            "query_index": idx,
            "query_text": queries[idx] if format == "detailed" else None,
            "event_ids": event_ids,
            "relevance_scores": [float(d) for d in distances],
            "results_found": len(event_ids),
        }

        if format == "detailed":
            result_entry["metadata"] = metadata

        results_per_query.append(result_entry)

    # Add index health info
    index_info = manager.get_index_info()
    total_vecs = index_info.get("total_vectors", 0)
    current_nlist = vector_store.nlist or 0
    optimal_nlist = int(4 * (total_vecs**0.5)) if total_vecs > 0 else 0
    nlist_ratio = current_nlist / optimal_nlist if optimal_nlist > 0 else 0.0

    return {
        "status": "success",
        "total_queries": len(queries),
        "results": results_per_query,
        "performance": {
            "gpu_enabled": vector_store.gpu_enabled,
            "used_batch_api": will_use_batch,
            "routing_reason": routing_reason,
            "nlist": current_nlist,
            "optimal_nlist": optimal_nlist,
            "nlist_ratio": nlist_ratio,
            "nprobe": vector_store.nprobe,
        },
        "index_health": {
            "status": (
                "optimal"
                if nlist_ratio >= 0.85
                else "acceptable" if nlist_ratio >= 0.5 else "suboptimal"
            ),
            "recommendation": (
                "Index optimal for batch search"
                if nlist_ratio >= 0.85
                else f"Suboptimal nlist may affect recall - consider retraining"
            ),
        },
    }


def main():
    """Entry point for uvx execution."""
    try:
        # Run STDIO transport (default)
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
