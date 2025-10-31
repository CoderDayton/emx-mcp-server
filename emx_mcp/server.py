"""Main EMX-LLM MCP server with STDIO transport and IVF indexing."""

import os
import sys
from pathlib import Path
from typing import Optional, Any
from fastmcp import FastMCP
from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config
from emx_mcp.utils.logging import setup_logging
from emx_mcp.metrics import setup_metrics, get_health_tracker

mcp = FastMCP("EMX Memory MCP Server", version="1.0.0")

# Initialize configuration and logging
config = load_config()
logger = setup_logging(config)

# Initialize OpenTelemetry metrics
try:
    setup_metrics(config)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        logger.info(f"Metrics initialized: console + OTLP ({otlp_endpoint})")
    else:
        logger.info("Metrics initialized: console exporter only")
except Exception as e:
    logger.warning(f"Failed to initialize metrics: {e}")

# Detect project path
project_path = os.getenv("EMX_PROJECT_PATH", os.getcwd())
global_path = os.getenv(
    "EMX_GLOBAL_PATH", str(Path.home() / ".emx-mcp" / "global_memories")
)

# Initialize memory manager
manager = ProjectMemoryManager(
    project_path=project_path, global_path=global_path, config=config
)

logger.info(f"Server started - Project: {project_path}")
logger.info(f"Global memory: {global_path}")
logger.info("Using IVF indexing")


@mcp.resource("memory://status")
def get_memory_status() -> dict:
    """
    Get comprehensive memory system status.
    
    Returns project memory stats, IVF index health, local context size,
    and global memory availability. Use this to understand current memory
    state before storing or retrieving memories.
    """
    stats = manager.get_stats()
    index_info = manager.get_index_info()
    
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
            "total_vectors": index_info.get("total_vectors", 0),
            "nlist": index_info.get("nlist", 0),
            "nprobe": index_info.get("nprobe", 8),
            "buffered_vectors": index_info.get("buffered_vectors", 0),
        },
    }


@mcp.resource("metrics://health")
def get_metrics_health() -> dict:
    """
    Get OpenTelemetry metrics export health status.
    
    Returns:
        - Last successful export timestamp
        - Total exports and failures
        - Success rate
        - Configured exporters (console, OTLP)
        - Last error if any
    
    Use this to verify metrics are flowing to Grafana Cloud or other
    observability backends. If exports are failing, check OTLP endpoint
    configuration and network connectivity.
    """
    tracker = get_health_tracker()
    
    if tracker is None:
        return {
            "status": "not_initialized",
            "message": "Metrics system not initialized. Check startup logs for errors.",
        }
    
    health = tracker.get_health()
    
    return {
        "status": "healthy" if health["healthy"] else "degraded",
        "exporters": health["exporters"],
        "last_success": health["last_success"],
        "last_failure": health["last_failure"],
        "stats": health["stats"],
    }


@mcp.tool()
def remember_context(
    content: str,
    metadata: Optional[dict[Any, Any]] = None,
    auto_segment: bool = True,
    gamma: float = 1.0,
) -> dict:
    """
    Store new information into project memory with automatic segmentation.
    
    Think of this like saving a conversation, document, or experience into
    long-term memory. The content is automatically:
    1. Tokenized (split into words/segments)
    2. Segmented into coherent episodic events (if auto_segment=True)
    3. Embedded using sentence-transformers
    4. Indexed for fast semantic retrieval
    
    Use this when you want to remember something for later retrieval.
    
    Args:
        content: Text content to remember (conversation, notes, code, etc)
        metadata: Optional tags/labels (e.g., {"topic": "debugging", "date": "2025-10-31"})
        auto_segment: Automatically split into semantic episodes (recommended: True)
        gamma: Boundary sensitivity for segmentation (1.0=balanced, >1=more segments)
    
    Returns:
        Event IDs created, segmentation info, and index health status
    """
    logger.info(f"Remembering content: {len(content)} chars, auto_segment={auto_segment}")
    
    # Simple tokenization (split on whitespace)
    tokens = content.split()
    
    if not tokens:
        return {"status": "error", "error": "Empty content provided"}
    
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
        
        # Store each segment as separate event
        for i in range(len(boundaries) - 1):
            segment_tokens = tokens[boundaries[i]:boundaries[i+1]]
            event_result = manager.add_event(
                segment_tokens, 
                embeddings=None, 
                metadata=metadata or {}
            )
            results["event_ids"].append(event_result["event_id"])
            
    else:
        # Store as single event without segmentation
        event_result = manager.add_event(tokens, embeddings=None, metadata=metadata or {})
        results["event_ids"].append(event_result["event_id"])
        results["num_segments"] = 1
    
    # Check index health
    index_info = manager.get_index_info()
    if not index_info.get("is_trained", False):
        results["index_status"] = "training_needed"
    elif index_info.get("buffered_vectors", 0) > 500:
        results["index_status"] = "retrain_recommended"
    else:
        results["index_status"] = "healthy"
    
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
    2. Searches the vector index for similar past events
    3. Optionally expands results with temporally adjacent events
    4. Returns formatted results optimized for your use case
    
    Use this when you need to recall past context, find related information,
    or build context for a new task.
    
    Args:
        query: Natural language query (e.g., "debugging React hooks", "meeting notes about Q3 goals")
        scope: Search scope - "project" (current codebase), "global" (all projects), "both"
        format: Response format - "concise" (IDs + snippets) or "detailed" (full events)
        k: Number of relevant events to retrieve (default: 10)
    
    Returns:
        Retrieved memories with relevance scores, formatted per your request
    """
    logger.info(f"Recalling memories: query='{query[:50]}...', scope={scope}, format={format}")
    
    if scope not in ["project", "global", "both"]:
        return {"status": "error", "error": f"Invalid scope: {scope}. Use: project, global, both"}
    
    if format not in ["concise", "detailed"]:
        return {"status": "error", "error": f"Invalid format: {format}. Use: concise, detailed"}
    
    # Encode query
    query_embedding = manager.encode_query(query)
    
    # Retrieve from project memory (always if scope != global)
    results = {"status": "success", "query": query, "scope": scope, "memories": []}
    
    if scope in ["project", "both"]:
        project_results = manager.retrieve_memories(
            query_embedding.tolist(), 
            k_similarity=k, 
            k_contiguity=5,
            use_contiguity=True
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
                memory_entry["snippet"] = " ".join(tokens[:20]) + ("..." if len(tokens) > 20 else "")
            
            results["memories"].append(memory_entry)
    
    # TODO: Add global scope retrieval when requested
    if scope == "global":
        results["warning"] = "Global scope not yet implemented, searching project only"
    
    results["total_retrieved"] = len(results["memories"])
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
    - "stats": Get detailed memory statistics and index health
    - "retrain": Rebuild IVF search index (use after adding many events)
    - "optimize": Prune least-accessed events and compress storage
    - "clear": Delete all project memory (requires confirm=True in options)
    
    Args:
        action: Operation to perform (stats, retrain, optimize, clear)
        options: Action-specific options (e.g., {"force": true} for retrain)
    
    Returns:
        Operation results and updated system status
    """
    logger.info(f"Memory management: action={action}")
    options = options or {}
    
    if action == "stats":
        stats = manager.get_stats()
        index_info = manager.get_index_info()
        
        return {
            "status": "success",
            "action": "stats",
            "project_events": stats["project_events"],
            "global_events": stats["global_events"],
            "local_context_tokens": stats["local_context_size"],
            "index_info": {
                "trained": index_info.get("is_trained", False),
                "total_vectors": index_info.get("total_vectors", 0),
                "buffered_vectors": index_info.get("buffered_vectors", 0),
                "nlist": index_info.get("nlist", 0),
                "nprobe": index_info.get("nprobe", 8),
            },
            "paths": {
                "project": str(manager.memory_dir),
                "global": str(manager.global_path),
            },
        }
    
    elif action == "retrain":
        force = options.get("force", False)
        result = manager.retrain_index(force=force)
        return {
            "status": "success",
            "action": "retrain",
            "force": force,
            "result": result,
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
            "available_actions": ["stats", "retrain", "optimize", "clear"],
        }


@mcp.tool()
def transfer_memory(
    action: str,
    path: str,
    merge: bool = False,
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
    
    Args:
        action: "export" or "import"
        path: File path for archive (.tar.gz)
        merge: For import only - merge with existing memory (vs replace)
    
    Returns:
        Transfer status, file info, and event counts
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
        result = manager.import_memory(path, merge=merge)
        return {
            "status": "success",
            "action": "import",
            "path": path,
            "merge": merge,
            "events_imported": result["events"],
            "total_events_now": manager.get_stats()["project_events"],
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
    
    Most users should use recall_memories() instead - this is for specialized
    workloads like bulk analysis, testing, or benchmarking.
    
    Args:
        queries: List of query strings (max 1000 per call)
        k: Number of results per query
        format: "concise" (IDs only) or "detailed" (full events)
        force_batch: Override adaptive routing (debugging/benchmarking only)
    
    Returns:
        Results array with per-query matches and performance metadata
    """
    import numpy as np
    
    if len(queries) > 1000:
        return {
            "status": "error",
            "error": f"Query limit exceeded: {len(queries)} > 1000. Split into smaller batches."
        }
    
    if format not in ["concise", "detailed"]:
        return {"status": "error", "error": f"Invalid format: {format}. Use: concise, detailed"}
    
    logger.info(f"Batch search: {len(queries)} queries (k={k}, format={format})")
    
    # Encode all queries to embeddings
    query_embeddings = np.array([
        manager.encode_query(query).tolist() 
        for query in queries
    ], dtype=np.float32)
    
    vector_store = manager.project_store.vector_store
    
    # Determine routing decision
    will_use_batch = force_batch or vector_store._should_use_batch(len(queries))
    routing_reason = (
        "forced" if force_batch else
        "gpu_enabled" if vector_store.gpu_enabled else
        f"cpu_query_count>={100}" if will_use_batch else
        f"cpu_query_count<{100}"
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
    
    return {
        "status": "success",
        "total_queries": len(queries),
        "results": results_per_query,
        "performance": {
            "gpu_enabled": vector_store.gpu_enabled,
            "used_batch_api": will_use_batch,
            "routing_reason": routing_reason,
            "nlist": vector_store.nlist,
            "nprobe": vector_store.nprobe,
        }
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
