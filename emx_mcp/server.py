"""Main EMX-LLM MCP server with STDIO transport and IVF indexing."""

import os
import sys
from pathlib import Path
from typing import Optional, Any
from fastmcp import FastMCP
from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config
from emx_mcp.utils.logging import setup_logging
from emx_mcp.tools import (
    store_memory as store_memory_impl,
    recall_memories as recall_memories_impl,
    remove_memories as remove_memories_impl,
    manage_memory as manage_memory_impl,
    transfer_memory as transfer_memory_impl,
    search_memory_batch as search_memory_batch_impl,
)

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
                    else "suboptimal"
                    if nlist_ratio > 0
                    else "not_trained"
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
def store_memory(
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
    return store_memory_impl(
        manager, content, metadata, auto_segment, gamma, expected_tokens
    )


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
    pass expected_tokens to store_memory() for optimal nlist configuration.
    Check index health with manage_memory(action="stats").

    Args:
        query: Natural language query (e.g., "debugging React hooks", "meeting notes about Q3 goals")
        scope: Search scope - "project" (current codebase), "global" (all projects), "both"
        format: Response format - "concise" (IDs + snippets) or "detailed" (full events)
        k: Number of relevant events to retrieve (default: 10)

    Returns:
        Retrieved memories with relevance scores and index health info
    """
    return recall_memories_impl(manager, query, scope, format, k)


@mcp.tool()
def remove_memories(
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
        event_ids: List of event IDs to remove (get these from recall_memories)
        confirm: Safety flag - must be True to execute deletion

    Returns:
        Removal status, count of removed events, and updated system stats
    """
    return remove_memories_impl(manager, event_ids, confirm)


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
    return manage_memory_impl(manager, action, options)


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
    return transfer_memory_impl(manager, action, path, merge, expected_tokens)


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
    return search_memory_batch_impl(manager, queries, k, format, force_batch)


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
