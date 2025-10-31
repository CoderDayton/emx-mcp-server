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


@mcp.resource("memory://project/context")
def get_project_context() -> dict:
    """Get current project's local context (working memory)."""
    return {
        "local_context": manager.get_local_context(),
        "token_count": len(manager.get_local_context()),
        "project_path": project_path,
    }


@mcp.resource("memory://project/stats")
def get_memory_stats() -> dict:
    """Get memory statistics for current project."""
    stats = manager.get_stats()
    stats["index_info"] = manager.get_index_info()
    return stats


@mcp.resource("memory://global/context")
def get_global_context() -> dict:
    """Get shared global semantic context."""
    return {
        "global_context": manager.get_global_context(),
        "semantic_knowledge": manager.get_global_semantic(),
    }


@mcp.tool()
def segment_experience(
    tokens: list[str], gamma: float = 1.0, use_refinement: bool = True
) -> dict:
    """
    Segment token sequence into episodic events using surprise.

    Args:
        tokens: List of token strings or IDs
        gamma: Surprise threshold sensitivity (default: 1.0)
        use_refinement: Apply boundary refinement (default: True)

    Returns:
        Dictionary with event boundaries and metadata
    """
    logger.info(f"Segmenting {len(tokens)} tokens (gamma={gamma})")
    return manager.segment_tokens(tokens, gamma, use_refinement)


@mcp.tool()
def retrieve_memories(
    query: str,
    k_similarity: int = 10,
    k_contiguity: int = 5,
    use_contiguity: bool = True,
) -> dict:
    """
    Retrieve relevant memories using two-stage retrieval.

    Query is automatically encoded server-side.

    Args:
        query: Query text (not embedding)
        k_similarity: Number of similar events to retrieve
        k_contiguity: Size of temporal contiguity buffer
        use_contiguity: Enable contiguity buffer

    Returns:
        Dictionary with retrieved events and metadata
    """
    logger.info(f"Retrieving memories for query: '{query[:50]}...'")

    # Server encodes query to embedding
    query_embedding = manager.encode_query(query)

    return manager.retrieve_memories(
        query_embedding.tolist(), k_similarity, k_contiguity, use_contiguity
    )


@mcp.tool()
def add_episodic_event(
    tokens: list[str], metadata: Optional[dict[Any, Any]] = None
) -> dict:
    """
    Add new episodic event to project memory.

    Embeddings are automatically generated server-side from tokens.

    Args:
        tokens: Event tokens (words or text segments)
        metadata: Optional event metadata

    Returns:
        Event ID and storage confirmation
    """
    logger.info(f"Adding event with {len(tokens)} tokens")

    # Server generates embeddings from tokens
    result = manager.add_event(tokens, embeddings=None, metadata=metadata or {})

    # Check if retraining needed
    if result.get("retrain_recommended", False):
        logger.info("IVF index retraining recommended")

    return result


@mcp.tool()
def remove_episodic_events(event_ids: list[str]) -> dict:
    """
    Remove episodic events from project memory.

    Args:
        event_ids: List of event IDs to remove

    Returns:
        Removal confirmation and retraining status
    """
    logger.info(f"Removing {len(event_ids)} events")
    return manager.remove_events(event_ids)


@mcp.tool()
def retrain_index(force: bool = False) -> dict:
    """
    Retrain IVF index for optimal performance.

    Args:
        force: Force retraining even if not recommended

    Returns:
        Retraining status and new index statistics
    """
    logger.info(f"Retraining IVF index (force={force})")
    return manager.retrain_index(force)


@mcp.tool()
def optimize_memory(
    prune_old_events: bool = True, compress_embeddings: bool = False
) -> dict:
    """
    Optimize memory storage and performance.

    Args:
        prune_old_events: Remove least-accessed old events
        compress_embeddings: Apply PQ compression to embeddings

    Returns:
        Optimization results
    """
    logger.info("Optimizing project memory")
    return manager.optimize_memory(prune_old_events, compress_embeddings)


@mcp.tool()
def clear_project_memory(confirm: bool = False) -> dict:
    """
    Clear all project memory (destructive operation).

    Args:
        confirm: Must be True to execute

    Returns:
        Confirmation message
    """
    if not confirm:
        return {"error": "Must set confirm=True to clear memory"}

    logger.warning("Clearing project memory")
    manager.clear_memory()
    return {"status": "cleared", "project": project_path}


@mcp.tool()
def export_project_memory(output_path: str) -> dict:
    """
    Export project memory to portable format.

    Args:
        output_path: Destination file path (.tar.gz)

    Returns:
        Export status and file info
    """
    logger.info(f"Exporting memory to {output_path}")
    return manager.export_memory(output_path)


@mcp.tool()
def import_project_memory(input_path: str, merge: bool = False) -> dict:
    """
    Import memory from another project.

    Args:
        input_path: Source file path (.tar.gz)
        merge: Merge with existing memory (vs replace)

    Returns:
        Import status and event count
    """
    logger.info(f"Importing memory from {input_path} (merge={merge})")
    return manager.import_memory(input_path, merge)


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
