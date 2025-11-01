"""EMX MCP Server - Infinite context memory with episodic events."""

__version__ = "1.0.0"

from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.models.events import EpisodicEvent

__all__ = ["ProjectMemoryManager", "EpisodicEvent"]
