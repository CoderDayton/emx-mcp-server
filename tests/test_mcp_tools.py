"""Integration tests for MCP tool endpoints."""

import pytest

from emx_mcp.server import mcp


class TestMCPTools:
    """Test MCP tool registration and basic functionality."""

    @pytest.mark.asyncio
    async def test_store_memory_tool_registered(self):
        """Test store_memory tool is registered."""
        tool = await mcp.get_tool("store_memory")
        assert tool is not None
        assert tool.name == "store_memory"
        assert tool.description
        assert len(tool.description) > 20

    @pytest.mark.asyncio
    async def test_recall_memories_tool_registered(self):
        """Test recall_memories tool is registered."""
        tool = await mcp.get_tool("recall_memories")
        assert tool is not None
        assert tool.name == "recall_memories"
        assert tool.description
        assert len(tool.description) > 20

    @pytest.mark.asyncio
    async def test_manage_memory_tool_registered(self):
        """Test manage_memory tool is registered."""
        tool = await mcp.get_tool("manage_memory")
        assert tool is not None
        assert tool.name == "manage_memory"
        assert tool.description

    @pytest.mark.asyncio
    async def test_remove_memories_tool_registered(self):
        """Test remove_memories tool is registered."""
        tool = await mcp.get_tool("remove_memories")
        assert tool is not None
        assert tool.name == "remove_memories"
        assert tool.description

    @pytest.mark.asyncio
    async def test_transfer_memory_tool_registered(self):
        """Test transfer_memory tool is registered."""
        tool = await mcp.get_tool("transfer_memory")
        assert tool is not None
        assert tool.name == "transfer_memory"
        assert tool.description

    @pytest.mark.asyncio
    async def test_search_memory_batch_tool_registered(self):
        """Test search_memory_batch tool is registered."""
        tool = await mcp.get_tool("search_memory_batch")
        assert tool is not None
        assert tool.name == "search_memory_batch"
        assert tool.description

    @pytest.mark.asyncio
    async def test_all_tools_have_descriptions(self):
        """Test all tools have non-empty descriptions."""
        tools = await mcp.get_tools()
        assert len(tools) >= 6, "Expected at least 6 tools registered"
        for tool in tools.values():
            assert tool.description, f"Tool {tool.name} missing description"

    @pytest.mark.asyncio
    async def test_tool_count(self):
        """Test expected number of tools are registered."""
        tools = await mcp.get_tools()
        assert len(tools) == 6, f"Expected 6 tools, got {len(tools)}"
