<div align="center">
  <img src="https://i.imgur.com/oGBS08t.png" width="175">

  [![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB)](https://docs.python.org/3/)
  [![protocol](https://img.shields.io/badge/protocol-Model%20Context%20Protocol-8A2BE2)](https://modelcontextprotocol.io/)
  [![embedding](https://img.shields.io/badge/embedding-sentence--transformers-FF6B6B)](https://github.com/sentence-transformers/sentence-transformers)
  [![Open Source](https://img.shields.io/badge/open_source-MIT-green)](#-license)

  ⭐ Star us on GitHub — your support motivates us a lot! 🙏😊

  🔥 Why EMX-MCP Server is the best choice for AI agent memory — find out in our [technical overview](https://resources.abblix.com/pdf/emx-mcp-technical-overview.pdf) 📑
</div>

## Table of Contents
- [About](#-about)
- [What's New](#-whats-new)
- [How It Works](#-how-it-works)
- [How to Build](#-how-to-build)
- [How to Use](#-how-to-use)
- [Feedback and Contributions](#-feedback-and-contributions)
- [License](#-license)
- [Contacts](#%EF%B8%8F-contacts)

## 🚀 About

**EMX-MCP Server** is a Python library designed to provide comprehensive support for infinite-context episodic memory for AI agents. It implements the Human-inspired Episodic Memory for Infinite Context LLMs (EM-LLM) algorithm using embedding-based approaches for maximum compatibility and performance. The system is built for seamless integration with AI coding agents (GitHub Copilot, Claude Code, Cursor, etc.) via the Model Context Protocol (MCP).

The server employs advanced machine learning techniques including:

- **Embedding-Based Surprise Calculation**: Uses sentence-transformers to compute semantic distances from local context for boundary detection
- **FAISS IVF Indexing**: Scales to 10M+ vectors with sub-millisecond search performance
- **Hierarchical Memory Storage**: Three-tier memory system with automatic disk offloading
- **Two-Stage Retrieval**: Combines similarity search with temporal contiguity
- **Project + Global Memory**: Per-workspace and shared contexts for enhanced relevance

This approach ensures high performance, scalability, and seamless integration with existing AI development workflows while maintaining the semantic understanding required for effective episodic memory management.

## ✨ What's New

### Version 1.0.0 (Latest)

🚀 **Major Release: Workflow-Oriented MCP Tools**
- **Modular Tool Architecture**: Tools organized into dedicated modules in `emx_mcp/tools/` for maintainability
- **Refined Tool API**: 6 focused workflow tools (`store_memory`, `recall_memories`, `remove_memories`, `manage_memory`, `transfer_memory`, `search_memory_batch`)
- **Token-Efficient Responses**: Added `format` parameter (concise vs detailed) - 72% token reduction for concise mode
- **Natural Language Interface**: Tools designed around agent workflows (store → recall → remove) rather than system primitives
- **Single Resource**: Consolidated `memory://status` for unified system health monitoring

🧠 **Embedding-Based Architecture with Boundary Refinement**
- **Complete Embedding-Based Implementation**: Replaced LLM-dependent operations with embedding-based approaches for maximum compatibility
- **Graph-Theoretic Boundary Refinement**: Implements Algorithm 1 from EM-LLM paper with modularity/conductance optimization (10-29% accuracy boost)
- **Enhanced Surprise Calculation**: Uses semantic distances from local context centroids for boundary detection
- **O(nm) Complexity Control**: Chunked refinement with configurable `max_refinement_window` prevents performance degradation
- **Optimized Performance**: 433 tokens/sec throughput on 60k token corpus while maintaining semantic accuracy

🔧 **New Features**
- **Adaptive GPU Routing**: Batch search automatically routes CPU (<100 queries) vs GPU (all batches) for optimal performance
- **Selective Memory Deletion**: New `remove_memories` tool for granular memory management without clearing everything
- **Response Format Control**: All retrieval tools support concise (IDs + snippets) and detailed (full events) modes
- **Context Window Support**: Configurable context windows for embedding-based surprise calculation
- **Comprehensive Test Suite**: 112/115 tests passing (97.4% success rate) with boundary refinement validation

> **Migration Note**: This is a **breaking change** for direct tool usage. Legacy tools replaced with consolidated workflow tools. See tool documentation for new API.

## 🧠 How It Works

### Architecture Overview

The EMX-MCP Server implements the EM-LLM algorithm using an embedding-based approach that replaces LLM-dependent operations with semantic similarity computations:

```
AI Agent (Copilot/Claude/Cursor)
        ↓ MCP Protocol
   EMX-MCP Server
        ├─ EmbeddingEncoder (sentence-transformers)
        │   └─ encode_tokens_with_context()
        ├─ SurpriseSegmenter (O(n) + O(nm) refinement)
        │   ├─ _compute_embedding_surprises()
        │   ├─ _compute_embedding_adjacency()
        │   └─ _refine_boundaries() [modularity/conductance]
        ├─ ProjectMemoryManager
        │   └─ HierarchicalMemoryStore
        │       ├─ FAISS IVF Vector Store (SQ8 compression)
        │       ├─ SQLite Graph Store (temporal links)
        │       └─ Memory-Mapped Disk Manager
        └─ CachedBatchRetrieval (LRU cache)
            ├─ Similarity Search (FAISS IVF)
            └─ Temporal Contiguity (graph neighbors)
```

### Core Algorithm: Embedding-Based Surprise + Boundary Refinement

The system implements Algorithm 1 from the EM-LLM paper using a two-phase approach:

**Phase 1: Surprise-Based Segmentation** (O(n))
1. **Token Encoding**: Each token is encoded with local context using sentence-transformers
2. **Context Centroid**: Calculate centroid of previous tokens (configurable window size)
3. **Distance Calculation**: Measure cosine distance from current embedding to context centroid
4. **Adaptive Threshold**: Use μ + γσ from local window for boundary detection

**Phase 2: Graph-Theoretic Refinement** (O(nm), where m << n)
1. **Adjacency Matrix**: Compute cosine similarity between tokens in each segment
2. **Modularity Optimization**: Find boundary position maximizing community structure (Equation 3)
3. **Conductance Minimization**: Alternative metric for boundary quality (Equation 4)
4. **Chunked Processing**: Segments limited to `max_refinement_window` (default: 512) for performance

```python
# Simplified embedding-based surprise + refinement
def segment_tokens(token_embeddings, gamma=1.0, enable_refinement=True):
    # Phase 1: O(n) surprise-based boundaries
    surprises = compute_embedding_surprises(token_embeddings)
    boundaries = identify_boundaries(surprises, gamma)

    # Phase 2: O(nm) graph-theoretic refinement (m=512 default)
    if enable_refinement:
        boundaries = refine_boundaries(token_embeddings, boundaries, metric="modularity")

    return boundaries
```

### Memory Hierarchy

The system employs a three-tier memory architecture:

- **Tier 1 (Initial)**: First 1000 tokens for attention sinks
- **Tier 2 (Local)**: Recent 4096 tokens in rolling window
- **Tier 3 (Episodic)**: FAISS-indexed events with disk offloading

### Performance Characteristics

- **Embedding Generation**: ~2s for 10,000 tokens (sentence-transformers on GPU)
- **Boundary Detection**: O(n) base + O(nm) refinement where m=512 (sub-second for 100k tokens)
- **Refinement Impact**: 10-29% accuracy improvement on retrieval tasks (paper benchmarks)
- **Memory Retrieval**: <500ms for similarity + contiguity search with LRU caching
- **Storage Scale**: Tested up to 10M vectors with FAISS IVF+SQ8 indexing (4x compression)
- **Batch Throughput**: 433 tokens/sec on 60k token corpus (GPU-accelerated)

## 🚀 How to Use

### Quick Setup with MCP

The easiest way to get started is by integrating EMX-MCP Server with your AI coding agent using the Model Context Protocol:

Most AI agents support MCP integration. Add the server configuration:

```json
{
  "mcpServers": {
    "emx-memory": {
      "command": "uvx",
      "args": ["emx-mcp-server"],
      "env": {}
    }
  }
}
```

### Available MCP Tools

Once integrated, your AI agent will have access to these high-level memory tools:

#### 📝 Core Memory Operations
- **`store_memory(content, metadata, auto_segment, gamma)`** - Store conversations/documents with automatic segmentation and boundary refinement
- **`recall_memories(query, scope, format, k)`** - Semantic search across project/global memory with concise or detailed results
- **`remove_memories(event_ids, confirm)`** - Selectively delete specific memories by event IDs (requires confirm=True)
- **`search_memory_batch(queries, k, format)`** - Advanced: High-throughput batch retrieval with adaptive CPU/GPU routing

#### 🔧 System Management
- **`manage_memory(action, options)`** - Administrative operations: stats, retrain, optimize, clear, estimate
- **`transfer_memory(action, path, merge, expected_tokens)`** - Import/export memory archives with optimal nlist hints

#### 📊 Resources
- **`memory://status`** - Comprehensive memory system health with FAISS IVF nlist diagnostics

> **Design Philosophy**: Tools are consolidated around **workflows** (store → recall → remove) rather than low-level primitives. All tools support **format control** (concise vs detailed) for token efficiency.

### Tool Usage Examples

#### Store a Conversation
```python
# AI agent calls this automatically when you ask to "remember this"
store_memory(
    content="Discussed React hooks optimization. useCallback prevents re-renders...",
    metadata={"topic": "react", "date": "2025-10-31"},
    auto_segment=True,  # Automatically splits into semantic episodes with refinement
    gamma=1.0  # Boundary sensitivity (higher = more segments)
)
# Returns: {"event_ids": ["event_1730368800"], "num_segments": 3, "index_status": "optimal"}
```

#### Retrieve Relevant Context
```python
# AI agent calls this when you ask "what did we discuss about React?"
recall_memories(
    query="React hooks optimization techniques",
    scope="project",  # Search current project only
    format="concise",  # Return IDs + snippets (token-efficient)
    k=10
)
# Returns: {"memories": [{"event_id": "...", "snippet": "useCallback prevents...", "relevance_score": 0.92}]}
```

#### Remove Outdated Memories
```python
# Delete specific memories when they're no longer relevant
remove_memories(
    event_ids=["event_1730368800", "event_1730368900"],
    confirm=True  # Safety flag required for deletion
)
# Returns: {"removed_count": 2, "remaining_events": 1245, "index_health": {...}}
```

#### Batch Analysis
```python
# For high-throughput retrieval (testing, bulk analysis)
search_memory_batch(
    queries=["debugging", "performance", "testing"],
    k=5,
    format="concise"  # Adaptive routing: GPU batches all, CPU sequential if <100 queries
)
# Returns: {"results": [...], "performance": {"used_batch_api": true, "routing_reason": "gpu_enabled"}}
```

#### System Maintenance
```python
# Get memory statistics with FAISS nlist diagnostics
manage_memory(action="stats")
# Returns: {"project_events": 1247, "index_info": {"nlist": 184, "optimal_nlist": 184, "nlist_ratio": 1.0}}

# Estimate optimal configuration for expected corpus size
manage_memory(action="estimate", options={"expected_tokens": 60000})
# Returns: {"expected_vectors": 54000, "optimal_nlist": 929, "recommendation": "Set EMX_EXPECTED_TOKENS=60000"}

# Optimize storage (prune least-accessed events)
manage_memory(action="optimize", options={"prune_old_events": True})

# Backup memory with optimal nlist hint
transfer_memory(
    action="export",
    path="/backups/memory-2025-10-31.tar.gz",
    expected_tokens=60000  # Hint for import optimization
)
```

### Configuration

EMX-MCP Server is configured via **environment variables** set in your MCP client configuration:

```json
{
  "mcpServers": {
    "emx-memory": {
      "command": "uvx",
      "args": ["emx-mcp-server"],
      "env": {
        "EMX_MODEL_DEVICE": "cuda",
        "EMX_MEMORY_GAMMA": "1.5",
        "EMX_STORAGE_VECTOR_DIM": "384",
        "EMX_SEGMENTATION_ENABLE_REFINEMENT": "true",
        "EMX_SEGMENTATION_REFINEMENT_METRIC": "modularity"
      }
    }
  }
}
```

**📖 Configuration Documentation:**
- **[ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md)** - Complete reference for all configuration variables:
  - Model selection and hardware acceleration (GPU/CPU)
  - Boundary detection tuning (gamma, context windows, refinement)
  - Memory hierarchy sizing (init/local/episodic tiers)
  - FAISS IVF index configuration (nlist, nprobe, SQ compression)
  - Batch encoding and retrieval optimization
  - Recommended configurations for different use cases

## 📝 How to Build

To build and install the EMX-MCP Server, follow these steps:

```shell
# Open a terminal

# Ensure Python 3.12+ is installed
python --version  # Check the installed version of Python
# Visit https://python.org to install or update if necessary

# Clone the repository
git clone https://github.com/coderdayton/emx-mcp-server.git

# Navigate to the project directory
cd emx-mcp-server

# Install with uv (recommended)
uv sync --dev

# Or install with pip
pip install -e .

# Install sentence-transformers dependency
uv add sentence-transformers
# or pip install sentence-transformers

# Run the server (uses .env if present)
emx-mcp-server

# Or set environment variables inline
EMX_MODEL_DEVICE=cuda EMX_MEMORY_GAMMA=1.5 emx-mcp-server
```

### Memory Structure

- **Project Memory**: `<your-project>/.memories/` (git-ignored, per-project)
- **Global Memory**: `~/.emx-mcp/global_memories/` (shared across projects)

## 🤝 Feedback and Contributions

We've made every effort to implement all the core aspects of the EM-LLM algorithm in the best possible way using embedding-based approaches with graph-theoretic boundary refinement. However, the development journey doesn't end here, and your input is crucial for our continuous improvement.

> Whether you have feedback on features, have encountered any bugs, or have suggestions for enhancements, we're eager to hear from you. Your insights help us make the EMX-MCP Server library more robust and user-friendly.

Please feel free to contribute by [submitting an issue](https://github.com/coderdayton/emx-mcp-server/issues) or [joining the discussions](https://github.com/coderdayton/emx-mcp-server/discussions). Each contribution helps us grow and improve.

We appreciate your support and look forward to making our product even better with your help!

## 📃 License

This project is distributed under the MIT License. You can review the full license agreement at the following link: [MIT License](https://github.com/coderdayton/emx-mcp-server/blob/main/LICENSE.md).

## 🗨️ Contacts

For more details about the EMX-MCP Server project, or any general information regarding implementation and development, feel free to reach out. We are here to provide support and answer any questions you may have. Below are the best ways to contact our team:

- **GitHub Issues**: [Submit an issue](https://github.com/coderdayton/emx-mcp-server/issues) for bugs, feature requests, or questions.
- **Email**: Send inquiries to [coderdayton14@gmail.com](mailto:coderdayton14@gmail.com).
- **Repository**: Visit the official EMX-MCP Server repository: [EMX-MCP Server](https://github.com/coderdayton/emx-mcp-server).

##

<div align="center">

  [Back to top](#top)

</div>