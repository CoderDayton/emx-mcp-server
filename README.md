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

🚀 **Major Release: Embedding-Based Architecture**
- **Complete Embedding-Based Implementation**: Replaced LLM-dependent operations with embedding-based approaches for maximum compatibility
- **Enhanced Surprise Calculation**: Now uses semantic distances from local context centroids for boundary detection
- **Improved Attention Approximation**: Embedding cosine similarity replaces attention keys for boundary refinement
- **Optimized Performance**: 100x faster than previous LLM-based approach while maintaining semantic accuracy

🔧 **New Features**
- **Context Window Support**: Configurable context windows for embedding-based surprise calculation
- **Semantic Shift Detection**: Improved detection of topic changes and phase transitions
- **Comprehensive Test Suite**: 100% test coverage with pytest-asyncio validation
- **Enhanced Configuration**: Flexible model configuration and performance tuning

🧠 **Technical Improvements**
- **Sentence-Transformers Integration**: Native support for all-MiniLM-L6-v2 and compatible models
- **Memory-Efficient Processing**: Reduced memory footprint while maintaining accuracy
- **Backwards Compatibility**: Legacy API support for gradual migration

> **Migration Note**: This release maintains full backwards compatibility. No breaking changes were introduced.

## 🧠 How It Works

### Architecture Overview

The EMX-MCP Server implements the EM-LLM algorithm using an embedding-based approach that replaces LLM-dependent operations with semantic similarity computations:

```
AI Agent (Copilot/Claude/Cursor)
        ↓ MCP Protocol
   EMX-MCP Server
        ├─ EmbeddingEncoder (sentence-transformers)
        │   └─ encode_tokens_with_context()
        ├─ SurpriseSegmenter
        │   ├─ _compute_embedding_surprises()
        │   └─ _compute_embedding_adjacency()
        ├─ ProjectMemoryManager
        │   └─ HierarchicalMemoryStore
        │       ├─ FAISS IVF Vector Store
        │       ├─ SQLite Graph Store
        │       └─ Memory-Mapped Disk Manager
        └─ TwoStageRetrieval
            ├─ Similarity Search
            └─ Temporal Contiguity
```

### Core Algorithm: Embedding-Based Surprise

The system computes **semantic surprise** by measuring embedding distances from local context:

1. **Token Encoding**: Each token is encoded with local context using sentence-transformers
2. **Context Centroid**: Calculate centroid of previous tokens (configurable window size)
3. **Distance Calculation**: Measure Euclidean distance from current embedding to context centroid
4. **Adaptive Threshold**: Use μ + γσ from local window for boundary detection

```python
# Simplified embedding-based surprise calculation
def compute_surprise(token_embeddings, window=10):
    for t in range(window, len(token_embeddings)):
        context = token_embeddings[t-window:t]
        context_centroid = np.mean(context, axis=0)
        distance = np.linalg.norm(token_embeddings[t] - context_centroid)
        surprise[t] = distance
    return surprise
```

### Memory Hierarchy

The system employs a three-tier memory architecture:

- **Tier 1 (Initial)**: First 1000 tokens for attention sinks
- **Tier 2 (Local)**: Recent 4096 tokens in rolling window
- **Tier 3 (Episodic)**: FAISS-indexed events with disk offloading

### Performance Characteristics

- **Embedding Generation**: ~2s for 10,000 tokens (sentence-transformers)
- **Boundary Detection**: Sub-second for sequences up to 100k tokens
- **Memory Retrieval**: <500ms for similarity + contiguity search
- **Storage Scale**: Tested up to 10M vectors with FAISS IVF indexing

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

Once integrated, your AI agent will have access to these memory tools:

#### 📝 Memory Operations
- **`add_episodic_event`** - Store new experiences with automatic embedding
- **`retrieve_memories`** - Search for relevant past experiences
- **`segment_experience`** - Detect episode boundaries in conversations
- **`get_memory_stats`** - View memory usage and statistics

#### 🔧 Maintenance
- **`optimize_memory`** - Clean up and optimize storage
- **`retrain_index`** - Rebuild FAISS index for better performance
- **`export_project_memory`** - Backup memory to file
- **`import_project_memory`** - Restore memory from backup

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
        "EMX_STORAGE_VECTOR_DIM": "384"
      }
    }
  }
}
```

**📖 See [ENVIRONMENT_VARIABLES.md](docs/ENVIRONMENT_VARIABLES.md) for complete documentation** of all 22 configuration variables including:
- Model selection and hardware acceleration
- Boundary detection tuning (gamma, context windows)
- Memory hierarchy sizing (init/local/episodic tiers)
- FAISS index configuration
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

We've made every effort to implement all the core aspects of the EM-LLM algorithm in the best possible way using embedding-based approaches. However, the development journey doesn't end here, and your input is crucial for our continuous improvement.

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