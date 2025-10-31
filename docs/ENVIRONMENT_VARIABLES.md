# Environment Variables Reference

EMX-MCP Server is configured via environment variables. When using with MCP clients (Claude Desktop, Cline, etc.), set these in your client's configuration file under the `env: {}` section.

## Quick Example: MCP Client Configuration

```json
{
  "mcpServers": {
    "emx-memory": {
      "command": "uvx",
      "args": ["emx-mcp-server"],
      "env": {
        "EMX_MODEL_DEVICE": "cuda",
        "EMX_MEMORY_GAMMA": "1.5",
        "EMX_STORAGE_VECTOR_DIM": "768"
      }
    }
  }
}
```

---

## 📋 Complete Variable Reference

### Model Configuration

#### `EMX_MODEL_NAME`
**Default**: `all-MiniLM-L6-v2`  
**Valid Values**: Any HuggingFace sentence-transformers model name  
**Description**: Embedding model for semantic encoding. Must match the vector dimension you configure.

**Common Models**:
- `all-MiniLM-L6-v2` (384-dim, fastest, recommended for most use cases)
- `all-mpnet-base-v2` (768-dim, better quality, slower)
- `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual support)

**Example**:
```json
"EMX_MODEL_NAME": "all-mpnet-base-v2"
```

---

#### `EMX_MODEL_DEVICE`
**Default**: `cpu`  
**Valid Values**: `cpu`, `cuda`  
**Description**: Hardware device for model inference.

**Guidelines**:
- Use `cpu` for compatibility and reasonable performance (<1k tokens/sec)
- Use `cuda` if you have NVIDIA GPU (5-10x faster, critical for >10k token workloads)

**Example**:
```json
"EMX_MODEL_DEVICE": "cuda"
```

---

#### `EMX_MODEL_BATCH_SIZE`
**Default**: `32`  
**Valid Range**: `1` to `512`  
**Description**: Number of tokens processed per inference batch.

**Tuning Guidelines**:
- **CPU**: 16-32 (sweet spot for memory/speed)
- **GPU**: 64-256 (utilize GPU parallelism)
- Lower if you encounter OOM errors
- Higher doesn't always mean faster (diminishing returns)

**Example**:
```json
"EMX_MODEL_BATCH_SIZE": "128"
```

---

### Memory Configuration

#### `EMX_MEMORY_GAMMA`
**Default**: `1.0`  
**Valid Range**: `0.1` to `10.0`  
**Description**: Boundary detection sensitivity (threshold = mean + gamma × stddev).

**Tuning Guidelines**:
- **Lower (0.5-1.0)**: More boundaries, fine-grained episodes (chatty conversations)
- **Higher (1.5-3.0)**: Fewer boundaries, coarse episodes (focused work sessions)
- Start with `1.0` and adjust based on boundary quality

**Example**:
```json
"EMX_MEMORY_GAMMA": "1.5"
```

---

#### `EMX_MEMORY_CONTEXT_WINDOW`
**Default**: `10`  
**Valid Range**: `1` to `100`  
**Description**: Number of preceding tokens used for contextual embedding.

**Tuning Guidelines**:
- **Smaller (5-10)**: Faster, good for short-form content
- **Larger (20-50)**: Better semantic understanding for long-form text
- Affects embedding quality and compute time

**Example**:
```json
"EMX_MEMORY_CONTEXT_WINDOW": "20"
```

---

#### `EMX_MEMORY_WINDOW_OFFSET`
**Default**: `128`  
**Valid Range**: `1` to `2048`  
**Description**: Lookback window for surprise calculation (adaptive threshold).

**Guidelines**:
- Larger window = more stable thresholds but slower adaptation to topic shifts
- 128-256 works well for most conversation patterns

**Example**:
```json
"EMX_MEMORY_WINDOW_OFFSET": "256"
```

---

#### `EMX_MEMORY_MIN_BLOCK_SIZE`
**Default**: `8`  
**Valid Range**: `1` to `1024`  
**Description**: Minimum tokens per episode (prevents micro-segmentation).

**Guidelines**:
- Too small = noisy boundaries from every sentence
- Too large = misses genuine topic shifts
- 8-16 tokens ≈ 1-2 sentences

**Example**:
```json
"EMX_MEMORY_MIN_BLOCK_SIZE": "12"
```

---

#### `EMX_MEMORY_MAX_BLOCK_SIZE`
**Default**: `128`  
**Valid Range**: `1` to `4096`  
**Must be**: ≥ `EMX_MEMORY_MIN_BLOCK_SIZE`  
**Description**: Maximum tokens per episode (forces split for very long segments).

**Guidelines**:
- Prevents runaway episodes in monotonous text
- 128-512 tokens ≈ 1-4 paragraphs

**Example**:
```json
"EMX_MEMORY_MAX_BLOCK_SIZE": "512"
```

---

#### `EMX_MEMORY_N_INIT`
**Default**: `128`  
**Valid Range**: `1` to `10000`  
**Description**: Tier 1 memory - initial tokens always kept (attention sinks).

**Guidelines**:
- Preserves conversation start/system prompts
- 128-256 is sufficient for most use cases

**Example**:
```json
"EMX_MEMORY_N_INIT": "256"
```

---

#### `EMX_MEMORY_N_LOCAL`
**Default**: `4096`  
**Valid Range**: `1` to `100000`  
**Must be**: ≥ `EMX_MEMORY_N_MEM`  
**Description**: Tier 2 memory - recent tokens in rolling window.

**Guidelines**:
- Equivalent to "working memory"
- 4k-8k covers typical multi-turn conversations
- Increase for long coding sessions

**Example**:
```json
"EMX_MEMORY_N_LOCAL": "8192"
```

---

#### `EMX_MEMORY_N_MEM`
**Default**: `2048`  
**Valid Range**: `1` to `100000`  
**Must be**: ≤ `EMX_MEMORY_N_LOCAL`  
**Description**: Tier 3 memory - episodic memories indexed by FAISS.

**Guidelines**:
- Retrieved based on semantic similarity + recency
- 2k-4k balances retrieval quality and speed

**Example**:
```json
"EMX_MEMORY_N_MEM": "4096"
```

---

#### `EMX_MEMORY_REPR_TOPK`
**Default**: `4`  
**Valid Range**: `1` to `50`  
**Description**: Number of representative tokens per episode for indexing.

**Guidelines**:
- Higher = better episode representation but more storage
- 3-5 is optimal for most use cases

**Example**:
```json
"EMX_MEMORY_REPR_TOPK": "5"
```

---

#### `EMX_MEMORY_REFINEMENT_METRIC`
**Default**: `modularity`  
**Valid Values**: `modularity`, `conductance`, `coverage`  
**Description**: Graph-based metric for boundary refinement.

**Guidelines**:
- `modularity`: Best all-around (finds natural clusters)
- `conductance`: Good for overlapping topics
- `coverage`: Maximizes intra-segment similarity

**Example**:
```json
"EMX_MEMORY_REFINEMENT_METRIC": "conductance"
```

---

### Storage Configuration

#### `EMX_STORAGE_VECTOR_DIM`
**Default**: `384`  
**Valid Range**: `1` to `4096`  
**Description**: Embedding vector dimension (MUST match your model).

**⚠️ CRITICAL**: Must match `EMX_MODEL_NAME` output dimension:
- `all-MiniLM-L6-v2` → `384`
- `all-mpnet-base-v2` → `768`
- `paraphrase-multilingual-*` → check model card

**Example**:
```json
"EMX_STORAGE_VECTOR_DIM": "768"
```

---

#### `EMX_STORAGE_NPROBE`
**Default**: `8`  
**Valid Range**: `1` to `1024`  
**Description**: FAISS IVF clusters to search (higher = better recall, slower).

**Tuning Guidelines**:
- **Fast search (latency-critical)**: 4-8
- **Balanced**: 16-32
- **Exhaustive (quality-critical)**: 64-128
- Only matters after index training (>1000 vectors)

**Example**:
```json
"EMX_STORAGE_NPROBE": "16"
```

---

#### `EMX_STORAGE_DISK_OFFLOAD_THRESHOLD`
**Default**: `300000`  
**Valid Range**: `1` to `10000000`  
**Description**: Number of vectors before switching from memory to disk-backed index.

**Guidelines**:
- Lower = saves RAM but slower retrieval
- Higher = faster but RAM-intensive
- 300k vectors ≈ 400MB (for 384-dim floats)

**Example**:
```json
"EMX_STORAGE_DISK_OFFLOAD_THRESHOLD": "500000"
```

---

#### `EMX_STORAGE_MIN_TRAINING_SIZE`
**Default**: `1000`  
**Valid Range**: `100` to `1000000`  
**Description**: Minimum vectors before training IVF index (uses Flat index below this).

**Guidelines**:
- Don't change unless you know FAISS internals
- 1000-5000 is optimal for IVF training stability

**Example**:
```json
"EMX_STORAGE_MIN_TRAINING_SIZE": "2000"
```

---

#### `EMX_STORAGE_INDEX_TYPE`
**Default**: `IVF`  
**Valid Values**: `IVF`, `Flat`, `HNSW`  
**Description**: FAISS index type (IVF = inverted file, Flat = exact, HNSW = graph-based).

**Guidelines**:
- `IVF`: Best for >10k vectors (fast approximate search)
- `Flat`: <5k vectors or when you need exact results
- `HNSW`: Experimental (faster than IVF but more memory)

**Example**:
```json
"EMX_STORAGE_INDEX_TYPE": "Flat"
```

---

#### `EMX_STORAGE_METRIC`
**Default**: `cosine`  
**Valid Values**: `cosine`, `euclidean`, `dot`  
**Description**: Distance metric for vector similarity.

**Guidelines**:
- `cosine`: Best for semantic similarity (normalized)
- `euclidean`: Raw L2 distance (sensitive to magnitude)
- `dot`: Inner product (faster but less intuitive)

**Example**:
```json
"EMX_STORAGE_METRIC": "cosine"
```

---

### Logging Configuration

#### `EMX_LOGGING_LEVEL`
**Default**: `INFO`  
**Valid Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`  
**Description**: Minimum log level to emit.

**Guidelines**:
- `INFO`: Production (shows major operations)
- `DEBUG`: Development (verbose, shows all internal state)
- `WARNING`: Quiet production (errors + warnings only)

**Example**:
```json
"EMX_LOGGING_LEVEL": "DEBUG"
```

---

#### `EMX_LOGGING_FORMAT`
**Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`  
**Valid Values**: Any Python logging format string  
**Description**: Log message format template.

**Example**:
```json
"EMX_LOGGING_FORMAT": "[%(levelname)s] %(message)s"
```

---

### Runtime Paths

#### `EMX_PROJECT_PATH`
**Default**: Current working directory  
**Valid Values**: Any valid filesystem path  
**Description**: Override automatic project detection (uses git root or cwd by default).

**Use Cases**:
- Force specific project when running from subdirectory
- Testing with isolated memory stores
- Multi-project setups

**Example**:
```json
"EMX_PROJECT_PATH": "/home/user/my-project"
```

---

#### `EMX_GLOBAL_PATH`
**Default**: `~/.emx-mcp/global_memories`  
**Valid Values**: Any valid filesystem path  
**Description**: Global memory storage location (shared across all projects).

**Use Cases**:
- Centralized memory for cross-project knowledge
- Custom backup/sync location
- Network-mounted storage

**Example**:
```json
"EMX_GLOBAL_PATH": "/mnt/shared/emx-global"
```

---

## 🔧 How Configuration Works

EMX-MCP Server uses a layered configuration system:

1. **Defaults** - Sensible defaults built into the code
2. **`.env` file** - Shipped with the package, provides base configuration
3. **Environment variables** - Set in MCP client config under `env: {}`; these override `.env` defaults

**You configure by**: Adding variables to your MCP client's `env: {}` section. Only specify what you want to change from defaults.

---

## 📊 Configuration Validation

All environment variables are validated at startup using Pydantic. Invalid values will produce clear error messages:

```
ValidationError: 1 validation error for ModelConfig
batch_size
  Input should be less than or equal to 512 [type=less_than_equal, input_value=1024]
```

**Common validation errors**:
- Range violations (e.g., `EMX_MEMORY_GAMMA=15.0` → must be ≤ 10.0)
- Type mismatches (e.g., `EMX_MODEL_BATCH_SIZE=large` → must be integer)
- Relationship constraints (e.g., `EMX_MEMORY_MIN_BLOCK_SIZE=200` > `EMX_MEMORY_MAX_BLOCK_SIZE=128`)

---

## 🎯 Recommended Configurations

### Minimal (Default)
Fast, low-memory, good for most conversations:
```json
"env": {}
```

### High Quality
Better boundary detection, more context:
```json
"env": {
  "EMX_MEMORY_GAMMA": "1.5",
  "EMX_MEMORY_CONTEXT_WINDOW": "20",
  "EMX_MEMORY_N_LOCAL": "8192"
}
```

### GPU Accelerated
For large-scale workloads:
```json
"env": {
  "EMX_MODEL_DEVICE": "cuda",
  "EMX_MODEL_BATCH_SIZE": "128",
  "EMX_STORAGE_NPROBE": "32"
}
```

### Multilingual
Support for non-English text:
```json
"env": {
  "EMX_MODEL_NAME": "paraphrase-multilingual-MiniLM-L12-v2",
  "EMX_STORAGE_VECTOR_DIM": "384"
}
```

---

## ❓ Troubleshooting

**Q: Changes in `env: {}` aren't taking effect**  
A: Restart your MCP client completely (not just reload server)

**Q: Getting "Model not found" errors**  
A: First run downloads the model from HuggingFace. Check internet connection.

**Q: Out of memory errors**  
A: Reduce `EMX_MODEL_BATCH_SIZE` or `EMX_MEMORY_N_LOCAL`

**Q: Too many/few boundaries detected**  
A: Adjust `EMX_MEMORY_GAMMA` (higher = fewer boundaries, lower = more boundaries)

**Q: Vector dimension mismatch errors**  
A: Ensure `EMX_STORAGE_VECTOR_DIM` matches your `EMX_MODEL_NAME` output dimension
