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
        "EMX_MEMORY_GAMMA": "1.5"
      }
    }
  }
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
A: Remove `EMX_STORAGE_VECTOR_DIM` from your config to enable auto-detection (recommended). Only set manually for advanced use cases.


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

#### `EMX_SEGMENTATION_ENABLE_REFINEMENT`
**Default**: `true`
**Valid Values**: `true`, `false`
**Description**: Enable graph-theoretic boundary refinement (Algorithm 1 from EM-LLM paper).

**Guidelines**:
- `true`: Apply O(nm) refinement for 10-29% accuracy improvement (recommended)
- `false`: Use O(n) surprise-only segmentation (faster but less accurate)
- Refinement optimizes boundaries using adjacency matrix of token embeddings
- Performance controlled by `EMX_SEGMENTATION_MAX_REFINEMENT_WINDOW`

**Example**:
```json
"EMX_SEGMENTATION_ENABLE_REFINEMENT": "true"
```

---

#### `EMX_SEGMENTATION_REFINEMENT_METRIC`
**Default**: `modularity`
**Valid Values**: `modularity`, `conductance`
**Description**: Graph-based metric for boundary refinement optimization.

**Guidelines**:
- `modularity`: Maximizes community structure Q = (1/4m) Σ [A_ij - (k_i k_j)/(2m)] δ(c_i, c_j) (best all-around)
- `conductance`: Minimizes cut ratio φ(S) = cut(S, V\S) / min(vol(S), vol(V\S)) (good for overlapping topics)
- Both metrics use cosine similarity adjacency from token embeddings
- See EM-LLM paper Section 3.2 for mathematical details

**Example**:
```json
"EMX_SEGMENTATION_REFINEMENT_METRIC": "modularity"
```

---

#### `EMX_SEGMENTATION_MAX_REFINEMENT_WINDOW`
**Default**: `512`
**Valid Range**: `64` to `4096`
**Description**: Maximum tokens per segment for refinement (prevents O(n²) complexity explosion).

**Tuning Guidelines**:
- Refinement is O(nm) where m = segment size (capped at this value)
- **Conservative (256-512)**: Fast refinement, handles typical segments well
- **Aggressive (1024-2048)**: Better quality for long segments but slower
- Segments larger than this threshold skip refinement
- 512 tokens ≈ 3-4 paragraphs, covers 95% of natural segments

**Performance Impact**:
- 256 window: ~5ms per segment
- 512 window: ~20ms per segment
- 1024 window: ~80ms per segment
- Batched across segments for amortized cost

**Example**:
```json
"EMX_SEGMENTATION_MAX_REFINEMENT_WINDOW": "512"
```

---

### Storage Configuration

#### `EMX_STORAGE_VECTOR_DIM`
**Default**: Auto-detected from `EMX_MODEL_NAME` (recommended)
**Valid Range**: `1` to `4096`
**Description**: Embedding vector dimension. Auto-detected from the model by default; only set manually if you need to override.

**Auto-Detection (Recommended)**:
- Leave unset (or omit from `env: {}`), and the server will automatically detect the dimension from your model
- Eliminates manual configuration errors
- Safer: impossible to mismatch model and index dimensions

**Manual Override (Advanced)**:
- Only needed for edge cases (custom models, pre-existing indexes)
- Common dimensions:
  - `all-MiniLM-L6-v2` → `384`
  - `all-mpnet-base-v2` → `768`
  - `paraphrase-multilingual-*` → check model card
- ⚠️ If set incorrectly, will cause dimension mismatch errors at runtime

**Example (auto-detect, recommended)**:
```json
"env": {
  "EMX_MODEL_NAME": "all-mpnet-base-v2"
}
```

**Example (manual override)**:
```json
"env": {
  "EMX_MODEL_NAME": "all-mpnet-base-v2",
  "EMX_STORAGE_VECTOR_DIM": "768"
}
```

---

#### `EMX_STORAGE_NPROBE`
**Default**: `16`
**Valid Range**: `1` to `1024`
**Description**: FAISS IVF clusters to search (higher = better recall, slower).

**Tuning Guidelines**:
- **Fast search (latency-critical)**: 4-8
- **Balanced (recommended)**: 16-32 (optimal for SQ8 quantization)
- **Exhaustive (quality-critical)**: 64-128
- Default increased to 16 for better recall with scalar quantization
- Only matters after index training (>1000 vectors)

**Example**:
```json
"EMX_STORAGE_NPROBE": "16"
```

---

#### `EMX_STORAGE_USE_SQ`
**Default**: `true`
**Valid Values**: `true`, `false`
**Description**: Enable 8-bit Scalar Quantization (SQ8) for 4x memory compression.

**Guidelines**:
- `true`: Use SQ8 quantization (4x smaller index, minimal accuracy loss <2%)
- `false`: Use full float32 vectors (4x more memory but maximum precision)
- SQ8 converts float32 vectors to uint8 with per-dimension min/max scaling
- Recommended for production (dramatically reduces memory without quality loss)
- Requires `EMX_STORAGE_NPROBE >= 16` for optimal recall

**Memory Impact Example**:
- 1M vectors × 384 dim: 1.5GB float32 → 375MB SQ8 (75% savings)
- 10M vectors × 768 dim: 30GB float32 → 7.5GB SQ8 (75% savings)

**Example**:
```json
"EMX_STORAGE_USE_SQ": "true"
```

---

#### `EMX_STORAGE_EXPECTED_TOKENS`
**Default**: `null` (auto-scale based on actual data)
**Valid Range**: `1000` to `100000000`
**Description**: Expected total tokens for optimal FAISS nlist pre-calculation.

**Guidelines**:
- If set, calculates optimal nlist as `4 * sqrt(expected_vectors)` at initialization
- If unset, nlist adapts dynamically as data grows (recommended for most cases)
- Use when you know corpus size upfront for optimal index structure from start
- Prevents suboptimal nlist during initial growth phase
- Expected vectors ≈ `expected_tokens * 0.9` (90% encoding efficiency)

**Calculation Example**:
- 60k tokens → ~54k vectors → optimal nlist = 929
- 1M tokens → ~900k vectors → optimal nlist = 3795

**Example**:
```json
"EMX_STORAGE_EXPECTED_TOKENS": "60000"
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

#### `EMX_STORAGE_AUTO_RETRAIN`
**Default**: `true`
**Valid Values**: `true`, `false`
**Description**: Enable automatic IVF index retraining when nlist drift is detected.

**Guidelines**:
- `true`: Index automatically optimizes as vectors grow (recommended for production)
- `false`: Manual retraining only (useful for testing or controlled environments)
- Retraining triggers when current nlist drifts beyond `EMX_STORAGE_NLIST_DRIFT_THRESHOLD`
- Each optimization event is logged to `.emx_optimization_history.json`

**How It Works**:
- System calculates optimal nlist as `4 * sqrt(n_vectors)` bounded by `[128, n_vectors/39]`
- Compares current nlist to optimal: if `ratio > threshold`, retrain index
- Typical progression: 1k vectors → nlist=128, 10k → nlist=400, 100k → nlist=1265
- Formula changed from `sqrt(n)` to `4*sqrt(n)` for better IVF partitioning with SQ8

**Example**:
```json
"EMX_STORAGE_AUTO_RETRAIN": "true"
```

---

#### `EMX_STORAGE_NLIST_DRIFT_THRESHOLD`
**Default**: `2.0`
**Valid Range**: `1.1` to `10.0`
**Description**: Ratio threshold for triggering automatic index retraining.

**Tuning Guidelines**:
- **Conservative (1.5-2.0)**: Keeps index near-optimal (more retraining, better performance)
- **Moderate (2.0-3.0)**: Balanced approach (default, good for most workloads)
- **Aggressive (3.0-10.0)**: Tolerates drift (fewer retrains, accepts performance degradation)
- Lower threshold = more frequent retraining = better search quality
- Higher threshold = less overhead but suboptimal nlist longer

**Performance Impact**:
- Retraining cost: ~100ms per 10k vectors (one-time operation)
- Optimal nlist improves search speed 2-5x and recall by 5-15%
- Drift >5x can degrade search quality significantly

**Example (aggressive optimization)**:
```json
"EMX_STORAGE_NLIST_DRIFT_THRESHOLD": "1.5"
```

**Example (conservative, minimize retraining)**:
```json
"EMX_STORAGE_NLIST_DRIFT_THRESHOLD": "5.0"
```

---

### Batch Encoding Configuration

#### `EMX_MEMORY_BATCH_EVENT_THRESHOLD`
**Default**: `10`
**Valid Range**: `1` to `100`
**Description**: Number of events to buffer before flushing to storage (batch write optimization).

**Tuning Guidelines**:
- **Low latency (1-5)**: Events written immediately, minimal buffering (good for real-time apps)
- **Balanced (10-20)**: Amortizes write costs while keeping latency reasonable (recommended)
- **High throughput (50-100)**: Maximum write efficiency for bulk ingestion
- Lower threshold = more frequent writes = lower memory but more I/O
- Higher threshold = fewer writes = better throughput but more memory

**Performance Impact**:
- Threshold=1: ~100 events/sec (one write per event)
- Threshold=10: ~400 events/sec (batched writes)
- Threshold=50: ~600 events/sec (optimal for bulk loads)

**Example**:
```json
"EMX_MEMORY_BATCH_EVENT_THRESHOLD": "10"
```

---

#### `EMX_MEMORY_BATCH_ENCODING_THRESHOLD`
**Default**: `50`
**Valid Range**: `1` to `10000`
**Description**: Minimum tokens required to use batch encoding (falls back to per-token encoding below this).

**Tuning Guidelines**:
- Batch encoding has ~50ms fixed overhead that only pays off for larger inputs
- **Small conversations (<50 tokens)**: Per-token encoding is faster
- **Medium conversations (50-500 tokens)**: Crossover point, batch encoding starts winning
- **Large conversations (>500 tokens)**: Batch encoding provides 1.4x speedup
- Lower if you have very efficient batch processing, higher for CPU-bound systems

**Performance Example (60k tokens)**:
- Per-token: 187.43s (320 tokens/sec)
- Batch (threshold=50): 138.79s (433 tokens/sec) - 1.35x faster

**Example**:
```json
"EMX_MEMORY_BATCH_ENCODING_THRESHOLD": "50"
```

---

### GPU Optimization Configuration

#### `EMX_GPU_ENABLE_PINNED_MEMORY`
**Default**: `true`
**Valid Values**: `true`, `false`
**Description**: Enable pinned memory pool for async CPU→GPU transfers.

**Guidelines**:
- `true`: Use pinned memory for non-blocking transfers (recommended if PyTorch + CUDA available)
- `false`: Disable pinned memory (fallback for CPU-only systems or PyTorch unavailable)
- Only beneficial for batch_size ≥ 32 due to allocation overhead
- Requires PyTorch with CUDA support

**Example**:
```json
"EMX_GPU_ENABLE_PINNED_MEMORY": "true"
```

---

#### `EMX_GPU_PINNED_BUFFER_SIZE`
**Default**: `4`
**Valid Range**: `1` to `32`
**Description**: Number of reusable pinned memory buffers in pool.

**Tuning Guidelines**:
- **Low concurrency (1-2 parallel ops)**: 2-4 buffers
- **Medium concurrency (3-5 parallel ops)**: 4-8 buffers
- **High concurrency (6+ parallel ops)**: 8-16 buffers
- Each buffer pre-allocates `pinned_max_batch × vector_dim × 4 bytes` (e.g., 128 × 384 × 4 = 196KB per buffer)
- More buffers = more memory but less blocking on acquire()

**Example**:
```json
"EMX_GPU_PINNED_BUFFER_SIZE": "8"
```

---

#### `EMX_GPU_PINNED_MAX_BATCH`
**Default**: `128`
**Valid Range**: `32` to `512`
**Description**: Maximum batch size per pinned buffer.

**Tuning Guidelines**:
- Should match or exceed `EMX_MODEL_BATCH_SIZE`
- Higher values allow larger batches but increase per-buffer memory
- 128-256 is optimal for most GPU workloads

**Example**:
```json
"EMX_GPU_PINNED_MAX_BATCH": "256"
```

---

#### `EMX_GPU_PINNED_MIN_BATCH_THRESHOLD`
**Default**: `32`
**Valid Range**: `1` to `256`
**Description**: Minimum batch size to use pinned memory (falls back to regular tensors below this).

**Tuning Guidelines**:
- Pinned memory has allocation overhead that only pays off for larger batches
- 32-64 is the crossover point where async transfer benefits exceed overhead
- Lower if you have very fast DMA, higher if CPU-GPU transfers are slow

**Example**:
```json
"EMX_GPU_PINNED_MIN_BATCH_THRESHOLD": "64"
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
Better boundary detection with refinement, more context:
```json
"env": {
  "EMX_MEMORY_GAMMA": "1.5",
  "EMX_MEMORY_CONTEXT_WINDOW": "20",
  "EMX_MEMORY_N_LOCAL": "8192",
  "EMX_SEGMENTATION_ENABLE_REFINEMENT": "true",
  "EMX_SEGMENTATION_REFINEMENT_METRIC": "modularity"
}
```

### GPU Accelerated
For large-scale workloads with SQ8 compression:
```json
"env": {
  "EMX_MODEL_DEVICE": "cuda",
  "EMX_MODEL_BATCH_SIZE": "128",
  "EMX_STORAGE_NPROBE": "16",
  "EMX_STORAGE_USE_SQ": "true",
  "EMX_GPU_ENABLE_PINNED_MEMORY": "true",
  "EMX_GPU_PINNED_BUFFER_SIZE": "8"
}
```

### Multilingual
Support for non-English text:
```json
"env": {
  "EMX_MODEL_NAME": "paraphrase-multilingual-MiniLM-L12-v2"
}
```

### Large-Scale Production
Optimized for millions of vectors with adaptive index management, SQ8 compression, and batch optimization:
```json
"env": {
  "EMX_MODEL_DEVICE": "cuda",
  "EMX_MODEL_BATCH_SIZE": "128",
  "EMX_STORAGE_NPROBE": "16",
  "EMX_STORAGE_USE_SQ": "true",
  "EMX_STORAGE_AUTO_RETRAIN": "true",
  "EMX_STORAGE_NLIST_DRIFT_THRESHOLD": "1.5",
  "EMX_STORAGE_DISK_OFFLOAD_THRESHOLD": "500000",
  "EMX_STORAGE_EXPECTED_TOKENS": "1000000",
  "EMX_MEMORY_BATCH_EVENT_THRESHOLD": "20",
  "EMX_MEMORY_BATCH_ENCODING_THRESHOLD": "50",
  "EMX_SEGMENTATION_ENABLE_REFINEMENT": "true"
}
```
