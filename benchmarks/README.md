# EMX Benchmarks

Comprehensive benchmarks for measuring EMX-LLM memory system performance across different workloads and configurations.

## Available Benchmarks

### 1. E2E 30k Token Benchmark

**File**: `e2e_30k_tokens_benchmark.py`

Simulates real-world LLM usage by processing a 30,000 token multi-domain conversation through the complete pipeline:

**Pipeline Stages Measured**:
- **Tokenization**: Whitespace-based splitting
- **Segmentation**: O(n) linear boundary detection with configurable gamma
- **Embedding**: sentence-transformers model inference (CPU/GPU)
- **Indexing**: FAISS IVF with adaptive nlist
- **Retrieval**: Semantic similarity + temporal contiguity expansion

**Corpus Structure** (4 semantic domains):
- Technical debugging session (~7k tokens)
- Product planning discussion (~8k tokens)
- Code review with architectural debate (~10k tokens)
- Performance optimization analysis (~5k tokens)

**Metrics Collected**:
- End-to-end latency (remember + recall phases)
- Per-stage timing breakdowns
- Memory footprint (RSS delta in MB)
- Segmentation accuracy (number of boundaries detected)
- Index health (training status, nlist, vector count)
- Retrieval latency (avg ms per query)

## Running Benchmarks

### Prerequisites

Install dependencies:
```bash
uv sync
```

Ensure you have a valid `.env` configuration (or use defaults):
```bash
# Optional: Configure for GPU acceleration
export EMX_MODEL_DEVICE=cuda
export EMX_MODEL_BATCH_SIZE=64

# Optional: Tune segmentation sensitivity
export EMX_MEMORY_GAMMA=1.0
```

### Execute E2E Benchmark

```bash
# Run with default config
python benchmarks/e2e_30k_tokens_benchmark.py

# Or use uv
uv run benchmarks/e2e_30k_tokens_benchmark.py
```

**Output**:
- Real-time progress printed to stdout
- Final summary with key metrics
- JSON results saved to `benchmarks/benchmark_results.json`

### Example Output

```
======================================================================
EMX E2E Benchmark: 30k Token Corpus
======================================================================
Generated corpus: 30,247 tokens (target: 30,000)

=== REMEMBER PHASE ===
Tokenized: 30,247 tokens in 12.45ms
Segmented: 8 events in 245.32ms
  Method: linear_surprise
  Boundaries: [0, 4102, 8334, 12891, 18765, 22103, 25890, 28456, 30247]
Stored: 8 events in 3.42s
  Memory delta: +87.3 MB
  Index: trained=True, vectors=8, nlist=2

=== RECALL PHASE ===
  Query: 'debugging database connection pool exhaustion...' → 10 results in 18.23ms
  Query: 'JWT authentication security best practices...' → 10 results in 15.67ms
  Query: 'performance optimization JSON serialization...' → 10 results in 16.92ms
  Query: 'file attachment feature roadmap planning...' → 10 results in 14.88ms
  Query: 'rate limiting token refresh endpoint...' → 10 results in 17.41ms
Completed: 5 queries in 0.08s (avg: 16.62ms per query)

======================================================================
BENCHMARK SUMMARY
======================================================================
Total tokens processed: 30,247
Segmentation: 8 events in 245.32ms
Storage (embed + index): 3.42s
E2E Remember latency: 3.68s
Memory footprint delta: +87.3 MB
Retrieval (avg): 16.62ms per query
Device: cuda
Index: trained=True, nlist=2, vectors=8
```

## Interpreting Results

### Remember Phase Performance

**Target**: `< 5s` for 30k tokens (typical LLM conversation length)

**Bottlenecks**:
- **Embedding generation**: Dominant cost (~70-80% of total)
  - GPU acceleration provides 3-5x speedup over CPU
  - Batch size matters: 64 is optimal for most GPUs
- **Segmentation**: Should be `< 500ms` (O(n) linear complexity)
- **Index training**: Triggered when vector count exceeds `min_training_size` (default: 1000)

**Memory Usage**:
- Expect `~3MB per 1000 tokens` for embeddings (384-dim float32)
- RSS delta includes model weights (~90MB for `all-MiniLM-L6-v2`)

### Recall Phase Performance

**Target**: `< 50ms p95` for k=10 queries

**Factors**:
- **Index type**: IVF faster than Flat for >10k vectors
- **nprobe**: Higher = more accurate but slower (default: 8)
- **Contiguity expansion**: Adds temporal context but increases latency

### Segmentation Quality

**Expected boundaries**: 4-12 segments for 30k token corpus with gamma=1.0

**Tuning gamma**:
- `gamma < 1.0`: Fewer, larger segments (more tolerance)
- `gamma > 1.0`: More, smaller segments (higher sensitivity)

If you see only 1-2 segments, increase gamma or verify corpus has semantic diversity.

## Configuration Testing

Test different configurations by setting environment variables:

### CPU vs GPU Comparison
```bash
# CPU baseline
EMX_MODEL_DEVICE=cpu python benchmarks/e2e_30k_tokens_benchmark.py

# GPU accelerated
EMX_MODEL_DEVICE=cuda python benchmarks/e2e_30k_tokens_benchmark.py
```

### Segmentation Sensitivity
```bash
# Coarse segmentation (fewer events)
EMX_MEMORY_GAMMA=0.5 python benchmarks/e2e_30k_tokens_benchmark.py

# Fine segmentation (more events)
EMX_MEMORY_GAMMA=2.0 python benchmarks/e2e_30k_tokens_benchmark.py
```

### Batch Size Optimization
```bash
# Small batches (lower memory, slower)
EMX_MODEL_BATCH_SIZE=32 python benchmarks/e2e_30k_tokens_benchmark.py

# Large batches (higher memory, faster)
EMX_MODEL_BATCH_SIZE=128 python benchmarks/e2e_30k_tokens_benchmark.py
```

## CI Integration

To track performance regressions, save baseline results:

```bash
# Run benchmark and save results
python benchmarks/e2e_30k_tokens_benchmark.py

# Compare with baseline
python benchmarks/compare_results.py \
  --baseline benchmarks/baseline_results.json \
  --current benchmarks/benchmark_results.json
```

**Regression thresholds**:
- E2E latency increase > 20% → investigate
- Memory footprint increase > 30% → investigate
- Retrieval latency increase > 50% → investigate

## Troubleshooting

### Benchmark Fails with CUDA OOM

Reduce batch size or switch to CPU:
```bash
EMX_MODEL_BATCH_SIZE=32 python benchmarks/e2e_30k_tokens_benchmark.py
```

### Segmentation Produces Only 1 Event

Increase gamma sensitivity:
```bash
EMX_MEMORY_GAMMA=2.0 python benchmarks/e2e_30k_tokens_benchmark.py
```

### Retrieval Returns Empty Results

Check index training status in output. If `trained=False`, not enough vectors:
- The benchmark should automatically train with 8+ events
- Minimum training size is 1000 vectors (configurable via `EMX_STORAGE_MIN_TRAINING_SIZE`)

### Import Errors

Ensure dependencies installed:
```bash
uv sync
```

## Future Benchmarks

**Planned**:
- `batch_search_benchmark.py`: High-throughput batch retrieval (100-1000 queries)
- `scaling_benchmark.py`: Memory growth with corpus size (10k → 1M tokens)
- `latency_percentiles_benchmark.py`: p50/p95/p99 latency tracking
- `gpu_stream_benchmark.py`: CUDA stream parallelism efficiency

## Results Archive

Historical results stored in `benchmarks/archive/` for regression tracking:
```
archive/
  2025-10-31_cuda_baseline.json
  2025-10-31_cpu_baseline.json
  2025-11-01_after_ivf_optimization.json
```
