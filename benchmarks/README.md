# EMX-MCP Benchmarks

Performance benchmarks for EMX-MCP Server's adaptive nlist optimization system.

## Available Benchmarks

### `adaptive_nlist_benchmark.py`

Tests FAISS IVF index performance with adaptive nlist optimization across realistic workload scales.

**What it measures:**
- Index training time
- Vector insertion throughput
- Search latency (p50, p95, p99)
- Optimization event frequency
- Memory usage

**Usage:**

```bash
# Quick test (100k vectors, ~30 seconds)
uv run python benchmarks/adaptive_nlist_benchmark.py --max-vectors 100000

# Standard test (1M vectors, ~5 minutes)
uv run python benchmarks/adaptive_nlist_benchmark.py --max-vectors 1000000

# Large-scale test (10M vectors, ~1 hour)
uv run python benchmarks/adaptive_nlist_benchmark.py --max-vectors 10000000

# With GPU acceleration (if available)
uv run python benchmarks/adaptive_nlist_benchmark.py --device cuda --max-vectors 1000000

# Test without auto-retrain (manual mode)
uv run python benchmarks/adaptive_nlist_benchmark.py --no-auto-retrain

# Custom drift threshold (more aggressive optimization)
uv run python benchmarks/adaptive_nlist_benchmark.py --drift-threshold 1.5

# Save results to JSON
uv run python benchmarks/adaptive_nlist_benchmark.py --output results.json
```

**Expected Results (CPU, 1M vectors):**

| Phase | Vectors | Duration | Throughput | nlist | Optimal | Drift |
|-------|---------|----------|------------|-------|---------|-------|
| Initial | 10k | ~2s | ~5k/s | 32 | 100 | 0.32x |
| Growth | 100k | ~20s | ~4.5k/s | 316 | 316 | 1.0x |
| Scale | 1M | ~200s | ~4k/s | 1000 | 1000 | 1.0x |

**Optimization Events:**
- Typically 2-3 retraining events from 10k→1M vectors
- Each retrain takes ~100-200ms per 100k vectors
- Drift threshold of 2.0x triggers optimization when `current/optimal > 2.0` or `< 0.5`

**Search Latency (1M vectors):**
- p50: 5-10ms
- p95: 15-25ms
- p99: 30-50ms

**Memory Usage:**
- 100k vectors: ~50 MB
- 1M vectors: ~500 MB
- 10M vectors: ~5 GB

## Interpreting Results

### Good Performance Indicators

1. **Throughput stays consistent** (~4k-5k vectors/sec on CPU)
2. **Few optimization events** (2-4 retrains from 10k→1M)
3. **Drift ratio near 1.0x** after each phase
4. **Search latency p95 < 50ms** at 1M vectors

### Performance Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Throughput drops >50% | Too many retrains | Increase `drift_threshold` to 3.0-5.0 |
| Search latency >100ms | nlist too small | Decrease `drift_threshold` to 1.5 |
| Memory usage excessive | Large batch sizes | Reduce batch size in benchmark |
| Many optimizations | Aggressive threshold | Use default 2.0 or higher |

## Configuration Testing

Test different adaptive settings:

```bash
# Conservative (minimize retraining)
uv run python benchmarks/adaptive_nlist_benchmark.py \
  --drift-threshold 5.0 \
  --max-vectors 1000000

# Aggressive (optimize frequently)
uv run python benchmarks/adaptive_nlist_benchmark.py \
  --drift-threshold 1.5 \
  --max-vectors 1000000

# Manual mode (no auto-optimization)
uv run python benchmarks/adaptive_nlist_benchmark.py \
  --no-auto-retrain \
  --max-vectors 1000000
```

## Adding New Benchmarks

When adding new benchmark scripts:

1. **Use realistic data**: Mimic actual embedding distributions (clustered, normalized)
2. **Measure end-to-end**: Include index creation, training, search
3. **Track resources**: Memory, CPU, disk I/O
4. **Document expected results**: Add baseline metrics for comparison
5. **Export results**: Support JSON output for automated testing

## CI Integration

To run benchmarks in CI (quick mode):

```bash
# Fast validation (10k vectors, <10 seconds)
uv run python benchmarks/adaptive_nlist_benchmark.py \
  --max-vectors 10000 \
  --output benchmark_results.json

# Check for regressions
# (Compare against baseline in version control)
```

## Troubleshooting

**"Index not trained yet" warnings:**
- Normal for first 1k vectors (training buffer)
- If persists beyond 1k, check `min_training_size` setting

**OOM errors:**
- Reduce `--max-vectors`
- Benchmark generates vectors in batches to minimize memory
- 1M vectors = ~500MB, 10M = ~5GB

**Slow performance:**
- Check CPU usage (should be near 100% during insertion)
- GPU mode requires `sentence-transformers` with CUDA support
- Disk I/O can bottleneck if `/tmp` is slow (change `store_path` in code)

## Related Documentation

- [Environment Variables](../docs/ENVIRONMENT_VARIABLES.md) - Configuration reference
- [Performance Report](../docs/PERFORMANCE_REPORT.md) - Optimization analysis
- [VectorStore Implementation](../emx_mcp/storage/vector_store.py) - Source code
