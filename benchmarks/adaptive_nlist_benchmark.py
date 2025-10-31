#!/usr/bin/env python3
"""
Real-World Adaptive nlist Benchmark
=====================================

Tests FAISS IVF index performance with adaptive nlist optimization across
realistic workload scales (100k → 1M → 10M vectors).

Measures:
- Index training time
- Vector insertion throughput
- Search latency (p50, p95, p99)
- Optimization event frequency
- Memory usage

Usage:
    python benchmarks/adaptive_nlist_benchmark.py [--max-vectors N] [--device cpu|cuda]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import tracemalloc

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emx_mcp.storage.vector_store import VectorStore
from emx_mcp.embeddings.encoder import EmbeddingEncoder


class BenchmarkResult:
    """Container for benchmark metrics."""
    
    def __init__(self):
        self.phases: List[Dict] = []
        self.optimization_events: List[Dict] = []
        self.search_latencies: Dict[str, List[float]] = {}
        self.memory_snapshots: List[Tuple[int, float]] = []
    
    def add_phase(self, phase: str, n_vectors: int, duration: float, 
                  throughput: float, nlist: int, optimal_nlist: int):
        """Record a benchmark phase."""
        self.phases.append({
            "phase": phase,
            "n_vectors": n_vectors,
            "duration_sec": round(duration, 3),
            "throughput_vecs_per_sec": round(throughput, 1),
            "current_nlist": nlist,
            "optimal_nlist": optimal_nlist,
            "drift_ratio": round(nlist / optimal_nlist, 2) if optimal_nlist > 0 else 0
        })
    
    def add_optimization(self, n_vectors: int, old_nlist: int, new_nlist: int, 
                        duration: float):
        """Record an optimization event."""
        self.optimization_events.append({
            "at_n_vectors": n_vectors,
            "old_nlist": old_nlist,
            "new_nlist": new_nlist,
            "duration_sec": round(duration, 3)
        })
    
    def add_search_latency(self, n_vectors: int, latency_ms: float):
        """Record a search latency measurement."""
        key = str(n_vectors)
        if key not in self.search_latencies:
            self.search_latencies[key] = []
        self.search_latencies[key].append(latency_ms)
    
    def add_memory_snapshot(self, n_vectors: int, memory_mb: float):
        """Record memory usage."""
        self.memory_snapshots.append((n_vectors, memory_mb))
    
    def compute_search_percentiles(self) -> Dict:
        """Compute p50, p95, p99 for each vector count."""
        percentiles = {}
        for n_vectors_str, latencies in self.search_latencies.items():
            if not latencies:
                continue
            sorted_lat = sorted(latencies)
            percentiles[n_vectors_str] = {
                "p50": round(np.percentile(sorted_lat, 50), 2),
                "p95": round(np.percentile(sorted_lat, 95), 2),
                "p99": round(np.percentile(sorted_lat, 99), 2),
                "mean": round(np.mean(sorted_lat), 2),
                "samples": len(latencies)
            }
        return percentiles
    
    def to_dict(self) -> Dict:
        """Export results as dictionary."""
        return {
            "phases": self.phases,
            "optimization_events": self.optimization_events,
            "search_latency_percentiles": self.compute_search_percentiles(),
            "memory_usage_mb": [
                {"n_vectors": n, "memory_mb": round(mem, 1)} 
                for n, mem in self.memory_snapshots
            ]
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        print("\n📊 INSERTION PHASES:")
        print("-" * 70)
        for phase in self.phases:
            print(f"  {phase['phase']:20s} | {phase['n_vectors']:>10,} vectors")
            print(f"    Duration:   {phase['duration_sec']:>8.2f}s  |  "
                  f"Throughput: {phase['throughput_vecs_per_sec']:>10,.0f} vec/s")
            print(f"    nlist:      {phase['current_nlist']:>8}     |  "
                  f"Optimal:    {phase['optimal_nlist']:>10}")
            print(f"    Drift:      {phase['drift_ratio']:>8.2f}x")
            print()
        
        print("\n🔄 OPTIMIZATION EVENTS:")
        print("-" * 70)
        if self.optimization_events:
            for opt in self.optimization_events:
                print(f"  @ {opt['at_n_vectors']:>10,} vectors: "
                      f"nlist {opt['old_nlist']:>4} → {opt['new_nlist']:>4} "
                      f"({opt['duration_sec']:.3f}s)")
        else:
            print("  (No optimizations triggered)")
        
        print("\n⚡ SEARCH LATENCY (ms):")
        print("-" * 70)
        percentiles = self.compute_search_percentiles()
        for n_vectors_str in sorted(percentiles.keys(), key=lambda x: int(x)):
            stats = percentiles[n_vectors_str]
            print(f"  {int(n_vectors_str):>10,} vectors: "
                  f"p50={stats['p50']:>6.2f}  p95={stats['p95']:>6.2f}  "
                  f"p99={stats['p99']:>6.2f}  (n={stats['samples']})")
        
        print("\n💾 MEMORY USAGE:")
        print("-" * 70)
        if self.memory_snapshots:
            for n_vectors, mem_mb in self.memory_snapshots:
                print(f"  {n_vectors:>10,} vectors: {mem_mb:>8.1f} MB")
        
        print("\n" + "="*70 + "\n")


def generate_realistic_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """
    Generate synthetic embeddings with realistic properties:
    - Clustered structure (simulates semantic topics)
    - Normalized (unit vectors like sentence-transformers output)
    """
    # Create 10 cluster centroids
    n_clusters = min(10, n // 100)
    centroids = np.random.randn(n_clusters, dim).astype('float32')
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Assign vectors to clusters and add noise
    cluster_assignments = np.random.randint(0, n_clusters, size=n)
    vectors = centroids[cluster_assignments].copy()
    noise = np.random.randn(n, dim).astype('float32') * 0.1  # 10% noise
    vectors += noise
    
    # Re-normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    
    return vectors


def benchmark_insertion_phase(
    vector_store: VectorStore,
    vectors: np.ndarray,
    ids: List[str],
    metadata: List[Dict],
    phase_name: str,
    result: BenchmarkResult
) -> Tuple[int, int]:
    """Benchmark vector insertion with adaptive optimization tracking."""
    n_vectors = len(vectors)
    
    # Record optimization state before
    info_before = vector_store.get_info()
    nlist_before = info_before.get("nlist", 0)
    
    # Time the insertion
    start_time = time.perf_counter()
    vector_store.add_vectors(vectors, ids, metadata)
    duration = time.perf_counter() - start_time
    
    # Record optimization state after
    info_after = vector_store.get_info()
    nlist_after = info_after.get("nlist", 0)
    optimal_nlist = vector_store._calculate_optimal_nlist(
        info_after.get("total_vectors", 0)
    )
    
    # Check if optimization occurred
    if nlist_after != nlist_before and nlist_before > 0:
        result.add_optimization(
            n_vectors=info_after.get("total_vectors", 0),
            old_nlist=nlist_before,
            new_nlist=nlist_after,
            duration=duration  # Includes optimization time
        )
    
    # Record phase metrics
    throughput = n_vectors / duration if duration > 0 else 0
    result.add_phase(
        phase=phase_name,
        n_vectors=info_after.get("total_vectors", 0),
        duration=duration,
        throughput=throughput,
        nlist=nlist_after,
        optimal_nlist=optimal_nlist
    )
    
    return nlist_before, nlist_after


def benchmark_search_latency(
    vector_store: VectorStore,
    query_vectors: np.ndarray,
    n_vectors: int,
    result: BenchmarkResult,
    k: int = 10,
    n_queries: int = 100
):
    """Benchmark search latency with percentile analysis."""
    n_queries = min(n_queries, len(query_vectors))
    
    for i in range(n_queries):
        query = query_vectors[i:i+1]
        start_time = time.perf_counter()
        vector_store.search(query, k=k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        result.add_search_latency(n_vectors, latency_ms)


def measure_memory_usage() -> float:
    """Measure current memory usage in MB."""
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024
    return 0.0


def run_benchmark(
    max_vectors: int = 1_000_000,
    device: str = "cpu",
    auto_retrain: bool = True,
    drift_threshold: float = 2.0
) -> BenchmarkResult:
    """
    Execute full benchmark across multiple scales.
    
    Progression:
    - Phase 1: 10k vectors (initial training)
    - Phase 2: 100k vectors (first optimization likely)
    - Phase 3: 1M vectors (multiple optimizations)
    - Phase 4: 10M vectors (if max_vectors allows)
    """
    print("\n" + "="*70)
    print(f"ADAPTIVE NLIST BENCHMARK")
    print("="*70)
    print(f"  Max vectors:      {max_vectors:,}")
    print(f"  Device:           {device}")
    print(f"  Auto-retrain:     {auto_retrain}")
    print(f"  Drift threshold:  {drift_threshold}x")
    print("="*70 + "\n")
    
    # Initialize
    result = BenchmarkResult()
    tracemalloc.start()
    
    # Create temporary vector store
    store_path = Path("/tmp/emx_benchmark_store")
    store_path.mkdir(exist_ok=True)
    
    vector_store = VectorStore(
        storage_path=str(store_path),
        dimension=384,
        auto_retrain=auto_retrain,
        nlist_drift_threshold=drift_threshold
    )
    
    # Define phases based on max_vectors
    phases = [
        (10_000, "Phase 1: Initial"),
        (100_000, "Phase 2: Growth"),
        (1_000_000, "Phase 3: Scale"),
    ]
    if max_vectors >= 10_000_000:
        phases.append((10_000_000, "Phase 4: Large-Scale"))
    
    # Filter phases that exceed max_vectors
    phases = [(n, name) for n, name in phases if n <= max_vectors]
    
    total_inserted = 0
    query_vectors = generate_realistic_embeddings(1000, dim=384)  # Reuse for queries
    
    for phase_size, phase_name in phases:
        print(f"\n🚀 Starting {phase_name}: inserting {phase_size:,} vectors...")
        
        # Generate vectors in batches to avoid memory issues
        batch_size = 10_000
        n_batches = phase_size // batch_size
        
        for batch_idx in range(n_batches):
            vectors = generate_realistic_embeddings(batch_size, dim=384)
            ids = [f"event_{total_inserted + i}" for i in range(batch_size)]
            metadata = [{"batch": batch_idx, "idx": i} for i in range(batch_size)]
            
            # Benchmark this batch
            batch_name = f"{phase_name} (batch {batch_idx+1}/{n_batches})"
            benchmark_insertion_phase(
                vector_store, vectors, ids, metadata, batch_name, result
            )
            
            total_inserted += batch_size
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"  ✓ Inserted {total_inserted:,} / {phase_size:,} vectors")
        
        # Measure search latency at this scale
        print(f"  📊 Measuring search latency at {total_inserted:,} vectors...")
        benchmark_search_latency(
            vector_store, query_vectors, total_inserted, result,
            n_queries=100
        )
        
        # Measure memory
        mem_mb = measure_memory_usage()
        result.add_memory_snapshot(total_inserted, mem_mb)
        print(f"  💾 Memory usage: {mem_mb:.1f} MB")
    
    tracemalloc.stop()
    
    # Cleanup
    import shutil
    shutil.rmtree(store_path, ignore_errors=True)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark adaptive nlist optimization with realistic workloads"
    )
    parser.add_argument(
        "--max-vectors", 
        type=int, 
        default=1_000_000,
        help="Maximum number of vectors to test (default: 1M)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for embedding generation (default: cpu)"
    )
    parser.add_argument(
        "--no-auto-retrain",
        action="store_true",
        help="Disable automatic retraining (test manual mode)"
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=2.0,
        help="nlist drift threshold for retraining (default: 2.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save JSON results to file"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    result = run_benchmark(
        max_vectors=args.max_vectors,
        device=args.device,
        auto_retrain=not args.no_auto_retrain,
        drift_threshold=args.drift_threshold
    )
    
    # Print summary
    result.print_summary()
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
