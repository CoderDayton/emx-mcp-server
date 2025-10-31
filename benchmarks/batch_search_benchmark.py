"""Benchmark batch search performance for VectorStore.

Tests batch search throughput improvements compared to sequential search
across different batch sizes and index sizes.
"""

import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from emx_mcp.storage.vector_store import VectorStore
from emx_mcp.embeddings.encoder import EmbeddingEncoder


def generate_sample_corpus(n_docs: int) -> list[str]:
    """Generate diverse sample text corpus."""
    templates = [
        "Machine learning algorithms process {topic} efficiently.",
        "Python is used for {topic} and data science applications.",
        "Neural networks learn {topic} from training datasets.",
        "Deep learning models handle {topic} with high accuracy.",
        "Natural language processing analyzes {topic} in text.",
        "Computer vision systems interpret {topic} from images.",
        "Reinforcement learning optimizes {topic} through experience.",
        "Data preprocessing improves {topic} for better results.",
        "Feature engineering enhances {topic} in predictive models.",
        "Cross-validation prevents {topic} in machine learning.",
    ]
    
    topics = [
        "large datasets", "complex patterns", "sequential data", "spatial features",
        "semantic meaning", "visual information", "temporal sequences", "structured data",
        "high-dimensional spaces", "sparse representations", "dense embeddings", 
        "categorical variables", "numerical features", "text corpora", "image collections",
        "time series", "graph structures", "hierarchical data", "multi-modal inputs",
        "streaming data", "batch processing", "real-time inference", "model optimization",
    ]
    
    corpus = []
    for i in range(n_docs):
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        corpus.append(template.format(topic=topic))
    
    return corpus


def benchmark_batch_vs_sequential(
    n_index_vectors: int,
    batch_sizes: list[int],
    k: int = 10,
    use_gpu: bool = False,
):
    """Benchmark batch search vs sequential search."""
    print(f"\n{'='*80}")
    print(f"Benchmark: {n_index_vectors} vectors, k={k}, GPU={'ON' if use_gpu else 'OFF'}")
    print(f"{'='*80}\n")
    
    # Initialize
    encoder = EmbeddingEncoder(
        model_name="all-MiniLM-L6-v2",
        device="cuda" if use_gpu else "cpu",
        batch_size=32,
    )
    
    tmp_path = Path("/tmp/batch_search_benchmark")
    tmp_path.mkdir(exist_ok=True)
    
    store = VectorStore(
        dimension=384,
        storage_path=str(tmp_path / f"vectors_{n_index_vectors}"),
        use_gpu=use_gpu,
        nlist_formula="sqrt",  # Default formula
    )
    
    # Generate and add corpus
    print(f"Generating {n_index_vectors} sample vectors...")
    corpus = generate_sample_corpus(n_index_vectors)
    # encode_batch expects List[List[str]], so wrap each text as token list
    embeddings = encoder.encode_batch([text.split() for text in corpus])
    event_ids = [f"event_{i}" for i in range(n_index_vectors)]
    metadata = [{"text": text, "index": i} for i, text in enumerate(corpus)]
    
    store.add_vectors(embeddings, event_ids, metadata)
    
    print(f"Index trained: {store.index.is_trained}")
    print(f"Total vectors: {store.index.ntotal}")
    print(f"Recommended batch size: {store.get_recommended_batch_size()}")
    
    # Test different batch sizes
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # Generate queries
        query_texts = [
            f"machine learning topic {i}" for i in range(batch_size)
        ]
        query_embeddings = encoder.encode_batch([text.split() for text in query_texts])
        
        # Batch search
        start = time.perf_counter()
        batch_results = store.search_batch(query_embeddings, k=k)
        batch_time = time.perf_counter() - start
        
        # Sequential search
        start = time.perf_counter()
        sequential_results = [
            store.search(query_embeddings[i:i+1], k=k)
            for i in range(len(query_embeddings))
        ]
        sequential_time = time.perf_counter() - start
        
        # Calculate metrics
        speedup = sequential_time / batch_time
        batch_qps = batch_size / batch_time
        sequential_qps = batch_size / sequential_time
        
        print(f"  Batch search:      {batch_time*1000:.2f} ms ({batch_qps:.1f} queries/sec)")
        print(f"  Sequential search: {sequential_time*1000:.2f} ms ({sequential_qps:.1f} queries/sec)")
        print(f"  Speedup:           {speedup:.2f}x")
        
        # Verify consistency (spot check first query)
        batch_ids, batch_dists, _ = batch_results[0]
        seq_ids, seq_dists, _ = sequential_results[0]
        
        consistency = np.array_equal(batch_ids, seq_ids)
        print(f"  Result consistency: {'✓ PASS' if consistency else '✗ FAIL'}")
        
        results.append({
            "batch_size": batch_size,
            "batch_time": batch_time,
            "sequential_time": sequential_time,
            "speedup": speedup,
            "batch_qps": batch_qps,
            "sequential_qps": sequential_qps,
            "consistent": consistency,
        })
    
    return results


def benchmark_gpu_acceleration(n_index_vectors: int, batch_size: int, k: int = 10):
    """Compare CPU vs GPU performance for batch search."""
    print(f"\n{'='*80}")
    print(f"GPU Acceleration Benchmark: {n_index_vectors} vectors, batch={batch_size}, k={k}")
    print(f"{'='*80}\n")
    
    results = {}
    
    for device in ["cpu", "gpu"]:
        try:
            use_gpu = device == "gpu"
            print(f"\n--- {device.upper()} Benchmark ---")
            
            encoder = EmbeddingEncoder(
                model_name="all-MiniLM-L6-v2",
                device="cuda" if use_gpu else "cpu",
                batch_size=32,
            )
            
            tmp_path = Path("/tmp/batch_search_benchmark")
            tmp_path.mkdir(exist_ok=True)
            
            store = VectorStore(
                dimension=384,
                storage_path=str(tmp_path / f"vectors_{device}"),
                use_gpu=use_gpu,
            )
            
            # Generate corpus
            corpus = generate_sample_corpus(n_index_vectors)
            embeddings = encoder.encode_batch([text.split() for text in corpus])
            event_ids = [f"event_{i}" for i in range(n_index_vectors)]
            metadata = [{"text": text, "index": i} for i, text in enumerate(corpus)]
            
            store.add_vectors(embeddings, event_ids, metadata)
            
            # Generate queries
            query_texts = [f"test query {i}" for i in range(batch_size)]
            query_embeddings = encoder.encode_batch([text.split() for text in query_texts])
            
            # Warmup
            _ = store.search_batch(query_embeddings, k=k)
            
            # Benchmark
            num_runs = 5
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = store.search_batch(query_embeddings, k=k)
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            qps = batch_size / avg_time
            
            print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            print(f"  Throughput:   {qps:.1f} queries/sec")
            
            results[device] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "qps": qps,
            }
            
        except Exception as e:
            print(f"  {device.upper()} benchmark failed: {e}")
            results[device] = None
    
    # Compare
    if results.get("cpu") and results.get("gpu"):
        speedup = results["cpu"]["avg_time"] / results["gpu"]["avg_time"]
        print(f"\n{'='*80}")
        print(f"GPU Speedup: {speedup:.2f}x")
        print(f"{'='*80}")
    
    return results


def main():
    """Run comprehensive batch search benchmarks."""
    print("\n" + "="*80)
    print("BATCH SEARCH PERFORMANCE BENCHMARK")
    print("="*80)
    
    # Test 1: Small index, varying batch sizes
    print("\n\n## Test 1: Small Index (5K vectors)")
    benchmark_batch_vs_sequential(
        n_index_vectors=5000,
        batch_sizes=[10, 25, 50, 100],
        k=10,
        use_gpu=False,
    )
    
    # Test 2: Medium index, varying batch sizes
    print("\n\n## Test 2: Medium Index (50K vectors)")
    benchmark_batch_vs_sequential(
        n_index_vectors=50000,
        batch_sizes=[10, 50, 100, 200],
        k=10,
        use_gpu=False,
    )
    
    # Test 3: GPU acceleration (if available)
    try:
        import torch
        if torch.cuda.is_available():
            print("\n\n## Test 3: GPU Acceleration")
            benchmark_gpu_acceleration(
                n_index_vectors=50000,
                batch_size=100,
                k=10,
            )
        else:
            print("\n\n## Test 3: GPU Acceleration - SKIPPED (no CUDA)")
    except ImportError:
        print("\n\n## Test 3: GPU Acceleration - SKIPPED (torch not available)")
    
    print("\n\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
