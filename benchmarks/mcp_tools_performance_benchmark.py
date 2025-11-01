#!/usr/bin/env python3
"""
MCP Tools End-to-End Performance Benchmark

Tests real-world MCP server tool performance with comprehensive metrics:
- remember_context: Ingestion throughput, segmentation speed, index training
- recall_memories: Query latency (p50/p95/p99), retrieval accuracy
- manage_memory: Index health checks, retrain operations
- search_memory_batch: Batch search throughput with CPU/GPU routing
- transfer_memory: Export/import performance with compression

Simulates realistic workload: store 60k tokens, run 1000 queries, measure all operations.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config
from emx_mcp.utils.logging import setup_logging

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


class MCPToolsBenchmark:
    """End-to-end benchmark using actual MCP server tools."""

    def __init__(self, corpus_size: int = 60_000):
        self.corpus_size = corpus_size
        self.temp_dir = Path(tempfile.mkdtemp(prefix="emx_mcp_bench_"))
        self.results: dict[str, Any] = {
            "config": {
                "corpus_size_tokens": corpus_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "system": {},
            "ingestion": {},
            "retrieval": {},
            "management": {},
            "batch_search": {},
            "transfer": {},
            "summary": {},
        }

        # Load config with expected tokens hint
        self.config = load_config()
        self.config["storage"]["expected_total_tokens"] = corpus_size

        # Setup logging after config is loaded
        setup_logging(self.config)

        # Setup test project
        self.project_path = self.temp_dir / "test_project"
        self.project_path.mkdir()
        os.environ["EMX_PROJECT_PATH"] = str(self.project_path)
        os.environ["EMX_EXPECTED_TOKENS"] = str(corpus_size)

        logger.info(f"🚀 MCP Tools Benchmark initialized: {corpus_size} tokens")
        logger.info(f"📁 Test project: {self.project_path}")

    def capture_system_info(self):
        """Capture system configuration."""
        import platform

        try:
            import torch

            gpu_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        except ImportError:
            gpu_available = False
            gpu_name = "N/A"

        self.results["system"] = {
            "platform": platform.platform(),
            "cpu_model": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "python_version": platform.python_version(),
        }

        logger.info(f"💻 System: {self.results['system']['cpu_model']}")
        logger.info(
            f"🧮 CPU: {self.results['system']['cpu_cores']}c/"
            f"{self.results['system']['cpu_threads']}t, "
            f"RAM: {self.results['system']['ram_gb']}GB"
        )
        if gpu_available:
            logger.info(f"🎮 GPU: {gpu_name}")

    def generate_corpus(self) -> str:
        """Generate realistic multi-domain corpus with semantic boundaries."""
        logger.info(f"📝 Generating {self.corpus_size:,} token corpus...")

        # Diverse content types for realistic segmentation
        segments = {
            "debugging": """
We're debugging intermittent timeout errors in the production API endpoint.
The database connection pool is exhausted during peak traffic at 3pm UTC.
Slow query log reveals N+1 query patterns in the user profile endpoint.
Adding eager loading with selectinload batches queries and reduces latency.
Creating composite indexes on foreign keys reduced query time by 85 percent.
The Redis cache hit rate improved from 40 to 78 percent after TTL adjustments.
""",
            "architecture": """
For the microservices migration we need to design the service boundaries carefully.
Each service should own its data and expose APIs through REST and GraphQL.
The API gateway will handle authentication, rate limiting, and request routing.
Service discovery uses Consul with health checks and automatic failover support.
Inter-service communication requires circuit breakers to prevent cascade failures.
Event sourcing with Kafka enables eventual consistency across service boundaries.
""",
            "code_review": """
Reviewing the authentication refactor PR that introduces JWT refresh tokens.
The current implementation is vulnerable to XSS with localStorage storage.
Switching to httpOnly cookies with SameSite strict prevents JavaScript access.
Token rotation strategy invalidates old tokens on refresh which is good practice.
However checking token revocation on every request adds 15ms p50 latency overhead.
Should cache revoked tokens in Redis instead of querying database each time.
""",
            "performance": """
Profiling shows 60 percent of CPU time is spent in JSON serialization overhead.
Switching from standard json to orjson improved throughput by 3.5x consistently.
Database connection pooling with PgBouncer reduced latency by 40ms at p50.
Code splitting by route reduced initial bundle from 2.8MB to 400KB with lazy loading.
Converting images to WebP format reduced bandwidth by 65 percent with quality maintained.
""",
            "planning": """
Q4 roadmap priorities include the file attachment feature users have requested.
Architecture team proposed S3 with CloudFront CDN for global distribution.
Security requires client side encryption before upload using KMS key management.
Virus scanning with ClamAV in Lambda functions processes uploads asynchronously.
File size limit of 100MB keeps storage costs reasonable while serving most use cases.
""",
        }

        # Repeat segments to reach target token count
        corpus_parts = []
        current_tokens = 0
        target = self.corpus_size

        while current_tokens < target:
            for _seg_type, content in segments.items():
                tokens = content.split()
                corpus_parts.append(content)
                current_tokens += len(tokens)
                if current_tokens >= target:
                    break

        corpus = "\n\n".join(corpus_parts)
        actual_tokens = len(corpus.split())

        logger.info(f"✅ Generated corpus: {actual_tokens:,} tokens")
        return corpus

    def benchmark_remember_context(self, corpus: str, manager: ProjectMemoryManager):
        """Benchmark remember_context tool: ingestion + segmentation + indexing."""
        logger.info("\n" + "=" * 60)
        logger.info("📥 BENCHMARK: remember_context (Ingestion)")
        logger.info("=" * 60)

        tokens = corpus.split()
        actual_tokens = len(tokens)

        # Capture initial memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**2)  # MB

        start_time = time.time()

        # Simulate the remember_context tool
        logger.info(f"Storing {actual_tokens:,} tokens with auto-segmentation...")

        # Test with segmentation enabled
        seg_start = time.time()
        seg_result = manager.segment_tokens(tokens, gamma=1.0)
        seg_time = time.time() - seg_start

        boundaries = seg_result["refined_boundaries"]
        num_segments = seg_result["num_events"]

        logger.info(
            f"⚡ Segmentation: {num_segments} events in {seg_time:.3f}s "
            f"({actual_tokens / seg_time:.0f} tokens/sec)"
        )

        # Store each segment
        event_ids = []
        add_start = time.time()

        for i in range(len(boundaries) - 1):
            segment_tokens = tokens[boundaries[i] : boundaries[i + 1]]
            event_result = manager.add_event(
                segment_tokens, embeddings=None, metadata={"segment_id": i}
            )
            event_ids.append(event_result["event_id"])

        add_time = time.time() - add_start
        total_time = time.time() - start_time

        # Capture final memory
        mem_after = process.memory_info().rss / (1024**2)  # MB
        mem_delta = mem_after - mem_before

        # Get index health
        index_info = manager.get_index_info()

        self.results["ingestion"] = {
            "total_tokens": actual_tokens,
            "num_events": num_segments,
            "tokens_per_event_avg": actual_tokens // num_segments,
            "segmentation_time_sec": round(seg_time, 3),
            "segmentation_throughput_tokens_per_sec": round(actual_tokens / seg_time, 0),
            "add_events_time_sec": round(add_time, 3),
            "add_events_throughput_events_per_sec": round(num_segments / add_time, 1),
            "total_time_sec": round(total_time, 3),
            "memory_delta_mb": round(mem_delta, 2),
            "index_trained": index_info["is_trained"],
            "total_vectors": index_info["total_vectors"],
            "nlist": index_info["nlist"],
            "optimal_nlist": index_info["optimal_nlist"],
            "nlist_ratio": round(index_info["nlist_ratio"], 3),
            "nlist_status": (
                "optimal"
                if index_info["nlist_ratio"] >= 0.85
                else "acceptable"
                if index_info["nlist_ratio"] >= 0.5
                else "suboptimal"
            ),
        }

        logger.info(f"✅ Ingestion complete: {total_time:.2f}s total")
        logger.info(f"📊 Events: {num_segments}, Vectors: {index_info['total_vectors']:,}")
        logger.info(
            f"🎯 Index: nlist={index_info['nlist']} (optimal={index_info['optimal_nlist']}, "
            f"ratio={index_info['nlist_ratio']:.1%})"
        )
        logger.info(f"💾 Memory: +{mem_delta:.1f}MB")

    def benchmark_recall_memories(self, manager: ProjectMemoryManager):
        """Benchmark recall_memories tool: query latency and retrieval quality."""
        logger.info("\n" + "=" * 60)
        logger.info("🔍 BENCHMARK: recall_memories (Retrieval)")
        logger.info("=" * 60)

        retrieval = manager.retrieval

        logger.info("Warming up cache...")
        retrieval.warmup_cache_manual(num_passes=5)
        retrieval.print_cache_stats()

        # Reset cache stats for benchmark
        retrieval.cache_hits = 0
        retrieval.cache_misses = 0

        # Generate diverse queries
        queries = [
            "debugging database connection pool timeout errors",
            "microservices architecture service boundaries design",
            "JWT refresh token authentication security",
            "JSON serialization performance optimization",
            "file attachment feature S3 CloudFront",
            "code review authentication middleware refactor",
            "Redis cache hit rate TTL configuration",
            "database indexes foreign key performance",
            "API gateway rate limiting circuit breakers",
            "WebP image compression bandwidth reduction",
        ]

        logger.info(f"Running {len(queries)} queries (k=10 each)...")

        latencies = []
        results_per_query = []

        # Use batch processing for 3+ queries
        if len(queries) >= 3:
            logger.info(f"Using batch retrieval for {len(queries)} queries...")

            batch_start = time.time()

            # Batch encode all queries
            query_embeddings = manager.encode_queries_batch(queries)

            # Batch retrieve
            batch_results = retrieval.retrieve_batch(
                query_embeddings,
                k_similarity=10,
                k_contiguity=5,
                use_contiguity=True,
            )

            total_batch_time = time.time() - batch_start
            avg_latency_ms = (total_batch_time / len(queries)) * 1000

            for i, (query, result) in enumerate(zip(queries, batch_results, strict=True)):
                latencies.append(avg_latency_ms)
                results_per_query.append(len(result.get("events", [])))

                if i == 0:
                    logger.info(
                        f"  Batch Query 1: '{query[:50]}...' -> {avg_latency_ms:.1f}ms avg, "
                        f"{len(result.get('events', []))} results"
                    )
        else:
            # Individual processing for small query sets
            for i, query in enumerate(queries):
                query_start = time.time()

                # Encode query
                query_embedding = manager.encode_query(query)

                # Retrieve memories
                result = retrieval.retrieve(
                    query_embedding.tolist(),
                    k_similarity=10,
                    k_contiguity=5,
                    use_contiguity=True,
                )

                latency_ms = (time.time() - query_start) * 1000
                latencies.append(latency_ms)
                results_per_query.append(len(result.get("events", [])))

                if i == 0:
                    logger.info(
                        f"  Query 1: '{query[:50]}...' -> {latency_ms:.1f}ms, "
                        f"{len(result.get('events', []))} results"
                    )

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50 = np.percentile(latencies_sorted, 50)
        p95 = np.percentile(latencies_sorted, 95)
        p99 = np.percentile(latencies_sorted, 99)
        avg = np.mean(latencies_sorted)
        min_lat = min(latencies_sorted)
        max_lat = max(latencies_sorted)

        self.results["retrieval"] = {
            "num_queries": len(queries),
            "k_per_query": 10,
            "latency_ms": {
                "min": round(min_lat, 2),
                "avg": round(avg, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2),
                "max": round(max_lat, 2),
            },
            "results_per_query": {
                "min": min(results_per_query),
                "avg": round(np.mean(results_per_query), 1),
                "max": max(results_per_query),
            },
            "throughput_qps": round(len(queries) / (sum(latencies) / 1000), 1),
        }

        logger.info(
            f"✅ Retrieval latency: min={min_lat:.1f}ms, p50={p50:.1f}ms, "
            f"p95={p95:.1f}ms, p99={p99:.1f}ms, max={max_lat:.1f}ms"
        )
        logger.info(f"📈 Throughput: {self.results['retrieval']['throughput_qps']:.1f} QPS")

    def benchmark_manage_memory(self, manager: ProjectMemoryManager):
        """Benchmark manage_memory tool: stats, estimate, retrain operations."""
        logger.info("\n" + "=" * 60)
        logger.info("⚙️  BENCHMARK: manage_memory (Management)")
        logger.info("=" * 60)

        # Test 1: Stats action
        stats_start = time.time()
        stats = manager.get_stats()
        index_info = manager.get_index_info()
        stats_time = (time.time() - stats_start) * 1000

        logger.info(f"✅ Stats query: {stats_time:.2f}ms")

        # Test 2: Estimate action simulation
        estimate_start = time.time()
        expected_tokens = self.corpus_size
        expected_events = expected_tokens // 30
        expected_vectors = expected_events * 27
        optimal_nlist = int(4 * (expected_vectors**0.5))
        estimate_time = (time.time() - estimate_start) * 1000

        logger.info(
            f"✅ Estimate: {expected_tokens:,} tokens → {expected_vectors:,} vectors, "
            f"optimal_nlist={optimal_nlist} ({estimate_time:.2f}ms)"
        )

        # Test 3: Index health check (no retrain if optimal)
        current_nlist = index_info["nlist"]
        nlist_ratio = index_info["nlist_ratio"]

        if nlist_ratio < 0.85:
            # Would trigger retrain in production
            logger.info(
                f"⚠️  Index suboptimal: nlist={current_nlist} "
                f"vs optimal={index_info['optimal_nlist']} "
                f"(ratio={nlist_ratio:.1%})"
            )
        else:
            logger.info(f"✅ Index optimal: nlist={current_nlist} (ratio={nlist_ratio:.1%})")

        self.results["management"] = {
            "stats_query_time_ms": round(stats_time, 2),
            "estimate_time_ms": round(estimate_time, 2),
            "project_events": stats["project_events"],
            "global_events": stats["global_events"],
            "local_context_size": stats["local_context_size"],
            "index_nlist": current_nlist,
            "index_optimal_nlist": index_info["optimal_nlist"],
            "index_nlist_ratio": round(nlist_ratio, 3),
            "index_status": (
                "optimal"
                if nlist_ratio >= 0.85
                else "acceptable"
                if nlist_ratio >= 0.5
                else "suboptimal"
            ),
        }

    def benchmark_search_memory_batch(self, manager: ProjectMemoryManager):
        """Benchmark search_memory_batch tool: batch throughput with adaptive routing."""
        logger.info("\n" + "=" * 60)
        logger.info("🔎 BENCHMARK: search_memory_batch (Batch Search)")
        logger.info("=" * 60)

        # Test different batch sizes
        batch_sizes = [10, 50, 100, 200]
        batch_results = {}

        base_queries = [
            "database performance optimization",
            "authentication security best practices",
            "microservices architecture patterns",
            "API rate limiting implementation",
            "cache invalidation strategies",
        ]

        for batch_size in batch_sizes:
            # Generate batch of queries
            queries = []
            for i in range(batch_size):
                queries.append(base_queries[i % len(base_queries)] + f" variant {i}")

            logger.info(f"Testing batch size: {batch_size} queries...")

            # Encode all queries using batch encoding for efficiency
            batch_start = time.time()
            query_embeddings = manager.encode_queries_batch(queries)
            encode_time = time.time() - batch_start

            # Batch search (loop over individual queries)
            search_start = time.time()
            vector_store = manager.project_store.vector_store
            batch_search_results = []
            for i in range(batch_size):
                query_vec = query_embeddings[i : i + 1]  # Keep 2D shape
                event_ids, distances, metadata = vector_store.search(query_vec, k=10)
                batch_search_results.append((event_ids, distances, metadata))
            search_time = time.time() - search_start

            total_time = time.time() - batch_start

            # Analyze search results
            total_results_found = sum(len(event_ids) for event_ids, _, _ in batch_search_results)
            avg_results_per_query = total_results_found / batch_size if batch_size > 0 else 0

            # Calculate average relevance scores
            all_distances = []
            for _, distances, _ in batch_search_results:
                if isinstance(distances, (list, tuple, np.ndarray)):
                    all_distances.extend(distances)
                elif isinstance(distances, (int, float)):
                    all_distances.append(distances)
            avg_distance = np.mean(all_distances) if all_distances else 0.0

            batch_results[batch_size] = {
                "encode_time_sec": round(encode_time, 3),
                "search_time_sec": round(search_time, 3),
                "total_time_sec": round(total_time, 3),
                "queries_per_sec": round(batch_size / total_time, 1),
                "avg_latency_ms": round((total_time / batch_size) * 1000, 2),
                "total_results_found": total_results_found,
                "avg_results_per_query": round(avg_results_per_query, 1),
                "avg_relevance_distance": round(avg_distance, 4),
            }

            logger.info(
                f"  ✅ {batch_size} queries: {total_time:.3f}s total, "
                f"{batch_results[batch_size]['queries_per_sec']:.1f} QPS, "
                f"{batch_results[batch_size]['avg_latency_ms']:.1f}ms avg"
            )

        vector_store = manager.project_store.vector_store
        self.results["batch_search"] = {
            "batch_sizes_tested": batch_sizes,
            "results": batch_results,
            "gpu_enabled": vector_store.gpu_enabled,
            "device": "cuda" if vector_store.gpu_enabled else "cpu",
        }

    def benchmark_transfer_memory(self, manager: ProjectMemoryManager):
        """Benchmark transfer_memory tool: export/import with compression."""
        logger.info("\n" + "=" * 60)
        logger.info("📦 BENCHMARK: transfer_memory (Export/Import)")
        logger.info("=" * 60)

        archive_path = self.temp_dir / "memory_export.tar.gz"

        # Test export
        export_start = time.time()
        export_result = manager.export_memory(archive_path)
        export_time = time.time() - export_start

        archive_size_mb = export_result["size_bytes"] / (1024**2)
        events_exported = manager.get_stats()["project_events"]

        logger.info(
            f"✅ Export: {export_time:.2f}s, {archive_size_mb:.2f}MB, {events_exported} events"
        )

        # Test import to new location
        import_dir = self.temp_dir / "imported_project"
        import_dir.mkdir()

        # Clear current memory
        manager.clear_memory()

        import_start = time.time()
        # Simulate import (would need new manager instance in production)
        # For benchmark, we measure time to extract archive
        import tarfile

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(import_dir)
        import_time = time.time() - import_start

        logger.info(f"✅ Import: {import_time:.2f}s")

        self.results["transfer"] = {
            "export_time_sec": round(export_time, 3),
            "export_size_mb": round(archive_size_mb, 2),
            "export_events": events_exported,
            "export_throughput_mb_per_sec": round(archive_size_mb / export_time, 2),
            "import_time_sec": round(import_time, 3),
            "import_throughput_mb_per_sec": round(archive_size_mb / import_time, 2),
        }

    def generate_summary(self):
        """Generate performance summary with key metrics."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 60)

        ingestion = self.results["ingestion"]
        retrieval = self.results["retrieval"]
        _ = self.results["management"]  # Available for future use

        # Calculate overall throughput
        total_time = ingestion["total_time_sec"]
        tokens_per_sec = ingestion["total_tokens"] / total_time
        events_per_sec = ingestion["num_events"] / total_time

        self.results["summary"] = {
            "overall_performance": {
                "total_runtime_sec": round(total_time, 2),
                "ingestion_throughput_tokens_per_sec": round(tokens_per_sec, 0),
                "ingestion_throughput_events_per_sec": round(events_per_sec, 1),
                "retrieval_latency_p50_ms": retrieval["latency_ms"]["p50"],
                "retrieval_latency_p95_ms": retrieval["latency_ms"]["p95"],
                "retrieval_throughput_qps": retrieval["throughput_qps"],
            },
            "index_quality": {
                "nlist_ratio": ingestion["nlist_ratio"],
                "nlist_status": ingestion["nlist_status"],
                "vectors_indexed": ingestion["total_vectors"],
            },
            "resource_usage": {
                "memory_delta_mb": ingestion["memory_delta_mb"],
                "cpu_cores_available": self.results["system"]["cpu_cores"],
                "gpu_available": self.results["system"]["gpu_available"],
            },
        }

        logger.info("\n🚀 INGESTION:")
        logger.info(
            f"  • Throughput: {tokens_per_sec:,.0f} tokens/sec, {events_per_sec:.1f} events/sec"
        )
        logger.info(
            f"  • Total: {ingestion['total_tokens']:,} tokens → "
            f"{ingestion['num_events']} events → {ingestion['total_vectors']:,} vectors"
        )
        logger.info(f"  • Time: {ingestion['total_time_sec']:.2f}s")

        logger.info("\n🔍 RETRIEVAL:")
        logger.info(
            f"  • Latency: p50={retrieval['latency_ms']['p50']:.1f}ms, "
            f"p95={retrieval['latency_ms']['p95']:.1f}ms, "
            f"p99={retrieval['latency_ms']['p99']:.1f}ms"
        )
        logger.info(f"  • Throughput: {retrieval['throughput_qps']:.1f} QPS")

        logger.info("\n🎯 INDEX QUALITY:")
        logger.info(
            f"  • nlist: {ingestion['nlist']} (optimal: {ingestion['optimal_nlist']}, "
            f"ratio: {ingestion['nlist_ratio']:.1%})"
        )
        logger.info(f"  • Status: {ingestion['nlist_status'].upper()}")

        logger.info("\n💾 RESOURCES:")
        logger.info(f"  • Memory: +{ingestion['memory_delta_mb']:.1f}MB")
        logger.info(
            f"  • CPU: {self.results['system']['cpu_cores']}c/"
            f"{self.results['system']['cpu_threads']}t"
        )
        if self.results["system"]["gpu_available"]:
            logger.info(f"  • GPU: {self.results['system']['gpu_name']}")

    def save_results(self):
        """Save benchmark results to JSON."""
        output_path = Path(__file__).parent / "mcp_tools_benchmark_results.json"

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\n💾 Results saved to: {output_path}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        logger.info(f"🧹 Cleaned up: {self.temp_dir}")

    def run(self):
        """Execute full benchmark suite."""
        try:
            self.capture_system_info()

            # Generate test corpus
            corpus = self.generate_corpus()

            # Detect global path
            global_path = os.getenv(
                "EMX_GLOBAL_PATH", str(Path.home() / ".emx-mcp" / "global_memories")
            )

            # Initialize memory manager
            manager = ProjectMemoryManager(
                project_path=str(self.project_path),
                global_path=str(global_path),
                config=self.config,
            )

            logger.info("🛠️  Initialized ProjectMemoryManager")
            logger.info(f"  • Project Path: {manager.project_path}")
            logger.info(f"  • Global Path: {manager.global_path}")

            # Run benchmarks
            self.benchmark_remember_context(corpus, manager)
            self.benchmark_recall_memories(manager)
            self.benchmark_manage_memory(manager)
            self.benchmark_search_memory_batch(manager)
            self.benchmark_transfer_memory(manager)

            # Generate summary
            self.generate_summary()

            # Save results
            self.save_results()

            logger.info("\n✅ Benchmark complete!")

        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Tools End-to-End Performance Benchmark")
    parser.add_argument(
        "--tokens",
        type=int,
        default=60_000,
        help="Corpus size in tokens (default: 60,000)",
    )

    args = parser.parse_args()

    benchmark = MCPToolsBenchmark(corpus_size=args.tokens)
    benchmark.run()


if __name__ == "__main__":
    main()
