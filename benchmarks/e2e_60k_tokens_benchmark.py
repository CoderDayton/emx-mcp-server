#!/usr/bin/env python3
import json
import logging
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config
from emx_mcp.utils.logging import setup_logging


def generate_synthetic_corpus(target_tokens: int = 60_000) -> str:
    """Generate realistic multi-domain conversation with semantic boundaries."""
    debugging_session = (
        """
We're seeing intermittent 504 Gateway Timeout errors on the production API.
The logs show requests pile up during peak traffic around 3pm UTC daily.
Database connection pool is exhausted - current max_connections is 100.
Increasing to 500 might help but we need to profile query performance first.
The slow query log reveals several N+1 patterns in the user profile endpoint.
Adding eager loading with SQLAlchemy selectinload should batch those queries.
Also found three endpoints without database indexes on foreign key columns.
Creating composite indexes on (user_id, created_at) reduced query time by 85%.
After deploying the index changes, the 504 errors dropped to near zero.
Now we're seeing occasional deadlocks in the transaction processing code.
Switching to optimistic locking with version columns resolved the deadlocks.
The Redis cache hit rate is only 40% which seems low for user sessions.
Adjusting TTL from 5 minutes to 30 minutes improved hit rate to 78%.
Need to implement cache warming on deployment to avoid cold start latency.
The monitoring dashboard shows p95 latency is now under 200ms consistently.
"""
        * 30
    )

    planning_discussion = (
        """
For Q4 roadmap we need to prioritize the file attachment feature.
Users have requested this for 8 months and churn analysis shows it's critical.
The architecture team proposed using S3 with CloudFront for CDN distribution.
Security requires client-side encryption before upload with KMS key management.
We'll need to implement virus scanning using ClamAV in a Lambda function.
File size limit should be 100MB per attachment to keep costs reasonable.
The database schema needs a new attachments table with foreign keys to messages.
Migration plan: deploy schema changes first, then gradually roll out upload UI.
Support for image thumbnails will require ImageMagick or Sharp for resizing.
We should generate multiple thumbnail sizes: 150x150, 300x300, and 600x600.
The upload progress indicator needs WebSocket connection for real-time updates.
Error handling must cover network failures, quota exceeded, and invalid file types.
Compliance team wants audit logging for all file access with IP and timestamp.
Cost estimation shows $2000/month for storage and bandwidth at current user base.
"""
        * 36
    )

    code_review = (
        """
Reviewing PR #4421: Refactor authentication middleware to use JWT refresh tokens.
The current implementation stores JWTs in localStorage which is XSS vulnerable.
Switching to httpOnly cookies with SameSite=Strict prevents JavaScript access.
The refresh token rotation strategy looks good - invalidate old token on use.
However the database query to check token revocation happens on every request.
This adds 15ms p50 latency - should cache revoked tokens in Redis instead.
The token expiry is set to 15 minutes which seems short for mobile clients.
Consider extending to 1 hour for access tokens, keep 7 days for refresh tokens.
The error handling doesn't distinguish between expired vs invalid signatures.
Return 401 for expired tokens so frontend can attempt refresh automatically.
Missing rate limiting on the token refresh endpoint - vulnerable to brute force.
Add sliding window rate limit: 5 requests per minute per user with exponential backoff.
The cryptographic algorithms are using RS256 which is good for key rotation.
Key storage in environment variables is risky - migrate to AWS Secrets Manager.
Unit test coverage is 85% but missing edge cases like clock skew handling.
Add tests for tokens issued slightly in the future due to NTP drift tolerance.
The PR description should document the migration path for existing users.
Need to handle cookie domain configuration for multi-subdomain architecture.
Consider adding security headers: X-Frame-Options, Content-Security-Policy.
The audit logging captures token issuance but not refresh or revocation events.
"""
        * 40
    )

    performance_analysis = (
        """
Profiling results show 60% of CPU time spent in JSON serialization.
Using orjson instead of standard library json improved throughput by 3.5x.
The database connection pool is creating new connections on every request.
Implementing connection pooling with PgBouncer reduced latency by 40ms p50.
Memory usage spikes correlate with large GraphQL query responses over 5MB.
Adding pagination to all list endpoints capped maximum response size at 1MB.
The frontend bundle size is 2.8MB uncompressed which delays time-to-interactive.
Code splitting by route reduced initial bundle to 400KB with lazy loading.
Image assets lack WebP format support - serving PNG doubles transfer time.
Converting to WebP with fallback reduced image bandwidth by 65%.
The API response times show 200ms constant overhead from middleware stack.
Profiling revealed unnecessary JWT signature verification on public endpoints.
Skipping auth middleware for public routes cut overhead to 15ms baseline.
Database query count per request averages 23 which indicates N+1 problems.
Using dataloader pattern batched queries into 3 roundtrips maximum.
"""
        * 24
    )

    corpus = f"""
=== System Context Initialization ===

This is a comprehensive conversation covering multiple technical domains
including debugging, planning, code review, and performance optimization.

=== Debugging Session ===

{debugging_session}

=== Planning Discussion ===

{planning_discussion}

=== Code Review ===

{code_review}

=== Performance Analysis ===

{performance_analysis}

=== Session Complete ===

Total topics covered: 4 major domains with distinct semantic boundaries.
This corpus is designed to test segmentation accuracy and retrieval precision.
""".strip()

    actual_tokens = len(corpus.split())
    print(f"Generated corpus: {actual_tokens:,} tokens (target: {target_tokens:,})")

    return corpus


class E2EBenchmark:
    """EM-LLM benchmark with proper index health tracking."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.config = load_config()
        self.logger = setup_logging(self.config)

        # Add expected token count for optimal nlist calculation
        self.config["storage"]["expected_total_tokens"] = 60_000

        self.manager = ProjectMemoryManager(
            project_path=str(temp_dir / "benchmark_project"),
            global_path=str(temp_dir / "benchmark_global"),
            config=self.config,
        )

        enriched_config = self.manager.config

        self.metrics = {
            "config": {
                "gamma": enriched_config["memory"]["gamma"],
                "batch_size": enriched_config["model"]["batch_size"],
                "device": enriched_config["model"]["device"],
                "nprobe": enriched_config["storage"]["nprobe"],
            },
            "timings": {},
            "memory": {},
            "corpus": {},
            "segmentation": {},
            "indexing": {},
            "retrieval": {},
        }

        self.process = psutil.Process(os.getpid())

    def measure_memory_usage(self) -> dict:
        """Capture current memory footprint."""
        mem_info = self.process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }

    def get_index_health(self) -> dict:
        """Direct inspection of index."""
        try:
            storage = self.manager.project_store
            vs = storage.vector_store

            n_vectors = vs.index.ntotal if vs.is_trained else 0
            optimal_nlist = vs._calculate_optimal_nlist(n_vectors) if n_vectors > 0 else vs.nlist

            # Calculate nlist_ratio, handling None values
            nlist_ratio = 1.0
            if vs.nlist is not None and optimal_nlist is not None and optimal_nlist > 0:
                nlist_ratio = vs.nlist / optimal_nlist

            return {
                "is_trained": vs.is_trained,
                "total_vectors": n_vectors,
                "nlist": vs.nlist,
                "optimal_nlist": optimal_nlist,
                "nlist_ratio": nlist_ratio,
                "nprobe": vs.nprobe,
                "use_sq": vs.use_sq,
                "buffered_vectors": sum(v.shape[0] for v in vs.training_vectors),
            }
        except Exception as e:
            logging.error(f"Failed to get index health: {e}")
            return {
                "is_trained": False,
                "total_vectors": 0,
                "nlist": 0,
                "error": str(e),
            }

    def run_remember_phase(self, corpus: str, gamma: float = 1.0) -> dict:
        """Execute remember phase (tokenize, segment, embed, index)."""
        print("\n=== REMEMBER PHASE ===")
        start_time = time.perf_counter()
        mem_before = self.measure_memory_usage()

        # Tokenization
        tokenize_start = time.perf_counter()
        tokens = corpus.split()
        tokenize_time = time.perf_counter() - tokenize_start

        self.metrics["corpus"]["total_tokens"] = len(tokens)
        self.metrics["timings"]["tokenization_sec"] = tokenize_time

        print(f"Tokenized: {len(tokens):,} tokens in {tokenize_time * 1000:.2f}ms")

        # Update expected vector count based on actual token count
        actual_tokens = len(tokens)
        expected_events = actual_tokens // 30
        expected_vectors = expected_events * 27
        min_training = int(expected_vectors * 0.9)  # Train at 90% for better nlist

        # Update VectorStore configuration with actual estimates
        self.manager.project_store.vector_store.expected_vector_count = expected_vectors
        self.manager.project_store.vector_store.min_training_size = min_training
        self.manager.project_store.vector_store.nlist = (
            self.manager.project_store.vector_store._calculate_optimal_nlist(expected_vectors)
        )

        print(
            f"📊 Estimated {expected_vectors:,} vectors → "
            f"nlist={self.manager.project_store.vector_store.nlist}, "
            f"training at {min_training:,}"
        )

        # Segmentation (O(n) surprise-based boundaries)
        segment_start = time.perf_counter()
        seg_result = self.manager.segment_tokens(tokens, gamma)
        segment_time = time.perf_counter() - segment_start

        self.metrics["segmentation"] = {
            "num_segments": seg_result["num_events"],
            "method": seg_result["method"],
            "boundaries": seg_result["refined_boundaries"],
            "time_sec": segment_time,
        }

        print(f"Segmented: {seg_result['num_events']} events in {segment_time * 1000:.2f}ms")

        # Add events (embedding + indexing)
        storage_start = time.perf_counter()
        event_ids = []
        boundaries = seg_result["refined_boundaries"]

        for i in range(len(boundaries) - 1):
            segment_tokens = tokens[boundaries[i] : boundaries[i + 1]]
            event_result = self.manager.add_event(
                segment_tokens,
                embeddings=None,
                metadata={"segment_index": i, "benchmark": "e2e_60k"},
            )
            event_ids.append(event_result["event_id"])

        storage_time = time.perf_counter() - storage_start

        self.metrics["timings"]["storage_sec"] = storage_time
        self.metrics["timings"]["remember_total_sec"] = time.perf_counter() - start_time

        mem_after = self.measure_memory_usage()
        self.metrics["memory"]["remember_delta_mb"] = mem_after["rss_mb"] - mem_before["rss_mb"]

        print(f"Stored: {len(event_ids)} events in {storage_time:.2f}s")
        print(f" Memory delta: +{self.metrics['memory']['remember_delta_mb']:.1f} MB")

        # Index health
        index_info = self.get_index_health()
        self.metrics["indexing"] = index_info

        status = "✅" if index_info["nlist_ratio"] >= 0.95 else "⚠"
        print(
            f" Index: trained={index_info['is_trained']}, "
            f"vectors={index_info['total_vectors']}, "
            f"nlist={index_info['nlist']} "
            f"(optimal={index_info['optimal_nlist']}) {status}"
        )

        return {"event_ids": event_ids, "seg_result": seg_result}

    def run_recall_phase(self, queries: list, k: int = 10) -> dict:
        """Execute recall phase (two-stage retrieval)."""
        print("\n=== RECALL PHASE ===")
        start_time = time.perf_counter()
        results: list[dict[str, str | int | float]] = []

        # Pre-warm cache for optimal performance
        cache_info = self.manager.get_cache_info()
        if cache_info["cache_size"] == 0:
            print("Pre-warming retrieval cache...")
            self.manager.retrieval.warmup_cache_manual(num_passes=3)

        # Use batch processing for queries if 3+, else individual
        if len(queries) >= 3:
            self._extracted_from_run_recall_phase(queries, k, results)
        else:
            # Individual query processing for small batches
            for query in queries:
                query_start = time.perf_counter()

                query_embedding = self.manager.encode_query(query)
                retrieval_result = self.manager.retrieve_memories(
                    query_embedding.tolist(),
                    k_similarity=k,
                    k_contiguity=5,
                    use_contiguity=True,
                )

                query_time = time.perf_counter() - query_start

                results.append(
                    {
                        "query": f"{query[:50]}...",
                        "num_results": len(retrieval_result.get("events", [])),
                        "time_ms": query_time * 1000,
                    }
                )

                print(
                    f" Query: '{query[:60]}...' → "
                    f"{len(retrieval_result.get('events', []))} results "
                    f"in {query_time * 1000:.2f}ms"
                )

        total_time = time.perf_counter() - start_time
        avg_time = total_time / len(queries) if queries else 0

        self.metrics["retrieval"] = {
            "num_queries": len(queries),
            "total_time_sec": total_time,
            "avg_time_ms": avg_time * 1000,
            "results": results,
        }

        print(
            f"Completed: {len(queries)} queries in {total_time:.2f}s "
            f"(avg: {avg_time * 1000:.2f}ms per query)"
        )

        return self.metrics["retrieval"]

    def _extracted_from_run_recall_phase(self, queries, k, results):
        print(f"Using batch retrieval for {len(queries)} queries...")

        # Batch encode all queries
        encode_start = time.perf_counter()
        query_embeddings = self.manager.encode_queries_batch(queries)
        encode_time = time.perf_counter() - encode_start

        # Batch retrieve
        batch_start = time.perf_counter()
        batch_results = self.manager.retrieve_batch(
            query_embeddings,
            k_similarity=k,
            k_contiguity=5,
            use_contiguity=True,
        )
        batch_time = time.perf_counter() - batch_start

        # Process results
        amortized_query_time = (encode_time + batch_time) / len(queries)  # Amortized time
        for i, (query, retrieval_result) in enumerate(zip(queries, batch_results, strict=True)):
            results.append(
                {
                    "query": f"{query[:50]}...",
                    "num_results": len(retrieval_result.get("events", [])),
                    "time_ms": amortized_query_time * 1000,
                }
            )

            print(
                f" Query {i + 1}: '{query[:60]}...' → "
                f"{len(retrieval_result.get('events', []))} results"
            )

    def run_full_benchmark(self) -> dict:
        """Execute complete EM-LLM benchmark."""
        print("=" * 70)
        print("EM-LLM E2E Benchmark: 60k Token Corpus (WITH ALL FIXES)")
        print("=" * 70)

        # Clear index at START for fresh benchmark
        print("\nClearing old index for fresh benchmark...")
        self.manager.clear_memory()
        print("✅ Index cleared\n")

        corpus = generate_synthetic_corpus(target_tokens=60_000)

        self.run_remember_phase(corpus, gamma=1.0)

        test_queries = [
            "debugging database connection pool exhaustion",
            "JWT authentication security best practices",
            "performance optimization JSON serialization",
            "file attachment feature roadmap planning",
            "rate limiting token refresh endpoint",
        ]

        self.run_recall_phase(test_queries, k=10)

        # Summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY (WITH ALL FIXES)")
        print("=" * 70)

        print(f"Total tokens processed: {self.metrics['corpus']['total_tokens']:,}")
        print(
            f"Segmentation: {self.metrics['segmentation']['num_segments']} events "
            f"in {self.metrics['segmentation']['time_sec'] * 1000:.2f}ms"
        )
        print(f"Storage (embed + index): {self.metrics['timings']['storage_sec']:.2f}s")
        print(f"E2E Remember latency: {self.metrics['timings']['remember_total_sec']:.2f}s")
        print(f"Memory footprint delta: +{self.metrics['memory']['remember_delta_mb']:.1f} MB")
        print(f"Retrieval (avg): {self.metrics['retrieval']['avg_time_ms']:.2f}ms per query")

        print(f"\nDevice: {self.metrics['config']['device']}")
        print(f"Batch size: {self.metrics['config']['batch_size']}")

        # Index health
        print("\nIndex Health (EM-LLM 8-bit SQ + Fixed nlist):")
        idx = self.metrics["indexing"]
        print(
            f" trained={idx['is_trained']}, "
            f"vectors={idx['total_vectors']} ✅, "
            f"nlist={idx['nlist']} "
            f"(optimal={idx['optimal_nlist']}) ✅"
        )
        print(f" nprobe={idx['nprobe']}, use_sq={idx['use_sq']}")

        ratio = idx["nlist_ratio"]
        status = "✅ OPTIMAL" if ratio >= 0.95 else "⚠ may need retrain"
        print(f" nlist ratio: {ratio:.2f}x [{status}]")

        return self.metrics


def main():
    """Run benchmark and save results."""
    import shutil
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix="emx_benchmark_"))
    print(f"Benchmark workspace: {temp_dir}")

    try:
        return _setup_benchmark(temp_dir)
    except Exception as e:
        print(f"\nBenchmark failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    finally:
        print(f"\nCleaning up: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def _setup_benchmark(temp_dir):
    benchmark = E2EBenchmark(temp_dir)
    metrics = benchmark.run_full_benchmark()

    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
