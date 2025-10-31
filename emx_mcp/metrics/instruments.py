"""Metric instruments for EMX MCP Server."""

import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

from opentelemetry import metrics

from emx_mcp.metrics.setup import get_meter


class MetricInstruments:
    """
    Centralized metric instruments for EMX MCP operations.
    
    SLI Focus (per component):
        - p95 latency (histograms)
        - error rate (counters)
    """
    
    def __init__(self):
        meter = get_meter()
        
        # Embedding generation metrics
        self.embedding_duration = meter.create_histogram(
            name="emx.embedding.duration",
            description="Embedding generation latency in seconds",
            unit="s",
        )
        
        # Vector search metrics
        self.vector_search_duration = meter.create_histogram(
            name="emx.vector_search.duration",
            description="Vector search latency in seconds",
            unit="s",
        )
        
        # Operation counters
        self.operations_total = meter.create_counter(
            name="emx.operations.total",
            description="Total operations by type",
            unit="1",
        )
        
        self.operations_errors = meter.create_counter(
            name="emx.operations.errors",
            description="Failed operations by type and reason",
            unit="1",
        )
        
        # Batch size tracking (for adaptive routing analysis)
        self.batch_size = meter.create_histogram(
            name="emx.batch.size",
            description="Batch size distribution",
            unit="1",
        )
    
    @contextmanager
    def track_embedding(self, batch_size: int, device: str):
        """
        Track embedding generation latency.
        
        Args:
            batch_size: Number of tokens/sequences in batch
            device: cpu/cuda
            
        Yields:
            None
            
        Example:
            with instruments.track_embedding(batch_size=32, device="cuda"):
                embeddings = model.encode(tokens)
        """
        start = time.perf_counter()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.operations_errors.add(
                1,
                {
                    "operation": "embedding",
                    "device": device,
                    "error_type": type(e).__name__,
                },
            )
            raise
        finally:
            duration = time.perf_counter() - start
            
            self.embedding_duration.record(
                duration,
                {
                    "device": device,
                    "batch_size_bucket": self._bucket_batch_size(batch_size),
                },
            )
            
            if not error_occurred:
                self.operations_total.add(
                    1,
                    {"operation": "embedding", "device": device},
                )
                self.batch_size.record(batch_size, {"operation": "embedding"})
    
    @contextmanager
    def track_vector_search(
        self,
        n_queries: int,
        k: int,
        device: str,
        use_batch_api: bool,
    ):
        """
        Track vector search latency.
        
        Args:
            n_queries: Number of query vectors
            k: Top-K results per query
            device: cpu/cuda
            use_batch_api: Whether batch API was used
            
        Yields:
            None
        """
        start = time.perf_counter()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.operations_errors.add(
                1,
                {
                    "operation": "vector_search",
                    "device": device,
                    "error_type": type(e).__name__,
                },
            )
            raise
        finally:
            duration = time.perf_counter() - start
            
            self.vector_search_duration.record(
                duration,
                {
                    "device": device,
                    "batch_api": str(use_batch_api),
                    "query_size_bucket": self._bucket_query_size(n_queries),
                    "k": str(k),
                },
            )
            
            if not error_occurred:
                self.operations_total.add(
                    1,
                    {
                        "operation": "vector_search",
                        "device": device,
                        "batch_api": str(use_batch_api),
                    },
                )
                self.batch_size.record(n_queries, {"operation": "vector_search"})
    
    def _bucket_batch_size(self, size: int) -> str:
        """Bucket batch sizes for cardinality control."""
        if size <= 16:
            return "1-16"
        elif size <= 64:
            return "17-64"
        elif size <= 256:
            return "65-256"
        else:
            return "257+"
    
    def _bucket_query_size(self, size: int) -> str:
        """Bucket query counts for cardinality control."""
        if size == 1:
            return "1"
        elif size <= 10:
            return "2-10"
        elif size <= 100:
            return "11-100"
        elif size <= 1000:
            return "101-1000"
        else:
            return "1000+"


# Singleton instance
_instruments: Optional[MetricInstruments] = None


def get_instruments() -> MetricInstruments:
    """
    Get singleton MetricInstruments instance.
    
    Returns:
        Active MetricInstruments instance
        
    Raises:
        RuntimeError: If metrics not initialized
    """
    global _instruments
    
    if _instruments is None:
        _instruments = MetricInstruments()
    
    return _instruments
