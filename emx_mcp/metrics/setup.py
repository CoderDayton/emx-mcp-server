"""OpenTelemetry metrics initialization."""

import os
import sys
import time
from typing import Optional, Dict, Any
from collections.abc import Sequence

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
    MetricExporter,
    MetricExportResult,
)
from opentelemetry.sdk.metrics._internal.export import MetricsData
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

_meter: Optional[metrics.Meter] = None
_health_tracker: Optional["MetricsHealthTracker"] = None


class MetricsHealthTracker:
    """
    Track metrics export health for observability.
    
    Provides /health MCP resource endpoint with:
    - Last successful export timestamp
    - Total exports (success + failures)
    - Export failures count
    - Current exporter status
    """
    
    def __init__(self):
        self.last_export_success: Optional[float] = None
        self.last_export_failure: Optional[float] = None
        self.total_exports: int = 0
        self.failed_exports: int = 0
        self.last_error: Optional[str] = None
        self.exporters: list[str] = []
    
    def record_success(self):
        """Record successful export."""
        self.last_export_success = time.time()
        self.total_exports += 1
    
    def record_failure(self, error: str):
        """Record failed export."""
        self.last_export_failure = time.time()
        self.total_exports += 1
        self.failed_exports += 1
        self.last_error = error
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Health status dictionary with timestamps, counters, and exporter info
        """
        now = time.time()
        
        return {
            "healthy": self.last_export_success is not None and self.failed_exports == 0,
            "exporters": self.exporters,
            "last_success": {
                "timestamp": self.last_export_success,
                "seconds_ago": (now - self.last_export_success) if self.last_export_success else None,
            },
            "last_failure": {
                "timestamp": self.last_export_failure,
                "seconds_ago": (now - self.last_export_failure) if self.last_export_failure else None,
                "error": self.last_error,
            } if self.last_export_failure else None,
            "stats": {
                "total_exports": self.total_exports,
                "failed_exports": self.failed_exports,
                "success_rate": (
                    (self.total_exports - self.failed_exports) / self.total_exports
                    if self.total_exports > 0
                    else 0.0
                ),
            },
        }


class HealthTrackingExporter(MetricExporter):
    """Wraps an exporter to track health status."""
    
    def __init__(self, inner: MetricExporter, tracker: MetricsHealthTracker, name: str):
        self.inner = inner
        self.tracker = tracker
        self.name = name
    
    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10000,
        **kwargs,
    ) -> MetricExportResult:
        """Export metrics and track success/failure."""
        try:
            result = self.inner.export(metrics_data, timeout_millis, **kwargs)
            
            if result == MetricExportResult.SUCCESS:
                self.tracker.record_success()
            else:
                self.tracker.record_failure(f"{self.name}: Export returned {result}")
            
            return result
        except Exception as e:
            self.tracker.record_failure(f"{self.name}: {type(e).__name__}: {str(e)}")
            raise
    
    def shutdown(self, timeout_millis: float = 30000, **kwargs) -> None:
        """Shutdown inner exporter."""
        return self.inner.shutdown(timeout_millis, **kwargs)
    
    def force_flush(self, timeout_millis: float = 10000) -> bool:
        """Force flush inner exporter."""
        return self.inner.force_flush(timeout_millis)


def setup_metrics(config: dict) -> metrics.Meter:
    """
    Initialize OpenTelemetry metrics with console and optional OTLP exporter.
    
    Environment Variables:
        OTEL_SERVICE_NAME: Service name for metrics (default: emx-mcp-server)
        OTEL_EXPORTER_OTLP_ENDPOINT: Optional OTLP collector endpoint
        OTEL_METRIC_EXPORT_INTERVAL: Export interval in milliseconds (default: 10000)
    
    Args:
        config: Server configuration dictionary
        
    Returns:
        Configured Meter instance for creating instruments
    """
    global _meter, _health_tracker
    
    if _meter is not None:
        return _meter
    
    # Initialize health tracker
    _health_tracker = MetricsHealthTracker()
    
    # Create resource with service metadata
    service_name = os.getenv("OTEL_SERVICE_NAME", "emx-mcp-server")
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("OTEL_ENVIRONMENT", "development"),
        }
    )
    
    # Configure exporters
    readers = []
    export_interval_ms = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "10000"))
    
    # Add console exporter only if explicitly enabled (writes to stderr to avoid STDIO interference)
    # For MCP servers using STDIO transport, use OTLP exporter instead
    if os.getenv("OTEL_METRICS_CONSOLE", "false").lower() == "true":
        console_exporter = HealthTrackingExporter(
            ConsoleMetricExporter(out=sys.stderr),
            _health_tracker,
            "console"
        )
        console_reader = PeriodicExportingMetricReader(
            console_exporter,
            export_interval_millis=export_interval_ms,
        )
        readers.append(console_reader)
        _health_tracker.exporters.append("console")
    
    # Add OTLP exporter if endpoint configured
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        otlp_exporter = HealthTrackingExporter(
            OTLPMetricExporter(endpoint=otlp_endpoint),
            _health_tracker,
            "otlp"
        )
        otlp_reader = PeriodicExportingMetricReader(
            otlp_exporter,
            export_interval_millis=export_interval_ms,
        )
        readers.append(otlp_reader)
        _health_tracker.exporters.append(f"otlp({otlp_endpoint})")
    
    # Initialize MeterProvider
    provider = MeterProvider(
        resource=resource,
        metric_readers=readers,
    )
    metrics.set_meter_provider(provider)
    
    _meter = metrics.get_meter(__name__)
    
    return _meter


def get_meter() -> metrics.Meter:
    """
    Get initialized Meter instance.
    
    Returns:
        Active Meter instance
        
    Raises:
        RuntimeError: If setup_metrics() not called first
    """
    if _meter is None:
        raise RuntimeError("Metrics not initialized. Call setup_metrics() first.")
    return _meter


def get_health_tracker() -> Optional[MetricsHealthTracker]:
    """
    Get metrics health tracker instance.
    
    Returns:
        Health tracker if metrics initialized, None otherwise
    """
    return _health_tracker
