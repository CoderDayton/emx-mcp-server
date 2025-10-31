"""OpenTelemetry metrics initialization."""

import os
from typing import Optional

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource

_meter: Optional[metrics.Meter] = None


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
    global _meter
    
    if _meter is not None:
        return _meter
    
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
    
    # Always add console exporter for local debugging
    console_reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(),
        export_interval_millis=export_interval_ms,
    )
    readers.append(console_reader)
    
    # Add OTLP exporter if endpoint configured
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        otlp_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
        otlp_reader = PeriodicExportingMetricReader(
            otlp_exporter,
            export_interval_millis=export_interval_ms,
        )
        readers.append(otlp_reader)
    
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
