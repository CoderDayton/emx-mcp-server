"""
Integration tests for metrics health tracking.

Validates HealthTrackingExporter wrapper doesn't break MetricExporter interface.
"""

import pytest
from unittest.mock import Mock, MagicMock
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from emx_mcp.metrics.setup import (
    MetricsHealthTracker,
    HealthTrackingExporter,
)


class MockExporter(MetricExporter):
    """Mock exporter with required SDK attributes."""
    
    def __init__(self):
        self._preferred_temporality = {}
        self._preferred_aggregation = {}
        self.exported_data = []
        self.should_fail = False
    
    def export(self, metrics_data, timeout_millis=10000, **kwargs):
        """Mock export implementation."""
        if self.should_fail:
            raise RuntimeError("Mock export failure")
        
        self.exported_data.append(metrics_data)
        return MetricExportResult.SUCCESS
    
    def shutdown(self, timeout_millis=30000, **kwargs):
        """Mock shutdown."""
        pass
    
    def force_flush(self, timeout_millis=10000):
        """Mock force flush."""
        return True


@pytest.fixture
def health_tracker():
    """Fresh health tracker for each test."""
    return MetricsHealthTracker()


@pytest.fixture
def mock_exporter():
    """Fresh mock exporter for each test."""
    return MockExporter()


def test_health_tracker_initialization(health_tracker):
    """Test health tracker starts in clean state."""
    assert health_tracker.last_export_success is None
    assert health_tracker.last_export_failure is None
    assert health_tracker.total_exports == 0
    assert health_tracker.failed_exports == 0
    assert health_tracker.last_error is None
    assert health_tracker.exporters == []


def test_health_tracker_record_success(health_tracker):
    """Test successful export recording."""
    health_tracker.record_success()
    
    assert health_tracker.last_export_success is not None
    assert health_tracker.last_export_failure is None
    assert health_tracker.total_exports == 1
    assert health_tracker.failed_exports == 0
    
    # Multiple successes
    health_tracker.record_success()
    assert health_tracker.total_exports == 2
    assert health_tracker.failed_exports == 0


def test_health_tracker_record_failure(health_tracker):
    """Test failed export recording."""
    health_tracker.record_failure("Connection timeout")
    
    assert health_tracker.last_export_failure is not None
    assert health_tracker.total_exports == 1
    assert health_tracker.failed_exports == 1
    assert health_tracker.last_error == "Connection timeout"


def test_health_tracker_get_health(health_tracker):
    """Test health status reporting."""
    # Initial state
    health = health_tracker.get_health()
    assert health["healthy"] is False  # No successful exports yet
    assert health["stats"]["total_exports"] == 0
    assert health["stats"]["success_rate"] == 0.0
    
    # After success
    health_tracker.record_success()
    health = health_tracker.get_health()
    assert health["healthy"] is True
    assert health["last_success"]["timestamp"] is not None
    assert health["stats"]["total_exports"] == 1
    assert health["stats"]["success_rate"] == 1.0
    
    # After failure
    health_tracker.record_failure("Test error")
    health = health_tracker.get_health()
    assert health["healthy"] is False  # Has failures now
    assert health["last_failure"] is not None
    assert health["last_failure"]["error"] == "Test error"
    assert health["stats"]["total_exports"] == 2
    assert health["stats"]["failed_exports"] == 1
    assert health["stats"]["success_rate"] == 0.5


def test_wrapper_proxies_export(mock_exporter, health_tracker):
    """Test HealthTrackingExporter proxies export calls correctly."""
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    # Mock metrics data
    metrics_data = Mock()
    
    result = wrapper.export(metrics_data, timeout_millis=5000)
    
    assert result == MetricExportResult.SUCCESS
    assert len(mock_exporter.exported_data) == 1
    assert mock_exporter.exported_data[0] == metrics_data
    assert health_tracker.total_exports == 1
    assert health_tracker.failed_exports == 0


def test_wrapper_records_success(mock_exporter, health_tracker):
    """Test wrapper records successful exports."""
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    metrics_data = Mock()
    wrapper.export(metrics_data)
    
    assert health_tracker.last_export_success is not None
    assert health_tracker.total_exports == 1
    assert health_tracker.failed_exports == 0


def test_wrapper_records_failure(mock_exporter, health_tracker):
    """Test wrapper records failed exports."""
    mock_exporter.should_fail = True
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    metrics_data = Mock()
    
    with pytest.raises(RuntimeError, match="Mock export failure"):
        wrapper.export(metrics_data)
    
    assert health_tracker.last_export_failure is not None
    assert health_tracker.total_exports == 1
    assert health_tracker.failed_exports == 1
    assert "RuntimeError" in health_tracker.last_error
    assert "Mock export failure" in health_tracker.last_error


def test_wrapper_proxies_sdk_attributes(mock_exporter, health_tracker):
    """
    Test wrapper proxies OpenTelemetry SDK internal attributes.
    
    This is the critical fix: PeriodicExportingMetricReader accesses
    _preferred_temporality and _preferred_aggregation during initialization.
    """
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    # Test attribute proxying via __getattr__
    assert hasattr(wrapper, "_preferred_temporality")
    assert hasattr(wrapper, "_preferred_aggregation")
    assert wrapper._preferred_temporality == mock_exporter._preferred_temporality
    assert wrapper._preferred_aggregation == mock_exporter._preferred_aggregation


def test_wrapper_with_periodic_reader(mock_exporter, health_tracker):
    """
    Test wrapper works with PeriodicExportingMetricReader.
    
    This integration validates the fix for AttributeError crash
    when OTLP endpoint is configured.
    """
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    # This used to crash with AttributeError before __getattr__ fix
    reader = PeriodicExportingMetricReader(
        wrapper,
        export_interval_millis=60000,  # 60 seconds
    )
    
    # Verify reader initialized successfully
    assert reader is not None
    
    # Clean shutdown
    reader.shutdown()


def test_wrapper_shutdown_propagation(mock_exporter, health_tracker):
    """Test wrapper propagates shutdown to inner exporter."""
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    # Mock to track shutdown calls
    mock_exporter.shutdown = Mock(return_value=None)
    
    wrapper.shutdown(timeout_millis=5000)
    
    # Wrapper passes timeout_millis as positional arg to inner
    mock_exporter.shutdown.assert_called_once_with(5000)


def test_wrapper_force_flush_propagation(mock_exporter, health_tracker):
    """Test wrapper propagates force_flush to inner exporter."""
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    # Mock to track force_flush calls
    mock_exporter.force_flush = Mock(return_value=True)
    
    result = wrapper.force_flush(timeout_millis=3000)
    
    assert result is True
    # Wrapper passes timeout_millis as positional arg to inner
    mock_exporter.force_flush.assert_called_once_with(3000)


def test_multiple_exports_tracking(mock_exporter, health_tracker):
    """Test tracking across multiple export cycles."""
    wrapper = HealthTrackingExporter(mock_exporter, health_tracker, "test")
    
    metrics_data = Mock()
    
    # 3 successful exports
    for _ in range(3):
        wrapper.export(metrics_data)
    
    assert health_tracker.total_exports == 3
    assert health_tracker.failed_exports == 0
    
    # 1 failed export
    mock_exporter.should_fail = True
    with pytest.raises(RuntimeError):
        wrapper.export(metrics_data)
    
    assert health_tracker.total_exports == 4
    assert health_tracker.failed_exports == 1
    
    # 2 more successful exports
    mock_exporter.should_fail = False
    for _ in range(2):
        wrapper.export(metrics_data)
    
    assert health_tracker.total_exports == 6
    assert health_tracker.failed_exports == 1
    assert health_tracker.get_health()["stats"]["success_rate"] == pytest.approx(5/6)
