# OTLP Exporter Manual Testing Guide

## Overview

This document provides manual testing procedures for the OpenTelemetry OTLP metrics exporter integration with Grafana Cloud.

**Context:** Integration tests validate the `HealthTrackingExporter` wrapper doesn't break the `MetricExporter` interface. Manual testing with a real OTLP endpoint validates end-to-end data flow.

---

## Prerequisites

1. **Grafana Cloud Account**: Sign up at https://grafana.com/products/cloud/
2. **OTLP Credentials**: Generate an API token from Grafana Cloud Access Policies
3. **Base64 Encoding**: Encode credentials as `<instance_id>:<api_token>`

---

## Test 1: OTLP Initialization (Local)

**Purpose:** Verify server starts without crash when OTLP endpoint configured.

```bash
cd /home/malu/.projects/emx-mcp-server

# Mock OTLP endpoint (no actual export)
OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/metrics" \
OTEL_SERVICE_NAME="emx-mcp-test" \
uv run python -c "
from emx_mcp.metrics.setup import setup_metrics
from emx_mcp.metrics import get_health_tracker

config = {}
meter = setup_metrics(config)
tracker = get_health_tracker()

print(f'Exporters: {tracker.exporters}')
print('✓ No AttributeError crash')
"
```

**Expected Output:**
```
Exporters: ['otlp(http://localhost:4318/v1/metrics)']
✓ No AttributeError crash
```

**Success Criteria:**
- ✓ No `AttributeError: '_preferred_temporality'`
- ✓ OTLP exporter registered in health tracker

---

## Test 2: MCP Server Startup with OTLP

**Purpose:** Verify full server initialization with OTLP exporter.

```bash
# Start server with OTLP endpoint (will fail to export but shouldn't crash)
OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/metrics" \
OTEL_SERVICE_NAME="emx-mcp-test" \
OTEL_METRIC_EXPORT_INTERVAL="60000" \
uv run python -m emx_mcp.server
```

**Expected Output:**
```
INFO: MCP server initialized successfully
INFO: 2 resources registered
INFO: 5 tools registered
INFO: Listening on stdin/stdout
```

**Success Criteria:**
- ✓ Server starts without crash
- ✓ No AttributeError in logs
- ✓ Server responds to MCP protocol requests

**Terminate:** Press `Ctrl+C` after verifying startup

---

## Test 3: Grafana Cloud Integration (Production)

**Purpose:** Validate metrics export to real Grafana Cloud backend.

### Step 1: Configure Credentials

```bash
# Get your Grafana Cloud OTLP endpoint
# Example: https://otlp-gateway-prod-us-central-0.grafana.net/otlp

# Encode credentials: <instance_id>:<api_token>
echo -n "YOUR_INSTANCE_ID:YOUR_API_TOKEN" | base64

# Export environment variables
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp-gateway-prod-YOUR-REGION.grafana.net/otlp"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic YOUR_BASE64_ENCODED_CREDENTIALS"
export OTEL_SERVICE_NAME="emx-mcp-server"
export OTEL_ENVIRONMENT="test"
export OTEL_METRIC_EXPORT_INTERVAL="10000"  # 10 seconds for testing
```

### Step 2: Start Server

```bash
uv run python -m emx_mcp.server
```

### Step 3: Generate Metrics

Interact with the server via MCP protocol (use Claude Desktop or MCP inspector):

```json
// Call memory operations to generate metrics
{
  "method": "tools/call",
  "params": {
    "name": "memory_operations",
    "arguments": {
      "operation": "add",
      "text": "Test embedding generation"
    }
  }
}
```

### Step 4: Verify in Grafana Cloud

1. Navigate to **Grafana Cloud → Explore**
2. Select **Prometheus** data source
3. Query metrics:
   ```promql
   # Check metric presence
   emx_operations_total{service_name="emx-mcp-server"}
   
   # Embedding latency percentiles
   histogram_quantile(0.95, rate(emx_embedding_duration_bucket[5m]))
   
   # Vector search performance
   histogram_quantile(0.99, rate(emx_vector_search_duration_bucket[5m]))
   
   # Error rate
   rate(emx_operations_errors[5m])
   ```

4. Import dashboard from `docs/grafana-dashboard-example.json`

**Success Criteria:**
- ✓ Metrics appear in Grafana Cloud within 30 seconds
- ✓ `metrics://health` resource shows successful exports
- ✓ Dashboard panels display data
- ✓ No errors in server logs

---

## Test 4: Health Resource Validation

**Purpose:** Verify `metrics://health` MCP resource tracks export status.

```bash
# Start server with OTLP
OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/metrics" \
uv run python -m emx_mcp.server
```

**MCP Request:**
```json
{
  "method": "resources/read",
  "params": {
    "uri": "metrics://health"
  }
}
```

**Expected Response:**
```json
{
  "contents": [
    {
      "uri": "metrics://health",
      "mimeType": "application/json",
      "text": "{
        \"healthy\": true,
        \"exporters\": [\"otlp(http://localhost:4318/v1/metrics)\"],
        \"last_success\": {
          \"timestamp\": 1698765432.123,
          \"seconds_ago\": 15
        },
        \"stats\": {
          \"total_exports\": 10,
          \"failed_exports\": 0,
          \"success_rate\": 1.0
        }
      }"
    }
  ]
}
```

**Success Criteria:**
- ✓ Resource returns JSON health status
- ✓ `last_success.timestamp` updates after exports
- ✓ `stats.success_rate` is 1.0 for successful exports
- ✓ `exporters` array contains OTLP endpoint

---

## Troubleshooting

### Issue: `AttributeError: '_preferred_temporality'`

**Symptom:** Server crashes on startup with OTLP endpoint configured.

**Resolution:** Update to commit with `__getattr__` fix in `emx_mcp/metrics/setup.py`.

**Validate Fix:**
```bash
cd /home/malu/.projects/emx-mcp-server
git log --oneline -1
# Should show commit with "Fix HealthTrackingExporter attribute proxying"

uv run pytest tests/test_metrics_health.py::test_wrapper_with_periodic_reader -v
# Should pass
```

---

### Issue: Metrics Not Appearing in Grafana Cloud

**Checklist:**
1. Verify OTLP endpoint format: `https://otlp-gateway-prod-REGION.grafana.net/otlp`
2. Check Authorization header: `Authorization=Basic <base64>`
3. Validate base64 encoding: `echo "<instance_id>:<api_token>" | base64`
4. Check `metrics://health` for export failures
5. Verify firewall allows HTTPS egress to Grafana Cloud

**Debug Command:**
```bash
# Enable verbose OTel logging
OTEL_LOG_LEVEL="debug" \
OTEL_EXPORTER_OTLP_ENDPOINT="..." \
OTEL_EXPORTER_OTLP_HEADERS="..." \
uv run python -m emx_mcp.server
```

---

### Issue: Export Failures in Health Resource

**Symptom:** `metrics://health` shows `failed_exports > 0`.

**Common Causes:**
- **Network timeout**: Increase `OTEL_METRIC_EXPORT_TIMEOUT` (default: 10000ms)
- **Invalid credentials**: Re-generate API token in Grafana Cloud
- **Rate limiting**: Increase `OTEL_METRIC_EXPORT_INTERVAL` (default: 10000ms)

**Resolution:**
```bash
# Query health for error details
# MCP request: resources/read with uri="metrics://health"
# Check "last_failure.error" field for specific error message
```

---

## Cleanup

```bash
# Unset environment variables
unset OTEL_EXPORTER_OTLP_ENDPOINT
unset OTEL_EXPORTER_OTLP_HEADERS
unset OTEL_SERVICE_NAME
unset OTEL_ENVIRONMENT
unset OTEL_METRIC_EXPORT_INTERVAL
```

---

## References

- **Grafana Cloud Setup**: `docs/GRAFANA_CLOUD_SETUP.md`
- **Dashboard Import**: `docs/grafana-dashboard-example.json`
- **ADR**: `docs/adr/001-opentelemetry-metrics.md`
- **Integration Tests**: `tests/test_metrics_health.py`
