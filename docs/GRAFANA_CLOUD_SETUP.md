# Grafana Cloud Metrics Setup

Quick guide to send EMX-MCP Server metrics to Grafana Cloud for monitoring and dashboards.

---

## What You Get

Once configured, you'll see real-time dashboards showing:
- **Embedding performance**: How fast tokens are encoded
- **Search latency**: Vector search response times
- **Memory usage**: Events stored, index size, retrieval patterns
- **Error rates**: Failed operations, timeouts

---

## Prerequisites

1. **Grafana Cloud account** (free tier available)
   - Sign up at https://grafana.com/auth/sign-up/create-user
   - Note your **instance URL** and **region** (e.g., `prod-us-east-0`)

2. **Generate API key**:
   - Go to **Grafana Cloud Portal → Administration → Access Policies**
   - Create new access policy with `metrics:write` permission
   - Generate token and **save it** (shown only once)

---

## Setup: 3 Steps

### Step 1: Get Your OTLP Endpoint

Your Grafana Cloud OTLP endpoint format:
```
https://otlp-gateway-<REGION>.grafana.net/otlp
```

**Example regions**:
- `prod-us-east-0` (US East)
- `prod-eu-west-0` (EU West)
- `prod-ap-southeast-0` (Asia Pacific)

Find your region in **Grafana Cloud Portal → Administration → Settings**.

---

### Step 2: Create Authorization Header

Encode your credentials as Base64:

```bash
# Format: <instance_id>:<api_token>
# Example instance_id: 123456
# Example token: glc_abc123xyz...

echo -n "123456:glc_abc123xyz..." | base64
```

You'll get something like: `MTIzNDU2OmdsY19hYmMxMjN4eXouLi4=`

---

### Step 3: Add to MCP Client Config

Update your MCP client configuration (e.g., Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "emx-memory": {
      "command": "uvx",
      "args": ["emx-mcp-server"],
      "env": {
        "OTEL_SERVICE_NAME": "emx-mcp-server",
        "OTEL_ENVIRONMENT": "production",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Basic MTIzNDU2OmdsY19hYmMxMjN4eXouLi4=",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_METRIC_EXPORT_INTERVAL": "10000"
      }
    }
  }
}
```

**Replace**:
- `prod-us-east-0` → your region
- `MTIzNDU2OmdsY19hYmMxMjN4eXouLi4=` → your Base64-encoded credentials

**Restart your MCP client** for changes to take effect.

---

## Verify It's Working

### 1. Check Metrics are Flowing

In **Grafana Cloud Portal → Explore**:
```promql
{service_name="emx-mcp-server"}
```

You should see metrics within 10-30 seconds after EMX server starts.

### 2. View Available Metrics

EMX server exports these metrics:

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `emx.embedding.duration` | Histogram | Time to encode tokens (seconds) | `device`, `batch_size_bucket` |
| `emx.vector_search.duration` | Histogram | Vector search latency (seconds) | `device`, `batch_api`, `query_size_bucket`, `k` |
| `emx.operations.total` | Counter | Total operations by type | `operation`, `device`, `batch_api` |
| `emx.operations.errors` | Counter | Failed operations | `operation`, `device`, `error_type` |
| `emx.batch.size` | Histogram | Batch size distribution | `operation` |

---

## Create a Dashboard

### Production-Ready Dashboard

A complete dashboard is available in this repository:

1. **Download the dashboard JSON**:
   - File: [`grafana-dashboard-example.json`](./grafana-dashboard-example.json)
   - Or view on GitHub: [docs/grafana-dashboard-example.json](https://github.com/yourusername/emx-mcp-server/blob/main/docs/grafana-dashboard-example.json)

2. **Import into Grafana Cloud**:
   - Go to **Grafana Cloud → Dashboards → New → Import**
   - Click **Upload JSON file** and select `grafana-dashboard-example.json`
   - Select your Prometheus data source
   - Click **Import**

3. **What you get**:
   - **Embedding Performance**: Latency percentiles (p50/p95/p99), batch size distribution
   - **Vector Search Performance**: Search latency by device/API, query count distribution
   - **Operation Rates**: Throughput and error rates by operation type
   - **System Overview**: Health stats and p95 latency summary

4. **Customize**:
   - Click any panel → Edit to adjust queries
   - Add more panels by clicking **Add panel**
   - Set thresholds and alerts via **Alert** tab

---

## Troubleshooting

### No metrics appearing in Grafana Cloud

**Check 1: Verify endpoint URL**
```bash
# Test connectivity
curl -v https://otlp-gateway-prod-us-east-0.grafana.net/otlp
# Should return 405 Method Not Allowed (means endpoint exists)
```

**Check 2: Validate authorization header**
```bash
# Decode your Base64 to verify format
echo "MTIzNDU2OmdsY19hYmMxMjN4eXouLi4=" | base64 -d
# Should output: 123456:glc_abc123xyz...
```

**Check 3: Check MCP client logs**
- Look for OTLP export errors in your MCP client's error logs
- Common issue: Wrong region in endpoint URL

**Check 4: Test with console exporter first**
```json
"env": {
  "OTEL_METRICS_CONSOLE": "true"
}
```
Run MCP server and redirect stderr to file:
```bash
uvx emx-mcp-server 2> metrics.log
```
If you see metrics in `metrics.log`, the problem is with Grafana Cloud config (not EMX).

---

### Metrics delayed or missing

**Issue**: Metrics export every 10 seconds by default.

**Solution**: Reduce export interval for faster feedback:
```json
"OTEL_METRIC_EXPORT_INTERVAL": "5000"
```

---

### "Invalid authorization" errors

**Issue**: Base64 encoding includes newlines or wrong format.

**Solution**: Use `-n` flag with echo:
```bash
# Correct (no newline)
echo -n "123456:glc_abc123xyz..." | base64

# Wrong (includes newline)
echo "123456:glc_abc123xyz..." | base64
```

---

## Environment Variables Reference

All metrics-related environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_SERVICE_NAME` | `emx-mcp-server` | Service identifier in Grafana |
| `OTEL_ENVIRONMENT` | `development` | Environment label (dev/staging/prod) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | _(none)_ | Grafana Cloud OTLP endpoint URL |
| `OTEL_EXPORTER_OTLP_HEADERS` | _(none)_ | `Authorization=Basic <token>` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | Protocol (required for Grafana Cloud) |
| `OTEL_METRIC_EXPORT_INTERVAL` | `10000` | Export interval (milliseconds) |
| `OTEL_METRICS_CONSOLE` | `false` | Enable stderr console output (debugging) |

**See [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) for complete documentation.**

---

## Cost Considerations

**Grafana Cloud Free Tier** (as of 2025):
- 10,000 series (unique metric combinations)
- 14-day retention
- 3 active users

**EMX-MCP Server metrics usage**:
- ~20 base metrics
- ~5-10 series per metric (depending on label cardinality)
- **Estimated total**: 100-200 series

**Recommendation**: Free tier is sufficient for individual use. Upgrade if running multiple EMX instances or need longer retention.

---

## Next Steps

Once metrics are flowing:

1. **Set up alerts**: Get notified when search latency exceeds thresholds
   - Go to **Grafana Cloud → Alerting → New Alert Rule**
   - Example: Alert if `p95 search latency > 1s` for 5 minutes

2. **Explore correlations**: Compare embedding performance vs GPU usage
   - Use `EMX_MODEL_DEVICE=cuda` env var
   - Compare metrics before/after GPU acceleration

3. **Track memory growth**: Monitor index size over time
   - Set alerts for unexpected growth (memory leaks, inefficient storage)

4. **Optimize based on data**: Use metrics to tune configuration
   - High encoding duration? Increase `EMX_MODEL_BATCH_SIZE`
   - Slow searches? Adjust `EMX_STORAGE_NPROBE`

---

## Additional Resources

- **Grafana Cloud Docs**: https://grafana.com/docs/grafana-cloud/send-data/otlp/
- **OpenTelemetry OTLP Spec**: https://opentelemetry.io/docs/specs/otlp/
- **EMX Environment Variables**: [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md)
- **Performance Benchmarks**: [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)
