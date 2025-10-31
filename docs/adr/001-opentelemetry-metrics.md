# ADR 001: OpenTelemetry Metrics for Production Observability

**Status:** Active  
**Date:** 2025-10-31  
**Context:** Production deployment requires visibility into embedding performance, vector search latency, and OTLP export health.

---

## Decision

Use **OpenTelemetry SDK** with **OTLP HTTP/protobuf exporter** for metrics collection, targeting Grafana Cloud as the observability backend.

### Key Instruments (5 total, ~20 time series)

1. **`emx.embedding.duration`** (Histogram)  
   - Labels: `device` (cpu/cuda), `batch_size_bucket` (1, 2-10, 11-50, 51+)
   - Rationale: Percentile analysis (p50/p95/p99) for tail latency tracking

2. **`emx.vector_search.duration`** (Histogram)  
   - Labels: `device`, `batch_api` (single/batch), `query_size_bucket` (1, 2-10, 11-50, 51+), `k` (results returned)
   - Rationale: Distinguish IVF single vs batch search performance

3. **`emx.operations.total`** (Counter)  
   - Labels: `operation` (embedding/vector_search), `device`, `batch_api`
   - Rationale: Throughput rates via `rate()` queries

4. **`emx.operations.errors`** (Counter)  
   - Labels: `operation`, `device`, `error_type` (index_not_trained/faiss_error/cuda_oom)
   - Rationale: Error rate alerting and failure mode analysis

5. **`emx.batch.size`** (Histogram)  
   - Label: `operation`
   - Rationale: Track actual batch sizes for GPU/CPU routing decisions

### Export Configuration

- **Default interval:** 10 seconds (`OTEL_METRIC_EXPORT_INTERVAL=10000`)
- **Console exporter:** Opt-in via `OTEL_METRICS_CONSOLE=true` (writes to stderr, MCP STDIO-safe)
- **OTLP exporter:** Enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` set (Grafana Cloud endpoint)
- **Health tracking:** `HealthTrackingExporter` wrapper records export success/failure for `metrics://health` MCP resource

---

## Rationale

### Why OpenTelemetry?

- **Vendor-neutral:** Not locked to Grafana; compatible with Prometheus, Datadog, AWS CloudWatch
- **Industry standard:** Part of CNCF, widely adopted in cloud-native systems
- **SDK maturity:** Python SDK stable since 2022, active maintenance

### Why OTLP over Direct Prometheus?

- **Push model:** MCP servers are short-lived processes; pull-based scraping unreliable
- **Grafana Cloud native:** OTLP is the primary ingestion path for Grafana Cloud Metrics
- **Reduced surface:** No need to expose HTTP endpoint for scraping (security benefit for STDIO transport)

### Why Histograms for Latency?

- **Percentile calculations:** Grafana's `histogram_quantile()` computes p50/p95/p99 server-side
- **Aggregatable:** Unlike summaries, histograms can be aggregated across dimensions (e.g., sum across devices)
- **Trade-off:** 13 buckets per histogram (~13 time series per instrument) vs 1 gauge

### Why Counters for Operations?

- **Rate derivation:** `rate(emx_operations_total[1m])` gives ops/sec without client-side calculation
- **Monotonic guarantees:** Counters never decrease, simplifying alerting logic (vs gauges which can spike/drop)

### Cardinality Control Strategy

**Problem:** Unbounded cardinality (e.g., per-query-ID labels) explodes storage and query costs.

**Solution:** Bucketed labels reduce cardinality from O(n queries) to O(1 buckets):

- `batch_size_bucket`: 4 values (1, 2-10, 11-50, 51+) instead of 1000 unique batch sizes
- `query_size_bucket`: 4 values instead of unbounded query counts
- `k`: 3-5 common values (5, 10, 20) instead of arbitrary top-k sizes

**Result:** ~20 active time series (manageable for Grafana Cloud free tier: 10k series limit).

---

## Consequences

### Benefits

✅ **Production visibility:** Real-time latency/throughput/error dashboards without log parsing  
✅ **Debugging aid:** Health resource (`metrics://health`) exposes OTLP export failures for connectivity troubleshooting  
✅ **Benchmark validation:** Performance claims (4-5x GPU speedup) verifiable in production  
✅ **SLO foundation:** p95 latency and error rate metrics enable future SLI/SLO definitions

### Costs

⚠️ **Export overhead:** 10-second intervals add ~0.1-0.5ms per operation (instrumentation + aggregation)  
⚠️ **Dependency weight:** OpenTelemetry SDK adds 3 packages (~2MB installed size)  
⚠️ **Learning curve:** Histogram bucketing and PromQL queries require operator training

### Mitigations

- **Overhead:** Negligible compared to embedding generation (10-50ms) and vector search (5-20ms)
- **Dependency size:** Acceptable for server deployment (not embedded contexts)
- **Learning curve:** Dashboard JSON provided in `/docs/grafana-dashboard-example.json` with pre-built queries

---

## Alternatives Considered

### 1. Structured Logging Only (e.g., loguru with JSON)

**Pros:** No additional dependencies, works with existing logging setup  
**Cons:** Requires log aggregation (Loki/CloudWatch Logs), no histogram/percentile support, expensive at scale

**Why rejected:** Metrics are purpose-built for aggregation; logs are for event investigation.

### 2. Direct Prometheus Client (pull-based)

**Pros:** Lightweight, simple `/metrics` HTTP endpoint  
**Cons:** Requires exposing HTTP port (conflicts with MCP STDIO transport), unreliable for short-lived processes

**Why rejected:** Push-based OTLP better for MCP server lifecycle.

### 3. StatsD + Telegraf

**Pros:** Battle-tested, low overhead  
**Cons:** Additional daemon dependency (Telegraf), less ecosystem support than OTel

**Why rejected:** OpenTelemetry consolidates traces/metrics/logs under single standard.

---

## Enforcement

This ADR is enforced by:

1. **Config validation:** `tests/test_config_integration.py` ensures OTLP endpoint validation
2. **Health resource:** `metrics://health` fails if tracker not initialized (runtime check)
3. **Dashboard contract:** `/docs/grafana-dashboard-example.json` queries exact metric names from `instruments.py`

**Update trigger:** Changes to instrument names, labels, or export intervals require updating:
- This ADR (rationale section)
- `emx_mcp/metrics/instruments.py` (instrument definitions)
- `docs/grafana-dashboard-example.json` (dashboard queries)
- `docs/GRAFANA_CLOUD_SETUP.md` (environment variables)

---

## References

- [OpenTelemetry Metrics API Spec](https://opentelemetry.io/docs/specs/otel/metrics/api/)
- [Grafana Cloud OTLP Setup](../GRAFANA_CLOUD_SETUP.md)
- [Production Dashboard](../grafana-dashboard-example.json)
- [Performance Benchmark Results](../PERFORMANCE_REPORT.md)
