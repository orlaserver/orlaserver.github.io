# Monitoring and Metrics

The Orla daemon exposes Prometheus metrics at `GET /metrics` and returns per-request timing and cost data in every inference response. Together these give you real-time visibility into queue depth, latency, cost, and routing decisions across all your backends.

## Per-request metrics

Every inference response includes a `metrics` object with timing and token data. When using the LangChain integration, these appear in `response_metadata`:

```python
llm = stage.as_chat_model()
reply = llm.invoke(messages)
m = reply.response_metadata

print(f"Queue wait:       {m.get('queue_wait_ms')}ms")
print(f"Backend latency:  {m.get('backend_latency_ms')}ms")
print(f"Prompt tokens:    {m.get('prompt_tokens')}")
print(f"Completion tokens:{m.get('completion_tokens')}")
print(f"Estimated cost:   ${m.get('estimated_cost_usd', 0):.6f}")
```

When using `stage.execute()` directly:

```python
response = stage.execute("What is 24 * 17?")
m = response.metrics

print(f"Queue wait:       {m.queue_wait_ms}ms")
print(f"Backend latency:  {m.backend_latency_ms}ms")
print(f"Estimated cost:   ${m.estimated_cost_usd:.6f}")
```

Available fields:

| Field | Description |
|-------|-------------|
| `queue_wait_ms` | Time the request waited in the scheduler queue before a worker picked it up |
| `scheduler_decision_ms` | Time the scheduler spent selecting this request from the queue |
| `dispatch_ms` | Total time from dequeue to response |
| `backend_latency_ms` | Time spent in the LLM backend (non-streaming only) |
| `prompt_tokens` | Number of input tokens |
| `completion_tokens` | Number of output tokens |
| `estimated_cost_usd` | Estimated cost in USD based on the backend's registered cost model (null if no cost model) |
| `ttft_ms` | Time to first token (streaming only) |
| `tpot_ms` | Time per output token (streaming only) |

## Prometheus endpoint

The daemon serves Prometheus metrics in the standard exposition format at `GET /metrics`. Point your Prometheus scrape config at the daemon's listen address:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: orla
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:8081"]
```

## Available metrics

### Counters

`orla_requests_total` tracks the total number of inference requests, labeled by backend and status:

```
orla_requests_total{backend="sglang-qwen", status="success"} 1423
orla_requests_total{backend="sglang-qwen", status="error"} 12
```

`orla_estimated_cost_usd_total` tracks cumulative estimated spend per backend:

```
orla_estimated_cost_usd_total{backend="bedrock-strong"} 4.82
orla_estimated_cost_usd_total{backend="bedrock-cheap"} 0.31
```

`orla_accuracy_routing_total` tracks accuracy-based routing decisions when using cost policies:

```
orla_accuracy_routing_total{selected_backend="bedrock-cheap", status="ok"} 890
orla_accuracy_routing_total{selected_backend="bedrock-strong", status="ok"} 533
orla_accuracy_routing_total{selected_backend="bedrock-cheap", status="fallback"} 17
```

### Histograms

`orla_queue_wait_seconds` captures the distribution of time requests spend waiting in the queue:

```
orla_queue_wait_seconds_bucket{backend="sglang-qwen", le="0.016"} 1200
orla_queue_wait_seconds_bucket{backend="sglang-qwen", le="0.064"} 1380
```

`orla_backend_latency_seconds` captures the distribution of backend inference time:

```
orla_backend_latency_seconds_bucket{backend="sglang-qwen", le="1.6"} 1100
orla_backend_latency_seconds_bucket{backend="sglang-qwen", le="6.4"} 1400
```

`orla_estimated_cost_usd` captures the per-request cost distribution:

```
orla_estimated_cost_usd_bucket{backend="bedrock-strong", le="0.01"} 200
orla_estimated_cost_usd_bucket{backend="bedrock-strong", le="0.05"} 520
```

### Gauges

`orla_queue_depth` shows the current number of requests waiting per backend:

```
orla_queue_depth{backend="sglang-qwen"} 3
```

## Useful queries

Total request rate per backend:

```promql
rate(orla_requests_total[5m])
```

Error rate as a percentage:

```promql
sum(rate(orla_requests_total{status="error"}[5m])) by (backend)
/
sum(rate(orla_requests_total[5m])) by (backend)
* 100
```

P95 queue wait time:

```promql
histogram_quantile(0.95, rate(orla_queue_wait_seconds_bucket[5m]))
```

P99 backend latency:

```promql
histogram_quantile(0.99, rate(orla_backend_latency_seconds_bucket[5m]))
```

Spend per hour by backend:

```promql
rate(orla_estimated_cost_usd_total[1h]) * 3600
```

Average cost per request:

```promql
rate(orla_estimated_cost_usd_total[5m])
/
rate(orla_requests_total{status="success"}[5m])
```

## Alerting

Set up alerts on the metrics that matter for your deployment. Some starting points:

High error rate:

```yaml
# alertmanager rule
- alert: OrlaHighErrorRate
  expr: >
    sum(rate(orla_requests_total{status="error"}[5m])) by (backend)
    /
    sum(rate(orla_requests_total[5m])) by (backend)
    > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Orla backend {{ $labels.backend }} error rate above 5%"
```

Queue backing up:

```yaml
- alert: OrlaQueueBacklog
  expr: orla_queue_depth > 50
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "Orla backend {{ $labels.backend }} has {{ $value }} queued requests"
```

Spend exceeding budget:

```yaml
- alert: OrlaHighSpend
  expr: rate(orla_estimated_cost_usd_total[1h]) * 3600 > 10
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Orla backend {{ $labels.backend }} spending above $10/hour"
```

## Integrating with Grafana

Import the Prometheus data source in Grafana and create dashboards using the queries above. A useful layout:

- Top row: request rate and error rate per backend
- Middle row: P50/P95/P99 latency and queue wait time
- Bottom row: cost per hour, queue depth, accuracy routing breakdown

All metrics are labeled by `backend`, so you can filter or group by backend in any panel.
