# API Reference

The Orla daemon (`orla serve`) exposes an HTTP API at `http://localhost:8081` by default. All endpoints accept and return JSON unless noted otherwise.

The full [OpenAPI 3.0 spec](https://github.com/harvard-cns/orla/blob/main/docs/openapi.yaml) is available in the Orla repo.

## Health check

```
GET /api/v1/health
```

Returns `{"status": "healthy"}` when the daemon is ready.

## Execute inference

```
POST /api/v1/execute
```

Run a chat completion against a registered backend.

Request body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `backend` | string | yes | Name of a registered backend |
| `stage_id` | string | no | Stage identifier for queue grouping (default `"default"`) |
| `prompt` | string | no | User prompt (appended as a user message) |
| `messages` | array | no | Chat message history |
| `tools` | array | no | Tool definitions in MCP format |
| `stream` | boolean | no | Enable SSE streaming (default `false`) |
| `max_tokens` | integer | no | Maximum tokens to generate |
| `temperature` | number | no | Sampling temperature |
| `top_p` | number | no | Top-p sampling |
| `scheduling_policy` | string | no | Stage-level scheduling: `"fcfs"` or `"priority"` |
| `request_scheduling_policy` | string | no | Request-level scheduling: `"fcfs"` or `"priority"` |
| `scheduling_hints` | object | no | `{"priority": int, "request_priority": int}` |
| `workflow_id` | string | no | Groups requests by workflow for memory management |
| `cache_policy` | string | no | Cache override: `"preserve"` or `"flush"` |
| `accuracy` | number | no | Quality floor (0.0 to 1.0) for cost-optimized backend selection |
| `accuracy_policy` | string | no | Fallback behavior: `"prefer"` (default) or `"strict"` |

Response (non-streaming):

```json
{
  "success": true,
  "response": {
    "content": "The answer is 42.",
    "thinking": "",
    "tool_calls": [],
    "metrics": {
      "prompt_tokens": 24,
      "completion_tokens": 8,
      "queue_wait_ms": 3,
      "scheduler_decision_ms": 0,
      "dispatch_ms": 450,
      "backend_latency_ms": 447,
      "estimated_cost_usd": 0.000032
    }
  }
}
```

Streaming: set `stream: true` to receive Server-Sent Events with `content`, `thinking`, and `done` event types.

## Register a backend

```
POST /api/v1/backends
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Backend identifier (used in `execute` requests) |
| `endpoint` | string | yes | Base URL of the inference server |
| `type` | string | yes | `"openai"` or `"sglang"` |
| `model_id` | string | yes | Model identifier (e.g. `"openai:Qwen/Qwen3-4B"`) |
| `api_key_env_var` | string | no | Environment variable name for the API key |
| `max_concurrency` | integer | no | Max parallel requests to this backend (default `1`) |
| `queue_capacity` | integer | no | Max queued requests before rejecting (default `4096`) |
| `cost_model` | object | no | `{"input_cost_per_mtoken": float, "output_cost_per_mtoken": float}` |
| `quality` | number | no | Capability score (0.0 to 1.0) for accuracy-based routing |

## List backends

```
GET /api/v1/backends
```

Returns `{"backends": ["backend-1", "backend-2"]}`.

## Update a backend

```
PATCH /api/v1/backends/{name}
```

Live-update mutable fields on a registered backend. Only supplied fields are changed.

| Field | Type | Description |
|-------|------|-------------|
| `cost_model` | object | Updated pricing |
| `quality` | number | Updated quality score |
| `max_concurrency` | integer | Updated concurrency limit |

## Notify workflow complete

```
POST /api/v1/workflow/complete
```

Signals that a workflow has finished so the memory manager can flush cached state.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow_id` | string | yes | The workflow ID assigned during execution |
| `backends` | array | no | List of backend names the workflow used |

## Prometheus metrics

```
GET /metrics
```

Returns Prometheus-formatted metrics. See [Monitoring and Metrics](monitoring.md) for the full list of available metrics and example queries.
