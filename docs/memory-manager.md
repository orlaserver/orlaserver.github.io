# Using the Memory Manager

When an LLM processes a sequence of tokens, it builds a KV cache that speeds up generation for subsequent tokens. In a multi-stage workflow, consecutive stages often share context. If the cache is discarded between stages, the next stage pays the full prefill cost again. Orla's memory manager decides when to preserve the cache across stages and when to flush it, so later stages can pick up where earlier ones left off without redundant computation.

The memory manager operates automatically. If you do nothing, Orla uses a default policy that preserves the cache when two consecutive stages use the same backend and model with a small token delta, and flushes it at workflow boundaries or when the backend changes. You can override this behavior per stage or write your own policy.

## Default behavior

The default policy makes two decisions:

1. If the next stage uses the same backend and model as the previous stage, and the token delta between them is 256 tokens or fewer, preserve the cache.
2. If the workflow completes or the backend/model changes between stages, flush the cache.

This means a classify-then-reply workflow on the same backend keeps the cache warm between stages, while a workflow that routes different stages to different backends flushes cleanly at each transition.

## Explicit cache control per stage

You can override the default policy on any stage with `set_cache_policy`:

```python
from pyorla import Stage
from pyorla.types import CACHE_POLICY_PRESERVE, CACHE_POLICY_FLUSH

# Always preserve cache going into this stage, even if the default policy would flush
classify = Stage("classify", backend)
classify.set_cache_policy(CACHE_POLICY_PRESERVE)

# Always flush cache before this stage starts
reply = Stage("reply", backend)
reply.set_cache_policy(CACHE_POLICY_FLUSH)
```

Stage-level overrides take priority over the default policy. Use `CACHE_POLICY_PRESERVE` when you know two stages share context and want to guarantee the cache is kept. Use `CACHE_POLICY_FLUSH` when a stage starts a fresh context and stale cache would waste memory.

## Tuning the preserve threshold

The default threshold for preserving cache is 256 tokens. If the token delta between two consecutive stages on the same backend exceeds this, the default policy does not preserve. You can adjust this with `CacheHints`:

```python
from pyorla.types import CacheHints

stage = Stage("summarize", backend)
stage.set_cache_hints(CacheHints(preserve_threshold_tokens=512))
```

A higher threshold preserves the cache more aggressively, which helps when stages share large contexts. A lower threshold flushes more often, which frees memory for other workflows sharing the same backend.

## Workflow lifecycle

The memory manager tracks cache state per workflow. When you use the `Workflow` class, this happens automatically:

```python
from pyorla import OrlaClient, Stage, Workflow

client = OrlaClient("http://localhost:8081")

classify = Stage("classify", backend)
reply = Stage("reply", backend)

wf = Workflow(client)
wf.add_stage(classify)
wf.add_stage(reply)
results = wf.execute()
```

During `execute()`, Orla assigns a workflow ID to all stages, sends it with every inference request, and notifies the daemon when the workflow completes. On completion, the daemon flushes the cache for all backends that workflow used.

If you are using LangGraph instead of the Workflow class, the workflow completion notification is not sent automatically. You can send it manually after the graph finishes:

```python
client.workflow_complete(workflow_id, ["backend-name"])
```

## How cache decisions are made

When a request arrives at the daemon, the memory manager evaluates the transition in this order:

1. If the stage has an explicit cache policy set, use it. `"preserve"` always preserves, `"flush"` always flushes.
2. Otherwise, evaluate the policy chain. The preserve-on-small-increment policy checks whether the same backend and model are used and the token delta is within threshold. If so, preserve.
3. If the first policy does not act, the flush-at-boundary policy checks whether the workflow completed or the backend/model changed. If so, flush.
4. If neither policy acts, do nothing and let the backend's natural LRU eviction handle it.

Flushing has two levels. A soft flush marks the cache as stale in the daemon's tracker. A hard flush calls the backend's cache controller to actually evict the data. Hard flushes only happen when no other workflows are using that backend, to avoid disrupting in-flight requests.

## Memory pressure monitoring

The daemon runs a background monitor that polls each backend's memory usage every two seconds. When a backend's KV cache usage exceeds 85%, the monitor identifies the oldest idle workflow with preserved cache on that backend and flushes it. This prevents a single long-running workflow from monopolizing cache memory.

This runs automatically when the daemon starts. The pressure threshold is 85% by default.

## Writing your own memory policy

On the Python side, you can implement a custom policy by subclassing `MemoryPolicy`:

```python
from pyorla.memory import MemoryPolicy, CacheEvent
from pyorla.types import CACHE_POLICY_PRESERVE, CACHE_POLICY_FLUSH, CACHE_POLICY_AUTO

class AlwaysPreservePolicy(MemoryPolicy):
    def decide(self, event: CacheEvent) -> str:
        if event.transition_type == "workflow_complete":
            return CACHE_POLICY_FLUSH
        return CACHE_POLICY_PRESERVE
```

A policy receives a `CacheEvent` with information about the transition:

- `prev_stage_backend` and `next_stage_backend` tell you whether the backend changed.
- `delta_tokens` is the difference in token count between stages.
- `transition_type` is either `"stage"` or `"workflow_complete"`.

Return `CACHE_POLICY_PRESERVE` to keep the cache, `CACHE_POLICY_FLUSH` to evict it, or `CACHE_POLICY_AUTO` to defer to the next policy in the chain.

A policy that preserves cache only for stages on a specific backend:

```python
class PreserveOnGPUBackend(MemoryPolicy):
    def __init__(self, gpu_backend: str):
        self.gpu_backend = gpu_backend

    def decide(self, event: CacheEvent) -> str:
        if event.transition_type == "workflow_complete":
            return CACHE_POLICY_FLUSH
        if event.next_stage_backend == self.gpu_backend:
            return CACHE_POLICY_PRESERVE
        return CACHE_POLICY_AUTO
```

A policy that flushes when the token delta is large to free memory for other workflows:

```python
class FlushOnLargeDelta(MemoryPolicy):
    def __init__(self, max_delta: int = 1024):
        self.max_delta = max_delta

    def decide(self, event: CacheEvent) -> str:
        if event.delta_tokens > self.max_delta:
            return CACHE_POLICY_FLUSH
        return CACHE_POLICY_AUTO
```

## Backends that support cache control

Cache preservation works with any backend since it is a daemon-side decision about whether to include previous context in subsequent requests. Hard flush, where the daemon tells the backend to evict cached data, currently requires an SGLang backend. For other backends like vLLM and Ollama, soft flush marks the cache as stale in the tracker but does not send a flush command to the backend.
