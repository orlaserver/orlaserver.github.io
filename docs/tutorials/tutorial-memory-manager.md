# Tutorial: Managing KV Cache with the Memory Manager

LLM backends like vLLM and SGLang maintain a KV cache that stores intermediate attention state. In multi-stage workflows, stages often share context. A triage stage's output feeds into a resolver stage, which feeds into a reviewer. If the KV cache from earlier stages is evicted prematurely, the backend must recompute the prefix from scratch, wasting GPU cycles. Conversely, holding onto cache from a completed workflow wastes GPU memory that new requests need.

Orla's Memory Manager solves this by making workflow-aware decisions about when to preserve or flush KV cache at stage and workflow boundaries. It implements three policies by default:

- **Preserve on Small Increment**: keep the cache when the new stage adds only a small number of tokens to the existing context.
- **Flush at Boundary**: release cache when a workflow completes or the backend/model changes between stages.
- **Flush Under Pressure**: automatically evict idle workflow caches when GPU memory is tight (handled server-side).

## What you need

- A running Orla server with at least one LLM backend. See [Using Orla with vLLM](tutorials/tutorial-vllm-lily.md) for setup.
- Go 1.25 or later.
- The Orla Go client (`github.com/harvard-cns/orla/pkg/api`).

## Run the example

The example uses **SGLang** by default so you can verify the hard cache flush end-to-end. From the Orla repo root, start the workflow-demo stack (Orla + SGLang):

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
```

Wait for SGLang to load its model (check logs with `docker compose -f deploy/docker-compose.workflow-demo.yaml logs -f sglang`), then run:

```bash
go run ./examples/memory_manager_demo/cmd/memory_manager_demo
```

The example uses `NewFlushAtBoundaryPolicy()`, which flushes the KV cache when the workflow completes. With SGLang, that triggers a hard flush via `POST /flush_cache`, clearing the RadixAttention cache. The source is in `examples/memory_manager_demo/`. To use vLLM instead (soft flush only), set `VLLM_URL=http://vllm:8000/v1`.

## 1. Zero-config: the default behavior

If you create a Workflow without setting a memory policy, Orla uses `NewDefaultMemoryPolicy()` automatically. This composes the preserve-on-small-increment and flush-at-boundary policies with sensible defaults (preserve threshold of 256 tokens). For most workloads this is all you need:

```go
wf := orla.NewWorkflow(client)
// No SetMemoryPolicy call -- the default policy applies.
wf.AddStage(classifyStage)
wf.AddStage(prioritizeStage)
wf.AddStage(draftStage)
wf.AddDependency(prioritizeStage.ID, classifyStage.ID)
wf.AddDependency(draftStage.ID, prioritizeStage.ID)

results, err := wf.Execute(ctx)
```

Under the hood, the Memory Manager:
- Preserves cache between stages on the same backend when the context grows by fewer than 256 tokens (avoiding recomputation of the shared prefix).
- Flushes cache when a workflow completes, freeing GPU memory for the next workflow.
- Flushes cache when a stage switches to a different backend or model (the old backend's cache is no longer useful).

## 2. Tuning the workflow-level policy

You can adjust the preserve threshold to match your workload. A higher threshold means the Memory Manager preserves cache even when stages add more tokens, trading GPU memory for compute savings:

```go
wf := orla.NewWorkflow(client)
wf.SetMemoryPolicy(orla.NewDefaultMemoryPolicy(
	orla.WithPreserveThreshold(512), // preserve up to 512 new tokens
))
```

When to raise the threshold:
- Long multi-turn conversations where each turn appends several hundred tokens.
- Pipelines where the resolver adds significant analysis text that the reviewer needs in its prefix.

When to lower it (or leave the default):
- Many short, independent workflows competing for the same GPU.
- Backends with limited GPU memory where you want aggressive reclamation.

## 3. Stage-level overrides

For fine-grained control, you can override the policy on individual stages. Stage-level overrides take precedence over the workflow-level policy:

```go
// Force this stage to preserve cache, regardless of the workflow policy.
reviewStage := orla.NewStage("final_review", heavyBackend)
reviewStage.SetCachePolicy(orla.CachePolicyPreserve)

// Force this stage to flush cache when it completes.
cleanupStage := orla.NewStage("cleanup", heavyBackend)
cleanupStage.SetCachePolicy(orla.CachePolicyFlush)
```

The three policy constants are:

| Constant | Behavior |
|----------|----------|
| `orla.CachePolicyPreserve` | Always preserve cache for this stage. |
| `orla.CachePolicyFlush` | Flush cache when this stage completes. |
| `orla.CachePolicyAuto` | Defer to the workflow-level policy (default). |

You can also pass hints that influence the preserve-on-small-increment threshold for a specific stage:

```go
stage.SetCacheHints(&orla.CacheHints{
	PreserveThresholdTokens: intPtr(1024), // override threshold for this stage only
})
```

## 4. Built-in policies

Orla ships three built-in policies that you can use individually or compose:

### PreserveOnSmallIncrement

Preserves cache when the context delta is below a token threshold and the backend/model hasn't changed between stages:

```go
policy := orla.NewPreserveOnSmallIncrementPolicy(256)
wf.SetMemoryPolicy(policy)
```

### FlushAtBoundary

Flushes cache at workflow completion and when the backend or model switches between stages:

```go
policy := orla.NewFlushAtBoundaryPolicy()
wf.SetMemoryPolicy(policy)
```

### DefaultMemoryPolicy (composed)

Chains preserve-on-small-increment and flush-at-boundary in order. The first policy to return a non-auto decision wins:

```go
policy := orla.NewDefaultMemoryPolicy(
	orla.WithPreserveThreshold(256), // optional, 256 is the default
)
wf.SetMemoryPolicy(policy)
```

This is equivalent to manually composing the two policies, and is what Orla uses when no policy is set.

## 5. Writing a custom policy

The `MemoryPolicy` interface has a single method:

```go
type MemoryPolicy interface {
	Decide(ctx context.Context, event CacheEvent) string
}
```

`CacheEvent` provides context about the stage transition:

```go
type CacheEvent struct {
	PrevStageBackend string
	PrevStageModel   string
	NextStageBackend string
	NextStageModel   string
	DeltaTokens      int
	TotalTokens      int
	TransitionType   string // "stage", "workflow_complete"
}
```

Return `orla.CachePolicyPreserve`, `orla.CachePolicyFlush`, or `orla.CachePolicyAuto` (defer to the next policy in the chain).

Here is an example that always preserves cache for a specific backend:

```go
type alwaysPreserveForBackend struct {
	targetBackend string
}

func (p *alwaysPreserveForBackend) Decide(_ context.Context, event orla.CacheEvent) string {
	if event.NextStageBackend == p.targetBackend {
		return orla.CachePolicyPreserve
	}
	return orla.CachePolicyAuto
}

// Usage:
wf.SetMemoryPolicy(&alwaysPreserveForBackend{targetBackend: "heavy-vllm"})
```

## 6. How it works under the hood

The server-side Memory Manager (`internal/serving/memory`) handles the actual cache lifecycle:

1. **Workflow tracking**: the client generates a unique workflow ID when `Execute()` is called. The server automatically registers the workflow the first time it sees a request with that ID.
2. **Stage lifecycle signals**: the scheduler emits `StageStart` and `StageComplete` signals to the Memory Manager for every request that carries a workflow ID. These signals feed the policy chain so it can make preserve/flush decisions.
3. **In-flight awareness**: the scheduler records each in-flight request. Flush decisions are deferred if other requests for the same workflow are still running on the backend.
4. **Workflow completion**: when `Workflow.Execute()` finishes on the client, it sends a `POST /api/v1/workflow/complete` notification to the server with the workflow ID and backends used. The server emits `TransitionWorkflowComplete` signals, which trigger the flush-at-boundary policy, then deregisters the workflow from tracking.
5. **Cache flush execution**: when the Memory Manager decides to flush, the behavior depends on the backend type:
   - **SGLang**: Orla automatically registers an SGLang cache controller when you add an SGLang backend. The controller calls SGLang's `POST /flush_cache` endpoint for a hard flush, which clears the RadixAttention cache. This is a global operation (SGLang does not support per-session eviction), so it is only executed when no other workflows are in-flight on the same backend. With the default `SetMaxConcurrency(1)`, this is safe because only one request runs at a time. With higher concurrency, the hard flush is deferred until the backend is idle, falling back to soft-flush behavior in the meantime.
   - **vLLM / Ollama**: These backends do not expose cache eviction APIs. The Memory Manager performs a soft flush: it marks the session as stale and stops actively preserving its cache, relying on the backend's natural LRU eviction to reclaim memory.
6. **Pressure monitor**: a background goroutine starts when the server starts and periodically queries SGLang backends for KV cache utilization via `GET /get_server_info`. When pressure exceeds the configured threshold (default 85%), it identifies idle workflow caches and marks them for eviction, oldest first. For backends without a memory stats API, the pressure monitor is a no-op.

Stage-level `CachePolicy` overrides (set via `SetCachePolicy`) are sent with each request and take precedence over the server-side policy chain. The server-side `DefaultManager` evaluates the policy chain and enforces decisions while accounting for global state like in-flight requests and memory pressure.

## Full example: customer support workflow with memory management

Putting it all together with a multi-stage workflow:

```go
package main

import (
	"context"
	"fmt"
	"log"

	orla "github.com/harvard-cns/orla/pkg/api"
)

func main() {
	client := orla.NewOrlaClient("http://localhost:8081")
	ctx := context.Background()

	heavy := orla.NewVLLMBackend("Qwen/Qwen3-8B", "http://vllm:8000/v1")
	heavy.SetMaxConcurrency(4)
	if err := client.RegisterBackend(ctx, heavy); err != nil {
		log.Fatal(err)
	}

	wf := orla.NewWorkflow(client)
	wf.SetMemoryPolicy(orla.NewDefaultMemoryPolicy(
		orla.WithPreserveThreshold(512),
	))

	classify := orla.NewStage("classify", heavy)
	classify.SetMaxTokens(128)
	classify.Prompt = "Classify this ticket: ..."

	prioritize := orla.NewStage("prioritize", heavy)
	prioritize.SetMaxTokens(128)
	prioritize.SetCachePolicy(orla.CachePolicyPreserve) // keep cache for draft
	prioritize.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		cr := results[classify.ID]
		if cr == nil || cr.Response == nil {
			return "", fmt.Errorf("missing classify result")
		}
		return fmt.Sprintf("Prioritize based on: %s", cr.Response.Content), nil
	})

	draft := orla.NewStage("draft", heavy)
	draft.SetMaxTokens(512)
	draft.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		pr := results[prioritize.ID]
		if pr == nil || pr.Response == nil {
			return "", fmt.Errorf("missing prioritize result")
		}
		return fmt.Sprintf("Draft a customer response based on: %s", pr.Response.Content), nil
	})

	wf.AddStage(classify)
	wf.AddStage(prioritize)
	wf.AddStage(draft)
	wf.AddDependency(prioritize.ID, classify.ID)
	wf.AddDependency(draft.ID, prioritize.ID)

	results, err := wf.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Draft:", results[draft.ID].Response.Content)
}
```

The Memory Manager preserves cache between `classify` and `prioritize` (small increment on the same backend), honors the explicit `CachePolicyPreserve` on `prioritize` so the draft benefits from the cached prefix, and flushes everything when `Execute()` returns and the workflow-complete notification reaches the server.

## Next steps

- See [Concurrent Stages](tutorials/tutorial-concurrent-stages.md) for configuring backend concurrency alongside the Memory Manager.
- See the [Workflow tutorial](research/orla_workflow_customer_support.md) for the full seven-stage customer support pipeline.
