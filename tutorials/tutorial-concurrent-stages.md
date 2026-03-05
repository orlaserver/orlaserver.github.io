# Tutorial: Concurrent Stages with Backend Concurrency

Orla allows multiple stages to dispatch inference requests to the same backend concurrently. This is useful when your backend supports continuous batching (vLLM, SGLang) and you want to maximize GPU utilization. By default, Orla dispatches one request at a time per backend. With `SetMaxConcurrency`, you can raise this limit so independent stages or parallel workflows are served simultaneously.

## What you need

- A running Orla server with at least one LLM backend (vLLM or SGLang recommended). See [Using Orla with vLLM](tutorials/tutorial-vllm-lily.md) for setup.
- Go 1.25 or later.
- The Orla Go client (`github.com/dorcha-inc/orla/pkg/api`).

## Run the example

From the Orla repo root, start Orla and vLLM:

```bash
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

Wait for the backend to load, then run the concurrent stages demo:

```bash
go run ./examples/concurrent_stages_demo/cmd/concurrent_stages_demo
```

The example registers a backend with `SetMaxConcurrency(4)` and runs two parallel stages (summarize and extract_entities) on the same backend. The source is in `examples/concurrent_stages_demo/`.

## Why concurrency matters

Consider a workflow where two agents fan out after triage and run in parallel on the same heavy backend. Without concurrency, the resolver and escalation requests queue behind each other on the heavy backend, even though the GPU could batch them together. With `SetMaxConcurrency(4)`, up to four requests can be in-flight on the heavy backend simultaneously, and the backend's internal scheduler handles GPU batching.

## 1. Register a backend with concurrency

When creating a backend, call `SetMaxConcurrency` before registering it with the Orla server:

```go
package main

import (
	"context"
	"fmt"
	"log"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

func main() {
	client := orla.NewOrlaClient("http://localhost:8081")
	ctx := context.Background()

	backend := orla.NewVLLMBackend("Qwen/Qwen3-8B", "http://vllm:8000/v1")
	backend.SetMaxConcurrency(4)

	if err := client.RegisterBackend(ctx, backend); err != nil {
		log.Fatal("register backend: ", err)
	}

	fmt.Println("Backend registered with max concurrency 4")
}
```

A value of `0` or `1` means serial dispatch (the default). Any value above `1` spawns that many worker goroutines inside the Orla server for this backend.

## 2. Run parallel stages in an Agent DAG

With concurrency enabled, independent stages in an Agent DAG naturally benefit. When two stages have no dependency between them, the Agent executor launches them concurrently, and both requests reach the backend workers in parallel:

```go
package main

import (
	"context"
	"fmt"
	"log"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

func main() {
	client := orla.NewOrlaClient("http://localhost:8081")
	ctx := context.Background()

	backend := orla.NewVLLMBackend("Qwen/Qwen3-8B", "http://vllm:8000/v1")
	backend.SetMaxConcurrency(4)
	if err := client.RegisterBackend(ctx, backend); err != nil {
		log.Fatal("register backend: ", err)
	}

	agent := orla.NewAgent(client)
	agent.Name = "parallel_demo"

	stageA := orla.NewStage("summarize", backend)
	stageA.SetMaxTokens(256)
	stageA.Prompt = "Summarize the key findings of this report: ..."

	stageB := orla.NewStage("extract_entities", backend)
	stageB.SetMaxTokens(256)
	stageB.Prompt = "Extract all named entities from this report: ..."

	if err := agent.AddStage(stageA); err != nil {
		log.Fatal(err)
	}
	if err := agent.AddStage(stageB); err != nil {
		log.Fatal(err)
	}
	// No dependency between stageA and stageB -- they run concurrently.

	results, err := agent.ExecuteDAG(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Summary:", results[stageA.ID].Response.Content)
	fmt.Println("Entities:", results[stageB.ID].Response.Content)
}
```

Without `SetMaxConcurrency(4)`, `summarize` would finish before `extract_entities` starts. With it, both requests are dispatched to the GPU at the same time.

## 3. Concurrent workflows

Concurrency also helps when multiple workflows target the same backend. For example, if you submit 10 SWE-bench instances concurrently and they all route to the heavy backend, a concurrency of `4` means four instances can be in-flight simultaneously instead of queuing one-by-one:

```go
backend := orla.NewVLLMBackend("Qwen/Qwen3-8B", "http://vllm:8000/v1")
backend.SetMaxConcurrency(4) // 4 concurrent requests to this backend
```

The Orla scheduler still applies its stage scheduling policy (FCFS or priority) to decide *which* requests are admitted from the queue. Concurrency controls how many are dispatched at once.

## 4. Choosing the right concurrency value

| Backend | Recommended starting value | Notes |
|---------|---------------------------|-------|
| vLLM | 4--8 | vLLM's continuous batching handles many concurrent requests well. Start with 4 and increase based on GPU memory. |
| SGLang | 4--8 | Similar to vLLM. RadixAttention benefits from concurrent requests sharing prefixes. |
| Ollama | 1 | From our understanding, Ollama serializes requests internally. Raising concurrency might add overhead without benefit. |

Monitor your backend's GPU memory usage (`nvidia-smi`) when tuning. More concurrent requests means more active KV cache entries competing for GPU memory. If you see out-of-memory errors, reduce concurrency or reduce `max_tokens`.

## 5. How concurrency interacts with scheduling

Concurrency and scheduling work together. With `SetMaxConcurrency(4)` and priority scheduling:

1. The Orla scheduler selects the 4 highest-priority requests from the stage queues.
2. All 4 are dispatched to backend workers simultaneously.
3. As each completes, the scheduler admits the next highest-priority request.

This means SJF (Shortest Job First) or priority scheduling still controls ordering, but now multiple requests run in parallel instead of strictly one-at-a-time.

## Next steps

- Combine concurrency with the [Memory Manager](tutorials/tutorial-memory-manager.md) to manage KV cache lifecycle across concurrent workflows.
- See the [Multi-Agent Workflow tutorial](research/orla_workflow_customer_support.md) for a full workflow example with fan-out across agents and backends.
