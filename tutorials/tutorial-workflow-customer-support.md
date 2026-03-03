# Tutorial: Multi-Agent Workflow with Orla

This tutorial runs a multi-agent workflow using [Orla](https://github.com/dorcha-inc/orla) and two [SGLang](https://sgl-project.github.io/) backends. The workflow processes a customer support ticket through a four-stage pipeline spanning two agents: a triage agent (light model) classifies and prioritizes the ticket, then a resolver agent (heavy model) drafts a reply and checks it for quality. This demonstrates Orla's full abstraction stack: Workflow, Agent, Stage DAG, Stage Mapping, Scheduling, and Context Passing.

## Architecture

<!-- ```
Workflow
  |
  +-- Agent "triage" (light model, FCFS scheduling)
  |     +-- Stage "classify"    -> structured JSON: category, sentiment, key issue
  |     +-- Stage "prioritize"  -> severity rating (depends on "classify")
  |
  +-- Agent "resolver" (heavy model, Priority scheduling; depends on "triage")
        +-- Stage "draft_response" -> personalized customer reply
        +-- Stage "qa_check"       -> policy compliance review (depends on "draft_response")
``` -->

This demo exercises every layer of the Orla abstraction stack. Within each agent, stages form a DAG: for example, `prioritize` cannot run until `classify` finishes, because it needs the classification output to decide severity. The workflow itself is also a DAG of agents: the `resolver` agent depends on the `triage` agent, so the heavy model never starts work until triage is complete.

Before execution, Stage Mapping validates the plan. Each stage is explicitly assigned to a backend (light or heavy), and `ExplicitStageMapping` checks that every stage points to a registered backend. This catches configuration errors before any inference happens.

On the server side, Orla uses two-level scheduling. The triage agent's stages run with FCFS (first-come, first-served), which is appropriate for lightweight classification work. The resolver agent's stages use Priority scheduling with scheduling hints, so when multiple tickets compete for the heavy backend, urgent ones are served first.

The `classify` stage produces structured JSON output via a schema, ensuring the downstream stages always receive machine-parseable triage data rather than free-form text. Finally, context passing connects the two agents: when the triage agent finishes, its output is automatically injected into the resolver agent's prompt, so the heavy model has full context about the ticket's category, sentiment, and severity before drafting a response.

## What you need

- Docker and Docker Compose (Compose V2).
- An NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- The [Orla repo](https://github.com/dorcha-inc/orla) cloned.
- Go 1.25 or later.

## 1. Start the backends and Orla

The workflow uses two SGLang backends: a **light** model (Qwen3-4B) for triage and a **heavy** model (Qwen3-8B) for resolution. Start the stack from the Orla repo root:

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml up -d sglang sglang-light orla
```

This starts the heavy SGLang backend on port 30000, the light SGLang backend on port 30001, and the Orla server on port 8081. Wait for the backends to finish loading their models (check logs with `docker compose -f deploy/docker-compose.swebench-lite.yaml logs -f sglang sglang-light`).

Verify Orla is healthy:

```bash
curl http://localhost:8081/api/v1/health
```

If you only have one GPU, you can run both backends on the same model by setting the environment variables in step 3.

## 2. Understand the workflow code

The workflow is defined in `examples/workflow_demo/workflow_demo.go`. Here is a walkthrough of the key parts.

### Backends

The demo registers two SGLang backends with Orla:

```go
lightBackend := orla.NewSGLangBackend("Qwen/Qwen3-4B-Instruct-2507", "http://sglang-light:30000/v1")
heavyBackend := orla.NewSGLangBackend("Qwen/Qwen3-8B", "http://sglang:30000/v1")
client.RegisterBackend(ctx, lightBackend)
client.RegisterBackend(ctx, heavyBackend)
```

### Agent 1: triage

The triage agent has two stages in a DAG. The `classify` stage uses structured output (JSON schema) to extract ticket metadata:

```go
triage := orla.NewAgent(client)
triage.Name = "triage"

classifyStage := orla.NewStage("classify", lightBackend)
classifyStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_triage", triageSchema))
classifyStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
classifyStage.Prompt = fmt.Sprintf("You are a customer support triage system. Classify this ticket...\n\n%s", ticket)

prioritizeStage := orla.NewStage("prioritize", lightBackend)
prioritizeStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    classification := results[classifyStage.ID]
    return fmt.Sprintf("Assign a severity...\n\n%s", classification.Response.Content), nil
})

triage.AddStage(classifyStage)
triage.AddStage(prioritizeStage)
triage.AddDependency(prioritizeStage.ID, classifyStage.ID)
```

The `prioritize` stage uses a **PromptBuilder** that reads the classify result at execution time to build its prompt dynamically.

### Agent 2: resolver

The resolver agent also has two stages. It uses Priority scheduling so urgent tickets are served first when multiple tickets compete for the heavy backend:

```go
resolver := orla.NewAgent(client)
resolver.Name = "resolver"

draftStage := orla.NewStage("draft_response", heavyBackend)
draftStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
priority := 5
draftStage.SetSchedulingHints(&orla.SchedulingHints{Priority: &priority})

qaStage := orla.NewStage("qa_check", heavyBackend)
qaStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    draft := results[draftStage.ID]
    return fmt.Sprintf("Check this draft for policy compliance...\n\n%s", draft.Response.Content), nil
})

resolver.AddStage(draftStage)
resolver.AddStage(qaStage)
resolver.AddDependency(qaStage.ID, draftStage.ID)
```

### Stage mapping

Before execution, `ExplicitStageMapping` validates that every stage is assigned to a registered backend:

```go
allStages := []*orla.Stage{classifyStage, prioritizeStage, draftStage, qaStage}
mapping := &orla.ExplicitStageMapping{}
output, _ := mapping.Map(&orla.StageMappingInput{
    Stages:   allStages,
    Backends: []*orla.LLMBackend{lightBackend, heavyBackend},
})
orla.ApplyStageMappingOutput(allStages, output)
```

### Workflow with context passing

The workflow defines the inter-agent dependency and a context passing function that feeds triage output into the resolver:

```go
wf := orla.NewWorkflow()
wf.AddAgent(triage)
wf.AddAgent(resolver)
wf.AddDependency("resolver", "triage")

wf.SetContextPassingFn(func(upstream map[string]*orla.AgentResult, downstream *orla.Agent) error {
    if downstream.Name != "resolver" {
        return nil
    }
    triageResult := upstream["triage"]
    // Build prompt for draft_response using triage output + original ticket
    // ...
    return nil
})

results, _ := wf.Execute(ctx)
```

## 3. Run the demo

From the Orla repo root, run the workflow demo:

```bash
go run ./examples/workflow_demo/cmd/workflow_demo
```

The demo includes a built-in sample ticket (a duplicate billing charge complaint). To use your own ticket, set the `TICKET_PATH` environment variable:

```bash
TICKET_PATH=/path/to/ticket.txt go run ./examples/workflow_demo/cmd/workflow_demo
```

You can also override the backend URLs and models with environment variables if your setup differs:

```bash
ORLA_URL=http://localhost:8081 \
LIGHT_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
HEAVY_MODEL=Qwen/Qwen3-8B \
SGLANG_LIGHT_URL=http://sglang-light:30000/v1 \
SGLANG_HEAVY_URL=http://sglang:30000/v1 \
go run ./examples/workflow_demo/cmd/workflow_demo
```

You should see output similar to:

```
2025/07/20 14:30:01 ================================================
2025/07/20 14:30:01 Running customer support workflow demo
2025/07/20 14:30:01 ================================================
2025/07/20 14:30:01 Stage mapping validated: 4 stages assigned to backends
2025/07/20 14:30:01 Executing customer support workflow...
2025/07/20 14:30:03 === Agent: triage ===
2025/07/20 14:30:03   Stage classify:
2025/07/20 14:30:03     {"category":"billing","product":"Pro subscription","customer_sentiment":"frustrated","key_issue":"Duplicate $49.99 charge on credit card for Pro subscription"}
2025/07/20 14:30:04   Stage prioritize:
2025/07/20 14:30:04     Severity: HIGH. Customer has been double-charged $49.99, causing immediate financial impact...
2025/07/20 14:30:07 === Agent: resolver ===
2025/07/20 14:30:07   Stage draft_response:
2025/07/20 14:30:07     Dear Alex, Thank you for reaching out and for being a loyal Pro subscriber for 2 years...
2025/07/20 14:30:09   Stage qa_check:
2025/07/20 14:30:09     APPROVED. The response addresses the billing issue with a clear refund commitment...
```

## 4. Stop the stack

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml down
```

## Control Flow

The workflow executor walks the agent dependency graph and runs agents in topological order. Because `resolver` depends on `triage`, the triage agent always runs first.

Inside the triage agent, stages also execute in DAG order. The `classify` stage fires first, sending the raw ticket to the light model and receiving structured JSON back (category, product, sentiment, key issue). Once that completes, the `prioritize` stage's `PromptBuilder` reads the classification result and constructs a new prompt asking the light model to assign a severity level. This two-step decomposition keeps each inference call focused on a single task, which improves output quality compared to asking the model to do everything at once.

When the triage agent finishes, the workflow executor calls the context passing function before launching the next agent. This function takes the triage agent's stage results and injects them into the resolver agent's `draft_response` stage prompt, so the heavy model sees the full triage analysis alongside the original ticket. Context passing is the mechanism that connects agents without coupling their internal stage logic.

The resolver agent then runs on the heavy backend. The `draft_response` stage generates a personalized customer reply informed by the triage context, and once that completes, the `qa_check` stage reviews the draft for policy compliance, professional tone, and completeness. If the QA check fails, the output tells you exactly what to fix, which could feed into a retry loop in a production system.

On the server side, each backend maintains per-stage request queues. The triage agent's stages use FCFS scheduling, which is sufficient for lightweight classification. The resolver agent's stages use Priority scheduling with hints, so when multiple tickets are competing for the heavy backend simultaneously, higher-priority tickets (e.g. critical billing issues) are served before lower-priority ones.

For more on Orla's scheduling and stage routing, see [Using Tools with Orla](tutorial-tools-vllm-ollama-sglang.md). For the SWE-bench experiment that uses Priority scheduling with score prediction, see [Orla SWE-bench Lite](../research/orla_swebench_lite.md).
