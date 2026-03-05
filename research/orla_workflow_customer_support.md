# Multi-Stage Workflow with Orla

This tutorial runs a multi-stage workflow using [Orla](https://github.com/dorcha-inc/orla) and two [SGLang](https://sgl-project.github.io/) backends. The workflow processes a customer support ticket through a four-stage pipeline: a classification stage (light model) extracts structured ticket metadata; a policy check stage (heavy model) reads company policy via tool call and renders an accept/deny decision; a reply stage (heavy model) composes and sends a customer email via tool call; and a routing stage (heavy model) acknowledges receipt, looks up internal teams, and creates an internal ticket -- all via tool calls. This demonstrates Orla's full abstraction stack: Workflow, Stage DAG, Stage Mapping, Scheduling, and tool-calling agent loops.

## Architecture

<img src="research/workflow_demo.svg" alt="Workflow DAG: classify, policy_check, reply, and route_ticket stages" style="max-width: 560px; height: auto; display: block; margin-left: auto; margin-right: auto;" />

<!-- ```
Workflow
  |
  +-- Stage "classify"       (light model, single-shot, FCFS) -> structured JSON: category, product, key issue, customer request
  |
  +-- Stage "policy_check"   (heavy model, agent-loop, Priority; depends on classify)
  |     tool: read_policy_yaml -> structured JSON: accept/deny + reasoning
  |
  +-- Stage "reply"          (heavy model, agent-loop, Priority; depends on policy_check)
  |     tool: send_email -> structured JSON: email_sent + summary
  |
  +-- Stage "route_ticket"   (heavy model, agent-loop, Priority; depends on classify)
        tools: send_email, read_team_descriptions, send_ticket -> free-text summary
``` -->

<!-- ```
classify ──┬──▶ policy_check ──▶ reply
           └──▶ route_ticket
``` -->

The demo exercises every layer of the Orla abstraction stack. Stage 1 (`classify`) is a single-shot structured-output call on the light model that extracts the ticket category, product, key issue, and what the customer is actually asking for. Stage 2 (`policy_check`) runs as an agent loop on the heavy model: it calls the `read_policy_yaml` tool to retrieve company policy for the ticket's category, then renders an accept/deny decision with reasoning as structured output. Stage 3 (`reply`) composes a professional customer email based on the policy decision and classification, then sends it via the `send_email` tool. Stage 4 (`route_ticket`) runs in parallel with Stages 2 and 3 (it only depends on `classify`): it sends an acknowledgment email, reads internal team descriptions, and creates a routed internal ticket -- executing three tool calls in sequence.

Before execution, Stage Mapping validates the plan. Each stage is explicitly assigned to a backend (light or heavy), and `ExplicitStageMapping` checks that every stage points to a registered backend. This catches configuration errors before any inference happens.

On the server side, Orla uses two-level scheduling. The classification stage runs with FCFS (first-come, first-served), appropriate for a lightweight single-shot call. The remaining stages use Priority scheduling with hints set dynamically based on the classification output, so when multiple tickets compete for the heavy backend, urgent ones are served first.

Two of the four stages produce structured JSON output via schemas (`classify` and `policy_check`), ensuring downstream stages always receive machine-parseable data. The agent-loop stages (`policy_check`, `reply`, `route_ticket`) demonstrate Orla's tool-calling support: each stage is given tools, and the agent-loop executor handles the Gen → Tool → Gen cycle automatically. Context between stages is passed via `PromptBuilder` functions: each downstream stage reads upstream results from the `map[string]*StageResult` and constructs its prompt accordingly.

## What you need

- Docker and Docker Compose (Compose V2).
- An NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- The [Orla repo](https://github.com/dorcha-inc/orla) cloned.
- Go 1.25 or later.

## 1. Start the backends and Orla

The workflow uses two SGLang backends: a **light** model (Qwen3-4B) for classification, and a **heavy** model (Qwen3-8B) for policy checking, reply composition, and ticket routing. Use the **workflow-demo** compose file so you get a clean stack and avoid network conflicts with other compose projects (e.g. SWE-bench Lite). You can run with **SGLang** (default) or **vLLM**. From the Orla repo root:

**SGLang** (default):

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
```

**vLLM** (two vLLM containers: heavy on 8000, light on 8001):

```bash
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml up -d
```

This starts the heavy SGLang backend on port 30000, the light SGLang backend on port 30001, and the Orla server on port 8081. Wait for the backends to finish loading their models (check logs with `docker compose -f deploy/docker-compose.workflow-demo.yaml logs -f sglang sglang-light`).

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

### Stage 1: classify (single-shot, structured output)

The classification stage extracts structured metadata from the raw ticket. It runs as a single-shot call on the light model with a JSON schema enforcing the output format:

```go
wf := orla.NewWorkflow(client)

classifyStage := orla.NewStage("classify", lightBackend)
classifyStage.SetMaxTokens(512)
classifyStage.SetTemperature(0)
classifyStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
classifyStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_classify", classifySchema))
classifyStage.Prompt = fmt.Sprintf(
    "You are a customer support triage system. Classify this support ticket.\n"+
        "Extract the category, product, a one-sentence summary of the core issue, "+
        "and what the customer is actually asking for.\n\nTicket:\n%s", ticket)
```

### Stage 2: policy_check (agent-loop with tool)

The policy check stage runs as an agent loop. It has access to a `read_policy_yaml` tool that retrieves company policy for a given category. The model calls the tool, reads the policy, then renders a structured accept/deny decision:

```go
policyTool, _ := readPolicyYAMLTool()

policyStage := orla.NewStage("policy_check", heavyBackend)
policyStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
policyStage.SetMaxTurns(5)
policyStage.SetResponseFormat(orla.NewStructuredOutputRequest("policy_decision", policyDecisionSchema))
policyStage.AddTool(policyTool)
policyStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    classification := results[classifyStage.ID]
    return fmt.Sprintf(
        "Use the read_policy_yaml tool to retrieve the policy for the ticket's category. "+
            "Then decide whether to ACCEPT or DENY the customer's request.\n\n"+
            "Ticket Classification:\n%s\n\nOriginal Ticket:\n%s",
        classification.Response.Content, ticket), nil
})

wf.AddStage(policyStage)
wf.AddDependency(policyStage.ID, classifyStage.ID)
```

The agent-loop executor handles the Gen → Tool(read_policy_yaml) → Gen → Structured Output cycle automatically.

### Stage 3: reply (agent-loop with tool)

The reply stage composes a customer email based on the policy decision and sends it using the `send_email` tool. It depends on `policy_check`:

```go
emailTool, _ := sendEmailTool()

replyStage := orla.NewStage("reply", heavyBackend)
replyStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
replyStage.SetMaxTurns(5)
replyStage.SetResponseFormat(orla.NewStructuredOutputRequest("reply_confirmation", replyConfirmationSchema))
replyStage.AddTool(emailTool)
replyStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    classification := results[classifyStage.ID]
    policyResult := results[policyStage.ID]
    return fmt.Sprintf(
        "Compose a professional reply based on the policy decision and send it "+
            "using the send_email tool.\n\n"+
            "Policy Decision:\n%s\n\nTicket Classification:\n%s\n\nOriginal Ticket:\n%s",
        policyResult.Response.Content, classification.Response.Content, ticket), nil
})

wf.AddStage(replyStage)
wf.AddDependency(replyStage.ID, policyStage.ID)
```

### Stage 4: route_ticket (agent-loop with multiple tools)

The routing stage runs in parallel with Stages 2 and 3 because it only depends on `classify`. It has three tools: `send_email` (acknowledgment), `read_team_descriptions` (team lookup), and `send_ticket` (internal routing). The agent loop executes all three tool calls in sequence:

```go
routeStage := orla.NewStage("route_ticket", heavyBackend)
routeStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
routeStage.SetMaxTurns(10)
routeStage.AddTool(emailToolRoute)
routeStage.AddTool(teamsTool)
routeStage.AddTool(ticketTool)
routeStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    classification := results[classifyStage.ID]
    return fmt.Sprintf(
        "1. Send an acknowledgment email to the customer.\n"+
            "2. Read the team descriptions to find the right team.\n"+
            "3. Create an internal support ticket for that team.\n\n"+
            "Ticket Classification:\n%s\n\nOriginal Ticket:\n%s",
        classification.Response.Content, ticket), nil
})

wf.AddStage(routeStage)
wf.AddDependency(routeStage.ID, classifyStage.ID)
```

### Stage mapping

Before execution, `ExplicitStageMapping` validates that every stage is assigned to a registered backend:

```go
allStages := []*orla.Stage{classifyStage, policyStage, replyStage, routeStage}
mapping := &orla.ExplicitStageMapping{}
output, _ := mapping.Map(&orla.StageMappingInput{
    Stages:   allStages,
    Backends: []*orla.LLMBackend{lightBackend, heavyBackend},
})
orla.ApplyStageMappingOutput(allStages, output)
```

### Execute the workflow

With all stages and dependencies added, execute the workflow:

```go
results, _ := wf.Execute(ctx)
```

The DAG executor starts `classify` first (no dependencies). Once it completes, both `policy_check` and `route_ticket` become unblocked and run in parallel. `policy_check` calls its tool, renders a decision, and when it finishes, `reply` becomes unblocked and runs. Meanwhile, `route_ticket` independently executes its three-tool sequence. Context between stages is handled by `PromptBuilder` functions on each stage.

## 3. Run the demo

From the Orla repo root, run the workflow demo:

```bash
go run ./examples/workflow_demo/cmd/workflow_demo
```

The demo includes a built-in sample ticket (a duplicate billing charge complaint). To use your own ticket, set the `TICKET_PATH` environment variable:

```bash
TICKET_PATH=/path/to/ticket.txt go run ./examples/workflow_demo/cmd/workflow_demo
```

You can override the backend URLs and models with environment variables. When using the **vLLM** stack, set `VLLM_LIGHT_URL` and `VLLM_HEAVY_URL` (these must be URLs the Orla container can resolve, e.g. `http://vllm-light:8000/v1` and `http://vllm-heavy:8000/v1` when Orla runs in the same compose):

```bash
# SGLang (default)
ORLA_URL=http://localhost:8081 \
SGLANG_LIGHT_URL=http://sglang-light:30000/v1 \
SGLANG_HEAVY_URL=http://sglang:30000/v1 \
go run ./examples/workflow_demo/cmd/workflow_demo

# vLLM (when using docker-compose.workflow-demo.vllm.yaml)
ORLA_URL=http://localhost:8081 \
VLLM_LIGHT_URL=http://vllm-light:8000/v1 \
VLLM_HEAVY_URL=http://vllm-heavy:8000/v1 \
go run ./examples/workflow_demo/cmd/workflow_demo
```

## 4. Stop the stack

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml down
# or, if you used vLLM:
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml down
```

If you see a "network not found" or permission error when starting containers, try bringing the stack down and removing orphan containers first: `docker compose -f deploy/docker-compose.workflow-demo.yaml down --remove-orphans`, then run `up -d` again. On Linux you may need `sudo` for Docker commands.

## Control Flow

The workflow executor walks the stage dependency graph and runs stages in topological order. `classify` is the only root with no dependencies, so the DAG executor fires it first on the light backend. It sends the raw ticket and receives structured JSON back (category, product, key issue, and customer request).

Once `classify` completes, two stages become unblocked: `policy_check` and `route_ticket`. Both run in parallel on the heavy backend. `policy_check` enters an agent loop: the model generates a request to call `read_policy_yaml` with the ticket's category, the tool returns the relevant policy document, and the model then generates a structured accept/deny decision with reasoning. Meanwhile, `route_ticket` enters its own agent loop with three tools: it first calls `send_email` to acknowledge receipt to the customer, then calls `read_team_descriptions` to discover available internal teams, and finally calls `send_ticket` to create an internal ticket routed to the appropriate team. The agent-loop executor manages the Gen → Tool → Gen cycle for each stage automatically.

Once `policy_check` finishes, `reply` becomes unblocked. It enters an agent loop, composes a professional customer reply based on the policy decision and ticket classification, then calls `send_email` to send it. The `reply` stage produces a structured confirmation indicating whether the email was sent and a brief summary.

Scheduling hints are set dynamically in the `PromptBuilder` for the `reply` stage based on the classification output: billing and technical tickets get higher priority (8) than other categories (5). Because the agent-loop stages (`policy_check`, `reply`, `route_ticket`) all use Priority scheduling on the heavy backend, Orla's scheduler can prioritize urgent tickets when multiple workflows compete for the same backend simultaneously.
