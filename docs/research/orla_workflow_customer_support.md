# Multi-Stage Workflow with Orla

This tutorial runs a multi-stage workflow using [Orla](https://github.com/harvard-cns/orla) and two [SGLang](https://sgl-project.github.io/) backends. The workflow processes a customer support ticket through a four-stage pipeline: a classification stage (light model) extracts structured ticket metadata and decides whether the ticket needs human escalation; a policy check stage (heavy model) reads company policy via tool call and renders an accept/deny decision; a reply stage (heavy model) either sends a brief escalation acknowledgment or a full resolution email via tool call; and a routing stage (heavy model) conditionally either routes an escalated ticket to a human team or notifies the team that the ticket was resolved automatically. The policy check → reply chain and the route_ticket stage run in parallel after classification. This demonstrates Orla's full abstraction stack: Workflow, Stage DAG, Stage Mapping, Scheduling, and tool-calling agent loops.

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

  <!--
classify ──┬──▶ policy_check ──▶ reply
           └──▶ route_ticket
-->

The demo exercises every layer of the Orla abstraction stack. Stage 1 (`classify`) is a single-shot structured-output call on the light model that extracts the ticket category, product, key issue, customer request, and whether the ticket needs human escalation (`needs_escalation`). Stage 2 (`policy_check`) runs as an agent loop on the heavy model: it calls the `read_policy_yaml` tool to retrieve company policy for the ticket's category, then renders an accept/deny decision with reasoning as structured output. Stage 3 (`reply`) also branches on `needs_escalation`: if the ticket is escalated, it sends a brief acknowledgment to the customer; otherwise it composes a full resolution email based on the policy decision. Stage 4 (`route_ticket`) runs in parallel with Stages 2 and 3 (it only depends on `classify`): if the classifier decided the ticket needs escalation, it reads team descriptions, creates a routed internal ticket, and emails the responsible team; if not, it notifies the team that the ticket is being resolved automatically.

Before execution, Stage Mapping validates the plan. Each stage is explicitly assigned to a backend (light or heavy), and `ExplicitStageMapping` checks that every stage points to a registered backend. This catches configuration errors before any inference happens.

On the server side, Orla uses two-level scheduling. The classification stage runs with FCFS (first-come, first-served), appropriate for a lightweight single-shot call. The remaining stages use Priority scheduling with hints set dynamically based on the classification output, so when multiple tickets compete for the heavy backend, urgent ones are served first.

Two of the four stages produce structured JSON output via schemas (`classify` and `policy_check`), ensuring downstream stages always receive machine-parseable data. The agent-loop stages (`policy_check`, `reply`, `route_ticket`) demonstrate Orla's tool-calling support: each stage is given tools, and the agent-loop executor handles the Gen → Tool → Gen cycle automatically. Context between stages is passed via `PromptBuilder` functions: each downstream stage reads upstream results from the `map[string]*StageResult` and constructs its prompt accordingly.

## Choose Your Setup

<div style="text-align:center; margin: 1.5em 0;">
  <img src="assets/choose-adventure.png" alt="Choose Your Own Adventure" style="max-width: 320px; height: auto; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.12);" />
</div>

Pick the path that matches your hardware. Each one walks you through the full workflow demo end to end. We recommend the first path if you have a well-provisioned NVIDIA GPU as it demonstrates Orla running on top of SGLang and vLLM backends. The second path runs Orla on top of Ollama. The third path lets you try Orla as a standalone agent but, due to infrastructure requirements, cannot run the full customer support workflow.

<div class="adventure-cards" data-adventure-group="hw">
  <div class="adventure-card" data-adventure="gpu" data-adventure-group="hw">
    <span class="card-icon">🖥️</span>
    <div class="card-title">NVIDIA GPU</div>
    <div class="card-desc">Server or workstation with an NVIDIA GPU and at least 98 GB of GPU Memory.</div>
  </div>
  <div class="adventure-card" data-adventure="laptop" data-adventure-group="hw">
    <span class="card-icon">💻</span>
    <div class="card-title">Mac / Linux</div>
    <div class="card-desc">Laptop or server with at least 16 GB of main memory.</div>
  </div>
  <div class="adventure-card" data-adventure="standalone" data-adventure-group="hw">
    <span class="card-icon">🤷</span>
    <div class="card-title">Any Mac or Linux Computer</div>
    <div class="card-desc">Try Orla as a standalone agent.</div>
  </div>
</div>

<!-- GPU sub-selection -->
<div data-adventure-section="gpu-sub" data-adventure-section-group="hw" class="adventure-hidden">

**Which LLM backend do you prefer?**

<div class="adventure-cards" data-adventure-group="gpu-backend">
  <div class="adventure-card" data-adventure="sglang" data-adventure-group="gpu-backend">
    <span class="card-icon">🎲</span>
    <div class="card-title">SGLang</div>
    <!-- <div class="card-desc">RadixAttention, continuous batching, fast structured output.</div> -->
  </div>
  <div class="adventure-card" data-adventure="vllm" data-adventure-group="gpu-backend">
    <span class="card-icon">🪙</span>
    <div class="card-title">vLLM</div>
    <!-- <div class="card-desc">PagedAttention, broad model support, tensor parallelism.</div> -->
  </div>
</div>
</div>

<!-- ── SGLang path ─────────────────────────────────── -->
<div data-adventure-section="sglang" data-adventure-section-group="gpu-backend" class="adventure-hidden">

### SGLang Path

<span class="adventure-reset">← Start over</span>

**Prerequisites:**

- Docker and Docker Compose (Compose V2).
- An NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- The [Orla repo](https://github.com/harvard-cns/orla) cloned.
- Go 1.25 or later.

**1. Start the backends and Orla**

The workflow uses two SGLang backends: a **light** model (Qwen3-4B) for classification and a **heavy** model (Qwen3-8B) for the agent-loop stages. From the Orla repo root:

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
```

Wait for models to load (check logs):

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml logs -f sglang sglang-light
```

Verify Orla is healthy:

```bash
curl http://localhost:8081/api/v1/health
```

If you only have one GPU, you can run both backends on the same model by setting the environment variables in the run step.

**2. Run the demo**

```bash
go run ./examples/workflow_demo/cmd/workflow_demo
```

SGLang is the default backend — no `BACKEND` env var needed.

**3. Stop the stack**

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml down
```

</div>

<!-- ── vLLM path ───────────────────────────────────── -->
<div data-adventure-section="vllm" data-adventure-section-group="gpu-backend" class="adventure-hidden">

### vLLM Path

<span class="adventure-reset">← Start over</span>

**Prerequisites:**

- Docker and Docker Compose (Compose V2).
- An NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- The [Orla repo](https://github.com/harvard-cns/orla) cloned.
- Go 1.25 or later.

**1. Start the backends and Orla**

Two vLLM containers: heavy on port 8000, light on port 8001. From the Orla repo root:

```bash
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml up -d
```

Wait for models to load, then verify Orla is healthy:

```bash
curl http://localhost:8081/api/v1/health
```

**2. Run the demo**

```bash
BACKEND=vllm go run ./examples/workflow_demo/cmd/workflow_demo
```

**3. Stop the stack**

```bash
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml down
```

</div>

<!-- ── Ollama path ──────────────────────────────────── -->
<div data-adventure-section="laptop" data-adventure-section-group="hw" class="adventure-hidden">

### Ollama Path

<span class="adventure-reset">← Start over</span>

**Prerequisites:**

- Docker and Docker Compose (Compose V2).
- At least 16 GB of main memory.
- The [Orla repo](https://github.com/harvard-cns/orla) cloned.
- Go 1.25 or later.

**1. Start the backends and Orla**

From the Orla repo root:

```bash
docker compose -f deploy/docker-compose.workflow-demo.ollama.yaml up -d
```

The first run downloads `qwen3:0.6b` and `qwen3:1.7b` into a Docker volume, so it may take a few minutes. Subsequent starts reuse the cached models.

Verify Orla is healthy:

```bash
curl http://localhost:8081/api/v1/health
```

Both models are served by a single Ollama process; the light model (qwen3:0.6b) handles classification and the heavy model (qwen3:1.7b) handles the agent-loop stages. Output quality will be lower than the GPU models, but the full workflow — structured output, tool calls, DAG execution — works the same way.

**2. Run the demo**

```bash
BACKEND=ollama go run ./examples/workflow_demo/cmd/workflow_demo
```

**3. Stop the stack**

```bash
docker compose -f deploy/docker-compose.workflow-demo.ollama.yaml down
```

</div>

<!-- ── Standalone agent path ────────────────────────── -->
<div data-adventure-section="standalone" data-adventure-section-group="hw" class="adventure-hidden">

### Standalone Agent (Homebrew)

<span class="adventure-reset">← Start over</span>

You don't need Docker or a GPU for this path. Install Orla as a standalone agent via Homebrew and start experimenting right away.

Head over to the **[Getting Started guide](https://orlaserver.github.io/docs/#/getting-started)** for installation instructions.

</div>

---

## Understand the workflow code

The workflow is defined in `examples/workflow_demo/workflow_demo.go`. Here is a walkthrough of the key parts.

### Backends

The demo selects backends based on environment variables. The default is SGLang; set `OLLAMA_URL` for Ollama or `VLLM_LIGHT_URL`+`VLLM_HEAVY_URL` for vLLM:

```go
// SGLang (default)
lightBackend := orla.NewSGLangBackend("Qwen/Qwen3-4B-Instruct-2507", "http://sglang-light:30000/v1")
heavyBackend := orla.NewSGLangBackend("Qwen/Qwen3-8B", "http://sglang:30000/v1")

// Ollama (laptop — set OLLAMA_URL)
lightBackend := orla.NewOllamaBackend("qwen3:0.6b", "http://localhost:11434")
heavyBackend := orla.NewOllamaBackend("qwen3:1.7b", "http://localhost:11434")

client.RegisterBackend(ctx, lightBackend)
client.RegisterBackend(ctx, heavyBackend)
```

### Stage 1: classify (single-shot, structured output)

The classification stage extracts structured metadata from the raw ticket and decides whether the ticket needs human escalation. It runs as a single-shot call on the light model with a JSON schema enforcing the output format (fields: `category`, `product`, `key_issue`, `customer_request`, `needs_escalation`, `escalation_reason`):

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
        "and what the customer is actually asking for.\n"+
        "Also decide if this ticket needs human escalation.\n\nTicket:\n%s", ticket)
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

### Stage 3: reply (agent-loop with tool, conditional on escalation)

The reply stage depends on `policy_check` and branches on the `needs_escalation` flag from the classify output:

- **Needs escalation:** sends a brief acknowledgment email telling the customer their request is being escalated to a specialist team. It does not resolve the issue or make promises about the outcome.
- **No escalation:** composes a full resolution email based on the policy decision (confirming the action or explaining a denial) and sends it via `send_email`.

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
    var classifyData struct {
        NeedsEscalation bool `json:"needs_escalation"`
    }
    json.Unmarshal([]byte(classification.Response.Content), &classifyData)

    if classifyData.NeedsEscalation {
        return fmt.Sprintf("... escalation acknowledgment prompt ..."), nil
    }
    return fmt.Sprintf(
        "Compose a professional reply based on the policy decision and send it "+
            "using the send_email tool.\n\n"+
            "Policy Decision:\n%s\n\nTicket Classification:\n%s\n\nOriginal Ticket:\n%s",
        policyResult.Response.Content, classification.Response.Content, ticket), nil
})

wf.AddStage(replyStage)
wf.AddDependency(replyStage.ID, policyStage.ID)
```

### Stage 4: route_ticket (agent-loop with multiple tools, conditional on escalation)

The routing stage runs in parallel with Stages 2 and 3 because it only depends on `classify`. It has three tools: `send_email`, `read_team_descriptions`, and `send_ticket`. Its behavior branches on the `needs_escalation` flag from the classify output:

- **Needs escalation:** reads team descriptions, creates a routed internal ticket for the appropriate human team, and emails the team about the escalation.
- **No escalation:** reads team descriptions and sends an informational email to the responsible team letting them know the ticket is being handled automatically by the resolver (Stages 2 and 3).

```go
routeStage := orla.NewStage("route_ticket", heavyBackend)
routeStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
routeStage.SetMaxTurns(10)
routeStage.AddTool(emailToolRoute)
routeStage.AddTool(teamsTool)
routeStage.AddTool(ticketTool)
routeStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    classification := results[classifyStage.ID]
    // Parse needs_escalation from classify output to branch behavior
    var classifyOut struct {
        NeedsEscalation  bool   `json:"needs_escalation"`
        EscalationReason string `json:"escalation_reason"`
    }
    json.Unmarshal([]byte(classification.Response.Content), &classifyOut)

    if classifyOut.NeedsEscalation {
        // Route to human team
        return fmt.Sprintf("... escalation prompt with reason: %s ...",
            classifyOut.EscalationReason), nil
    }
    // Notify team of auto-resolution
    return fmt.Sprintf("... notification prompt ..."), nil
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

The DAG executor starts `classify` first (no dependencies). Once it completes, both `policy_check` and `route_ticket` become unblocked and run in parallel. `policy_check` calls its tool, renders a decision, and when it finishes, `reply` becomes unblocked and runs. Meanwhile, `route_ticket` reads the `needs_escalation` flag from the classification and either routes an escalated ticket to a human team or notifies the team of auto-resolution. Context between stages is handled by `PromptBuilder` functions on each stage.

## Example output

The demo includes a built-in sample ticket (a duplicate billing charge complaint). To use your own ticket, set the `TICKET_PATH` environment variable:

```bash
TICKET_PATH=/path/to/ticket.txt go run ./examples/workflow_demo/cmd/workflow_demo
```

You can override individual URLs or models if needed (e.g. `SGLANG_LIGHT_URL`, `VLLM_HEAVY_URL`, `OLLAMA_URL`, `LIGHT_MODEL`, `HEAVY_MODEL`). When using Ollama, both models are served by the same process; the demo defaults to `qwen3:0.6b` (light) and `qwen3:1.7b` (heavy).

Here is an output on SGLang on a server where the ticket was escalated:

```bash
2026/03/05 15:32:32 ================================================
2026/03/05 15:32:32 Running customer support workflow demo
2026/03/05 15:32:32 ================================================
2026/03/05 15:32:32 Stage mapping validated: 4 stages assigned to backends
2026/03/05 15:32:32 Executing customer support workflow...
2026/03/05 15:32:32   Stage DAG:
2026/03/05 15:32:32     classify ──┬──▶ policy_check ──▶ reply
2026/03/05 15:32:32                └──▶ route_ticket
2026/03/05 15:32:37 [send_ticket] Team: billing_ops | Priority: high
2026/03/05 15:32:37 [send_email] To: billing-ops@company.com | Subject: Escalated Ticket: Refund Request for Duplicate Charges
2026/03/05 15:32:39   classify:
2026/03/05 15:32:39     {
  "category": "billing",
  "customer_request": "Refund for two duplicate charges of $49.99 made on October 3 and October 5 for my Pro subscription",
  "key_issue": "The customer was charged twice for their Pro subscription in October and is requesting a refund for the duplicate charges.",
  "needs_escalation": true
  ,
  "product": "Pro subscription ($49.99/month)"
}
2026/03/05 15:32:39   policy_check:
2026/03/05 15:32:39     {
  "applicable_policy": "billing",
  "decision": "accept",
  "reasoning": "The customer was charged twice for their Pro subscription and is requesting a refund for the duplicate charges. According to the billing policy, duplicate charges should be refunded within 5 business days. The customer's request aligns with the policy, and the needs_escalation flag is set to true, indicating the matter should be escalated for prompt resolution."
}
2026/03/05 15:32:39   reply:
2026/03/05 15:32:39     {
  "email_sent": true,
  "summary": "An email has been sent to the customer, acknowledging the receipt of their request and informing them that it is being escalated to a specialist team for further review."
}
2026/03/05 15:32:39   route_ticket:
2026/03/05 15:32:39     The ticket has been successfully escalated to the appropriate team. Here's a summary of what was done:

1. **Team Identification**: The team responsible for handling refund requests, subscription changes, payment disputes, and invoice corrections (billing_ops) was identified.
2. **Ticket Creation**: An internal support ticket was created and routed to the billing_ops team with the ticket ID `TKT-billing_ops-42`. The escalation reason was included.
3. **Email Notification**: An email was sent to ...
2026/03/05 15:32:39     (tool calls executed: 3)
```

Here is a second output with Ollama on a Macbook where the ticket was not escalated:

```bash
2026/03/06 07:42:36 ================================================
2026/03/06 07:42:36 Running customer support workflow demo
2026/03/06 07:42:36 ================================================
2026/03/06 07:42:36 Stage mapping validated: 4 stages assigned to backends
2026/03/06 07:42:36 Executing customer support workflow...
2026/03/06 07:42:36   Stage DAG:
2026/03/06 07:42:36     classify ──┬──▶ policy_check ──▶ reply
2026/03/06 07:42:36                └──▶ route_ticket
2026/03/06 07:43:29 [send_email] To: leonhard.euler@email.com | Subject: Auto-Resolved Ticket: Duplicate Charges Refund Request
2026/03/06 07:45:34   classify:
2026/03/06 07:45:34     {"category": "billing", "customer_request": "a refund for duplicate charges", "key_issue": "duplicate charges in Pro subscription", "needs_escalation": false, "product": "Pro ($49.99/month)"}


2026/03/06 07:45:34   policy_check:
2026/03/06 07:45:34     {
  "applicable_policy": "billing",
  "decision": "accept",
  "reasoning": "The company supports refunds for duplicate charges as per the billing policy. The customer's request aligns with the policy's provisions for resolving duplicate charges."
}
2026/03/06 07:45:34   reply:
2026/03/06 07:45:34     {
  "email_sent": true,
  "summary": "Email sent to leonhard.euler@email.com confirming refund processing and ETA of 48 hours. Refund amount $50.00 will be processed within 48 hours. Customer can expect a refund within 48 hours and will be notified via email once processed."
}
2026/03/06 07:45:34   route_ticket:
2026/03/06 07:45:34     The system identified the "billing_ops" team as responsible for the duplicate charges refund request. An automated email was sent to this team informing them of the auto-resolution status, including the customer's account details and ticket summary. The email was successfully delivered (status: sent).

**Summary:**
- Team: billing_ops
- Action: Sent auto-resolution notification to billing team with customer details
- Status: Email delivered successfully
2026/03/06 07:45:34     (tool calls executed: 2)
```

## Control Flow

The workflow executor walks the stage dependency graph and runs stages in topological order. `classify` is the only root with no dependencies, so the DAG executor fires it first on the light backend. It sends the raw ticket and receives structured JSON back (category, product, key issue, customer request, and `needs_escalation`).

Once `classify` completes, two stages become unblocked: `policy_check` and `route_ticket`. Both run in parallel on the heavy backend. `policy_check` enters an agent loop: the model generates a request to call `read_policy_yaml` with the ticket's category, the tool returns the relevant policy document, and the model then generates a structured accept/deny decision with reasoning. Meanwhile, `route_ticket` reads the `needs_escalation` flag from the classification output. If escalation is needed, it enters an agent loop that reads team descriptions, creates a routed internal ticket for the appropriate human team, and emails the team about the escalation. If escalation is not needed, it instead sends an informational email to the responsible team letting them know the ticket is being handled automatically by the resolver stages.

Once `policy_check` finishes, `reply` becomes unblocked. It reads the `needs_escalation` flag from the classification. If escalation is needed, it enters an agent loop that sends a brief acknowledgment email to the customer letting them know their request is being routed to a specialist team -- it does not resolve the issue. If escalation is not needed, it composes a full resolution email based on the policy decision and sends it. Either way, the `reply` stage produces a structured confirmation indicating whether the email was sent and a brief summary.

Scheduling hints are set dynamically in the `PromptBuilder` for the `reply` stage based on the classification output: billing and technical tickets get higher priority (8) than other categories (5). Because the agent-loop stages (`policy_check`, `reply`, `route_ticket`) all use Priority scheduling on the heavy backend, Orla's scheduler can prioritize urgent tickets when multiple workflows compete for the same backend simultaneously.
