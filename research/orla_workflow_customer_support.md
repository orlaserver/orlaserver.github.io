# Multi-Stage Workflow with Orla

This tutorial runs a multi-stage workflow using [Orla](https://github.com/dorcha-inc/orla) and two [SGLang](https://sgl-project.github.io/) backends. The workflow processes a customer support ticket through a four-stage pipeline: a classification stage (light model) extracts structured ticket metadata and decides whether the ticket needs human escalation; a policy check stage (heavy model) reads company policy via tool call and renders an accept/deny decision; a reply stage (heavy model) either sends a brief escalation acknowledgment or a full resolution email via tool call; and a routing stage (heavy model) conditionally either routes an escalated ticket to a human team or notifies the team that the ticket was resolved automatically. The policy check → reply chain and the route_ticket stage run in parallel after classification. This demonstrates Orla's full abstraction stack: Workflow, Stage DAG, Stage Mapping, Scheduling, and tool-calling agent loops.

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

## 3. Run the demo

From the Orla repo root, run the workflow demo:

```bash
go run ./examples/workflow_demo/cmd/workflow_demo
```

The output should be something like

```bash
2026/03/05 13:46:28 ================================================
2026/03/05 13:46:28 Running customer support workflow demo
2026/03/05 13:46:28 ================================================
2026/03/05 13:46:28 Stage mapping validated: 4 stages assigned to backends
2026/03/05 13:46:28 Executing customer support workflow...
2026/03/05 13:46:28   Stage DAG:
2026/03/05 13:46:28     classify ──┬──▶ policy_check ──▶ reply
2026/03/05 13:46:28                └──▶ route_ticket
2026/03/05 13:46:32 [send_email] To: leonhard.euler@email.com | Subject: Acknowledgment of Your Ticket
2026/03/05 13:46:32 [send_ticket] Team: billing | Priority: high
2026/03/05 13:46:36   classify:
2026/03/05 13:46:36     {
  "category": "billing",
  "customer_request": "Refund for a duplicate charge of $49.99 made on October 3 and October 5 for my Pro subscription, as I did not authorize a second subscription.",
  "key_issue": "The customer was charged twice for their Pro subscription in October, resulting in a total of $99.98, and is requesting a refund for the duplicate payment.",
  "product": "Pro subscription"
}
2026/03/05 13:46:36   policy_check:
2026/03/05 13:46:36     {
  "applicable_policy": "billing",
  "decision": "accept",
  "reasoning": "The customer was charged twice for their Pro subscription, which is a billing issue. The company's policy states that duplicate charges should be refunded within 5 business days. The customer is requesting a refund for the duplicate payment, which aligns with the policy. The dashboard performance issue is unrelated to the refund request and can be addressed separately."
}
2026/03/05 13:46:36   reply:
2026/03/05 13:46:36     {
  "email_sent": true,
  "summary": "The refund for the duplicate charge has been processed and will be issued within 5 business days. The dashboard performance issue will be addressed separately. Thank you for your patience."
}
2026/03/05 13:46:36   route_ticket:
2026/03/05 13:46:36     I have completed the following steps:

1. Sent an acknowledgment email to the customer, Leonhard Euler, confirming their ticket was received.
2. Read the team descriptions and determined that the "billing_ops" team should handle this ticket, as it involves a refund and payment dispute.
3. Created an internal support ticket with the following details:
   - **Ticket ID:** TKT-billing-42
   - **Team:** billing_ops
   - **Priority:** High (as the customer marked the issue as urgent and is on a tight...
2026/03/05 13:46:36     (tool calls executed: 3)
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

The workflow executor walks the stage dependency graph and runs stages in topological order. `classify` is the only root with no dependencies, so the DAG executor fires it first on the light backend. It sends the raw ticket and receives structured JSON back (category, product, key issue, customer request, and `needs_escalation`).

Once `classify` completes, two stages become unblocked: `policy_check` and `route_ticket`. Both run in parallel on the heavy backend. `policy_check` enters an agent loop: the model generates a request to call `read_policy_yaml` with the ticket's category, the tool returns the relevant policy document, and the model then generates a structured accept/deny decision with reasoning. Meanwhile, `route_ticket` reads the `needs_escalation` flag from the classification output. If escalation is needed, it enters an agent loop that reads team descriptions, creates a routed internal ticket for the appropriate human team, and emails the team about the escalation. If escalation is not needed, it instead sends an informational email to the responsible team letting them know the ticket is being handled automatically by the resolver stages.

Once `policy_check` finishes, `reply` becomes unblocked. It reads the `needs_escalation` flag from the classification. If escalation is needed, it enters an agent loop that sends a brief acknowledgment email to the customer letting them know their request is being routed to a specialist team -- it does not resolve the issue. If escalation is not needed, it composes a full resolution email based on the policy decision and sends it. Either way, the `reply` stage produces a structured confirmation indicating whether the email was sent and a brief summary.

Scheduling hints are set dynamically in the `PromptBuilder` for the `reply` stage based on the classification output: billing and technical tickets get higher priority (8) than other categories (5). Because the agent-loop stages (`policy_check`, `reply`, `route_ticket`) all use Priority scheduling on the heavy backend, Orla's scheduler can prioritize urgent tickets when multiple workflows compete for the same backend simultaneously.
