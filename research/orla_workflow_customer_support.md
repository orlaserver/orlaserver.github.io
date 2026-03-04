# Multi-Agent Workflow with Orla

This tutorial runs a multi-agent workflow using [Orla](https://github.com/dorcha-inc/orla) and two [SGLang](https://sgl-project.github.io/) backends. The workflow processes a customer support ticket through a seven-stage pipeline spanning three agents: a triage agent (light model) classifies, analyzes sentiment, and prioritizes the ticket in a parallel diamond DAG; then a resolver agent (heavy model) and an escalation agent (light model) run in parallel -- the resolver drafts and reviews a reply while the escalation agent decides whether human intervention is needed. This demonstrates Orla's full abstraction stack: Workflow, Agent, Stage DAG, Stage Mapping, Scheduling, and Context Passing, including both intra-agent parallelism (diamond DAGs) and inter-agent parallelism (workflow-level fan-out).

## Architecture

<!-- ```
Workflow
  |
  +-- Agent "triage" (light model, FCFS)
  |     +-- Stage "classify"    -> structured JSON: category, product, key issue
  |     +-- Stage "sentiment"   -> structured JSON: sentiment, urgency signals
  |     |   (classify and sentiment run in parallel)
  |     +-- Stage "prioritize"  -> structured JSON: severity, priority, reasoning
  |         (depends on both classify and sentiment)
  |
  +-- Agent "resolver" (heavy model, Priority; depends on "triage")
  |     +-- Stage "draft_response" -> personalized customer reply
  |     +-- Stage "policy_check"   -> applicable policies and constraints
  |     |   (draft_response and policy_check run in parallel)
  |     +-- Stage "final_review"   -> compliance review (depends on both)
  |
  +-- Agent "escalation" (light model, FCFS; depends on "triage", parallel with "resolver")
        +-- Stage "route_ticket"   -> structured JSON: escalate, reason, suggested team
``` -->

This demo exercises every layer of the Orla abstraction stack. Within each agent, stages form a DAG. The triage agent has a diamond-shaped DAG: `classify` and `sentiment` are independent roots that run in parallel, and `prioritize` depends on both (fan-in), waiting for both to complete before deciding severity. The resolver agent mirrors this shape: `draft_response` and `policy_check` run in parallel, and `final_review` depends on both.

The workflow itself is also a DAG of agents. The `resolver` and `escalation` agents both depend on the `triage` agent, so after triage completes they run in parallel -- the resolver on the heavy backend and the escalation agent on the light backend. This demonstrates fan-out at the workflow level.

Before execution, Stage Mapping validates the plan. Each stage is explicitly assigned to a backend (light or heavy), and `ExplicitStageMapping` checks that every stage points to a registered backend. This catches configuration errors before any inference happens.

On the server side, Orla uses two-level scheduling. The triage and escalation agents' stages run with FCFS (first-come, first-served), appropriate for lightweight classification work. The resolver agent's stages use Priority scheduling, with the numeric priority produced directly by the triage agent's `prioritize` stage and injected via context passing, so when multiple tickets compete for the heavy backend, urgent ones are served first.

Four of the seven stages produce structured JSON output via schemas (`classify`, `sentiment`, `prioritize`, and `route_ticket`), ensuring downstream stages and the context passing function always receive machine-parseable data rather than free-form text. Context passing connects the agents: when the triage agent finishes, its structured outputs are injected into both the resolver and escalation agents' prompts, and the numeric priority from the `prioritize` stage is set as the scheduling hint on the resolver's stages.

## What you need

- Docker and Docker Compose (Compose V2).
- An NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- The [Orla repo](https://github.com/dorcha-inc/orla) cloned.
- Go 1.25 or later.

## 1. Start the backends and Orla

The workflow uses two SGLang backends: a **light** model (Qwen3-4B) for triage and escalation, and a **heavy** model (Qwen3-8B) for resolution. Use the **workflow-demo** compose file so you get a clean stack and avoid network conflicts with other compose projects (e.g. SWE-bench Lite). You can run with **SGLang** (default) or **vLLM**. From the Orla repo root:

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

### Agent 1: triage (diamond DAG)

The triage agent has three stages in a diamond DAG. `classify` and `sentiment` are independent roots that run in parallel, both analyzing the raw ticket. `prioritize` depends on both and produces the final severity and numeric scheduling priority:

```go
triage := orla.NewAgent(client)
triage.Name = "triage"

classifyStage := orla.NewStage("classify", lightBackend)
classifyStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_classify", classifySchema))
classifyStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
classifyStage.Prompt = fmt.Sprintf("Classify this support ticket...\n\n%s", ticket)

sentimentStage := orla.NewStage("sentiment", lightBackend)
sentimentStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_sentiment", sentimentSchema))
sentimentStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
sentimentStage.Prompt = fmt.Sprintf("Determine the customer's sentiment and urgency signals...\n\n%s", ticket)

prioritizeStage := orla.NewStage("prioritize", lightBackend)
prioritizeStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_priority", prioritySchema))
prioritizeStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    classification := results[classifyStage.ID]
    sentiment := results[sentimentStage.ID]
    return fmt.Sprintf("Assign a severity and priority (1-10)...\n\n%s\n\n%s",
        classification.Response.Content, sentiment.Response.Content), nil
})

triage.AddStage(classifyStage)
triage.AddStage(sentimentStage)
triage.AddStage(prioritizeStage)
triage.AddDependency(prioritizeStage.ID, classifyStage.ID)
triage.AddDependency(prioritizeStage.ID, sentimentStage.ID)
```

Because `classify` and `sentiment` have no dependency between them, Orla's DAG executor fires both concurrently. The `prioritize` stage waits for both to finish before running.

### Agent 2: resolver (diamond DAG)

The resolver agent also has a diamond DAG. `draft_response` and `policy_check` run in parallel on the heavy backend, and `final_review` depends on both. No priority hint is set statically -- it is injected by the context passing function using the triage agent's structured output:

```go
resolver := orla.NewAgent(client)
resolver.Name = "resolver"

draftStage := orla.NewStage("draft_response", heavyBackend)
draftStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)

policyStage := orla.NewStage("policy_check", heavyBackend)
policyStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)

reviewStage := orla.NewStage("final_review", heavyBackend)
reviewStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
reviewStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
    draft := results[draftStage.ID]
    policies := results[policyStage.ID]
    return fmt.Sprintf("Review this draft against the policies...\n\n%s\n\n%s",
        draft.Response.Content, policies.Response.Content), nil
})

resolver.AddStage(draftStage)
resolver.AddStage(policyStage)
resolver.AddStage(reviewStage)
resolver.AddDependency(reviewStage.ID, draftStage.ID)
resolver.AddDependency(reviewStage.ID, policyStage.ID)
```

### Agent 3: escalation

The escalation agent has a single stage that decides whether the ticket needs human intervention. It runs on the light backend in parallel with the resolver:

```go
escalation := orla.NewAgent(client)
escalation.Name = "escalation"

routeStage := orla.NewStage("route_ticket", lightBackend)
routeStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
routeStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_escalation", escalationSchema))

escalation.AddStage(routeStage)
```

### Stage mapping

Before execution, `ExplicitStageMapping` validates that every stage is assigned to a registered backend:

```go
allStages := []*orla.Stage{classifyStage, sentimentStage, prioritizeStage,
    draftStage, policyStage, reviewStage, routeStage}
mapping := &orla.ExplicitStageMapping{}
output, _ := mapping.Map(&orla.StageMappingInput{
    Stages:   allStages,
    Backends: []*orla.LLMBackend{lightBackend, heavyBackend},
})
orla.ApplyStageMappingOutput(allStages, output)
```

### Workflow with context passing

The workflow defines the agent dependency graph and a context passing function. Both `resolver` and `escalation` depend on `triage`, so they run in parallel after triage completes. The context passing function feeds triage output into both downstream agents and sets the resolver's scheduling priority from the `prioritize` stage's structured JSON:

```go
wf := orla.NewWorkflow()
wf.AddAgent(triage)
wf.AddAgent(resolver)
wf.AddAgent(escalation)
wf.AddDependency("resolver", "triage")
wf.AddDependency("escalation", "triage")

wf.SetContextPassingFn(func(upstream map[string]*orla.AgentResult, downstream *orla.Agent) error {
    triageResult := upstream["triage"]

    // Read structured outputs from each triage stage.
    classifyOutput := triageResult.StageResults[classifyStage.ID].Response.Content
    sentimentOutput := triageResult.StageResults[sentimentStage.ID].Response.Content
    prioritizeOutput := triageResult.StageResults[prioritizeStage.ID].Response.Content

    var p struct{ Priority int `json:"priority"` }
    json.Unmarshal([]byte(prioritizeOutput), &p)

    switch downstream.Name {
    case "resolver":
        // Set scheduling priority and prompts for both parallel root stages.
        for _, s := range downstream.Stages() {
            s.SetSchedulingHints(&orla.SchedulingHints{Priority: &p.Priority})
            // ... set draft_response and policy_check prompts
        }
    case "escalation":
        // Set route_ticket prompt from all triage outputs.
        for _, s := range downstream.Stages() {
            // ... set route_ticket prompt
        }
    }
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

You should see output similar to:

```
2026/03/04 07:48:15 ================================================                                                            
2026/03/04 07:48:15 Running customer support workflow demo                                                                      
2026/03/04 07:48:15 ================================================                                                            
2026/03/04 07:48:15 Stage mapping validated: 7 stages assigned to backends                                                      
2026/03/04 07:48:15 Executing customer support workflow...                                                                      
2026/03/04 07:48:16 Triage assigned scheduling priority 9 to resolver                                                           
2026/03/04 07:48:30 === Agent: triage ===                                                                                       
2026/03/04 07:48:30   sentiment (stage id: trusting_snyder):                                                                    
2026/03/04 07:48:30     {
  "sentiment": "frustrated",
  "urgency_signals": [
    "URGENT",
    "I need a refund for the duplicate charge ASAP",
    "I'm on a tight budget this month and that extra $50 really hurts"
  ]
}
2026/03/04 07:48:30   classify (stage id: determined_goldwasser):
2026/03/04 07:48:30     {
  "category": "billing",
  "key_issue": "Customer reports a duplicate charge of $49.99 for their Pro subscription, requesting a refund.",
  "product": "Pro subscription"
}
2026/03/04 07:48:30   prioritize (stage id: competent_buck):
2026/03/04 07:48:30     {  
  "priority": 9,  
  "reasoning": "The customer is frustrated and explicitly requests an urgent refund due to a duplicate charge that impacts their budget, indicating high financial and emotional urgency."  
  ,  
  "severity": "high"  
}
2026/03/04 07:48:30 === Agent: escalation ===
2026/03/04 07:48:30   route_ticket (stage id: stupefied_mirzakhani):
2026/03/04 07:48:30     {
  "escalate": true,
  "reason": "The ticket requires human escalation due to the high severity, customer sentiment, and financial impact. The customer reports a duplicate charge of $49.99 (totaling $99.98), which is a direct billing error with clear financial consequences. The customer is explicitly frustrated and requests an urgent refund, citing a tight budget and emotional distress. While automated systems may detect duplicate charges, the need for manual verification of subscription record...
2026/03/04 07:48:30 === Agent: resolver ===
2026/03/04 07:48:30   policy_check (stage id: relaxed_easley):
2026/03/04 07:48:30     Based on the **ticket classification** and **sentiment**, here is a comprehensive breakdown of **applicable support policies, refund rules, SLA commitments, and constraints** the response agent must follow:

---

### ✅ **1. Ticket Classification Summary**
- **Category:** Billing  
- **Key Issue:** Customer reports a duplicate charge of $49.99 for their Pro subscription, requesting a refund.  
- **Product:** Pro subscription ($49.99/month)  
- **Customer Name:** Leonhard Euler  
- **Email:** le...
2026/03/04 07:48:30   draft_response (stage id: wonderful_chaum):
2026/03/04 07:48:30     Subject: Refund for Duplicate Charge - Urgent

Dear Leonhard Euler,

Thank you for reaching out and for bringing this issue to our attention. We sincerely apologize for the inconvenience and frustr
ation caused by the duplicate charge on your Pro subscription. We understand how concerning this must be, especially given your 
tight budget, and we are committed to resolving this matter as quickly as possible.

We have already initiated a refund for the duplicate charge of $49.99 and it should be pro...
2026/03/04 07:48:30   final_review (stage id: vigorous_keldysh): 
2026/03/04 07:48:30     **REVIEW OUTCOME: APPROVED**

**Summary:**
The draft reply complies with all applicable policies, maintains a professional and empathetic tone, and addresses all aspects o
f the customer's concern comprehensively.

- **Policy Compliance:** The response acknowledges the duplicate charge, confirms the refund process, and aligns with the 5–7 bu
siness day refund timeline. It also addresses the urgency signal and financial hardship, which requires an expedited refund. The
 mention of investigating th...
```

Note that the resolver and escalation agents run concurrently -- the escalation result may appear before the resolver finishes, since it runs on the lighter model.

## 4. Stop the stack

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml down
# or, if you used vLLM:
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml down
```

If you see a "network not found" or permission error when starting containers, try bringing the stack down and removing orphan containers first: `docker compose -f deploy/docker-compose.workflow-demo.yaml down --remove-orphans`, then run `up -d` again. On Linux you may need `sudo` for Docker commands.

## Control Flow

The workflow executor walks the agent dependency graph and runs agents in topological order. Because both `resolver` and `escalation` depend on `triage`, the triage agent always runs first. Once triage completes, the executor launches the resolver and escalation agents concurrently -- this is the workflow-level fan-out.

Inside the triage agent, stages execute in DAG order. The `classify` and `sentiment` stages are both roots with no dependencies, so the DAG executor fires them in parallel on the light backend. `classify` sends the raw ticket and receives structured JSON back (category, product, key issue), while `sentiment` independently analyzes the customer's emotional state and extracts urgency signals. Once both complete, the `prioritize` stage's `PromptBuilder` reads both results and constructs a prompt asking the model to assign a severity and numeric scheduling priority (1--10), also as structured JSON. This diamond decomposition -- parallel analysis followed by a join -- keeps each inference call focused and demonstrates that Orla's stage DAG executor can exploit parallelism within a single agent.

When the triage agent finishes, the workflow executor calls the context passing function once for each downstream agent before launching it. For the resolver, the function injects the full triage output into `draft_response`'s prompt, passes the classification and sentiment specifically to `policy_check`, and reads the numeric `priority` field from the `prioritize` stage's structured JSON to set as the scheduling hint on all resolver stages. For the escalation agent, it passes the classification, sentiment, and priority assessment to `route_ticket` so the model can make an informed escalation decision. Because the triage agent produces the priority directly as an integer, no client-side mapping is needed.

The resolver and escalation agents then run concurrently. Inside the resolver, `draft_response` generates a personalized customer reply and `policy_check` identifies applicable support policies -- both running in parallel on the heavy backend. Once both complete, `final_review` reads both results and checks the draft against the identified policies for compliance, tone, and completeness. Meanwhile, the escalation agent's single `route_ticket` stage runs on the light backend, producing a structured JSON decision about whether the ticket needs human escalation and which team should handle it.

On the server side, each backend maintains per-stage request queues. The triage and escalation agents' stages use FCFS scheduling, sufficient for lightweight work on the light model. The resolver agent's stages use Priority scheduling with hints taken directly from the triage agent's structured output, so when multiple tickets compete for the heavy backend simultaneously, higher-priority tickets (e.g. a critical billing issue assigned priority 10 by the triage model) are served before lower-priority ones (e.g. a general inquiry assigned priority 2).

For more on Orla's scheduling and stage routing, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For the SWE-bench experiment that uses Priority scheduling with score prediction, see [Orla SWE-bench Lite](orla_swebench_lite.md).
