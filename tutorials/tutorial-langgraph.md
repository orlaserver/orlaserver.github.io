# Tutorial: Using Orla with LangGraph (Python)

This tutorial shows how to use **pyorla**, Orla's Python SDK, to run LLM-powered agentic workflows with [LangGraph](https://langchain-ai.github.io/langgraph/). Orla handles inference scheduling, multi-backend orchestration, and KV cache management while LangGraph provides graph-based workflow orchestration.

## Install

pyorla requires Python 3.11+. You need [uv](https://docs.astral.sh/uv/) installed. Assume you have cloned the Orla repo. From a project inside the repo:

```bash
cd orla
uv init my-langgraph-app
cd my-langgraph-app
uv add ../pyorla
```

If your project is outside the repo, use the path to the pyorla directory (e.g. `uv add ../orla/pyorla`). This installs pyorla and its dependencies (langchain-core, langgraph, httpx, pydantic).

## Docker Compose (Orla + vLLM)

You need Orla and at least one LLM backend running. The Orla repo includes Docker Compose files you can reuse.

**Tier 1 (single model):** Start the basic vLLM stack from the Orla repo root:

```bash
git clone https://github.com/dorcha-inc/orla.git
cd orla
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

This runs vLLM on port 8000 and Orla on port 8081. From your Python process on the host, use `http://localhost:8081` for Orla and `http://localhost:8000/v1` for the vLLM backend. See [Tutorial: Run a simple agent with Orla and vLLM](tutorial-vllm-lily.md) for prerequisites (Docker, NVIDIA GPU, etc.).

**Tier 2–3 (multi-stage, light + heavy models):** For workflows that use separate backends for classification vs. response, start the workflow-demo stack:

```bash
cd orla
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml up -d
```

This runs vLLM heavy on port 8000 and vLLM light on port 8001. Set `VLLM_LIGHT_URL=http://localhost:8001/v1` and `VLLM_HEAVY_URL=http://localhost:8000/v1` when configuring your backends (or use those URLs directly in `new_vllm_backend`).

## Tier 1: Simple — One model, one call

The simplest way to use Orla with LangGraph. A single `ChatOrla` wraps a registered backend and works like any other LangChain chat model. Run with the basic vLLM stack (`docker-compose.vllm.yaml`).

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from pyorla import OrlaClient, Stage, new_vllm_backend
from typing import TypedDict

# Connect to Orla and register a backend (use localhost when running from host)
client = OrlaClient("http://localhost:8081")
backend = new_vllm_backend("Qwen/Qwen3-4B-Instruct-2507", "http://localhost:8000/v1")
client.register_backend(backend)

# Create a ChatOrla
stage = Stage("my-stage", backend)
stage.client = client
stage.set_max_tokens(512)
llm = stage.as_chat_model()

# LangGraph
class State(TypedDict):
    question: str
    answer: str

def answer_node(state):
    resp = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": resp.content}

graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.set_entry_point("answer")
graph.add_edge("answer", END)
app = graph.compile()

result = app.invoke({"question": "What is Orla?"})
print(result["answer"])
```

## Tier 2: Multi-stage — Different models for different tasks

Use Orla's scheduling and stage mapping to route different parts of your workflow to different backends with different parameters. Run with the workflow-demo stack (`docker-compose.workflow-demo.vllm.yaml`).

```python
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from pyorla import (
    OrlaClient, Stage,
    ExplicitStageMapping, StageMappingInput,
    apply_stage_mapping_output,
    new_vllm_backend, StructuredOutputRequest,
    SCHEDULING_POLICY_FCFS, SCHEDULING_POLICY_PRIORITY,
)
from typing import TypedDict

# Connect and register backends (localhost when running from host)
client = OrlaClient("http://localhost:8081")
light = new_vllm_backend("Qwen/Qwen3-4B-Instruct-2507", "http://localhost:8001/v1")
heavy = new_vllm_backend("Qwen/Qwen3-8B", "http://localhost:8000/v1")
client.register_backend(light)
client.register_backend(heavy)

# Stage 1: fast classification on light model
classify_stage = Stage("classify", light)
classify_stage.client = client
classify_stage.set_max_tokens(256)
classify_stage.set_temperature(0)
classify_stage.set_scheduling_policy(SCHEDULING_POLICY_FCFS)
classify_stage.set_response_format(StructuredOutputRequest(
    name="classify",
    schema={"type": "object", "properties": {"category": {"type": "string"}}},
))
classify_llm = classify_stage.as_chat_model()

# Stage 2: detailed response on heavy model
respond_stage = Stage("respond", heavy)
respond_stage.client = client
respond_stage.set_max_tokens(1024)
respond_stage.set_temperature(0.3)
respond_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
respond_llm = respond_stage.as_chat_model()

# Validate stage mapping
all_stages = [classify_stage, respond_stage]
mapping = ExplicitStageMapping()
output = mapping.map(StageMappingInput(stages=all_stages, backends=[light, heavy]))
apply_stage_mapping_output(all_stages, output)

# LangGraph: classify → respond
class State(TypedDict):
    question: str
    category: str
    answer: str

def classify_node(state):
    resp = classify_llm.invoke([HumanMessage(content=state["question"])])
    return {"category": resp.content}

def respond_node(state):
    resp = respond_llm.invoke([
        HumanMessage(content=f"Question: {state['question']}\nCategory: {state['category']}\n\nProvide a detailed answer."),
    ])
    return {"answer": resp.content}

graph = StateGraph(State)
graph.add_node("classify", classify_node)
graph.add_node("respond", respond_node)
graph.set_entry_point("classify")
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)
app = graph.compile()

result = app.invoke({"question": "What is Orla and how does it schedule inference?"})
print(result["answer"])
```

Each `ChatOrla` carries its Stage's scheduling policy, backend assignment, and inference parameters — Orla's server-side scheduler handles the rest.

## Tier 3: Full workflow — Agent loops, tools, and parallel stages

This mirrors the Go `workflow_demo`: a customer support triage pipeline with four stages, tool calls, structured output, and parallel execution. Run with the workflow-demo stack.

```
classify ──┬──▶ policy_check ──▶ reply
           └──▶ route_ticket
```

```python
import json
import operator
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from pyorla import (
    OrlaClient, Stage, Message,
    ExplicitStageMapping, StageMappingInput,
    apply_stage_mapping_output,
    new_vllm_backend, StructuredOutputRequest,
    SchedulingHints, EXECUTION_MODE_AGENT_LOOP,
    SCHEDULING_POLICY_FCFS, SCHEDULING_POLICY_PRIORITY,
)
from pyorla.tools import Tool, tool_runner_from_schema
from typing import TypedDict, Annotated

# --- Setup ---
client = OrlaClient("http://localhost:8081")
light = new_vllm_backend("Qwen/Qwen3-4B-Instruct-2507", "http://localhost:8001/v1")
heavy = new_vllm_backend("Qwen/Qwen3-8B", "http://localhost:8000/v1")
client.register_backend(light)
client.register_backend(heavy)

# Mock tools
def read_policy(input_args):
    category = input_args.get("category", "general")
    return {"policy_document": f"Policy for {category}: refund within 30 days, etc."}

def send_email(input_args):
    return {"status": "sent", "message_id": f"msg-{input_args.get('to', '')}-001"}

def read_teams(_):
    return {"teams": [{"name": "billing_ops", "email": "billing@co.com"}, {"name": "tech_support", "email": "tech@co.com"}]}

def send_ticket(input_args):
    return {"ticket_id": f"TKT-{input_args.get('team', '')}-42", "status": "created"}

policy_tool = Tool("read_policy_yaml", "Look up company policy for a category",
    {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]},
    run=tool_runner_from_schema(read_policy))
email_tool = Tool("send_email", "Send email to recipient",
    {"type": "object", "properties": {"to": {}, "subject": {}, "body": {}}, "required": ["to", "subject", "body"]},
    run=tool_runner_from_schema(send_email))
teams_tool = Tool("read_team_descriptions", "List internal teams", {"type": "object", "properties": {}},
    run=tool_runner_from_schema(read_teams))
ticket_tool = Tool("send_ticket", "Create internal ticket",
    {"type": "object", "properties": {"team": {}, "priority": {}, "summary": {}}, "required": ["team", "priority", "summary"]},
    run=tool_runner_from_schema(send_ticket))

# Stages
classify_stage = Stage("classify", light)
classify_stage.client = client
classify_stage.set_max_tokens(512)
classify_stage.set_temperature(0)
classify_stage.set_scheduling_policy(SCHEDULING_POLICY_FCFS)
classify_stage.set_response_format(StructuredOutputRequest("classify", {
    "type": "object", "properties": {"category": {}, "needs_escalation": {}}, "required": ["category", "needs_escalation"]
}))
classify_llm = classify_stage.as_chat_model()

policy_stage = Stage("policy_check", heavy)
policy_stage.client = client
policy_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
policy_stage.set_max_turns(5)
policy_stage.set_max_tokens(1024)
policy_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
policy_stage.add_tool(policy_tool)

reply_stage = Stage("reply", heavy)
reply_stage.client = client
reply_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
reply_stage.set_max_turns(5)
reply_stage.set_max_tokens(1024)
reply_stage.add_tool(email_tool)

route_stage = Stage("route_ticket", heavy)
route_stage.client = client
route_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
route_stage.set_max_turns(10)
route_stage.add_tool(teams_tool)
route_stage.add_tool(email_tool)
route_stage.add_tool(ticket_tool)

mapping = ExplicitStageMapping()
output = mapping.map(StageMappingInput(stages=[classify_stage, policy_stage, reply_stage, route_stage], backends=[light, heavy]))
apply_stage_mapping_output([classify_stage, policy_stage, reply_stage, route_stage], output)

# --- Agent-loop helper ---
def run_agent_loop(stage, prompt):
    from pyorla.types import Message
    messages = [Message(role="user", content=prompt)]
    last_content = ""
    for _ in range(stage.max_turns or 10):
        resp = stage.execute_with_messages(messages)
        last_content = resp.content
        messages.append(Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls))
        if not resp.tool_calls:
            break
        for tr in stage.run_tool_calls_in_response(resp):
            m = tr.to_message_dict()
            messages.append(Message(role="tool", content=m["content"], tool_call_id=m.get("tool_call_id", ""), tool_name=m.get("tool_name", "")))
    return last_content

# --- LangGraph ---
class WorkflowState(TypedDict, total=False):
    ticket: str
    classify_result: str
    policy_result: str
    reply_result: str
    route_result: str
    completed_stages: Annotated[list[str], operator.add]

def classify_node(state):
    resp = classify_llm.invoke([HumanMessage(content=f"Classify this support ticket:\n\n{state['ticket']}")])
    return {"classify_result": resp.content, "completed_stages": ["classify"]}

def policy_check_node(state):
    prompt = f"Check policy for this classification:\n{state['classify_result']}\n\nOriginal ticket:\n{state['ticket']}\n\nUse read_policy_yaml first, then decide accept or deny."
    return {"policy_result": run_agent_loop(policy_stage, prompt), "completed_stages": ["policy_check"]}

def reply_node(state):
    classify_data = json.loads(state["classify_result"]) if state["classify_result"].strip().startswith("{") else {}
    priority = 8 if classify_data.get("category") in ("billing", "technical") else 5
    reply_stage.set_scheduling_hints(SchedulingHints(priority=priority))
    prompt = f"Policy decision:\n{state['policy_result']}\n\nClassification:\n{state['classify_result']}\n\nTicket:\n{state['ticket']}\n\nCompose and send a reply using send_email."
    return {"reply_result": run_agent_loop(reply_stage, prompt), "completed_stages": ["reply"]}

def route_ticket_node(state):
    prompt = f"Classification:\n{state['classify_result']}\n\nTicket:\n{state['ticket']}\n\nUse read_team_descriptions, then send_ticket or send_email as needed."
    return {"route_result": run_agent_loop(route_stage, prompt), "completed_stages": ["route_ticket"]}

graph = StateGraph(WorkflowState)
graph.add_node("classify", classify_node)
graph.add_node("policy_check", policy_check_node)
graph.add_node("reply", reply_node)
graph.add_node("route_ticket", route_ticket_node)
graph.set_entry_point("classify")
graph.add_edge("classify", "policy_check")
graph.add_edge("classify", "route_ticket")
graph.add_edge("policy_check", "reply")
graph.add_edge("reply", END)
graph.add_edge("route_ticket", END)
app = graph.compile()

ticket = "I was charged twice for my subscription. Need a refund. - customer@example.com"
result = app.invoke({"ticket": ticket, "completed_stages": []})
print("Reply:", result.get("reply_result", "")[:200])
print("Route:", result.get("route_result", "")[:200])
```

The `reply_node` uses dynamic priority: billing and technical tickets get priority 8; others get 5.

## KV cache management

Orla's Memory Manager tracks KV cache across stages within a workflow. When using LangGraph, set `workflow_id` on your stages and call `workflow_complete` when the workflow finishes so Orla can flush caches and clean up:

```python
import uuid
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from pyorla import OrlaClient, Stage, new_vllm_backend
from typing import TypedDict

client = OrlaClient("http://localhost:8081")
light = new_vllm_backend("Qwen/Qwen3-4B-Instruct-2507", "http://localhost:8001/v1")
heavy = new_vllm_backend("Qwen/Qwen3-8B", "http://localhost:8000/v1")
client.register_backend(light)
client.register_backend(heavy)

classify_stage = Stage("classify", light)
classify_stage.client = client
policy_stage = Stage("policy_check", heavy)
policy_stage.client = client
classify_llm = classify_stage.as_chat_model()

wf_id = str(uuid.uuid4())[:8]
for stage in [classify_stage, policy_stage]:
    stage.set_workflow_id(wf_id)

class State(TypedDict):
    question: str
    classify_result: str
    policy_result: str

def classify_node(state):
    resp = classify_llm.invoke([HumanMessage(content=state["question"])])
    return {"classify_result": resp.content}

def policy_node(state):
    prompt = f"Classification: {state['classify_result']}\n\nDecide accept or deny."
    resp = policy_stage.execute(prompt)
    return {"policy_result": resp.content}

graph = StateGraph(State)
graph.add_node("classify", classify_node)
graph.add_node("policy", policy_node)
graph.set_entry_point("classify")
graph.add_edge("classify", "policy")
graph.add_edge("policy", END)
app = graph.compile()

result = app.invoke({"question": "I was charged twice, need refund."})
print(result["policy_result"])

client.workflow_complete(wf_id, [light.name, heavy.name])
```

## Why add Orla to your LangGraph pipelines

LangGraph gives you workflow orchestration including state, control flow, conditional edges, checkpointing. Orla adds production-grade agentic serving underneath:

- **Multi-backend routing**: Route different stages to different models (e.g., fast classifier on a small model, detailed reasoning on a larger one) without changing your graph. Orla handles backend registration and stage mapping.

- **Server-side scheduling**: When many requests hit the same backend, Orla schedules them with configurable policies (FCFS, priority). You can set per-request priority from your LangGraph state (e.g., escalate billing tickets).

- **KV cache management**: Orla tracks workflow context across stages and manages KV cache lifecycle. Preserve cache when the next stage adds few tokens; flush at workflow boundaries or when switching models. Reduces redundant computation and latency.

- **Self-hosted at scale**: Orla is built for self-hosted vLLM, SGLang, and Ollama. If you run your own models, Orla gives you scheduling, memory management, and multi-model orchestration that hosted APIs don't provide.

- **Lower cloud API costs**: Use stage mapping to route simple tasks (classification, extraction) to cheaper cloud models, and reserve expensive ones for complex reasoning. Orla also lets you mix backends, including self-hosted and cloud, in the same workflow.

If you're already on LangGraph and hitting limits with direct API calls — queueing, cache waste, juggling multiple backends, or runaway API costs — Orla slots in as the agentic serving layer.

## Full example

See `pyorla/examples/workflow_demo/` in the Orla repository for the complete customer support triage workflow implemented in both:

- `run_workflow.py` — native pyorla Workflow API
- `run_langgraph.py` — LangGraph StateGraph with pyorla stages
