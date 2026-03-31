# Using Scheduling Policies

When multiple stages share a backend, Orla needs to decide which request to serve next. Scheduling policies control that decision. By default every request is served in the order it arrives (FCFS). When your workflow has stages with different urgency levels, you can switch to priority scheduling so that critical stages jump ahead of less important ones.

Orla's scheduler operates at two levels. The first level picks which stage queue to serve next. The second level picks which request within that stage queue to dequeue. Both levels default to FCFS and both can be switched to priority mode independently.

## Stage-level scheduling

Stage-level scheduling controls which stage gets served next when multiple stages are queued on the same backend. FCFS is the default and requires no configuration.

To use priority mode, set the policy and assign a priority value to each stage. Higher numbers mean higher priority. Ties are broken by arrival time, so older requests win:

```python
from pyorla import Stage
from pyorla.types import SCHEDULING_POLICY_PRIORITY, SchedulingHints

classify = Stage("classify", backend)
classify.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
classify.set_scheduling_hints(SchedulingHints(priority=3))

reply = Stage("reply", backend)
reply.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
reply.set_scheduling_hints(SchedulingHints(priority=8))
```

With this configuration, `reply` requests are dequeued before `classify` requests whenever both are waiting.

## Request-level scheduling

Request-level scheduling controls the order within a single stage's queue. This is useful when you have many concurrent requests for the same stage and some are more urgent than others.

To enable it, set the request scheduling policy and provide a `request_priority` in the hints:

```python
from pyorla.types import REQUEST_SCHEDULING_POLICY_PRIORITY, SchedulingHints

stage = Stage("solve", backend)
stage.set_request_scheduling_policy(REQUEST_SCHEDULING_POLICY_PRIORITY)
stage.set_scheduling_hints(SchedulingHints(request_priority=10))
```

When request-level priority is enabled, the scheduler scans the entire stage queue and picks the request with the highest `request_priority` value rather than always taking the head.

`priority` and `request_priority` are independent fields in `SchedulingHints`. Setting one does not affect the other. This lets you use different scheduling strategies at each level.

## Dynamic priority

Priority does not have to be static. You can change a stage's priority at runtime based on the content of earlier stages. This is useful when you want to fast-track certain categories of work.

The customer support example in the Orla repo demonstrates this pattern. A classification stage labels each ticket, and the reply stage adjusts its priority based on the label:

```python
from pyorla import Stage
from pyorla.types import SCHEDULING_POLICY_PRIORITY, SchedulingHints

classify = Stage("classify", light_backend)

reply = Stage("reply", heavy_backend)
reply.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)

def reply_node(state):
    category = state["classify_result"].get("category", "")
    if category in ("billing", "technical"):
        priority = 8   # urgent
    else:
        priority = 3   # normal

    reply.set_scheduling_hints(SchedulingHints(priority=priority))
    reply_llm = reply.as_chat_model()
    response = reply_llm.invoke(state["messages"])
    return {"reply_result": response.content}
```

Billing and technical tickets get priority 8 and jump ahead of general inquiries at priority 3. The scheduling decision happens inside the Orla daemon with no changes to the LangGraph graph structure.

## How the scheduler works

The daemon maintains one queue per stage per backend. When a worker becomes available, the scheduler runs two selection steps:

1. Pick the stage. In FCFS mode, it picks the stage whose head request arrived earliest. In priority mode, it picks the stage whose head request has the highest `priority` value.

2. Pick the request. In FCFS mode, it takes the head of the selected stage queue. In priority mode, it scans the queue and takes the request with the highest `request_priority`.

Tie-breaking at both levels uses enqueue time: the older request wins. This guarantees fairness when priorities are equal.

## Adding your own scheduling policy

Orla ships with two built-in policies, FCFS and priority, both at the stage level and at the request level. You can take advantage of Orla's dynamic priorities to specify your own scheduling policies.

### Shortest-job-first at the stage level

Estimate how expensive each stage is and give cheaper stages higher priority so they finish quickly and free up the backend:

```python
from pyorla import Stage
from pyorla.types import SCHEDULING_POLICY_PRIORITY, SchedulingHints

def sjf_priority(prompt: str) -> int:
    tokens = len(prompt.split())
    if tokens < 50:
        return 10
    if tokens < 200:
        return 5
    return 1

stage = Stage("solve", backend)
stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)

def solve_node(state):
    prompt = state["messages"][-1].content
    stage.set_scheduling_hints(SchedulingHints(priority=sjf_priority(prompt)))
    llm = stage.as_chat_model()
    return {"messages": [llm.invoke(state["messages"])]}
```

### Urgency-based at the request level

When many concurrent requests hit the same stage, serve the most urgent ones first. Here the urgency comes from the triage label assigned in an earlier stage:

```python
from pyorla import Stage
from pyorla.types import REQUEST_SCHEDULING_POLICY_PRIORITY, SchedulingHints

URGENCY = {"critical": 10, "high": 7, "normal": 3, "low": 1}

stage = Stage("reply", backend)
stage.set_request_scheduling_policy(REQUEST_SCHEDULING_POLICY_PRIORITY)

def reply_node(state):
    label = state.get("triage_label", "normal")
    stage.set_scheduling_hints(SchedulingHints(request_priority=URGENCY.get(label, 3)))
    llm = stage.as_chat_model()
    return {"messages": [llm.invoke(state["messages"])]}
```

### Combining both levels

You can set different policies at each level. For example, suppose you want deadline-based scheduling across stages so that urgent stages go first, and shortest-job-first within each stage so that short prompts are served before long ones:

```python
from pyorla import Stage
from pyorla.types import (
    SCHEDULING_POLICY_PRIORITY,
    REQUEST_SCHEDULING_POLICY_PRIORITY,
    SchedulingHints,
)

def deadline_stage_priority(stage_name: str) -> int:
    """Urgent stages jump the queue."""
    urgent = {"solve", "reply"}
    return 8 if stage_name in urgent else 3

def sjf_request_priority(prompt: str) -> int:
    """Shorter prompts get served first within a stage."""
    tokens = len(prompt.split())
    if tokens < 50:
        return 10
    if tokens < 200:
        return 5
    return 1

# Stage level: deadline-based
triage = Stage("triage", backend)
triage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
triage.set_scheduling_hints(SchedulingHints(priority=deadline_stage_priority("triage")))

solve = Stage("solve", backend)
solve.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
solve.set_scheduling_hints(SchedulingHints(priority=deadline_stage_priority("solve")))

# Request level: shortest-job-first within the solve stage
solve.set_request_scheduling_policy(REQUEST_SCHEDULING_POLICY_PRIORITY)

def solve_node(state):
    prompt = state["messages"][-1].content
    solve.set_scheduling_hints(SchedulingHints(
        priority=deadline_stage_priority("solve"),
        request_priority=sjf_request_priority(prompt),
    ))
    llm = solve.as_chat_model()
    return {"messages": [llm.invoke(state["messages"])]}
```

The stage-level policy decides that solve requests get dequeued before triage requests. The request-level policy decides that within the solve queue, shorter requests go first.

Since the policy is just a function that returns an integer, you can iterate on it quickly: run your workload, check `queue_wait_ms` in the response metrics, adjust thresholds, and run again. There is no configuration file to edit and no deployment step between experiments.

## Monitoring scheduling performance

Every response from Orla includes scheduling metrics:

```python
llm = stage.as_chat_model()
reply = llm.invoke(state["messages"])
metrics = reply.response_metadata

print(f"Queue wait:    {metrics.get('queue_wait_ms')}ms")
print(f"Decision time: {metrics.get('scheduler_decision_ms')}ms")
print(f"Dispatch time: {metrics.get('dispatch_ms')}ms")
```

`queue_wait_ms` tells you how long the request waited before a worker picked it up. If this is consistently high for a particular stage, that stage may benefit from higher priority or a dedicated backend with more concurrency.

The Orla daemon also exposes these metrics at the Prometheus `/metrics` endpoint for monitoring across your entire deployment.

