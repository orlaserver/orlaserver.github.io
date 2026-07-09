# Using a custom scheduler

By default Orla serves each backend's queue first-come-first-served. When two requests wait for the same busy backend, the one that arrived first goes first. This tutorial replaces that with an external scheduling service that orders the queue however you want. Here it orders by a priority tag, so an urgent request jumps ahead of ones already waiting.

The scheduling service only reorders a queue. It never decides which backend a request goes to, never rejects a request, and never blocks correctness. If it is slow or unreachable, Orla falls back to first-come-first-served on its own.

## Prerequisites

You need a working Orla setup from the [Quickstart](v2/quickstart.md), so Postgres running, Orla built, and at least one backend registered. You also need [uv](https://docs.astral.sh/uv/) to run the example service, and the `openai` Python client for the test agent.

## Run the example scheduling service

The Orla repo ships a tiny priority-first scheduler under `examples/scheduler_service`. It admits the queued request with the highest `priority` tag, breaking ties by the longest wait.

```bash
cd examples/scheduler_service
uv run uvicorn app:app --host 127.0.0.1 --port 8090
```

Leave it running. It answers on `http://127.0.0.1:8090/v1/schedule/next`.

## Point Orla at it

The scheduling policy is control-plane state, managed with `orlactl` the same way you register backends and map stages. Set it once. It is stored in Postgres, takes effect on every backend immediately, and survives restarts. No daemon restart is needed.

```bash
orlactl scheduler policy set --url http://127.0.0.1:8090/v1/schedule/next
orlactl scheduler policy show
```

A malformed URL is rejected. To go back to first-come-first-served at any time, run `orlactl scheduler policy disable`.

## Make ordering observable

Ordering only matters when requests queue, and requests only queue when a backend is busy. Register a backend with a concurrency of one so a single in-flight request forces the rest to wait. Map a stage to it.

```bash
orlactl backend create --name solo --endpoint http://localhost:11434/v1 \
  --model ollama:llama3.2:1b --api-key-env OLLAMA_API_KEY --max-concurrency 1

orlactl stage map reply solo
```

## Send competing requests

Each request names the `reply` stage and carries a priority through the `X-Orla-Tag-Priority` header. Orla forwards that tag to the scheduling service. This script fires several at once so they pile up behind the single slot, then prints the order they finished.

```python
import threading
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="unused")

done = []
lock = threading.Lock()

def fire(label: str, priority: int) -> None:
    client.chat.completions.create(
        model="unused",
        messages=[{"role": "user", "content": f"say {label}"}],
        extra_headers={"X-Orla-Stage": "reply", "X-Orla-Tag-Priority": str(priority)},
    )
    with lock:
        done.append(label)

threads = [
    threading.Thread(target=fire, args=("low", 1)),
    threading.Thread(target=fire, args=("high", 9)),
    threading.Thread(target=fire, args=("medium", 5)),
]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("finished order:", done)
```

The first request to arrive starts immediately, because the slot is free. The rest queue. When the slot frees, the scheduling service picks the highest priority among those waiting, so `high` is served before `medium`, and `medium` before `low`. Run it a few times. The exact head of the list depends on which request won the free slot first, but among the queued ones the priority order holds.

## Watch the decisions

Every decision is counted at the Prometheus endpoint.

```bash
curl -s http://localhost:8081/metrics | grep scheduler_policy
```

`orla_scheduler_policy_decisions_total{outcome="ok"}` counts decisions the service made. The `fallback_timeout`, `fallback_error`, and `fallback_invalid` outcomes count the times Orla had to fall back to first-come-first-served. `orla_scheduler_policy_decision_seconds` records how long the service took.

## Try the fallback

Stop the scheduling service while the agent keeps running. Orla keeps serving, now first-come-first-served, and the `fallback_error` counter climbs. Nothing errors and nothing stalls. This is the point of keeping the service advisory. A scheduler you are still tuning cannot take your traffic down.

## Write your own

The service is an HTTP endpoint that receives the pending queue and returns the id to run next. Swap the priority rule in `app.py` for anything you can compute from the request metadata Orla sends, the stage, the model, the tags, and how long each request has waited. Deadline-aware, shortest-job-first by a token estimate in a tag, or per-tenant fairness are all a few lines. The full request and response contract is in `docs/scheduling-service.md` in the Orla repo.
