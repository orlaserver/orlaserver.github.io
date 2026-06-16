# Quickstart

This quickstart builds a two-stage agent on two local Ollama models, routes it through Orla, then re-routes a stage live without touching the agent.

## Prerequisites

You need Postgres running locally, [Ollama](https://ollama.com), and Go 1.26 or newer to build Orla.

## Pull two small models

Pull two models small enough to run on a laptop. Ollama serves them on an OpenAI-compatible endpoint at `http://localhost:11434/v1`.

```bash
ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
```

## Run Orla

Create a database, install the daemon and the `orlactl` control-plane CLI, then start the daemon. `go install` puts `orla` and `orlactl` on your PATH, under `$(go env GOPATH)/bin`. Orla runs its migrations on first start and listens on `localhost:8081`. Leave it running and open a second terminal for the rest of the steps.

```bash
createdb orla
git clone https://github.com/harvard-cns/orla
cd orla
go install ./cmd/orla ./cmd/orlactl
ORLA_DATABASE_URL=postgres://localhost/orla?sslmode=disable orla serve
```

## Register the backends

A backend is one inference endpoint. Register both Ollama models with `orlactl`. The model id is written as `ollama:<model>`, where the part before the first colon is a free label and the rest is the model name Orla passes to the endpoint. Ollama needs no API key, so `--api-key-env` can name a variable you never set.

```bash
orlactl backend create --name qwen-05b --endpoint http://localhost:11434/v1 \
  --model ollama:qwen2.5:0.5b --api-key-env OLLAMA_API_KEY --max-concurrency 2

orlactl backend create --name llama-1b --endpoint http://localhost:11434/v1 \
  --model ollama:llama3.2:1b --api-key-env OLLAMA_API_KEY --max-concurrency 2
```

## Map stages to backends

A stage is a label for what a call is doing. The agent here has two stages, `plan` and `answer`. Map each one to a backend so Orla knows where to send it.

```bash
orlactl stage map plan   qwen-05b
orlactl stage map answer llama-1b
```

## Write the agent

The agent uses the standard OpenAI client pointed at Orla. It names a stage on each call, never a model. Install the client first.

```bash
pip install openai
```

```python
from openai import OpenAI

# Point the OpenAI client at Orla. Ollama needs no real key.
client = OpenAI(base_url="http://localhost:8081/v1", api_key="orla")

def call(stage: str, prompt: str) -> str:
    resp = client.chat.completions.create(
        model="set-by-orla",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=80,
        extra_headers={"X-Orla-Stage": stage},
    )
    print(f"[{stage}] served by {resp.model}")
    return resp.choices[0].message.content

question = "Should I water a cactus every day?"
plan = call("plan", f"List two things to consider when answering: {question}")
answer = call("answer", f"Question: {question}\nNotes: {plan}\nAnswer in one sentence.")
print("\n" + answer.strip())
```

Running it sends the `plan` call to `qwen-05b` and the `answer` call to `llama-1b`. The `model` field on each response is the backend Orla chose.

```
[plan] served by qwen-05b
[answer] served by llama-1b
```

## Re-route a stage without touching the agent

This is the point of the control plane. Send the planning stage to the larger model with a single call.

```bash
orlactl stage map plan llama-1b
```

Run the agent again. The `plan` call is now served by `llama-1b`, and the agent code did not change. Put it back when you are done.

```bash
orlactl stage map plan qwen-05b
```

## Inspect mappings and report outcomes

You can list the current stage mappings at any time.

```bash
orlactl stage ls
```

After a call completes, you report a rating for the stage it used. A mapper process reads these ratings and re-maps stages to better backends over time. In production your agent posts this from code, but you can also do it by hand with the completion id from the response.

```bash
orlactl feedback <completion-id> --stage answer --rating 1.0
```

## Takeaways

Your agent's routing now lives outside its code. The same loop scales up. You register more backends, split the workflow into more stages, and let a mapper tune each stage to the cheapest model that holds quality.
