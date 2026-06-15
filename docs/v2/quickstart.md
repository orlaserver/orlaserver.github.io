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

Create a database, build the binary, and start the daemon. Orla runs its migrations on first start and listens on `localhost:8081`. Leave it running and open a second terminal for the rest of the steps.

```bash
createdb orla
git clone https://github.com/harvard-cns/orla
cd orla
go build -o bin/orla ./cmd/orla
ORLA_DATABASE_URL=postgres://localhost/orla?sslmode=disable ./bin/orla serve
```

## Register the backends

A backend is one inference endpoint. Register both Ollama models through the control plane. The `model_id` is written as `ollama:<model>`, where the part before the first colon is a free label and the rest is the model name Orla passes to the endpoint. Ollama needs no API key, so `api_key_env_var` can name a variable you never set.

```bash
curl -X POST localhost:8081/api/v1/backends -H 'Content-Type: application/json' -d '{
  "name": "qwen-05b",
  "endpoint": "http://localhost:11434/v1",
  "model_id": "ollama:qwen2.5:0.5b",
  "api_key_env_var": "OLLAMA_API_KEY",
  "max_concurrency": 2
}'

curl -X POST localhost:8081/api/v1/backends -H 'Content-Type: application/json' -d '{
  "name": "llama-1b",
  "endpoint": "http://localhost:11434/v1",
  "model_id": "ollama:llama3.2:1b",
  "api_key_env_var": "OLLAMA_API_KEY",
  "max_concurrency": 2
}'
```

## Map stages to backends

A stage is a label for what a call is doing. The agent here has two stages, `plan` and `answer`. Map each one to a backend so Orla knows where to send it.

```bash
curl -X PUT localhost:8081/api/v1/stages/plan   -H 'Content-Type: application/json' -d '{"backend": "qwen-05b"}'
curl -X PUT localhost:8081/api/v1/stages/answer -H 'Content-Type: application/json' -d '{"backend": "llama-1b"}'
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
curl -X PATCH localhost:8081/api/v1/stages/plan -H 'Content-Type: application/json' -d '{"backend": "llama-1b"}'
```

Run the agent again. The `plan` call is now served by `llama-1b`, and the agent code did not change. Put it back when you are done.

```bash
curl -X PATCH localhost:8081/api/v1/stages/plan -H 'Content-Type: application/json' -d '{"backend": "qwen-05b"}'
```

## Inspect mappings and report outcomes

You can list the current stage mappings at any time.

```bash
curl localhost:8081/api/v1/stages
```

After a call completes, your agent can post a rating for the stage it used. A mapper process reads these ratings and re-maps stages to better backends over time. Attach the rating to the completion id from the response.

```bash
curl -X POST localhost:8081/v1/feedback -H 'Content-Type: application/json' -d '{
  "completion_id": "<id from the response>",
  "stage_id": "answer",
  "rating": 1.0
}'
```

## Takeaways

Your agent's routing now lives outside its code. The same loop scales up. You register more backends, split the workflow into more stages, and let a mapper tune each stage to the cheapest model that holds quality.
