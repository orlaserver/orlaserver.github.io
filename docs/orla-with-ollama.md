# Orla with Ollama

Ollama runs models locally on your machine with no GPU required. This makes it the fastest way to try Orla without any cloud credentials or Docker GPU setup.

## Install Ollama

If you do not have Ollama installed:

```bash
brew install ollama
```

Start the Ollama server and pull a model:

```bash
ollama serve &
ollama pull qwen3:0.6b
```

## Register the backend

Use `new_ollama_backend` to register Ollama with Orla. The function takes the model name and the Ollama server URL:

```python
from pyorla import Stage, new_ollama_backend, orla_runtime

with orla_runtime(quiet=True) as client:
    backend = new_ollama_backend("qwen3:0.6b", "http://127.0.0.1:11434")
    client.register_backend(backend)

    stage = Stage("classify", backend)
    stage.client = client
    stage.set_max_tokens(512)
    stage.set_temperature(0)

    model_with_tools = stage.as_chat_model().bind_tools(tools)
```

The rest of your LangGraph graph stays the same. Build the `StateGraph`, add nodes, compile, and invoke as usual.

## Using Ollama with Docker

For production or reproducible environments, run Ollama in Docker:

```bash
docker compose -f deploy/docker-compose.ollama.yaml up -d
docker compose -f deploy/docker-compose.ollama.yaml exec ollama ollama pull qwen3:0.6b
```

The Orla daemon and Ollama both run in Docker. The daemon API is available at `http://localhost:8081`.

## Multiple models

You can register multiple Ollama backends with different models. This is useful for cost-aware routing where a cheap model handles simple tasks and a larger model handles complex ones:

```python
light = new_ollama_backend("qwen3:0.6b", "http://127.0.0.1:11434")
heavy = new_ollama_backend("qwen3:8b", "http://127.0.0.1:11434")
client.register_backend(light)
client.register_backend(heavy)

triage = Stage("triage", light)
solve = Stage("solve", heavy)
```

## Tool calling

Not all Ollama models support tool calling. If your graph uses tools, choose a model that supports them. The Qwen3 family works well for this. If tool calls are not working, check that the model you pulled supports the tool calling format LangGraph expects.
