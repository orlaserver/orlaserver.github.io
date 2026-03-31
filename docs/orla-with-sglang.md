# Orla with SGLang

SGLang is an inference engine optimized for structured generation and large batch throughput on NVIDIA GPUs. It exposes an OpenAI-compatible API and supports KV cache management, which Orla's memory manager can use for cache-aware scheduling.

## Prerequisites

You need an NVIDIA GPU with drivers installed and Docker with GPU support. See the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Start SGLang with Docker

The Orla repo includes a Docker Compose file that runs SGLang with Qwen3-8B by default:

```bash
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

To use a different model:

```bash
export SGLANG_MODEL=Qwen/Qwen3-32B
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

For gated models that require authentication:

```bash
export HF_TOKEN=your_token
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

## Register the backend

Use `new_sglang_backend` to register SGLang with Orla:

```python
from pyorla import Stage, new_sglang_backend, orla_runtime

with orla_runtime(quiet=True) as client:
    backend = new_sglang_backend(
        "Qwen/Qwen3-8B",
        "http://127.0.0.1:30000/v1",
    )
    client.register_backend(backend)

    stage = Stage("solve", backend)
    stage.client = client
    stage.set_max_tokens(512)
    stage.set_temperature(0)

    model_with_tools = stage.as_chat_model().bind_tools(tools)
```

## KV cache integration

SGLang is the only backend that currently supports hard cache flush from Orla's memory manager. When a workflow completes and the memory manager decides to flush, it sends a request to SGLang's `/flush_cache` endpoint. This frees KV cache memory for other workflows.

This happens automatically when you use the `Workflow` class or call `client.workflow_complete()` after a LangGraph run. See [Using the Memory Manager](memory-manager.md) for details.

## Memory pressure monitoring

When running with SGLang, Orla's background pressure monitor polls SGLang's `/get_server_info` endpoint to check KV cache usage. If cache usage exceeds 85%, the memory manager flushes the oldest idle workflow to free space. This prevents a single workflow from monopolizing cache memory on a shared backend.

## Multiple models

Run multiple SGLang instances on different ports:

```python
light = new_sglang_backend("Qwen/Qwen3-4B", "http://127.0.0.1:30000/v1")
heavy = new_sglang_backend("Qwen/Qwen3-32B", "http://127.0.0.1:30001/v1")
client.register_backend(light)
client.register_backend(heavy)
```

The multi-backend Docker Compose files in `deploy/` set this up for you:

```bash
docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
```

This starts two SGLang instances, one with a light model and one with a heavy model, along with the Orla daemon.
