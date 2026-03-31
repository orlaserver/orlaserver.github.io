# Orla with vLLM

vLLM is a high-throughput inference engine that runs on NVIDIA GPUs. It exposes an OpenAI-compatible API, which Orla connects to directly.

## Prerequisites

You need an NVIDIA GPU with drivers installed and either Docker with GPU support or a local vLLM installation. See the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for Docker GPU setup.

## Start vLLM with Docker

The Orla repo includes a Docker Compose file that runs vLLM with Qwen3-4B by default:

```bash
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

To use a different model:

```bash
export VLLM_MODEL=Qwen/Qwen3-8B
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

Wait for vLLM to finish loading the model. You can check with:

```bash
curl -s http://127.0.0.1:8000/health
```

## Register the backend

Use `new_vllm_backend` to register vLLM with Orla. The function takes the model ID and the vLLM server URL:

```python
from pyorla import Stage, new_vllm_backend, orla_runtime

with orla_runtime(quiet=True) as client:
    backend = new_vllm_backend(
        "Qwen/Qwen3-4B-Instruct-2507",
        "http://127.0.0.1:8000/v1",
    )
    client.register_backend(backend)

    stage = Stage("solve", backend)
    stage.client = client
    stage.set_max_tokens(512)
    stage.set_temperature(0)

    model_with_tools = stage.as_chat_model().bind_tools(tools)
```

The `model_id` must match the model vLLM is serving. If you changed `VLLM_MODEL`, update the first argument to `new_vllm_backend` accordingly.

## Tool calling

vLLM needs tool calling enabled in its server flags for LangGraph tool use to work. The Orla Docker Compose file configures this automatically. If you are running vLLM manually, add:

```bash
--enable-auto-tool-choice --tool-call-parser hermes
```

The tool call parser depends on your model family. `hermes` works with Qwen models. Check the [vLLM documentation](https://docs.vllm.ai/en/latest/) for other model families.

## Multiple models on separate GPUs

Run multiple vLLM instances on different ports, each serving a different model. Register each as a separate backend:

```python
light = new_vllm_backend("Qwen/Qwen3-4B-Instruct-2507", "http://127.0.0.1:8000/v1")
heavy = new_vllm_backend("Qwen/Qwen3-32B", "http://127.0.0.1:8001/v1")
client.register_backend(light)
client.register_backend(heavy)
```

Combine this with Orla's cost policies to route cheap stages to the small model and expensive stages to the large one.

## Endpoint note

When registering vLLM, use the URL that Orla can reach. If both Orla and vLLM run on the same host, `http://127.0.0.1:8000/v1` works. If vLLM runs inside Docker and Orla runs on the host, use the exposed port. If both are in Docker, use the Docker service name.
