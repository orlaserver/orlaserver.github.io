# Docker Deployment

The Orla repo includes Docker Compose files for running the daemon alongside LLM backends. This is the recommended setup for production or reproducible environments.

## Prerequisites

- Docker and Docker Compose
- For vLLM and SGLang: NVIDIA GPU with drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- For Ollama: optional GPU for faster inference, runs on CPU otherwise

## Quick start

### Ollama

```bash
docker compose -f deploy/docker-compose.ollama.yaml up -d

# Pull a model
docker compose -f deploy/docker-compose.ollama.yaml exec ollama ollama pull qwen3:0.6b
```

### vLLM

Defaults to Qwen3-4B. Override with `VLLM_MODEL`:

```bash
docker compose -f deploy/docker-compose.vllm.yaml up -d

# Or with a different model:
VLLM_MODEL=Qwen/Qwen3-8B docker compose -f deploy/docker-compose.vllm.yaml up -d
```

### SGLang

Defaults to Qwen3-8B. Override with `SGLANG_MODEL`:

```bash
docker compose -f deploy/docker-compose.sglang.yaml up -d

# For gated models:
HF_TOKEN=your_token docker compose -f deploy/docker-compose.sglang.yaml up -d
```

All three setups expose the Orla daemon API at `http://localhost:8081`.

## Multi-backend stacks

For workflows that use both a light and heavy model, the repo includes multi-backend Compose files:

```bash
# Two SGLang instances (light + heavy)
docker compose -f deploy/docker-compose.workflow-demo.yaml up -d

# Two vLLM instances (light + heavy)
docker compose -f deploy/docker-compose.workflow-demo.vllm.yaml up -d
```

Two GPUs are recommended. With one GPU, run the two model containers on different devices or one at a time.

## Configuration

Each backend has a corresponding config file:

| Backend | Compose file | Config file |
|---------|-------------|-------------|
| Ollama | `docker-compose.ollama.yaml` | `orla-ollama.yaml` |
| vLLM | `docker-compose.vllm.yaml` | `orla-vllm.yaml` |
| SGLang | `docker-compose.sglang.yaml` | `orla-sglang.yaml` |

Edit the `orla-*.yaml` file to change backend endpoints, model names, and inference options. Restart the Orla service to reload:

```bash
docker compose -f deploy/docker-compose.ollama.yaml restart orla
```

## Building the Orla image

The Compose files build the daemon image from the repo root Dockerfile:

```bash
docker compose -f deploy/docker-compose.ollama.yaml build orla
```

Or build once and reuse:

```bash
docker build -t orla:latest .
```

## Connecting from Python

When both your Python code and the Orla daemon run on the host:

```python
from pyorla import OrlaClient

client = OrlaClient("http://localhost:8081")
```

When your Python code runs on the host and the backend runs in Docker, register the backend with the URL that the Orla container can reach. If both Orla and the backend are in the same Docker network, use the service name:

```python
backend = new_vllm_backend("Qwen/Qwen3-4B", "http://vllm:8000/v1")
```

If Orla runs on the host and the backend is in Docker, use the exposed port:

```python
backend = new_vllm_backend("Qwen/Qwen3-4B", "http://127.0.0.1:8000/v1")
```

## GPU setup on Linux

Install Docker Engine and the NVIDIA Container Toolkit:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
