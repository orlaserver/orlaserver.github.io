# Tutorial: Run a simple agent with Orla and SGLang

This tutorial runs a single agent using [Orla](https://github.com/dorcha-inc/orla) and [SGLang](https://sgl-project.github.io/) via Docker Compose. The agent has one job: tell a short story about a cat called Lily.

## What you need

- Docker and Docker Compose (Compose V2 plugin).
- An NVIDIA GPU with drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so the SGLang container can use the GPU.
- The Orla repo cloned so you can run the deploy compose file from the repo root.
- Go 1.25 or later (for the main tutorial; the appendix shows a curl-based alternative).

The steps below assume Linux. For other platforms, see Docker’s [install docs](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) guide for your OS.

## Installing prerequisites (Linux)

### 1. Docker Engine and Docker Compose

Install Docker Engine and the Compose V2 plugin. On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
```

Add your user to the `docker` group, then log out and back in:

```bash
sudo usermod -aG docker $USER
```

Verify Docker and Compose:

```bash
docker --version
docker compose version
```

### 2. NVIDIA drivers and Container Toolkit

Install the proprietary NVIDIA driver for your GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so Docker containers can use the GPU. After installation:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify that a container can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

### 3. Orla repository

Clone the Orla repo (if you haven’t already):

```bash
git clone https://github.com/dorcha-inc/orla.git
cd orla
```

### 4. Go

You need Go 1.25 or later. Install or upgrade using the [official install guide](https://go.dev/doc/install), then run `go version` to confirm.

## 1. Start SGLang and Orla

From the root of the Orla repository:

```bash
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

This starts:

- **SGLang** on port 30000 (Ollama-compatible API). Default model: `Qwen/Qwen3-8B`. The Orla server (in its container) reaches it at `http://sglang:30000`.
- **Orla** on port 8081. The Orla server starts with no LLM backends; the Go program in step 3 registers the SGLang backend and runs the request.

### Optional

You can use a different model or set the Hugging Face token:

```bash
export SGLANG_MODEL=Qwen/Qwen3-4B-Instruct-2507
export HF_TOKEN=your_token   # only for gated models
docker compose -f deploy/docker-compose.sglang.yaml up
```

If you change the model, ensure the Orla backend’s model identifier matches what SGLang serves (e.g. `ollama:Qwen/Qwen3-8B` for the default).

## 2. Check that the daemon is up

```bash
curl http://localhost:8081/api/v1/health
```

You should get a successful response (e.g. HTTP 200).

## 3. Run the “Lily” story request

The Orla API exposes **`POST /api/v1/execute`**: you send a `backend` name (a backend you registered), a `prompt` (or `messages`), and optional `max_tokens` and `stream`. The response contains the model output in `response.content`.

### Using the Go client and Agent API

Orla talks to SGLang using the **Ollama-compatible** API (type `"ollama"`). Create a file `main.go` in the Orla repo root:

```go
package main

import (
	"context"
	"fmt"
	"log"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

func main() {
	client := orla.NewOrlaClient("http://localhost:8081")
	ctx := context.Background()

	// Register the SGLang backend. Orla uses the Ollama-compatible API to talk to SGLang.
	// When Orla and SGLang run in Docker Compose, use the service name "sglang".
	backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
		Name:     "sglang",
		Endpoint: "http://sglang:30000",
		Type:     "ollama",
		ModelID:  "ollama:Qwen/Qwen3-8B",
	})
	if err != nil {
		log.Fatal("register backend: ", err)
	}

	agent := orla.NewAgent(client, backend)
	agent.SetMaxTokens(512)

	prompt := "Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough."
	resp, err := agent.Execute(ctx, prompt)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Content)
}
```

From the repo root, run:

```bash
go run .
```

You should see the model’s story about Lily printed to the terminal.

### Streamed output

To see the story as it is generated, use the agent’s **ExecuteStream** and **ConsumeStream**:

```go
package main

import (
	"context"
	"fmt"
	"log"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

func main() {
	client := orla.NewOrlaClient("http://localhost:8081")
	ctx := context.Background()

	backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
		Name:     "sglang",
		Endpoint: "http://sglang:30000",
		Type:     "ollama",
		ModelID:  "ollama:Qwen/Qwen3-8B",
	})
	if err != nil {
		log.Fatal("register backend: ", err)
	}

	agent := orla.NewAgent(client, backend)
	agent.SetMaxTokens(512)

	prompt := "Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough."
	stream, err := agent.ExecuteStream(ctx, prompt)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := agent.ConsumeStream(ctx, stream, func(ev orla.StreamEvent) error {
		if ev.Type == "content" {
			fmt.Print(ev.Content)
		}
		if ev.Type == "thinking" {
			fmt.Print(ev.Thinking)
		}
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	if resp.Metrics != nil {
		fmt.Printf("\n\n[TTFT: %d ms, TPOT: %d ms]\n", resp.Metrics.TTFTMs, resp.Metrics.TPOTMs)
	}
}
```

Run with `go run .`. Text appears incrementally; ConsumeStream returns the full InferenceResponse when the stream finishes.

## 4. Stop the stack

When you’re done:

```bash
docker compose -f deploy/docker-compose.sglang.yaml down
```

Use `down -v` if you also want to remove the SGLang model cache volume.

You’ve registered a backend via the API, sent a prompt to the Orla server, and received a story about Lily from SGLang. For tool calling with SGLang, see [Tool calling with Orla (vLLM, Ollama, SGLang)](tutorial-tools-vllm-ollama-sglang.md).

## Appendix: Running the request with curl

You can skip the Go client and use only `curl`. First register the SGLang backend, then call the execute endpoint.

### Register the backend (curl)

```bash
curl -X POST http://localhost:8081/api/v1/backends \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sglang",
    "endpoint": "http://sglang:30000",
    "type": "ollama",
    "model_id": "ollama:Qwen/Qwen3-8B",
    "api_key_env_var": ""
  }'
```

You should get `{"success":true}`. List backends: `curl -s http://localhost:8081/api/v1/backends`

### Execute with curl

Install `jq` if needed: `sudo apt-get install -y jq`.

```bash
curl -X POST http://localhost:8081/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "sglang",
    "prompt": "Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough.",
    "max_tokens": 512,
    "stream": false
  }' | jq .
```

To print only the story text:

```bash
curl -X POST http://localhost:8081/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"backend": "sglang", "prompt": "Tell me a short story about a cat called Lily.", "max_tokens": 512}' \
  | jq -r '.response.content'
```

Response shape: `{"success": true, "response": {"content": "...", "thinking": "", ...}}` or `{"success": false, "error": "..."}`.
