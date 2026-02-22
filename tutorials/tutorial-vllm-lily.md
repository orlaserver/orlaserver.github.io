# Tutorial: Run a simple agent with Orla and vLLM

This tutorial runs a single agent using [Orla](https://github.com/dorcha-inc/orla) and [vLLM](https://docs.vllm.ai/) via Docker Compose. The agent has one job: tell a short story about a cat called Lily.

## What you need

- Docker and Docker Compose (Compose V2 plugin).
- An NVIDIA GPU with drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so the vLLM container can use the GPU.
- The Orla repo cloned so you can run the deploy compose file from the repo root.
- Go 1.25 or later (for the main tutorial; the appendix shows a curl-based alternative).

The steps below assume Linux. For other platforms, see Docker’s [install docs](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) guide for your OS.

## Installing prerequisites (Linux)

### 1. Docker Engine and Docker Compose

Install Docker Engine and the Compose V2 plugin. On Ubuntu or Debian you can use the distro packages (no extra repository):

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
```

If you prefer the latest Docker release from Docker’s official repo, see [Install Docker Engine](https://docs.docker.com/engine/install/) for your distribution.

Add your user to the `docker` group so you can run Docker without `sudo`, then log out and back in:

```bash
sudo usermod -aG docker $USER
```

Verify Docker and Compose:

```bash
docker --version
docker compose version
```

### 2. NVIDIA drivers

Install the proprietary NVIDIA driver for your GPU. On Ubuntu, you can use the default package manager or the [NVIDIA driver download](https://www.nvidia.com/drivers) page. Example (Ubuntu, generic):

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535   # or newer; check ubuntu-drivers list
sudo reboot
```

After reboot, confirm the GPU is visible:

```bash
nvidia-smi
```

### 3. NVIDIA Container Toolkit

This lets Docker containers use the GPU. See the [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for your distro. Ubuntu/Debian (after adding NVIDIA’s package repository as in the guide):

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify that a container can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

You should see the same GPU info as on the host.

### 4. Orla repository

Clone the Orla repo (if you haven’t already):

```bash
git clone https://github.com/dorcha-inc/orla.git
cd orla
```

### 5. Go

The main tutorial uses the Orla Go client. You need Go 1.25 or later. Install or upgrade using the [official install guide](https://go.dev/doc/install), then run `go version` to confirm.

## 1. Start vLLM and Orla

From the root of the Orla repository:

```bash
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

Side note: remove the `-d` if you want to see the streamed output.

This starts:

- vLLM on port 8000 (OpenAI-compatible API), serving the default model `Qwen/Qwen3-4B-Instruct-2507`.
- Orla on port 8081. The Orla server starts with no LLM backends; the Go program in step 3 registers the vLLM backend and runs the request. If you use the curl-only flow (appendix), register the backend with curl first.

### Optional

You can use a different model or set the Hugging Face token

```bash
export VLLM_MODEL=Qwen/Qwen3-8B
export HF_TOKEN=your_token   # only for gated models
docker compose -f deploy/docker-compose.vllm.yaml up
```

If you change the model, ensure the Orla backend’s model identifier matches what vLLM serves (e.g. `openai:Qwen/Qwen3-4B-Instruct-2507`).

## 2. Check that the daemon is up

```bash
curl http://localhost:8081/api/v1/health
```

You should get a successful response (e.g. HTTP 200 with `{"status":"healthy"}`).

## 3. Run the “Lily” story request

The Orla API exposes **`POST /api/v1/execute`**: you send a `backend` name (a backend you registered), a `prompt` (or `messages`), and optional `max_tokens` and `stream`. The response contains the model output in `response.content`.

### Using the Go client

Create a file `main.go` in the Orla repo root. The program registers the vLLM backend, then runs the execute request:

```go
package main

import (
	"context"
	"fmt"
	"log"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

func main() {
	client := orla.NewClient("http://localhost:8081")
	ctx := context.Background()

	// Register the vLLM backend. When Orla and vLLM run in Docker Compose, use the
	// service name "vllm" so the Orla server (in its container) can reach the vLLM container.
	_, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
		Name:     "vllm",
		Endpoint: "http://vllm:8000/v1",
		Type:     "openai",
		ModelID:  "openai:Qwen/Qwen3-4B-Instruct-2507",
	})

	if err != nil {
		log.Fatal("register backend: ", err)
	}

	resp, err := client.Execute(ctx, &orla.ExecuteRequest{
		Backend:   "vllm",
		Prompt:    "Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough.",
		MaxTokens: 512,
		Stream:    false,
	})
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

You should see the model’s story about Lily printed to the terminal. Example output:

```
Lily the cat lived in a sun-dappled cottage at the edge of a meadow, where daisies waved in the breeze and butterflies danced like tiny rainbows. Every morning, she would stretch like a furry sunbeam, then leap onto the windowsill to watch the world wake up. Her favorite pastime? Tiptoeing through the garden to chase a fluttering butterfly, only to gently nudge it back into the daisy patch with a soft purr. The neighbors often chuckled and said, “Lily has the most joyful spirit in all the village!”

One bright afternoon, a tiny blue bird with a broken wing landed near her window, trembling and unable to fly. Lily didn’t just watch—she sat beside it, her tail wrapped gently around the bird’s side, and began humming a soft, melodic tune. The bird, startled but calm, slowly leaned into her warmth. With a little help from a kind neighbor, they found a patch of healing herbs, and by sunset, the bird could flutter again. As it soared into the sky, it left a trail of golden light—and Lily, beaming with pride, curled up in the grass, dreaming of more sunny adventures. After all, happiness, she thought, is just a purr away. 🌞🐾
```

## 4. Stop the stack

When you’re done:

```bash
docker compose -f deploy/docker-compose.vllm.yaml down
```

Use `down -v` if you also want to remove the vLLM model cache volume.

You’ve registered a backend via the API, sent a prompt to the Orla server, and received a story about Lily from vLLM. For more on the execute and backend-registration APIs, configuration, and deployment, see the [Orla repo](https://github.com/dorcha-inc/orla) and the [deploy README](https://github.com/dorcha-inc/orla/blob/main/deploy/README.md).

## Appendix: Running the request with curl

You can skip the Go client and use only `curl`. First register the vLLM backend, then call the execute endpoint.

### Register the backend (curl)

Orla does not read backends from config. Register the vLLM backend with **`POST /api/v1/backends`**. When Orla and vLLM run in Docker Compose, use endpoint **`http://vllm:8000/v1`** (the Compose service name) so the Orla server container can reach the vLLM container:

```bash
curl -X POST http://localhost:8081/api/v1/backends \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm",
    "endpoint": "http://vllm:8000/v1",
    "type": "openai",
    "model_id": "openai:Qwen/Qwen3-4B-Instruct-2507",
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
    "backend": "vllm",
    "prompt": "Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough.",
    "max_tokens": 512,
    "stream": false
  }' | jq .
```

To print only the story text:

```bash
curl -X POST http://localhost:8081/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"backend": "vllm", "prompt": "Tell me a short story about a cat called Lily.", "max_tokens": 512}' \
  | jq -r '.response.content'
```

Response shape: `{"success": true, "response": {"content": "...", "thinking": "", ...}}` or `{"success": false, "error": "..."}`.
