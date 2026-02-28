# Tutorial: Run a simple agent with Orla and Ollama

This tutorial runs a single agent using [Orla](https://github.com/dorcha-inc/orla) and [Ollama](https://ollama.com/) via Docker Compose. The agent has one job: tell a short story about a cat called Lily.

## What you need

- Docker and Docker Compose (Compose V2 plugin).
- The Orla repo cloned so you can run the deploy compose file from the repo root.
- Go 1.25 or later (for the main tutorial; the appendix shows a curl-based alternative).

Ollama can run on CPU or GPU. No NVIDIA Container Toolkit is required for CPU-only.

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

Verify:

```bash
docker --version
docker compose version
```

### 2. Orla repository

Clone the Orla repo (if you haven’t already):

```bash
git clone https://github.com/dorcha-inc/orla.git
cd orla
```

### 3. Go

You need Go 1.25 or later. Install or upgrade using the [official install guide](https://go.dev/doc/install), then run `go version` to confirm.

## 1. Start Ollama and Orla

From the root of the Orla repository:

```bash
docker compose -f deploy/docker-compose.ollama.yaml up -d
```

This starts:

- **Ollama** on port 11434 (API at `http://localhost:11434`). The Orla server (in its container) reaches it at `http://ollama:11434`.
- **Orla** on port 8081. The Orla server starts with no LLM backends; the Go program in step 3 registers the Ollama backend and runs the request.

Pull a model inside the Ollama container (e.g. Llama 3.2 3B):

```bash
docker compose -f deploy/docker-compose.ollama.yaml exec ollama ollama pull llama3.2:3b
```

If you use a different model, ensure the Go program’s `model_id` matches (e.g. `ollama:llama3.2:3b`).

## 2. Check that the daemon is up

```bash
curl http://localhost:8081/api/v1/health
```

You should get a successful response (e.g. HTTP 200).

## 3. Run the “Lily” story request

The Orla API exposes **`POST /api/v1/execute`**: you send a `backend` name (a backend you registered), a `prompt` (or `messages`), and optional `max_tokens` and `stream`. The response contains the model output in `response.content`.

### Using the Go client and Agent API

Create a file `main.go` in the Orla repo root. The program registers the Ollama backend, creates an agent with the prompt, and runs the execute request:

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

	// Register the Ollama backend. When Orla and Ollama run in Docker Compose, use the
	// service name "ollama" so the Orla server (in its container) can reach the Ollama container.
	backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
		Name:     "ollama",
		Endpoint: "http://ollama:11434",
		Type:     "ollama",
		ModelID:  "ollama:llama3.2:3b",
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
		Name:     "ollama",
		Endpoint: "http://ollama:11434",
		Type:     "ollama",
		ModelID:  "ollama:llama3.2:3b",
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
docker compose -f deploy/docker-compose.ollama.yaml down
```

Use `down -v` if you also want to remove the Ollama data volume.

You’ve registered a backend via the API, sent a prompt to the Orla server, and received a story about Lily from Ollama. For tool calling with Ollama, see [Tool calling with Orla (vLLM, Ollama, SGLang)](tutorial-tools-vllm-ollama-sglang.md).

## Appendix: Running the request with curl

You can skip the Go client and use only `curl`. First register the Ollama backend, then call the execute endpoint.

### Register the backend (curl)

```bash
curl -X POST http://localhost:8081/api/v1/backends \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama",
    "endpoint": "http://ollama:11434",
    "type": "ollama",
    "model_id": "ollama:llama3.2:3b",
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
    "backend": "ollama",
    "prompt": "Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough.",
    "max_tokens": 512,
    "stream": false
  }' | jq .
```

To print only the story text:

```bash
curl -X POST http://localhost:8081/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"backend": "ollama", "prompt": "Tell me a short story about a cat called Lily.", "max_tokens": 512}' \
  | jq -r '.response.content'
```

Response shape: `{"success": true, "response": {"content": "...", "thinking": "", ...}}` or `{"success": false, "error": "..."}`.
