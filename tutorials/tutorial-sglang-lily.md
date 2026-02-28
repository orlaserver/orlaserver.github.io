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

- **SGLang** on port 30000 (OpenAI- and Ollama-compatible APIs). Default model: `Qwen/Qwen3-8B`. The Orla server (in its container) reaches it at `http://sglang:30000`.
- **Orla** on port 8081. The Orla server starts with no LLM backends; the Go program in step 3 registers the SGLang backend and runs the request.

### Optional

You can use a different model or set the Hugging Face token:

```bash
export SGLANG_MODEL=Qwen/Qwen3-4B-Instruct-2507
export HF_TOKEN=your_token   # only for gated models
docker compose -f deploy/docker-compose.sglang.yaml up
```

If you change the model, use the same model name in the Go backend (e.g. `Qwen/Qwen3-8B` for the default).

## 2. Check that the daemon is up

```bash
curl http://localhost:8081/api/v1/health
```

You should get a successful response (e.g. HTTP 200).

## 3. Run the “Lily” story request

The Orla API exposes **`POST /api/v1/execute`**: you send a `backend` name (a backend you registered), a `prompt` (or `messages`), and optional `max_tokens` and `stream`. The response contains the model output in `response.content`.

### Using the Go client and Agent API

Orla talks to SGLang using the **OpenAI-compatible** API (`/v1/chat/completions`) so you get TTFT/TPOT metrics in streaming responses. Use `orla.NewSGLangBackend` to create a backend with the correct endpoint and model ID. Create a file `main.go` in the Orla repo root:

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

	// Register the SGLang backend. When Orla and SGLang run in Docker Compose, use the service name "sglang".
	backend := orla.NewSGLangBackend("Qwen/Qwen3-8B", "http://sglang:30000/v1")
	if err := client.RegisterBackend(ctx, backend); err != nil {
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

```bash
<think>
Okay, the user wants a short, cheerful story about a cat named Lily. Let me start by thinking about the key elements. The story should be two or three paragraphs, so I need to keep it concise but engaging.

First, I should introduce Lily. Maybe give her some characteristics that make her cheerful. Perhaps she's playful and curious. Setting is important—maybe a cozy home with a garden. That way, there's a nice environment for her to explore.

Next, I need a simple plot. Maybe she does something cute, like chasing a butterfly or interacting with other animals. Including a positive outcome would keep the story cheerful. Maybe she helps a friend or finds something delightful. 

I should make sure the language is warm and upbeat. Use words that evoke happiness, like "gleamed," "whiskers twitched," "sunlight danced." Maybe include some sensory details to make it vivid. 

Wait, the user mentioned two or three paragraphs. Let me outline: first paragraph introduces Lily and her daily activities. Second paragraph could be an incident where she does something special, maybe helping a bird or another animal. Third paragraph could wrap up with her contentment and the positive effect on her surroundings. 

I need to check for flow and ensure each paragraph transitions smoothly. Also, make sure the story has a satisfying ending. Maybe end with her sleeping contentedly, showing she's happy. Avoid any sad elements. Keep the tone light and joyful throughout. 

Let me start drafting. First paragraph: Lily in her garden, playful, chasing things. Second paragraph: she helps a bird, showing her kindness. Third paragraph: the garden thrives because of her, and she's happy. That should work. Now, check the word count and make sure it's not too long. Use simple sentences to keep it accessible. Alright, that should do it.
</think>

Lily was a sprightly tabby with emerald eyes that gleamed like polished gems. Every morning, she’d leap from her sun-warmed windowsill, tail flicking with mischief, and race through the garden where tulips swayed in the breeze. She’d chase fireflies at dusk, pounce on floating dandelion seeds, and nap in the shade of the old oak tree, her purr a soft melody that seemed to charm the very flowers. Though small, Lily had a knack for making the world feel brighter, especially when she’d curl up beside the mailman’s daughter, teaching her to tickle the feathers of a curious
```

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

	backend := orla.NewSGLangBackend("Qwen/Qwen3-8B", "http://sglang:30000/v1")
	if err := client.RegisterBackend(ctx, backend); err != nil {
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

```bash
<think>
Okay, the user wants a short, cheerful story about a cat named Lily. Let me start by thinking about the key elements. The story should be two or three paragraphs, so I need to keep it concise but engaging.

First, I should introduce Lily. Maybe give her some characteristics that make her cheerful. Maybe she's curious or loves adventures. Then, think of a simple plot. Perhaps she does something playful, like chasing a butterfly or exploring the garden. Including some positive interactions with other animals or humans would add to the cheerfulness.

I need to make sure the tone is upbeat. Use vivid descriptions to paint a happy scene. Maybe include some sensory details like the smell of flowers or the sound of birds. Also, think about a small conflict or challenge that Lily overcomes, but keep it light. For example
, she might help a friend or find something special. Ending on a positive note will keep the story cheerful.

Wait, the user mentioned two or three paragraphs. Let me outline: first paragraph introduces Lily and her environment. Second paragraph
 could be her adventure or interaction. Third paragraph might be the resolution or a happy ending. But maybe two paragraphs are enough if I combine some elements. Let me check the example response they provided earlier. Oh, right, the example had two paragraphs. So maybe stick to that structure.

Also, ensure the name Lily is prominent. Maybe give her some unique traits, like a favorite toy or a special ability. Avoid making it too complex. Keep the language simple and warm. Use words that evoke happiness, like "sunlight," "laughter," "paws," "butterflies." Maybe include a friendly animal friend to add depth. Alright, let me start drafting.
</think>

Lily the cat lived in a sunlit cottage where the windows always seemed to wink at her. Her fur was the color of honey, and her tail twitched with curiosity as she prowled the garden, chasing fireflies at dusk. Every morning, she’d leap onto the porch swing, purring as the breeze tousled her ears, and greet the day with a yowl that sounded more like a cheerful song than a cat’s call. The neighbors often
 smiled at her antics, especially when she’d sneak up behind the mailman and nudge his hat with her nose, leaving him laughing despite himself.  

One sunny afternoon, Lily discovered a hidden patch of wildflowers behind the fence, their colors brighter than any garden she’d seen. She brought her favorite feather toy, twirling it through the

[TTFT: 26 ms, TPOT: 11 ms]
```

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

Use the OpenAI API and `/v1` so streaming responses include TTFT/TPOT:

```bash
curl -X POST http://localhost:8081/api/v1/backends \
  -H "Content-Type: application/json" \
  -d '{
    "name": "sglang",
    "endpoint": "http://sglang:30000/v1",
    "type": "openai",
    "model_id": "openai:Qwen/Qwen3-8B"
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
