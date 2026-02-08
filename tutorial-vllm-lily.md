---
layout: default
title: Tutorial | Run a simple agent with Orla and vLLM
---

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

## 1. Start Orla and vLLM

From the root of the Orla repository:

```bash
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

This starts:

- vLLM on port 8000 (OpenAI-compatible API), serving the default model `Qwen/Qwen3-4B-Instruct-2507`.
- Orla daemon on port 8081, using the config in `deploy/orla-vllm.yaml` (including the `single_agent` workflow).

Optional: use a different model or Hugging Face token:

```bash
export VLLM_MODEL=Qwen/Qwen3-8B
export HF_TOKEN=your_token   # only for gated models
docker compose -f deploy/docker-compose.vllm.yaml up -d
```

If you change the model, set `agentic_serving.llm_servers[0].model` in `deploy/orla-vllm.yaml` to match (e.g. `openai:Qwen/Qwen3-8B`), then restart the Orla service:

```bash
docker compose -f deploy/docker-compose.vllm.yaml restart orla
```

## 2. Check that the daemon is up

```bash
curl -s http://localhost:8081/api/v1/health
```

You should get a successful response (e.g. HTTP 200).

## 3. Run the “Lily” story workflow

The included config defines a workflow named `single_agent` with one task that uses the default agent profile (vLLM). We’ll call it from a small Go program using the Orla API client.

Create a file `main.go` in the Orla repo root:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/dorcha-inc/orla/pkg/api"
)

func main() {
	client := api.NewClient("http://localhost:8081")
	exec, err := client.StartWorkflow(context.Background(), "single_agent")
	if err != nil {
		log.Fatal(err)
	}
	task, taskIndex, complete, _, err := client.GetNextTask(context.Background(), exec)
	if err != nil || complete {
		log.Fatal(err)
	}
	prompt := "Tell me a short, cheerful story about a cat called Lily."
	if task.Prompt != "" {
		prompt = task.Prompt
	}
	resp, err := client.ExecuteTask(context.Background(), exec, taskIndex, prompt, &api.ExecuteTaskOptions{MaxTokens: 512, Stream: false})
	if err != nil {
		log.Fatal(err)
	}
	_ = client.CompleteTask(context.Background(), exec, taskIndex, resp)
	fmt.Println(resp.Content)
}
```

From the repo root, run:

```bash
go run .
```

You should see the model’s story about Lily printed to the terminal. For a curl-based version of the same workflow, see the appendix below.

## 4. Stop the stack

When you’re done:

```bash
docker compose -f deploy/docker-compose.vllm.yaml down
```

Use `down -v` if you also want to remove the vLLM model cache volume.

You’ve run a simple agent that tells a story about a cat called Lily using Orla’s daemon and vLLM. For more workflows, multi-step agents, and configuration options, see the [Orla repo](https://github.com/dorcha-inc/orla) and the [deploy README](https://github.com/dorcha-inc/orla/blob/main/deploy/README.md).

## Appendix: Running the workflow with curl

If you prefer not to use Go, you can drive the same workflow with `curl` (and `jq` for parsing JSON). Install `jq` if needed: `sudo apt-get install -y jq`.

### Step 1: Start the workflow

```bash
EXEC=$(curl -s -X POST http://localhost:8081/api/v1/workflow/start \
  -H "Content-Type: application/json" \
  -d '{"workflow_name":"single_agent"}' | jq -r '.execution_id')
echo "Execution ID: $EXEC"
```

### Step 2: Get the next task

```bash
TASK_JSON=$(curl -s "http://localhost:8081/api/v1/workflow/task/next?execution_id=$EXEC")
echo "$TASK_JSON" | jq .
```

You should see `"complete": false` and one task with `"task_index": 0`.

### Step 3: Execute the task (ask for the story)

```bash
RESPONSE=$(curl -s -X POST http://localhost:8081/api/v1/workflow/task/execute \
  -H "Content-Type: application/json" \
  -d "{
    \"execution_id\": \"$EXEC\",
    \"task_index\": 0,
    \"prompt\": \"Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough.\",
    \"options\": { \"max_tokens\": 512, \"stream\": false }
  }")
echo "$RESPONSE" | jq .
```

To print just the story: `echo "$RESPONSE" | jq -r '.response.content'`

### Step 4: Complete the task

```bash
BODY=$(echo "$RESPONSE" | jq -c '{execution_id: "'"$EXEC"'", task_index: 0, response: .response}')
curl -s -X POST http://localhost:8081/api/v1/workflow/task/complete \
  -H "Content-Type: application/json" \
  -d "$BODY" | jq .
```

### Step 5: Confirm the workflow is complete

```bash
curl -s "http://localhost:8081/api/v1/workflow/task/next?execution_id=$EXEC" | jq .
```

You should see `"complete": true`.

One-shot script: save as `run_lily_story.sh` in the Orla repo root and run `./run_lily_story.sh` (requires `jq`):

```bash
#!/usr/bin/env bash
set -e
BASE=http://localhost:8081/api/v1

EXEC=$(curl -s -X POST "$BASE/workflow/start" -H "Content-Type: application/json" \
  -d '{"workflow_name":"single_agent"}' | jq -r '.execution_id')
echo "Execution ID: $EXEC"

TASK=$(curl -s "$BASE/workflow/task/next?execution_id=$EXEC")
IDX=$(echo "$TASK" | jq -r '.task_index')
COMPLETE=$(echo "$TASK" | jq -r '.complete')

if [ "$COMPLETE" = "true" ]; then
  echo "Workflow has no tasks."
  exit 0
fi

RESPONSE=$(curl -s -X POST "$BASE/workflow/task/execute" -H "Content-Type: application/json" \
  -d "{
    \"execution_id\": \"$EXEC\",
    \"task_index\": $IDX,
    \"prompt\": \"Tell me a short, cheerful story about a cat called Lily. Two or three paragraphs is enough.\",
    \"options\": { \"max_tokens\": 512, \"stream\": false }
  }")

echo "--- Story (Lily) ---"
echo "$RESPONSE" | jq -r '.response.content'

COMPLETE_BODY=$(echo "$RESPONSE" | jq -c '{execution_id: "'"$EXEC"'", task_index: '"$IDX"', response: .response}')
curl -s -X POST "$BASE/workflow/task/complete" -H "Content-Type: application/json" -d "$COMPLETE_BODY" > /dev/null
echo "--- Done ---"
```
