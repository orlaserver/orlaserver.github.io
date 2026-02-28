# Tutorial: Using Tools with Orla

This tutorial shows how to use **tools** with Orla and any of the supported backends: [vLLM](https://docs.vllm.ai/), [Ollama](https://ollama.com/), or [SGLang](https://sgl-project.github.io/). The model receives tool definitions, can request tool calls in its response, and you run the tools locally and send the results back in the next request.

## What you need

- Docker and Docker Compose (Compose V2).
- For **vLLM** or **SGLang**: an NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- For **Ollama**: no GPU required (CPU is fine for small models).
- The [Orla repo](https://github.com/dorcha-inc/orla) cloned.
- Go 1.25 or later.

Pick one backend and start its stack (see [tutorial: vLLM](tutorial-vllm-lily.md), [Ollama](tutorial-ollama-lily.md), or [SGLang](tutorial-sglang-lily.md) for full setup). Then register that backend with Orla and run the Go program below.

## 1. Start a backend and Orla

From the Orla repo root, start one of:

```bash
# vLLM (GPU)
docker compose -f deploy/docker-compose.vllm.yaml up -d

# Ollama (CPU or GPU). Then: docker compose exec ollama ollama pull llama3.2:3b
docker compose -f deploy/docker-compose.ollama.yaml up -d

# SGLang (GPU)
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

Ensure Orla is healthy:

```bash
curl http://localhost:8081/api/v1/health
```

## 2. Register the backend

Register the backend that matches the stack you started. Use the **service name** as the host when Orla runs in the same Compose (so Orla in its container can reach the backend).

**vLLM:**

```go
backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
    Name:     "vllm",
    Endpoint: "http://vllm:8000/v1",
    Type:     "openai",
    ModelID:  "openai:Qwen/Qwen3-4B-Instruct-2507",
})
```

**Ollama:**

```go
backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
    Name:     "ollama",
    Endpoint: "http://ollama:11434",
    Type:     "ollama",
    ModelID:  "ollama:llama3.2:3b",
})
```

**SGLang** (Ollama-compatible API on port 30000):

```go
backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
    Name:     "sglang",
    Endpoint: "http://sglang:30000",
    Type:     "ollama",
    ModelID:  "ollama:Qwen/Qwen3-8B",
})
```

## 3. Define a tool

Tools have a **name**, **description**, **input** and **output** schemas (for the model), and a **Run** function that you execute locally when the model requests the tool.

Example: a simple “get weather” tool that returns a fixed string. The model will see the tool spec and can “call” it; your code runs it and sends the result back as a message.

```go
tool, err := orla.NewTool(
    "get_weather",
    "Get the current weather for a location.",
    orla.ToolSchema{
        "type": "object",
        "properties": map[string]any{
            "location": map[string]any{"type": "string", "description": "City or place name"},
        },
        "required": []any{"location"},
    },
    orla.ToolSchema{
        "type": "object",
        "properties": map[string]any{
            "temperature": map[string]any{"type": "number"},
            "conditions": map[string]any{"type": "string"},
        },
    },
    orla.ToolRunnerFromSchema(func(ctx context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
        location, _ := input["location"].(string)
        if location == "" {
            location = "unknown"
        }
        return orla.ToolSchema{
            "temperature": 22.0,
            "conditions": "sunny in " + location,
        }, nil
    }),
)
if err != nil {
    log.Fatal(err)
}
```

## 4. Agent loop with tools

Create an agent, attach the tool, and run a loop: send messages (including tool-result messages), and stop when the model returns no tool calls.

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

    // Register backend (use the block that matches your stack: vLLM, Ollama, or SGLang)
    backend, err := client.RegisterBackend(ctx, &orla.RegisterBackendRequest{
        Name:     "vllm",
        Endpoint: "http://vllm:8000/v1",
        Type:     "openai",
        ModelID:  "openai:Qwen/Qwen3-4B-Instruct-2507",
    })
    if err != nil {
        log.Fatal("register backend: ", err)
    }

    agent := orla.NewAgent(client, backend)
    agent.SetMaxTokens(512)

    // Define and add the tool
    tool, err := orla.NewTool(
        "get_weather",
        "Get the current weather for a location.",
        orla.ToolSchema{
            "type": "object",
            "properties": map[string]any{
                "location": map[string]any{"type": "string", "description": "City or place name"},
            },
            "required": []any{"location"},
        },
        orla.ToolSchema{
            "type": "object",
            "properties": map[string]any{
                "temperature": map[string]any{"type": "number"},
                "conditions": map[string]any{"type": "string"},
            },
        },
        orla.ToolRunnerFromSchema(func(ctx context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
            location, _ := input["location"].(string)
            if location == "" {
                location = "unknown"
            }
            return orla.ToolSchema{
                "temperature": 22.0,
                "conditions": "sunny in " + location,
            }, nil
        }),
    )
    if err != nil {
        log.Fatal(err)
    }
    if err := agent.AddTool(tool); err != nil {
        log.Fatal(err)
    }

    // Conversation: start with one user message
    prompt := "What's the weather in Paris? Reply in one sentence."
    messages := []orla.Message{{Role: "user", Content: prompt}}

    for {
        resp, err := agent.ExecuteWithMessages(ctx, messages)
        if err != nil {
            log.Fatal(err)
        }

        if len(resp.ToolCalls) == 0 {
            fmt.Println("Model reply:", resp.Content)
            break
        }

        // Append assistant turn (content) then tool-result messages
        messages = append(messages, orla.Message{Role: "assistant", Content: resp.Content})

        toolMessages, err := agent.RunToolCallsInResponse(ctx, resp)
        if err != nil {
            log.Fatal(err)
        }
        for _, m := range toolMessages {
            messages = append(messages, *m)
        }
    }
}
```

- **ExecuteWithMessages** sends the current `messages` and the agent’s tools to the backend. The model may return text and/or **tool_calls**.
- **RunToolCallsInResponse** parses `resp.ToolCalls`, runs each tool by name (using your `Run`), and returns the corresponding **tool-result messages** (role `"tool"`, content, tool_call_id, tool_name).
- You append the assistant message (the model’s content) and those tool messages, then call **ExecuteWithMessages** again. The loop ends when the model responds with no tool calls.

This flow is the same for **vLLM**, **Ollama**, and **SGLang**; only the backend registration (name, endpoint, type, model_id) changes.

## 5. Backend-specific notes

- **vLLM**: Uses the OpenAI-compatible API. Tool calls include a unique `id`; you must send it back as `tool_call_id` in the tool result message so the model can match results to calls.
- **Ollama**: Uses `tool_name` and content for tool results. Ollama does not yet support per-call IDs, so when the same tool is called multiple times in one turn, matching is by order.
- **SGLang**: In this setup SGLang is used with the Ollama-compatible API (same as the Ollama tutorial). Behavior matches Ollama for tool results.

## 6. Stop the stack

```bash
docker compose -f deploy/docker-compose.vllm.yaml down   # or ollama / sglang
```

You’ve run a tool-calling agent with Orla and one of vLLM, Ollama, or SGLang. For a minimal run without tools, see [Using Orla with vLLM](tutorial-vllm-lily.md), [Ollama](tutorial-ollama-lily.md), or [SGLang](tutorial-sglang-lily.md).
