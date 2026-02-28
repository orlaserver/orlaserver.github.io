# Tutorial: Using Tools with Orla

This tutorial shows how to use **tools** with Orla and any of the supported backends: [vLLM](https://docs.vllm.ai/), [Ollama](https://ollama.com/), or [SGLang](https://sgl-project.github.io/). The model receives tool definitions, can request tool calls in its response, and you run the tools locally and send the results back in the next request.

## What you need

- Docker and Docker Compose (Compose V2).
- For **vLLM** or **SGLang**: an NVIDIA GPU and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- For **Ollama**: no GPU required (CPU is fine for small models).
- The [Orla repo](https://github.com/dorcha-inc/orla) cloned.
- Go 1.25 or later.

Pick one backend and start its stack (see [vLLM](tutorials/tutorial-vllm-lily.md), [Ollama](tutorials/tutorial-ollama-lily.md), or [SGLang](tutorials/tutorial-sglang-lily.md) for full setup). Then register that backend with Orla and run the Go program below.

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

> [!NOTE]
> Tool calling with vLLM only works when the server is started with `--enable-auto-tool-choice` and `--tool-call-parser`. The Orla deploy Compose file [`deploy/docker-compose.vllm.yaml`](https://github.com/dorcha-inc/orla/blob/main/deploy/docker-compose.vllm.yaml) already includes these flags for the default Qwen model using the `hermes` parser. If you use a different model or see errors about tool choice, check [vLLM’s Tool Calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/) for the correct `--tool-call-parser` for your model and update the `command` in the Compose file if needed.

## 2. Register the backend

Register the backend that matches the stack you started. Use the **service name** as the host when Orla runs in the same Compose (so Orla in its container can reach the backend).

**vLLM:**

```go
backend := orla.NewVLLMBackend("Qwen/Qwen3-4B-Instruct-2507", "http://vllm:8000/v1")
if err := client.RegisterBackend(ctx, backend); err != nil { log.Fatal(err) }
```

**Ollama:**

```go
backend := orla.NewOllamaBackend("llama3.2:3b", "http://ollama:11434")
if err := client.RegisterBackend(ctx, backend); err != nil { log.Fatal(err) }
```

**SGLang** (OpenAI API on port 30000; use `/v1` for TTFT/TPOT):

```go
backend := orla.NewSGLangBackend("Qwen/Qwen3-8B", "http://sglang:30000/v1")
if err := client.RegisterBackend(ctx, backend); err != nil { log.Fatal(err) }
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

    // Register backend (use the helper that matches your stack: vLLM, Ollama, or SGLang)
    backend := orla.NewVLLMBackend("Qwen/Qwen3-4B-Instruct-2507", "http://vllm:8000/v1")
    if err := client.RegisterBackend(ctx, backend); err != nil {
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

The output should be something like:


```bash
Model reply: <think>
Okay, the user asked for the weather in Paris, and I called the get_weather function. The response came back with sunny conditions and a temperature of 22°C. Now I need to reply in one sentence. Let me combine those details smoothly. Maybe start with the main condition and mention the temperature. Check if the units are specified, but since it's not mentioned, just use Celsius. Alright, the sentence should be clear and concise.
</think>

The weather in Paris is sunny with a temperature of 22°C.
```

- **ExecuteWithMessages** sends the current `messages` and the agent’s tools to the backend. The model may return text and/or **tool_calls**.
- **RunToolCallsInResponse** parses `resp.ToolCalls`, runs each tool by name (using your `Run`), and returns the corresponding **tool-result messages** (role `"tool"`, content, tool_call_id, tool_name).
- You append the assistant message (the model’s content) and those tool messages, then call **ExecuteWithMessages** again. The loop ends when the model responds with no tool calls.

This flow is the same for **vLLM**, **Ollama**, and **SGLang**; only the backend registration (name, endpoint, type, model_id) changes.

## 5. Streaming with tools

You can stream the model’s reply (and thinking, if the backend supports it) while still doing the same tool loop. Use **ExecuteStreamWithMessages** and **ConsumeStream**: the stream delivers `content`, `thinking`, and `tool_call` events, then a **done** event with the full response (including **ToolCalls**). Run tools on that response and continue the loop as before.

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

    backend := orla.NewVLLMBackend("Qwen/Qwen3-4B-Instruct-2507", "http://vllm:8000/v1")
    if err := client.RegisterBackend(ctx, backend); err != nil {
        log.Fatal("register backend: ", err)
    }

    agent := orla.NewAgent(client, backend)
    agent.SetMaxTokens(512)

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

    prompt := "What's the weather in Tokyo? One sentence."
    messages := []orla.Message{{Role: "user", Content: prompt}}

    for {
        stream, err := agent.ExecuteStreamWithMessages(ctx, messages)
        if err != nil {
            log.Fatal(err)
        }

        // Print content and thinking as they arrive; ConsumeStream returns the full response on "done"
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

        if len(resp.ToolCalls) == 0 {
            fmt.Println()
            break
        }

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

The output should be something like:

```bash
<think>
Okay, the user is asking for the weather in Tokyo in one sentence. Let me check the tools provided. There's a function called get_weather that takes a location parameter. So I need to call that function with Tokyo as the location. I'll make sure to format the tool call correctly within the XML tags. Just need to specify the name as get_weather and arguments with location set to Tokyo. That should do it
.
</think>

<think>
Okay, the user asked for the weather in Tokyo in one sentence. I called the get_weather function with Tokyo as the location. The response came back with conditions "sunny in Tokyo" and temperature 22. Now I need to present this information concisely. Let me check if the
 temperature is in Celsius or Fahrenheit. Since it's not specified, I'll assume Celsius. The user wants a single sentence, so I'll combine the conditions and temperature into one statement. Maybe something like, "It's sunny in Tokyo with a temperature of 22°C." That should cover both details in one sentence.
</think>

It's sunny in Tokyo with a temperature of 22°C.
```

- **ExecuteStreamWithMessages** takes the same `messages` (and tools) as the non-streaming call but returns a channel of events.
- **ConsumeStream** reads the channel, optionally runs your handler for each event (e.g. print content/thinking), and returns the full **InferenceResponse** when the stream sends **done**. That response includes **ToolCalls**.
- The rest of the loop is unchanged: append the assistant message and tool-result messages, then call **ExecuteStreamWithMessages** again until the model returns no tool calls.

## 6. Backend-specific notes

- **vLLM**: Uses the OpenAI-compatible API. Tool calls include a unique `id`; you must send it back as `tool_call_id` in the tool result message so the model can match results to calls.
- **Ollama**: Uses `tool_name` and content for tool results. Ollama does not yet support per-call IDs, so when the same tool is called multiple times in one turn, matching is by order.
- **SGLang**: Use the OpenAI-compatible API (`NewSGLangBackend` with endpoint `http://sglang:30000/v1`). Behavior matches vLLM for tool results and you get TTFT/TPOT in streaming.

## 7. Stop the stack

```bash
docker compose -f deploy/docker-compose.vllm.yaml down   # or ollama / sglang
```

You’ve run a tool-calling agent with Orla and one of vLLM, Ollama, or SGLang. For a minimal run without tools, see [Using Orla with vLLM](tutorial-vllm-lily.md), [Ollama](tutorial-ollama-lily.md), or [SGLang](tutorial-sglang-lily.md).
