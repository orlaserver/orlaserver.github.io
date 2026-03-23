# Configuration

Orla uses defaults when no config file is given. To customize, pass a single config file:

## Orla Service

Pass a config file with `-c` / `--config`, e.g. 

```bash
orla serve --config orla.yaml
```

## Orla Standalone Agent

Pass a config file with `-c` / `--config`, e.g. 

```bash
orla agent "Hello" --config orla.yaml
```

If omitted, built-in defaults are used. Override the model with `-m` / `--model`, e.g. 

```bash
orla agent "Hello" --model ollama:llama3
```

## Configuration options for the service

The service reads the config file for listen address and logging. Backends are registered programmatically by the host application, not from the config file.

| Option | Description | Default |
|--------|-------------|---------|
| `listen_address` | Address to bind (e.g. `localhost:8081`, `:8081`) | `localhost:8081` |
| `log_format` | Log format: `"pretty"` or `"json"` | `"json"` |
| `log_level` | Log level: `"debug"`, `"info"`, `"warn"`, `"error"`, `"fatal"` | `"info"` |

**CLI overrides:** `-l` / `--listen-address` overrides `listen_address`; `--pretty` overrides `log_format` (forces pretty-printed logs).

### Example orla.yaml

```yaml
listen_address: localhost:8081
log_format: json
log_level: info
```

## Configuration options for the standalone agent

| Option | Description | Default |
|--------|-------------|---------|
| `model` | Model identifier (e.g. `ollama:ministral-3:3b`, `ollama:qwen3:0.6b`) | `ollama:qwen3:0.6b` |
| `streaming` | Enable streaming responses | `true` |
| `output_format` | Output format: `auto`, `rich`, or `plain` | `auto` |
| `show_thinking` | Show thinking trace for thinking-capable models | `false` |
| `show_tool_calls` | Show detailed tool call information | `false` |
| `show_progress` | Show progress when UI is disabled (e.g. stdin piped) | `false` |
| `log_format` | Log format: `"pretty"` or `"json"` | `"json"` |
| `log_level` | Log level: `"debug"`, `"info"`, `"warn"`, `"error"`, `"fatal"` | `"info"` |

### Example orla.yaml

```yaml
log_format: json
log_level: info
model: ollama:llama3
streaming: true
output_format: auto
show_thinking: false
show_tool_calls: true
```

## Environment variables

You can override some settings via environment variables, for example:

```bash
export ORLA_MODEL=ollama:qwen3:1.7b
export ORLA_SHOW_TOOL_CALLS=true
```

## LLM backend

Configure the LLM backend (endpoint, type, optional API key env var) under `llm_backend` in your config. For remote Ollama, set `endpoint` and `type: ollama`. For OpenAI-compatible APIs, set `type: openai` and `api_key_env_var` to the name of the environment variable holding the API key.
