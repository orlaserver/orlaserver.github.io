# Usage

Orla has two modes: **`orla agent`** for one-shot runs and **`orla serve`** for the agent engine (HTTP API).

## orla agent

One-shot execution. Give a prompt (and optional stdin); Orla uses the configured model and returns the result.

**Basic:**

```bash
orla agent "List all files in the current directory"
```

**With stdin (e.g. pipe):**

```bash
orla agent "summarize this code" < main.go
```

```bash
cat data.json | orla agent "extract all email addresses" | sort -u
```

```bash
git status | orla agent "Draft a short, imperative-mood commit message for these changes"
```

**Override model:**

```bash
orla agent "Hello" --model ollama:ministral-3:3b
```

**Redirect output:**

```bash
orla agent "find all TODO comments in *.c files in pwd" > todos.txt
```

## orla serve

Run the Orla agent engine as a long-lived service. It exposes an HTTP API for inference and coordination. You register the LLM backends programmatically using Orla's API.

```bash
orla serve --config orla.yaml
```

Set the bind address in config or with `--listen-address`. Default: `localhost:8081`. The API provides:

### Health

```bash
GET /api/v1/health
```

### Backend registration

```bash
POST /api/v1/backends
```

Registers an LLM backend.

```bash
GET /api/v1/backends
``` 

Lists registered backend names.

Register a backend before calling execute with that `backend` name.

### Agent Execution

```bash
POST /api/v1/execute
``` 

Use the [Orla Go client](https://github.com/dorcha-inc/orla/tree/main/pkg/api) or call the HTTP API directly. See the [API package](https://github.com/dorcha-inc/orla/tree/main/internal/serving/api) for request/response types.
