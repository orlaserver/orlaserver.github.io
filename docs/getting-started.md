# Getting Started

## Installation

The easiest and recommended way to install Orla on macOS and Linux is using [Homebrew](https://brew.sh/):

```bash
brew install --cask harvard-cns/orla/orla
```

Alternatively, use the installation script (may require `sudo`). It can install Orla, Ollama, and set everything up:

```bash
curl -fsSL https://raw.githubusercontent.com/harvard-cns/orla/main/scripts/install.sh | sh
```

### Installing without local Ollama

If you already have a remote Ollama server or prefer to manage Ollama separately, skip the local Ollama installation:

**Homebrew:**

```bash
HOMEBREW_ORLA_SKIP_OLLAMA=1 brew install --cask harvard-cns/orla/orla
```

**Install script:**

```bash
curl -fsSL https://raw.githubusercontent.com/harvard-cns/orla/main/scripts/install.sh | sh -s -- --skip-ollama
```

After installation, point Orla at your remote Ollama server via environment variable or config:

```bash
export OLLAMA_HOST=http://your-ollama-server:11434
```

Or in `orla.yaml`:

```yaml
llm_backend:
  endpoint: http://your-ollama-server:11434
  type: ollama
```

## Quick start

Try Orla:

```bash
orla agent "Hello"
```

One-shot with stdin:

```bash
orla agent "summarize this code" < main.go
```

To uninstall, see [Uninstalling Orla](uninstalling.md).

## Python SDK (pyorla)

For Python apps and [LangGraph](tutorials/tutorial-langgraph.md), use **pyorla** from the Orla repository (`pyorla/`). With a clone of [github.com/harvard-cns/orla](https://github.com/harvard-cns/orla):

```bash
cd pyorla
uv sync
```

Or add it as a path dependency from your project: `uv add ../pyorla` (adjust the path).

Full SDK documentation — remote and local clients, tools, LangChain, notebooks — is on **[Python SDK (pyorla)](pyorla.md)**.
