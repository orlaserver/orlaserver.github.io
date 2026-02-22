# Getting Started

## Installation

The easiest and recommended way to install Orla on macOS and Linux is using [Homebrew](https://brew.sh/):

```bash
brew install --cask dorcha-inc/orla/orla
```

Alternatively, use the installation script (may require `sudo`). It can install Orla, Ollama, and set everything up:

```bash
curl -fsSL https://raw.githubusercontent.com/dorcha-inc/orla/main/scripts/install.sh | sh
```

### Installing without local Ollama

If you already have a remote Ollama server or prefer to manage Ollama separately, skip the local Ollama installation:

**Homebrew:**

```bash
HOMEBREW_ORLA_SKIP_OLLAMA=1 brew install --cask dorcha-inc/orla/orla
```

**Install script:**

```bash
curl -fsSL https://raw.githubusercontent.com/dorcha-inc/orla/main/scripts/install.sh | sh -s -- --skip-ollama
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
