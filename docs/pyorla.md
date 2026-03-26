# Python SDK (pyorla)

**pyorla** is Orla’s Python SDK for the HTTP API: clients, stages, workflows, tools, and LangChain integration. It targets Python 3.11+ and ships as the `pyorla` package in the Orla repository under the `pyorla/` directory.

## Install

From a clone of the Orla repo, in the `pyorla` directory:

```bash
cd pyorla
uv sync
```

From another project, add it as a path dependency (adjust the path):

```bash
uv add ../pyorla
```

## Connect to a running daemon

Point `OrlaClient` at wherever `orla serve` is listening (see [Usage](usage.md#orla-serve)):

```python
from pyorla import OrlaClient

client = OrlaClient("http://localhost:8081")
client.health()
```

You can also use **`OrlaClient.from_env()`**, which reads **`ORLA_URL`** (default `http://localhost:8081` if unset).

Register backends and run inference via `Stage`, `Workflow`, or `ChatOrla` as in the tutorials below.

## Local server from Python

For development or notebooks, **`orla_runtime()`** is a context manager that starts `orla serve` on loopback, waits until `/api/v1/health` succeeds, and tears the process down when the block exits. You need the **`orla` CLI on `PATH`**, or set **`ORLA_BIN`** to the binary path.

```python
from pyorla import orla_runtime

with orla_runtime() as client:
    client.health()
    # register backends, call execute, etc.
```

Under the hood this runs `orla serve --listen-address 127.0.0.1:<ephemeral-port>`.

## Notebooks and Colab

Hosted notebooks (e.g. Colab) cannot reach `localhost` on your laptop. Use one of:

- An Orla daemon on a **reachable URL** (VM, Kubernetes, etc.) and pass that URL to `OrlaClient`, or  
- A **tunnel** (ngrok, Cloudflare Tunnel, etc.) from the machine where `orla serve` runs to a URL you use in the notebook.

## API surface (high level)

| Area | Typical entry points |
|------|----------------------|
| HTTP client | `OrlaClient`, `OrlaClient.from_env()`, `OrlaError` |
| Local daemon | `orla_runtime`, `OrlaBinaryNotFoundError`, `resolve_orla_binary` |
| Stages / workflows | `Stage`, `Workflow`, backend helpers (`new_vllm_backend`, etc.), stage mapping types |
| Tools | `@orla_tool`, `Tool`, `new_tool`, `tool_from_langchain` |
| LangChain | `ChatOrla`, message helpers in `pyorla.messages` |

For REST shape of `/api/v1/execute` and related routes, see [Usage](usage.md); pyorla request types match those JSON payloads.

## Tutorials and examples

- [Using Orla with LangGraph](tutorials/tutorial-langgraph.md) — pyorla with `StateGraph`, multi-backend stages, and tools.
- Example projects and workflow demos live in the Orla repo under `pyorla/examples/` (e.g. `workflow_demo`).

## Hacking on pyorla

From the `pyorla` directory in the Orla repo:

```bash
uv sync
uv run ty check
uv run pytest
```

(See also [Developer’s Guide](developers-guide.md).)
