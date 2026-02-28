# Tutorial: Running mini-SWE-agent with SWE-bench Lite

This tutorial walks through running [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) on [SWE-bench Lite](https://www.swebench.com/lite.html) using the same Docker Compose setup as Orla: SGLang as the LLM backend. mini-SWE-agent is a minimal Python agent (bash-only tools, linear history) and is a common baseline for SWE-bench. Using our compose keeps the backend identical so you can compare runs with [Orla + SWE-bench Lite](orla_swebench_lite.md).

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for the SGLang container.
- **Orla repo** cloned (for the run script and to start SGLang via the same compose).
- **[uv](https://docs.astral.sh/uv/)** so you can run with a single sync and no manual venv.

For full SWE-bench Lite runs, mini-swe-agent uses **Docker** (or another supported sandbox) to run each instance in isolation. Ensure Docker is available and your user can run containers.

## What is mini-SWE-agent?

[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) is a minimal software-engineering agent: it uses only a **bash** tool and a **linear message history**. The model is called repeatedly with the same conversation; each turn can include one or more bash commands, and their output is appended before the next call. It supports many backends via LiteLLM (including Ollama-compatible APIs), so we point it at SGLang. The Orla repo includes `run.bash` in that directory; it runs `mini-extra swebench` with the right options and environment for our setup.

## 1. Start SGLang with Docker Compose

From the root of the Orla repository, start at least SGLang (the same stack used for Orla):

```bash
cd orla
docker compose -f deploy/docker-compose.sglang.yaml up -d sglang
```

This starts **SGLang** on port 30000 with an Ollama-compatible API. Default model: `Qwen/Qwen3-8B`. You can start the full stack (Orla + SGLang) if you like; mini-SWE-agent only talks to SGLang:

```bash
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

Check that SGLang is reachable (Ollama-style list models):

```bash
curl -s http://localhost:30000/api/tags | head -20
```

You should see a JSON response (e.g. with a `models` array).

## 2. Install dependencies with uv

The `mini_swe_agent` directory has a `pyproject.toml` that depends on `mini-swe-agent[full]` (includes the SWE-bench `mini-extra` command). From that directory, sync the environment once:

```bash
cd examples/swe_bench_lite/cache_strategy/mini_swe_agent
uv sync
```

This creates a `.venv` and installs the package. You don’t create or activate a virtualenv yourself; `uv run` (below) will use it.

## 3. Point mini-SWE-agent at SGLang

mini-SWE-agent uses LiteLLM, which can talk to an Ollama-compatible server via `OLLAMA_HOST`. Set it to your SGLang host (default port 30000):

```bash
export OLLAMA_HOST=http://localhost:30000
```

If SGLang runs on another machine or port, use that URL instead.

## 4. Run SWE-bench Lite

From the `mini_swe_agent` directory, run the script with **uv** so it uses the project’s `.venv` (where `mini-extra` is installed):

```bash
export OLLAMA_HOST=http://localhost:30000
uv run ./run.bash
```

The script runs `mini-extra swebench` with our defaults (subset **lite**, split **dev**). Predictions are written under `./mini_swe_agent_preds` by default. Each instance runs in a Docker container (by default) if your environment supports it.

### Run a small slice first

To test with only a few instances (e.g. first 3):

```bash
uv run ./run.bash --slice 0:3
```

### Full test set

To run the full 300 test instances:

```bash
uv run ./run.bash --split test
```

## 5. Model name

The script uses a default model name that matches what SGLang serves over the Ollama API (e.g. `ollama/qwen3:8b`). If your SGLang model differs, set `MINI_SWE_MODEL` before running:

```bash
export MINI_SWE_MODEL=ollama/your-model-name
uv run ./run.bash
```

The exact name depends on how SGLang exposes the model; check SGLang logs or the `/api/tags` response if needed.

## 6. Evaluating predictions

The run produces a predictions file (e.g. in the output directory). You can evaluate it with the [SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/) or the [sb-cli](https://www.swebench.com/sb-cli/) tool. See the SWE-bench docs for details.

## 7. Stop the stack

When you’re done:

```bash
docker compose -f deploy/docker-compose.sglang.yaml down
```

Use `down -v` if you want to remove the SGLang model cache volume.

## Conclusion

You’ve run mini-SWE-agent on SWE-bench Lite with SGLang from our Docker Compose setup. For the same benchmark using the Orla Go client and the same backend, see [Orla with SWE-bench Lite](research/orla_swebench_lite.md).
