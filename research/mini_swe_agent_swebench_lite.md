# Tutorial: Running mini-SWE-agent with SWE-bench Lite

This tutorial walks through running [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) on [SWE-bench Lite](https://www.swebench.com/lite.html) using the same Docker Compose setup as Orla: SGLang as the LLM backend. mini-SWE-agent is a minimal Python agent (bash-only tools, linear history) and is a common baseline for SWE-bench. Using our compose keeps the backend identical so you can compare runs with [Orla + SWE-bench Lite](orla_swebench_lite.md). For **comparable runs**, start the **full stack** (`docker compose -f deploy/docker-compose.sglang.yaml up -d`) so SGLang is started with `--tool-call-parser qwen`; Orla uses the OpenAI API to the same SGLang with the same model (Qwen/Qwen3-8B).

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for the SGLang container.
- **Orla repo** cloned (for the run script and to start SGLang via the same compose).
- **[uv](https://docs.astral.sh/uv/)** so you can run with a single sync and no manual venv.
- **(Optional)** **[sb-cli](https://www.swebench.com/sb-cli/)** to evaluate or submit predictions. See [Setting up sb-cli for SWE-bench Lite](sb_cli_swebench_lite.md) for installation with uv.

For full SWE-bench Lite runs, mini-swe-agent uses **Docker** (or another supported sandbox) to run each instance in isolation. Ensure Docker is available and your user can run containers without `sudo`. If you see "permission denied" or exit **126** when containers start, add your user to the `docker` group and start a new login session so the change takes effect:

```bash
sudo usermod -aG docker $USER
# Then log out and SSH back in, or in the current shell run:
newgrp docker
```

Verify with `docker run --rm hello-world`. 

## What is mini-SWE-agent?

[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) is a minimal software-engineering agent: it uses only a **bash** tool and a **linear message history**. The model is called repeatedly with the same conversation; each turn can include one or more bash commands, and their output is appended before the next call. It supports many backends via LiteLLM; we point it at SGLang’s OpenAI-compatible API. The Orla repo includes `run.bash` in that directory; it runs `mini-extra swebench` with the right options and environment for our setup.

## 1. Start SGLang with Docker Compose

From the root of the Orla repository, start SGLang (the same stack used for Orla). For **comparable runs with Orla**, start the **full stack** so SGLang is launched with `--tool-call-parser qwen` (included in the compose):

```bash
cd orla
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

This starts **SGLang** on port 30000 (OpenAI-compatible API) and **Orla** on 8081. Default model: `Qwen/Qwen3-8B`. mini-SWE-agent only talks to SGLang on port 30000; you can ignore the Orla daemon if you are only running mini-SWE-agent. To start only SGLang:

```bash
docker compose -f deploy/docker-compose.sglang.yaml up -d sglang
```

Check that SGLang is reachable (OpenAI-compatible list models):

```bash
curl -s http://localhost:30000/v1/models
```

You should see a JSON response (e.g. with a `models` array).

```bash
{"object":"list","data":[{"id":"Qwen/Qwen3-8B","object":"model","created":1772313846,"owned_by":"sglang","root":"Qwen/Qwen3-8B","parent":null,"max_model_len":40960}]}
```

## 2. Install dependencies with uv

The `mini_swe_agent` directory has a `pyproject.toml` that depends on `mini-swe-agent[full]` (includes the SWE-bench `mini-extra` command). From that directory, sync the environment once:

```bash
cd examples/swe_bench_lite/cache_strategy/mini_swe_agent
uv sync
```

This creates a `.venv` and installs the package (including a minimal Python package so the build backend can create an editable install). You don’t create or activate a virtualenv yourself; `uv run` (below) will use it.

## 3. Point mini-SWE-agent at SGLang

mini-SWE-agent uses LiteLLM with the **OpenAI-compatible API** (same as Orla). The run script sets `OPENAI_BASE_URL` to `http://localhost:30000/v1` by default. If SGLang runs on another host or port, set it before running:

```bash
export OPENAI_BASE_URL=http://your-host:30000/v1
```

## 4. Run SWE-bench Lite

From the `mini_swe_agent` directory, run the script with **uv** so it uses the project’s `.venv` (where `mini-extra` is installed):

```bash
uv run ./run.bash
```

The script runs `mini-extra swebench` with our defaults (subset **lite**, split **dev**). Predictions are written under `./mini_swe_agent_preds` by default. Each instance runs in a Docker container (by default) if your environment supports it.

If you see **"Skipping N existing instances"** and **"Running on 0 instances"**, the tool is treating prior runs as complete. To re-run all instances (e.g. after fixing Docker or to retry), pass `--redo-existing`:

```bash
uv run ./run.bash --redo-existing
```

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

The script uses `openai/Qwen/Qwen3-8B` by default (same model as the Orla example). The `openai/` prefix tells LiteLLM to use the OpenAI-compatible endpoint (`OPENAI_BASE_URL`). If your SGLang model differs, set `MINI_SWE_MODEL` before running:

```bash
export MINI_SWE_MODEL=openai/Your/Model-Name
uv run ./run.bash
```

## 6. Evaluating predictions

The run produces a predictions file (e.g. `mini_swe_agent_preds/preds.json`). To evaluate or submit with **sb-cli** (including setup with uv), see [Setting up sb-cli for SWE-bench Lite](sb_cli_swebench_lite.md). You can also use the [SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/) directly; see the SWE-bench docs for details.

## 7. Stop the stack

When you’re done:

```bash
docker compose -f deploy/docker-compose.sglang.yaml down
```

Use `down -v` if you want to remove the SGLang model cache volume.

## Conclusion

You’ve run mini-SWE-agent on SWE-bench Lite with SGLang from our Docker Compose setup. For the same benchmark using the Orla Go client and the same backend, see [Orla with SWE-bench Lite](orla_swebench_lite.md).
