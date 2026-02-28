# Tutorial: Running Orla with SWE-bench Lite

This tutorial runs the [Orla](https://github.com/dorcha-inc/orla) SWE-bench Lite baseline in Docker: one **run_bash** tool and a ReAct-style agent loop against [SWE-bench Lite](https://www.swebench.com/lite.html) instances.

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for SGLang.
- **Orla repo** cloned so you can run from the repo root.

## What is SWE-bench Lite?

[SWE-bench Lite](https://www.swebench.com/lite.html) is a curated benchmark of 300 test instances (plus 23 dev instances) from real GitHub issues. Each instance has a **problem statement**, a **repository**, and a **base commit**. The agent’s job is to produce a patch that fixes the issue. The baseline uses a single **run_bash** tool: the model runs commands (e.g. explore the repo, edit files, run tests), and the results are appended until the model stops or hits a step limit.


## 1. Start the stack

A single compose stack runs SGLang, Orla, and the baseline in containers. The baseline container starts only after Orla is healthy (Orla waits for SGLang). The baseline image includes Go, git, and jq; the entrypoint clones the instance repo, checks out `base_commit`, and runs the agent. No Go or git on the host required.

From the **Orla repo root**:

```bash
docker compose -f deploy/docker-compose.swebench-baseline.yaml up -d
```

This starts SGLang (GPU), then Orla (after SGLang is healthy). The baseline service is defined but does not run until you run it manually (see step 2).

## 2. Prepare an instance and run the baseline

Create directories and put a SWE-bench Lite instance JSON in `deploy/instances/`. Paths are relative to the `deploy/` directory:

```bash
mkdir -p deploy/instances deploy/output
cp examples/swe_bench_lite/baseline_orla_sglang/sample_instance.json deploy/instances/instance.json
```

Run the baseline (one instance per run):

```bash
docker compose -f deploy/docker-compose.swebench-baseline.yaml run --rm baseline
```

Predictions are written to **`deploy/output/predictions.jsonl`**.

To run a different instance file or pass flags (e.g. `-max-steps 10`):

```bash
docker compose -f deploy/docker-compose.swebench-baseline.yaml run --rm baseline /instances/other.json -max-steps 10
```

## 3. Stop the stack

```bash
docker compose -f deploy/docker-compose.swebench-baseline.yaml down
```

Use `down -v` to remove the SGLang model cache volume.

## Instance JSON format

Each instance is a single JSON object with at least:

- **`instance_id`** – Identifier (e.g. `django__django-11099`).
- **`repo`** – Repository (e.g. `django/django`).
- **`base_commit`** – Git commit the agent should work from.
- **`problem_statement`** – The issue description (what to fix).

The sample file is at `examples/swe_bench_lite/baseline_orla_sglang/sample_instance.json`. You can download real instances from the [SWE-bench Lite dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) or [SWE-bench/SWE-bench_Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite).

Predictions are one JSON object per line (instance_id, model_name_or_path, model_patch). You can run the [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/) against the output file.

## How the baseline works

The example in `examples/swe_bench_lite/baseline_orla_sglang/main.go`:

1. **Registers** the SGLang backend with the Orla daemon (OpenAI-compatible API at `http://sglang:30000/v1`). SGLang must be started with `--tool-call-parser qwen` (the deploy compose includes it).
2. **Adds a single tool**, `run_bash`, that runs one bash command in `-workdir` and returns stdout, stderr, and exit code.
3. **Loads the instance** and builds a user message with the problem statement, repo, and base commit.
4. **Runs a loop**: `ExecuteWithMessages` → if the model returns tool calls, runs them via the agent, appends assistant + tool-result messages, and repeats until there are no tool calls or `-max-steps` is reached.

This is the same pattern as [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md): **ExecuteWithMessages** and **RunToolCallsInResponse** implement a ReAct-style loop.

## Conclusion 

You’ve run the Orla SWE-bench Lite baseline with SGLang and one bash tool. For tool-calling basics, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For a minimal SGLang-only run, see [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md).
