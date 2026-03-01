# Tutorial: Running Orla with SWE-bench Lite

This tutorial runs the [Orla](https://github.com/dorcha-inc/orla) SWE-bench Lite baseline in Docker: one **run_bash** tool and a ReAct-style agent loop against [SWE-bench Lite](https://www.swebench.com/lite.html) instances.

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for SGLang.
- **Orla repo** cloned so you can run from the repo root.

## What is SWE-bench Lite?

[SWE-bench Lite](https://www.swebench.com/lite.html) is a curated benchmark of 300 test instances (plus 23 dev instances) from real GitHub issues. Each instance has a **problem statement**, a **repository**, and a **base commit**. The agent’s job is to produce a patch that fixes the issue. The baseline uses a single **run_bash** tool: the model runs commands (e.g. explore the repo, edit files, run tests), and the results are appended until the model stops or hits a step limit.


## 1. Start the stack

From the **Orla repo root**:

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml up -d
```

This starts SGLang (GPU), then Orla (after SGLang is healthy). The baseline image includes the dataset, Go, and git; the baseline discovers all instances under `/dataset` and runs them sequentially. The baseline service does not run until you run it manually (see step 2).

## 2. Run the full SWE-bench Lite benchmark

Create the output directory and run the baseline once. It will process every instance in the dataset (dev + test) in order:

```bash
mkdir -p deploy/output
docker compose -f deploy/docker-compose.swebench-lite.yaml run --rm run baseline
```

Predictions are appended to **`deploy/output/predictions.jsonl`** (one JSON object per line per instance).

## 3. Stop the stack

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml down
```

Use `down -v` to remove the SGLang model cache volume.

## Instance JSON format

Each instance is a single JSON object with at least:

- **`instance_id`** – Identifier (e.g. `django__django-11099`).
- **`repo`** – Repository (e.g. `django/django`).
- **`base_commit`** – Git commit the agent should work from.
- **`problem_statement`** – The issue description (what to fix).

The baseline image includes the dataset from `examples/swe_bench_lite/dataset/` (populated via `fetch_dataset.py` from [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)).

Predictions are one JSON object per line (instance_id, model_name_or_path, model_patch). You can run the [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/) against the output file.

## How the baseline works

The baseline experiment lives in `examples/swe_bench_lite/baseline/` (package baseline) and is invoked via `cmd/baseline` or `make baseline`:

1. **Discovers instances**: walks `/dataset` (e.g. `dataset/dev/` and `dataset/test/`) for all `*.json` files and runs them in sorted order.
2. **Registers** the SGLang backend with the Orla daemon (OpenAI-compatible API at `http://sglang:30000/v1`). SGLang must be started with `--tool-call-parser qwen` (the deploy compose includes it).
3. **Adds a single tool**, `run_bash`, that runs one bash command in the current instance’s workdir (`/workdir/<instance_id>`) and returns stdout, stderr, and exit code.
4. **For each instance**: ensures the repo is cloned and at `base_commit`, builds a user message from the problem statement, runs a ReAct-style loop (`ExecuteWithMessages` and `RunToolCallsInResponse`) until the model stops or the step limit is reached, then writes one prediction line (the **git diff** of the workdir) to `predictions.jsonl`.

No per-run arguments: the full SWE-bench Lite benchmark runs in one container invocation. This is the same pattern as [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md): **ExecuteWithMessages** and **RunToolCallsInResponse** implement the ReAct loop.

## Conclusion 

You’ve run the Orla SWE-bench Lite baseline with SGLang and one bash tool. For tool-calling basics, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For a minimal SGLang-only run, see [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md).
