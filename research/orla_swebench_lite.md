# Tutorial: Experiment with Orla Stage Mapping with SWE-bench Lite

This tutorial is for anyone who wants to research, reproduce, extend, or test Orla’s stage mapping functionality. It runs [Orla](https://github.com/dorcha-inc/orla) SWE-bench Lite experiments in Docker: one **run_bash** tool and a ReAct-style agent loop against [SWE-bench Lite](https://www.swebench.com/lite.html) instances. Two experiments are available: **baseline** (single model) and **two_stage_mapping** (router + light/heavy model by task complexity).

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for SGLang.
- **Orla repo** cloned so you can run from the repo root.

## What is SWE-bench Lite?

[SWE-bench Lite](https://www.swebench.com/lite.html) is a curated benchmark of 300 test instances (plus 23 dev instances) from real GitHub issues. Each instance has a **problem statement**, a **repository**, and a **base commit**. The agent’s job is to produce a patch that fixes the issue. The baseline uses a single **run_bash** tool: the model runs commands (e.g. explore the repo, edit files, run tests), and the results are appended until the model stops or hits a step limit.


## 1. Start the stack and run an experiment

From the **Orla repo root**, create the output directory and start the stack with the experiment you want. Compose will start the SGLang services (heavy and light model) and Orla, then run the experiment.

**Baseline** (single model, Qwen3-8B):

```bash
mkdir -p deploy/output
docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

**Two-stage mapping** (router on heavy model, then light or heavy model per instance):

```bash
mkdir -p deploy/output
RUN_TARGET=two_stage_mapping docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

This starts all services (SGLang, Orla, and the experiment runner) and attaches to their logs in the foreground. Predictions are written to **`deploy/output/predictions.jsonl`** and timing metrics to **`deploy/output/metrics.json`** (set `METRICS_PATH` to use a different path). To run an experiment again without bringing the stack up first, use `docker compose run --rm run baseline` or `docker compose run --rm run two_stage_mapping` (with the stack already running via `up -d`).

## 2. Stop the stack

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

The image includes the dataset as `dataset.zip` (unzipped at build time to `/dataset`). The baseline runs instances from `/dataset/test`. Source: [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite).

Predictions are one JSON object per line (instance_id, model_name_or_path, model_patch). You can run the [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/) against the output file.

## How the experiments work

Both experiments live under `examples/swe_bench_lite/` and use shared helpers from `shared/` (including metrics recording). They are invoked via `make baseline` or `make two_stage_mapping` (or the same targets as the container command).

**Baseline** (`baseline/`, `cmd/baseline`):

1. **Loads instances** from `/dataset/test` (sorted order).
2. **Registers** one SGLang backend (Qwen3-8B at `http://sglang:30000/v1`) with the Orla daemon. SGLang must use `--tool-call-parser qwen` (the deploy compose includes it).
3. **Adds** the `run_bash` tool (runs one bash command in the instance workdir `/workdir/<instance_id>`).
4. **For each instance**: prepares the workdir, runs a ReAct loop (`ExecuteWithMessages` and `RunToolCallsInResponse`) until the model stops or the step limit, then appends one prediction (git diff) to `predictions.jsonl`. Writes **metrics** (end-to-end, per-instance, per-step times) to `metrics.json`.

**Two-stage mapping** (`two_stage_mapping/`, `cmd/two_stage_mapping`):

1. **Registers** two SGLang backends: heavy (Qwen3-8B) and light (Qwen3-4B).
2. **For each instance**: runs a **router** (heavy model) to classify the task as light or heavy, then runs the same ReAct loop as the baseline using the **light** or **heavy** model accordingly. Appends predictions to `predictions.jsonl` (model_name_or_path `orla-two-stage`) and writes metrics to `metrics.json`.

No per-run arguments: each experiment runs in one container invocation. This matches [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md): **ExecuteWithMessages** and **RunToolCallsInResponse** implement the ReAct loop.

## Conclusion

You’ve run the Orla SWE-bench Lite experiments (baseline and/or two-stage mapping) with SGLang and one bash tool. For tool-calling basics, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For a minimal SGLang-only run, see [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md).
