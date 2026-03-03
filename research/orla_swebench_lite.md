# Tutorial: Experiment with Orla Stage Mapping with SWE-bench Lite

This tutorial is for anyone who wants to research, reproduce, extend, or test Orla’s stage mapping and scheduling functionality. It runs [Orla](https://github.com/dorcha-inc/orla) SWE-bench Lite experiments in Docker: a **text-based** ReAct loop (model outputs THOUGHT + bash code blocks; we parse and execute) against [SWE-bench Lite](https://www.swebench.com/lite.html) instances. Three experiments are available:

- **baseline** — single model (Qwen3-8B).
- **two_stage_mapping** — router on light model, then light or heavy model per instance; light and heavy backends run **concurrently**.
- **two_stage_mapping_complexity_sched** — same routing as above, plus **complexity-aware scheduling** on the heavy backend: a `ScorePredictor` estimates task complexity (1–5), and the heavy queue is sorted simplest-first (Shortest Job First) using Orla’s Priority scheduling.

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for SGLang.
- **Orla repo** cloned so you can run from the repo root.

## What is SWE-bench Lite?

[SWE-bench Lite](https://www.swebench.com/lite.html) is a curated benchmark of 300 test instances (plus 23 dev instances) from real GitHub issues. Each instance has a **problem statement**, a **repository**, and a **base commit**. The agent’s job is to produce a patch that fixes the issue. The agent uses a **text-based** loop (no tool-calling API): the model outputs a THOUGHT and a single bash command in a code block; we parse it, run it in the instance workdir, and feed the output back. This repeats until the model stops or hits the step limit.


## 1. Build the images (first time or after code changes)

From the **Orla repo root**, build the Orla and experiment-runner images used by the compose file:

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml build
```

SGLang services use the pre-built `lmsysorg/sglang:latest` image and do not need to be built.

## 2. Start the stack and run an experiment

Create the output directory and start the stack with the experiment you want. Compose will start the SGLang services (heavy and light model) and Orla, then run the experiment.

**Baseline** (single model, Qwen3-8B):

```bash
mkdir -p deploy/output
docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

**Two-stage mapping** (router on light model, then light or heavy model per instance; light and heavy run in parallel):

```bash
mkdir -p deploy/output
RUN_TARGET=two_stage_mapping docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

**Two-stage mapping with complexity scheduling** (same routing + SJF scheduling on heavy backend):

```bash
mkdir -p deploy/output
RUN_TARGET=two_stage_mapping_complexity_sched docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

If the run container already exists (e.g. you ran baseline before), add `--force-recreate` so it picks up the new `RUN_TARGET`: 

```bash
RUN_TARGET=two_stage_mapping docker compose -f deploy/docker-compose.swebench-lite.yaml up --force-recreate
```

With `sudo`, use: 

```bash
sudo env RUN_TARGET=two_stage_mapping docker compose -f deploy/docker-compose.swebench-lite.yaml up --force-recreate
```

This starts all services (SGLang, Orla, and the experiment runner) and attaches to their logs in the foreground. Predictions are written to **`deploy/output/predictions.jsonl`** and metrics (timing and token counts per instance/step) to **`deploy/output/metrics.json`** (set `METRICS_PATH` to use a different path). 

To run an experiment again without bringing the stack up first (with the stack already running via `up -d`), use 

```bash
docker compose run --rm -e RUN_TARGET=baseline run
```

or 

```bash
docker compose run --rm -e RUN_TARGET=two_stage_mapping run
```

or

```bash
docker compose run --rm -e RUN_TARGET=two_stage_mapping_complexity_sched run
```

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

The image includes the dataset as `dataset.zip` (unzipped at build time to `/dataset`). The baseline runs instances from `/dataset/test`. Source: [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite).

Predictions are one JSON object per line (instance_id, model_name_or_path, model_patch). You can run the [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/) against the output file.

## How the experiments work

All experiments live under `examples/swe_bench_lite/` and use shared helpers from `shared/` (including metrics recording). They are invoked via `make baseline`, `make two_stage_mapping`, or `make two_stage_mapping_complexity_sched` (or the same targets as the container command).

**Baseline** (`baseline/`, `cmd/baseline`):

1. **Loads instances** from `/dataset/test` (sorted order).
2. **Registers** one SGLang backend (Qwen3-8B at `http://sglang:30000/v1`) with the Orla daemon.
3. **For each instance**: prepares the workdir, runs a **text-based ReAct loop**: call `ExecuteWithMessages`, parse the first `\`\`\`orla_bash` code block from the response, execute that command in `/workdir/<instance_id>`, append the output as a user message, repeat until no bash block is found or the step limit. Appends one prediction (git diff) to `predictions.jsonl`. Writes **metrics** (end-to-end, per-instance, per-step times and token counts) to `metrics.json`.

**Two-stage mapping** (`two_stage_mapping/`, `cmd/two_stage_mapping`):

1. **Registers** two SGLang backends: heavy (Qwen3-8B) and light (Qwen3-4B).
2. **Phase 1**: For each instance, prepares the workdir and runs a **router** (on the **light** model) to classify the task as light or heavy; builds a light queue and a heavy queue.
3. **Phase 2**: Runs **one worker for the light backend** and **one worker for the heavy backend** concurrently. Each worker drains its queue (same text-based ReAct loop as baseline), so light and heavy instances are processed in parallel. Appends predictions to `predictions.jsonl` (model_name_or_path `orla-two-stage`) and writes metrics to `metrics.json`.

No per-run arguments: each experiment runs in one container invocation. The ReAct loop uses **ExecuteWithMessages** and parses bash commands from the model’s text output (no tool-calling API).

**Two-stage mapping with complexity scheduling** (`two_stage_mapping_complexity_sched/`, `cmd/two_stage_mapping_complexity_sched`):

1. **Registers** the same two SGLang backends as two-stage mapping: heavy (Qwen3-8B) and light (Qwen3-4B).
2. **Phase 1 – Routing**: Same as two-stage mapping: each instance is routed to light or heavy via the `OneBitStageMapper`.
3. **Phase 1b – Complexity prediction**: For each **heavy** instance, a `ScorePredictor` (backed by the light model) estimates task complexity on a 1–5 scale. The scheduling priority is set to `6 - complexity`, implementing **Shortest Job First (SJF)**: simpler tasks get higher priority and are scheduled before complex ones on the heavy backend.
4. **Phase 2**: Heavy jobs are sorted by priority (simplest first). Light and heavy workers run concurrently, same as two-stage mapping. Each heavy request carries its priority via `SchedulingHints`, and the heavy backend uses `SchedulingPolicyPriority` to schedule accordingly.
5. **Metrics**: In addition to the standard per-instance timing and token counts, this experiment records `complexity` (predicted 1–5 score) and `queue_position` (position in the sorted heavy queue) for each instance.

The hypothesis is that SJF scheduling reduces **average instance completion time** and **queue wait time** on the heavy backend, since shorter tasks no longer wait behind longer ones. The additional overhead is one lightweight inference call per heavy instance for complexity prediction.

## Conclusion

You’ve run the Orla SWE-bench Lite experiments (baseline, two-stage mapping, and/or complexity scheduling) with SGLang using the text-based agent loop. For tool-calling with Orla, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For a multi-agent workflow demo, see [Multi-Agent Workflow](tutorials/tutorial-workflow-customer-support.md). For a minimal SGLang-only run, see [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md).
