# Tutorial: Single-Shot SWE-bench Lite Experiments

This tutorial runs [Orla](https://github.com/dorcha-inc/orla) single-shot SWE-bench Lite experiments in Docker. Each of the 300 [SWE-bench Lite](https://www.swebench.com/lite.html) instances gets **one inference call** containing the problem statement and oracle-provided source files. All instances are submitted **concurrently**, stressing Orla's server-side scheduling. Three experiment modes are available:

- **baseline** — all instances go to a single heavy model (Qwen3-8B), FCFS scheduling, submitted concurrently.
- **stage_mapping** — `OneBitStageMapper` routes each instance to a light model (Qwen3-4B) or the heavy model, FCFS scheduling per backend, concurrent submission.
- **sjf** — same routing as stage_mapping, plus **Shortest Job First** priority scheduling on the heavy backend: shorter prompts (by character count) are scheduled first, since prompt length correlates with prefill time and likely output length. No predictor LLM call is needed.

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for SGLang or vLLM.
- **Orla repo** cloned so you can run from the repo root.

## What is SWE-bench Lite?

[SWE-bench Lite](https://www.swebench.com/lite.html) is a curated benchmark of 300 test instances from real GitHub issues. Each instance has a **problem statement**, a **repository**, and a **base commit**. The task is to produce a unified diff patch that fixes the issue.

In our single-shot design, we provide the model with **oracle context**: the source files that the gold patch modifies, read from the repository at the base commit. The model must produce a correct unified diff in a single response.

## 1. Build the images (first time or after code changes)

From the **Orla repo root**, build the Orla and experiment-runner images:

**SGLang** (default):

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml build
```

**vLLM**:

```bash
docker compose -f deploy/docker-compose.swebench-lite.vllm.yaml build
```

## 2. Start the stack and run an experiment

Create the output directory and start the stack with the experiment you want. Use `docker-compose.swebench-lite.yaml` for **SGLang** or `docker-compose.swebench-lite.vllm.yaml` for **vLLM**.

### SGLang

**Baseline** (single heavy model, FCFS):

```bash
mkdir -p deploy/output
RUN_TARGET=single_shot_baseline docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

**Stage mapping** (router on light model, FCFS per backend):

```bash
mkdir -p deploy/output
RUN_TARGET=single_shot_stage_mapping docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

**SJF** (stage mapping + shortest-job-first on heavy):

```bash
mkdir -p deploy/output
RUN_TARGET=single_shot_sjf docker compose -f deploy/docker-compose.swebench-lite.yaml up
```

### vLLM

The same experiments work with vLLM. Use `docker-compose.swebench-lite.vllm.yaml` instead; the vLLM compose sets `BACKEND_PROVIDER=vllm` and the backend URLs automatically.

**Baseline**:

```bash
mkdir -p deploy/output
RUN_TARGET=single_shot_baseline docker compose -f deploy/docker-compose.swebench-lite.vllm.yaml up
```

**Stage mapping**:

```bash
mkdir -p deploy/output
RUN_TARGET=single_shot_stage_mapping docker compose -f deploy/docker-compose.swebench-lite.vllm.yaml up
```

**SJF**:

```bash
mkdir -p deploy/output
RUN_TARGET=single_shot_sjf docker compose -f deploy/docker-compose.swebench-lite.vllm.yaml up
```

### Tips

If the run container already exists, add `--force-recreate` so it picks up the new `RUN_TARGET`:

```bash
RUN_TARGET=single_shot_sjf docker compose -f deploy/docker-compose.swebench-lite.yaml up --force-recreate
```

Predictions are written to **`deploy/output/predictions.jsonl`** and metrics to **`deploy/output/metrics.json`**. To re-run with the stack already up (`up -d`):

```bash
docker compose run --rm -e RUN_TARGET=single_shot_baseline run
docker compose run --rm -e RUN_TARGET=single_shot_stage_mapping run
docker compose run --rm -e RUN_TARGET=single_shot_sjf run
```

## 3. Stop the stack

```bash
docker compose -f deploy/docker-compose.swebench-lite.yaml down
```

Use `down -v` to remove the model cache volume.

## Instance JSON format

Each instance is a single JSON object with:

- **`instance_id`** — Identifier (e.g. `django__django-11099`).
- **`repo`** — Repository (e.g. `django/django`).
- **`base_commit`** — Git commit the agent should work from.
- **`problem_statement`** — The issue description (what to fix).
- **`patch`** — The gold unified diff (used for oracle context gathering).

The image includes the dataset as `dataset.zip` (unzipped at build time to `/dataset`). Source: [princeton-nlp/SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite). To regenerate with the latest data, run `python examples/swe_bench_lite/scripts/prepare_dataset.py`.

Predictions are one JSON object per line (`instance_id`, `model_name_or_path`, `model_patch`). You can run the [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/) against the output file.

## How the experiments work

All experiments are in a single binary (`single_shot`) under `examples/swe_bench_lite/`. The mode is selected by the `EXPERIMENT_MODE` environment variable (set automatically by the Makefile targets). The binary reads the dataset, prepares workdirs, builds prompts with oracle context, and submits all instances concurrently to Orla.

**Phase 1 — Prompt building** (all modes):

For each instance, the experiment parses the gold patch to extract modified file paths, reads those files from the repository at the base commit, and assembles a single-shot prompt: system message (produce a unified diff), problem statement, repository info, and the full source of each relevant file.

**Phase 1b — Routing** (stage_mapping and sjf modes):

The `OneBitStageMapper` classifies each instance as "light" or "heavy" by asking the light model whether the fix looks simple (single file, clear bug, config tweak) or complex (multiple files, unclear root cause, API changes). Light instances go to the Qwen3-4B backend; heavy instances go to Qwen3-8B.

**Phase 1c — SJF priority assignment** (sjf mode only):

Heavy instances are sorted by prompt length (ascending). Each heavy instance gets `priority = maxPromptLen - len(prompt)`, so shorter prompts receive higher priority. The heavy stage is configured with `SchedulingPolicyPriority`, and each request carries its priority via `SchedulingHints`. Orla's server-side scheduler picks the highest-priority request first.

**Phase 2 — Concurrent submission** (all modes):

All instances are submitted simultaneously as goroutines, each calling `stage.Execute`. Orla's server-side queues handle contention. This creates the realistic multi-agent scenario where scheduling policy matters.

**Metrics collected per instance:**

| Metric | Description |
|--------|-------------|
| `prompt_length` | Prompt length in characters |
| `prompt_tokens` | Prompt tokens (from response) |
| `completion_tokens` | Completion tokens (from response) |
| `queue_position` | Position in submission order for that backend |
| `queue_wait_ms` | Time spent in Orla's server queue |
| `ttft_ms` | Time to first token |
| `tpot_ms` | Time per output token |
| `duration_ms` | Wall clock from submit to response |
| `backend_latency_ms` | Backend-reported latency |
| `mapped_stage` | Which backend (light/heavy/single) |
| `priority` | Scheduling priority (sjf mode) |

The SJF hypothesis is that shorter prompts correlate with faster inference (less prefill, often shorter output). By scheduling them first, the heavy backend clears quick jobs faster, reducing average completion time and queue wait time across all instances.

## Conclusion

You've run the Orla single-shot SWE-bench Lite experiments (baseline, stage mapping, and/or SJF scheduling). For tool-calling with Orla, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For a multi-agent workflow demo, see [Multi-Agent Workflow (Customer Support)](research/orla_workflow_customer_support.md). For a minimal SGLang-only run, see [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md).
