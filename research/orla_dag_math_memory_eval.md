# Tutorial: DAG-Math Memory Management Evaluation

This tutorial runs [Orla](https://github.com/dorcha-inc/orla) memory management experiments using the [DAG-MATH-Formatted-CoT](https://huggingface.co/datasets/liminho123/DAG-MATH-Formatted-CoT) dataset. Each of the 2,894 mathematical reasoning problems is modeled as an Orla **workflow DAG**, where each reasoning step becomes a separate stage with dependencies matching the original proof structure. Two experiment modes compare KV cache flushing strategies:

- **flush_per_request** (baseline) -- every stage gets `CachePolicy: "flush"`, so the KV cache is evicted after each LLM call. This simulates a system with no cross-request memory management.
- **flush_per_workflow** -- Orla's default policy preserves KV cache across stages within a workflow and only flushes when the entire workflow completes. This is Orla's Memory Manager in action.

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for SGLang.
- **Orla repo** cloned so you can run from the repo root.

## What is DAG-Math?

[DAG-Math](https://arxiv.org/pdf/2510.19842) is a benchmark of 2,894 mathematical reasoning problems from [Omni-MATH](https://omni-math.github.io/). Each problem's solution is represented as a **directed acyclic graph (DAG)** where:

- **Nodes** are conclusions (intermediate or final results).
- **Edges** are inferences that derive a conclusion from premises.
- **Dependencies** (`direct_dependent_steps`) link each step to the prior steps it builds on.

This DAG structure maps naturally to Orla's workflow model: each step becomes a `Stage`, and `direct_dependent_steps` becomes `AddDependency()`. Independent steps execute concurrently; dependent steps wait for their upstream stages.

## How the experiment works

Each DAG-Math problem is converted into an Orla workflow:

1. For each step in the problem, a `Stage` is created with a prompt containing the problem text, the step's inference (edge), and the step's conclusion (node).
2. Dependencies from `direct_dependent_steps` are wired via `Workflow.AddDependency()`.
3. Upstream stage results are passed into dependent stages via `MessagesBuilder`, so each stage can reference prior conclusions.
4. In **flush_per_request** mode, `stage.SetCachePolicy("flush")` is called on every stage, forcing KV cache eviction after each LLM call.
5. In **flush_per_workflow** mode, no override is set, so Orla's default Memory Manager policy applies: `PreserveOnSmallIncrementPolicy` keeps the KV cache across stages, and `FlushAtBoundaryPolicy` flushes only when the workflow completes.

Workflows run **sequentially** (one at a time) to isolate KV cache effects. This ensures that cache hits/misses are attributable to the flushing strategy, not interference from concurrent workflows.

**Model:** Qwen3-8B via SGLang (single GPU).

## 1. Prepare the dataset (first time only)

If you don't already have the `dataset.zip` in the `examples/dag_math_eval/` directory, generate it:

```bash
pip install datasets
python examples/dag_math_eval/scripts/prepare_dataset.py
```

This downloads the dataset from HuggingFace and creates `examples/dag_math_eval/dataset.zip`.

## 2. Build the images

From the **Orla repo root**:

```bash
docker compose -f deploy/docker-compose.dag-math.yaml build
```

## 3. Run the experiments

Create the output directory and start the stack with the desired experiment mode.

**Flush per workflow** (Orla default -- preserves KV cache across stages):

```bash
mkdir -p deploy/output
RUN_TARGET=dag_math_flush_per_workflow docker compose -f deploy/docker-compose.dag-math.yaml up
```

**Flush per request** (baseline -- evicts KV cache after every stage):

```bash
mkdir -p deploy/output
RUN_TARGET=dag_math_flush_per_request docker compose -f deploy/docker-compose.dag-math.yaml up
```

### Tips

To run a quick subset for testing, set `MAX_INSTANCES`:

```bash
MAX_INSTANCES=50 RUN_TARGET=dag_math_flush_per_workflow docker compose -f deploy/docker-compose.dag-math.yaml up
```

If the run container already exists, add `--force-recreate`:

```bash
RUN_TARGET=dag_math_flush_per_request docker compose -f deploy/docker-compose.dag-math.yaml up --force-recreate
```

Metrics are written to **`deploy/output/metrics.json`**. To re-run with the stack already up (`up -d`):

```bash
docker compose run --rm -e RUN_TARGET=dag_math_flush_per_workflow run
docker compose run --rm -e RUN_TARGET=dag_math_flush_per_request run
```

## 4. Stop the stack

```bash
docker compose -f deploy/docker-compose.dag-math.yaml down
```

Use `down -v` to remove the model cache volume.

## Dataset JSON format

Each problem is a single JSON object with:

- **`problem_id`** -- Integer identifier.
- **`problem_text`** -- The mathematical problem statement.
- **`final_answer`** -- The expected answer.
- **`difficulty`** -- Numeric difficulty from 1 (easiest) to 6 (hardest).
- **`domain`** -- Topic taxonomy (e.g., `["Mathematics -> Number Theory -> ..."]`).
- **`steps`** -- List of step objects, each with:
  - **`step_id`** -- Unique integer within the problem.
  - **`edge`** -- The logical inference from premises to conclusion.
  - **`node`** -- The conclusion of this step.
  - **`direct_dependent_steps`** -- List of `step_id` values this step depends on (null for root steps).

Source: [liminho123/DAG-MATH-Formatted-CoT](https://huggingface.co/datasets/liminho123/DAG-MATH-Formatted-CoT). To regenerate, run `python examples/dag_math_eval/scripts/prepare_dataset.py`.

## Metrics collected per workflow

| Metric | Description |
|--------|-------------|
| `problem_id` | DAG-Math problem identifier |
| `num_stages` | Number of stages (steps) in the workflow |
| `difficulty` | Problem difficulty (1--6) |
| `duration_ms` | Wall clock time for the entire workflow |
| `total_prompt_tokens` | Sum of prompt tokens across all stages |
| `total_completion_tokens` | Sum of completion tokens across all stages |

Per-stage metrics within each workflow:

| Metric | Description |
|--------|-------------|
| `step_id` | DAG-Math step identifier |
| `prompt_tokens` | Prompt tokens for this stage |
| `completion_tokens` | Completion tokens for this stage |
| `queue_wait_ms` | Time spent in Orla's server queue |
| `ttft_ms` | Time to first token |
| `tpot_ms` | Time per output token |
| `backend_latency_ms` | Backend-reported latency |

## What to expect

The **flush_per_workflow** mode should show lower per-workflow latency than **flush_per_request**, particularly for workflows with many stages that share context. When Orla's Memory Manager preserves the KV cache across stages, subsequent stages benefit from cached prefix tokens, reducing prefill time. This effect is most visible in:

- **TTFT (Time to First Token)**: Should be lower for non-root stages in flush_per_workflow mode, since the shared prefix is already in the KV cache.
- **Total workflow duration**: Cumulative savings from reduced prefill across all stages.
- **Higher-difficulty problems**: These tend to have more stages and deeper dependency chains, amplifying the cache benefit.

## Conclusion

You've run the Orla DAG-Math memory management evaluation comparing request-level and workflow-level KV cache flushing. For other Orla experiments, see [Single-Shot SWE-bench Experiments](research/orla_swebench_lite.md). For a multi-stage workflow demo, see [Multi-Stage Workflow (Customer Support)](research/orla_workflow_customer_support.md).
