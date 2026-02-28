# Tutorial: Running Orla with SWE-bench Lite

This tutorial walks through running [Orla](https://github.com/dorcha-inc/orla) with [SWE-bench Lite](https://www.swebench.com/lite.html) using the setup in the Orla repository. You’ll start the Orla daemon and SGLang, then run an agent that uses a bash tool to work through SWE-bench Lite instances in a ReAct-style loop.

## What you need

- **Docker and Docker Compose** (Compose V2).
- **NVIDIA GPU** and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so the SGLang container can use the GPU.
- **Orla repo** cloned so you can run the example and deploy from the repo root.
- **Go 1.25 or later** to run the Orla SWE-bench Lite example.

If you’ve already completed [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md), you have the same stack; this tutorial adds the SWE-bench Lite example and instance format.

## What is SWE-bench Lite?

[SWE-bench Lite](https://www.swebench.com/lite.html) is a curated benchmark of 300 test instances (plus 23 dev instances) from real GitHub issues. Each instance has a **problem statement**, a **repository**, and a **base commit**. The agent’s job is to produce a patch that fixes the issue. The Orla example uses a single **bash tool**: the model calls it to run commands (e.g. explore the repo, edit files, run tests), and the results are appended to the conversation until the model stops or hits a step limit.

## 1. Start SGLang and Orla

From the root of the Orla repository:

```bash
cd orla
docker compose -f deploy/docker-compose.sglang.yaml up -d
```

This starts:

- **SGLang** on port 30000 (Ollama-compatible API). Default model: `Qwen/Qwen3-8B`.
- **Orla** on port 8081. The example program will register the SGLang backend and run the agent.

Check that the daemon is up:

```bash
curl http://localhost:8081/api/v1/health
```

You should get a successful response (e.g. HTTP 200).

## 2. Run the example with a sample instance

The Orla repo includes an example that loads a SWE-bench Lite–style instance, registers SGLang, attaches a **run_bash** tool, and runs an agent loop (execute → tool calls → run tools → append results → repeat).

From the **Orla repo root**, run with the bundled sample instance (no repo clone needed; good for testing the loop):

```bash
go run ./examples/swe_bench_lite/cache_strategy/orla_sglang \
  -instance ./examples/swe_bench_lite/cache_strategy/sample_instance.json \
  -max-steps 10
```

You’ll see the model receive the problem statement, call `run_bash` to run commands, and eventually finish. The final model output is printed at the end.

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-orla-url` | Orla daemon URL | `http://localhost:8081` |
| `-instance` | Path to SWE-bench Lite instance JSON | (required) |
| `-workdir` | Working directory for the bash tool | current directory |
| `-max-steps` | Maximum agent steps | 25 |
| `-output` | JSONL file to append predictions (for evaluation) | (none) |

## 3. Instance JSON format

Each instance is a single JSON object. The example expects at least:

- **`instance_id`** – Identifier (e.g. `django__django-11099`).
- **`repo`** – Repository (e.g. `django/django`).
- **`base_commit`** – Git commit the agent should work from.
- **`problem_statement`** – The issue description (what to fix).

The sample file is at `examples/swe_bench_lite/cache_strategy/sample_instance.json`. You can download real instances from the [SWE-bench Lite dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) or [SWE-bench/SWE-bench_Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) and point `-instance` at a local JSON file.

## 4. Running with a real repository (optional)

To run the agent inside a real repo (so bash commands see the codebase):

1. **Clone the repo** and check out the instance’s **base_commit**:
   ```bash
   git clone https://github.com/owner/repo.git /path/to/workdir
   cd /path/to/workdir
   git checkout <base_commit from instance JSON>
   ```

2. **Run the example** with `-workdir` and, if you want to collect predictions for evaluation, `-output`:
   ```bash
   go run ./examples/swe_bench_lite/cache_strategy/orla_sglang \
     -instance /path/to/instance.json \
     -workdir /path/to/workdir \
     -max-steps 25 \
     -output predictions.jsonl
   ```

Predictions are appended as one JSON object per line (instance_id, model_name_or_path, model_patch). You can then run the [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/) against that file.

## 5. How the example works

The example in `examples/swe_bench_lite/cache_strategy/orla_sglang/main.go`:

1. **Registers** the SGLang backend with the Orla daemon (Ollama-compatible, `http://sglang:30000`).
2. **Adds a single tool**, `run_bash`, that runs one bash command in `-workdir` and returns stdout, stderr, and exit code.
3. **Loads the instance** and builds a user message with the problem statement, repo, and base commit.
4. **Runs a loop**: `ExecuteWithMessages` → if the model returns tool calls, runs them via the agent, appends assistant + tool-result messages, and repeats until there are no tool calls or `-max-steps` is reached.

This is the same pattern as [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md): the agent uses **ExecuteWithMessages** and **RunToolCallsInResponse** to implement a ReAct-style loop.

## 6. Comparing with mini-SWE-agent

The same directory includes a **mini-SWE-agent** setup so you can run SWE-bench Lite with the same SGLang backend for comparison. mini-SWE-agent is a minimal Python agent (bash-only tools, linear history) used as a baseline. See `examples/swe_bench_lite/cache_strategy/README.md` in the Orla repo for:

- Installing `mini-swe-agent` and `mini-swe-agent-extra`.
- Setting `OLLAMA_HOST=http://localhost:30000` and running `run_swebench_lite.sh` to drive mini-SWE-agent against SGLang.

## 7. Stop the stack

When you’re done:

```bash
docker compose -f deploy/docker-compose.sglang.yaml down
```

Use `down -v` if you also want to remove the SGLang model cache volume.

---

You’ve run Orla with SWE-bench Lite: SGLang as the backend, one bash tool, and an agent loop over instance problem statements. For tool-calling basics, see [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md). For a minimal SGLang-only run, see [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md).
