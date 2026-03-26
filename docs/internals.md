# Orla Internals

This section is for hands-on, end-to-end guides after you are comfortable with the basics. If you are new to Orla, start with [Overview](overview.md), [Quickstart](quickstart.md) (LangGraph + pyorla), and [Configuration](configuration.md). For the daemon HTTP API, see [`docs/openapi.yaml`](https://github.com/harvard-cns/orla/blob/main/docs/openapi.yaml) in the Orla repository.

The pages below walk through running Orla against real backends (vLLM, Ollama, SGLang), tool calling, workflow-oriented features (concurrent stages, memory manager), and research-style experiments (benchmarks and evaluation workflows). They include config snippets, compose or launch commands, and example programs. For LangGraph + pyorla, use the [Quickstart](quickstart.md).

## Backend setup

Pick the stack that matches your environment; each tutorial registers the backend with `orla serve` and runs a minimal flow.

- [Using Orla with vLLM](tutorials/tutorial-vllm-lily.md)
- [Using Orla with Ollama](tutorials/tutorial-ollama-lily.md)
- [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md)

## Tools and workflows

- [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md): tool definitions and execution across backends
- [Concurrent Stages](tutorials/tutorial-concurrent-stages.md): parallelism and backend concurrency limits
- [Memory Manager](tutorials/tutorial-memory-manager.md): KV cache lifecycle across workflow stages

## Research tutorials

Experiment write-ups and reproducible evaluation workflows:

- [Single-Shot SWE-bench Experiments](research/orla_swebench_lite.md)
- [DAG-Math Memory Management Evaluation](research/orla_dag_math_memory_eval.md)
- [Multi-Stage Workflow (Customer Support)](research/orla_workflow_customer_support.md)
