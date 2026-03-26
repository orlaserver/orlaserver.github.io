# Orla Internals

This section is for hands-on, end-to-end guides after you are comfortable with the basics. If you are new to Orla, start with [Getting Started](getting-started.md), [Usage](usage.md), and the [Python SDK (pyorla)](pyorla.md). They cover install, CLI, HTTP API, and client libraries without assuming a specific GPU stack.

The pages below walk through running Orla against real backends (vLLM, Ollama, SGLang), tool calling, workflow-oriented features (concurrent stages, memory manager), LangGraph integration, and research-style experiments (benchmarks and evaluation workflows). They include config snippets, compose or launch commands, and example programs.

## Backend setup

Pick the stack that matches your environment; each tutorial registers the backend with `orla serve` and runs a minimal flow.

- [Using Orla with vLLM](tutorials/tutorial-vllm-lily.md)
- [Using Orla with Ollama](tutorials/tutorial-ollama-lily.md)
- [Using Orla with SGLang](tutorials/tutorial-sglang-lily.md)

## Tools and workflows

- [Using Tools with Orla](tutorials/tutorial-tools-vllm-ollama-sglang.md): tool definitions and execution across backends
- [Concurrent Stages](tutorials/tutorial-concurrent-stages.md): parallelism and backend concurrency limits
- [Memory Manager](tutorials/tutorial-memory-manager.md): KV cache lifecycle across workflow stages

## Python / LangGraph

- [Using Orla with LangGraph](tutorials/tutorial-langgraph.md): `StateGraph`, multi-backend stages, and tools via pyorla

## Research tutorials

Experiment write-ups and reproducible evaluation workflows:

- [Single-Shot SWE-bench Experiments](research/orla_swebench_lite.md)
- [DAG-Math Memory Management Evaluation](research/orla_dag_math_memory_eval.md)
- [Multi-Stage Workflow (Customer Support)](research/orla_workflow_customer_support.md)
