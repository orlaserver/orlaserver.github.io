---
layout: default
title: Orla
---

## Overview

Large Language Models (LLMs) are increasingly used in agentic systems that perform specialized tasks 
through iterative loops of inference and tool invocation. These systems introduce serving challenges 
that differ from traditional LLM workloads. They have separate execution stages that differ in computational requirements
and they can coordinate across heterogeneous backends including datacenter GPUs and edge devices. 
They also benefit from managing inference-state lifecycles such as  KV cache management at agent boundaries 
rather than request boundaries. Existing LLM serving systems like SGLang, vLLM, and Ollama optimize at the 
request level and cannot address these agent-level concerns.

Orla is an execution engine for agents and agentic workflows that that sits above LLM backends such as Ollama, SGLang, and vLLM. It lets users specify inference controls at agent-level granularity. These controls include which model and backend to use per step, how to manage inference state (e.g. KV cache), and how to coordinate multiple agents or workflow stages. LLM serving systems optimize at the *request* level using speculative decoding, prefill/decode, batching, per-request cache, and other techniques. Agentic systems decompose tasks into multiple steps that share context, have different computational requirements, and span heterogeneous backends. Orla addresses that gap. It optimizes for *end-to-end* agent completion time and cost, with policies expressed per agent profile and per workflow stage.

<!-- Orla is a server and orchestrator for agents and agentic workflows that sits above LLM backends such as Ollama, SGLang, and vLLM. It lets users specify inference controls at agent-level granularity. These controls include which model and backend to use per step, how to manage inference state (e.g. KV cache), and how to coordinate multiple agents or workflow stages.

LLM serving systems like SGLang, vLLM, and Ollama optimize at the *request* level using speculative decoding, prefill/decode, batching, per-request cache, and other techniques. Agentic systems decompose tasks into multiple steps that share context, have different computational requirements, and may span heterogeneous backends. Orla addresses that gap. It optimizes for *end-to-end* agent completion time and cost, with policies expressed per agent profile and per workflow stage. -->


## Tutorials

- [Run a simple agent with Orla and vLLM]({{ '/tutorial-vllm-lily.html' | relative_url }})

## Source code

Orla's [github repository](https://github.com/dorcha-inc/orla).

## Team

The Orla agentic server was originally developed in [Dr. Minlan Yu](https://minlanyu.seas.harvard.edu/)'s lab at Harvard University as part of a broader research investigation on agentic systems.

- [Hayder Tirmazi](https://jadidbourbaki.github.io/jadidbourbaki/): developer and maintainer of Orla.
- [Dr. Rana Shahout](https://sites.google.com/view/ranash): primary researcher of the agentic systems project at Harvard.
- [Dr. Minlan Yu](https://minlanyu.seas.harvard.edu/): principal investigator.

## Sponsors and Acknowledgements

We would like to thank [Dr. Minlan Yu](https://minlanyu.seas.harvard.edu/)'s lab at Harvard, the [Kempner Institute for the Study of Natural and Artificial Intelligence](https://kempnerinstitute.harvard.edu/) at Harvard, and
[Akamai Technologies](https://www.akamai.com/) for providing the compute resources that made developing Orla and conducting research with it possible. We would like to thank the [Craig McLuckie](https://www.linkedin.com/in/craigmcluckie/) (founder of kubernetes) for early insight into Orla server and providing us with some sage advice on
developing and releasing open-source software.

## Preliminary Benchmarks

![End-to-end completion time: Orla vs SGLang (datacenter) and Orla vs Ollama (edge)](end_to_end_comparison.png)

Figure: End-to-end task completion time for 20 tasks across four configurations. Orla-optimized configurations show significant improvements: 29.6\% faster for SGLang and 47.9\% faster for Ollama compared to their respective baselines. Error bars show standard deviation across 3 runs (warmup run excluded).

We benchmark Orla on a Github Issue solving agent. Our evaluation uses a three-stage workflow inspired by [SWE-Bench](https://github.com/SWE-bench/SWE-bench) that processes software engineering tasks:

- Issue Analysis: Understand the problem and identify what needs to be fixed.
- Code Generation: Generate the fixed code with proper error handling.
- Summary: Summarize the fix in 2-3 sentences.

We use 20 realistic software engineering issues covering bug fixes, optimizations, and security patches.

We compare four configurations. The first two configurations evaluate a datacenter setting with SGLang. The SGLang baseline 
uses Mistral-7B-Instruct-v0.3 directly via SGLang with Orla serving only as a lightweight agent-level orchestrator that 
routes all agent task to SGLang. The Orla-enabled configuration takes advantage of Orla's agent profiles and 
routes the two light-weight agent stages, Analysis and Summary, to use a smaller model Qwen2.5-0.5B-Instruct on SGLang, and utilizes
the large model Mistral-7B only for code generation.

The second two configurations evaluate an edge setting with Ollama. The Ollama baseline uses Mistral-7B directly via Ollama with Orla again serving only as a lightweight agent-level orchestrator that 
routes all agent task to Ollama. The Orla-enabled configuration behaves similarly to the datacenter setting, but uses Ollama instead of SGLang.

All experiments run on the same hardware, an AMD EPYC 7313 server equipped with an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. 
We measure latency per stage, total task completion time, and inference cost calculated from token counts across all stages. We run each configuration 4 times.
We discard the first run as a warmup and report the mean and standard deviation
for the 3 remaining non-warmup runs.