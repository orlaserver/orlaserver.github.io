---
layout: default
title: Orla
---

Orla is a server and orchestrator for agents and agentic workflows that sits above LLM backends such as Ollama, SGLang, and vLLM. It lets users specify inference controls at agent-level granularity. These controls include which model and backend to use per step, how to manage inference state (e.g. KV cache), and how to coordinate multiple agents or workflow stages.

LLM serving systems like SGLang, vLLM, and Ollama optimize at the *request* level using speculative decoding, prefill/decode, batching, per-request cache, and other techniques. Agentic systems decompose tasks into multiple steps that share context, have different computational requirements, and may span heterogeneous backends. Orla addresses that gap. It optimizes for *end-to-end* agent completion time and cost, with policies expressed per agent profile and per workflow stage.


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