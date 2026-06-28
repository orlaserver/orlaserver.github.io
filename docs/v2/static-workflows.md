# Static Workflows

This tutorial takes a multi-stage agent and runs it through Orla, unchanged, on three different model providers: a model on your laptop, a model on AWS Bedrock, and a model on Anthropic. You will watch the same agent produce different accuracy and cost on each, swap a provider in a single command, and then run different stages of the workflow on different providers. The agent code never changes. Every routing decision happens in Orla.

The agent is a multi-hop question-answering agent for [HotpotQA](https://hotpotqa.github.io/), and it ships in the Orla repo under `examples/hotpotqa-distractor`.

## A three-stage workflow

A multi-hop question takes more than one hop to answer. "What year did the director of Jaws release his first feature film?" means first finding the director, then finding that person's first film. HotpotQA's distractor setting gives the agent ten passages for each question, two that hold the answer and eight that are there to mislead, and asks it to reason across them.

The agent answers every question with the same three **stages**, in the same order, written here as `select -> hop -> answer`. A stage is Orla's term for a named point in a workflow that you route to a model on its own, separately from the others.

- **select** reads the ten passages and picks the two or three that are relevant.
- **hop** reasons across those passages, one hop at a time, and drafts an answer.
- **answer** reduces the draft to a short final answer, such as a name or a yes or no.

This shape never changes from one question to the next. There is no branching, and no loop whose length depends on the input. That is what makes the workflow static, and it is the simplest kind to run on Orla, because every stage is known before the first call.

The agent tags each call with its stage name and lets Orla decide which model serves it. Each stage returns a typed result rather than free text, so the next stage reads structured fields. Every provider in this tutorial supports structured output over the OpenAI protocol, which is why the one agent runs unchanged against all of them.

The agent that runs this workflow ships in the Orla repo. Clone it and move into the example directory, where the agent code and the runner script live.

```bash
git clone https://github.com/harvard-cns/orla
cd orla/examples/hotpotqa-distractor
```

You run the example with `uv run run.py`. That script loads a sample of HotpotQA questions, runs each one through the agent, scores the agent's answer against the known correct answer, and reports the score back to Orla. Two environment variables control it: `ORLA_BASE_URL`, where the Orla daemon listens, defaulting to `http://localhost:8081/v1`, and `N`, how many questions to run, defaulting to 10. Ten questions is enough to see the behavior without spending much, and it is the sample used throughout this tutorial.

None of this works without a running Orla daemon and the `orlactl` command on your PATH. The [quickstart](v2/quickstart.md) walks through both. Leave the daemon running in its own terminal and run everything below from a second one.

## Inference providers

A **backend** is one model endpoint Orla can send calls to. Before you can route a stage anywhere, you have to tell Orla about at least one backend, which you do with `orlactl backend create`.

Register whichever of the three providers below you have access to. Any single one is enough to follow the rest of the tutorial. They differ only in the endpoint, the model name, and the credential.

One detail is shared by all three. The `--model` flag takes a value of the form `label:model-name`. Orla keeps the part before the first colon as a free label for your own reference and sends the part after it to the endpoint as the real model name. So `ollama:llama3.2:1b` routes to the model `llama3.2:1b`.

Orla reads each backend's API key from an environment variable, named by `--api-key-env`. That variable has to be set in the shell that started the daemon, because the daemon is what reads it when it forwards a call. Export your keys before you run `orla serve`.

### A model on your laptop, with Ollama

Ollama serves models locally on an OpenAI-compatible endpoint. It needs no API key and costs nothing to call, which makes it the easiest place to start. Pull a small model, then register it as a backend.

```bash
ollama pull llama3.2:1b

orlactl backend create --name ollama-llama \
  --endpoint http://localhost:11434/v1 \
  --model ollama:llama3.2:1b \
  --api-key-env OLLAMA_API_KEY --max-concurrency 2
```

The first command downloads the model into your local Ollama server. The second registers it with Orla as a backend named `ollama-llama`, the handle you will use when mapping stages. `--endpoint` is the OpenAI-compatible URL Orla calls, on Ollama's default port 11434. `--model ollama:llama3.2:1b` is the model id, so following the convention above Orla labels the backend `ollama` and asks the endpoint for `llama3.2:1b`. `--api-key-env OLLAMA_API_KEY` names the variable Orla would read a key from, but Ollama needs none, so that variable can stay unset. `--max-concurrency 2` caps how many calls Orla sends at once, so a small local model is not overwhelmed. There is no cost flag, because local inference is free.

### A model on AWS Bedrock

Bedrock exposes an OpenAI-compatible endpoint and authenticates with a bearer token you place in `AWS_BEARER_TOKEN_BEDROCK`. `gemma-3-12b-it` is one of its cheaper models.

```bash
orlactl backend create --name bedrock-gemma \
  --endpoint https://bedrock-mantle.us-east-2.api.aws/v1 \
  --model openai:google.gemma-3-12b-it \
  --api-key-env AWS_BEARER_TOKEN_BEDROCK \
  --max-concurrency 4 --input-cost 0.090 --output-cost 0.290 --rate 2
```

This registers the Bedrock model as a backend named `bedrock-gemma`. `--endpoint` is Bedrock's OpenAI-compatible URL, and `--model openai:google.gemma-3-12b-it` asks that endpoint for `google.gemma-3-12b-it`. `--api-key-env AWS_BEARER_TOKEN_BEDROCK` is the variable holding your Bedrock bearer token. `--input-cost` and `--output-cost` are the model's price in dollars per million input and output tokens, copied from the provider's pricing page. Orla multiplies them by the tokens each call uses and stores the dollar cost on the call, which is what lets you compare providers on price later. `--rate 2` caps the backend at two requests per second, and `--max-concurrency 4` at four calls in flight at once.

### A model on Anthropic

Anthropic also speaks the OpenAI protocol and authenticates with `ANTHROPIC_API_KEY`. `claude-haiku-4-5` is its cheap, fast tier.

```bash
orlactl backend create --name anthropic-haiku \
  --endpoint https://api.anthropic.com/v1 \
  --model anthropic:claude-haiku-4-5-20251001 \
  --api-key-env ANTHROPIC_API_KEY \
  --max-concurrency 4 --input-cost 1.0 --output-cost 5.0 --rate 2
```

This registers Anthropic's Haiku as a backend named `anthropic-haiku`. `--model anthropic:claude-haiku-4-5-20251001` asks the endpoint for `claude-haiku-4-5-20251001`, `--api-key-env ANTHROPIC_API_KEY` points at the variable holding your Anthropic key, and the cost, rate, and concurrency flags work exactly as they did for Bedrock. Haiku costs more per token than Gemma, which the cost comparison later reflects.

## Routing the workflow

Registering a backend does not route anything to it yet. You connect the workflow's three stages to a backend with `orlactl stage map STAGE BACKEND`. This loop maps all three stages, `select`, `hop`, and `answer`, to the Anthropic backend.

```bash
for s in select hop answer; do orlactl stage map $s anthropic-haiku; done
```

Now run the example. Every call the agent makes carries a stage name, and Orla now sends all three stages to `anthropic-haiku`.

```bash
uv run run.py
```

You will see one line per question and a summary at the end.

```
F1 1.00  pred='Yes'                  gold='yes'
F1 1.00  pred='Animorphs'            gold='Animorphs'
...
10 questions  |  EM 70%  |  answer F1 0.861
```

Each line shows the agent's answer, `pred`, next to the correct answer, `gold`, and an F1 score for how much they overlap, where 1.00 is an exact match. The summary reports two standard HotpotQA metrics over the whole sample. **EM** is the fraction of answers that match exactly. **Answer F1** is the average word overlap. Higher is better for both, and this run scores 70% EM and 0.861 F1 on Anthropic's Haiku.

Notice that the agent named a stage on each call and never named a model. Orla resolved each stage to `anthropic-haiku` and recorded the score against all three stages as feedback, which the later sections build on.

Because routing lives in Orla and not in the agent, moving the whole workflow to a different provider is just three more `stage map` calls. Point the same three stages at the Bedrock backend and run again.

```bash
for s in select hop answer; do orlactl stage map $s bedrock-gemma; done
uv run run.py
```

Run it once more against `ollama-llama` the same way. Across the three providers, on the same ten questions, the results come out like this:

| Backend | Answer F1 | Cost for 10 questions |
|---|---|---|
| `ollama-llama` (local) | 0.475 | free |
| `bedrock-gemma` | 0.624 | ~$0.002 |
| `anthropic-haiku` | 0.861 | ~$0.033 |

This is the comparison Orla exists to make visible. Anthropic's model costs about sixteen times as much as the Bedrock one, and on this task it earns the difference in accuracy. The local model costs nothing and stays accurate enough for a quick check. The agent cannot see any of this, and producing the comparison took no change to it.

## Routing each stage differently

So far every run sent all three stages to one backend, but stages route independently and do not have to. Look at what each stage does. The **hop** does the real reasoning, while **select** and **answer** are nearly mechanical. A sensible split, then, is to spend the expensive model only on the hop and leave the cheap model on the other two.

```bash
orlactl stage map select bedrock-gemma
orlactl stage map hop    anthropic-haiku
orlactl stage map answer bedrock-gemma
uv run run.py
```

```
10 questions  |  EM 70%  |  answer F1 0.757
```

This mix costs about $0.013, roughly a third of the all-Anthropic price, and it keeps most of the accuracy. Searching for splits like this by hand is exactly the job an optimizer does automatically, and the lever it pulls, the stage-to-backend mapping, is the same one you just used.

## Cost, latency, and feedback

Orla records every call it routes, and you can read the aggregates for any stage. This asks for the per-backend numbers on the `hop` stage.

```bash
curl 'http://localhost:8081/api/v1/stages/hop/metrics'
```

```json
{"metrics":[
  {"backend":"anthropic-haiku","count":10,"avg_latency_ms":2397,"total_cost_usd":0.011075,"error_count":0},
  {"backend":"bedrock-gemma","count":10,"avg_latency_ms":1319,"total_cost_usd":0.000536,"error_count":0}
]}
```

Each row is one backend that has served this stage. It shows how many calls the backend handled in `count`, its average latency, the total dollars it cost, and how many calls errored. This is the cost-and-speed half of the picture.

The quality half comes from the scores the example reported after each answer. The same per-stage view exposes that feedback.

```bash
curl 'http://localhost:8081/api/v1/stages/hop/feedback'
```

An optimizer reads both halves. It weighs each backend's cost and latency against the quality scores and re-maps every stage to the backend that holds quality for the least money. That is the automated version of the manual swapping you did in the last two sections.

## Takeaways

A static workflow runs the same stages in the same order every time, and Orla turns each stage into a routing choice you make from outside the agent. In this tutorial you ran one unchanged agent on a local model, on Bedrock, and on Anthropic, compared their accuracy and cost, and then split the workflow so the expensive model ran only where it mattered. The [example's README](https://github.com/harvard-cns/orla/tree/main/examples/hotpotqa-distractor) has the full agent and scorer if you want to read the code behind it.
