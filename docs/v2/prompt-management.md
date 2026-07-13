# Managing Prompts

This tutorial adds a second lever to the workflow from [Static Workflows](v2/static-workflows.md). There, you routed each stage to a model from outside the agent. Here you set each stage's prompt from outside the agent too. You will override the instructions a stage runs under, from Orla, in one command, and watch the agent's behavior change without editing a line of its code.

The mapping and the prompt are the two levers on how a stage behaves. The mapping decides which model serves the stage. The prompt decides the instructions that model runs under. You already pulled the first lever. This is the second.

The tutorial continues with the same HotpotQA agent from the static-workflows tutorial, so work through that one first if you have not. You need the Orla daemon running, at least one backend registered, and the `select -> hop -> answer` stages mapped to it. Run everything below from a second terminal, in `orla/examples/hotpotqa-distractor`.

## The prompt the agent already sends

Every call the agent makes carries a system prompt. You can see it in `agent.py`, where each stage sends a system message with its instructions and a user message with the task.

```python
messages=[
    {"role": "system", "content": system},
    {"role": "user", "content": user},
],
extra_headers={"X-Orla-Stage": stage},
```

That `system` string is the stage's default prompt, written into the agent. When a stage has no prompt set in Orla, this default passes through untouched, which is how the agent has behaved so far. Setting a stage prompt in Orla overrides it.

## Override a stage's prompt

Look at the `answer` stage. Its job is to reduce the hop's draft to a short final answer. HotpotQA scores an exact match, so how tersely the stage answers matters. A draft that says "The answer is Steven Spielberg." scores worse than one that says "Steven Spielberg", even though both are right.

Set a prompt on the `answer` stage that forces the terse form.

```bash
orlactl stage prompt answer \
  "Output only the exact answer span, with no leading words and no trailing punctuation. For a yes or no question, output yes or no."
```

Orla confirms the prompt is stored.

```
set prompt for stage "answer" (135 chars)
```

Now run the example again. The agent code has not changed, and it still sends its own default system prompt on every `answer` call, but Orla now replaces that message with the prompt you just set.

```bash
uv run run.py
```

The `hop` stage still does the reasoning, so the agent finds the same answers. What changes is the `answer` stage's formatting, and because exact-match scoring is sensitive to formatting, the exact-match number moves. You are tuning the workflow through the prompt, from outside the agent, the same way you tuned it through the mapping.

## Inspect what is set

`orlactl stage list` shows which stages carry a prompt, so you can tell at a glance which stages Orla is overriding.

```bash
orlactl stage list
```

```
STAGE     BACKEND          PROMPT  CAPTURE
answer    anthropic-haiku  set     off
hop       anthropic-haiku  -       off
select    anthropic-haiku  -       off
```

The `PROMPT` column reads `set` for a stage with a prompt and `-` for one without. The `CAPTURE` column reads `on` when a stage is recording its request and response, and `off` here because none of these stages is. The [Capturing Stage I/O](v2/capturing-io.md) tutorial covers that switch.

`orlactl stage get answer` shows the full record, including the prompt text.

```bash
orlactl stage get answer
```

```json
{
  "id": "answer",
  "backend": "anthropic-haiku",
  "reasoning_effort": "",
  "prompt": "Output only the exact answer span, with no leading words and no trailing punctuation. For a yes or no question, output yes or no.",
  "capture_io": false,
  "labels": {}
}
```

## How the override applies

The rule Orla follows is "the stage's prompt is the leading instruction message." The instruction message is the system or developer message the request opens with, since SDKs differ on which role they use for instructions. This agent sends a system message, and the Vercel AI SDK used elsewhere in Meddit sends a developer message, so Orla accepts either. On a call tagged with a stage that has a prompt set, Orla replaces that first message and keeps its role, or prepends a system message when the first message is neither. Everything after the leading instruction message is left alone.

That last point matters for multi-step stages. A tool-calling loop sends the same system message on every step, alongside a growing record of what it has done so far. Orla swaps that system message on each step and leaves the record intact, so the instructions change while the accumulated work survives. The `answer` stage here is a single call, so there is only one message to swap, but the same rule carries a retrieval loop or an agent that calls tools.

The override is opt-in. A stage with no prompt forwards the agent's own messages exactly as before, so setting a prompt on `answer` leaves `select` and `hop` untouched. Set a stage prompt only when the stage sends its instructions as a leading system or developer message, which this agent does for every stage. An agent that folds its instructions into a user turn instead should leave the stage prompt empty and keep applying the prompt itself.

## Clear it

Clearing the prompt returns the stage to the agent's own default.

```bash
orlactl stage prompt answer --clear
```

```
cleared prompt for stage "answer"
```

The next `answer` call sends the agent's built-in system prompt again, with nothing from Orla in the way.

You can also set a longer prompt from a file, which is easier than quoting a paragraph on the command line.

```bash
orlactl stage prompt answer --file answer-prompt.txt
```

## Toward optimizing prompts

Setting a prompt by hand is to the prompt lever what mapping a stage by hand is to the routing lever. Both are the manual version of what an optimizer does automatically. An optimizer is a separate process that reads a stage's accuracy and cost from the feedback and metrics you saw in the static-workflows tutorial, proposes a better prompt, and writes it onto the stage the same way you just did. Orla ships the API surface an optimizer needs, not the optimizer itself, the same as it does for the mapper that re-routes stages. The next call runs the new prompt. Because the prompt lives in Orla rather than in the agent, an optimizer can evolve a prompt that exists nowhere in the agent's code and put it live without a redeploy.

## Takeaways

A stage has two levers, and both live in Orla, not in the agent. You already routed each stage to a model. In this tutorial you set a stage's prompt from Orla, watched the agent's behavior change with no edit to its code, inspected and cleared the override, and saw how the same move an optimizer makes for the mapping applies to the prompt. The agent kept sending its own default prompt the whole time. Orla decided when to override it.
