# Capturing Stage I/O

The telemetry Orla records on every call tells you a stage was slow or expensive. It does not tell you what the stage saw or what it produced. This tutorial turns on Orla's per-stage capture, which stores the request and response content of a stage, and reads it back grouped by workflow run. You will run a small agent, capture both of its stages, and inspect exactly what one stage handed to the next.

Capture is the tool for attribution. When a two-stage agent gives a wrong answer, capture is how you tell whether the first stage surfaced the wrong information or the second stage misused the right information. That is a question the metadata alone cannot answer.

The tutorial uses a self-contained example that ships in the Orla repo, so you do not need to have worked through the other v2 tutorials first. You do need the Orla daemon running and the `orlactl` command on your PATH. The [quickstart](v2/quickstart.md) walks through both. Leave the daemon running in its own terminal and run everything below from a second one.

## A two-stage agent

The example is a question-answering agent with two stages, written as `retrieve -> answer`. The `retrieve` stage lists the facts a question needs, and the `answer` stage writes the answer from those facts. A stage is Orla's term for a named point in a workflow that you route to a model on its own.

```bash
git clone https://github.com/harvard-cns/orla
cd orla/examples/capture-io
```

Both stages need a backend before they can run. Register one with `orlactl backend create` as the quickstart shows, then map each stage to it. This loop points both stages at the same backend, named `anthropic-haiku` here.

```bash
for s in retrieve answer; do orlactl stage map $s anthropic-haiku; done
```

Each command connects a stage to the backend, so a call tagged with that stage is served by that model. The agent tags every call with its stage and lets Orla decide the model, so the agent code never names one.

## Turn capture on

Capture is a per-stage switch called `capture_io`, and it is off by default. Turn it on for both stages with `orlactl stage capture`.

```bash
for s in retrieve answer; do orlactl stage capture $s on; done
```

Each call confirms the new state.

```
capture for stage "retrieve" is on
capture for stage "answer" is on
```

`orlactl stage list` shows the switch under a `CAPTURE` column, so you can see at a glance which stages are recording.

```bash
orlactl stage list
```

```
STAGE     BACKEND          PROMPT  CAPTURE
answer    anthropic-haiku  -       on
retrieve  anthropic-haiku  -       on
```

The switch is live. The next call on either stage records its content, and turning it off stops the recording without a restart or a code change, the same way stage mapping and stage prompts work.

## Run the agent and read the capture

Run the example.

```bash
uv run run.py
```

The script sends one question through `retrieve -> answer`. Both calls carry the same workflow run id, through the `X-Orla-Workflow-Run` header, so Orla groups them as one run. The script then reads the captured content back through `GET /api/v1/workflows/{run}/completions` and prints the request and response for each stage. It also sets `capture_io` on itself through the control API, so the example works even if you skipped the step above.

```
capture on for retrieve, answer

question:     Which planet in our solar system has the most known moons?
answer:       Saturn has the most known moons of any planet in our solar system.
workflow run: 3f9c1a2b7e6d4f08a1c5b9e2d7043a6c

=== retrieve  (completion chatcmpl-a1) ===
  request:  {"model":"orla","messages":[{"role":"system","content":"List the facts needed to answer the question, one per line. Do not answer it."},{"role":"user","content":"Which planet in our solar system has the most known moons?"}], ...}
  response: {"id":"chatcmpl-a1","choices":[{"message":{"role":"assistant","content":"- The number of known moons for each planet\n- Saturn's known moon count\n- Jupiter's known moon count"}}], ...}

=== answer  (completion chatcmpl-a2) ===
  request:  {"model":"orla","messages":[{"role":"system","content":"Answer the question in one sentence, using only the facts provided."},{"role":"user","content":"Question: Which planet in our solar system has the most known moons?\n\nFacts:\n- The number of known moons for each planet\n- Saturn's known moon count\n- Jupiter's known moon count"}], ...}
  response: {"id":"chatcmpl-a2","choices":[{"message":{"role":"assistant","content":"Saturn has the most known moons of any planet in our solar system."}}], ...}
```

Read the two blocks together and the handoff is visible. The `retrieve` stage's response is a list of facts, and those same facts appear inside the `answer` stage's request, under `Facts:`. The request side is the raw body the agent sent, and the response side is the full response Orla returned. This is the attribution view: if the answer were wrong, you could see whether `retrieve` failed to name the right fact or `answer` ignored a fact it was given.

## Read a run yourself

The script reads the run through an endpoint you can call directly. Pass the workflow run id the script printed.

```bash
curl 'http://localhost:8081/api/v1/workflows/3f9c1a2b7e6d4f08a1c5b9e2d7043a6c/completions'
```

The response is one object per captured stage of that run.

```json
{
  "completions": [
    {
      "completion_id": "chatcmpl-a1",
      "workflow_run": "3f9c1a2b7e6d4f08a1c5b9e2d7043a6c",
      "stage_id": "retrieve",
      "request_content": "{\"model\":\"orla\",\"messages\":[ ... ]}",
      "response_content": "{\"id\":\"chatcmpl-a1\",\"choices\":[ ... ]}",
      "created_at": "2026-07-13T18:30:01Z"
    },
    {
      "completion_id": "chatcmpl-a2",
      "workflow_run": "3f9c1a2b7e6d4f08a1c5b9e2d7043a6c",
      "stage_id": "answer",
      "request_content": "{\"model\":\"orla\",\"messages\":[ ... ]}",
      "response_content": "{\"id\":\"chatcmpl-a2\",\"choices\":[ ... ]}",
      "created_at": "2026-07-13T18:30:02Z"
    }
  ]
}
```

`stage_id` names the stage and `completion_id` matches the id on the completion record, so you can join a run's content back to its cost and latency. `request_content` is the request body and `response_content` is the response, each captured only for a stage with `capture_io` on. A stage with capture off contributes no object, so the endpoint returns only the stages you opted in. Either field is null when Orla captured only the other side, for example a call that errored before it produced a response.

## Where the content lives

Captured content is kept apart from the metadata Orla records on every call. The lean per-call record, with tokens, cost, latency, and status, stays in the `completion_records` table that the mapper reads. The request and response bodies go to a separate `completion_io` table with its own access control and retention, so an operator can grant a mapper the metrics without granting it the content, and can purge captured bodies on a shorter schedule.

The capture write is best-effort. If it fails, Orla logs the loss and moves on without touching the metadata write or the response to the caller, so turning capture on cannot slow or break a live call. Rows lost this way are counted in the `orla_completion_io_drops_total` metric, so you can tell whether any content is being dropped.

## Turn it off

Capture stays on until you turn it off, and content keeps accruing for every call on those stages, so turn it off when you are done inspecting.

```bash
for s in retrieve answer; do orlactl stage capture $s off; done
```

`orlactl stage list` now shows `off` in the `CAPTURE` column for both stages, and the next call on either one records nothing.

## Toward diagnostics

Capturing by hand is the manual version of what a diagnostic does automatically. A diagnostic is a process that turns capture on for the stages of a workflow, runs a set of questions, reads each run's content back through the workflow endpoint, and scores where the workflow went wrong, whether the retrieval surfaced the right evidence or the composer used it. Orla ships the capture switch and the read endpoint that a diagnostic needs, and keeps the content in a table it can query, so the diagnostic lives outside the agent and needs no change to the agent's code. When it finishes, you turn capture back off and the workflow runs lean again.

## Takeaways

Capture is a per-stage switch you control from Orla, off by default and reversible. In this tutorial you turned it on for both stages of an agent, ran one question, and read the run back to see the facts one stage produced arriving in the next stage's request. You saw that the content lives in a table separate from the metadata, that the write is best-effort so it cannot harm a live call, and that a single endpoint returns a whole run's input and output grouped by workflow run. The agent's code never changed. Orla recorded what each stage saw and produced, on demand, because you asked it to.
