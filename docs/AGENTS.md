# AGENTS.md

Guidance for writing the Orla documentation. The CLAUDE.md symlink
resolves to this file. Everything under `docs/` follows these rules:
the quickstart, the tutorials, and the concept and reference pages.
Read it before writing or editing a page.

The reader is a developer meeting Orla for the first time. Write so
they can follow a page top to bottom and run it without guessing.

## Voice

Write in active voice and full, connected sentences. Lead with the
noun or the result, not the qualifier. Address the reader as "you". A
page should read like prose a person follows, not a list of clipped
notes.

Do not write in fragments or in a punchy, aphoristic style. Short
clipped clauses strung together read like a parable, not like
documentation.

Wrong:

> No key, no cost, runs on a laptop.

Right:

> Ollama serves models locally, so it needs no API key and costs
> nothing to call.

Wrong:

> This is the picture Orla exists to give you.

Right:

> These numbers are the cost and accuracy comparison Orla makes
> visible.

## Mechanics

These are hard rules, not preferences.

- No em-dashes. The character `—` never appears in prose. Split into
  two sentences or use a comma. The same goes for en-dashes.
- No semicolons in prose. Use a period and start a new sentence.
- No unnecessary parentheses. A parenthetical aside that pauses the
  reader belongs in its own sentence. Parens are fine for a genuine
  clarification, such as a default value or a one-word gloss.
- No ASCII diagrams. Describe relationships in prose. A single inline
  arrow like `select -> hop -> answer` is fine, boxes and arrows are
  not.
- No emoji unless the page explicitly calls for them.

## Explain every code block

Never drop a code block into a page without saying what it does.
Before or after each block, explain what the commands do and what
every meaningful flag means. The reader should understand the block
without already knowing the tool. Show command output too, and when
you do, say what its columns or fields mean.

Wrong:

> ```bash
> orlactl backend create --name ollama-llama \
>   --endpoint http://localhost:11434/v1 \
>   --model ollama:llama3.2:1b --api-key-env OLLAMA_API_KEY --max-concurrency 2
> ```
> `--max-concurrency 2` caps concurrency.

Right:

> The command registers the model with Orla as a backend named
> `ollama-llama`, the handle you map stages to. `--endpoint` is the
> OpenAI-compatible URL Orla calls, `--model ollama:llama3.2:1b` asks
> the endpoint for `llama3.2:1b`, `--api-key-env` names the variable
> Orla reads the key from, and `--max-concurrency 2` caps how many
> calls Orla sends at once.

## Headings

A heading names what a section contains. Make it a meaningful noun
phrase, such as "Inference providers" or "Cost, latency, and
feedback".

Avoid two failure modes. The first is the narrating heading, such as
"Now we register the backends" or "Map the stages and run". State the
subject, not the act of doing it. The second is the rhetorical
heading, such as "What makes a workflow static". Answer the question
in prose rather than posing it as a title.

Do not over-chunk. A heading breaks the reader's flow, so add one
only where a genuinely new section begins. When two short sections are
really one idea, merge them and let a sentence carry the transition.
Prefer a handful of meaningful sections over a heading every few
lines.

## Terminology

Use Orla's vocabulary exactly and consistently. A backend is one
model endpoint. A stage is a named point in a workflow that you route
to a backend. You map a stage to a backend. Say "stages", not
"steps", when you mean Orla stages.

Define any term the reader may not know on its first use, then use the
same word for it everywhere. Do not switch synonyms mid-page.

## Cut filler

Remove words that earn nothing. If "static" already carries the
meaning, do not also write "fixed". Do not lean on one adjective
across a passage. Read each sentence back and delete the words that
would not be missed.
