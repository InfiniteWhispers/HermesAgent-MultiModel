---
name: mixture-of-agents
description: "Run /localmoa or @localmoa to query multiple local Ollama models in parallel and synthesize a single answer via the local_moa CLI (~/.hermes/bin/local_moa.py)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
triggers:
  - "/localmoa <question>"
  - "@localmoa <question>"
  - "ask local_moa about X"
  - "ask local_moa to explain X"
  - "use local_moa to examine X"
  - "run local_moa on X"
  - "what does local_moa say about X"
metadata:
  hermes:
    tags: [moa, mixture-of-agents, ollama, local, localmoa, synthesis]
    related_skills: [tool-routing-decomposition]
---

# Mixture-of-Agents (local_moa)

`local_moa` sends one prompt to several local Ollama models in parallel,
anonymizes their replies (`Model 1`, `Model 2`, …), then asks an aggregator
model to synthesize a single best answer.

## Invocation Patterns

All of these trigger this skill — run the CLI and return the result:

| Pattern | Example |
|---|---|
| `/localmoa <question>` | `/localmoa explain RSA vs DSA` |
| `@localmoa <question>` | `@localmoa what are the risks of MXFP4 quantization?` |
| `ask local_moa about X` | `ask local_moa about gradient descent` |
| `ask local_moa to explain X` | `ask local_moa to explain transformer attention` |
| `use local_moa to examine X` | `use local_moa to examine this config` |
| `run local_moa on X` | `run local_moa on this question` |
| `what does local_moa say about X` | `what does local_moa say about RSA` |

The `X` or `<question>` becomes the `--prompt` value verbatim.

## On-the-fly Configuration via Inline Flags

`/localmoa` and `@localmoa` accept inline flags appended directly to the invocation.
Parse them from the user's message and pass to the CLI verbatim:

```
/localmoa explain gradient descent --references gpt-oss-20b qwen3:14b qwythos-9b --aggregator qwen3:14b --temperature 0.3 --verbose
```

```
@localmoa what are the main RSA attack vectors? --system "You are a cybersecurity expert." --temperature 0.3 --timeout 240
```

Supported inline flags (all optional — omit to use defaults):

| Flag | What it controls |
|---|---|
| `--references M1 M2 …` | Which models draft in parallel (Ollama pull tags) |
| `--aggregator MODEL` | Which model synthesizes (Ollama pull tag) |
| `--system "TEXT"` | System prompt for reference models only |
| `--temperature FLOAT` | 0.2–0.4 factual; unset for general/creative |
| `--timeout INT` | Per-request timeout in seconds |
| `--num-ctx INT` | Context window (default: 65536) |
| `--verbose` / `-v` | Print each model's raw response before synthesis |

If no flags are provided, use the defaults below.

## Default Models

```bash
# Default invocation — no flags needed
~/.hermes/bin/local_moa.py --prompt "..."
# References:  qwen3:14b  ornith-9b
# Aggregator:  gpt-oss-20b
```

> NOTE: local_moa requires Ollama pull tags, not Hermes aliases.
> Alias → Ollama tag mapping for your roster:
>
> | Hermes alias        | Ollama pull tag          |
> |---------------------|--------------------------|
> | qwen3-14b-think     | qwen3:14b                |
> | qwythos-9b          | qwythos-9b               |
> | gpt-oss-20b         | gpt-oss-20b              |
> | ornith-9b           | ornith-9b                |
> | gemma4-heretic-12b  | gemma4-heretic-12b       |
> | qwen25-coder-7b     | qwen2.5-coder:7b         |
> | qwen3-vl-8b         | qwen3-vl:8b              |

## How to Run It

Always use the full path — the bare symlink may not be present:

```bash
~/.hermes/bin/local_moa.py --prompt "{question}"
```

> **Run it in the background.** local_moa cold-loads multiple Ollama models on a
> shared GPU and routinely takes longer than the terminal tool's foreground timeout
> (causing `[Command timed out after Ns]` failures from the agent). When invoking via
> the terminal tool, set **`background=true` and `notify_on_complete=true`**, and pass
> **`--timeout 240`** to the script. You'll be notified with the synthesized answer when
> it finishes instead of blocking and timing out.

### Common Invocations

```bash
# Default (qwen3:14b + ornith-9b → gpt-oss-20b aggregator)
~/.hermes/bin/local_moa.py --prompt "Explain RSA vs DSA"

# Security research — uncensored drafters, reasoning aggregator
~/.hermes/bin/local_moa.py \
  --prompt "Main attack vectors against RSA implementations?" \
  --system "You are a cybersecurity expert." \
  --references gpt-oss-20b qwythos-9b \
  --aggregator qwen3:14b \
  --temperature 0.3 \
  --timeout 240

# Three drafters for maximum coverage
~/.hermes/bin/local_moa.py \
  --prompt "Architecture tradeoffs for event-driven vs request-response systems" \
  --references gpt-oss-20b qwen3:14b qwythos-9b \
  --aggregator gpt-oss-20b \
  --timeout 240

# Agentic coding review — coding-focused drafters
~/.hermes/bin/local_moa.py \
  --prompt "Review this Python script for correctness and edge cases" \
  --references ornith-9b qwen2.5-coder:7b \
  --aggregator gpt-oss-20b

# Debug — see each model's raw answer before synthesis
~/.hermes/bin/local_moa.py --prompt "Explain gradient descent" --verbose
```

## Options

| Flag | Default | Purpose |
|---|---|---|
| `--prompt TEXT` | *(required)* | Question sent to every reference model |
| `--references M1 M2 …` | `qwen3:14b ornith-9b` | Reference models (Ollama pull tags) |
| `--aggregator MODEL` | `gpt-oss-20b` | Synthesizer model (Ollama pull tag) |
| `--system TEXT` | *(none)* | System prompt for reference models only |
| `--temperature FLOAT` | model default | 0.2–0.4 factual; unset for creative |
| `--num-ctx INT` | `65536` | Context window tokens |
| `--timeout INT` | `120` | Per-request timeout (s); aggregator gets +60s |
| `--ollama-url URL` | `http://localhost:11434/api/chat` | Override endpoint |
| `-v`, `--verbose` | off | Print raw responses before aggregation |

## Requirements

- Ollama running on `localhost:11434` (override via `OLLAMA_BASE_URL`)
- All referenced models pulled in Ollama — if missing, `ollama pull <tag>`
- If a reference model times out or fails, run continues with remaining models;
  fails only if ALL reference models fail

## Tips

- Cold-loading a model on 16 GB GDDR7 takes 60–90s — use `--timeout 240`
  when running 3+ models that may not all be pre-loaded
- `--temperature 0.2–0.4` for factual/security work; omit for general reasoning
- `--system` scopes to reference models only — aggregator sees anonymized drafts
  plus the original question, not the system prompt
- `--verbose` is your best debugging tool — shows what each model actually said
  before synthesis so you can spot where the aggregator diverged
- qwythos-9b as a reference drafter adds uncensored analytical depth that other
  models hedge on — especially useful for security and exploit-adjacent research
- A reasoning-capable aggregator (qwen3:14b) often synthesizes better than
  reusing the same model as both reference and aggregator

## More Detail

- `~/.hermes/bin/local_moa.py --help` — authoritative option list
- `~/.hermes/bin/local_moa_guide.md` — full guide (architecture, examples, env vars)
- `references/local_moa_usage.md` — quick usage notes alongside this skill
