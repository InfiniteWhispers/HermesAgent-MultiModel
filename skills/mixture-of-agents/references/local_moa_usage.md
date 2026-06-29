# local_moa Usage Guide

## Invocation Shortcuts

| Shortcut | Behavior |
|---|---|
| `/localmoa <question> [flags]` | Run local_moa CLI with question + any inline flags |
| `@localmoa <question> [flags]` | Same as /localmoa |

Inline flags are parsed from the user message and passed to the CLI verbatim.
All `--references`, `--aggregator`, `--system`, `--temperature`, `--timeout`,
`--num-ctx`, and `--verbose` flags are supported inline.

## Defaults

- `--references` default: `qwen3:14b ornith-9b`
- `--aggregator` default: `gpt-oss-20b`
- `--num-ctx` default: `65536`
- `--timeout` default: `120s` per reference; aggregator gets `+60s`

## Key Behaviors

- Run `~/.hermes/bin/local_moa.py --help` to view all options
- `--references` specifies Ollama pull tags queried in parallel. Order sets
  `Model 1`, `Model 2`, … which the aggregator sees (anonymized)
- `--aggregator` is the model that synthesizes responses
- `--verbose` (`-v`) prints each model's raw answer before aggregation —
  essential for debugging synthesis quality
- Ollama service must be running at `http://localhost:11434/api/chat`;
  override with `OLLAMA_BASE_URL` env var
- Default timeout: 120s per reference call, +60s for aggregator.
  Adjust with `--timeout` when running 3+ cold models (cold load = 60–90s)

## Alias → Ollama Tag Mapping

Model tags must be Ollama pull tags, not Hermes aliases:

| Hermes alias        | Ollama pull tag    |
|---------------------|--------------------|
| qwen3-14b-think     | qwen3:14b          |
| qwythos-9b          | qwythos-9b         |
| gpt-oss-20b         | gpt-oss-20b        |
| ornith-9b           | ornith-9b          |
| gemma4-heretic-12b  | gemma4-heretic-12b |
| qwen25-coder-7b     | qwen2.5-coder:7b   |
| qwen3-vl-8b         | qwen3-vl:8b        |

## Examples

```bash
# Default — qwen3:14b + ornith-9b drafters, gpt-oss-20b aggregator
~/.hermes/bin/local_moa.py --prompt "Top 3 agentic platforms like Hermes"

# Explicit models
~/.hermes/bin/local_moa.py \
  --prompt "Explain RSA vs DSA" \
  --references gpt-oss-20b qwen3:14b \
  --aggregator gpt-oss-20b \
  --verbose

# Security research with system prompt
~/.hermes/bin/local_moa.py \
  --prompt "Main attack vectors against RSA implementations?" \
  --system "You are a cybersecurity expert." \
  --references gpt-oss-20b qwythos-9b \
  --aggregator qwen3:14b \
  --temperature 0.3 \
  --timeout 240

# Three drafters, debug output
~/.hermes/bin/local_moa.py \
  --prompt "Architecture tradeoffs for event-driven systems" \
  --references gpt-oss-20b qwen3:14b qwythos-9b \
  --aggregator gpt-oss-20b \
  --timeout 240 \
  --verbose 2>&1 | less
```

## Difference: /localmoa vs /moa

| | `/localmoa` | `/moa` |
|---|---|---|
| Engine | External Python CLI (`local_moa.py`) | Hermes built-in MoA |
| Config | Inline flags at runtime | `moa:` block in config.yaml |
| Flexibility | Full — any models, any flags on the fly | Fixed to configured presets |
| Default preset | `qwen3:14b + ornith-9b → gpt-oss-20b` | `local` preset in config |
| Use when | You want runtime control over models/params | You want the configured default behavior |

- `~/.hermes/bin/local_moa.py --help` — authoritative option list
- `~/.hermes/bin/local_moa_guide.md` — full architecture guide and env vars
