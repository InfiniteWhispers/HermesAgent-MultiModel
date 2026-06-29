# local_moa — Usage Guide

**Mixture-of-Agents for local Ollama models.**  
Queries multiple models in parallel, then asks an aggregator to synthesize the best answer from all responses.

Script: `~/.hermes/bin/local_moa.py`

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running on `localhost:11434`
- Models pulled in Ollama (e.g. `ollama pull ornith-9b`)

---

## Basic Usage

```bash
~/.hermes/bin/local_moa.py --prompt "Your question here"
```

Uses default reference models (`qwen3:14b`, `ornith-9b`) and `gpt-oss-20b` as the aggregator.

```bash
~/.hermes/bin/local_moa.py --help
```

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--prompt TEXT` | *(required)* | The question or task |
| `--references M1 M2 ...` | `qwen3:14b ornith-9b` | Reference models (Ollama pull tags) |
| `--aggregator MODEL` | `gpt-oss-20b` | Model that synthesizes the final answer |
| `--system TEXT` | *(none)* | System prompt applied to all reference models |
| `--temperature FLOAT` | *(model default)* | Sampling temperature (e.g. `0.3` for factual, `0.8` for creative) |
| `--timeout INT` | `200` | Per-request socket timeout in seconds; aggregator gets +60s |
| `--num-ctx INT` | `65536` | Context window size passed to Ollama |
| `--ollama-url URL` | `http://localhost:11434/api/chat` | Override Ollama endpoint (or set `OLLAMA_BASE_URL`) |
| `-v / --verbose` | off | Print each model's raw response to stderr before aggregating |

> **Model tags** must be Ollama pull tags, not Hermes aliases. See alias table below.

---

## Alias → Ollama Tag Mapping

| Hermes alias        | Ollama pull tag    |
|---------------------|--------------------|
| qwen3-14b-think     | qwen3:14b          |
| qwythos-9b          | qwythos-9b         |
| gpt-oss-20b         | gpt-oss-20b        |
| ornith-9b           | ornith-9b          |
| gemma4-heretic-12b  | gemma4-heretic-12b |
| qwen25-coder-7b     | qwen2.5-coder:7b   |
| qwen3-vl-8b         | qwen3-vl:8b        |

---

## Examples

**Default models (qwen3:14b + ornith-9b → gpt-oss-20b):**
```bash
~/.hermes/bin/local_moa.py --prompt "Research RSA and DSA and summarize the key differences"
```

**Explicit models:**
```bash
~/.hermes/bin/local_moa.py \
  --prompt "Explain RSA vs DSA" \
  --references gpt-oss-20b qwen3:14b \
  --aggregator gpt-oss-20b
```

**Three reference models with a domain-specific system prompt:**
```bash
~/.hermes/bin/local_moa.py \
  --prompt "What are the main attack vectors against RSA implementations?" \
  --system "You are a cybersecurity expert." \
  --references gpt-oss-20b qwen3:14b qwythos-9b \
  --aggregator qwen3:14b \
  --temperature 0.3 \
  --timeout 240
```

**Agentic coding review:**
```bash
~/.hermes/bin/local_moa.py \
  --prompt "Review this Python script for correctness and edge cases" \
  --references ornith-9b qwen2.5-coder:7b \
  --aggregator gpt-oss-20b
```

**Debug mode — see what each model said before synthesis:**
```bash
~/.hermes/bin/local_moa.py --prompt "Explain gradient descent" --verbose 2>&1 | less
```

---

## How It Works

```
User prompt
    │
    ├──► Model 1 ──┐
    ├──► Model 2 ──┼──► Aggregator ──► Final answer
    └──► Model N ──┘
         (parallel)      (synthesis)
```

1. All `--references` models receive the same prompt simultaneously (parallel HTTP requests to Ollama).
2. Responses are anonymized (`Model 1`, `Model 2`, …) to prevent aggregator bias.
3. The `--aggregator` model receives all responses plus the original question and synthesizes a single definitive answer.
4. If a reference model times out or fails, a warning is printed and the run continues with remaining models. The run only fails if **all** reference models fail.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OLLAMA_BASE_URL` | Override the Ollama endpoint. Accepts bare host (`http://localhost:11434`), `/v1` base, or full `/api/chat` path — the script normalizes all forms. |

---

## Tips

- **Timeout tuning:** Cold-loading a model on 16 GB GDDR7 takes 60–90 s. Use `--timeout 240` when running 3+ models that may not all be pre-loaded.
- **Background runs:** When invoking via the Hermes terminal tool, set `background=true` and `notify_on_complete=true` to avoid foreground timeout failures.
- **Temperature:** Use `0.2`–`0.4` for factual/technical questions; leave unset for general or creative tasks.
- **Aggregator choice:** A reasoning-capable aggregator (e.g. `qwen3:14b`) often synthesizes better than reusing the same model as both reference and aggregator.
- **System prompt scope:** `--system` applies only to reference models. The aggregator receives only the anonymized drafts plus the original question, not the system prompt.
- **qwythos-9b** as a reference drafter adds uncensored analytical depth on questions other models hedge on — especially useful for security and exploit-adjacent research.
