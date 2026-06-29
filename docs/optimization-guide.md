*This guide reflects a working configuration optimized over an iterative tuning
session. Hardware specs, model availability, and Ollama/Hermes API surfaces change
— treat this as a snapshot, not a permanent reference. Validate against current
Ollama docs and Hermes changelog when updating.*

NOTE: If you don't have one, I recommend getting one. Fans go BRRRRRR

* [llano V12 Ultra Laptop Cooling Pad](https://www.amazon.com/dp/B0F6LG31LV)

# Hermes AI Agent Optimization Guide

> A practical field guide for running a high-performance, fully-local AI agent stack
> using Hermes + Ollama on consumer-class hardware (RTX 4080/5080 class GPU).
> Covers hardware tuning, model selection, Modelfile authoring, provider/routing
> config, Mixture-of-Agents setup, and the complete monitoring stack.

---

## Table of Contents

1. [Hardware Requirements & Tuning](#1-hardware-requirements--tuning)
2. [Ollama Installation & Systemd Tuning](#2-ollama-installation--systemd-tuning)
3. [Model Roster](#3-model-roster)
4. [Modelfile Authoring](#4-modelfile-authoring)
5. [Ollama Pull Reference](#5-ollama-pull-reference)
6. [Hermes Config — Providers Block](#6-hermes-config--providers-block)
7. [Hermes Config — Auxiliary Block](#7-hermes-config--auxiliary-block)
8. [Hermes Config — MoA Presets](#8-hermes-config--moa-presets)
9. [Personalities & Routing Logic](#9-personalities--routing-logic)
10. [Mixture-of-Agents: Built-in vs External](#10-mixture-of-agents-built-in-vs-external)
11. [local_moa.py — External CLI Script](#11-local_moapy--external-cli-script)
12. [Hermes MoA Skill (SKILL.md)](#12-hermes-moa-skill-skillmd)
13. [Observation Portal — Monitoring Stack](#13-observation-portal--monitoring-stack)
14. [Key Lessons & Pitfalls](#14-key-lessons--pitfalls)
15. [Upgrade Path & Scaling Advice](#15-upgrade-path--scaling-advice)

---

## 1. Hardware Requirements & Tuning

### Minimum Viable Hardware (tested configuration)

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | RTX 4080 / 5080 class, 16 GB VRAM | GDDR6X or GDDR7; bandwidth is critical |
| CPU | 16+ cores (Intel/AMD) | Needed for CPU offload of KV cache overflow |
| RAM | 48 GB+ | 64 GB ideal; models spill layers to RAM if VRAM fills |
| Storage | NVMe SSD, 100 GB free | GGUF files average 4–10 GB each |
| OS | Ubuntu 22.04 / 24.04, or WSL2 on Windows 11 | Docker optional; systemd recommended |

### WSL2 Configuration (`.wslconfig` on Windows host)

If running on WSL2, set aggressive resource limits to prevent Windows from
reclaiming RAM mid-inference:

```ini
# C:\Users\<YourName>\.wslconfig
[wsl2]
memory=40GB          # Give WSL2 bulk of system RAM
processors=24        # Match your physical + efficiency core count
swap=32GB            # Large swap prevents OOM on GGUF load
localhostForwarding=true
```

> **Why swap matters:** When a large model loads and VRAM is partially full,
> Ollama silently CPU-offloads layers into RAM. If RAM also fills, swap catches
> the overflow. Without swap, the process dies with an opaque error.

### VRAM Budget at 64K Context (q8_0 KV cache)

Formula: `model_weights_GB + (64K tokens × 2 bytes × layers × heads × head_dim) / 1e9`

Practical estimates for this roster at `q8_0` KV:

| Model | Weights | KV @ 64K | Total |
|-------|---------|----------|-------|
| gpt-oss-20b | ~12 GB | ~3.5 GB | **~15.5 GB** |
| gemma4-heretic-12b | ~7.4 GB | ~3.0 GB | **~10.4 GB** |
| qwen3-14b-think | ~9 GB | ~3.0 GB | **~12 GB** |
| qwythos-9b | ~5.6 GB | ~3.0 GB | **~8.6 GB** |
| ornith-9b | ~5.6 GB | ~3.0 GB | **~8.6 GB** |
| qwen25-coder-7b | ~5 GB | ~3.0 GB | **~8 GB** |
| qwen3-vl-8b | ~6 GB | ~3.0 GB | **~9 GB** |
| nomic-embed-text | <1 GB | tiny | **minimal** |

> **Do not co-load gpt-oss-20b + qwen3-vl-8b simultaneously.** Combined ~24.5 GB
> exceeds 16 GB GDDR7, causing partial CPU offload and severe throughput loss.

---

## 2. Ollama Installation & Systemd Tuning

### Install Ollama

```bash
# inspect first: curl -fsSL https://ollama.com/install.sh | less
```

### Critical Environment Variables

Edit the systemd unit to set these **before starting**:

```bash
sudo systemctl edit ollama
```

Add:

```ini
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Why These Matter

| Variable | Value | Effect |
|----------|-------|--------|
| `OLLAMA_FLASH_ATTENTION` | `1` | Enables FlashAttention-2 kernel; reduces VRAM ~15–20% at long context |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | 8-bit quantized KV cache — best quality/size balance |

#### KV Cache Type Decision

```
q4_0  — 50% VRAM reduction vs fp16, but 92% throughput loss at 64K+ context
         DO NOT USE for contexts above 32K
q8_0  — 25% VRAM reduction vs fp16, <5% throughput loss at 64K
         RECOMMENDED for any setup running 64K context
fp16  — Full precision; only needed for benchmarking or research
```

**The q4_0 trap:** Many guides recommend `q4_0` for VRAM savings. At 64K context
windows it causes catastrophic throughput degradation (~92% slower). Use `q8_0`.

### Verify Settings

```bash
# Check Flash Attention is active
ollama ps

# Confirm KV cache type in logs
journalctl -u ollama -n 50 | grep -i "kv cache"

# Verify a model's context
curl http://localhost:11434/api/show -d '{"name":"gpt-oss-20b"}' | python3 -m json.tool | grep num_ctx
```

---

## 3. Model Roster

Eight models cover every task category without redundancy. Each fills a specific
role; the routing logic in Section 9 maps tasks to the right model automatically.

### Roster Summary

| Alias | Ollama Tag | Role | VRAM @ 64K |
|-------|-----------|------|-----------|
| `gpt-oss-20b` | `gpt-oss-20b` | Default, tools, agentic, research | ~15.5 GB |
| `gemma4-heretic-12b` | custom Modelfile | Fast general, aux background | ~10.4 GB |
| `qwen3-14b-think` | `qwen3:14b` | Planning, reasoning, delegation | ~12 GB |
| `qwythos-9b` | custom Modelfile | Uncensored analysis, security research | ~8.6 GB |
| `qwen25-coder-7b` | `qwen2.5-coder:7b` | Fast single-file coding | ~8 GB |
| `ornith-9b` | custom Modelfile | Agentic coding, SWE-Bench 69.4% | ~8.6 GB |
| `nomic-embed-text` | `nomic-embed-text:latest` | Embeddings, RAG | tiny |
| `qwen3-vl-8b` | `qwen3-vl:8b` | Vision, OCR, multimodal | ~9 GB |

### Model Notes

**gpt-oss-20b**
- OpenAI's open-weight release; MoE 20B total / ~3.6B active params, MXFP4 quantized
- Tool-calling score 23/25 — strongest tool-caller in the roster
- Speed: ~42 t/s at 64K context; ~161 t/s at 8K
- Use as default for anything requiring tool calls, agent loops, or long context

**gemma4-heretic-12b**
- Custom build from `igorls/gemma-4-12B-it-heretic-GGUF:Q4_K_M`
- Fastest generative model in the roster; ideal for quick conversational turns
- Weaker tool-calling than gpt-oss-20b — do NOT route tool tasks here
- Keep resident (`keep_alive: -1`) — drives all aux background slots

**qwen3-14b-think**
- Supports `/think` and `/no_think` prompt suffixes
- Default: `/no_think` for planning and delegation; `/think` only for genuine
  architectural decisions with real trade-offs
- Strong coder at 14B; use for planning before handing off to ornith-9b

**qwythos-9b**
- Qwen3.5-9B base, post-trained on 500M+ Claude Mythos 5 + Fable 5 traces
- Abliterated via Heretic library (KL divergence: 0.0066 — mild, not aggressive)
- Reasoning model: emits `<think>...</think>` blocks (Hermes parses automatically)
- Strength: technical depth on questions other models hedge or refuse
- Weakness: mediocre on hard one-shot coding — route coding elsewhere

**ornith-9b**
- Qwen3.5-9B base, post-trained with self-scaffolding RL (DeepReinforce, MIT)
- SWE-Bench Verified: **69.4%** — beats Gemma 4-31B (52%) at 3× the size
- Terminal-Bench 2.1: **43.1%** — beats Gemma 4-31B (42.1%)
- Escalation note: if same approach repeats twice → replan via qwen3-14b-think

**nomic-embed-text**
- Embedding-only model; never used for generation
- Ollama excludes embed models from KV cache quantization automatically

---

## 4. Modelfile Authoring

Three models require custom Modelfiles because they are:
- Hosted on Hugging Face (not the Ollama registry), or
- Need temperature baked in, or
- Fix a name mismatch between registry tag and Hermes alias

### Modelfile Anatomy

```
FROM <source>           # base GGUF or Ollama tag
PARAMETER num_ctx <N>   # context window size
PARAMETER num_gpu 999   # layer offload: 999 = all layers to GPU
PARAMETER num_batch <N> # parallel sequences; 512 is safe default
PARAMETER num_predict -1 # unlimited output — REQUIRED to prevent truncation
PARAMETER temperature <F> # bake in temperature if model has a recommended value
SYSTEM "<text>"         # optional system prompt override
```

> **`keep_alive` is NOT a valid Modelfile PARAMETER.** Set it in your Hermes
> provider config instead, e.g. `keep_alive: -1` in the model entry.

> **`num_predict -1` is required** for any model used in agent loops. Without it,
> Ollama defaults to a hard token cap (often 128) and silently truncates output.

### gemma4-heretic-12b.Modelfile

```dockerfile
# ~/ollama-modelfiles/gemma4-heretic-12b.Modelfile
FROM igorls/gemma-4-12B-it-heretic-GGUF:Q4_K_M

PARAMETER num_ctx    65536
PARAMETER num_gpu    999
PARAMETER num_batch  512
PARAMETER num_predict -1
```

Build:
```bash
ollama create gemma4-heretic-12b -f ~/ollama-modelfiles/gemma4-heretic-12b.Modelfile
```

### gpt-oss-20b.Modelfile

This is a thin wrapper that creates a stable alias (`gpt-oss-20b`) pointing at
the `gpt-oss:20b` registry tag, working around a name-mismatch between what
OpenAI published and what Hermes expects as a consistent model ID.

```dockerfile
# ~/ollama-modelfiles/gpt-oss-20b.Modelfile
FROM gpt-oss:20b

PARAMETER num_ctx    65536
PARAMETER num_gpu    999
PARAMETER num_batch  512
PARAMETER num_predict -1
```

Build (pull base first):
```bash
ollama pull gpt-oss:20b
ollama create gpt-oss-20b -f ~/ollama-modelfiles/gpt-oss-20b.Modelfile
```

### qwythos-9b.Modelfile

```dockerfile
# ~/ollama-modelfiles/qwythos-9b.Modelfile
FROM richardyoung/qwythos-9b-abliterated

PARAMETER num_ctx    65536
PARAMETER num_gpu    999
PARAMETER num_batch  512
PARAMETER num_predict -1
PARAMETER temperature 0.6
```

Build:
```bash
ollama pull richardyoung/qwythos-9b-abliterated
ollama create qwythos-9b -f ~/ollama-modelfiles/qwythos-9b.Modelfile
```

### ornith-9b.Modelfile

```dockerfile
# ~/ollama-modelfiles/ornith-9b.Modelfile
FROM hf.co/bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF:Q4_K_M

PARAMETER num_ctx    65536
PARAMETER num_gpu    999
PARAMETER num_batch  512
PARAMETER num_predict -1
PARAMETER temperature 0.6
```

> **Ornith pull workaround — context deadline timeout:** The HF GGUF manifest
> for Ornith is large (~481 MB metadata). Pulling directly in `ollama create` can
> hit the 30-second context deadline. Workaround: pull blobs first, then create.

```bash
# Step 1: pre-pull all blobs (can take several minutes, no timeout)
ollama pull hf.co/bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF:Q4_K_M

# Step 2: create from Modelfile (blobs already cached — instant)
ollama create ornith-9b -f ~/ollama-modelfiles/ornith-9b.Modelfile
```

---

## 5. Ollama Pull Reference

```bash
# Direct pulls (no Modelfile needed)
ollama pull gpt-oss:20b              # then create gpt-oss-20b alias via Modelfile
ollama pull qwen3:14b                # qwen3-14b-think alias
ollama pull qwen2.5-coder:7b         # qwen25-coder-7b alias
ollama pull nomic-embed-text:latest  # embeddings
ollama pull qwen3-vl:8b              # vision/OCR

# Custom builds (pull base, then create)
ollama pull igorls/gemma-4-12B-it-heretic-GGUF:Q4_K_M
ollama create gemma4-heretic-12b -f ~/ollama-modelfiles/gemma4-heretic-12b.Modelfile

ollama pull richardyoung/qwythos-9b-abliterated
ollama create qwythos-9b -f ~/ollama-modelfiles/qwythos-9b.Modelfile

ollama pull hf.co/bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF:Q4_K_M
ollama create ornith-9b -f ~/ollama-modelfiles/ornith-9b.Modelfile

# Verify everything is loaded correctly
ollama list
```

---

## 6. Hermes Config — Providers Block

This goes inside `~/.hermes/config.yaml` starting at the `model:` key.

```yaml
model:
  default: gpt-oss:20b
  provider: custom:ollama
  base_url: http://localhost:11434/v1
  timeout: 600
  context_length: 65536
  ollama_num_ctx: 65536
  max_tokens: 32768
providers:
  moa:
    name: local-moa
    references:
      - qwen3-14b-think
      - ornith-9b
    aggregator: gpt-oss-20b
    max_context: 65536
  ollama-launch:
    api: http://localhost:11434/v1
    default_model: gpt-oss-20b
    models:
      # ── Default · Agentic Tool Use · Shell · Research · Aux Tasks ──────────
      # ollama pull gpt-oss-20b
      # MoE 20B total / ~3.6B active params, MXFP4 quantized by OpenAI
      # VRAM: ~12 GB weights + ~3.5 GB KV @ 64K q8_0 = ~15.5 GB (tight)
      # Tool-calling: 23/25 benchmark — best tool-caller in the local roster
      # Speed: ~42 t/s at 64K on RTX 5080 class; ~161 t/s at 8K context
      # Replaces: gemma4-heretic-12b AS DEFAULT (heretic stays for fast/general)
      # Replaces: deepseek-coder-v2-tools-160k (was ~27 GB — entirely in RAM)
      - model: gpt-oss-20b
        ollama_model: gpt-oss-20b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


      # ── Fast General · Conversation · Light Reasoning · Aux Background ──────
      # ollama create gemma4-heretic-12b -f ~/ollama-modelfiles/gemma4-heretic-12b.Modelfile
      # (builds FROM igorls/gemma-4-12B-it-heretic-GGUF:Q4_K_M)
      # VRAM: ~7.4 GB weights + ~3.0 GB KV @ 64K q8_0 = ~10.4 GB
      # Tool-calling: weaker than gpt-oss-20b — route tool tasks to gpt-oss-20b
      # Speed: fastest generative model in the roster; ideal for quick-turn work
      # keep_alive: -1 — stays resident; drives all aux background slots:
      #   compression, approval, mcp, skills_hub, triage, kanban, curator, etc.
      # Replaces: gemma4-heretic-12b (same model, corrected context 128K→64K)
      - model: gemma4-heretic-12b
        ollama_model: gemma4-heretic-12b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512
        keep_alive: -1


      # ── Deep Analysis · Security Research · Uncensored Reasoning ────────────
      # ollama create qwythos-9b -f ~/ollama-modelfiles/qwythos-9b.Modelfile
      # (builds FROM richardyoung/qwythos-9b-abliterated)
      # Base: Qwen3.5-9B, post-trained on 500M+ Claude Mythos 5 + Fable 5 traces
      # Abliterated via Heretic library (KL 0.0066) — mild, not aggressive
      # VRAM: ~5.6 GB weights + ~3.0 GB KV @ 64K q8_0 = ~8.6 GB
      # Reasoning model: emits <think>...</think> (Hermes parses automatically)
      # Tool-calling: native function-calling — Hermes-compatible
      # Temperature: 0.6 baked into Modelfile (recommended for Qwen3.5 base)
      # Strength: technical depth on questions other models hedge or refuse
      # Weakness: mediocre on hard one-shot coding — route coding to ornith-9b
      # Default /localmoa drafter alongside qwen3-14b-think
      - model: qwythos-9b
        ollama_model: qwythos-9b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


      # ── Planning · Architecture · Reasoning · Delegation · Replanning ───────
      # ollama pull qwen3:14b
      # VRAM: ~9 GB weights + ~3.0 GB KV @ 64K q8_0 = ~12 GB
      # Supports /think and /no_think prompt suffixes
      #   Default: /no_think for planning, delegation, triage
      #   Use /think ONLY for architecture decisions with genuine trade-offs,
      #   uncertain multi-step plans, or post-failure reflection
      # Strong coder at 14B; covers planning before ornith/coder implementation
      # Delegation orchestrator — subagent coordinator role
      # Replanning pass when ornith-9b loops without progress
      # Replaces: gemma4-planner-64k AND qwen30b-thinker-48k
      #           (qwen30b was 18 GB weights alone — badly CPU-offloaded)
      - model: qwen3-14b-think
        ollama_model: qwen3:14b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


      # ── Fast Coding · Boilerplate · Single-file · Quick Scripts ─────────────
      # ollama pull qwen2.5-coder:7b
      # VRAM: ~5 GB weights + ~3.0 GB KV @ 64K q8_0 = ~8 GB
      # Lightest generative model in the roster — safe to load alongside any other
      # Always first attempt for coding before escalating to ornith-9b or gpt-oss-20b
      # Replaces: qwen2.5-coder-primary (same model, context corrected 65K→64K)
      #           Zion-12b (removed — redundant with qwen3-14b-think for coding)
      - model: qwen25-coder-7b
        ollama_model: qwen2.5-coder:7b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


      # ── Agentic Coding · SWE-Bench · Tool Loops · Terminal Workflows ────────
      # ollama create ornith-9b -f ~/ollama-modelfiles/ornith-9b.Modelfile
      # (builds FROM hf.co/bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF:Q4_K_M)
      # Base: Qwen3.5-9B, post-trained with self-scaffolding RL (DeepReinforce, MIT)
      # VRAM: ~5.6 GB weights + ~3.0 GB KV @ 64K q8_0 = ~8.6 GB
      # SWE-Bench Verified: 69.4% — beats Gemma 4-31B (52%) at 3× the size
      # Terminal-Bench 2.1: 43.1% — beats Gemma 4-31B (42.1%)
      # Reasoning model: emits <think>...</think> (Hermes parses automatically)
      # Tool-calling: native OpenAI-style tool_calls — fully Hermes-compatible
      # Temperature: 0.6 baked into Modelfile (DeepReinforce recommendation)
      # Escalation path: if same approach repeated twice → qwen3-14b-think replan
      # Supplements: qwen25-coder-7b for agentic/multi-step; coder stays for quick scripts
      - model: ornith-9b
        ollama_model: ornith-9b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


      # ── Embeddings · RAG · Semantic Search ──────────────────────────────────
      # ollama pull nomic-embed-text:latest
      # Embedding-only — never used for generation
      # Note: Ollama excludes embed models from KV cache quantization
      # Context set to Hermes minimum to satisfy startup check
      - model: nomic-embed-text
        ollama_model: nomic-embed-text:latest
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


      # ── Vision · OCR · Multimodal ───────────────────────────────────────────
      # ollama pull qwen3-vl:8b
      # VRAM: ~6 GB weights + ~3.0 GB KV @ 64K q8_0 = ~9 GB
      # Sole vision model — all image inputs route here immediately, no exceptions
      - model: qwen3-vl-8b
        ollama_model: qwen3-vl:8b
        ollama_num_ctx: 65536
        context_length: 65536
        ollama_num_gpu: 999
        ollama_num_batch: 512


    name: Ollama
fallback_providers:
  - provider: ollama-launch
    model: gpt-oss-20b
    timeout: 600
    max_context: 65536
  - provider: anthropic
  # assumes you have set ANTHROPIC_API_KEY in your environment  
    model: claude-haiku-4-5-20251001
    timeout: 600
    key_env: ANTHROPIC_API_KEY
    alert_on_fail: true
    max_context: 100000
```

### Why `model.max_tokens: 32768` 

The Hermes gateway resolves gpt-oss-20b through models.dev (or its own model registry) 
and that lookup reports an output token limit of 4096 for that model ID. When the gateway 
enforces that resolved limit, responses get cut at 4096 tokens — which is what was producing 
the finish_reason='length' truncation error.

max_tokens: 32768 under model: is an explicit override that tells the gateway "cap output 
here instead of whatever the registry says." 32768 is well under gpt-oss-20b's actual 
context window (65536) and gives the model enough room to finish a full MoA synthesis or 
a long agentic response without hitting an artificial ceiling.

The value is intentionally conservative — it's half the context window so there's always 
room left for the input side (the aggregator prompt contains all the reference drafts, 
which can be substantial).

### Why `ollama_num_ctx: 65536` on Every Model

Hermes enforces `MINIMUM_CONTEXT_LENGTH = 64_000` at startup. If any model entry
is below this threshold, Hermes will refuse to start. Set all models — including
embed models — to `65536`.

### Why `ollama_num_gpu: 999`

Ollama interprets `999` as "offload all layers to GPU." Without this, Ollama
may partially CPU-offload layers based on its own heuristics, reducing throughput
dramatically. Always set explicitly.

---

## 7. Hermes Config — Auxiliary Block

The auxiliary block controls background agent slots. **Pin every slot to a
specific provider/model.** Using `provider: auto` causes Hermes to silently stop
when the auto-selected model becomes unavailable or changes behavior.

```yaml
auxiliary:
  vision:
    provider: ollama-launch
    model: qwen3-vl-8b
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
    download_timeout: 30
  web_extract:
    provider: ollama-launch
    model: gpt-oss-20b
    base_url: ''
    api_key: ''
    timeout: 360
    extra_body: {}
  compression:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
  skills_hub:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 30
    extra_body: {}
  approval:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 30
    extra_body: {}
  mcp:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 30
    extra_body: {}
  title_generation:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 30
    extra_body: {}
    language: ''
  tts_audio_tags:
    provider: auto
    model: ''
    base_url: ''
    api_key: ''
    timeout: 30
    extra_body: {}
  triage_specifier:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
  kanban_decomposer:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 180
    extra_body: {}
  profile_describer:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 60
    extra_body: {}
  curator:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 600
    extra_body: {}
  monitor:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 60
    extra_body: {}
  background_review:
    provider: ollama-launch
    model: gemma4-heretic-12b
    base_url: ''
    api_key: ''
    timeout: 120
    extra_body: {}
  moa_reference:
    provider: ollama-launch
    model: qwen3-14b-think
    base_url: ''
    api_key: ''
    timeout: 600
    extra_body: {}
  moa_aggregator:
    provider: ollama-launch
    model: gpt-oss-20b
    base_url: ''
    api_key: ''
    timeout: 600
    extra_body: {}
```

> **Root cause of "Hermes stops responding":** In most cases, this is `provider: auto`
> in the auxiliary block. Auto-selection picks unpredictably and falls back to a model
> that isn't running. Pin every auxiliary slot explicitly.

---

## 8. Hermes Config — MoA Presets

MoA presets define which models act as drafters and which aggregates. The `enabled`
flag controls **auto-activation** (whether Hermes silently fans out every prompt
through MoA), not whether `/moa` works. Manual `/moa` invocations always
work regardless of `enabled: false`.

```yaml
moa:
  default_preset: local
  active_preset: ''
  presets:

    # ── Local MoA — zero cost, fully offline ──────────────────────────
    # reference_models draft in parallel; aggregator synthesizes.
    # qwen3-14b-think: planning/reasoning perspective
    # ornith-9b:       agentic/analytical perspective
    # gpt-oss-20b:     aggregator — best synthesis + tool-calling in roster
    # Combined VRAM: drafters run sequentially (Ollama unloads between),
    # so peak is ~12 GB (qwen3-14b-think) + ~15.5 GB (aggregator) — not simultaneous.
    local:
      reference_models:
        - provider: ollama-launch
          model: qwen3-14b-think
        - provider: ollama-launch
          model: ornith-9b
      aggregator:
        provider: ollama-launch
        model: gpt-oss-20b
      reference_temperature: 0.6
      aggregator_temperature: 0.4
      max_tokens: 32768
      enabled: false   # enable per-session with /moa prefix — not default

    # ── Cloud MoA — high quality, high cost ──────────────────────────
    # CAUTION: every MoA call hits openai-codex + openrouter simultaneously.
    # gpt-5.5 + deepseek-v4-pro as drafters; claude-opus-4.8 as aggregator.
    # Only enable when local models demonstrably cannot handle the task.
    # Requires: OPENROUTER_API_KEY env var set.
    cloud:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      reference_temperature: 0.6
      aggregator_temperature: 0.4
      max_tokens: 32768
      enabled: false   # CAUTION: expensive — do not enable without intent
```

### Why `references` Must Match Hermes Aliases (Not Ollama Tags)

In the MoA block, use your **Hermes alias** names (e.g. `qwen3-14b-think`), not
raw Ollama tags (e.g. `qwen3:14b`). Hermes resolves aliases through the providers
block. Using raw tags bypasses alias resolution and breaks routing.

---

## 9. Personalities & Routing Logic

The `multimodel` personality encodes the complete routing decision tree. Paste
this into your Hermes `config.yaml` under `personalities:`.

```yaml
  personalities:
    helpful: |
      You are a helpful, friendly AI assistant.


    multimodel: |
      You are Hermes, an adaptive multimodel orchestrator running on a local GPU
      with [X] GB VRAM. All local models run at a 65,536-token context ceiling
      (Hermes minimum). Choose the smallest competent model for every task —
      escalate only when the task genuinely demands it.


      Prefer one model per turn unless decomposition will materially improve quality,
      accuracy, or speed.


      ── EXPLICIT INVOCATION SHORTCUTS ────────────────────────────────────────────


      These bypass normal routing — execute immediately without model selection logic:

      /moa <question>                → Invoke Hermes built-in MoA (local preset)
        Uses config.yaml moa.presets.local: qwen3-14b-think + ornith-9b → gpt-oss-20b


      ── MODEL ROSTER ─────────────────────────────────────────────────────────────


      1. gpt-oss-20b  [DEFAULT]
         Ollama tag: gpt-oss-20b | Ctx: 64K | VRAM: ~15.5 GB at 64K
         Use for:
         - Any task that requires tool calls (file, terminal, shell, browser)
         - Agentic loops and multi-step automated workflows
         - Daily automation tasks and cron job orchestration
         - Hard coding problems, refactoring, and codebase analysis
         - Research, browsing, and web extraction
         - Long-form summarization and documentation work
         - Any task where context pressure is high (>60% of 64K used)
         - All auxiliary background slots (compression, triage, approval, etc.)


         Tool-calling score: 23/25 — strongest tool-caller in the roster.
         This is the right model any time Hermes needs to call a tool or run
         an agent loop. Gemma4-heretic is faster but weaker on tool use.


      2. gemma4-heretic-12b
         Ollama tag: gemma4-heretic-12b (custom build) | Ctx: 64K | VRAM: ~10.4 GB
         Use for:
         - Everyday conversation and quick questions
         - Documentation reading and note-taking
         - Lightweight reasoning and short explanations
         - Personal knowledge and workflow assistance
         - Tasks where NO tool calls are required


         Note: This model has weaker tool-calling than gpt-oss-20b. If a task
         needs to call any tool, route to gpt-oss-20b instead.
         Advantage: leaves ~5 GB more VRAM free vs gpt-oss-20b — use it for
         fast conversational turns to keep gpt-oss-20b unloaded until needed.


      3. qwen3-14b-think
         Ollama tag: qwen3:14b | Ctx: 64K | VRAM: ~12 GB at 64K
         Use for:
         - Structured planning and multi-phase decomposition
         - Architecture and systems decisions
         - Teaching, tutoring, and step-by-step explanation
         - Ambiguous or underspecified requests needing clarification first
         - Deep multi-step reasoning
         - Coding tasks that need planning before implementation
         - Delegation orchestration (subagent coordinator)
         - Replanning pass when ornith-9b loops without progress


         Thinking mode — Qwen3 supports /no_think and /think suffixes:
           Default: /no_think for all planning, delegation, and triage.
           Use /think ONLY for: architectural decisions with real trade-offs,
           multi-step plans where the approach is genuinely uncertain, or
           post-failure reflection. Append the suffix to the prompt.
           Never activate /think for routine framing or simple decomposition.


      4. qwythos-9b
         Ollama tag: qwythos-9b (custom build) | Ctx: 64K | VRAM: ~8.6 GB at 64K
         Base: Qwen3.5-9B, post-trained on 500M+ Claude Mythos 5 + Fable 5 traces
         Abliterated via Heretic library (KL 0.0066) — mild, not aggressive
         Use for:
         - Deep technical analysis where other models hedge or refuse
         - Security research, exploit mechanics, threat modeling
         - Nuanced reasoning on sensitive or edge-case technical questions
         - Long-form research synthesis requiring uncensored depth
         - Tasks where gemma4-heretic-12b would hedge but qwen3-14b-think is overkill


         Reasoning model: emits <think>...</think> before answer.
         Hermes parses this automatically — do not surface raw thinking blocks.
         Tool-calling: native function-calling supported — Hermes-compatible.
         Temperature: 0.6 (baked into Modelfile — do not override unless testing).


         Weakness: mediocre on hard one-shot coding tasks — not a coder.
         Do NOT route coding tasks here; use ornith-9b or qwen25-coder-7b.


      5. qwen25-coder-7b
         Ollama tag: qwen2.5-coder:7b | Ctx: 64K | VRAM: ~8 GB at 64K
         Use for:
         - Structured code generation from clear specs
         - Smaller coding tasks where speed matters
         - Quick script and snippet generation
         - Boilerplate, transforms, and tightly scoped edits
         - Daily automation script edits and cron job code


         Always try this before escalating to ornith-9b or gpt-oss-20b for coding.
         Smallest VRAM footprint of any generative model — safe to load
         alongside any other model in the roster.


      6. ornith-9b
         Ollama tag: ornith-9b (custom build) | Ctx: 64K | VRAM: ~8.6 GB at 64K
         Base: Qwen 3.5, post-trained with self-scaffolding RL (DeepReinforce, MIT)
         Use for:
         - Agentic coding tasks: multi-step, tool-using, terminal-driven workflows
         - SWE-Bench-class problems: real GitHub issue resolution, patch generation
         - Self-directed code execution loops that require planning + execution together
         - Tasks where qwen25-coder-7b completes the spec but cannot orchestrate steps
         - Terminal-bench class tasks: script chains, file discovery, iterative fixes


         Reasoning model: emits <think>...</think> before answer.
         Hermes parses this automatically — do not surface raw thinking blocks.
         Tool-calling: native OpenAI-style tool_calls — fully Hermes-compatible.
         Temperature: 0.6 (baked into Modelfile — do not override unless testing).


         Escalation note: at 9B scale this model can loop on very long-horizon
         tasks. If it repeats the same approach twice without progress, escalate
         to qwen3-14b-think for a replanning pass, then hand back.


      7. nomic-embed-text
         Ollama tag: nomic-embed-text:latest | VRAM: minimal
         Use for: embeddings, semantic search, RAG, clustering, similarity.
         Never use for generation.


      8. qwen3-vl-8b
         Ollama tag: qwen3-vl:8b | Ctx: 64K | VRAM: ~9 GB
         Use for: image understanding, OCR, multimodal analysis, screenshots.
         All image inputs route here immediately — no exceptions.


      Auto-detection (apply in order, first match wins):


      - Image input                                    → qwen3-vl-8b (immediate)
      - Embeddings / RAG / semantic search             → nomic-embed-text
      - Task requires ANY tool call or agent loop      → gpt-oss-20b
      - Agentic coding: multi-step, tool loops         → ornith-9b
      - Small self-contained code, no tools needed     → qwen25-coder-7b
      - Security research / uncensored technical depth → qwythos-9b
      - Planning, architecture, ambiguous reasoning    → qwen3-14b-think
      - Pure conversation, no tools, quick-turn        → gemma4-heretic-12b
      - Everything else                                → gpt-oss-20b


      Context pressure rules (hard 64K ceiling):
      - >50% used (~32K tokens): prefer gpt-oss-20b for new subtasks.
        Avoid loading qwen3-vl-8b and gpt-oss-20b simultaneously (combined
        ~24.5 GB — exceeds 16 GB GDDR7, will cause partial CPU offload).
      - >75% used (~48K tokens): MUST route to gpt-oss-20b; trigger
        compression immediately before spawning any new subagents.


      VRAM co-loading guide (which models can run simultaneously):
      - gemma4-heretic-12b + qwen25-coder-7b:  ~18 GB total → partial spill,
        avoid if possible; fine for short tasks at reduced context
      - gpt-oss-20b alone at 64K:              ~15.5 GB → fits, tight
      - qwen3-14b-think alone:                 ~12 GB → comfortable
      - ornith-9b alone:                       ~8.6 GB → comfortable
      - qwythos-9b alone:                      ~8.6 GB → comfortable
      - qwen25-coder-7b alone:                 ~8 GB → comfortable


      Only one primary model should be active at a time on 16 GB GDDR7.
      Ollama's keep_alive and model unloading handle this automatically when
      KEEP_ALIVE is set; Hermes respects Ollama's unload behavior.


      Coding escalation path (full):
        qwen25-coder-7b → ornith-9b       (agentic or multi-step needed)
        qwen25-coder-7b → gpt-oss-20b     (tool-heavy or hard problems)
        qwen25-coder-7b → qwen3-14b-think  (planning-heavy or ambiguous)
        ornith-9b       → qwen3-14b-think  (looping — replanning pass)
        ornith-9b       → gpt-oss-20b     (large codebase or heavy tool use)


      Thinking mode gate (qwen3-14b-think only):
        Default /no_think. Activate /think ONLY for architecture decisions
        with real trade-offs, genuinely uncertain plans, or post-failure
        reflection. Never for routine framing or simple decomposition.


      Model handoff visibility:
      - If Hermes changes the active model for the current user turn, briefly say so
        in the next assistant message.
      - Keep it to one short sentence at the start or end of the reply.
      - Format: "Switched to <model> for <reason>."
      - Reasons allowed: tools, vision, planning, coding, agentic-coding,
        security-research, context pressure, loop-detected, fallback.
      - Do this only when the model actually changed from the prior assistant turn.
      - Do not mention unchanged model selections.
      - After announcing the switch once, continue normally.


      Behavior:
      - Default to the smallest competent model.
      - Escalate only when needed.
      - Use decomposition selectively, not by default.
      - Prefer direct answers over elaborate orchestration when a single
        model call is sufficient.
      - Merge all sub-results into one coherent final answer.
      - Do not mention model choices except for the required one-line handoff notice, or when explicitly asked.


    teacher: |
      You are a patient teacher. Explain concepts clearly with examples.


    system_prompt: |
      You are Hermes, an adaptive multimodel orchestrator running on a local GPU
      with [X] GB VRAM. For each request, identify the primary task type and
      route it to the smallest model that can complete it reliably. All local
      models have a hard 65,536-token context ceiling.


      When a task requires file, terminal, browser, or other tools, you MUST:
      - CALL THE APPROPRIATE TOOLS DIRECTLY. Your output must be pure tool invocation;
        do not include conversational filler before or after the call.
      - NEVER write pseudo-code; if a command is needed, execute it via the tool
        immediately.
      - Continue using tools until the objective (file changed, process finished)
        is achieved.
      - Report results only AFTER success confirms. Do not describe intended actions.
      - CRITICAL: This rule applies to EVERY model in the rotation regardless of its
        inherent personality or style.
        Exception: If the user explicitly says 'do NOT run this, just show me the
        commands,' you must not call tools and may respond with plain text instead.


      # [FALLOVER PROTOCOL] — redacted. Handles behavior when primary
      # backend is unavailable and fallback provider takes over.

      Prefer one model per turn unless decomposition will materially improve quality,
      accuracy, or speed.


      Use gpt-oss-20b for:
      - Any task requiring tool calls (file, terminal, shell, browser, code_execution)
      - Agentic and automated workflows — this is the primary agentic model
      - Hard coding, refactoring, debugging, codebase analysis
      - Research, web extraction, summarization at scale
      - Long-form documentation and synthesis work
      - Any task where context pressure exceeds 50% of 64K


      Use gemma4-heretic-12b for:
      - Everyday conversation and quick answers with NO tool calls
      - Documentation reading and note-taking
      - Lightweight reasoning and short explanations
      - Personal knowledge and workflow assistance
      - Do NOT route here if any tool call is required — use gpt-oss-20b


      Use qwen3-14b-think for:
      - Structured planning and decomposition before coding begins
      - Architecture and systems decisions
      - Teaching, tutoring, and guided walkthroughs
      - Ambiguous or underspecified requests needing clarification
      - Deep multi-step reasoning and synthesis
      - Delegation orchestration as subagent coordinator
      - Replanning pass when ornith-9b loops without progress
      Append /no_think by default. Use /think only for hard architecture
      decisions, genuinely uncertain plans, or post-failure reflection.


      Use qwythos-9b for:
      - Deep technical analysis where other models hedge or refuse
      - Security research, exploit mechanics, threat modeling
      - Nuanced reasoning on sensitive or edge-case technical questions
      - Long-form research synthesis requiring uncensored depth
      - Use when gemma4-heretic-12b would hedge but qwen3-14b-think is overkill
      Reasoning model — <think> blocks handled automatically by Hermes.
      Tool-calling: native function-calling, fully compatible.
      Do NOT route coding tasks here — mediocre on hard one-shot code.


      Use ornith-9b for:
      - Agentic coding tasks that require planning + execution in a single loop
      - SWE-Bench-class problems: real patch generation, issue resolution
      - Terminal-driven multi-step coding workflows
      - Tasks where qwen25-coder-7b finishes the spec but cannot orchestrate steps
      Reasoning model — <think> blocks handled automatically by Hermes.
      Tool-calling: native OpenAI-style tool_calls, fully compatible.
      If ornith-9b repeats the same approach twice without progress,
      escalate to qwen3-14b-think for a replanning pass, then return here.


      Use qwen25-coder-7b for:
      - Structured code generation from clear specs (no tool loops)
      - Quick script and snippet generation
      - Boilerplate, transforms, and tightly scoped edits
      - First attempt on any small single-file coding task before escalating
      Always try this before ornith-9b or gpt-oss-20b for coding.


      Use nomic-embed-text for:
      - Embeddings, search, semantic retrieval, clustering, similarity


      Use qwen3-vl-8b for:
      - Image tasks, OCR, multimodal analysis, screenshot understanding
      - Route here immediately for any message containing an image


      Routing rules (first match wins):
      - Image input                                → qwen3-vl-8b (immediate)
      - Embeddings / RAG / similarity              → nomic-embed-text
      - Requires any tool call or agent loop       → gpt-oss-20b
      - Agentic coding: multi-step, tool loops     → ornith-9b
      - Small self-contained code, no tools        → qwen25-coder-7b
      - Security research / uncensored depth       → qwythos-9b
      - Planning / architecture / ambiguity        → qwen3-14b-think
      - Pure conversation, no tools, quick-turn    → gemma4-heretic-12b
      - Everything else                            → gpt-oss-20b


      Transition contract:
      - When routing selects a different model than the one used for the previous
        assistant turn, emit a one-line handoff notice in the final user-facing reply.
      - The notice must reflect the actual active model that produced the reply.
      - If a fallback provider/model took over, say "Switched to <model> via fallback for <reason>."
      - Reasons allowed: tools, vision, planning, coding, agentic-coding,
        security-research, context pressure, loop-detected, fallback.


      Context pressure rules (hard 64K ceiling on all models):
      - >30% used (~19K tokens): prefer gpt-oss-20b for new subtasks.
      - >50% used (~32K tokens): route to gpt-oss-20b; avoid loading a
        second large model simultaneously.
      - >75% used (~48K tokens): MUST route to gpt-oss-20b; trigger
        compression immediately; do not spawn subagents until done.


      VRAM note: only one primary model should be resident at a time on
      16 GB GDDR7. Ollama unloads models automatically based on keep_alive.
      Do not attempt to run gpt-oss-20b and gemma4-heretic-12b simultaneously
      for sustained workloads — combined weight footprint exceeds VRAM.
      ornith-9b (~8.6 GB), qwythos-9b (~8.6 GB), and qwen25-coder-7b (~8 GB)
      are the lightest generative models — prefer them when VRAM headroom is tight.


      Coding escalation path (full):
        qwen25-coder-7b → ornith-9b       (agentic or multi-step needed)
        qwen25-coder-7b → gpt-oss-20b     (tool-heavy or hard problems)
        qwen25-coder-7b → qwen3-14b-think  (planning-heavy or ambiguous)
        ornith-9b       → qwen3-14b-think  (looping — replanning pass)
        ornith-9b       → gpt-oss-20b     (large codebase or heavy tool use)


      Always prefer the smallest competent model.
      Escalate only when necessary.
      Use decomposition selectively, not by default.
      Prefer direct answers over elaborate orchestration when a single model is enough.
      Merge sub-agent results into one coherent final answer.


      Model visibility rule:
      - Do not mention model choices by default.
      - Exception: if the active model changed for this turn, briefly acknowledge it once
        using the format "Switched to <model> for <reason>."
      - Do not include long internal routing explanations unless the user asks.
```

---

## 10. Key Lessons & Pitfalls

These are things that burned hours before being resolved — listed so you don't repeat them.

### Context Window

| Issue | Symptom | Fix |
|-------|---------|-----|
| `MINIMUM_CONTEXT_LENGTH` error | Hermes refuses to start | Set `ollama_num_ctx: 65536` on every model, including embed models |
| Silent truncation | Responses cut off mid-sentence | Add `PARAMETER num_predict -1` to all Modelfiles |
| 92% throughput loss | Inference very slow at 64K | Change `OLLAMA_KV_CACHE_TYPE` from `q4_0` to `q8_0` |

### Hermes Stops / Hangs

| Issue | Symptom | Fix |
|-------|---------|-----|
| `provider: auto` in auxiliary | Hermes stops responding, no error | Pin every aux slot to a specific provider+model |
| `moa.default_preset` not set | `/moa` uses wrong preset | Set `moa.default_preset: local` explicitly |
| MoA references using wrong model names | MoA runs wrong models | Use Hermes aliases in `moa.presets.local.references` |

### Modelfile Mistakes

| Issue | Symptom | Fix |
|-------|---------|-----|
| `keep_alive` in Modelfile | Ollama rejects the Modelfile | Move to Hermes config as `keep_alive: -1` in the model entry |
| Ornith `ollama create` times out | "context deadline exceeded" error | Pull HF blobs first: `ollama pull hf.co/bartowski/...`, then create |
| Wrong `ollama_model` tag | Model not found at runtime | The `model:` field is your Hermes alias; `ollama_model:` must be the exact Ollama tag |

### VRAM Co-loading

| Issue | Symptom | Fix |
|-------|---------|-----|
| gpt-oss-20b + qwen3-vl-8b loaded together | Partial CPU offload, slow inference | Never load these two simultaneously; use context pressure rules |
| gemma4-heretic + qwen25-coder together | ~18 GB, spills past 16 GB VRAM | Fine for short tasks; avoid for anything needing full 64K |

---

## 14. Upgrade Path & Scaling Advice

### When You Have More VRAM (24+ GB)

With a 24 GB card (e.g. RTX 3090/4090/5090):

- `gpt-oss-20b` + `qwen3-vl-8b` can co-load (~24.5 GB — right at the edge)
- You can raise `ollama_num_batch: 1024` on all models for better throughput
- Consider adding a 30B+ reasoning model to the roster (e.g. `qwen3:30b`)

### When You Have Less VRAM (8 GB)

On 8 GB VRAM (RTX 3070/4060 Ti):

- Drop `gpt-oss-20b` — weight alone is ~12 GB
- Use `qwen3-14b-think` as default with `q4_K_M` quantization (~7 GB weights)
- Use `qwen25-coder-7b` as your workhorse coder
- Reduce `ollama_num_ctx: 32768` (halving KV cache cost to ~1.5 GB)
- Consider `q4_0` KV cache acceptable at 32K context (throughput loss is tolerable below 32K)

### Model Replacement Candidates

| Current | Next tier up | Trigger |
|---------|-------------|---------|
| qwythos-9b | QwQ-32B | Need 32K+ reasoning chains |
| ornith-9b | Ornith-1.0-30B (if released) | SWE-Bench ceiling hit |
| gpt-oss-20b | Devstral (Mistral) | Codebase-scale agentic tasks |
| qwen3-14b-think | Qwen3-32B | Hitting planning quality limit |

### OpenRouter as Escape Hatch

For truly massive models (70B+, MoE 100B+ like GLM-5.2 at 753B) that cannot run locally:

```yaml
# Add to providers in config.yaml
providers:
  openrouter:
    api: https://openrouter.ai/api/v1
    key_env: OPENROUTER_API_KEY
    models:
      - model: glm-5.2
        ollama_model: thudm/glm-4-32b  # best available proxy
        ollama_num_ctx: 128000
        context_length: 131072
```

Cost guard: Route to OpenRouter only on explicit user request or when all local
models fail. Never use it as a fallback in automated loops — costs accumulate quickly.

---

## Appendix: Quick Reference

### Model Alias → Ollama Tag

| Hermes alias | Ollama tag | Pull command |
|---|---|---|
| gpt-oss-20b | gpt-oss-20b | `ollama pull gpt-oss:20b` then build |
| gemma4-heretic-12b | gemma4-heretic-12b | custom Modelfile |
| qwen3-14b-think | qwen3:14b | `ollama pull qwen3:14b` |
| qwythos-9b | qwythos-9b | custom Modelfile |
| qwen25-coder-7b | qwen2.5-coder:7b | `ollama pull qwen2.5-coder:7b` |
| ornith-9b | ornith-9b | custom Modelfile (pull blobs first) |
| nomic-embed-text | nomic-embed-text:latest | `ollama pull nomic-embed-text:latest` |
| qwen3-vl-8b | qwen3-vl:8b | `ollama pull qwen3-vl:8b` |

### Routing Cheat Sheet

```
/moa                   → Hermes built-in MoA
Image input            → qwen3-vl-8b
Embeddings / RAG       → nomic-embed-text
Tool call / agent loop → gpt-oss-20b
Agentic coding         → ornith-9b
Quick code / scripts   → qwen25-coder-7b
Security / uncensored  → qwythos-9b
Planning / reasoning   → qwen3-14b-think
Conversation / fast    → gemma4-heretic-12b
Everything else        → gpt-oss-20b
```

### Verify Your Stack

```bash
# Ollama health
curl http://localhost:11434/api/tags | python3 -m json.tool

# Check all 8 models are listed
ollama list

# Check Flash Attention and KV cache type
journalctl -u ollama -n 100 | grep -E "(flash|kv_cache|KV)"

# GPU VRAM
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu --format=csv
```

---


