# HermesAgent-MultiModel

**A fully-local, multi-model AI agent framework with Mixture-of-Agents (MoA) orchestration.** Run a high-performance agentic stack on consumer GPUs (RTX 4080/5080) using Hermes + Ollama, with zero cloud dependencies.

## 🎯 Quick Summary

HermesAgent-MultiModel implements a **local-first approach** to multi-model inference, routing tasks to specialized models (planning, coding, reasoning, vision) via an intelligent orchestration layer. It includes:

- **8-model roster**: gpt-oss-20b (tools), gemma4-heretic (fast), qwen3-14b (planning), qwythos-9b (reasoning), qwen25-coder (scripts), ornith-9b (SWE-Bench 69.4%), embeddings, vision
- **Mixture-of-Agents (MoA)**: Query multiple models in parallel, synthesize via aggregator
- **Local Ollama backend**: No API keys, full privacy, 64K context windows
- **Hermes integration**: Task routing, personalities, tool-calling, agentic loops
- **Optimized for RTX 4080/5080** (16 GB VRAM, 48+ GB RAM)

## 📦 What's in This Repo

```
├── README.md                          # This file — start here
├── .hermes/
│   ├── config.yaml                   # Hermes providers + routing config
│   └── bin/local_moa.py             # External Mixture-of-Agents CLI
├── docs/
│   └── optimization-guide.md         # 15-section field guide: hardware, models, tuning
├── skills/
│   └── mixture-of-agents/
│       └── README.md                 # MoA usage & examples
└── .github/                          # CI/CD workflows (optional)
```

## 🚀 Getting Started

### 1. **Prerequisites**
- **GPU**: RTX 4080 / 5080 or equivalent (16 GB VRAM minimum)
- **CPU**: 16+ cores (Intel/AMD)
- **RAM**: 48+ GB (64 GB ideal)
- **Storage**: 100 GB free NVMe SSD
- **OS**: Ubuntu 22.04+, WSL2 on Windows 11, or macOS 12+
- **Python**: 3.10+

### 2. **Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. **Pull & Build Models**

See the full [Ollama Pull Reference](./docs/optimization-guide.md#5-ollama-pull-reference) for all commands. Quick start:

```bash
# Direct pulls (registry)
ollama pull gpt-oss:20b
ollama pull qwen3:14b
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text:latest
ollama pull qwen3-vl:8b

# Custom builds (pull base, then create)
ollama pull igorls/gemma-4-12B-it-heretic-GGUF:Q4_K_M
ollama create gemma4-heretic-12b -f ~/ollama-modelfiles/gemma4-heretic-12b.Modelfile

ollama pull richardyoung/qwythos-9b-abliterated
ollama create qwythos-9b -f ~/ollama-modelfiles/qwythos-9b.Modelfile

ollama pull hf.co/bartowski/deepreinforce-ai_Ornith-1.0-9B-GGUF:Q4_K_M
ollama create ornith-9b -f ~/ollama-modelfiles/ornith-9b.Modelfile

# Verify
ollama list
```

### 4. **Configure Hermes**

Edit `~/.hermes/config.yaml` with your model provider block. Full reference: [Hermes Config — Providers Block](./docs/optimization-guide.md#6-hermes-config--providers-block).

**Essential settings** (paste into providers block):
```yaml
model:
  default: gpt-oss-20b
  provider: custom:ollama
  base_url: http://localhost:11434/v1
  timeout: 600
  context_length: 65536
  max_tokens: 32768
  ollama_num_ctx: 65536
```

### 5. **Tune Ollama for Performance**

```bash
sudo systemctl edit ollama
```

Add:
```ini
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

Then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

See [Ollama Installation & Systemd Tuning](./docs/optimization-guide.md#2-ollama-installation--systemd-tuning) for details on why these matter.

## 📚 Model Roster at a Glance

| Model | Role | VRAM @ 64K | Strength |
|-------|------|-----------|----------|
| **gpt-oss-20b** | Default, tools, agentic | ~15.5 GB | Best tool-caller (23/25); MoE 20B |
| **gemma4-heretic-12b** | Fast general, aux background | ~10.4 GB | Fastest generative model |
| **qwen3-14b-think** | Planning, reasoning, delegation | ~12 GB | Supports `/think` suffix |
| **qwythos-9b** | Deep analysis, security research | ~8.6 GB | Mild abliteration; reasoning blocks |
| **qwen25-coder-7b** | Fast single-file coding | ~8 GB | Lightest; first attempt for scripts |
| **ornith-9b** | Agentic coding, SWE-Bench | ~8.6 GB | **69.4% SWE-Bench**; best for multi-step |
| **nomic-embed-text** | Embeddings, RAG, search | <1 GB | Never for generation |
| **qwen3-vl-8b** | Vision, OCR, multimodal | ~9 GB | Sole vision model |

**Co-loading rule:** Do NOT load gpt-oss-20b + qwen3-vl-8b simultaneously (~24.5 GB exceeds RTX 5080). One or the other at a time.

## 🔄 Mixture-of-Agents (MoA)

Query multiple models in parallel, then synthesize via an aggregator:

```bash
~/.hermes/bin/local_moa.py --prompt "Your question here"
```

**Default setup:**
- References: `qwen3:14b`, `ornith-9b`  
- Aggregator: `gpt-oss-20b`

**Custom example:**
```bash
~/.hermes/bin/local_moa.py \
  --prompt "Explain RSA vs DSA" \
  --references gpt-oss-20b qwen3:14b \
  --aggregator qwen3:14b \
  --temperature 0.3 \
  --timeout 240
```

See [local_moa — Usage Guide](./skills/mixture-of-agents/README.md) for full documentation.

## 🛠️ Advanced Configuration

### Hardware Tuning
If running on **WSL2**, set aggressive resource limits in `C:\Users\<YourName>\.wslconfig`:
```ini
[wsl2]
memory=40GB
processors=24
swap=32GB
localhostForwarding=true
```

### Modelfile Authoring
Three models need custom Modelfiles (HF-hosted, temperature-baked, or name mismatch fixes). See [Modelfile Authoring](./docs/optimization-guide.md#4-modelfile-authoring) for examples.

### KV Cache Tuning
- **q4_0**: 50% VRAM savings but **92% throughput loss** at 64K context → DO NOT USE  
- **q8_0**: 25% VRAM savings, <5% throughput loss → **RECOMMENDED**  

### Context Window & VRAM Budget
Formula: `model_weights_GB + (64K tokens × 2 bytes × layers × heads × head_dim) / 1e9`

Examples at 64K context, q8_0 KV cache:
- gpt-oss-20b: 12 GB weights + 3.5 GB KV = **15.5 GB total**
- ornith-9b: 5.6 GB weights + 3.0 GB KV = **8.6 GB total**

See [VRAM Budget at 64K Context](./docs/optimization-guide.md#vram-budget-at-64k-context-q8_0-kv-cache) for the full table.

## 📖 Complete Documentation

- **[optimization-guide.md](./docs/optimization-guide.md)** — 15-section deep dive
  - Hardware requirements & WSL2 tuning
  - Ollama systemd configuration
  - Model roster notes & comparisons
  - Modelfile authoring & pull reference
  - Hermes provider & routing config
  - MoA presets & personalities
  - Key pitfalls & troubleshooting
  - Upgrade paths & scaling advice

- **[local_moa Usage Guide](./skills/mixture-of-agents/README.md)** — MoA CLI reference
  - Basic usage & examples
  - Model alias → Ollama tag mapping
  - Option flags & environment variables

- **[Hermes Config Skeleton](./hermes/config.yaml)** — Copy-paste provider block

## 🧪 Troubleshooting

### Ollama Hangs / Timeouts
- Cold load (first inference after restart) can take 60–90 s with large models  
- Set `--timeout 240` when running 3+ models in MoA
- Enable FlashAttention: `OLLAMA_FLASH_ATTENTION=1`

### Out-of-Memory (OOM)
- Check co-loading rules (gpt-oss-20b + qwen3-vl-8b = too big)
- Verify WSL2 memory allocation if on Windows
- Review VRAM budget table; may need smaller models or less context

### Modelfile Build Fails
- Ensure Ollama base is pulled first: `ollama pull <base-tag>`
- For Hugging Face models, allow extra timeout (metadata is large)
- Verify `num_predict -1` is set (prevents silent truncation in agent loops)

### Model Not Appearing in `ollama list`
- Check create command succeeded: `ollama list | grep <model-name>`
- If missing, re-run: `ollama create <alias> -f <Modelfile>`
- Verify `FROM` line in Modelfile points to correct base

## 🎓 Key Concepts

**Task Routing:** Hermes automatically selects the right model based on task type (tool use → gpt-oss-20b, fast conversation → gemma4-heretic, coding → qwen25-coder or ornith-9b).

**Agentic Loops:** Models like ornith-9b and gpt-oss-20b support tool-calling and multi-turn planning, enabling autonomous task completion.

**Reasoning Blocks:** qwythos-9b and ornith-9b emit `<think>...</think>` blocks; Hermes parses these automatically for better plan transparency.

**MoA Synthesis:** Responses from reference models are anonymized to prevent aggregator bias. Aggregator sees only the anonymous drafts + original question.

## 📋 Project Status

- **Created:** 2026-06-29
- **Language:** Python 100%
- **License:** (See LICENSE, if present)
- **Maintainer:** InfiniteWhispers

## 🔗 Related Links

- [Ollama](https://ollama.com)
- [Hermes AI Agent](https://github.com/hermes-ai/hermes) *(or relevant Hermes repo)*
- [SWE-Bench](https://github.com/aider-ai/aider/blob/main/docs/install.md)

---

**Ready to start?**  
1. Ensure GPU + 48+ GB RAM
2. Install Ollama
3. Pull & build 8 models (takes ~60 min on good internet)
4. Configure Hermes provider block
5. Run your first query with `local_moa.py`

For detailed tuning, see [optimization-guide.md](./docs/optimization-guide.md).
```

---

## Key Improvements

1. **Executive Summary** — Lead with what the project *does*, not just the tagline
2. **Complete Getting Started** — 5 concrete steps from zero to first inference
3. **Visual Model Table** — Quick reference for routing decisions
4. **Co-loading Rules** — Critical gotcha (gpt-oss-20b + qwen3-vl-8b)
5. **MoA Quick Example** — Shows both default and custom usage
6. **Troubleshooting Section** — Addresses common hangs, OOM, config issues
7. **Cross-References** — Links to detailed docs for deeper dives
8. **Hardware Tuning** — WSL2, KV cache, VRAM budget formulas upfront
9. **Concepts Explained** — Task routing, agentic loops, reasoning blocks in plain language
10. **Call to Action** — 5-step path to first working inference

This README balances **depth** (links to 15-section optimization guide) with **actionability** (you can follow Getting Started without reading everything else).
