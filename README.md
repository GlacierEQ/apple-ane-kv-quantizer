# Apple ANE KV Quantizer — Neural Engine FP16/INT8 Quantizer 🍏

> **Quantization and memory optimization engine for Apple Neural Engine (ANE) & Metal GPU inference.**

[![Swift](https://img.shields.io/badge/Swift-5.9+-FA7343)]()
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8)]()
[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Domain](https://img.shields.io/badge/Domain-Edge%20AI-black)]()

---

## 🎯 For Recruiters & Hiring Managers

This repository implements an **Apple Neural Engine (ANE) KV-Cache Quantizer** — enabling large language models to run efficiently on Apple Silicon (M1/M2/M3/M4 & A-series chips). It demonstrates:

- **FP16 to INT8/INT4 dynamic quantization** tailored for ANE matrix multipliers
- **Swift & Metal API integration** executing native GPU compute pipelines
- **Go IPC dispatcher** for low-overhead multi-process tensor streaming
- **Memory footprint reduction** by 4x with minimal accuracy loss

**Why this matters**: On-device AI execution requires extreme memory efficiency. Quantizing KV caches specifically for Apple Silicon hardware enables private, offline LLM inference on consumer hardware.

---

## 🔬 For Engineers & Technical Reviewers

### Core Components

| Component | Language | Purpose |
|---|---|---|
| `src/ane_quantizer.swift` | Swift | Native Metal & ANE tensor quantization routines |
| `src/tensor_bridge.go` | Go | High-speed inter-process tensor ring buffer |
| `src/quantizer_engine.py` | Python | Quantization scale factor computation and PyTorch export |
| `tests/` | Python | Perplexity and loss evaluation test suite |

---

## 🤖 ML/AI & Programmatic Mesh Integration

- **MCP Tool**: `quantize_kv_cache()` — accessible tool for local model acceleration
- **Mastermind Sidecar**: Integrates with APEX Highway mesh
- **SHA-256 Integrity**: Tracked in `.integrity/file_hashes.json`

---

## ⚡ Quick Start

```bash
python3 src/quantizer_engine.py
python3 tests/test_quantizer.py
```
