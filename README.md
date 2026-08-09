# Apple-Silicon KV Quantization Study

Independent GlacierEQ portfolio work exploring deterministic KV-cache storage tradeoffs and local quantization examples for Apple-Silicon-oriented scenarios.

**Status:** local model + reference implementations.  
**Evidence token:** `MODELED_SCENARIO_NOT_HARDWARE_MEASUREMENT`

This repository is **not affiliated with, endorsed by, or operated by Apple**. It does not claim proprietary Apple access, production deployment, Neural Engine kernel execution, measured device performance, or preserved model accuracy.

## What is verified here

The canonical runnable proof surface is `src/apple_ane_kv_quantizer.py` plus `tests/`.

It deterministically computes, from explicit inputs:

- FP16 KV-cache storage size;
- modeled 4-bit or 8-bit storage size;
- arithmetic storage reduction;
- transfer-time estimate under an explicitly configured memory-bandwidth assumption.

For example, converting the **storage representation** from 16 bits to 4 bits is arithmetically a 75% byte reduction. That is a modeled representation result, **not** a measured ANE bandwidth, latency, perplexity, or accuracy result.

## Engineering anatomy

| Surface | Current evidence-bound role |
|---|---|
| `src/apple_ane_kv_quantizer.py` | Deterministic storage/transfer estimator; canonical tested capability |
| `tests/test_ane.py` | Regression tests for 4-bit/8-bit estimates and fail-closed inputs |
| `tests/test_quantizer.py` | Small independent quantization-scale example |
| `src/quantizer.go` | Local INT8 quantization example; not a hardware benchmark |
| `src/MetalComputeEngine.swift` | Metal API source example; not proof of ANE execution and not exercised by Linux CI |
| `mastermind_sidecar.py` | Local status-report helper only; it does not establish mesh/runtime integration |

Historical or architectural files remain available for provenance, but they do not raise the evidence level of the tested capability.

## Native proof

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

The repository-owned Public Truth Gate runs the Python proof on supported CI versions and verifies that the public README retains the modeled-evidence and non-affiliation boundaries.

## Explicit nonclaims

Current repository evidence does **not** establish:

- Apple Neural Engine kernel dispatch;
- zero-copy unified-memory operation;
- live Metal compute execution on Apple hardware;
- 4× measured end-to-end memory improvement;
- minimal or preserved model accuracy/perplexity;
- production inference performance;
- MCP tool registration;
- APEX/AKOS/Mastermind live runtime connectivity;
- Apple employment, affiliation, endorsement, or proprietary access.

Those are higher evidence states and require their own hardware/runtime receipts before public promotion.

## Why the capability matters

The useful engineering mechanism here is the separation of **quantization arithmetic from hardware claims**. The local model makes representation assumptions explicit and reproducible, which provides a clean substrate for future hardware experiments without reporting those experiments as complete before they exist.
