# Apple-Silicon KV Quantization + Adaptive Frontier Planner

Independent GlacierEQ portfolio work exploring deterministic KV-cache storage tradeoffs, recency-tiered precision planning, and local quantization examples for Apple-Silicon-oriented scenarios.

**Status:** executable local model + adaptive planning engine + repository-native proof.  
**Evidence tokens:** `MODELED_SCENARIO_NOT_HARDWARE_MEASUREMENT` and `MODELED_KV_FRONTIER_NOT_HARDWARE_OR_QUALITY_MEASUREMENT`

This repository is **not affiliated with, endorsed by, or operated by Apple**. It does not claim proprietary Apple access, production deployment, Neural Engine kernel execution, measured device performance, or preserved model accuracy.

## What became stronger

The original estimator remains compatible, but the repository now has a second, deeper capability in `src/kv_frontier.py`:

1. models the complete K+V footprint across tokens, hidden width, layers, batch size, and an explicit GQA/MQA KV-width ratio;
2. generates uniform and recency-tiered 4/8/16-bit cache candidates;
3. computes exact modeled memory footprint and transfer-time arithmetic from explicit bandwidth assumptions;
4. derives a **precision-pressure** value from representation bits removed from cold tokens. This is an arithmetic representation-pressure proxy, not a model-quality estimate;
5. applies explicit memory, transfer, and precision-pressure constraints;
6. rejects impossible policies instead of silently returning a weak plan;
7. preserves every non-dominated tradeoff on a Pareto frontier before any preference ordering;
8. emits a selected plan plus a SHA-256-linked execution receipt through `scripts/kv_plan.py`.

That turns the repository from a single calculation into a small planning system with a refusal path, an inspectable tradeoff frontier, machine-readable artifacts, and a shared CI proof path.

## Example

```bash
python scripts/kv_plan.py \
  --tokens 8192 \
  --hidden-dim 4096 \
  --layers 32 \
  --kv-width-ratio 0.25 \
  --bandwidth-gbps 800 \
  --max-memory-mb 4096 \
  --max-precision-pressure 0.60 \
  --preference balanced \
  --output .verification-artifacts/apple-kv-plan.json
```

The generated JSON keeps the full non-dominated frontier as well as the selected candidate. Its receipt binds the artifact bytes to a SHA-256 digest.

## Engineering lineage actually adopted

This repository explicitly applies mechanisms from several stronger GlacierEQ systems rather than merely linking to them:

| Source | Mechanism adopted here | Repository-local implementation |
|---|---|---|
| `GlacierEQ/pro-code` | failure paths are first-class; significant execution leaves an audit trail; complexity must earn capability | impossible constraints raise `NoFeasiblePlan`; CLI emits content-hashed receipts; planner is typed and intentionally small |
| `GlacierEQ/Pro_Code` | doctrine is not proof until the target repo adopts and verifies it | adoption is implemented in source/tests/CI here rather than claimed by relationship alone |
| `GlacierEQ/glaciereq-excellence-core` | Pareto dominance precedes weighted or preference ordering | `KVFrontierPlanner.pareto_frontier()` preserves non-dominated memory/latency/precision/coordination tradeoffs |
| `GlacierEQ/apex-control-plane` | generated != executed != verified; state claims remain evidence-bound | model and hardware claims stay separately labeled; receipt state is `DETERMINISTIC_MODEL_EXECUTED` only |
| `GlacierEQ/public-actions-runner-host` | reusable strict verification with repository-owned proof scripts and uploaded artifacts | `.github/workflows/elite-core.yml` calls the shared runner and executes `scripts/ci/verify_elite_core.sh` |

This is an exercised code/CI integration for the mechanisms named above. It does **not** establish live APEX, MCP, Mastermind, or cross-repository runtime connectivity.

## What is verified here

The runnable proof surfaces are:

- `src/apple_ane_kv_quantizer.py` for the original deterministic storage/transfer estimator;
- `src/kv_frontier.py` for workload topology, candidate generation, constraints, Pareto frontier, selection, and refusal;
- `tests/test_ane.py` for the original model;
- `tests/test_kv_frontier.py` for full-footprint arithmetic, hybrid plans, frontier non-domination, and adversarial failure inputs;
- `scripts/kv_plan.py` for executable JSON plans and receipt hashes;
- `scripts/verify_public_truth.py` for public-claim boundaries;
- `scripts/ci/verify_elite_core.sh` for the repository-owned integrated proof path.

The original estimator deterministically computes, from explicit inputs:

- FP16 KV-cache storage size;
- modeled 4-bit or 8-bit storage size;
- arithmetic storage reduction;
- transfer-time estimate under an explicitly configured memory-bandwidth assumption.

For example, converting the **storage representation** from 16 bits to 4 bits is arithmetically a 75% byte reduction. That is a modeled representation result, **not a measured ANE bandwidth**, latency, perplexity, or accuracy result.

The frontier planner additionally models the transformer cache topology that the first estimator intentionally omitted. Its GQA/MQA width factor is an explicit input so reduced KV-head width is visible rather than buried in an assumption.

## Engineering anatomy

| Surface | Current evidence-bound role |
|---|---|
| `src/apple_ane_kv_quantizer.py` | Backward-compatible deterministic storage/transfer estimator |
| `src/kv_frontier.py` | Adaptive recency-tiered planner, constraints, Pareto frontier, fail-closed selection |
| `scripts/kv_plan.py` | Executable planner CLI + JSON artifact + SHA-256 receipt |
| `tests/test_ane.py` | Regression tests for 4-bit/8-bit estimates and fail-closed inputs |
| `tests/test_kv_frontier.py` | Deterministic and adversarial frontier proof |
| `src/quantizer.go` | Local INT8 quantization example; not a hardware benchmark |
| `src/MetalComputeEngine.swift` | Metal API source example; not proof of ANE execution and not exercised by Linux CI |
| `mastermind_sidecar.py` | Local status-report helper only; it **does not establish mesh/runtime integration** |
| `.github/workflows/public-truth.yml` | 3-version Python truth-boundary verification |
| `.github/workflows/elite-core.yml` | Shared GlacierEQ runner integration with artifact-producing repository proof |

## Native proof

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/verify_public_truth.py
bash scripts/ci/verify_elite_core.sh
```

The repository-owned Public Truth Gate runs the Python proof on supported CI versions and verifies that the public README retains the modeled-evidence and non-affiliation boundaries. The Elite Core gate adds the shared GlacierEQ verification runner and requires a generated, hash-linked KV planning artifact.

## Explicit nonclaims

Current repository evidence does **not** establish:

- Apple Neural Engine kernel dispatch;
- zero-copy unified-memory operation;
- live Metal compute execution on Apple hardware;
- 4× measured end-to-end memory improvement;
- minimal or preserved model accuracy/perplexity;
- production inference performance;
- MCP tool registration;
- live APEX/AKOS/Mastermind runtime connectivity;
- Apple employment, affiliation, endorsement, or proprietary access.

Those are higher evidence states and require their own hardware/runtime receipts before public promotion.

## Why the capability matters

The repository now separates three things that are often carelessly mixed together:

```text
REPRESENTATION ARITHMETIC
        ↓
SYSTEM-LEVEL KV PLANNING
        ↓
HARDWARE / MODEL-QUALITY EVIDENCE
```

The first two are executable here. The third remains explicitly outside the current evidence state.

The useful innovation is the composition: transformer topology + recency-tiered precision + explicit constraints + Pareto preservation + refusal + content-hashed execution receipts. None of those primitives is magical alone. Together they form an inspectable planning engine that can be driven by future real-device measurements without pretending those measurements already exist.
