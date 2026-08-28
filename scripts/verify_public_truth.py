#!/usr/bin/env python3
"""Fail-closed truth checks for the Apple-Silicon quantization study."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"PUBLIC_TRUTH_FAIL: {message}")


def main() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = readme.replace("**", "").replace("`", "")
    caps = json.loads((ROOT / "machine/capabilities.json").read_text(encoding="utf-8"))
    state = json.loads(
        (ROOT / "machine/excellence-state.json").read_text(encoding="utf-8")
    )

    require(
        "MODELED_SCENARIO_NOT_HARDWARE_MEASUREMENT" in readme,
        "modeled-evidence token missing",
    )
    require(
        "MODELED_KV_FRONTIER_NOT_HARDWARE_OR_QUALITY_MEASUREMENT" in readme,
        "frontier-evidence token missing",
    )
    require(
        "not affiliated with, endorsed by, or operated by Apple" in normalized,
        "Apple non-affiliation boundary missing",
    )
    require(
        "not a measured ANE bandwidth" in normalized,
        "hardware-measurement boundary missing",
    )
    require(
        "does not establish mesh/runtime integration" in normalized,
        "mesh boundary missing",
    )

    allowed = {
        "deterministic-kv-storage-size-modeling",
        "modeled-4bit-and-8bit-storage-reduction",
        "explicit-bandwidth-transfer-time-estimation",
        "validated-modeled-quantization-inputs",
        "full-kv-topology-footprint-modeling",
        "adaptive-recency-tiered-kv-planning",
        "constraint-based-kv-plan-selection",
        "pareto-kv-plan-frontier",
        "receipt-hashed-kv-plan-artifacts",
    }
    require(set(caps.get("capabilities", [])) == allowed, "capability allowlist drift")
    require(
        caps.get("operational_authority") is False,
        "operational authority must be false",
    )
    require(
        caps.get("apple_hardware_measurement") is False,
        "hardware measurement claim must be false",
    )
    require(
        caps.get("ane_kernel_execution") is False, "ANE execution claim must be false"
    )
    require(
        caps.get("metal_runtime_execution_proven") is False,
        "Metal runtime claim must be false",
    )
    require(
        caps.get("model_accuracy_preservation_proven") is False,
        "accuracy claim must be false",
    )
    require(
        caps.get("live_mcp_apex_mastermind_integration") is False,
        "live mesh claim must be false",
    )

    require(
        state.get("principal_state") == "FUNCTIONAL_CANDIDATE",
        "stale promotion restored",
    )
    require(
        state.get("operational_authority") is False,
        "state grants operational authority",
    )
    proof = state.get("gates", {}).get("DETERMINISTIC_PROOF_GREEN", {})
    require(
        proof.get("status") == "PENDING_CANONICAL_CI",
        "fresh exact-head proof gate missing",
    )

    print("PUBLIC_TRUTH_PASS")


if __name__ == "__main__":
    main()
