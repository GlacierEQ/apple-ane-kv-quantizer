#!/usr/bin/env python3
"""Generate an evidence-bound adaptive KV plan and a content-hashed receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kv_frontier import KVConstraints, KVFrontierPlanner, KVWorkload  # noqa: E402


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tokens", type=int, required=True)
    cli.add_argument("--hidden-dim", type=int, required=True)
    cli.add_argument("--layers", type=int, default=32)
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--kv-width-ratio", type=float, default=1.0)
    cli.add_argument("--bandwidth-gbps", type=float, default=800.0)
    cli.add_argument("--max-memory-mb", type=float)
    cli.add_argument("--max-transfer-ms", type=float)
    cli.add_argument("--max-precision-pressure", type=float)
    cli.add_argument(
        "--preference",
        choices=("balanced", "memory", "latency", "quality"),
        default="balanced",
    )
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--receipt", type=Path)
    return cli


def main() -> int:
    args = parser().parse_args()
    workload = KVWorkload(
        tokens=args.tokens,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        batch_size=args.batch_size,
        kv_width_ratio=args.kv_width_ratio,
        memory_bandwidth_gbps=args.bandwidth_gbps,
    )
    constraints = KVConstraints(
        max_memory_mb=args.max_memory_mb,
        max_transfer_ms=args.max_transfer_ms,
        max_precision_pressure=args.max_precision_pressure,
    )
    planner = KVFrontierPlanner(workload)
    frontier = planner.feasible_frontier(constraints)
    selected = planner.select(constraints, preference=args.preference)

    payload = {
        "schema": "glaciereq.apple-kv-frontier-plan.v1",
        "evidence_state": selected.evidence_state,
        "workload": asdict(workload),
        "constraints": asdict(constraints),
        "preference": args.preference,
        "candidate_count": len(planner.candidates()),
        "frontier_count": len(frontier),
        "selected": selected.to_dict(),
        "frontier": [plan.to_dict() for plan in frontier],
        "claims_not_established": [
            "Apple Neural Engine execution",
            "Metal runtime execution",
            "model quality or perplexity preservation",
            "measured device bandwidth or latency",
            "production inference performance",
        ],
        "integration_lineage": {
            "selection_law": "GlacierEQ/glaciereq-excellence-core Pareto-first APEX pattern",
            "engineering_law": "GlacierEQ/pro-code failure/audit/comprehensibility principles",
            "execution_state": "GlacierEQ/apex-control-plane proof-bound state discipline",
        },
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)

    receipt_path = args.receipt or args.output.with_suffix(
        args.output.suffix + ".receipt.json"
    )
    receipt = {
        "schema": "glaciereq.apple-kv-frontier-receipt.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": os.environ.get(
            "GITHUB_REPOSITORY", "GlacierEQ/apple-ane-kv-quantizer"
        ),
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "artifact": str(args.output),
        "artifact_bytes": len(encoded),
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "selected_plan": selected.to_dict(),
        "verified_state": "DETERMINISTIC_MODEL_EXECUTED",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
