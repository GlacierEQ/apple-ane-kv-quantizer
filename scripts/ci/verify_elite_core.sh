#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR=".verification-artifacts"
PLAN="${ARTIFACT_DIR}/apple-kv-plan.json"
RECEIPT="${ARTIFACT_DIR}/apple-kv-plan.receipt.json"
mkdir -p "${ARTIFACT_DIR}"

python -m compileall -q src tests scripts mastermind_sidecar.py
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v \
  | tee "${ARTIFACT_DIR}/unittest.txt"
python scripts/verify_public_truth.py \
  | tee "${ARTIFACT_DIR}/public-truth.txt"

python scripts/kv_plan.py \
  --tokens 8192 \
  --hidden-dim 4096 \
  --layers 32 \
  --batch-size 1 \
  --kv-width-ratio 0.25 \
  --bandwidth-gbps 800 \
  --max-memory-mb 4096 \
  --max-precision-pressure 0.60 \
  --preference balanced \
  --output "${PLAN}" \
  --receipt "${RECEIPT}" \
  | tee "${ARTIFACT_DIR}/kv-plan-cli.txt"

python - <<'PY'
import hashlib
import json
from pathlib import Path

plan_path = Path('.verification-artifacts/apple-kv-plan.json')
receipt_path = Path('.verification-artifacts/apple-kv-plan.receipt.json')
plan = json.loads(plan_path.read_text(encoding='utf-8'))
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))

assert plan['schema'] == 'glaciereq.apple-kv-frontier-plan.v1'
assert plan['selected']['evidence_state'] == 'MODELED_KV_FRONTIER_NOT_HARDWARE_OR_QUALITY_MEASUREMENT'
assert plan['frontier_count'] >= 1
assert plan['selected']['memory_mb'] <= 4096
assert plan['selected']['precision_pressure'] <= 0.60
actual = hashlib.sha256(plan_path.read_bytes()).hexdigest()
assert receipt['artifact_sha256'] == actual
assert receipt['verified_state'] == 'DETERMINISTIC_MODEL_EXECUTED'
print(json.dumps({
    'elite_core': 'PASS',
    'frontier_count': plan['frontier_count'],
    'selected': plan['selected'],
    'artifact_sha256': actual,
}, indent=2))
PY
