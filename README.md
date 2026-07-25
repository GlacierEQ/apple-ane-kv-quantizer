# Apple ANE KV-Quantizer

> **Production Solution for Apple Silicon ANE & Unified Memory Optimization**

## Overview
Zero-copy unified memory KV-cache quantizer and INT4/FP16 kernel compiler for Apple Neural Engine (ANE).

## Verification
```bash
PYTHONPATH=src python3 tests/test_ane.py
python3 mastermind_sidecar.py
```
