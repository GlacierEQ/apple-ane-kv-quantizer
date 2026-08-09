"""Deterministic KV-cache storage and transfer estimator for Apple-Silicon scenarios.

This module does not invoke Apple Neural Engine hardware, compile an ANE kernel, or
measure device latency. It models storage reduction and transfer time from explicit
inputs so those assumptions remain testable and reproducible.
"""

from __future__ import annotations

import time
from typing import Any


class AppleANEKVQuantizer:
    """Model KV-cache storage reduction under explicit Apple-Silicon assumptions."""

    def __init__(self, memory_bandwidth_gbps: float = 800.0, ane_cores: int = 16):
        if memory_bandwidth_gbps <= 0:
            raise ValueError("memory_bandwidth_gbps must be positive")
        if ane_cores <= 0:
            raise ValueError("ane_cores must be positive")
        self.memory_bandwidth_gbps = float(memory_bandwidth_gbps)
        self.ane_cores = int(ane_cores)

    def quantize_kv_cache(
        self, tokens_count: int, hidden_dim: int = 4096, target_bits: int = 4
    ) -> tuple[dict[str, Any], float]:
        """Return a modeled storage/transfer estimate for an FP16 KV-cache.

        The method preserves the historical public API name for compatibility, but
        it performs deterministic arithmetic only. ``target_bits`` may be 4 or 8.
        """

        if tokens_count <= 0:
            raise ValueError("tokens_count must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if target_bits not in {4, 8}:
            raise ValueError("target_bits must be 4 or 8")

        started = time.perf_counter()
        raw_bytes = tokens_count * hidden_dim * 2  # FP16 = 16 bits = 2 bytes.
        quantized_bytes = tokens_count * hidden_dim * target_bits / 8
        storage_reduction_percent = (1.0 - quantized_bytes / raw_bytes) * 100.0
        modeled_transfer_time_ms = (
            quantized_bytes / (self.memory_bandwidth_gbps * 1e9 / 8)
        ) * 1000.0

        metrics = {
            "tokens_count": tokens_count,
            "hidden_dim": hidden_dim,
            "target_bits": target_bits,
            "raw_size_mb": round(raw_bytes / (1024 * 1024), 4),
            "quantized_size_mb": round(quantized_bytes / (1024 * 1024), 4),
            "storage_reduction_percent": round(storage_reduction_percent, 2),
            "modeled_transfer_time_ms": round(modeled_transfer_time_ms, 6),
            "assumed_memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "configured_ane_cores": self.ane_cores,
            "evidence_state": "MODELED_SCENARIO_NOT_HARDWARE_MEASUREMENT",
        }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return metrics, elapsed_ms
