"""Adaptive, evidence-bound KV-cache precision planner.

This module models precision allocation across KV-cache segments. It does not
dispatch ANE/Metal kernels or claim measured model quality. The planner exposes
the assumptions, computes a Pareto frontier, and fails closed when budgets are
impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from typing import Iterable

EVIDENCE_STATE = "MODELED_ADAPTIVE_PLAN_NOT_HARDWARE_MEASUREMENT"
_BITS = (4, 8, 16)
_RISK_MULTIPLIER = {4: 1.0, 8: 0.35, 16: 0.0}


@dataclass(frozen=True)
class KVSegment:
    name: str
    tokens: int
    hidden_dim: int
    importance: float = 0.5
    access_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("segment name must be non-empty")
        if self.tokens <= 0:
            raise ValueError("tokens must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not isfinite(self.importance) or not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be finite and within [0, 1]")
        if not isfinite(self.access_weight) or self.access_weight <= 0:
            raise ValueError("access_weight must be finite and positive")

    @property
    def fp16_bytes(self) -> int:
        return self.tokens * self.hidden_dim * 2


@dataclass(frozen=True)
class SegmentDecision:
    name: str
    bits: int
    bytes: int
    modeled_risk_contribution: float
    modeled_transfer_time_ms: float


@dataclass(frozen=True)
class KVPlan:
    decisions: tuple[SegmentDecision, ...]
    total_bytes: int
    memory_mib: float
    modeled_quality_risk: float
    modeled_transfer_time_ms: float
    evidence_state: str = EVIDENCE_STATE

    @property
    def precision_map(self) -> dict[str, int]:
        return {decision.name: decision.bits for decision in self.decisions}

    def as_dict(self) -> dict[str, object]:
        return {
            "precision_map": self.precision_map,
            "total_bytes": self.total_bytes,
            "memory_mib": round(self.memory_mib, 6),
            "modeled_quality_risk": round(self.modeled_quality_risk, 8),
            "modeled_transfer_time_ms": round(self.modeled_transfer_time_ms, 8),
            "evidence_state": self.evidence_state,
        }


class AdaptiveKVPlanner:
    """Search precision allocations under explicit memory and risk budgets."""

    def __init__(
        self,
        *,
        memory_bandwidth_gbps: float = 800.0,
        allowed_bits: Iterable[int] = _BITS,
        max_segments: int = 10,
    ) -> None:
        if not isfinite(memory_bandwidth_gbps) or memory_bandwidth_gbps <= 0:
            raise ValueError("memory_bandwidth_gbps must be finite and positive")
        bits = tuple(sorted(set(int(value) for value in allowed_bits)))
        if not bits or any(value not in _BITS for value in bits):
            raise ValueError("allowed_bits must be a non-empty subset of {4, 8, 16}")
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.memory_bandwidth_gbps = float(memory_bandwidth_gbps)
        self.allowed_bits = bits
        self.max_segments = int(max_segments)

    def _plan_for(self, segments: tuple[KVSegment, ...], precisions: tuple[int, ...]) -> KVPlan:
        total_fp16 = sum(segment.fp16_bytes for segment in segments)
        decisions: list[SegmentDecision] = []
        total_bytes = 0
        weighted_risk = 0.0
        transfer_ms = 0.0

        for segment, bits in zip(segments, precisions, strict=True):
            size = segment.fp16_bytes * bits // 16
            risk = segment.fp16_bytes * segment.importance * _RISK_MULTIPLIER[bits]
            segment_transfer = (
                size / (self.memory_bandwidth_gbps * 1e9 / 8)
            ) * 1000.0 * segment.access_weight
            total_bytes += size
            weighted_risk += risk
            transfer_ms += segment_transfer
            decisions.append(
                SegmentDecision(
                    name=segment.name,
                    bits=bits,
                    bytes=size,
                    modeled_risk_contribution=(risk / total_fp16 if total_fp16 else 0.0),
                    modeled_transfer_time_ms=segment_transfer,
                )
            )

        return KVPlan(
            decisions=tuple(decisions),
            total_bytes=total_bytes,
            memory_mib=total_bytes / (1024 * 1024),
            modeled_quality_risk=weighted_risk / total_fp16,
            modeled_transfer_time_ms=transfer_ms,
        )

    @staticmethod
    def _dominates(left: KVPlan, right: KVPlan) -> bool:
        no_worse = (
            left.total_bytes <= right.total_bytes
            and left.modeled_quality_risk <= right.modeled_quality_risk
            and left.modeled_transfer_time_ms <= right.modeled_transfer_time_ms
        )
        strictly_better = (
            left.total_bytes < right.total_bytes
            or left.modeled_quality_risk < right.modeled_quality_risk
            or left.modeled_transfer_time_ms < right.modeled_transfer_time_ms
        )
        return no_worse and strictly_better

    def enumerate_plans(self, segments: Iterable[KVSegment]) -> tuple[KVPlan, ...]:
        normalized = tuple(segments)
        if not normalized:
            raise ValueError("at least one segment is required")
        if len(normalized) > self.max_segments:
            raise ValueError(
                f"segment count {len(normalized)} exceeds max_segments={self.max_segments}"
            )
        names = [segment.name for segment in normalized]
        if len(names) != len(set(names)):
            raise ValueError("segment names must be unique")

        return tuple(
            self._plan_for(normalized, precisions)
            for precisions in product(self.allowed_bits, repeat=len(normalized))
        )

    def pareto_frontier(self, segments: Iterable[KVSegment]) -> tuple[KVPlan, ...]:
        plans = self.enumerate_plans(segments)
        frontier = [
            plan
            for plan in plans
            if not any(
                self._dominates(other, plan)
                for other in plans
                if other is not plan
            )
        ]
        frontier.sort(
            key=lambda plan: (
                plan.total_bytes,
                plan.modeled_quality_risk,
                plan.modeled_transfer_time_ms,
                tuple(sorted(plan.precision_map.items())),
            )
        )
        return tuple(frontier)

    def choose(
        self,
        segments: Iterable[KVSegment],
        *,
        memory_budget_mib: float,
        max_modeled_quality_risk: float,
    ) -> KVPlan:
        if not isfinite(memory_budget_mib) or memory_budget_mib <= 0:
            raise ValueError("memory_budget_mib must be finite and positive")
        if (
            not isfinite(max_modeled_quality_risk)
            or not 0.0 <= max_modeled_quality_risk <= 1.0
        ):
            raise ValueError("max_modeled_quality_risk must be within [0, 1]")

        feasible = [
            plan
            for plan in self.pareto_frontier(segments)
            if plan.memory_mib <= memory_budget_mib
            and plan.modeled_quality_risk <= max_modeled_quality_risk
        ]
        if not feasible:
            raise ValueError(
                "no feasible precision plan satisfies both memory and modeled-risk budgets"
            )
        return min(
            feasible,
            key=lambda plan: (
                plan.modeled_transfer_time_ms,
                plan.total_bytes,
                plan.modeled_quality_risk,
                tuple(sorted(plan.precision_map.items())),
            ),
        )
