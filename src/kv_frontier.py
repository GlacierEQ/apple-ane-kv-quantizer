"""Adaptive KV-cache planning with evidence-bound Pareto selection.

The planner deliberately models representation, memory traffic, and precision pressure.
It does not claim model-quality preservation, ANE execution, or measured Apple hardware
performance. The design borrows GlacierEQ excellence-core's Pareto-first selection law:
real tradeoffs remain visible instead of being collapsed into a single magic score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Iterable, Literal

EVIDENCE_STATE = "MODELED_KV_FRONTIER_NOT_HARDWARE_OR_QUALITY_MEASUREMENT"
_ALLOWED_BITS = (4, 8, 16)
Preference = Literal["balanced", "memory", "latency", "quality"]


class NoFeasiblePlan(ValueError):
    """Raised when explicit constraints reject every generated candidate."""


@dataclass(frozen=True)
class KVWorkload:
    """Transformer KV-cache workload expressed only in inspectable inputs."""

    tokens: int
    hidden_dim: int
    layers: int = 32
    batch_size: int = 1
    kv_width_ratio: float = 1.0
    memory_bandwidth_gbps: float = 800.0

    def __post_init__(self) -> None:
        for name in ("tokens", "hidden_dim", "layers", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.kv_width_ratio) or not 0 < self.kv_width_ratio <= 1:
            raise ValueError("kv_width_ratio must be finite and in (0, 1]")
        if not isfinite(self.memory_bandwidth_gbps) or self.memory_bandwidth_gbps <= 0:
            raise ValueError("memory_bandwidth_gbps must be finite and positive")

    @property
    def kv_values_per_token(self) -> int:
        # K and V each own a cache tensor. GQA/MQA can reduce the KV width relative
        # to the model hidden width, so the ratio is explicit rather than assumed.
        return ceil(2 * self.layers * self.hidden_dim * self.batch_size * self.kv_width_ratio)

    @property
    def fp16_bytes(self) -> int:
        return self.tokens * self.kv_values_per_token * 2


@dataclass(frozen=True)
class KVPlan:
    """One modeled recency-tiered precision plan."""

    hot_tokens: int
    hot_bits: int
    cold_bits: int
    memory_bytes: int
    modeled_transfer_ms: float
    compression_ratio_vs_fp16: float
    precision_pressure: float
    coordination_cost: int
    evidence_state: str = EVIDENCE_STATE

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    def dominates(self, other: "KVPlan") -> bool:
        """Pareto dominance over cost dimensions.

        Lower memory, transfer time, precision pressure, and coordination cost are
        better. At least one dimension must be strictly better.
        """

        mine = (
            self.memory_bytes,
            self.modeled_transfer_ms,
            self.precision_pressure,
            self.coordination_cost,
        )
        theirs = (
            other.memory_bytes,
            other.modeled_transfer_ms,
            other.precision_pressure,
            other.coordination_cost,
        )
        return all(left <= right for left, right in zip(mine, theirs)) and any(
            left < right for left, right in zip(mine, theirs)
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["memory_mb"] = round(self.memory_mb, 6)
        data["modeled_transfer_ms"] = round(self.modeled_transfer_ms, 9)
        data["compression_ratio_vs_fp16"] = round(self.compression_ratio_vs_fp16, 6)
        data["precision_pressure"] = round(self.precision_pressure, 6)
        return data


@dataclass(frozen=True)
class KVConstraints:
    max_memory_mb: float | None = None
    max_transfer_ms: float | None = None
    max_precision_pressure: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_memory_mb", "max_transfer_ms", "max_precision_pressure"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.max_precision_pressure is not None and self.max_precision_pressure > 1:
            raise ValueError("max_precision_pressure must be <= 1")

    def accepts(self, plan: KVPlan) -> bool:
        return (
            (self.max_memory_mb is None or plan.memory_mb <= self.max_memory_mb)
            and (
                self.max_transfer_ms is None
                or plan.modeled_transfer_ms <= self.max_transfer_ms
            )
            and (
                self.max_precision_pressure is None
                or plan.precision_pressure <= self.max_precision_pressure
            )
        )


class KVFrontierPlanner:
    """Generate, constrain, Pareto-filter, and select adaptive KV plans."""

    DEFAULT_HOT_FRACTIONS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)

    def __init__(
        self,
        workload: KVWorkload,
        *,
        cold_bits: Iterable[int] = _ALLOWED_BITS,
        hot_fractions: Iterable[float] = DEFAULT_HOT_FRACTIONS,
    ) -> None:
        self.workload = workload
        self.cold_bits = tuple(dict.fromkeys(int(bits) for bits in cold_bits))
        if not self.cold_bits or any(bits not in _ALLOWED_BITS for bits in self.cold_bits):
            raise ValueError("cold_bits must contain only 4, 8, or 16")
        fractions = tuple(dict.fromkeys(float(value) for value in hot_fractions))
        if not fractions or any(not isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError("hot_fractions must be finite values in [0, 1]")
        self.hot_fractions = fractions

    def _candidate(self, cold_bits: int, hot_fraction: float) -> KVPlan:
        hot_tokens = min(self.workload.tokens, round(self.workload.tokens * hot_fraction))
        cold_tokens = self.workload.tokens - hot_tokens
        values = self.workload.kv_values_per_token
        total_bits = values * (hot_tokens * 16 + cold_tokens * cold_bits)
        memory_bytes = ceil(total_bits / 8)
        bandwidth_bytes_per_second = self.workload.memory_bandwidth_gbps * 1e9 / 8
        transfer_ms = memory_bytes / bandwidth_bytes_per_second * 1000
        compression_ratio = self.workload.fp16_bytes / memory_bytes

        # This is deliberately a representation-pressure proxy, not a model-quality
        # estimate: it is exactly the normalized precision removed from cold tokens.
        cold_fraction = cold_tokens / self.workload.tokens
        precision_pressure = cold_fraction * ((16 - cold_bits) / 16)

        # Hybrid hot/cold precision needs one extra coordination boundary. Uniform
        # plans do not. The cost is ordinal and used only as a Pareto dimension.
        coordination_cost = int(hot_tokens not in {0, self.workload.tokens} and cold_bits != 16)
        return KVPlan(
            hot_tokens=hot_tokens,
            hot_bits=16,
            cold_bits=cold_bits,
            memory_bytes=memory_bytes,
            modeled_transfer_ms=transfer_ms,
            compression_ratio_vs_fp16=compression_ratio,
            precision_pressure=precision_pressure,
            coordination_cost=coordination_cost,
        )

    def candidates(self) -> list[KVPlan]:
        unique: dict[tuple[int, int, int], KVPlan] = {}
        for bits in self.cold_bits:
            for fraction in self.hot_fractions:
                plan = self._candidate(bits, fraction)
                key = (plan.hot_tokens, plan.hot_bits, plan.cold_bits)
                unique[key] = plan
        return sorted(
            unique.values(),
            key=lambda plan: (
                plan.memory_bytes,
                plan.precision_pressure,
                plan.coordination_cost,
            ),
        )

    @staticmethod
    def pareto_frontier(plans: Iterable[KVPlan]) -> list[KVPlan]:
        items = list(plans)
        frontier = [
            candidate
            for candidate in items
            if not any(other.dominates(candidate) for other in items if other is not candidate)
        ]
        return sorted(
            frontier,
            key=lambda plan: (
                plan.precision_pressure,
                plan.memory_bytes,
                plan.coordination_cost,
            ),
        )

    def feasible_frontier(self, constraints: KVConstraints | None = None) -> list[KVPlan]:
        active = constraints or KVConstraints()
        feasible = [plan for plan in self.candidates() if active.accepts(plan)]
        if not feasible:
            raise NoFeasiblePlan(
                "no KV plan satisfies the requested memory, transfer, and precision-pressure constraints"
            )
        return self.pareto_frontier(feasible)

    def select(
        self,
        constraints: KVConstraints | None = None,
        *,
        preference: Preference = "balanced",
    ) -> KVPlan:
        frontier = self.feasible_frontier(constraints)
        if preference not in {"balanced", "memory", "latency", "quality"}:
            raise ValueError("preference must be balanced, memory, latency, or quality")
        if preference == "memory":
            return min(frontier, key=lambda plan: (plan.memory_bytes, plan.precision_pressure))
        if preference == "latency":
            return min(frontier, key=lambda plan: (plan.modeled_transfer_ms, plan.precision_pressure))
        if preference == "quality":
            return min(frontier, key=lambda plan: (plan.precision_pressure, plan.memory_bytes))

        # Balanced ordering only orders the already non-dominated frontier. Each
        # dimension is normalized to its frontier range, preserving Pareto primacy.
        dimensions = (
            "memory_bytes",
            "modeled_transfer_ms",
            "precision_pressure",
            "coordination_cost",
        )
        bounds = {
            name: (
                min(float(getattr(plan, name)) for plan in frontier),
                max(float(getattr(plan, name)) for plan in frontier),
            )
            for name in dimensions
        }

        def score(plan: KVPlan) -> tuple[float, float, int]:
            normalized = []
            for name in dimensions:
                low, high = bounds[name]
                value = float(getattr(plan, name))
                normalized.append(0.0 if high == low else (value - low) / (high - low))
            return (sum(normalized), plan.precision_pressure, plan.memory_bytes)

        return min(frontier, key=score)
