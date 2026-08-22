"""Adversarial and deterministic proof for the adaptive KV frontier planner."""

import unittest

from kv_frontier import (
    EVIDENCE_STATE,
    KVConstraints,
    KVFrontierPlanner,
    KVWorkload,
    NoFeasiblePlan,
)


class TestKVFrontierPlanner(unittest.TestCase):
    def setUp(self):
        # Exact FP16 footprint: 1024 tokens * (K+V) * 2 layers * 1024 width * 2 bytes = 8 MiB.
        self.workload = KVWorkload(
            tokens=1024,
            hidden_dim=1024,
            layers=2,
            batch_size=1,
            kv_width_ratio=1.0,
            memory_bandwidth_gbps=100.0,
        )
        self.planner = KVFrontierPlanner(self.workload)

    def test_workload_models_full_kv_topology(self):
        self.assertEqual(self.workload.fp16_bytes, 8 * 1024 * 1024)
        self.assertEqual(self.workload.kv_values_per_token, 4096)

    def test_uniform_4bit_candidate_is_exact_quarter_storage(self):
        plan = next(
            candidate
            for candidate in self.planner.candidates()
            if candidate.hot_tokens == 0 and candidate.cold_bits == 4
        )
        self.assertEqual(plan.memory_bytes, 2 * 1024 * 1024)
        self.assertEqual(plan.compression_ratio_vs_fp16, 4.0)
        self.assertEqual(plan.precision_pressure, 0.75)
        self.assertEqual(plan.evidence_state, EVIDENCE_STATE)

    def test_constraints_select_hybrid_recency_plan(self):
        selected = self.planner.select(
            KVConstraints(max_memory_mb=4.0, max_precision_pressure=0.60),
            preference="quality",
        )
        self.assertGreater(selected.hot_tokens, 0)
        self.assertLess(selected.hot_tokens, self.workload.tokens)
        self.assertLessEqual(selected.memory_mb, 4.0)
        self.assertLessEqual(selected.precision_pressure, 0.60)
        self.assertEqual(selected.coordination_cost, 1)

    def test_pareto_frontier_contains_no_dominated_plan(self):
        frontier = self.planner.feasible_frontier()
        for candidate in frontier:
            self.assertFalse(
                any(
                    other.dominates(candidate)
                    for other in frontier
                    if other is not candidate
                )
            )

    def test_impossible_constraints_fail_closed(self):
        with self.assertRaises(NoFeasiblePlan):
            self.planner.select(
                KVConstraints(max_memory_mb=1.0, max_precision_pressure=0.05)
            )

    def test_invalid_workload_and_policy_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            KVWorkload(tokens=0, hidden_dim=1024)
        with self.assertRaises(ValueError):
            KVWorkload(tokens=1, hidden_dim=1, kv_width_ratio=float("nan"))
        with self.assertRaises(ValueError):
            KVConstraints(max_precision_pressure=1.1)
        with self.assertRaises(ValueError):
            KVFrontierPlanner(self.workload, cold_bits=(3, 4))
        with self.assertRaises(ValueError):
            self.planner.select(preference="fastest")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
