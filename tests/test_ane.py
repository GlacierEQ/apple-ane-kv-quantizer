"""Regression tests for the deterministic Apple-Silicon KV-cache estimator."""

import unittest

from apple_ane_kv_quantizer import AppleANEKVQuantizer


class TestAppleANEKVQuantizer(unittest.TestCase):
    def test_kv_storage_model(self):
        quantizer = AppleANEKVQuantizer(memory_bandwidth_gbps=800.0, ane_cores=16)
        metrics, elapsed = quantizer.quantize_kv_cache(
            tokens_count=8192,
            hidden_dim=4096,
            target_bits=4,
        )

        self.assertEqual(metrics["storage_reduction_percent"], 75.0)
        self.assertEqual(metrics["target_bits"], 4)
        self.assertEqual(
            metrics["evidence_state"],
            "MODELED_SCENARIO_NOT_HARDWARE_MEASUREMENT",
        )
        self.assertGreater(metrics["modeled_transfer_time_ms"], 0.0)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_int8_storage_model(self):
        metrics, _ = AppleANEKVQuantizer().quantize_kv_cache(
            tokens_count=1024,
            hidden_dim=1024,
            target_bits=8,
        )
        self.assertEqual(metrics["storage_reduction_percent"], 50.0)

    def test_invalid_model_inputs_fail_closed(self):
        quantizer = AppleANEKVQuantizer()
        with self.assertRaises(ValueError):
            quantizer.quantize_kv_cache(tokens_count=0)
        with self.assertRaises(ValueError):
            quantizer.quantize_kv_cache(tokens_count=1, target_bits=3)
        with self.assertRaises(ValueError):
            AppleANEKVQuantizer(memory_bandwidth_gbps=0)


if __name__ == "__main__":
    unittest.main()
