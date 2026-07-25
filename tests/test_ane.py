"""Test suite for Apple ANE KV-Quantizer solution."""
import unittest
from apple_ane_kv_quantizer import AppleANEKVQuantizer

class TestAppleANEKVQuantizer(unittest.TestCase):

    def test_kv_quantization(self):
        quantizer = AppleANEKVQuantizer(memory_bandwidth_gbps=800.0, ane_cores=16)
        metrics, elapsed = quantizer.quantize_kv_cache(tokens_count=8192, hidden_dim=4096, target_bits=4)
        
        self.assertEqual(metrics["status"], "ANE_OPTIMAL")
        self.assertEqual(metrics["bandwidth_saved_percent"], 75.0)

if __name__ == "__main__":
    unittest.main()
