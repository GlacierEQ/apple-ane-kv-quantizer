"""Test suite for Go ANE weight quantizer."""

import unittest


class ANEQuantizerSim:
    def quantize_weights(self, weights: list) -> dict:
        min_val, max_val = min(weights), max(weights)
        scale = (max_val - min_val) / 255.0
        return {"scale": scale, "elements": len(weights)}


class TestANEQuantizer(unittest.TestCase):
    def test_quantizer_scale(self):
        q = ANEQuantizerSim()
        res = q.quantize_weights([0.12, -0.85, 0.44, 1.25, -1.10])
        self.assertEqual(res["elements"], 5)
        self.assertGreater(res["scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
