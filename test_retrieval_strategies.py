from __future__ import annotations

import unittest

import numpy as np

from retrieval import minmax_normalize, stable_ranks


class HybridSensitivityTests(unittest.TestCase):
    def test_minmax_constant_uses_neutral_value(self):
        normalized, constant = minmax_normalize(np.array([2.0, 2.0, 2.0]))
        self.assertTrue(constant)
        np.testing.assert_allclose(normalized, [0.5, 0.5, 0.5])

    def test_stable_ranks_are_one_based_and_reproducible(self):
        memories = [
            {"id": "b", "session_id": "s2", "round_index": 0},
            {"id": "a", "session_id": "s1", "round_index": 0},
            {"id": "c", "session_id": "s3", "round_index": 0},
        ]
        ranks = stable_ranks(np.array([1.0, 1.0, 0.5]), memories)
        np.testing.assert_array_equal(ranks, [2.0, 1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
