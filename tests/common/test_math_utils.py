from unittest import TestCase

from src.common.math_util import *
import numpy as np


class TestMathUtils(TestCase):
    def test_geometric_mean(self):
        result = geometric_mean(4, 8)
        self.assertEqual(5, result)

    def test_disc_sum(self):
        result = calc_disc_sum([1.0, 1.0, 1.0, -1.0, 0], 0.995)
        self.assertTrue(np.array_equal([1.9999501249999998, 1.004975, 0.0050000000000000044, -1.0, 0.0], result))
