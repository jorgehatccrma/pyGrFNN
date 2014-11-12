import unittest
import numpy as np

from pygrfnn.oscillator import Zparam
from pygrfnn.oscillator import zdot


class Test(unittest.TestCase):
    """Unit tests for pygrfnn.oscillator.Zparam"""

    def test_zparam(self):
        """Test nl(x, gamma) nonlinearity"""
        alpha = 1.0
        beta1 = 1.0
        beta2 = -1.0
        delta1 = -3.0
        delta2 = -5.0
        p = Zparam(alpha, beta1, beta2, delta1, delta2)
        self.assertEqual(p.a, alpha)
        self.assertEqual(p.b1, beta1 + 1j*delta1)
        self.assertEqual(p.b2, beta2 + 1j*delta2)

    def test_zdot(self):
        """
        Test zdot function
        """
        alpha = 1.0
        beta1 = 1.0
        beta2 = -1.0
        delta1 = -3.0
        delta2 = -5.0
        p = Zparam(alpha, beta1, beta2, delta1, delta2)
        x, z, f = 0.1, 0.1, 2.0
        zd = zdot(x, z, f, p)
        expected = 0.21210101010101012+1.2535865563854123j
        self.assertAlmostEqual(zd, expected, "simple zdot result doesn't match")


if __name__ == "__main__":
    unittest.main()
