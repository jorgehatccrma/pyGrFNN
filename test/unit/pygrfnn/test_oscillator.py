import unittest
import numpy as np

from pygrfnn.oscillator import Zparam
from pygrfnn.oscillator import zdot


class Test(unittest.TestCase):
    """Unit tests for pygrfnn.oscillator.Zparam"""

    def test_zparam(self):
        """Test canonical oscillator params"""
        alpha = 1.0
        beta1 = 1.0
        beta2 = -1.0
        delta1 = -3.0
        delta2 = -5.0
        epsilon = 1.0
        zp = Zparam(alpha, beta1, beta2, delta1, delta2, epsilon)
        self.assertEqual(zp.a, zp.alpha + 1j * 2 * np.pi)
        self.assertEqual(zp.b1, zp.beta1 + 1j * zp.delta1)
        self.assertEqual(zp.b2, zp.beta2 + 1j * zp.delta2)

    def test_zdot(self):
        """
        Test zdot function
        """
        alpha = 1.0
        beta1 = 1.0
        beta2 = -1.0
        delta1 = -3.0
        delta2 = -5.0
        epsilon = 1.0
        p = Zparam(alpha, beta1, beta2, delta1, delta2, epsilon)
        x, z, f = 0.1, 0.1, 2.0
        zd = zdot(x, z, f, p)
        # TODO: verify this values
        # expected = 0.21210101010101012+1.2535865563854123j
        expected = 0.301979797979798+1.2505360513349073j
        self.assertAlmostEqual(zd, expected, msg="simple zdot result doesn't match")


if __name__ == "__main__":
    unittest.main()
