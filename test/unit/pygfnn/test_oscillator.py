import unittest
from pygrfnn import oscillator as osc


class Test(unittest.TestCase):
    """Unit tests for pygrfnn.oscillator.Zparam"""

    def test_zparam(self):
        """Test nl(x, gamma) nonlinearity"""
        alpha = 1.0
        beta1 = 1.0
        beta2 = -1.0
        delta1 = -3.0
        delta2 = -5.0
        p = osc.Zparam(alpha, beta1, beta2, delta1, delta2)
        self.assertEqual(p.a, alpha)
        self.assertEqual(p.b1, beta1 + 1j*delta1)
        self.assertEqual(p.b2, beta2 + 1j*delta2)


if __name__ == "__main__":
    unittest.main()