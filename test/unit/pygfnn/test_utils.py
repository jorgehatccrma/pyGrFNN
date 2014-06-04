import unittest
from pygrfnn import utils

class Test(unittest.TestCase):
    """Unit tests for pygrfnn.utils."""

    def test_nl(self):
        """Test nl(x, gamma) nonlinearity"""
        self.assertEqual(utils.nl(0, 0), 1)
        gamma = 0.5
        self.assertEqual(utils.nl(1, gamma), 1/(1-gamma))
        self.assertRaises(ZeroDivisionError, utils.nl, 1, 1)


if __name__ == "__main__":
    unittest.main()