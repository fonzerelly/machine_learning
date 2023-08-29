import unittest
import numpy as np
from fractions import Fraction

from plot import gradient

class TestPlotMethods(unittest.TestCase):
    def test_gradient_w1(self):
        X = np.array([1,2,3])
        Y = np.array([1,2,3])
        w = 1
        b = 0
        self.assertEqual(gradient(X,Y,w,b)[0], 0)

    def test_gradient_w2(self):
        X = np.array([1,2,3])
        Y = np.array([1,2,3])
        w = 2
        b = 0

        # X * w
        # [1*2, 2*2, 3*2] = [2,4,6]

        # X * w - Y
        # [2-1, 4-2, 6-3] = [1,2,3]

        # X * (X * w - Y)
        # [1*1, 2*2, 3*3] = [1,4,9]

        # Durchschnitt
        # (1+4+9) /3 = 14/3

        # mal 2
        # 2 * 14/3 = 28/3

        self.assertEqual(gradient(X,Y,w,b)[0], float(Fraction (28, 3)))

    def test_gradient_b1(self):
        X = np.array([1,2,3])
        Y = np.array([1,2,3])
        w = 0
        b = 1
        self.assertEqual(gradient(X,Y,w,b)[1], -2)

    def test_gradient_b2(self):
        X = np.array([1,2,3])
        Y = np.array([1,2,3])
        w = 0
        b = 2
        self.assertEqual(gradient(X,Y,w,b)[1], 0)

if __name__ == '__main__':
    unittest.main()