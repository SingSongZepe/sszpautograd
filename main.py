
import numpy as np
from tensor import *
from unittest import TestCase, main

class TestTensor(TestCase):
    def test_add(self):
        x = Tensor(3)
        y = Tensor(4)
        z = x + y
        self.assertEqual(z.val, 7)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, 1)

    def test_neg(self):
        x = Tensor(3)
        y = -x
        self.assertEqual(y.val, -3)
        self.assertEqual(y.grad, 0)
        y.backward()
        self.assertEqual(x.grad, -1)

    def test_sub(self):
        x = Tensor(3)
        y = Tensor(4)
        z = x - y
        self.assertEqual(z.val, -1)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, -1)

    def test_mul(self):
        x = Tensor(3)
        y = Tensor(4)
        z = x * y
        self.assertEqual(z.val, 12)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 4)
        self.assertEqual(y.grad, 3)

    def test_truediv(self):
        x = Tensor(3)
        y = Tensor(4)
        z = x / y
        self.assertEqual(z.val, 0.75)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 0.25)
        self.assertEqual(y.grad, -3/16)

    def test_pow(self):
        x = Tensor(3)
        y = x ** 2
        self.assertEqual(y.val, 9)
        self.assertEqual(y.grad, 0)
        y.backward()
        self.assertEqual(x.grad, 6)

    def test_atan(self):
        x = Tensor(0.5)
        y = Tensor.atan(x)
        self.assertAlmostEqual(y.val, 0.4636476090008061)
        self.assertEqual(y.grad, 0)
        y.backward()



if __name__ == '__main__':
    main()
