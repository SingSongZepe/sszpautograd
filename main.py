
import numpy as np
from tensor import *
from unittest import TestCase, main

class TestTensor(TestCase):
    def test_add(self):
        print('test add')
        
        x = Tensor(3)
        y = Tensor(4)
        z = x + y
        self.assertEqual(z.val, 7)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, 1)

    def test_neg(self):
        print('test neg')
        
        x = Tensor(3)
        y = -x
        self.assertEqual(y.val, -3)
        self.assertEqual(y.grad, 0)
        y.backward()
        self.assertEqual(x.grad, -1)

    def test_sub(self):
        print('test sub')

        x = Tensor(3)
        y = Tensor(4)
        z = x - y
        self.assertEqual(z.val, -1)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 1)
        self.assertEqual(y.grad, -1)

    def test_mul(self):
        print('test mul')

        x = Tensor(3)
        y = Tensor(4)
        z = x * y
        self.assertEqual(z.val, 12)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 4)
        self.assertEqual(y.grad, 3)

    def test_truediv(self):
        print('test div')

        x = Tensor(3)
        y = Tensor(4)
        z = x / y
        self.assertEqual(z.val, 0.75)
        self.assertEqual(z.grad, 0)
        z.backward()
        self.assertEqual(x.grad, 0.25)
        self.assertEqual(y.grad, -3/16)

    def test_pow(self):
        print('test pow')

        x = Tensor(3)
        y = x ** 2
        self.assertEqual(y.val, 9)
        self.assertEqual(y.grad, 0)
        y.backward()
        self.assertEqual(x.grad, 6)

    def test_atan(self):
        print('test atan')

        x = Tensor(0.5)
        y = Tensor.atan(x)
        y.backward()
        self.assertEqual(x.grad, 0.8)

    def test_sin(self):
        print('test sin')

        x = Tensor(math.pi/4)
        y = Tensor.sin(x)
        
        y.backward()
        self.assertEqual(x.grad, math.cos(math.pi/4))


if __name__ == '__main__':
    main()
