
import numpy as np
from tensor import *
from unittest import TestCase, main

class TestTensor(TestCase):
    # def test_add(self):
    #     print('test add')
        
    #     x = Tensor(3)
    #     y = Tensor(4)
    #     z = x + y
    #     self.assertEqual(z.val, 7)
    #     self.assertEqual(z.grad, 0)
    #     z.backward()
    #     self.assertEqual(x.grad, 1)
    #     self.assertEqual(y.grad, 1)

    # def test_neg(self):
    #     print('test neg')
        
    #     x = Tensor(3)
    #     y = -x
    #     self.assertEqual(y.val, -3)
    #     self.assertEqual(y.grad, 0)
    #     y.backward()
    #     self.assertEqual(x.grad, -1)

    # def test_sub(self):
    #     print('test sub')

    #     x = Tensor(3)
    #     y = Tensor(4)
    #     z = x - y
    #     self.assertEqual(z.val, -1)
    #     self.assertEqual(z.grad, 0)
    #     z.backward()
    #     self.assertEqual(x.grad, 1)
    #     self.assertEqual(y.grad, -1)

    # def test_mul(self):
    #     print('test mul')

    #     x = Tensor(3)
    #     y = Tensor(4)
    #     z = x * y
    #     self.assertEqual(z.val, 12)
    #     self.assertEqual(z.grad, 0)
    #     z.backward()
    #     self.assertEqual(x.grad, 4)
    #     self.assertEqual(y.grad, 3)

    # def test_truediv(self):
    #     print('test div')

    #     x = Tensor(3)
    #     y = Tensor(4)
    #     z = x / y
    #     self.assertEqual(z.val, 0.75)
    #     self.assertEqual(z.grad, 0)
    #     z.backward()
    #     self.assertEqual(x.grad, 0.25)
    #     self.assertEqual(y.grad, -3/16)

    # def test_pow(self):
    #     print('test pow')

    #     x = Tensor(3)
    #     y = x ** 2
    #     self.assertEqual(y.val, 9)
    #     self.assertEqual(y.grad, 0)
    #     y.backward()
    #     self.assertEqual(x.grad, 6)

    # def test_atan(self):
    #     print('test atan')

    #     x = Tensor(0.5)
    #     y = Tensor.atan(x)
    #     y.backward()
    #     self.assertEqual(x.grad, 0.8)

    # def test_sin(self):
    #     print('test ')

    #     x = Tensor(math.pi/4)
    #     y = Tensor.sin(x)
        
    #     y.backward()
    #     self.assertEqual(x.grad, math.cos(math.pi/4))

    # def test_cos(self):
    #     x = Tensor(math.pi*3/4)
    #     y = Tensor.cos(x)

    #     y.backward()

    #     self.assertEqual(x.grad, -math.sin(math.pi*3/4))

    def test_pow2(self):
        x = Tensor(2)
        y = x ** 0.5
        y.backward()
        log.ln(x.grad)
    
    def test_ln(self):
        x = Tensor(10)
        y = Tensor.ln(x)
        y.backward()
        log.ln(x.grad)

    def test_tan(self):
        x = Tensor(math.pi / 4)
        y = Tensor.tan(x)

        y.backward()

        log.ln(x.grad)
        log.ln(1/math.cos(math.pi/4)**2)

    def test_ctg(self):
        x_val = math.pi/6
        x = Tensor(x_val)
        y = Tensor.ctg(x)

        y.backward()

        log.ln(x.grad)
        log.ln(-1/math.sin(x_val)**2)
        



if __name__ == '__main__':
    main()
