
from typing import List, Tuple, Callable
import math

class Tensor:
    pass

class Tensor:
    def __init__(self, 
                 val: float, 
                 requires_grad: List[Tuple[Tensor, Callable[[float], float]]] = [], 
                 name='none'):
                 
        self.val = val
        self.grad = 0.0
        self.required_grad = requires_grad
        self.name = name

    def backward(self, grad: float = None):
        if grad == None:
            self.grad = 1
            grad = 1
        else:
            self.grad += grad

        for tensor, gfn in self.required_grad:
            tensor.backward(gfn(grad))

    def zero_grad(self):
        self.grad = 0.0
        for tensor, _ in self.required_grad:
            tensor.zero_grad()

    def __add__(self, other: Tensor):
        '''
        y = x + other
        '''
        def grad_func(grad: float):
            return grad
        
        return Tensor(
            val=self.val + other.val,
            requires_grad=[(self, grad_func), (other, grad_func)]
        )

    def __neg__(self):
        return Tensor(
            val=-self.val,
            requires_grad=[(self, lambda grad: -grad)]
        )

    def __sub__(self, other: Tensor):
        '''
        y = x + other
        '''
        def grad_func1(grad: float):
            return grad
        def grad_func2(grad: float):
            return -grad

        return Tensor(
            val=self.val - other.val,
            requires_grad=[(self, grad_func1), (other, grad_func2)]
        )

    def __mul__(self, other: Tensor):
        '''
        y = x * other
        '''
        def grad_func1(grad: float):
            return grad * other.val
        def grad_func2(grad: float):
            return grad * self.val
        
        return Tensor(
            val=self.val * other.val,
            requires_grad=[(self, grad_func1), (other, grad_func2)]
        )

    def __truediv__(self, other: Tensor):
        '''
        y = x / other
        '''
        def grad_func1(grad: float):
            return grad / other.val
        def grad_func2(grad: float):
            return -grad * self.val / (other.val ** 2)
        
        return Tensor(
            val=self.val / other.val,
            requires_grad=[(self, grad_func1), (other, grad_func2)]
        )

    def __pow__(self, n: int):
        '''
        y = x ** n
        '''
        def grad_func(grad: float):
            return grad * n * self.val ** (n-1)
        
        return Tensor(
            val=self.val ** n,
            requires_grad=[(self, grad_func)]
        )

    # def __rmul__(self, other: Tensor):
    #     print('rmul')

    @staticmethod
    def atan(other: Tensor):
        print('atan')

        def grad_func(grad: float):
            return grad * 1 / (1 + other.val ** 2) 
            
        return Tensor(
            val=math.atan(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def sin(other: Tensor):
        print('sin')

        def grad_func(grad: float):
            return grad * math.cos(other.val)
        
        return Tensor(
            val=math.sin(other.val),
            requires_grad=[(other, grad_func)]
        )

    # other basic operator implementation...
