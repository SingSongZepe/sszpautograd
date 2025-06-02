
from typing import List, Tuple, Callable
import math

import utils.log as log

DEBUG_OUTPUT_OPERATOR_NAME = False

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
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('add operator')        

        def grad_func(grad: float):
            return grad
        
        return Tensor(
            val=self.val + other.val,
            requires_grad=[(self, grad_func), (other, grad_func)]
        )

    def __neg__(self):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('negative operator')

        return Tensor(
            val=-self.val,
            requires_grad=[(self, lambda grad: -grad)]
        )

    def __sub__(self, other: Tensor):
        '''
        y = x + other
        '''
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('substract operator')

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
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('mul operator')

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
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('div operator')

        if other.val == 0:
            raise RuntimeError('the divisor can\'t be 0')

        def grad_func1(grad: float):
            return grad / other.val
        def grad_func2(grad: float):
            return -grad * self.val / (other.val ** 2)
        
        return Tensor(
            val=self.val / other.val,
            requires_grad=[(self, grad_func1), (other, grad_func2)]
        )

    def __pow__(self, n: float):
        '''
        y = x ** n
        '''
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('pow operator')
        
        if (n < 0 and not isinstance(n, int)):
            log.lw(f'n is recommend to be a positive integer, your n: {n}')

        def grad_func(grad: float):
            return grad * n * self.val ** (n-1)
        
        return Tensor(
            val=self.val ** n,
            requires_grad=[(self, grad_func)]
        )
    
    def sqrt(other: float):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('sqrt operator')

        if (other.val < 0):
            log.le(f'out of the domain of sqrt func, you may got a complex number, your val: {other.val}')

        def grad_func(grad: float):
            return grad / 2 / math.sqrt(other.val)
        
        return Tensor(
            val=math.sqrt(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def sin(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('sin operator')

        def grad_func(grad: float):
            return grad * math.cos(other.val)
        
        return Tensor(
            val=math.sin(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def cos(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('cos operator')

        def grad_func(grad: float):
            return grad * -math.sin(other.val)

        return Tensor(
            val=math.cos(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def tan(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('tan operator')
        
        epsilon = 1e-5
        tmp = math.cos(other.val)
        if -epsilon < tmp < epsilon:
            raise RuntimeError(f'the domain of tan(x) is R(x!=pi/2+k*pi), where k is integer, tolerance is {epsilon},'            
             'but your value {other.val} within the tolerance')

        def grad_func(grad: float):
            return grad / math.cos(other.val) / math.sin(other.val)

        return Tensor(
            val=math.tan(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def ctg(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('ctg operator')
        
        epsilon = 1e-5
        tmp = math.sin(other.val)
        if -epsilon < tmp < epsilon:
            raise RuntimeError(f'the domain of ctg(x) is R(x!=k*pi), where k is integer, tolerance is {epsilon},'            
             'but your value {other.val} within the tolerance')
        
        def grad_func(grad: float):
            return -grad / math.sin(other.val) / math.sin(other.val) 

        return Tensor(
            val=1/math.tan(other.val) if math.cos(other.val) != 0 else 0,
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def asin(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('asin operator')
        
        if other.val > 1.0 or other.val < 1.0:
            raise RuntimeError(f'the domain of asin(x) is [-1, 1], but your value {other.val}')

        def grad_func(grad: float):
            return grad / math.sqrt(1 - other.val * other.val)
        
        return Tensor(
            val=math.asin(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def acos(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('acos operator')

        if other.val > 1.0 or other.val < 1.0:
            raise RuntimeError(f'the domain of acos(x) is [-1, 1], but your value {other.val}')

        def grad_func(grad: float):
            return -grad * math.sqrt(1 - other.val * other.val)

        return Tensor(
            val=math.acos(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def atan(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('arctan operator')

        def grad_func(grad: float):
            return grad * 1 / (1 + other.val ** 2) 
            
        return Tensor(
            val=math.atan(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def ln(other: Tensor):
        '''
        Logarithm to base e as commonly used
        '''
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('ln operator')

        def grad_func(grad: float):
            return grad / other.val

        return Tensor(
            val=math.log(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def lg(other: Tensor):
        '''
        Logarithm to base 10 as commonly used
        '''
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('lg operator')

        def grad_func(grad: float):
            return grad / math.log(10) / other.val

        return Tensor(
            val=math.log10(other.val),
            requires_grad=[(other, grad_func)]
        )

    @staticmethod
    def log(other: Tensor, base: float):
        '''
        Logarithm to base 'base'
        '''
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('log operator')

        def grad_func(grad: float):
            return grad / math.log(base) / other.val
        
        return Tensor(
            val=math.log(other.val, base),
            requires_grad=[(other, grad_func)]
        )

    def exp(other: Tensor):
        if DEBUG_OUTPUT_OPERATOR_NAME:
            log.ln('exp operator')
        
        def grad_func(grad: float):
            return grad * math.exp(other.val)

        return Tensor(
            val=math.exp(other.val),
            requires_grad=[(other, grad_func)]
        )

    # other basic operator implementation...
