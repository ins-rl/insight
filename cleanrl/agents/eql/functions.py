"""Functions for use with symbolic regression.

These functions encapsulate multiple implementations (sympy, Tensorflow, numpy) of a particular function so that the
functions can be used in multiple contexts."""

import torch
# import tensorflow as tf
import numpy as np
import sympy as sp

EPS = 1e-8

class BaseFunction:
    """Abstract class for primitive functions"""
    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """Sympy implementation"""
        return None

    def torch(self, x):
        """No need for base function"""
        return None

    def tf(self, x):
        """Automatically convert sympy to TensorFlow"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'tensorflow')(x)

    def np(self, x):
        """Automatically convert sympy to numpy"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)

    def name(self, x):
        return str(self.sp)


class Constant(BaseFunction):
    def torch(self, x):
        return torch.ones_like(x)

    def tf(self, x):
        return tf.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):
    def torch(self, x):
        return x / self.norm  # ??

    def tf(self, x):
        return tf.identity(x) / self.norm

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):
    def torch(self, x):
        return torch.where(torch.abs(x) < 1e3, torch.square(x), 0.0) / self.norm
        # return torch.clamp(torch.square(x) / self.norm,min=0,max=1e6)

    def tf(self, x):
        return tf.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm

class Sqrt(BaseFunction):
    def torch(self, x):
        # x = torch.clip(x,-20,20)
        # return torch.where(torch.abs(x) >1e-8, torch.sqrt(torch.abs(x)) / self.norm, 0.0) / self.norm
        return torch.sqrt(torch.abs(x)+1e-8)

    def tf(self, x):
        return tf.sqrt(x) / self.norm

    def sp(self, x):
        return sp.sqrt(sp.Abs(x))  / self.norm

    def np(self, x):
        return np.sqrt(np.abs(x)) / self.norm

class Pow(BaseFunction):
    def __init__(self, power, norm=1):
        BaseFunction.__init__(self, norm=norm)
        self.power = power

    def torch(self, x):
        return torch.pow(x, self.power) / self.norm

    def sp(self, x):
        return x ** self.power / self.norm

    def tf(self, x):
        return tf.pow(x, self.power) / self.norm


class Sin(BaseFunction):
    def torch(self, x):
        return torch.sin(x * 2 * 2 * np.pi) / self.norm

    def sp(self, x):
        return sp.sin(x * 2*2*np.pi) / self.norm


class Sigmoid(BaseFunction):
    def torch(self, x):
        return torch.sigmoid(x) / self.norm

    # def tf(self, x):
    #     return tf.sigmoid(x) / self.norm

    def sp(self, x):
        return 1 / (1 + sp.exp(-20*x)) / self.norm

    def np(self, x):
        return 1 / (1 + np.exp(-20*x)) / self.norm

    def name(self, x):
        return "sigmoid(x)"


class Exp(BaseFunction):
    def __init__(self, norm=1):
        super().__init__(norm)

    # ?? why the minus 1
    def torch(self, x):
        x = torch.where(x < 100, torch.exp(x), torch.tensor(0.0))
        return (x ) / self.norm

    def sp(self, x):
        return (sp.exp(x)) / self.norm


class Log(BaseFunction):
    def torch(self, x):
        x_safe = torch.abs(x) + 1e-8  # 使用绝对值并添加小的正常数
        return torch.log(x_safe) / self.norm
        # return torch.where(torch.abs(x) > 0.001, torch.log(torch.abs(x)), torch.tensor(0.0)) / self.norm

    def sp(self, x):
        # 使用绝对值并添加小的正常数，以确保 log 函数的输入总是正的
        x_safe = sp.Abs(x) + 1e-8
        # 使用 SymPy 的符号表达式进行对数运算
        return sp.log(x_safe) / self.norm
    
class Invx(BaseFunction):
    def __init__(self, norm=1):
        super().__init__(norm=norm)

    def torch(self, x):
        return 1/(x+torch.tensor(1e-4))
        # return torch.divide(x, y+torch.tensor(1e-2)) / self.norm

    def sp(self, x):
        # if sp.Abs(y)>1e-4:
        #     return x/y
        # else:
        #     return 0
        return 1/(x+1e-4) / self.norm

class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs"""
    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        """Sympy implementation"""
        return None

    def torch(self, x, y):
        return None

    def tf(self, x, y):
        """Automatically convert sympy to TensorFlow"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'tensorflow')(x, y)

    def np(self, x, y):
        """Automatically convert sympy to numpy"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)

    def name(self, x, y):
        return str(self.sp)


class Product(BaseFunction2):
    def __init__(self, norm=1):
        super().__init__(norm=norm)

    def torch(self, x, y):
        # return x * y / self.norm
        return torch.clip(x * y / self.norm,-10000,10000)

    def sp(self, x, y):
        return x*y / self.norm

class Add(BaseFunction2):
    def __init__(self, norm=1):
        super().__init__(norm=norm)

    def torch(self, x, y):
        # return x * y / self.norm
        return (x + y) / self.norm

    def sp(self, x, y):
        return (x + y)  / self.norm


class Div(BaseFunction2):
    def __init__(self, norm=1):
        super().__init__(norm=norm)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1e-8  # 小的正常数，避免除以零

    def torch(self, x, y):
        # const1 = torch.tensor(1.0)
        # const1 = const1.to(self.device)
        # return torch.where(torch.abs(y) > 0.001, torch.divide(x, y), const1) / self.norm
        #     # 在y接近零时增加稳定性，而不改变其符号
        y_denom = y + self.epsilon * torch.sign(y)
        y_denom = torch.where(y_denom == 0, torch.tensor(self.epsilon, device=y.device), y_denom)
        return torch.div(x, y_denom) / self.norm
        # return torch.divide(x, y+torch.tensor(1e-2)) / self.norm

    def sp(self, x, y):
        # 为避免除零，给 y 增加一个小的正数 epsilon
        y_denom = y + self.epsilon * sp.Symbol('sign(y)', real=True)
        # 使用 SymPy 的符号表达式进行除法
        return x / y_denom / self.norm
    
    
def count_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i


def count_double(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i


# default_func = [
#     Constant(),
#     Constant(),
#     Identity(),
#     Identity(),
#     Square(),
#     Square(),
#     Sin(),
#     Sigmoid(),
# ]

default_func = [
    *[Constant()] * 2,
    *[Identity()] * 4,
    *[Square()] * 4,
    *[Sin()] * 2,
    *[Exp()] * 2,
    *[Sigmoid()] * 2,
    *[Product()] * 2,
    # *[Div()] * 2,
]
