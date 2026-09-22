import numpy as np

from network_nn.autograd.function import Function, unbroadcast


class Add(Function):
    def forward(self, a, b):
        self.shapes = (a.shape, b.shape)
        return a + b

    def backward(self, grad):
        return unbroadcast(grad, self.shapes[0]), unbroadcast(grad, self.shapes[1])


class Sub(Function):
    def forward(self, a, b):
        self.shapes = (a.shape, b.shape)
        return a - b

    def backward(self, grad):
        return unbroadcast(grad, self.shapes[0]), unbroadcast(-grad, self.shapes[1])


class Mul(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        return a * b

    def backward(self, grad):
        return unbroadcast(grad * self.b, self.a.shape), unbroadcast(grad * self.a, self.b.shape)


class Div(Function):
    def forward(self, a, b):
        self.a, self.b = a, b
        return a / b

    def backward(self, grad):
        ga = grad / self.b
        gb = -grad * self.a / (self.b * self.b)
        return unbroadcast(ga, self.a.shape), unbroadcast(gb, self.b.shape)


class Pow(Function):
    def forward(self, a, exponent):
        self.a, self.p = a, exponent
        return np.power(a, exponent)

    def backward(self, grad):
        if self.p == 0:
            return np.zeros_like(self.a)
        return grad * self.p * np.power(self.a, self.p - 1)


class Neg(Function):
    def forward(self, a):
        return -a

    def backward(self, grad):
        return -grad


class Exp(Function):
    def forward(self, a):
        self.out = np.exp(a)
        return self.out

    def backward(self, grad):
        return grad * self.out


class Log(Function):
    def forward(self, a):
        self.a = a
        return np.log(a)

    def backward(self, grad):
        return grad / self.a


class Sqrt(Function):
    def forward(self, a):
        self.out = np.sqrt(a)
        return self.out

    def backward(self, grad):
        return grad * 0.5 / self.out


class Abs(Function):
    def forward(self, a):
        self.a = a
        return np.abs(a)

    def backward(self, grad):
        return grad * np.sign(self.a)


class Sin(Function):
    def forward(self, a):
        self.a = a
        return np.sin(a)

    def backward(self, grad):
        return grad * np.cos(self.a)


class Cos(Function):
    def forward(self, a):
        self.a = a
        return np.cos(a)

    def backward(self, grad):
        return -grad * np.sin(self.a)


class Tan(Function):
    def forward(self, a):
        self.out = np.tan(a)
        return self.out

    def backward(self, grad):
        return grad * (1.0 + self.out * self.out)


class Clip(Function):
    def forward(self, a, min_val, max_val):
        self.mask = (a >= min_val) & (a <= max_val)
        return np.clip(a, min_val, max_val)

    def backward(self, grad):
        return grad * self.mask


class MaskedFill(Function):
    def forward(self, a, mask, value):
        self.mask = np.broadcast_to(np.asarray(mask, dtype=bool), a.shape)
        return np.where(self.mask, np.asarray(value, dtype=a.dtype), a)

    def backward(self, grad):
        return np.where(self.mask, 0.0, grad)


class Where(Function):
    def forward(self, condition, a, b):
        self.cond = np.asarray(condition, dtype=bool)
        self.shapes = (a.shape, b.shape)
        return np.where(self.cond, a, b)

    def backward(self, grad):
        return unbroadcast(np.where(self.cond, grad, 0.0), self.shapes[0]), unbroadcast(
            np.where(self.cond, 0.0, grad), self.shapes[1]
        )


class Maximum(Function):
    def forward(self, a, b):
        self.mask = a >= b
        self.shapes = (a.shape, b.shape)
        return np.maximum(a, b)

    def backward(self, grad):
        return unbroadcast(grad * self.mask, self.shapes[0]), unbroadcast(grad * ~self.mask, self.shapes[1])
