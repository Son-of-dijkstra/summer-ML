import math
class Value:
    
    def __init__(self, data, _children = ()):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other,))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other, ))
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float power"
        out = Value(self.data ** other, (self, ))
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self, ))
        def _backward():
            self.grad += (1.0 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0.0 if self.data < 0.0 else self.data, (self, ))
        def _backward():
            self.grad += (out.data > 0.0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Value(1.0 / (1.0 + math.exp(-self.data)), (self, ))
        def _backward():
            self.grad += out.data * (1.0 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        used = set()
        order = []

        def topsort(v):
            used.add(v)
            for u in v._children:
                if u not in used:
                    topsort(u)
            order.append(v)

        topsort(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __repr__(self):
        return f'Value(data = {self.data})'