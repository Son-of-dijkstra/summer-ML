import random
from micrograd.autograd_engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []
    
class Neuron(Module):

    def __init__(self, nin, act = None):
        self._nin = nin
        self.ws = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
        self._act = act
    
    def __call__(self, x):
        val = sum((wi * xi for wi, xi in zip(self.ws, x)), self.bias)
        if self._act is None:
            return val
        if self._act == 'tanh':
            return val.tanh()
        if self._act == 'relu':
            return val.relu()
        if self._act == 'sigmoid':
            return val.sigmoid()
    
    def parameters(self):
        return self.ws + [self.bias]
    
    def __repr__(self):
        return f'Neuron(nin = {self._nin}, act = {self._act})'

class Layer(Module):

    def __init__(self, nin, nout, act = None):
        self._nin = nin
        self._nout = nout
        self._act = act
        self.neurons = [Neuron(nin, act) for _ in range(nout)]
    
    def __call__(self, x):
        assert len(x) == self._nin, f'size of x must be {self._nin}'
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f'Layer(nin = {self._nin}, nout = {self._nout}, act = {self._act})'

class MLP(Module):

    def __init__(self, nin, nouts, out_acts = None):
        sizes = [nin] + list(nouts)
        self._nin = nin
        self.layers = [Layer(sizes[i], sizes[i+1], None if out_acts is None else out_acts[i]) for i in range(len(sizes)-1)]
    
    def __call__(self, x):
        assert len(x) == self._nin, f'size of x must be {self._nin}'
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f'MLP(\n   ' +f',\n   '.join(str(layer) for layer in self.layers)+f'\n)'