from .module import Module, Parameter
from .tensor_functions import rand, zeros, tensor

class Linear(Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = Parameter(rand((in_size, out_size)))
        self.bias = Parameter(rand((1, out_size)))

    def forward(self, x):
        return x @ self.weights.value + self.bias.value