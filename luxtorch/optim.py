class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

class SGD(Optimizer):
    def __init__(self, parameters, lr=1.0):
        super().__init__(parameters)
        self.lr = lr
    
    def zero_grad(self):
        for param in self.parameters:
            if param.value.derivative is not None:
                param.value.derivative = None
    
    def step(self):
        for param in self.parameters:
            if param.value.derivative is not None:
                param.update(param.value - self.lr * param.value.derivative)