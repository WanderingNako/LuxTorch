from .module import Module, Parameter
from .tensor_functions import rand, zeros, tensor, normal, ones, dropout

class Linear(Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = Parameter(normal(0, 1, (in_size, out_size)))
        self.bias = Parameter(normal(0, 1, (1, out_size)))

    def forward(self, x):
        return x @ self.weights.value + self.bias.value
    
class Dropout(Module):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        return dropout(x, self.rate, self.training)
    
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embeddings = Parameter(normal(0, 1, (num_embeddings, embedding_dim)))

    def forward(self, x):
        return self.embeddings.value[x]
    
class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
    
class PositionWiseFFN(Module):
    def __init__(self, ffn_num_input, ffn_num_hidden, ffn_num_output):
        super().__init__()
        self.dense1 = Linear(ffn_num_input, ffn_num_hidden)
        self.relu = ReLU()
        self.dense2 = Linear(ffn_num_hidden, ffn_num_output)

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))
    
class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Parameter(ones(normalized_shape))
        self.beta = Parameter(zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1)
        std  = x.std(dim=-1)
        x_hat = (x -mean) / (std + self.eps).sqrt()
        return self.gamma.value * x_hat + self.beta.value
