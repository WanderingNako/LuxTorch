from .tensor_functions import tensor, matmul, normal

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    w = tensor(w)
    X = normal(0, 1, (num_examples, len(w)))
    y = matmul(X, w) + b
    y += normal(0, 0.01, y.shape)
    return X, y