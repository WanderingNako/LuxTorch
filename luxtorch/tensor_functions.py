"""
Implementation of the autodifferentiation Functions for Tensor.
"""
from .autodiff import FunctionBase
from .tensor_ops import TensorOps
import numpy as np
from . import operators
from .tensor import Tensor
import random

# Constructors
class Function(FunctionBase):
    data_type = Tensor

    @staticmethod
    def data(a):
        return (a._tensor, a.backend)
    
    @staticmethod
    def variable(raw, history):
        return Tensor(raw[0], history=history, backend=raw[1])
    
def make_tensor_backend(tensor_ops, is_cuda=False):
    """
    Dynamically construct a tensor backend based on a `tensor_ops` object
    that implements map, zip, and reduce higher-order functions.

    Args:
        tensor_ops (:class:`TensorOps`) : tensor operations object see `tensor_ops.py`
        is_cuda (bool) : is the operations object CUDA / GPU based

    Returns :
        backend : a collection of tensor functions

    """
    # Maps
    neg_map = tensor_ops.map(operators.neg)
    sigmoid_map = tensor_ops.map(operators.sigmoid)
    relu_map = tensor_ops.map(operators.relu)
    log_map = tensor_ops.map(operators.log)
    exp_map = tensor_ops.map(operators.exp)
    id_map = tensor_ops.map(operators.id)
    inv_map = tensor_ops.map(operators.inv)

    # Zips
    add_zip = tensor_ops.zip(operators.add)
    mul_zip = tensor_ops.zip(operators.mul)
    lt_zip = tensor_ops.zip(operators.lt)
    eq_zip = tensor_ops.zip(operators.eq)
    is_close_zip = tensor_ops.zip(operators.is_close)
    relu_back_zip = tensor_ops.zip(operators.relu_back)
    log_back_zip = tensor_ops.zip(operators.log_back)
    inv_back_zip = tensor_ops.zip(operators.inv_back)

    # Reduce
    add_reduce = tensor_ops.reduce(operators.add, 0.0)
    mul_reduce = tensor_ops.reduce(operators.mul, 1.0)
    max_reduce = tensor_ops.reduce(operators.max, -1e9)

    class Backend:
        cuda = is_cuda
        _id_map = id_map
        _add_reduce = add_reduce

        class Neg(Function):
            @staticmethod
            def forward(ctx, t1):
                return neg_map(t1)
            
            @staticmethod
            def backward(ctx, grad_output):
                # grad_output is an instance of Tensor
                return neg_map(grad_output)
            
        class Inv(Function):
            @staticmethod
            def forward(ctx, t1):
                ctx.save_for_backward(t1)
                return inv_map(t1)
            
            @staticmethod
            def backward(ctx, grad_output):
                t1 = ctx.saved_values
                return inv_back_zip(t1, grad_output)
        
        class Add(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1, t2)
                return add_zip(t1, t2)
            
            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_values
                grad_a, grad_b = a.expand(grad_output), b.expand(grad_output)
                return grad_a, grad_b
            
        class Mul(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return mul_zip(a, b)
            
            @staticmethod
            def backward(ctx, grad_output):
                a, b = ctx.saved_values
                grad_a, grad_b = a.expand(b * grad_output), b.expand(a * grad_output)
                return grad_a, grad_b
            
        class Sigmoid(Function):
            @staticmethod
            def forward(ctx, a):
                b = sigmoid_map(a)
                ctx.save_for_backward(b)
                return b

            @staticmethod
            def backward(ctx, grad_output):
                b = ctx.saved_values
                return mul_zip(grad_output, mul_zip(b, add_zip(tensor([1.0]), neg_map(b))))
        
        class ReLU(Function):
            @staticmethod
            def forward(ctx, a):
                return relu_map(a)
            
            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return relu_back_zip(a, grad_output)
        
        class Log(Function):
            @staticmethod
            def forward(ctx, a):
                ctx.save_for_backward(a)
                return log_map(a)

            @staticmethod
            def backward(ctx, grad_output):
                a = ctx.saved_values
                return log_back_zip(a, grad_output)
        
        class Exp(Function):
            @staticmethod
            def forward(ctx, a):
                b = exp_map(a)
                ctx.save_for_backward(b)
                return b

            @staticmethod
            def backward(ctx, grad_output):
                b = ctx.saved_values
                return mul_zip(b, grad_output)
            
        class Sum(Function):
            @staticmethod
            def forward(ctx, a, dim):
                ctx.save_for_backward(a.shape, dim)
                if dim is not None:
                    return add_reduce(a, dim)
                else:
                    # Flatten
                    return add_reduce(
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )
            
            @staticmethod
            def backward(ctx, grad_output):
                a_shape, dim = ctx.saved_values
                if dim is None:
                    out = grad_output.zeros(a_shape)
                    out._tensor._storage[:] = grad_output[0]
                    return out
                else:
                    return grad_output
            
        class All(Function):
            @staticmethod
            def forward(ctx, a, dim):
                if dim is not None:
                    return mul_reduce(a, dim)
                else:
                    return mul_reduce(
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )
        
        class LT(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a.shape, b.shape)
                return lt_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a_shape, b_shape = ctx.saved_values
                return grad_output.zeros(a_shape), grad_output.zeros(b_shape)
            
        class EQ(Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a.shape, b.shape)
                return eq_zip(a, b)

            @staticmethod
            def backward(ctx, grad_output):
                a_shape, b_shape = ctx.saved_values
                return grad_output.zeros(a_shape), grad_output.zeros(b_shape)

        class IsClose(Function):
            @staticmethod
            def forward(ctx, a, b):
                return is_close_zip(a,b)
        
        class Permute(Function):
            @staticmethod
            def forward(ctx, a, order):
                ctx.save_for_backward(order)
                return Tensor(a._tensor.permute(*order), backend=a.backend)
            
            @staticmethod
            def backward(ctx, grad_output):
                order = ctx.saved_values
                un_permute = [0 for _ in range(len(order))]
                for d1, d2 in enumerate(order):
                    un_permute[d2] = d1
                return Tensor(grad_output._tensor.permute(*un_permute), backend=grad_output.backend)
            
        class View(Function):
            @staticmethod
            def forward(ctx, a, shape):
                ctx.save_for_backward(a.shape)
                assert a._tensor.is_contiguous(), "Must be contiguous to view"
                return Tensor.make(a._tensor._storage, shape, backend=a.backend)
            
            @staticmethod
            def backward(ctx, grad_output):
                original = ctx.saved_values
                return Tensor.make(
                    grad_output._tensor._storage, original, backend=grad_output.backend
                )
            
        class Copy(Function):
            @staticmethod
            def forward(ctx, a):
                return id_map(a)
            
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output
            
        class MatMul(Function):
            @staticmethod
            def forward(ctx, t1, t2):
                ctx.save_for_backward(t1, t2)
                return tensor_ops.matrix_multiply(t1, t2)
            
            @staticmethod
            def backward(ctx, grad_output):
                t1, t2 = ctx.saved_values

                def transpose(a):
                    order = list(range(a.dims))
                    order[-2], order[-1] = order[-1], order[-2]
                    return a._new(a._tensor.permute(*order))
                
                return (
                    tensor_ops.matrix_multiply(grad_output, transpose(t2)),
                    tensor_ops.matrix_multiply(transpose(t1), grad_output)
                )
        
        class Max(Function):
            @staticmethod
            def forward(ctx, a, dim):
                ctx.save_for_backward(a, dim)
                if dim is not None:
                    return max_reduce(a, dim)
                else:
                    return max_reduce(
                        a.contiguous().view(int(operators.prod(a.shape))), 0
                    )
            
            @staticmethod
            def backward(ctx, grad_output):
                a, dim = ctx.saved_values
                if dim is None:
                    out = grad_output.zeros(a.shape)
                    out._tensor._storage[:] = grad_output[0] * argmax(a)
                return grad_output * argmax(a, dim)

        
    return Backend
            
TensorFunctions = make_tensor_backend(TensorOps)
# Helpers for Constructing tensors
def zeros(shape, backend=TensorFunctions):
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend

    Returns:
        :class:`Tensor` : new tensor
    """
    return Tensor.make([0] * int(operators.prod(shape)), shape, backend=backend)

def rand(shape, backend=TensorFunctions, requires_grad=False):
    """
    Produce a random tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def normal(mean, std, shape, backend=TensorFunctions, requires_grad=False):
    """
    Produce a random tensor of size `shape`.

    Args:
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.normalvariate(mu=mean, sigma=std) for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def _tensor(ls, shape=None, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls (list): data for tensor
        shape (tuple): shape of tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    tensor = Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def tensor(ls, backend=TensorFunctions, requires_grad=False):
    """
    Produce a tensor with data and shape from ls

    Args:
        ls (list): data for tensor
        backend (:class:`Backend`): tensor backend
        requires_grad (bool): turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def gen_shape(ls):
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + gen_shape(ls[0])
        else:
            return []

    def flatten(ls):
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape = gen_shape(ls)
    return _tensor(cur, tuple(shape), backend=backend, requires_grad=requires_grad)

def matmul(t1, t2):
    """
    Matrix multiply two tensors.

    Args:
        t1 (:class:`Tensor`): first tensor
        t2 (:class:`Tensor`): second tensor

    Returns:
        :class:`Tensor` : new tensor
    """
    return TensorFunctions.MatMul.apply(t1, t2)

def argmax(t1, dim=None):
    """
    Returns the index of the maximum value along a given dimension.

    Args:
        t1 (:class:`Tensor`): input tensor
        dim (int): dimension to reduce

    Returns:
        :class:`Tensor` : new tensor with the index of the maximum value along the given dimension
    """
    if dim is None:
        flatten = t1.contiguous().view(int(operators.prod(t1.shape)))
        out = TensorOps.reduce(operators.max, -1e9)(flatten, 0)
    else:
        out = TensorOps.reduce(operators.max, -1e9)(t1, dim)
    return out == t1