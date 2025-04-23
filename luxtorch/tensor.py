from .autodiff import Variable
from .tensor_data import TensorData
from . import operators
"""
Implementation of the core Tensor object for autodifferentiation.
"""
class Tensor(Variable):
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.

    Attributes:

        _tensor (:class:`TensorData`) : the tensor data storage
        backend : backend object used to implement tensor math (see `tensor_functions.py`)
    """
    def __init__(self, value, history=None, name=None, backend=None):
        assert isinstance(value, TensorData), "data must be a TensorData object"
        assert backend is not None, "backend must be provided"
        super().__init__(history, name=name)
        self._tensor = value
        self.backend = backend

    def to_numpy(self):
        """
        Returns:
             narray : converted to numpy array
        """
        return self.contiguous()._tensor._storage.reshape(self.shape)
    
    @property
    def shape(self):
        return self._tensor.shape
    
    @property
    def size(self):
        return self._tensor.size
    
    @property
    def dims(self):
        return self._tensor.dims
    
    def _ensure_tensor(self, b):
        "Turns a python number into a tensor with the same backend."
        if isinstance(b, (int, float)):
            b = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
        return b
    
    # Functions
    def __len__(self):
        return self._tensor.shape[0]
    
    def __add__(self, b):
        return self.backend.Add.apply(self, self._ensure_tensor(b))
    
    def __sub__(self, b):
        return self.backend.Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b):
        return self.backend.Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b):
        return self.backend.Mul.apply(
            self, self.backend.Inv.apply(self._ensure_tensor(b))
        )

    def __rtruediv__(self, b):
        return self.backend.Mul.apply(
            self._ensure_tensor(b), self.backend.Inv.apply(self)
        )

    def __matmul__(self, b):
        return self.backend.MatMul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b):
        return self.backend.LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b):
        return self.backend.EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b):
        return self.backend.LT.apply(self._ensure_tensor(b), self)

    def __neg__(self):
        return self.backend.Neg.apply(self)
    
    def all(self, dim=None):
        return self.backend.All.apply(self, dim)

    def is_close(self, y):
        return self.backend.IsClose.apply(self, y)

    def sigmoid(self):
        return self.backend.Sigmoid.apply(self)

    def relu(self):
        return self.backend.ReLU.apply(self)
    
    def sqrt(self):
        return self.backend.Sqrt.apply(self)

    def log(self):
        return self.backend.Log.apply(self)
    
    def item(self):
        assert self.size == 1
        it = self.view(1)
        return it[0]
    
    def exp(self):
        return self.backend.Exp.apply(self)
    
    def sum(self, dim=None):
        return self.backend.Sum.apply(self, dim)
    
    def max(self, dim=None):
        return self.backend.Max.apply(self, dim)
    
    def mean(self, dim=None):
        "Compute the mean over dimension `dim`"
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size
    
    def std(self, dim=None):
        "Compute the standard deviation over dimension `dim`"
        if dim is not None:
            return self.backend.Std.apply(self, dim)
        else:
            return self.backend.Std.apply(self)
        
    def permute(self, *order):
        "Permute tensor dimensions to *order"
        return self.backend.Permute.apply(self, order)

    def view(self, *shape):
        "Change the shape of the tensor to a new shape with the same size"
        return self.backend.View.apply(self, shape)
    
    def contiguous(self):
        "Return a contiguous tensor with the same data"
        return self.backend.Copy.apply(self)
    
    def __repr__(self):
        return self._tensor.to_string()
    
    def __getitem__(self, key):
        return self._tensor.get(key)
    
    def __setitem__(self, key, val):
        self._tensor.set(key, val)
    
    @property
    def grad(self):
        return self.derivative
    
    def _type_(self, backend):
        self.backend = backend
    
    def _new(self, tensor_data):
        return Tensor(tensor_data, backend=self.backend)
    
    @staticmethod
    def make(storage, shape, strides=None, backend=None):
        "Create a new tensor from data"
        return Tensor(TensorData(storage, shape, strides), backend=backend)
    
    def expand(self, other):
        """
        Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Parameters:
            other (class:`Tensor`): backward tensor (must broadcast with self)

        Returns:
            Expanded version of `other` with the right derivatives

        """

        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other
        
        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend._id_map(other, out=buf)
        if self.shape == true_shape:
            return buf
        
        # Case 3: Still different, reduce extra dims.
        out = buf
        ori_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if ori_shape[dim] == 1 and shape != 1:
                out = self.backend._add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        final = Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        return final
    
    def zeros(self, shape=None):
        def zero(shape):
            return Tensor.make(
                [0] * int(operators.prod(shape)), shape, backend=self.backend
            )
        
        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out
    
    def tuple(self):
        return self._tensor.tuple()
    
    def get_data(self):
        return Tensor(self._tensor, backend=self.backend)
    
    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = Tensor.make(self.size * [1.0], self.shape, backend=self.backend)
        super().backward(grad_output)
        