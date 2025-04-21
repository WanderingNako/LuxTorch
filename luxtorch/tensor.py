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
    
    def contiguous(self):
        "Return a contiguous tensor with the same data"
        return self.backend.Copy.apply(self)
    
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