variable_count = 1
# Variable is the main class for autodifferentiation logic for scalars and tensors.
class Variable:
    """
    Attributes:
        history (:class:`History` or None) : the Function calls that created this variable or None if constant
        derivative (variable type): the derivative with respect to this variable
        grad (variable type) : alias for derivative, used for tensors
        name (string) : a globally unique name of the variable
    """
    def __init__(self, history, name=None):
        global variable_count
        assert isinstance(history, History) or history is None, "history must be a History object or None"
        self.history = history
        self._derivative = None

        self.unique_id = "Variable" + str(variable_count)
        variable_count += 1

        if name is not None:
            self.name = name
        else:
            self.name = self.unique_id
        self.used = 0
    
    def requires_grad_(self, requires_grad):

        """
        Set the requires_grad flag to `val` on variable.

        Ensures that operations on this variable will trigger
        backpropagation.

        Args:
            requires_grad (bool): whether to require grad
        """
        if requires_grad:
            self.history = History(self)
        else:
            self.history = None
    
    @property
    def derivative(self):
        """
        Returns the derivative of this variable.
        """
        return self._derivative
    
    def is_leaf(self):
        "True if this variable created by the user (no `last_fn`)"
        return self.history.last_fn is None
    
    def accumulate_derivative(self, derivative):
        """
        Accumulate the derivative of this variable.

        Args:
            derivative (Variable): the derivative to accumulate
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self._derivative is None:
            self._derivative = self.zeros()
        self._derivative += derivative

    def zero_derivative_(self):
        """
        Reset the derivative of this variable.
        """
        self._derivative = self.zeros()

    def zero_grad_(self):
        """
        Reset the grad of this variable.
        """
        self.zero_derivative_()
    
    def expand(self, x):
        """
        Placeholder for tensor variables
        """
        return x
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def zeros(self):
        return 0.0
    
# Some helper functions for handling optional tuples.
def wrap_tuple(x):
    """
    Wraps a single value in a tuple.
    """
    if isinstance(x, tuple):
        return x
    else:
        return (x,)

def unwrap_tuple(x):
    """
    Unwraps a single value from a tuple.
    """
    if isinstance(x, tuple) and len(x) == 1:
        return x[0]
    else:
        return x
    
# Classes for Functions.

class Context: