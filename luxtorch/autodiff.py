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
    """
    Context class is used by `Function` to store information during the forward pass.

    Attributes:
        no_grad (bool) : do not save gradient information
        saved_values (tuple) : tuple of values saved for backward pass
        saved_tensors (tuple) : alias for saved_values
    """
    def __init__(self, no_grad=False):
        self._saved_values = None
        self.no_grad = no_grad

    def save_for_backward(self, *values):
        """
        Store the given `values` if they need to be used during backpropagation.

        Args:
            values (list of values) : values to save for backward
        """
        if self.no_grad:
            return
        self._saved_values = values
    
    @property
    def saved_values(self):
        assert not self.no_grad, "Doesn't require grad"
        assert self._saved_values is not None, "Did you forget to save values?"
        return unwrap_tuple(self._saved_values)
    
    @property
    def saved_tensors(self):
        return self.saved_values
    
class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn (:class:`FunctionBase`) : The last Function that was called.
        ctx (:class:`Context`): The context for that Function.
        inputs (list of inputs) : The inputs that were given when `last_fn.forward` was called.

    """

    def __init__(self, last_fn=None, ctx=None, inputs=None):
        self.last_fn = last_fn
        self.ctx = ctx
        self.inputs = inputs
    
    def backprop_step(self, d_output):
        """
        Run one step of backpropagation by calling chain rule.

        Args:
            d_output : a derivative with respect to this variable

        Returns:
            list of numbers : a derivative with respect to `inputs`
        """
        return self.last_fn.chain_rule(self.ctx, self.inputs, d_output)

class FunctionBase:
    """
    A function that can act on :class:`Variable` arguments to
    produce a :class:`Variable` output, while tracking the internal history.

    Call by :func:`FunctionBase.apply`.

    """

    @staticmethod
    def variable(raw, history):
        # Inplement by children class
        raise NotImplementedError()
    
    @classmethod
    def apply(cls, *vals)