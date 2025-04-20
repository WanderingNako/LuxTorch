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
        if not requires_grad:
            self.history = None
    
    def backward(self, d_output=None):
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)
    
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
        inputs (list of inputs)(list of Variables or constants) : The inputs that were given when `last_fn.forward` was called.

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
    def apply(cls, *vals):
        """
        Apply is called by the user to run the Function.
        Internally it does three things:

        a) Creates a Context for the function call.
        b) Calls forward to run the function.
        c) Attaches the Context to the History of the new variable.

        There is a bit of internal complexity in our implementation
        to handle both scalars and tensors.

        Args:
            vals (list of Variables or constants) : The arguments to forward

        Returns:
            `Variable` : The new variable produced

        """
        # Go through the variables to see if any needs grad.
        raw_vals = []
        need_grad = False
        for val in vals:
            if isinstance(val, Variable):
                if val.history is not None:
                    need_grad = True
                val.used += 1
                raw_vals.append(val.get_data())
            # constant
            else:
                raw_vals.append(val)
        
        # Create a context for the function call.
        ctx = Context(not need_grad)
        
        # Call forward with the variables.
        c = cls.forward(ctx, *raw_vals)
        assert isinstance(c, cls.data_type), "Expected return typ %s got %s" % (
            cls.data_type,
            type(c),
        )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = History(cls, ctx, vals)
        return cls.variable(cls.data(c), back)
    
    @classmethod
    def chain_rule(cls, ctx, inputs, d_output):
        """
        Chain rule is called by the `History` object to run the backward pass.

        Args:
            ctx (:class:`Context`) : The context from running forward
            inputs (list of args) : The args that were passed to :func:`FunctionBase.apply` (e.g. :math:`x, y`)
            d_output (number) : The `d_output` value in the chain rule.

        Returns:
            list of (`Variable`, number) : A list of non-constant variables with their derivatives
            (see `is_constant` to remove unneeded variables)
        """
        # Call backward with the context and d_output.
        derivatives =  cls.backward(ctx, d_output)
        result = []
        i = 0
        if isinstance(derivatives, tuple):
            for val in inputs:
                if not is_constant(val):
                    result.append((val, derivatives[i]))
                    i = i + 1
        else:
            for val in inputs:
                if not is_constant(val):
                    result.append((val, derivatives))
                i = i + 1
        return result
    
def is_constant(val):
    return not isinstance(val, Variable) or val.history is None

def topological_sort(variable):
    """
    Computes the topological order of the computation graph.

    Args:
        variable (:class:`Variable`): The right-most variable

    Returns:
        list of Variables : Non-constant Variables in topological order
                            starting from the right.
    """
    PermanentMarked = []
    TemporaryMarked = []
    result = []

    def visit(n):
        # Don't do anything with constant.
        if is_constant(n):
            return
        if n.unique_id in PermanentMarked:
            return
        elif n.unique_id in TemporaryMarked:
            raise RuntimeError("Graph is not a DAG")
        TemporaryMarked.append(n.unique_id)

        if n.is_leaf():
            pass
        else:
            for i in n.history.inputs:
                visit(i)
        TemporaryMarked.remove(n.unique_id)
        PermanentMarked.append(n.unique_id)
        result.insert(0, n)
    visit(variable)
    return result

def backpropagate(variable, deriv):
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    See :doc:`backpropagate` for details on the algorithm.

    Args:
        variable (:class:`Variable`): The right-most variable
        deriv (number) : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Get the topological order of the graph.
    order = topological_sort(variable)
    
    derivs = {variable.unique_id: deriv}

    for node in order:
        d_output = derivs[node.unique_id]
        if node.is_leaf():
            node.accumulate_derivative(d_output)
        else:
            for variable, derivative in node.history.backprop_step(d_output):
                if variable.unique_id not in derivs:
                    derivs[variable.unique_id] = 0.0
                derivs[variable.unique_id] += derivative