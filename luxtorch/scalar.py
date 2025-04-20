from .autodiff import FunctionBase, Variable, History
from . import operators

class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.
    """
    def __init__(self, value, back=History(), name=None):
        super().__init__(back, name)
        self.data = float(value)
    
    def __repr__(self):
        return "Scalar(%s)" % self.data
    
    def __mul__(self, b):
        return Mul.apply(self, b)
    
    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b):
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b):
        return Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b):
        return LT.apply(self, b)

    def __gt__(self, b):
        return LT.apply(b, self)

    def __eq__(self, b):
        return EQ.apply(self, b)

    def __sub__(self, b):
        return Add.apply(self, Neg.apply(b))

    def __neg__(self):
        return Neg.apply(self)

    def log(self):
        return Log.apply(self)

    def exp(self):
        return Exp.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def get_data(self):
        "Returns the raw float value"
        return self.data
    
class ScalarFunction(FunctionBase):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """
    @staticmethod
    def forward(ctx, *inputs):
        """
        Forward call, compute :math:`f(x_0 \ldots x_{n-1})`.

        Args:
            ctx (:class:`Context`): A container object to save
                                    any information that may be needed
                                    for the call to backward.
            *inputs (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`.

        Should return float the computation of the function :math:`f`.
        """
        pass  # pragma: no cover
        
    @staticmethod
    def backward(ctx, d_out):
        r"""
        Backward call, computes :math:`f'_{x_i}(x_0 \ldots x_{n-1}) \times d_{out}`.

        Args:
            ctx (Context): A container object holding any information saved during in the corresponding `forward` call.
            d_out (float): :math:`d_out` term in the chain rule.

        Should return the computation of the derivative function
        :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.

        """
        pass  # pragma: no cover

    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a

class Add(ScalarFunction):
    """
    Add two scalars together.
    """
    @staticmethod
    def forward(ctx, a, b):
        return a + b
    
    @staticmethod
    def backward(ctx, d_out):
        return d_out, d_out  # d_out is the derivative of the sum w.r.t. both inputs
    
class Log(ScalarFunction):
    "Log function :math:`f(x) = log(x)`"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.log_back(a, d_output)
    
class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx, d_output):
        a, b = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)
    
class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.inv_back(a, d_output)
    
class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx, a):
        return operators.mul(-1.0, a)

    @staticmethod
    def backward(ctx, d_output):
        return operators.mul(-1.0, d_output)
    
class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx, a):
        b = operators.sigmoid(a)
        ctx.save_for_backward(b)
        return b

    @staticmethod
    def backward(ctx, d_output):
        b = ctx.saved_values
        return operators.mul(d_output, operators.mul(b, operators.add(1, operators.neg(b))))
    
class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.relu_back(a, d_output)
    
class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.mul(operators.exp(a), d_output)
    
class LT(ScalarFunction):
    "Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx, a, b):
        return 1.0 if operators.lt(a, b) else 0.0

    @staticmethod
    def backward(ctx, d_output):
        return 0.0, 0.0

class EQ(ScalarFunction):
    "Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx, a, b):
        return 1.0 if operators.eq(a, b) else 0.0

    @staticmethod
    def backward(ctx, d_output):
        return 0.0, 0.0