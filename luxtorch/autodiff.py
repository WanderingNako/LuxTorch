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