class Module:
    """
    Module from a tree that stores parameters and other submodules. They make up the basis of neural network stacks.
    Attributes:
        _modules (dict of name x : class: `Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode
    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True
    
    def modules(self):
        """
        Returns the child modules of this module.(Auto updated)
        """
        return self._modules.values()

    def train(self):
        """
        Set the module of this module and all descendent modules to `train`.
        """
        def update(cur):
            cur.training = True
            for child in cur.modules():
                update(child)
        update(self)

    def eval(self):
        """
        Set the module of this module and all descendent modules to `eval`.
        """
        def update(cur):
            cur.training = False
            for child in cur.modules():
                update(child)
        update(self)
    
    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each descendent parameter.
        """
        res = {}
        def helper(name, node):
            prefix = name + '.' if name else ''
            for k, v in node._parameters.items():
                res[prefix + k] = v
            for k, v in node._modules.items():
                helper(prefix + k, v)
        helper('', self)
        return res
    
    def parameters(self):
        "Enumerate over all the parameters of this module and its descendents."
        return self.named_parameters().values()
    
    def add_parameter(self, name, x):
        """
        Add a parameter to this module.

        Args:
            name (str): The name of the parameter.
            x: The value of the parameter.
        """
        val = Parameter(x, name)
        self._parameters[name] = val
        return val
    
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        else:
            super().__setattr__(name, value)
    
    def __getattr__(self, name):
        if name in self._modules:
            return self._modules[name]
        elif name in self._parameters:
            return self._parameters[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def forward(self, *args, **kwds):
        assert False, "forward() not implemented in this module."

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s
        
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n" + "\n".join(lines) + "\n"
        main_str += ")"
        return main_str

class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """
    def __init__(self, x=None, name=None):
        self.value = x
        self.name  = name
        if hasattr(x, 'requires_grad_'):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name
    
    def update(self, x):
        """
        Update the value of this parameter.

        Args:
            x: The new value of this parameter.
        """
        self.value = x
        if hasattr(x, 'requires_grad_'):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)
    
    def __str__(self):
        return str(self.value)