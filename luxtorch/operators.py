"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

def mul(x, y):
    ":math:`f(x, y) = x * y`"
    return x * y

def id(x):
    ":math:`f(x) = x`"
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x):
    ":math:`f(x) = -x`"
    return -x


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    return x if x > y else y


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    return 1.0 if math.fabs(x - y) < 1e-2 else 0.0

def sigmoid(x):
    """
    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    return 1. / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1. + math.exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    return x if x > 0 else 0.


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    return d / (x + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    return - float(d) / (x**2)


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    return d if x > 0 else 0

def sqrt(x):
    ":math:`f(x) = \sqrt{x}`"
    return math.sqrt(x + EPS)

def sqrt_back(x, d):
    r"If :math:`f(x) = \sqrt{x}` compute d :math:`d \times f'(x)`"
    return 0.5 * d / sqrt(x + EPS)

def map(fn):
    """
    Higher-order map.

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def process(ls):
        arr = []
        for item in ls:
            arr.append(fn(item))
        return arr

    return process

def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)

def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    def process(ls1, ls2):
        arr = []
        for i in range(len(ls1)):
            arr.append(fn(ls1[i], ls2[i]))
        return arr

    return process

def addList(ls1, ls2):
    "Use :func:`zipWith` and :func:`add` to add two lists `ls1` and `ls2`"
    return zipWith(add)(ls1, ls2)

def reduce(fn, start):
    r"""
    Higher-order reduce.

    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def process(ls):
        ans = start
        for item in ls:
            ans = fn(item, ans)
        return ans

    return process

def sum(ls):
    "Use :func:`reduce` and :func:`add` to sum a list `ls`"
    return reduce(add, 0)(ls)

def prod(ls):
    "Use :func:`reduce` and :func:`mul` to product a list `ls`"
    return reduce(mul, 1)(ls)