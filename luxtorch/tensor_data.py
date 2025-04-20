import random
from .operators import prod
from numpy import array, float64, ndarray

MAX_DIMS = 32

class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass

def index_to_position(index, strides):
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index (array-like): index tuple of ints
        strides (array-like): tensor strides

    Returns:
        int : position in storage
    """
    pos = 0
    for i, v in enumerate(index):
        pos += v * strides[i]
    return pos