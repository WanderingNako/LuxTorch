import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

def tensor_map(fn):
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        out_index = np.array(out_shape)
        in_index = np.array(in_shape)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            data = in_storage[index_to_position(in_index, in_strides)]
            map_data = fn(data)
            out[index_to_position(out_index, out_strides)] = map_data
    return _map

def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out
    return ret

def tensor_zip(fn):
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    def _zip(out, out_shape, out_strides, a_storage, a_shape, a_strides,
             b_storage, b_shape, b_strides):
        out_index = np.array(out_shape)
        a_index = np.array(a_shape)
        b_index = np.array(b_shape)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            data_a = a_storage[index_to_position(a_index, a_strides)]
            data_b = b_storage[index_to_position(b_index, b_strides)]
            zip_data = fn(data_a, data_b)
            out[index_to_position(out_index, out_strides)] = zip_data
    return _zip

def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out
    return ret

def tensor_reduce(fn):
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """
    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides,
                reduce_dim):
        out_index = np.array(out_shape)
        for i in range(len(out)):
            to_index(i, out_shape, out_index)
            for j in range(a_shape[reduce_dim]):
                a_index = out_index.copy()
                a_index[reduce_dim] = j
                pos_a = index_to_position(a_index, a_strides)
                out[i] = fn(a_storage[pos_a], out[i])
    return _reduce

def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out
    return ret

def tensor_matrix_multiply(out, out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
    """
    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    out_index = np.array(out_shape)
    for i in range(len(out)):
        to_index(i, out_shape, out_index)
        middle_dim = a_shape[-1]
        for k in range(middle_dim):
            a_index = out_index.copy()
            a_index[-1] = k
            a_pos = index_to_position(a_index, a_strides)

            b_index = out_index.copy()
            b_index[-2] = k
            b_pos = index_to_position(b_index, b_strides) 
            out[i] += a_storage[a_pos] * b_storage[b_pos]
            


def matrix_multiply(a, b):
    """
    Tensor matrix multiply ::
          for i:
            for j:
              for k:
                out[i, j] += a[i, k] * b[k, j]

    Should work for tensor shapes of 2 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """
    # Make these always be a 2 dimensional multiply
    assert len(a.shape) == 2 and len(b.shape) == 2, "Only perform 2 dimensional multiply"
    assert a.shape[-1] == b.shape[-2], "Can't perform matrix multiplication"
    ls = []
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    out = a.zeros(tuple(ls))
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply