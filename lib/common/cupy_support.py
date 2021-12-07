# encoding: utf-8
# module cupy_support.py
# from common
# author: Kyle Millar (kyle.millar@adelaide.edu.au)

"""
cupy_support has been designed to provide a common interface for cupy and numpy operations. While the interface
between numpy and cupy are very similar there are edges cases where they have to be called differently. This is
mainly due to the current development stage of cupy.

A common interface between cupy and numpy was required to achieve a more flexible design. Cupy can greatly speed up
operations on large matrices; however, it requires a GPU with CUDA support (i.e., a Nvidia GPU). For systems without a
Nvidia GPU, Numpy will be used. While numpy is still usable, for cupy can see 10x-60x time improvement for large graphs.

!!! note "Small Matrices"
    Numpy is actually faster on small matrices. This is to due to the overhead of setting the operations up on the GPU.
    For operations with less than ~10^4 elements numpy will likely be faster.
"""

# ---
import scipy
from scipy.sparse import csr_matrix
import numpy as np
# ---

# Tests whether cupy is supported on this system.
# If cupy is not available, numpy will be used instead.
# ---
import cupy as cp
import cupyx

try:
    import cupy as cp
    import cupyx
    xp = cp
    _cupy_available = True
    print("Device: CUDA")
except ImportError:
    xp = np
    _cupy_available = False
    print("Device: CPU")
# ---

# - Sparse Matrices -------------------------------------------------------------------------------------------------- #
# Note: Currently only using scipy sparse matrices. Cupy's sparse matrices require further development before they are a
# usable replacement for scipy's sparse matrices. Currently, Cupy's sparse matrices are much slower to manipulate.
if _cupy_available:
    #csr_matrix = cupyx.scipy.sparse.csr_matrix
    csr_matrix = scipy.sparse.csr_matrix
else:
    csr_matrix = scipy.sparse.csr_matrix
# -------------------------------------------------------------------------------------------------------------------- #

# - scatter_add ------------------------------------------------------------------------------------------------------ #
# Scatter add, performs an in-line series of additions on a given matrix using vectorisation. This is much faster than
# performing each add operation individually.
# Note: cupyx.scatter_add is much faster than np.add.at
if _cupy_available:
    # https://docs.cupy.dev/en/stable/reference/generated/cupyx.scatter_add.html
    scatter_add = cupyx.scatter_add
else:
    # https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
    scatter_add = np.add.at
# -------------------------------------------------------------------------------------------------------------------- #


# - Mean Squared Error (MSE) ----------------------------------------------------------------------------------------- #
# TODO - Remove mean_squared_error function. No longer relevant since the remove of chainer.
def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error between matrices ``y_true`` and ``y_pred``. This function has been defined to
    provide a common mean_squared_error function for both cupy and Numpy.

    !!! note "Numpy MSE"
        Numpy does have a mse error function in the form of sklearn.metrics.mean_squared_error; however, it does not
        work for 3-dimensional arrays.

    **Parameters**
    > **y_true:** ``array`` -- True results.

    > **y_pred:** ``array`` -- Predicted results.

    **Returns**
    > **mse:** ``float`` -- Mean squared error between arrays - ``y_true`` and ``y_pred``.
    """
    return float(xp.mean(xp.square(y_true - y_pred)))
# -------------------------------------------------------------------------------------------------------------------- #


# - Round ------------------------------------------------------------------------------------------------------------ #
# TODO - use numpy around.
def xp_round(array, decimals=6):
    """
    Round the elements of an array to a given number of decimals. This has been defined as cupy uses ``around`` as its
    rounding function and numpy uses ``round``.

    **Parameters**
    > **array:** ``array`` -- The array to be rounded.

    > **decimals:** ``int`` -- The number of decimals to round each element to.

    **Returns**
    > **array:** ``array`` -- The rounded array.
    """

    if _cupy_available:
        return xp.around(array, decimals)
    else:
        return xp.round(array, decimals)
# -------------------------------------------------------------------------------------------------------------------- #


# - Sigmoid ---------------------------------------------------------------------------------------------------------- #
def sigmoid(x, b=1):
    """
    Passes input through the sigmoid function. Bounds input between [0, 1]. This function has been defined as cupy and
    numpy both use unique function calls for the sigmoid function.

    **Parameters**
    > **x:** ``array`` -- an n-dimensional array for which to compute the element-wise sigmoid upon.

    > **b:** ``float`` -- A variable used to modify the slope of the sigmoid function.

    **Returns**
    > **x:** ``array`` -- The result of applying an element-wise sigmoid to the input array.
    """
    ones = xp.ones(shape=x.shape)
    return xp.divide(ones, xp.add(ones, xp.exp(b * xp.negative(x))))
# -------------------------------------------------------------------------------------------------------------------- #

