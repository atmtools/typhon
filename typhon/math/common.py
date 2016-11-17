# -*- coding: utf-8 -*-

"""Common functions for typhon.math.
"""
import numpy as np
import functools


__all__ = [
    'integrate_column',
    'sum_digits',
    'nlogspace',
    "promote_maximally",
    "calculate_precisely"
]


def integrate_column(y, z, axis=None):
    """Calculate the column integral of given data.

    Parameters:
        y (np.array): Data array.
        z (np.array): Height levels.

    Returns:
        float: Column integral.

    """
    # TODO: Find a clean way to handle multidimensional input.
    y_level = (y[1:] + y[:-1]) / 2

    return np.nansum(y_level * np.diff(z), axis=axis)


def sum_digits(n):
    """Calculate the sum of digits.

    Parameters:
       n (int): Number.

    Returns:
        int: Sum of digitis of n.

    Examples:
        >>> sum_digits(42)
        6

    """
    s = 0
    while n:
        s += n % 10
        n //= 10

    return s


def nlogspace(start, stop, num=50):
    """Creates a vector with equally logarithmic spacing.

    Creates a vector with length num, equally logarithmically
    spaced between the given end values.

    Parameters:
        start (int): The starting value of the sequence.
        stop (int): The end value of the sequence,
            unless `endpoint` is set to False.
        num (int): Number of samples to generate.
            Default is 50. Must be non-negative.

    Returns: ndarray.
    """
    return np.exp(np.linspace(np.log(start), np.log(stop), num))

def promote_maximally(x):
    """Return copy of x with highest-precision dtype

    Currently only works for simple dtypes of type float, int, or uint.
    Anything else is returned unchanged.

    Experimental function.
    """
    try:
        q = x.m
        u = x.u
    except AttributeError: # not a pint quantity
        q = x
        u = None
    kind = q.dtype.kind
    if kind in "fiu":
        newx = q.astype(kind + "8")
        return newx*u if u else newx
    else:
        return x

def calculate_precisely(f):
    """Raise all arguments to their highest numpy precision.

    This decorator copies any floats to f8, ints to i8, preserving masked
    arrays and/or ureg units.

    Currently only works for simple dtypes of float, int, or uint.

    This makes a copy.  Therefore, it is memory-intensive and it does not
    work if function has need to change values in-place.

    Experimental function.

    See also: https://github.com/numpy/numpy/issues/593
    """

    # NB: this decorator supports pint but does not depend on it
    @functools.wraps(f)
    def inner(*args, **kwargs):
        newargs = []
        for arg in args:
            try:
                newargs.append(promote_maximally(arg))
            except AttributeError:
                newargs.append(arg)
        newkwargs = {}
        for (k, v) in kwargs.items():
            try:
                newkwargs[k] = promote_maximally(v)
            except AttributeError:
                newkwargs[k] = v
        return f(*newargs, **newkwargs)
    return inner
