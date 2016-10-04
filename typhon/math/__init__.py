"""Maths-related modules
"""
import numpy as np

from . import stats
from . import array

__all__ = ['stats',
           'array',
           'integrate_column',
           'sum_digits',
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
