# -*- coding: utf-8 -*-

"""Common functions for typhon.math.
"""
import numpy as np


__all__ = [
    'integrate_column',
    'sum_digits',
    'nlogspace',
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
