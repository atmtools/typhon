# -*- coding: utf-8 -*-

"""Common functions for typhon.math.
"""
import numpy as np
import functools

__all__ = [
    'cantor_pairing',
    'integrate_column',
    'interpolate_halflevels',
    'sum_digits',
    'nlogspace',
    'promote_maximally',
    'calculate_precisely',
    'squeezable_logspace',
]


def cantor_pairing(a, b, ):
    """Create an unique number from two natural numbers

    For more information about the Cantor pairing function, have a look at:
    https://en.wikipedia.org/wiki/Pairing_function

    This create an unique number from two natural numbers according to
    .. math::
        \pi (a,b):={\frac{1}{2}}(a+b)(a+b+1)+b.

    Args:
        a: A numpy.array with natural numbers, i.e. unsigned integer.
        b: A numpy.array with natural numbers, i.e. unsigned integer.

    Returns:
        A numpy.array with the unique values.
    """

    a_b_sum = a + b
    return (0.5 * a_b_sum * (a_b_sum+1) + b).astype("int")


def integrate_column(y, x=None, axis=0):
    """Integrate array along an arbitrary axis.

    Note:
        This function is just a wrapper for :func:`numpy.trapz`.

    Parameters:
        y (ndarray): Data array.
        x (ndarray): Coordinate array.
        axis (int): Axis to integrate along for multidimensional input.

    Returns:
        float or ndarray: Column integral.

    Examples:
        >>> import numpy as np
        >>> x = np.linspace(0, 1, 5)
        >>> y = np.arange(5)
        >>> integrate_column(y)
        8.0
        >>> integrate_column(y, x)
        2.0
    """
    return np.trapz(y, x, axis=axis)


def interpolate_halflevels(x, axis=0):
    """Returns the linear inteprolated halflevels for given array.

    Parameters:
        x (ndarray): Data array.
        axis (int): Axis to interpolate along.

    Returns:
        ndarray: Values at halflevels.

    Examples:
        >>> interpolate_halflevels([0, 1, 2, 4])
        array([ 0.5,  1.5,  3. ])
    """
    return (np.take(x, range(1, np.shape(x)[axis]), axis=axis) +
            np.take(x, range(0, np.shape(x)[axis] - 1), axis=axis)) / 2


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

    Examples:
        >>> nlogspace(10, 1000, 3)
        array([  10.,  100., 1000.])
    """
    return np.exp(np.linspace(np.log(start), np.log(stop), num))

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822.  That specifically applies to the functions
# promote_maximally and calculate_precisely.
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

def promote_maximally(x):
    """Return copy of x with high precision dtype.

    Converts input of 'f2', 'f4', or 'f8' to 'f8'.  Please don't pass f16.
    f16 is misleading and naughty.

    Converts input of 'u1', 'u2', 'u4', 'u8' to 'u8'.

    Converts input of 'i1', 'i2', 'i4', 'i8' to 'i8'.

    Naturally, this copies the data and increases memory usage.

    Anything else is returned unchanged.

    If you input a pint quantity you will get back a pint quantity.

    Experimental function.
    """
    try:
        q = x.m
        u = x.u
    except AttributeError:  # not a pint quantity
        q = x
        u = None
    try:
        kind = q.dtype.kind
    except AttributeError:  # not from numpy
        return q
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


def squeezable_logspace(start, stop, num=50, squeeze=1., fixpoint=0.):
    """Create a logarithmic grid that is squeezable around a fixpoint.

    Parameters:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int): Number of sample to generate (Default is 50).
        squeeze (float): Factor with which the first stepwidth is
            squeezed in logspace. Has to be between  ``(0, 2)``.
            Values smaller than one compress the gridpoints,
            while values greater than 1 strecht the spacing.
            The default is ``1`` (do not squeeze.)
        fixpoint (float): Relative fixpoint for squeezing the grid.
            Has to be between ``[0, 1]``. The  default is ``0`` (bottom).

    Examples:
        Constructing an unsqueezed grid in logspace.

        >>> squeezable_logspace(1, 100, num=5)
        array([1., 3.16227766, 10., 31.6227766, 100.])

        Constructing a grid that is squeezed at the start.

        >>> squeezable_logspace(1, 100, num=5, squeeze=0.5)
        array([1., 1.77827941, 4.64158883, 17.7827941, 100.])

        Constructing a grid that is squeezed in the middle.

        >>> squeezable_logspace(1, 100, num=5, squeeze=0.5, fixpoint=0.5)
        array([1., 5.62341325, 10., 17.7827941, 100.])

        Visualization of different fixpoint and squeeze factor combinations.

        .. plot::
            :include-source:

            import itertools

            from typhon.plots import profile_p_log
            from typhon.math import squeezable_logspace


            fixpoints = [0, 0.7]
            squeezefacotrs = [0.5, 1.5]
            combinations = itertools.product(fixpoints, squeezefacotrs)

            fig, axes = plt.subplots(len(fixpoints), len(squeezefacotrs),
                                    sharex=True, sharey=True)
            for ax, (fp, s) in zip(axes.flat, combinations):
                p = squeezable_logspace(1000e2, 0.01e2, 20,
                                        fixpoint=fp, squeeze=s)
                profile_p_log(p, np.ones(p.size),
                            marker='.', linestyle='none', ax=ax)
                ax.set_title('fixpoint={}, squeeze={}'.format(fp, s),
                            size='x-small')

    Returns:
        ndarray: (Squeezed) logarithmic grid.
    """
    # The squeeze factor has to be between 0 and 2. Otherwise, the order
    # of groidpoints is arbitrarily swapped as the scaled stepsizes
    # become negative.
    if squeeze <= 0 or squeeze >= 2:
        raise ValueError(
            'Squeeze factor has to be in the open interval (0, 2).'
        )

    # The fixpoint has to be between 0 and 1. It is used as a relative index
    # within the grid, values exceeding the limits result in an IndexError.
    if fixpoint < 0 or fixpoint > 1:
        raise ValueError(
            'The fixpoint has to be in the closed interval [0, 1].'
        )

    # Convert the (relative) fixpoint into an index dependent on gridsize.
    fixpoint_index = int(fixpoint * (num - 1))

    # Create a gridpoints with constant spacing in log-scale.
    samples = np.linspace(np.log(start), np.log(stop), num)

    # Select the bottom part of the grid. The gridpoint is included in the
    # bottom and top part to ensure right step widths.
    bottom = samples[:fixpoint_index + 1]

    # Calculate the stepsizes between each gridpoint.
    steps = np.diff(bottom)

    # Re-adjust the stepwidth according to given squeeze factor.
    # The squeeze factor is linearly changing in logspace.
    steps *= np.linspace(2 - squeeze, squeeze, steps.size)

    # Reconstruct the actual gridpoints by adding the first grid value and
    # the cumulative sum of the stepwidths.
    bottom = bottom[0] + np.cumsum(np.append([0], steps))

    # Re-adjust the top part as stated above.
    # **The order of squeezing is inverted!**
    top = samples[fixpoint_index:]
    steps = np.diff(top)
    steps *= np.linspace(squeeze, 2 - squeeze, steps.size)
    top = top[0] + np.cumsum(np.append([0], steps))

    # Combine the bottom and top parts to the final grid. Drop the fixpoint
    # in the bottom part to avoid duplicates.
    return np.exp(np.append(bottom[:-1], top))
