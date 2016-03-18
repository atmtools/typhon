"""Functions operating on arrays

"""

import numpy

def localmin(arr):
    """Find local minima for 1-D array

    Given a 1-dimensional numpy.ndarray, return the locations of any local
    minimum as a boolean array.  The first and last item are always
    considered False.

    Arguments:

        localmin (numpy.ndarray): 1-D ndarray for which to find local
            minima.  Should have a numeric dtype.

    Returns:
        
        numpy.ndarray with dtype `bool`.  True for any element that is
        strictly smaller than both neighbouring elements.  First and last
        element are always False.
    """

    localmin = numpy.hstack(
        (False,
        (arr[1:-1] < arr[0:-2]) & (arr[1:-1] < arr[2:]),
         False))

    return localmin

def limit_ndarray(M, limits):
    """Select elements from structured ndarray based on value ranges

    This function filters a structured ndarray based on ranges defined for
    zero or more fields.  For each field f with limits (lo, hi), it will
    select only those elements where lo<=X[f]<hi.

    >>> X = array([(2, 3), (4, 5), (8, 2), (5, 1)],
                   dtype=[("A", "i4"), ("B", "i4")])

    >>> print(limit_ndarray(X, {"A": (2, 5)}))
    [(2, 3) (4, 5)]

    >>> X = array([([2, 3], 3), ([4, 6], 5), ([8, 3], 2), ([5, 3], 1)],
                   dtype=[("A", "i4", 2), ("B", "i4")])

    >>> print(limit_ndarray(X, {"A": (2, 5, "all")}))
    [([2, 3], 3)]

    Arguments:
        
        M (numpy.ndarray): 1-D structured ndarray

        limits (dict): Dictionary with limits.  Keys must correspond to
            fields in M.  If this is a scalar field
            (`M.dtype[field].shape==()`), values are tuples (lo, hi).
            If this is a multidimensional field, values are tuples (lo,
            hi, mode), where mode must be either `all` or `any`.
            Values in the range [lo, hi) are retained, applying all or any
            when needed.

    Returns:

        ndarray subset of M.  This is a view, not a copy.
    """

    selection = numpy.ones(shape=M.shape, dtype="?")

    for (field, val) in limits.items():
        ndim = len(M.dtype[field].shape)
        if ndim == 0:
            (lo, hi) = val
            selection = selection & (M[field] >= lo) & (M[field] < hi)
        else:
            (lo, hi, mode) = val
            lelo = M[field] >= lo
            sthi = M[field] < hi
            while lelo.ndim > 1:
                lelo = getattr(lelo, mode)(-1)
                sthi = getattr(sthi, mode)(-1)
            selection = (selection & lelo & sthi)

    return M[selection]
