"""Functions operating on arrays."""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
#
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import numpy as np
import scipy.stats


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

    localmin = np.hstack(
        (False, (arr[1:-1] < arr[0:-2]) & (arr[1:-1] < arr[2:]), False)
    )

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

    selection = np.ones(shape=M.shape, dtype="?")

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
            selection = selection & lelo & sthi

    return M[selection]


def parity(v):
    """Vectorised parity-checking.

    For any ndarray with an nd.integer dtype, return an equally shaped
    array with the bit parity for each element.

    Arguments:

        v (numpy.ndarray): Array of integer dtype

    Returns:

        ndarray with uint8 dtype with the parity for each value in v
    """

    v = v.copy()  # don't ruin original
    parity = np.zeros(dtype=">u1", shape=v.shape)
    while v.any():
        parity[v != 0] += 1
        v &= v - 1
    return parity


def mad_outliers(arr, cutoff=10, mad0="raise"):
    """Mask out mad outliers

    Mask out any values that are more than N times the median absolute
    devitation from the median.

    Although I (Gerrit Holl) came up with this myself, it's also
    documented at:

    http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/

    except that I rolled by own approach for "what if mad==0".

    Note: If all values except one are constant, it is not possible to
    determine whether the remaining one is an outlier or “reasonably
    close” to the rest, without additional hints.  In this case, some
    outliers may go unnoticed.

    Arguments:

        arr (numpy.ndarray): n-D array with numeric dtype

        cutoff (int): Maximum tolerable normalised fractional distance

        mad0 (str): What to do if mad=0.  Can be 'raise', 'ignore', or
            'perc'.  In case of 'perc', will search for the lowest
            percentile at which the percentile absolute deviation is
            nonzero, increase the cutoff by the fractional approach toward
            percentile 100, and use that percentile instead.  So if the
            first non-zero is at percentile 75%, it will use the
            75th-percntile-absolute-deviation and increase the cutoff by
            a factor (100 - 50)/(100 - 75).

    Returns:

        ndarray with bool dtype, True for outliers
    """

    if arr.ptp() == 0:
        return np.zeros(shape=arr.shape, dtype="?")

    ad = abs(arr - np.ma.median(arr))
    mad = np.ma.median(ad)
    if mad == 0:
        if mad0 == "raise":
            raise ValueError("Cannot filter outliers, MAD=0")
        elif mad0 == "perc":
            # try other percentiles
            perc = np.r_[np.arange(50, 99, 1), np.linspace(99, 100, 100)]
            pad = scipy.stats.scoreatpercentile(ad, perc)
            if (pad == 0).all():  # all constant…?
                raise ValueError("These data are weird!")
            p_i = pad.nonzero()[0][0]
            cutoff *= (100 - 50) / (100 - perc[p_i])
            return (ad / pad[p_i]) > cutoff
    elif mad is np.ma.masked:
        # all are masked already…
        return np.ones(shape=ad.shape, dtype="?")
    else:
        return (ad / mad) > cutoff


def argclosest(array, value, retvalue=False):
    """Returns the index of the closest value in array.

    Parameters:
        array (ndarray): Input array.
        value (float): Value to compare to.
        retvalue (bool): If True, return the index and the closest value.

    Returns:
        int, float:
        Index of closest value, Closest value (if ``retvalue`` is True)

    """
    idx = np.abs(np.asarray(array) - value).argmin()

    return (idx, array[idx]) if retvalue else idx
