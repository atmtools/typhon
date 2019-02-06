"""Various statistical functions

"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import numbers

import numpy

import scipy.stats
import scipy.special


def bin(x, y, bins):
    """Bin/bucket y according to values of x.

    Returns list of arrays, one element with values for bin.
    Digitising happens with numpy.digitize.

    You can use this for geographical data, but if your x and
    y are in lat/lons the bins will be equirectangular (if you
    want equal-area bins you will have to reproject / change
    coordinates).

    Arguments:

        x (ndarray): Coordinate that y is binned along
        y (ndarray):  Data to be binned.  First dimension must match
            length of x.  All subsequent dimensions are left untouched.
        bins (ndarray): Bins according to which sort data.

    Returns:

        List of arrays, one element per bin.
    """
    if x.size == y.size == 0:
        return [y[()] for b in bins]
    digits = numpy.digitize(x, bins)
    binned = [y[digits == i, ...] for i in range(len(bins))]
    return binned


def bin_nd(binners, bins, data=None):
    """Bin/bucket data in arbitrary number of dimensions

    For example, one can bin geographical data according to lat/lon
    through:

    >>> binned = bin_nd([lats, lons], [lat_bins, lon_bins])

    The actually binned data are the indices for the arrays lats/lons,
    which hopefully corresponds to indices in your actual data.

    Data that does not fit in any bin, is not binned anywhere.

    You can use this for geographical data, but if your x and
    y are in lat/lons the bins will be equirectangular (if you
    want equal-area bins you will have to reproject / change
    coordinates).

    Note: do NOT pass the 3rd argument, `data`.  This is used purely for
    the implementation using recursion.  Passing anything here explicitly
    is a recipe for disaster.

    Arguments:

        binners (List[ndarray]): Axes that data is binned at.  This is
            akin to the x-coordinate in `:func:bin`.

        bins (List[ndarray]): Edges for the bins according to which bin
            data.

    Returns:
        n-D ndarray of type `object`, with indices describing what bin
        elements belong to.
    """

    if len(bins) != len(binners):
        raise ValueError("Length of bins must equal length of binners. "
                         "Found {} bins, {} binners.".format(
                             len(bins), len(binners)))

    for b in bins:
        if b.ndim != 1:
            raise ValueError("Bin-array must be 1-D. "
                             "Found {}-D array.".format(b.ndim))

    if not all([b.size == binners[0].size for b in binners[1:]]):
        raise ValueError("All binners must have same length.")

    dims = numpy.array([b.size for b in bins])

    nd = len(binners)

    if nd == 0:
        return numpy.array([], dtype=numpy.uint64)

    indices = numpy.arange(binners[0].size, dtype=numpy.uint64)
    if data is None:
        data = indices

    if nd > 1:
        # innerbinned = bin(binners[-1], data, bins[-1])
        innerbinned = bin(binners[-1], indices, bins[-1])
        outerbinned = []
        for (i, ib) in enumerate(innerbinned):
            obinners = [x[ib] for x in binners[:-1]]
            ob = bin_nd(obinners, bins[:-1], data[ib])
            outerbinned.append(ob)

        # go through some effort to make sure v[i, j, ...] is always
        # numpy.uint64, whereas v is numpy.object_
        # do this in steps, see comment in the else:-block below for
        # reasoning
        #
        # We have outerbinned, which has length n_N, and contains ndarrays
        # of size n_1 * n_2 * ... * n_{N-1}.
        #
        # We want V to be n_1 * n_2 * ... * n_N, where N is the number of
        # dimensions we are binning.
        #
        # The following could /probably/ be do with some sophisticated
        # list comprehension and permutation, but this is clearer.

        V = numpy.empty(shape=dims, dtype=numpy.object_)
        for i in range(len(outerbinned)):
            V[..., i] = outerbinned[i]

#        V.T[...] = [
#            [numpy.array(e.tolist(), dtype=numpy.uint64)
#                for e in l] for l in outerbinned]
        return V
        # return numpy.array(v, dtype=numpy.object_)
    else:
        # NB: I should not convert a list-of-ndarrays to an object-ndarray
        # directly.  If all nd-arrays have the same dimensions (such as
        # size x=0), the converted nd-array will have x as an additional
        # dimension, rather than having object arrays inside the
        # container.  To prevent this, explicitly initialise the ndarray.
        binned = bin(binners[0], data, bins[0])
        B = numpy.empty(shape=len(binned), dtype=numpy.object_)
        B[:] = binned
        return B


def binned_statistic(coords, data, bins, statistic=None):
    """Bin data and calculate statistics on them

    As :func:`scipy.stats.binned_statistic` but faster for simple
    statistics.

    Args:
        coords: 1-dimensional numpy.array with the coordinates that should be
            binned.
        data: A numpy.array on which the statistic should be applied.
        bins: The bins used for the binning.
        statistic: So far only *mean* is implemented.

    Returns:
        A numpy array with statistic results.
    """
    bin_indices = numpy.digitize(coords, bins)
    data_sum = numpy.bincount(bin_indices, weights=data, minlength=bins.size)
    data_number = numpy.bincount(bin_indices, minlength=bins.size)
    return data_sum / data_number

def get_distribution_as_percentiles(x, y,
                                    bins,
                                    ptiles=(5, 25, 5, 75, 95)):
    """get the distribution of y vs. x as percentiles.

    Bin y-data according to x-data (using :func:`typhon.math.stats.bin`).
    Then, within each bin, calculate percentiles.

    Arguments:

        x (ndarray): data for x-axis
        y (ndarray): data for y-axis
        bins (ndarray): Specific bins to use for dividing the x-data.
        ptiles (ndarray): Percentiles to use.
    """

    # explicitly get rid of masked data, because scoreatpercentile is not
    # masked-array aware
    try:
        good = (~x.mask) & (~y.mask)
    except AttributeError: # not a masked-array, leave as-is
        pass
    else: # surely masked arrays
        x = x[good].data
        y = y[good].data
    ybinned = bin(x, y, bins)
    return numpy.vstack([scipy.stats.scoreatpercentile(b, ptiles)
                         for b in ybinned])


def adev(x, dim=-1):
    r"""Calculate Allan deviation in its simplest form

    Arguments:

        x (ndarray or xarray DataArray): n-dim array for Allan calculation
        dim (int or str): optional, axis to operate along, defaults to
            last.  If you pass a str, x must be a xarray.DataArray and the
            dimension will be a name.

    .. math::
        \sigma = \sqrt{\frac{1}{2(N-1)} \sum_{i=1}^{N-1} (y_{i+1} - y_i)^2}

    Equation source: Jon Mittaz, personal communication, April 2016
    """

    
    if isinstance(dim, numbers.Integral):
        # dimension by number, probably ndarray
        x = x.swapaxes(-1, dim)
        N = x.shape[-1]
        return numpy.sqrt(1/(2*(N-1)) *
                          ((x[..., 1:] - x[..., :-1])**2).sum(-1))
    else:
        # dimension by name, should be xarray.Dataarray
        N = x.sizes[dim]
        return numpy.sqrt((1/(2*(N-1)) * x.diff(dim=dim)**2).sum(dim=dim))

def corrcoef(mat):
    """Calculate correlation coefficient with p-values

    Calculate correlation coefficients along with p-values.

    Arguments:

        mat (ndarray): 2-D array [p×N] for which the correlation matrix is
            calculated

    Returns:

        (r, p) where r is a p×p matrix with the correlation coefficients,
        obtained with numpy.corrcoef, and p is 

    Attribution:
    
        this code, or an earlier version was posted by user 'jingchao' on
        Stack Overflow at 2014-7-3 at
        http://stackoverflow.com/a/24547964/974555 and is licensed under
        CC BY-SA 3.0.  This notice may not be removed.
    """
    r = numpy.corrcoef(mat)
    rf = r[numpy.triu_indices(r.shape[0], 1)]
    df = mat.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = scipy.special.betainc(0.5 * df, 0.5, df / (df + ts))
    p = numpy.zeros(shape=r.shape)
    p[numpy.triu_indices(p.shape[0], 1)] = pf
    p[numpy.tril_indices(p.shape[0], -1)] = pf
    p[numpy.diag_indices(p.shape[0])] = numpy.ones(p.shape[0])
    return (r, p)
