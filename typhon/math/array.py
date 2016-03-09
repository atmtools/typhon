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

