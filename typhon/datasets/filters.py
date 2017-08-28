"""Collection of classes related to filtering

"""

import abc
import numpy

class OutlierFilter(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def filter_outliers(self, C):
        ...

class MEDMAD(OutlierFilter):
    """Outlier filter based on Median Absolute Deviation

    """

    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def filter_outliers(self, C):
        cutoff = self.cutoff
        if C.ndim == 3:
            med = numpy.ma.median(
                C.reshape(C.shape[0]*C.shape[1], C.shape[2]),
                0)
            mad = numpy.ma.median(
                abs(C - med).reshape(C.shape[0]*C.shape[1], C.shape[2]),
                0)
        elif C.ndim < 3:
            med = numpy.ma.median(C.reshape((-1,)))
            mad = numpy.ma.median(abs(C - med).reshape((-1,)))
        else:
            raise ValueError("Cannot filter outliers on "
                "input with {ndim:d}>3 dimensions".format(ndim=C.ndim))
        fracdev = ((C - med)/mad)
        return abs(fracdev) > cutoff
