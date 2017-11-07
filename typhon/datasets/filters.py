"""Collection of classes related to filtering

"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

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
