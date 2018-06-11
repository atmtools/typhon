import typhon
import numpy as np
import matplotlib.pyplot as plt
from .catalogues import Sparse

class Block(object):
    def __init__(self, i, j, row_start, column_start, inverse, matrix):
        self.i = i
        self.j = j
        self.row_start    = row_start
        self.column_start = column_start
        self.inverse = inverse
        self.matrix = matrix

    def write_xml(self, xmlwriter, attr = None):

        if attr is None:
            attr = {}

        attr["row_index"] = self.i
        attr["column_index"] = self.j
        attr["row_start"] = self.row_start
        attr["column_start"] = self.column_start
        attr["row_extent"], attr["column_extent"] = self.matrix.shape
        attr["is_inverse"] = int(self.inverse)

        if type(self.matrix) == Sparse:
            attr["type"] = "Sparse"
        else:
            attr["type"] = "Dense"

        xmlwriter.open_tag('Block', attr)
        xmlwriter.write_xml(self.matrix)
        xmlwriter.close_tag()

class CovarianceMatrix(object):
    """:class:`CovarianceMatrix` implements the same-named ARTS datatype."""
    def __init__(self):
        pass

    @classmethod
    def from_xml(cls, xmlelement):
        """Load a covariance matrix from an ARTS XML fiile.

        Returns:
           The loaded covariance matrix as :class:`CovarianceMatrix` object
        """
        obj = cls()

        n_blocks = xmlelement.get("n_blocks")

        obj.blocks = []
        for b in xmlelement.getchildren():
            i = b.get("row_index")
            j = b.get("column_index")
            row_start    = int(b.get("row_start"))
            column_start = int(b.get("column_start"))
            inverse = bool(b.get("is_inverse"))
            print(inverse)
            matrix = b[0].value()
            obj.blocks += [Block(i, j, row_start, column_start, inverse, matrix)]
        return obj

    def write_xml(self, xmlwriter, attr = None):

        if attr is None:
            attr = {}

        attr['n_blocks'] = len(self.blocks)
        xmlwriter.open_tag('CovarianceMatrix', attr)

        for b in self.blocks:
            xmlwriter.write_xml(b)

        xmlwriter.close_tag()

    def to_dense(self):
        """Conversion to dense representation.

        Converts the covariance matrix to a 2-dimensional numpy.ndarray.

        Returns:
            The covariance matrix as dense matrix.

        """
        m = max([b.row_start + b.matrix.shape[0] for b in self.blocks])
        n = max([b.column_start + b.matrix.shape[1] for b in self.blocks])
        mat = np.zeros((m, n))
        for b in self.blocks:
            m0 = b.row_start
            n0 = b.column_start
            dm = b.matrix.shape[0]
            dn = b.matrix.shape[1]
            mat[m0 : m0 + dm, n0 : n0 + dn] = b.matrix.todense()
        return mat

def plot_covariance_matrix(covariance_matrix, ax = None):

    if ax is None:
        ax = plt.gca()

    for b in covariance_matrix.blocks:
        y = np.arange(b.row_start, b.row_start + b.matrix.shape[0] + 1) - 0.5
        x = np.arange(b.column_start, b.column_start + b.matrix.shape[1] + 1) - 0.5
        print(x)
        print(y)
        print(b.matrix.todense())
        ax.pcolormesh(x, y, np.array(b.matrix.todense()))



    m = max([b.row_start + b.matrix.shape[0] for b in covariance_matrix.blocks])
    n = max([b.column_start + b.matrix.shape[1] for b in covariance_matrix.blocks])
    ax.set_xlim([-0.5, n + 0.5])
    ax.set_ylim([m + 0.5, -0.5])










