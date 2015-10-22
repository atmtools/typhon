# -*- coding: utf-8 -*-

"""Read and write Arts XML types

This module contains classes for reading and writing Arts XML types.
Most users will only need the `load` function.
"""

from __future__ import absolute_import
import xml.etree.ElementTree
import numpy as np

__all__ = ['load', 'parse']

# Source: ARTS developer guide, section 3.4
_dim_names = ['ncols', 'nrows', 'npages', 'nbooks', 'nshelves',
              'nvitrines', 'nlibraries']


class ARTSTypesLoadMultiplexer:
    def arts(self, elem):
        if (elem.attrib['format'] == 'binary'):
            raise RuntimeError(
                'Reading binary ARTS XML files is not yet supported')
        elif (elem.attrib['format'] != 'ascii'):
            raise RuntimeError('Unknown format in <arts> tag: {}'.format(
                elem.attrib['format']))

        return elem[0].value()

    def Array(self, elem):
        arr = [t.value() for t in elem]
        if len(arr) != int(elem.attrib['nelem']):
            raise RuntimeError('Expected {:s} elements in Array, found {:d}'
                               ' elements!'.format(elem.attrib['nelem'],
                                                   len(arr)))
        return arr

    def String(self, elem):
        return elem.text

    SpeciesTag = String

    def Index(self, elem):
        return int(elem.text)

    def Numeric(self, elem):
        return float(elem.text)

    def Vector(self, elem):
        # sep=' ' seems to work even when separated by newlines, see
        # http://stackoverflow.com/q/31882167/974555
        arr = np.fromstring(elem.text, sep=' ')
        if arr.size != int(elem.attrib['nelem']):
            raise RuntimeError('Expected {:s} elements in Vector, found {:d}'
                               ' elements!'.format(elem.attrib['nelem'],
                                                   arr.size))
        return arr

    def Matrix(self, elem):
        flatarr = np.fromstring(elem.text, sep=' ')
        # turn dims around: in ARTS, [10 x 1 x 1] means 10 pages, 1 row, 1 col
        dims = [dim for dim in _dim_names if dim in elem.attrib.keys()][::-1]
        return flatarr.reshape([int(elem.attrib[dim]) for dim in dims])

    Tensor3 = Tensor4 = Tensor5 = Tensor6 = Tensor7 = Matrix


_load_multiplexer = ARTSTypesLoadMultiplexer()


class ArtsElement(xml.etree.ElementTree.Element):
    """Element with value interpretation.
    """

    def value(self):
        return getattr(_load_multiplexer, self.tag)(self)


def load(file):
    """Load data from an ARTS XML file.

    The input file can be either a plain or gzipped XML file.

    Args:
        file (str): Name of ARTS XML file.

    Returns:
        Data from the XML file.

    """
    # TODO(OLE): Add support for gzipped files.
    return parse(file).getroot().value()


def parse(source):
    """Parse ArtsXML file from source.

    Args:
        source (str): Filename or file pointer.

    Returns:
        xml.etree.ElementTree.ElementTree: XML Tree of the ARTS data file.
    """
    return xml.etree.ElementTree.parse(source,
                                       parser=xml.etree.ElementTree.XMLParser(
                                           target=xml.etree.ElementTree.TreeBuilder(
                                               element_factory=ArtsElement)))
