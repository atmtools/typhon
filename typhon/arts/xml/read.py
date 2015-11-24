# -*- coding: utf-8 -*-

"""Read and write ARTS XML types

This module provides functionality for reading and writing ARTS XML files.
Most users will only need the `load` function.
"""

from __future__ import absolute_import

from xml.etree import ElementTree

import numpy as np

from .names import *
from .. import types

__all__ = ['parse']


class _ARTSTypesLoadMultiplexer:
    """Used by the xml.etree.ElementTree to parse ARTS variables.

    Tag names in the XML file are mapped to the corresponding parsing method.

    """

    @staticmethod
    def arts(elem):
        if (elem.attrib['format'] == 'binary'):
            raise RuntimeError(
                'Reading binary ARTS XML files is not yet supported')
        elif (elem.attrib['format'] != 'ascii'):
            raise RuntimeError('Unknown format in <arts> tag: {}'.format(
                elem.attrib['format']))

        return elem[0].value()

    @staticmethod
    def Array(elem):
        arr = [t.value() for t in elem]
        if len(arr) != int(elem.attrib['nelem']):
            raise RuntimeError('Expected {:s} elements in Array, found {:d}'
                               ' elements!'.format(elem.attrib['nelem'],
                                                   len(arr)))
        return arr

    @staticmethod
    def String(elem):
        if elem.text is None:
            return ''
        return elem.text[1:-1]

    SpeciesTag = String

    @staticmethod
    def Index(elem):
        return int(elem.text)

    @staticmethod
    def Numeric(elem):
        return float(elem.text)

    @staticmethod
    def Vector(elem):
        # sep=' ' seems to work even when separated by newlines, see
        # http://stackoverflow.com/q/31882167/974555
        arr = np.fromstring(elem.text, sep=' ')
        if arr.size != int(elem.attrib['nelem']):
            raise RuntimeError('Expected {:s} elements in Vector, found {:d}'
                               ' elements!'.format(elem.attrib['nelem'],
                                                   arr.size))
        return arr

    @staticmethod
    def Matrix(elem):
        flatarr = np.fromstring(elem.text, sep=' ')
        # turn dims around: in ARTS, [10 x 1 x 1] means 10 pages, 1 row, 1 col
        dims = [dim for dim in dimension_names if dim in elem.attrib.keys()][
               ::-1]
        return flatarr.reshape([int(elem.attrib[dim]) for dim in dims])

    Tensor3 = Tensor4 = Tensor5 = Tensor6 = Tensor7 = Matrix


class _ARTSElement(ElementTree.Element):
    """Element with value interpretation."""

    def value(self):
        if hasattr(types, self.tag):
            try:
                return types.classes[self.tag].from_xml(self)
            except AttributeError:
                raise RuntimeError('Type {} exists, but has no XML parsing '
                                   'support.'.format(self.tag))
        else:
            try:
                return getattr(_ARTSTypesLoadMultiplexer, self.tag)(self)
            except AttributeError:
                raise RuntimeError('Unknown ARTS type {}'.format(self.tag))


def parse(source):
    """Parse ArtsXML file from source.

    Args:
        source (str): Filename or file pointer.

    Returns:
        xml.etree.ElementTree: XML Tree of the ARTS data file.

    """
    return ElementTree.parse(source,
                             parser=ElementTree.XMLParser(
                                 target=ElementTree.TreeBuilder(
                                     element_factory=_ARTSElement)))
