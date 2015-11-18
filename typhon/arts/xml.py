# -*- coding: utf-8 -*-

"""Read and write ARTS XML types

This module provides functionality for reading and writing ARTS XML files.
Most users will only need the `load` function.
"""

from __future__ import absolute_import
from xml.etree import ElementTree
import gzip
import numpy as np

__all__ = ['load', 'save', 'parse', 'write_arts_xml', 'ARTSTag']

# Source: ARTS developer guide, section 3.4
_dim_names = [
    'ncols', 'nrows', 'npages', 'nbooks', 'nshelves', 'nvitrines', 'nlibraries']

_tensor_names = [
    'Vector', 'Matrix', 'Tensor3', 'Tensor4', 'Tensor5', 'Tensor6', 'Tensor7']

_arts_types = {
    'tuple': 'Array',
    'list': 'Array',
    'int': 'Index',
    'float': 'Numeric',
    'str': 'String',
}


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
        dims = [dim for dim in _dim_names if dim in elem.attrib.keys()][::-1]
        return flatarr.reshape([int(elem.attrib[dim]) for dim in dims])

    Tensor3 = Tensor4 = Tensor5 = Tensor6 = Tensor7 = Matrix


class _ARTSElement(ElementTree.Element):
    """Element with value interpretation."""

    def value(self):
        try:
            return getattr(_ARTSTypesLoadMultiplexer, self.tag)(self)
        except AttributeError:
            raise RuntimeError('Unknown ARTS type {}'.format(self.tag))


class ARTSTag:
    """Represents an XML tag and its attributes.

    This class is used to create an XML tag including attributes.
    A string of the opening and closing tag can be generated.

    The attributes are passed to the constructor as a dictionary.
    """

    def __init__(self, name='', attr=None):
        if attr is None:
            attr = {}
        self.name = name
        self.attributes = attr

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attr):
        if attr is not None and type(attr) is not dict:
            raise TypeError('Attributes must be a dictionary')

        self._attributes = attr

    def open(self, newline=True):
        """Returns the opening tag as a string."""
        ret = '<{}{}>'.format(self.name,
                              ''.join([' {}="{}"'.format(a, v) for a, v in
                                       self.attributes.items()]))
        if newline:
            ret += '\n'

        return ret

    def close(self):
        """Returns the closing tag as a string."""
        return '</{}>\n'.format(self.name)


def get_arts_typename(var):
    """Returns the ARTS type name for this variable.

    Args:
        var: Variable to get the ARTS type name for.

    Returns:
        str: ARTS type name.
    """
    return _arts_types[type(var).__name__]


def write_arts_xml(var, fp, precision='.7e', binaryfp=None):
    """Write a variable as XML.

    Writing basic matpack types is implemented here. Custom types (e.g.
    GriddedFields) must implement a class member function called
    'write_arts_xml'.

    Tuples and list are mapped to ARTS Array types.

    Args:
        var: Variable to be written as XML.
        fp: Text IO output stream.
        precision: Format string.
        binaryfp: Binary IO output stream.

    """
    if hasattr(var, 'write_as_arts_xml'):
        var.write_arts_xml(var, fp, precision, binaryfp)
    elif type(var) is np.ndarray:
        write_ndarray(var, fp, precision, binaryfp)
    elif type(var) is int:
        write_basic_type(var, fp, 'Index', binaryfp=binaryfp)
    elif type(var) is float:
        write_basic_type(var, fp, 'Numeric', fmt='{:' + precision + '}',
                         binaryfp=binaryfp)
    elif type(var) is str:
        write_basic_type('"' + var + '"', fp, 'String')
    elif type(var) in (list, tuple):
        try:
            arraytype = get_arts_typename(var[0])
        except IndexError:
            raise RuntimeError('Array must have at least one element.')

        at = ARTSTag('Array', attr={'nelem': len(var),
                                    'type': arraytype})

        fp.write(at.open())
        for i, v in enumerate(var):
            if get_arts_typename(v) != arraytype:
                raise RuntimeError(
                    'All array elements must have the same type. '
                    "Array type is '{}', but element {} has type '{}'".format(
                        arraytype, i, get_arts_typename(v)))
            write_arts_xml(v, fp, precision, binaryfp)
        fp.write(at.close())
    else:
        raise TypeError(
            "Can't map '{}' to any ARTS type.".format(type(var).__name__))


def write_basic_type(var, fp, name, fmt='{}', binaryfp=None):
    """Write a basic ARTS type as XML."""
    at = ARTSTag(name)
    fp.write(at.open(False))
    fp.write(fmt.format(var))
    fp.write(at.close())


def write_ndarray(var, fp, precision, binaryfp):
    """Convert ndarray to ARTS XML representation.

    For arguments see `write_arts_xml`.

    """
    ndim = var.ndim
    artstag = ARTSTag(_tensor_names[ndim - 1])
    # Vector
    if ndim == 1:
        artstag.attributes['nelem'] = var.shape[0]
        fp.write(artstag.open())
        fmt = "%" + precision
        for i in var:
            fp.write(fmt % i + '\n')
        fp.write(artstag.close())
    # Matrix and Tensors
    elif ndim <= len(_dim_names):
        for i in range(0, ndim):
            artstag.attributes[_dim_names[i]] = var.shape[ndim - 1 - i]

        fp.write(artstag.open())

        # Reshape for row-based linebreaks in XML file
        if (ndim > 2):
            var = var.reshape(-1, var.shape[-1])

        fmt = ' '.join(['%' + precision, ] * var.shape[1])

        for i in var:
            fp.write((fmt % tuple(i) + '\n'))
        fp.write(artstag.close())
    else:
        raise RuntimeError(
            'Dimensionality ({}) of ndarray too large for '
            'conversion to ARTS XML'.format(ndim))


def save(var, filename, precision='.7e'):
    """Save a variable to an ARTS XML file.

    Args:
        var: Variable to be stored.
        filename (str): Name of output XML file.
        precision (str): Format for output precision.

    """
    with open(filename, mode='w', encoding='UTF-8') as fp:
        fp.write('<?xml version="1.0"?>\n')
        artstag = ARTSTag('arts', {'version': 1, 'format': 'ascii'})
        fp.write(artstag.open())
        write_arts_xml(var, fp, precision=precision)
        fp.write(artstag.close())


def load(filename):
    """Load a variable from an ARTS XML file.

    The input file can be either a plain or gzipped XML file

    Args:
        filename (str): Name of ARTS XML file.

    Returns:
        Data from the XML file. Type depends on data in file.

    """
    if filename[-3:] == '.gz':
        xmlopen = gzip.open
    else:
        xmlopen = open

    with xmlopen(filename, 'rb') as fp:
        return parse(fp).getroot().value()


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
