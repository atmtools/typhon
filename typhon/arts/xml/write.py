# -*- coding: utf-8 -*-

"""Write ARTS XML types

This package contains the internal implementation for writing ARTS XML files.
"""

from __future__ import absolute_import

import numpy as np

from .names import *

__all__ = ['write_xml', 'ARTSTag']


class ARTSTag:
    """Represents an XML tag and its attributes.

    This class is used to create an XML tag including attributes.
    A string of the opening and closing tag can be generated.

    The attributes are passed to the constructor as a dictionary.
    """

    def __init__(self, name='', attr=None):
        """Construct an ARTSTag

        Args:
            name (str): Tag name corresponding to the ARTS type.
            attr (dict): XML tag attributes.

        """
        if attr is None:
            attr = {}
        self.name = name
        self.attributes = attr

    @property
    def name(self):
        """str: XML tag name."""
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def attributes(self):
        """dict: XML tag attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, attr):
        if attr is not None and type(attr) is not dict:
            raise TypeError('Attributes must be a dictionary')

        self._attributes = attr

    def open(self, newline=True):
        """Returns the opening tag as a string.

        Args:
            newline (bool): Put newline after tag.

        Returns:
            str: XML opening tag.
        """
        ret = '<{}{}>'.format(self.name,
                              ''.join([' {}="{}"'.format(a, v) for a, v in
                                       self.attributes.items()]))
        if newline:
            ret += '\n'

        return ret

    def close(self):
        """Returns the closing tag as a string.

        Returns:
            str: XML closing tag.
        """
        return '</{}>\n'.format(self.name)


def get_arts_typename(var):
    """Returns the ARTS type name for this variable.

    Args:
        var: Variable to get the ARTS type name for.

    Returns:
        str: ARTS type name.

    """
    return basic_types[type(var).__name__]


def write_xml(var, fp, precision='.7e', attr=None, binaryfp=None):
    """Write a variable as XML.

    Writing basic matpack types is implemented here. Custom types (e.g.
    GriddedFields) must implement a class member function called
    'write_xml'.

    Tuples and list are mapped to ARTS Array types.

    Args:
        var: Variable to be written as XML.
        fp: Text IO output stream.
        precision: Format string.
        binaryfp: Binary IO output stream.
        attr (dict): Additional attributes of the variable.

    """
    if hasattr(var, 'write_xml'):
        var.write_xml(fp, precision, attr, binaryfp)
    elif type(var) is np.ndarray:
        write_ndarray(var, fp, precision, attr, binaryfp)
    elif type(var) is int:
        write_basic_type(var, fp, 'Index', attr=attr, binaryfp=binaryfp)
    elif type(var) is float:
        write_basic_type(var, fp, 'Numeric', fmt='{:' + precision + '}',
                         attr=attr, binaryfp=binaryfp)
    elif type(var) is str:
        write_basic_type('"' + var + '"', fp, 'String', attr=attr)
    elif type(var) in (list, tuple):
        try:
            arraytype = get_arts_typename(var[0])
        except IndexError:
            raise RuntimeError('Array must have at least one element.')

        if attr is None:
            attr = {}
        else:
            attr = attr.copy()
        attr['nelem'] = len(var)
        attr['type'] = arraytype
        at = ARTSTag('Array', attr=attr)

        fp.write(at.open())
        for i, v in enumerate(var):
            if get_arts_typename(v) != arraytype:
                raise RuntimeError(
                    'All array elements must have the same type. '
                    "Array type is '{}', but element {} has type '{}'".format(
                        arraytype, i, get_arts_typename(v)))
            write_xml(v, fp, precision, binaryfp=binaryfp)
        fp.write(at.close())
    else:
        raise TypeError(
            "Can't map '{}' to any ARTS type.".format(type(var).__name__))


def write_basic_type(var, fp, name, fmt='{}', attr=None, binaryfp=None):
    """Write a basic ARTS type as XML.

    Args:
        var: See `write_xml`.
        fp: See `write_xml`.
        name: Variable type name.
        fmt (str): Output format string.
        attr: See `write_xml`.
        binaryfp: See `write_xml`.

    """
    at = ARTSTag(name, attr)
    fp.write(at.open(False))
    fp.write(fmt.format(var))
    fp.write(at.close())


def write_ndarray(var, fp, precision, attr=None, binaryfp=None):
    """Convert ndarray to ARTS XML representation.

    For arguments see `write_xml`.

    """
    ndim = var.ndim
    artstag = ARTSTag(tensor_names[ndim - 1], attr)
    # Vector
    if ndim == 1:
        artstag.attributes['nelem'] = var.shape[0]
        fp.write(artstag.open())
        fmt = "%" + precision
        for i in var:
            fp.write(fmt % i + '\n')
        fp.write(artstag.close())
    # Matrix and Tensors
    elif ndim <= len(dimension_names):
        for i in range(0, ndim):
            artstag.attributes[dimension_names[i]] = var.shape[ndim - 1 - i]

        fp.write(artstag.open())

        # Reshape for row-based linebreaks in XML file
        if np.prod(var.shape) != 0:
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
