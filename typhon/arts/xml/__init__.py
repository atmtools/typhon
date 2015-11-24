# -*- coding: utf-8 -*-

"""Read and write ARTS XML types

This module provides functionality for reading and writing ARTS XML files.
Most users will only need the `load` function.
"""

from __future__ import absolute_import

import gzip

from . import read
from . import write

__all__ = ['load', 'save']


def save(var, filename, precision='.7e'):
    """Save a variable to an ARTS XML file.

    Args:
        var: Variable to be stored.
        filename (str): Name of output XML file.
        precision (str): Format for output precision.

    """
    with open(filename, mode='w', encoding='UTF-8') as fp:
        fp.write('<?xml version="1.0"?>\n')
        artstag = write.ARTSTag('arts', {'version': 1, 'format': 'ascii'})
        fp.write(artstag.open())
        write.write_xml(var, fp, precision=precision)
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
        return read.parse(fp).getroot().value()
