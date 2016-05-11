# -*- coding: utf-8 -*-

"""This module provides functionality for reading and writing ARTS XML files.
"""

from __future__ import absolute_import

import gzip
from os.path import isfile

from . import read
from . import write

__all__ = ['load', 'save']


def save(var, filename, precision='.7e', format='ascii', comment=None):
    """Save a variable to an ARTS XML file.

    Args:
        var: Variable to be stored.
        filename (str): Name of output XML file.
            If the name ends in .gz, the file is compressed on the fly.
        precision (str): Format for output precision.
        format (str): Output format: 'ascii' (default) or 'binary'.
        comment (str): Comment string included in a tag above data.

    Note:
        Python's gzip module is extremely slow in writing. Consider
        compressing files manually after writing them normally.

    Example:
        >>> x = numpy.array([1.,2.,3.])
        >>> typhon.arts.xml.save(x, 'myvector.xml')

    """
    if filename[:-3] == '.gz':
        if format != 'ascii':
            raise RuntimeError(
                'For zipped files, the output format must be "ascii"')
        xmlopen = gzip.open
    else:
        xmlopen = open
    with xmlopen(filename, mode='w', encoding='UTF-8') as fp:
        if format == 'binary':
            with open(filename + '.bin', mode='wb') as binaryfp:
                axw = write.ARTSXMLWriter(fp, precision=precision,
                                          binaryfp=binaryfp)
                axw.write_header()
                if comment is not None:
                    axw.write_comment(comment)
                axw.write_xml(var)
                axw.write_footer()
        elif format == 'ascii':
            axw = write.ARTSXMLWriter(fp, precision=precision)
            axw.write_header()
            if comment is not None:
                axw.write_comment(comment)
            axw.write_xml(var)
            axw.write_footer()
        else:
            raise RuntimeError('Unknown output format "{}".'.format(format))


def load(filename):
    """Load a variable from an ARTS XML file.

    The input file can be either a plain or gzipped XML file

    Args:
        filename (str): Name of ARTS XML file.

    Returns:
        Data from the XML file. Type depends on data in file.

    Example:
        >>> typhon.arts.xml.load('tests/reference/matrix.xml')
        array([[ 0.,  1.],
               [ 2.,  3.]])

    """
    if filename[-3:] == '.gz':
        xmlopen = gzip.open
    else:
        xmlopen = open

    binaryfilename = filename + '.bin'
    with xmlopen(filename, 'rb') as fp:
        if isfile(binaryfilename):
            with open(binaryfilename, 'rb',) as binaryfp:
                return read.parse(fp, binaryfp).getroot().value()
        else:
            return read.parse(fp).getroot().value()
