# -*- coding: utf-8 -*-

"""This module provides functionality for reading and writing ARTS XML files.
"""

from __future__ import absolute_import

import gzip
import glob
import os
from os.path import isfile, join, basename, splitext, dirname

from . import read
from . import write

__all__ = [
    'load',
    'save',
    'load_directory',
    'load_indexed',
    'make_binary',
    'make_directory_binary',
]


def save(var, filename, precision='.7e', format='ascii', comment=None,
         parents=False):
    """Save a variable to an ARTS XML file.

    Args:
        var: Variable to be stored.
        filename (str): Name of output XML file.
            If the name ends in .gz, the file is compressed on the fly.
        precision (str): Format for output precision.
        format (str): Output format: 'ascii' (default) or 'binary'.
        comment (str): Comment string included in a tag above data.
        parents (bool): Create missing parent directories.

    Note:
        Python's gzip module is extremely slow in writing. Consider
        compressing files manually after writing them normally.

    Example:
        >>> x = numpy.array([1.,2.,3.])
        >>> typhon.arts.xml.save(x, 'myvector.xml')

    """
    if parents:
        os.makedirs(dirname(filename), exist_ok=True)

    if filename.endswith('.gz'):
        if format != 'ascii':
            raise RuntimeError(
                'For zipped files, the output format must be "ascii"')
        xmlopen = gzip.open
    else:
        xmlopen = open
    with xmlopen(filename, mode='wt', encoding='UTF-8') as fp:
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
    # If file is not found, try the gzipped version.
    if not isfile(filename):
        if not isfile(filename + '.gz'):
            raise FileNotFoundError("No such file: '{}'".format(filename))
        else:
            filename += '.gz'

    if filename.endswith('.gz'):
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


def load_directory(directory, exclude=None):
    """Load all XML files in a given directory.

    Search given directory  for files with `.xml` or `.xml.gz` extension and
    try to load them using :func:`load`.

    Parameters:
        directory (str): Path to the directory.
        exclude (list[str]): Filenames to exlude.

    Returns:
        dictionary: Dictionary, filenames without extension are used as key.

    Example:
        Load all files in foo except for the lookup table in abs_lookup.xml.

        >>> load_directory('foo', exclude=['abs_lookup.xml'])

    """
    if exclude is None:
        exclude = []

    xmlfiles = [f for f in glob.glob(join(directory, '*.xml'))
                if basename(f) not in exclude]

    # Append zipped files to list of XML files. Strip the `.gz` extension to
    # keep the dictionry keys clean, the `load` function finds zipped files
    # anyway.
    xmlfiles.extend(splitext(f)[0]
        for f in glob.glob(join(directory, '*.xml.gz'))
        if basename(f) not in exclude)

    return {splitext(basename(f))[0]: load(f) for f in xmlfiles}


def load_indexed(filename):
    """Load all indexed XML files matching the given filename.

    The function searches all files matching the pattern
    ``<filename>.<file_index>.xml`` or ``<filename>.<file_index>.xml.gz``.

    A list with the loaded file contents is returned. The list indices are
    equivalent to the file indices.

    Parameters:
        filename (str): Filename.

    Returns:
        list: List of file contents.

    Example:
        Load all files matching the pattern ``foo.<file_index>.xml``.

        >>> load_indexed_xml('foo')

    """
    iidx = -2  # Relative position of fileindex in splitted filename.

    # Get all files matching the indexed filename format.
    files = glob.glob('{}.*.xml'.format(filename))

    # If no files are found, try the gzipped version.
    if len(files) == 0:
        files = glob.glob('{}.*.xml.gz'.format(filename))
        iidx = -3  # Correct fileindex position for gzipped files.

    # Extract indices from filenames.
    maxindex = max(int(x.split('.')[iidx]) for x in files)

    # Pre-allocate a list according to the maximum index found.
    ret = (maxindex + 1) * [None]

    # Fill list with file contents (file index matching list index).
    for f in files:
        findex = int(f.split('.')[iidx])
        ret[findex] = load(f)

    return ret


def make_binary(filename, out='', absolute_out=False, parents=True):
    """Loads xml-file at filename and saves it back in binary format

    Parameters:
        filename (str): Filename path.
        out (str): Path to save the binary.  Empty causes overwrite of file.
        absolute_out (bool): If true, then write file to out-path rather than
            to the relative path out.  Does nothing if file is in the working
            folder and out is relative.
        parents (bool): Create missing parent directories.

    Returns:
        str: Path to the created binary file.

    Example:
        Load t_field.xml and save it back as binary it as ./binary/t_field.xml
        and ./binary/t_field.bin

        >>> make_binary('t_field.xml', out='binary')
        'binary/t_field.xml'
    """

    xml_data = load(filename)
    if absolute_out:
        outfile = join(out, basename(filename))
        save(xml_data, outfile, format='binary', parents=parents)
    else:
        outfile = join(dirname(filename), out, basename(filename))
        save(xml_data, outfile, format='binary', parents=parents)

    return outfile


def make_directory_binary(directory, out='', absolute_out=False, parents=True):
    """Loads xml-files in directory and saves them back in binary format

    Parameters:
        directory (str): Directory path.
        out (str): Path to save the binary.
        absolute_out (bool): If true, then write file to out-path rather than
            to the relative path out.  Does nothing if file is in the working
            folder and out is relative.
        parents (bool): Create missing parent directories.

    Returns:
        list[str]: Paths to the created binary files.

    Example:
        Load arts-xml-data/spectroscopy/cia/hitran2011/ and save it back as
        binary it at arts-xml-data-binary/spectroscopy/cia/hitran2011/

        >>> make_directory_binary('arts-xml-data/spectroscopy/cia/hitran2011',
            out='arts-xml-data-binary/spectroscopy/cia/hitran2011',
            absolute_out=True)
        ['arts-xml-data-binary/spectroscopy/cia/hitran2011/hitran_cia2012_adapted.xml']
    """

    directory_of_xmls = load_directory(directory)
    outfiles = []  # Empty list to store output filepaths.

    if absolute_out:
        for entry in directory_of_xmls:
            outfile = join(out, entry + '.xml')
            save(directory_of_xmls[entry],
                 outfile,
                 format='binary',
                 parents=parents)
            outfiles.append(outfile)
    else:
        for entry in directory_of_xmls:
            outfile = join(directory, out, entry + '.xml')
            save(directory_of_xmls[entry],
                 outfile,
                 format='binary',
                 parents=parents)
            outfiles.append(outfile)

    return outfiles
