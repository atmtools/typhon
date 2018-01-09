import os
import gzip
import bz2
import zipfile
import shutil
import tempfile
from contextlib import contextmanager

__all__ = [
    'compress', 'compress_as', 'decompress',
    'is_compression_format',
]

_known_compressions = {
    'gz': gzip.GzipFile,
    'bz2': bz2.BZ2File,
    'zip': zipfile.ZipFile,
}

try:
    import lzma
except ImportError:  # no lzma
    pass
else:
    _known_compressions['.xz'] = lzma.LZMAFile


@contextmanager
def compress(filename, fmt=None,):
    """Compress a file after writing to it.

    Supported compression formats are: gzip, bzip2, zip, and lzma (Python
    3.3 or newer only).

    This function is tailored for use in a with statement. It uses a context
    manager to automatically create a temporary file to which the data
    can be written. After writing to the file, it compresses and renames it.

    TODO: Preferably, it should be possible to write to the compressed file
    directly instead of copying it.
    TODO: Contains a bug. The original file won't be preserved, it changes
    to the name of the temporary file.

    Args:
        filename: The path and name of the compressed that should be
            created. The filename's extension decides to which format the
            file will be compressed (look at *fmt* for a list).
        fmt: If want to specify the compression format without using the
            filename's extension, use this argument.
            * *zip*: Uses the standard zip library.
            * *bz2*: Uses the bz2 library.
            * *gz*: Uses the GNU zip library.
            * *xz*: Uses the lzma format.

    Yields:
        Generator containing the path to the temporary file.

    Examples:
        >>> with typhon.files.compress('datafile.nc.gz') as file:
        >>>     f = netCDF4.Dataset(file, 'w')
        >>>     #...
    """
    filebase, fileext = os.path.splitext(filename)

    if fmt is None:
        fmt = fileext.lstrip(".")

    if not is_compression_format(fmt):
        yield filename
        return

    # Do not use NamedTemporaryFile as with statement due to this:
    # https://stackoverflow.com/q/15169101/9144990
    tfile = tempfile.NamedTemporaryFile(delete=False)
    try:
        yield tfile.name
        compress_as(tfile.name, fmt, filename, True)
    finally:
        os.remove(tfile.name)


def compress_as(filename, fmt, target=None, keep=True):
    """Compress an existing file.

    Supported compression formats are: gzip, bzip2, zip, and lzma (Python
    3.3 or newer only).

    Args:
        filename: The path and name of the uncompressed file.
        fmt: Decides to which format the file will be compressed.
            * *zip*: Uses the standard zip library.
            * *bz2*: Uses the bz2 library.
            * *gz*: Uses the GNU zip library.
            * *xz*: Uses the lzma format.
        target: The default output filename is *filename.fmt*.
            If you do not like it, you can set another filename here.
        keep: If true, keep the original file after compressing. Otherwise it
            will be deleted. Default is keeping.

    Returns:
        The filename of the newly created file.
    """

    if not is_compression_format(fmt):
        raise ValueError("Unknown compression format '%s'!" % fmt)

    if target is None:
        target = ".".join([filename, fmt])

    # The filename inside the zipped archive
    target_filename = os.path.basename(target)
    target_filename = target_filename.strip("."+fmt)

    # Read datafile in 100 MiB chunks for good performance/memory usage
    chunksize = 100 * 1024 * 1024
    compfile = get_compressor(fmt)
    try:
        if fmt == "zip":
            with compfile(target, 'w') as f_out:
                f_out.write(
                    filename, arcname=target_filename,
                    compress_type=zipfile.ZIP_DEFLATED
                )
        else:
            with open(filename, 'rb') as f_in:
                if fmt == "gz":
                    # Coming from https://stackoverflow.com/a/38020236
                    with open(target, 'wb') as f_out:
                        with compfile(filename, 'wb', fileobj=f_out) as f_out:
                            shutil.copyfileobj(f_in, f_out, length=chunksize)
                elif fmt == "bz2" or fmt == "xz":
                    with compfile(target, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out, length=chunksize)
    except Exception as e:
        raise e
    else:
        if not keep:
            os.unlink(filename)

    return target


@contextmanager
def decompress(filename, tmpdir=None):
    """Temporarily decompress file for reading.

    Returns the full path to the uncompressed temporary file or the original
    filename if it was not compressed.

    Supported compression formats are: gzip, bzip2, zip, and lzma (Python
    3.3 or newer only).

    This function is tailored for use in a with statement. It uses a context
    manager to automatically remove the decompressed file after use.

    Args:
        filename (str): Input file.
        tmpdir (str): Path to directory for temporary storage of the
            uncompressed file. The directory must exist. The default is the
            temporary dir of the system.

    Yields:
        Generator containing the path to the input filename.

    Example:
        >>> tmpdir = '/tmp'
        >>> with typhon.files.decompress('datafile.nc.gz', tmpdir) as file:
        >>>     f = netCDF4.Dataset(file)
        >>>     #...
    """
    if tmpdir is None:
        tmpdir = tempfile.gettempdir()

    filebase, fileext = os.path.splitext(filename)
    filebase = os.path.basename(filebase)
    fmt = fileext.lstrip(".")

    if is_compression_format(fmt):
        tmpfile = tempfile.NamedTemporaryFile(prefix=os.path.join(tmpdir, ''),
                                              delete=False)
        # Read datafile in 100 MiB chunks for good performance/memory usage
        chunksize = 100 * 1024 * 1024
        compfile = get_compressor(fmt)
        try:
            if fmt == 'zip':
                shutil.copyfileobj(compfile(filename, 'r').open(filebase, 'r'),
                                   tmpfile,
                                   chunksize)
            else:
                shutil.copyfileobj(compfile(filename, 'r'),
                                   tmpfile,
                                   chunksize)
            tmpfile.close()
            yield tmpfile.name
        finally:
            os.unlink(tmpfile.name)

    else:
        yield filename


def get_compressor(fmt):
    return _known_compressions[fmt]


def is_compression_format(fmt):
    return fmt in _known_compressions
