import os
import gzip
import bz2
import zipfile
import shutil
import tempfile
from contextlib import contextmanager

__all__ = ['decompress']

_known_compressions = {
    '.gz': gzip.GzipFile,
    '.bz2': bz2.BZ2File,
    '.zip': zipfile.ZipFile,
}

try:
    import lzma
except ImportError: # no lzma
    pass
else:
    _known_compressions['.xz'] = lzma.LZMAFile


@contextmanager
def decompress(filename, tmpdir):
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
            uncompressed file. The directory must exist.

    Returns:
        Generator containing the path to the input filename.

    Example:
        >>> tmpdir = '/tmp'
        >>> with typhon.files.decompress('datafile.nc.gz', tmpdir) as file:
        >>>     f = netCDF4.Dataset(file)
        >>>     #...
    """
    filebase, fileext = os.path.splitext(filename)
    filebase = os.path.basename(filebase)

    if _known_compressions.has_key(fileext):
        tmpfile = tempfile.NamedTemporaryFile(prefix=os.path.join(tmpdir, ''),
                                              delete=False)
        # Read datafile in 100 MiB chunks for good performance/memory usage
        chunksize = 100 * 1024 * 1024
        compfile = _known_compressions[fileext]
        try:
            if fileext == '.zip':
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
