# -*- coding: utf-8 -*-

"""This module contains convenience functions for any purposes.
"""

import ast
import contextlib
import operator
import os
import shutil
import subprocess
import time
from warnings import warn
from functools import (partial, wraps)

import xarray
import numpy as np

from . import cache
from . import metaclass

__all__ = [
    "cache",
    "deprecated",
    "metaclass",
    "extract_block_diag",
    "Timer",
    "path_append",
    "path_prepend",
    "path_remove",
    "image2mpeg",
]


def deprecated(func=None, message=None):
    """Decorator which can be used to mark functions as deprecated.

    Examples:
        Calling ``foo()`` will raise a ``DeprecationWarning``.

        >>> @deprecated
        ... def foo(x):
        ...     pass

        Display message with additional information:

        >>> @deprecated(message='Additional information message.')
        ... def foo(x):
        ...     pass
    """
    # Return partial when no arguments are passed.
    # This allows a plain call of the decorator.
    if func is None:
        return partial(deprecated, message=message)

    # Build warning message (with optional information).
    msg = 'Call to deprecated function {name}.'
    if message is not None:
        msg = '\n'.join((msg, message))

    # Wrapper that prints the warning before calling the deprecated function.
    @wraps(func)
    def wrapper(*args, **kwargs):
        warn(msg.format(name=func.__name__, message=message),
             category=DeprecationWarning,
             stacklevel=2,  # This ensures more useful stack information.
             )
        return func(*args, **kwargs)
    return wrapper


def extract_block_diag(M, n):
    """Extract diagonal blocks from square Matrix.

    Args:
        M (np.array): Square matrix.
        n (int): Number of blocks to extract.

    Example:
        >>> foo = np.array([[ 1.,  1.,  0.,  0.],
        ... [ 1.,  1.,  0.,  0.],
        ... [ 0.,  0.,  2.,  2.],
        ... [ 0.,  0.,  2.,  2.]])
        >>> extract_block_diag(foo, 2)
        [array([[ 1.,  1.],
                [ 1.,  1.]]), array([[ 2.,  2.],
                [ 2.,  2.]])]

    """
    return [np.split(m, n, axis=1)[i] for i, m in enumerate(np.split(M, n))]


class Timer(contextlib.ContextDecorator):
    """Provide a simple time profiling utility.

    Timer class adapted from blog entry [0].

    [0] https://www.huyng.com/posts/python-performance-analysis

    Parameters:
        verbose: Print results after stopping the timer.

    Examples:
        Timer in with statement:

        >>> import time
        >>> with Timer():
        ...     time.sleep(1)
        elapsed time: 1.001s

        Timer as object:

        >>> import time
        >>> t = Timer().start()
        >>> time.sleep(1)
        >>> t.stop()
        elapsed time: 1.001s

        As function decorator:

        >>> @Timer()
        ... def own_function(s):
        ...     import time
        ...     time.sleep(s)
        >>> own_function(1)
        elapsed time: 1.001s

    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start timer."""
        self.starttime = time.time()
        return self

    def stop(self):
        """Stop timer."""
        self.endtime = time.time()
        self.secs = self.endtime - self.starttime
        self.msecs = self.secs * 1000
        if self.verbose:
            print('elapsed time: {:d}m{:.3f}s'.format(
                int(self.secs // 60), self.secs % 60))


# This code, or a previous version thereof, was posted by user 'J. F.
# Sebastian' on http://stackoverflow.com/a/9558001/974555 on 2012-03-04
# and is licensed under CC-BY-SA 3.0.  This notice may not be removed.
operators = {ast.Add: operator.add,
             ast.Sub: operator.sub,
             ast.Mult: operator.mul,
             ast.Div: operator.truediv,
             ast.Pow: operator.pow,
             ast.BitXor: operator.xor,
             ast.USub: operator.neg}


def safe_eval(expr):
    """Safely evaluate string that may contain basic arithmetic
    """

    return _safe_eval_node(ast.parse(expr, mode="eval").body)


def _safe_eval_node(node):
    if isinstance(node, ast.Num):  # <number>
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return operators[type(node.op)](
            _safe_eval_node(node.left), _safe_eval_node(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](_safe_eval_node(node.operand))
    else:
        raise TypeError(node)
# End of snippet derived from http://stackoverflow.com/a/9558001/974555

def path_append(dirname, path='PATH'):
    """Append a directory to environment path variable.

    Append entries to colon-separated variables (e.g. the system path).
    If the entry is already in the list, it is moved to the end.
    A path variable is set, if not existing at function call.

    Parameters:
        dirname (str): Directory to add to the path.
        path (str): Name of the path variable to append to.
            Defaults to the system path 'PATH'.
    """
    if path in os.environ:
        dir_list = os.environ[path].split(os.pathsep)
        if dirname in dir_list:
            dir_list.remove(dirname)
        dir_list.append(dirname)
        os.environ[path] = os.pathsep.join(dir_list)
    else:
        os.environ[path] = dirname


def path_prepend(dirname, path='PATH'):
    """Prepend a directory to environment path variable.

    Append entries to colon-separated variables (e.g. the system path).
    If the entry is already in the list, it is moved to the end.
    A path variable is set, if not existing at function call.

    Parameters:
        dirname (str): Directory to add to the path.
        path (str): Name of the path variable to append to.
            Defaults to the system path 'PATH'.
    """
    if path in os.environ:
        dir_list = os.environ[path].split(os.pathsep)
        if dirname in dir_list:
            dir_list.remove(dirname)
        dir_list.insert(0, dirname)
        os.environ[path] = os.pathsep.join(dir_list)
    else:
        os.environ[path] = dirname


def path_remove(dirname, path='PATH'):
    """Remove a directory from environment path variable.

    Remove entries from colon-separated variables (e.g. the system path).
    If the path variable is not set, nothing is done.

    Parameters:
        dirname (str): Directory to add to the path.
        path (str): Name of the path variable to append to.
            Defaults to the system path 'PATH'.
    """
    if path in os.environ:
        dir_list = os.environ[path].split(os.pathsep)
        dir_list.remove(dirname)
        os.environ[path] = os.pathsep.join(dir_list)


def get_time_dimensions(ds):
    """From a xarray dataset or dataarray, get dimensions corresponding to time coordinates

    """

    return {k for (k, v) in ds.coords.items() if k in ds.dims and v.dtype.kind == "M"}

def get_time_coordinates(ds):
    """From a xarray dataset or dataarray, get coordinates with at least 1 time dimension

    """

    time_dims = get_time_dimensions(ds)
    return {k for (k, v) in ds.coords.items() if set(v.dims)&time_dims}


def concat_each_time_coordinate(*datasets):
    """Concatenate xarray datasets along each time coordinate

    Given two or more xarray datasets, concatenate seperately data
    variables with different time coordinates.  For example, one might
    have dimensions 'scanline' and 'calibration_cycle' that are each along
    time coordinates.  Data variables may have dimension either scanline
    or calibration_cycle or neither, but not both.  Both correspond to
    coordinates a datetime index.  Ordinary xarray.concat along either
    dimension will broadcast the other one in a way similar to repmat,
    thus exploding memory usage (test case for one FCDR HIRS granule: 89
    MB to 81 GB).  Instead, here, for each data variable, we will
    concatenate only along at most one time coordinate.

    Arguments:

        *datasets: xarray.Dataset objects to be concatenated
    """

    time_coords = get_time_coordinates(datasets[0])
    time_dims = get_time_dimensions(datasets[0])
    # ensure each data-variable has zero or one of those time coordinates
    # as dimensions
    for ds in datasets:
        if not all([len(set(v.dims) & time_coords) <= 1
                    for (k, v) in ds.data_vars.items()]):
            raise ValueError("Found vars with multiple time coords")

    new_sizes = {k: sum(g.dims[k] for g in datasets)
                     if k in time_coords
                     else datasets[0].dims[k]
                 for k in datasets[0].dims.keys()}
    # note data vars per time coordinate
    time_vars = {k: (set(v.dims)&time_coords).pop() for (k, v) in datasets[0].items() if set(v.dims)&time_coords}
    time_vars_per_time_dim = {k: {vn for (vn, dn) in time_vars.items() if dn==k} for k in time_coords}
    untimed_vars = datasets[0].data_vars.keys() - time_vars.keys()

    # allocate new
    new = xarray.Dataset(
        {k: (v.dims,
             np.zeros(shape=[new_sizes[d] for d in v.dims],
                         dtype=v.dtype))
                for (k, v) in datasets[0].data_vars.items()})
    # coordinates cannot be set in the same way so need to be allocated
    # separately
    new_coords = {k: xarray.DataArray(
                        np.zeros(shape=[new_sizes[d] for d in v.dims],
                                 dtype=datasets[0][k].dtype),
                        dims=v.dims)
                    for (k, v) in datasets[0].coords.items()}

    # copy over untimed vars
    for v in untimed_vars:
        new[v].values[...] = datasets[0][v].values

    # and untimed coords
    for c in datasets[0].coords.keys() - time_coords:
        new_coords[c][...] = datasets[0].coords[c]

    # keep track of progress per time dimension
    n_per_dim = dict.fromkeys(time_coords, 0)
    # copy timed vars dataset by dataset
    for ds in datasets:
        for (v, timedim) in time_vars.items():
            ncur = n_per_dim[timedim]
            nnew_cur = ds.dims[timedim]
            slc = {dim: slice(ncur, ncur+nnew_cur)
                        if dim==timedim else slice(None)
                   for dim in ds[v].dims}
            if v in time_coords: # time coordinate
                new_coords[v][slc] = ds[v]
            else:
                new[v].loc[slc] = ds[v]
        for timedim in time_dims:
            n_per_dim[timedim] += ds.dims[timedim]
    # copy attributes
    new.attrs.update(**datasets[0].attrs)
    for k in new.keys():
        new[k].attrs.update(**datasets[0][k].attrs)
    return new.assign_coords(**new_coords)


def image2mpeg(glob, outfile, framerate=12):
    """Combine image files to a video using ``ffmpeg``.

    Notes:
        The function is tested for ``ffmpeg`` versions 2.8.6 and 3.2.2.

    Parameters:
        glob (str): Glob pattern for input files.
        outfile (str): Path to output file.
            The file fileformat is determined by the extension.
        framerate (int or str): Number of frames per second.

    Raises:
        Exception: The function raises an exception if the
            underlying ``ffmpeg`` process returns a non-zero exit code.

    Example:
        >>> image2mpeg('foo_*.png', 'foo.mp4')
    """
    if not shutil.which('ffmpeg'):
        raise Exception('``ffmpeg`` not found.')

    p = subprocess.run(
        ['ffmpeg',
         '-framerate', str(framerate),
         '-pattern_type', 'glob', '-i', glob,
         '-s:v', '1280x720',
         '-c:v', 'libx264',
         '-profile:v', 'high',
         '-crf', '20',
         '-pix_fmt', 'yuv420p',
         '-y', outfile
         ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
        )

    # If the subprocess fails, raise exception including error message.
    if p.returncode != 0:
        raise Exception(p.stderr)
