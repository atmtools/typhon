# -*- coding: utf-8 -*-
import time

import numpy as np

from . import latex
from . import cache
from . import metaclass

__all__ = ["latex",
           "cache",
           "metaclass",
           "extract_block_diag",
           "Timer",
           ]


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


class Timer(object):
    """Provide a simple time profiling utility.

    Timer class adapted from blog entry [0].

    [0] https://www.huyng.com/posts/python-performance-analysis

    Parameters:
        verbose: Print results after stopping the timer.

    Examples:
        Timer in with block:

        >>> import time
        >>> with Timer():
        ...     time.sleep(1)
        elapsed time: 1.001s

        Timer object:

        >>> import time
        >>> r = Timer.start()
        >>> time.sleep(1)
        >>> t.stop()
        >>> print(t.secs)
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
                int(self.secs//60), self.secs % 60))
