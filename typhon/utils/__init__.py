# -*- coding: utf-8 -*-
import time
import ast
import operator

import numpy as np

from . import cache
from . import metaclass

__all__ = ["cache",
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
        >>> t = Timer.start()
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
                int(self.secs // 60), self.secs % 60))


# Next part from http://stackoverflow.com/a/9558001/974555

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
