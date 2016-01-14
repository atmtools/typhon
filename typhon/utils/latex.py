# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Implementation of functions related to LaTeX.

"""

import sys

import numpy as np


def texify_matrix(a, fmt="%f", filename=None, caption=None,
        align='r', delimiter=True):
    """Convert a np.ndarray into a LaTeX table.

    Note:
        the function only works with 2-dimensional arrays. If you want to
        process 1-dimensional arrays you still have to pass them in a
        2D-represenation to clearly determine if it is a row or column vector
        ([1, N] or [N, 1]).

    Args:
        a (np.ndarray): array to convert to LaTeX table.
        fmt (str): format string to specify the number format.
        filename (str): path to outputfile. if no file is given, the output is
            send so stdout.
        caption (str): table caption, if no caption is passed it is left empty.
        align (str): specify the alignment of numbers inside the cells.
        delimiter (bool): toggle the separation of cells through lines.

    Returns:
        the functions returns LaTeX source code either to a specified file or
        stdout.

    Examples:
        >>>> texify_matrix(np.random.randn(30, 8),
                           fmt="%.3f",
                           filename="matrix.tex",
                           caption="This is a test caption.",
                           align='c',
                           delimiter=False,
                          )

    """

    # only two-dimensional arrays are allowed
    if a.ndim != 2:
        raise Exception('Only 2D arrays can be conerted to matrices.')

    # check if alignment is valid
    if align not in ['l', 'c', 'r']:
        raise Exception('Valid alignment values are: "l", "c" and "r"')

    # define row and column delimiters
    if delimiter == True:
        dlm = '|'
        nwl = '\\hline'
    else:
        dlm = nwl = ''

    # if no caption is passed, leave it empty
    if caption is None:
        caption = ''
    elif not isinstance(caption, str):
        raise TypeError('Caption has to be of type string.')

    # if filename is passed, open file and redirect stdout
    if filename is not None:
        out = open(filename, 'w')
    else:
        out = sys.stdout

    # print table header
    out.write('\\begin{table}\n'
              '\\centering\n'
              '\\caption{' + caption + '}\n'
              '\\begin{tabular}{'
              + (dlm + (align + dlm) * a.shape[1])
              + '}' + nwl + '\n' )

    # print each matrix row
    for r in a:
        out.write(fmt % r[0])
        for c in r[1:]:
            out.write(' & '+ fmt % c)
        out.write('\\\\' + nwl + '\n')

    # close table environment
    out.write('\\end{tabular}\n'
                 '\\end{table}\n')

