# -*- coding: utf-8 -*-

"""This module contains functions to interact with ARTS.
"""

import collections
import os
import shutil
import subprocess
import sys

from . import xml

__all__ = ['xml',
           'run_arts',
           ]


def run_arts(controlfile=None, arts='arts', writetxt=False, **kwargs):
    """Start an ARTS Simulation.

    Parameters:
        controlfile (str): Path to the ARTS controlfile.
        arts (str): Path to the arts executable.
        writetxt (bool): Write stdout and stderr to ASCII files.
        **kwargs: Additional command line arguments passed as keyword.
            See `arts --help` for more details.

    Returns:
        Named tuple containing the fields stdout, stderr and retcode.

    Examples:
        Run a simple ARTS job and set the output directory
        and the report level:

        >>> run_arts('foo.arts', outdir='bar', reporting=020)

        If a keyword is set to True it is added as flag.
        Show the ARTS help message:

        >>> run_arts(help=True)

    """
    if not shutil.which(arts):
        raise Exception('ARTS executable not found at: {}'.format(arts))

    if controlfile is None:
        controlfile = ''
    elif not os.path.exists(controlfile):
        raise Exception('Controlfile not found at: {}'.format(controlfile))

    opts = []
    for kw, arg in kwargs.items():
        if type(arg) is bool and arg is True:
            if len(kw) == 1:
                opts.append('-{}'.format(kw))
            else:
                opts.append('--{}'.format(kw))
        elif len(kw) == 1:
            opts.append('-{}{}'.format(kw, arg))
        else:
            opts.append('--{}={}'.format(kw, arg))

    # Run ARTS job and redirect output.
    p = subprocess.run([arts, *opts, controlfile],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       universal_newlines=True
                       )

    # Write ARTS output and error to ASCII file.
    if writetxt:
        if controlfile.endswith('.arts'):
            outfile = controlfile.replace('.arts', '.out')
            errfile = controlfile.replace('.arts', '.err')
        else:
            outfile = 'arts.out'
            errfile = 'arts.err'

        for key in ['outdir', 'o']:
            if key in kwargs:
                outfile = os.path.join(kwargs[key], outfile)
                errfile = os.path.join(kwargs[key], errfile)

        with open(outfile, 'w') as out, open(errfile, 'w') as err:
            out.write(p.stdout)
            err.write(p.stderr)

    # Store ARTS output in namedtuple.
    ARTS_out = collections.namedtuple(
        'ARTS_output',
        ['stdout', 'stderr', 'retcode']
        )

    return ARTS_out(stdout=p.stdout, stderr=p.stderr, retcode=p.returncode)
