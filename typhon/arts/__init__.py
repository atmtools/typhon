# -*- coding: utf-8 -*-

"""This module contains functions to interact with ARTS.
"""

import collections
import os
import shutil
import subprocess

from . import sensor
from . import xml

from .griddedfield import GriddedField4

__all__ = ['xml',
           'sensor',
           'run_arts',
           'atm_fields_compact_get',
           'atm_fields_compact_update',
           ]


def run_arts(controlfile=None, arts='arts', writetxt=False,
             ignore_error=False, **kwargs):
    """Start an ARTS Simulation.

    Parameters:
        controlfile (str): Path to the ARTS controlfile.
        arts (str): Path to the arts executable.
        writetxt (bool): Write stdout and stderr to ASCII files.
        ignore_error (bool): If set to True, erros during the ARTS run do not
            result in an exception (default is False).
        **kwargs: Additional command line arguments passed as keyword.
            See `arts --help` for more details.

    Returns:
        Named tuple containing the fields stdout, stderr and retcode.

    Examples:
        Run a simple ARTS job and set the output directory
        and the report level:

        >>> run_arts('foo.arts', outdir='bar', reporting='020')

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
        if isinstance(arg, bool) and arg is True:
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

    # Throw exception if ARTS run failed.
    if p.returncode != 0 and ignore_error is not True:
        raise Exception('ARTS run failed:\n{}'.format(p.stderr))

    # Store ARTS output in namedtuple.
    arts_out = collections.namedtuple(
        'ARTS_output',
        ['stdout', 'stderr', 'retcode']
    )

    return arts_out(stdout=p.stdout, stderr=p.stderr, retcode=p.returncode)


def atm_fields_compact_get(abs_species, gf4):
    """Extract species from atm_fields_compact.

    Parameters:
        abs_species (list): Species to extract.
        gf4 (GriddedField4): GriddedField4.

    Returns:
        Extracted profiles.

    """
    if not isinstance(gf4, GriddedField4):
        raise Exception(
            'Expected GriddedField4 but got "{}".'.format(type(gf4).__name__))

    if not isinstance(abs_species, list):
        raise Exception('Absorption species have to be passed as list.')

    vmr_field = gf4.data[[gf4.grids[0].index(s) for s in abs_species]]

    return vmr_field


def atm_fields_compact_update(abs_species, gf4, vmr):
    """Update profile for given species.

    Parameters:
        abs_species (string): SpeciesTag.
        gf4 (GriddedField4): GriddedField4.
        vmr (ndarray): New VMR field.

    Returns:
        GriddedField4: Updated atm_fields_compact.

    """
    if not isinstance(gf4, GriddedField4):
        raise Exception(
            'Expected GriddedField4 but got "{}".'.format(type(gf4).__name__))

    if not isinstance(abs_species, list):
        raise Exception('Absorption species have to be passed as list.')

    species_index = [gf4.grids[0].index(s) for s in abs_species]
    gf4.data[species_index, :, :, :] = vmr

    gf4.check_dimension()

    return gf4
