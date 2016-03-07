# -*- coding: utf-8 -*-

__version__ = '0.2.7'

# Add revision number for development versions
# _branch = 'release'
_branch = 'dev'
_revision = ''.join(x for x in '$Revision$' if x.isdigit())
if _branch != 'release' and _revision:
    __version__ += '+r' + _revision

from . import arts
from . import files
from . import oem
from . import utils


def _runtest():
    """Run all tests."""
    from os.path import dirname
    from sys import argv
    import nose
    loader = nose.loader.TestLoader(workingDir=dirname(__file__))
    return nose.run(argv=[argv[0]], testLoader=loader)


test = _runtest
