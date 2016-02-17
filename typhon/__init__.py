# -*- coding: utf-8 -*-

__version__ = '0.2.4'

from . import arts
from . import cm
from . import files
from . import utils


def _runtest():
    """Run all tests."""
    from os.path import dirname
    import nose
    loader = nose.loader.TestLoader(workingDir=dirname(__file__))
    return nose.run(testLoader=loader)

test = _runtest
