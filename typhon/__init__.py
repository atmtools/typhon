# -*- coding: utf-8 -*-

__version__ = '0.0.4'

from . import arts
from . import files


def _runtest():
    """Run all tests."""
    import nose
    nose.run()

test = _runtest
