# -*- coding: utf-8 -*-

from .version import __version__

try:
    __TYPHON_SETUP__
except:
    __TYPHON_SETUP__ = False

if not __TYPHON_SETUP__:
    from . import arts
    from . import atmosphere
    from . import config
    from . import constants
    from . import files
    from . import geodesy
    from . import geographical
    from . import latex
    from . import math
    from . import nonlte
    from . import oem
    from . import physics
    from . import plots
    from . import spectroscopy
    from . import trees
    from . import utils
    from .environment import environ


    def test():
        """Use pytest to collect and run all tests in typhon.tests."""
        import pytest

        return pytest.main(['--pyargs', 'typhon.tests'])
