# -*- coding: utf-8 -*-

from .version import __version__

try:
    __TYPHON_SETUP__
except:
    __TYPHON_SETUP__ = False

if not __TYPHON_SETUP__:
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
