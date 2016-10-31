# -*- coding: utf-8 -*-

"""Simple deprecation warning."""
from warnings import warn
from typhon.plots.cm import *  # noqa
from typhon.plots.colors import *  # noqa


warn('Module has moved to typhon.plots.', DeprecationWarning, stacklevel=2)
