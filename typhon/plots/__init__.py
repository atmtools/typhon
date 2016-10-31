# -*- coding: utf-8 -*-

"""This module provides functions related to plot or to plot data.
"""

from typhon.plots import cm  # noqa
from typhon.plots.colors import *  # noqa
from typhon.plots.common import *  # noqa
from typhon.plots.plots import *  # noqa

__all__ = [s for s in dir() if not s.startswith('_')]
