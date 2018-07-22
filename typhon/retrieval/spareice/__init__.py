# -*- coding: utf-8 -*-

"""Implementation of SPARE-ICE

This is a reimplementation of the toolkit developed by Gerrit Holl for atmlab.

TODO: Extend documentation and credits.
"""

from .common import *

__all__ = [s for s in dir() if not s.startswith('_')]
