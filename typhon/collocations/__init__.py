"""
This module contains classes to find collocations between datasets. They are
inspired by the implemented CollocatedDataset classes in atmlab written by
Gerrit Holl.

TODO: I would like to have this package as typhon.collocations.

Created by John Mrziglod, June 2017
"""

from .common import *  # noqa

__all__ = [s for s in dir() if not s.startswith('_')]
