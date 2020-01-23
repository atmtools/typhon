# -*- coding: utf-8 -*-
from os.path import dirname, join


def get_version_info():
    """Parse version number from module-level ASCII file."""
    return open(join(dirname(__file__), "VERSION")).read().strip()


__version__ = get_version_info()
