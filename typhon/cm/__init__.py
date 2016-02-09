# -*- coding: utf-8 -*-
"""
This module provides colormaps to use for the visualisation of meteorological
data.

"""

from matplotlib.colors import ListedColormap

from ._cmocean import datad as _cmocean_datad


datad = _cmocean_datad

cmaps = {}
for (name, data) in datad.items():
    cmaps[name] = ListedColormap(data, name=name)
    cmaps[name + '_r'] = ListedColormap(data[::-1], name=name + '_r')

locals().update(cmaps)

__all__ = list(cmaps.keys())
