# -*- coding: utf-8 -*-
"""
This module provides colormaps to use for the visualisation of meteorological
data.

"""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

from ._cmocean import datad as _cmocean_datad


datad = _cmocean_datad

cmaps = {}
for (name, data) in datad.items():
    cmaps[name] = LinearSegmentedColormap.from_list(name, data)
    cmaps[name + '_r'] =LinearSegmentedColormap.from_list(name + '_r', data[::-1])

locals().update(cmaps)

for name, cmap in cmaps.items():
    register_cmap(name, cmap)

__all__ = list(cmaps.keys())
