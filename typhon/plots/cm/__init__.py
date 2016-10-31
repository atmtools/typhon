# -*- coding: utf-8 -*-
"""
This module provides colormaps to use for the visualisation of meteorological
data.

"""
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap

from ._cmocean import datad as _cmocean_datad
from ._cm import datad as _cm_datad


datad = _cmocean_datad
datad.update(_cm_datad)


def _rev_cdict(cdict):
    """Revert a dictionary containing specs for a LinearSegmentedColormap."""
    rev_cdict = {}
    for k, v in cdict.items():
        rev_cdict[k] = [(1.0 - x, y1, y0) for x, y0, y1 in reversed(v)]
    return rev_cdict

cmaps = {}
for (name, data) in datad.items():
    if 'red' in data:
        cmaps[name] = LinearSegmentedColormap(name, data)
        cmaps[name + '_r'] = LinearSegmentedColormap(
            name + '_r', _rev_cdict(data))
    else:
        cmaps[name] = LinearSegmentedColormap.from_list(name, data)
        cmaps[name + '_r'] = LinearSegmentedColormap.from_list(
            name + '_r', data[::-1])

locals().update(cmaps)

for name, cm in cmaps.items():
    register_cmap(name, cm)

__all__ = list(cmaps.keys())
