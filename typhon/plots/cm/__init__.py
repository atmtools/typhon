# -*- coding: utf-8 -*-
"""
This module provides colormaps to use for the visualisation of meteorological
data.  It heavily bases on the cmocean_ package developed by Kristen Thyng.
Most colormaps are directly inherited and renamed for meteorological
applications.

The colormaps are registered in matplotlib after importing typhon:

    >>> import typhon
    >>> plt.get_cmap('difference')

.. _cmocean: http://matplotlib.org/cmocean/
"""
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap)
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
    if isinstance(data, dict):
        cmaps[name] = LinearSegmentedColormap(name, data)
        cmaps[name + '_r'] = LinearSegmentedColormap(
            name + '_r', _rev_cdict(data))
    elif isinstance(data, list):
        cmaps[name] = ListedColormap(data, name)
        cmaps[name + '_r'] = ListedColormap(data[::-1], name + '_r')

locals().update(cmaps)

for name, cm in cmaps.items():
    register_cmap(name, cm)

__all__ = list(cmaps.keys())
