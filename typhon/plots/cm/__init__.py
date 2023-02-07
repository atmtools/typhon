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
import matplotlib as mpl

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
        cmaps[name] = mpl.colors.LinearSegmentedColormap(name, data)
        cmaps[name + '_r'] = mpl.colorsLinearSegmentedColormap(
            name + '_r', _rev_cdict(data))
    elif isinstance(data, list):
        cmaps[name] = mpl.colors.ListedColormap(data, name)
        cmaps[name + '_r'] = mpl.colors.ListedColormap(data[::-1], name + '_r')

locals().update(cmaps)

for name, cm in cmaps.items():
    mpl.colormaps.register(cmap=cm, name=name)

__all__ = list(cmaps.keys())
