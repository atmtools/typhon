# -*- coding: utf-8 -*-
"""
This module provides colormaps to use for the visualisation of meteorological
data.

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
import numpy as np

from ._cmocean import datad as _cmocean_datad
from ._cm import datad as _cm_datad

__all__ = ['mpl_colors']

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
            name + '_r',_rev_cdict(data))
    else:
        cmaps[name] = LinearSegmentedColormap.from_list(name, data)
        cmaps[name + '_r'] = LinearSegmentedColormap.from_list(
            name + '_r', data[::-1])

locals().update(cmaps)

for name, cmap in cmaps.items():
    register_cmap(name, cmap)

__all__ = __all__ + list(cmaps.keys())

def mpl_colors(cmap=None, N=10):
    """Return a list of RGB values.

    Parameters:
        cmap (str): Name of a registered colormap
        N (int): Number of colors to return

    Returns:
        np.array: Array with RGB and alpha values.

    Examples:
        >>> mpl_colors('viridis', 5)
        array([[ 0.267004,  0.004874,  0.329415,  1.      ],
            [ 0.229739,  0.322361,  0.545706,  1.      ],
            [ 0.127568,  0.566949,  0.550556,  1.      ],
            [ 0.369214,  0.788888,  0.382914,  1.      ],
            [ 0.993248,  0.906157,  0.143936,  1.      ]])
    """
    if cmap is None:
        cmap = plt.rcParams['image.cmap']

    return plt.get_cmap(cmap)(np.linspace(0, 1, N))
