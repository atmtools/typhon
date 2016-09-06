# -*- coding: utf-8 -*-
"""
This module provides colormaps to use for the visualisation of meteorological
data.

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import register_cmap
import numpy as np
import time

from ._cmocean import datad as _cmocean_datad
from ._cm import datad as _cm_datad

__all__ = ['mpl_colors',
           'cmap2txt',
           'cmap2cpt',
           ]

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

__all__ = __all__ + list(cmaps.keys())


def mpl_colors(cmap=None, n=10):
    """Return a list of RGB values.

    Parameters:
        cmap (str): Name of a registered colormap
        n (int): Number of colors to return

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

    return plt.get_cmap(cmap)(np.linspace(0, 1, n))


def cmap2txt(cmap, filename=None, comments='%'):
    """Export colormap to txt file.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.txt'
        comments (str): Character to start comments with.

    """
    colors = mpl_colors(cmap, 256)
    date = time.strftime('%c')
    header = 'Colormap "{}". Created {}'.format(cmap, date)

    if filename is None:
        filename = cmap + '.txt'

    np.savetxt(filename, colors[:, :3], header=header, comments=comments)


def cmap2cpt(cmap, filename=None):
    """Export colormap to cpt file.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.cpt'

    """
    colors = mpl_colors(cmap, 256)
    date = time.strftime('%c')
    header = ('# Colormap "{}". Created {}\n'
              '# COLOR_MODEL = RGB\n'.format(cmap, date))

    left = '{:<3d} {:0>3d}/{:0>3d}/{:0>3d} '.format
    right = '{:<3d} {:0>3d}/{:0>3d}/{:0>3d}\n'.format

    if filename is None:
        filename = cmap + '.cpt'

    with open(filename, 'w') as f:
        f.write(header)

        # For each level spefifz a ...
        for n in range(len(colors) - 1):
            # ... start color ...
            r, g, b = [int(c * 255) for c in colors[n, :3]]
            f.write(left(n, r, g, b))
            # ... and end color.
            r, g, b = [int(c * 255) for c in colors[n + 1, :3]]
            f.write(right(n + 1, r, g, b))
