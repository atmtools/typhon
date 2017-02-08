# -*- coding: utf-8 -*-

"""Utility functions related to plotting.
"""
import os
import re
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from warnings import warn

__all__ = ['mpl_colors',
           'colors2cmap',
           'cmap2txt',
           'cmap2cpt',
           'cmap2act',
           'cmap2c3g',
           'cmap2ggr',
           'cmap_from_act',
           'cmap_from_txt',
           ]


def mpl_colors(cmap=None, N=None):
    """Return a list of RGB values.

    Parameters:
        cmap (str): Name of a registered colormap.
        N (int): Number of colors to return.
            If ``None`` use the number of colors defined in the colormap.

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

    if N is None:
        N = plt.get_cmap(cmap).N

    return plt.get_cmap(cmap)(np.linspace(0, 1, N))


def _to_hex(c):
    """Convert arbitray color specification to hex string."""
    ctype = type(c)

    # Convert rgb to hex.
    if ctype is tuple or ctype is np.ndarray or ctype is list:
        return colors.rgb2hex(c)

    if ctype is str:
        # If color is already hex, simply return it.
        regex = re.compile('^#[A-Fa-f0-9]{6}$')
        if regex.match(c):
            return c

        # Convert named color to hex.
        return colors.cnames[c]

    raise Exception("Can't handle color of type: {}".format(ctype))


def colors2cmap(*args, name=None):
    """Create a colormap from a list of given colors.

    Parameters:
        *args: Arbitrary number of colors (Named color, HEX or RGB).
        name (str): Name with which the colormap is registered.

    Returns:
        LinearSegmentedColormap.

    Examples:
        >>> colors2cmap('darkorange', 'white', 'darkgreen', name='test')
    """
    if len(args) < 2:
        raise Exception("Give at least two colors.")

    cmap_data = [_to_hex(c) for c in args]
    cmap = colors.LinearSegmentedColormap.from_list(name, cmap_data)
    plt.register_cmap(name, cmap)

    return cmap


def cmap2txt(cmap, filename=None, N=None, comments='%'):
    """Export colormap to txt file.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.txt'
        N (int): Number of colors.
        comments (str): Character to start comments with.

    """
    colors = mpl_colors(cmap, N)
    header = 'Colormap "{}"'.format(cmap)

    if filename is None:
        filename = cmap + '.txt'

    np.savetxt(filename, colors[:, :3], header=header, comments=comments)


def cmap2cpt(cmap, filename=None, N=None):
    """Export colormap to cpt file.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.cpt'
        N (int): Number of colors.

    """
    colors = mpl_colors(cmap, N)
    header = ('# GMT palette "{}"\n'
              '# COLOR_MODEL = RGB\n'.format(cmap))

    left = '{:>3d} {:>3d} {:>3d} {:>3d}  '.format
    right = '{:>3d} {:>3d} {:>3d} {:>3d}\n'.format

    if filename is None:
        filename = cmap + '.cpt'

    with open(filename, 'w') as f:
        f.write(header)

        # For each level specify a ...
        for n in range(len(colors)):
            rgb = [int(c * 255) for c in colors[n, :3]]
            # ... start color ...
            f.write(left(n, *rgb))
            # ... and end color.
            f.write(right(n + 1, *rgb))


def cmap2act(cmap, filename=None, N=None):
    """Export colormap to Adobe Color Table file.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.cpt'
        N (int): Number of colors.

    """
    if filename is None:
        filename = cmap + '.act'

    # If the number of color levels to export is not set...
    if N is None:
        # ... use the number of colors defined in the colormap.
        N = plt.get_cmap(cmap).N

    if N > 256:
        N = 256
        warn('Maximum number of colors is 256.')

    colors = mpl_colors(cmap, N)[:, :3]

    rgb = np.zeros(256 * 3 + 2)
    rgb[:colors.size] = (colors.flatten() * 255).astype(np.uint8)
    rgb[768:770] = np.uint8(N // 2**8), np.uint8(N % 2**8)

    rgb.astype(np.uint8).tofile(filename)


def cmap2c3g(cmap, filename=None, N=None):
    """Export colormap ass CSS3 gradient.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.cpt'
        N (int): Number of colors.

    """
    if filename is None:
        filename = cmap + '.c3g'

    colors = mpl_colors(cmap, N)

    header = (
        '/*'
        '   CSS3 Gradient "{}"\n'
        '*/\n\n'
        'linear-gradient(\n'
        '  0deg,\n'
        ).format(cmap)

    color_spec = '  rgb({:>3d},{:>3d},{:>3d}) {:>8.3%}'.format

    with open(filename, 'w') as f:
        f.write(header)

        ncolors = len(colors)
        for n in range(ncolors):
            r, g, b = [int(c * 255) for c in colors[n, :3]]
            f.write(color_spec(r, g, b, n / (ncolors - 1)))
            if n < ncolors - 1:
                f.write(',\n')

        f.write('\n  );')


def cmap2ggr(cmap, filename=None, N=None):
    """Export colormap as GIMP gradient.

    Parameters:
        cmap (str): Colormap name.
        filename (str): Optional filename.
            Default: cmap + '.cpt'
        N (int): Number of colors.

    """
    if filename is None:
        filename = cmap + '.ggr'

    colors = mpl_colors(cmap, N)
    header = ('GIMP Gradient\n'
              'Name: {}\n'
              '{}\n').format(cmap, len(colors) - 1)

    line = ('{:.6f} {:.6f} {:.6f} '  # start, middle, stop
            '{:.6f} {:.6f} {:.6f} {:.6f} '  # RGBA
            '{:.6f} {:.6f} {:.6f} {:.6f} '  # RGBA next level
            '0 0\n').format

    def idx(x):
        return x / (len(colors) - 1)

    with open(filename, 'w') as f:
        f.write(header)

        for n in range(len(colors) - 1):
            rgb = colors[n, :]
            rgb_next = colors[n + 1, :]
            f.write(line(idx(n), idx(n + 0.5), idx(n + 1), *rgb, *rgb_next))


def cmap_from_act(file, name=None):
    """Import colormap from Adobe Color Table file.

    Parameters:
        file (str): Path to act file.
        name (str): Colormap name. Defaults to filename without extension.

    Returns:
        LinearSegmentedColormap.
    """
    # Extract colormap name from filename.
    if name is None:
        name = os.path.splitext(os.path.basename(file))[0]

    # Read binary file and determine number of colors
    rgb = np.fromfile(file, dtype=np.uint8)
    if rgb.shape[0] >= 770:
        ncolors = rgb[768] * 2**8 + rgb[769]
    else:
        ncolors = 256

    colors = rgb[:ncolors*3].reshape(ncolors, 3) / 255

    # Create and register colormap...
    cmap = LinearSegmentedColormap.from_list(name, colors, N=ncolors)
    plt.register_cmap(cmap=cmap)  # Register colormap.

    # ... and the reversed colormap.
    cmap_r = LinearSegmentedColormap.from_list(
            name + '_r', np.flipud(colors), N=ncolors)
    plt.register_cmap(cmap=cmap_r)

    return cmap


def cmap_from_txt(file, name=None, N=-1, comments='%'):
    """Import colormap from txt file.

    Reads colormap data (RGB/RGBA) from an ASCII file.
    Values have to be given in [0, 1] range.

    Parameters:
        file (str): Path to txt file.
        name (str): Colormap name. Defaults to filename without extension.
        N (int): Number of colors.
            ``-1`` means all colors (i.e., the complete file).
        comments (str): Character to start comments with.

    Returns:
        LinearSegmentedColormap.
    """
    # Extract colormap name from filename.
    if name is None:
        name = os.path.splitext(os.path.basename(file))[0]

    # Read binary file and determine number of colors
    rgb = np.genfromtxt(file, comments=comments)
    if N == -1:
        N = np.shape(rgb)[0]

    if np.min(rgb) < 0 or np.max(rgb) > 1:
        raise Exception('RGB value out of range: [0, 1].')

    # Create and register colormap...
    cmap = LinearSegmentedColormap.from_list(name, rgb, N=N)
    plt.register_cmap(cmap=cmap)

    # ... and the reversed colormap.
    cmap_r = LinearSegmentedColormap.from_list(
            name + '_r', np.flipud(rgb), N=N)
    plt.register_cmap(cmap=cmap_r)

    return cmap
