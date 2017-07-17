# -*- coding: utf-8 -*-

"""Utility functions related to plotting.
"""
import glob
import itertools
import os
import string

import numpy as np
import matplotlib.pyplot as plt

from typhon import constants


__all__ = [
    'center_colorbar',
    'figsize',
    'styles',
    'get_available_styles',
    'get_subplot_arrangement',
    'label_axes',
    'supcolorbar',
]


def center_colorbar(cb):
    """Center a diverging colorbar around zero.

    Convenience function to adjust the color limits of a colorbar. The function
    multiplies the absolute maximum of the data range by ``(-1, 1)`` and uses
    this range as new color limits.

    Note:
        The colormap used should be continuous. Resetting the clim for discrete
        colormaps may produce strange artefacts.

    Parameters:
        cb (matplotlib.colorbar.Colorbar): Colorbar to center.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon.plots import center_colorbar


        fig, ax = plt.subplots()
        sm = ax.pcolormesh(np.random.randn(10, 10) + 0.75, cmap='difference')
        cb = fig.colorbar(sm)
        center_colorbar(cb)

        plt.show()
    """
    # Set color limits to +- the absolute maximum of the data range.
    cb.set_clim(np.multiply((-1, 1), np.max(np.abs(cb.get_clim()))))


def figsize(w, portrait=False):
    """Return a figure size matching the golden ratio.

    This function takes a figure width and returns a tuple
    representing width and height in the golden ratio.
    Results can be returned for portrait orientation.

    Parameters:
        w (float): Figure width.
        portrait (bool): Return size for portrait format.

    Return:
        tuple: Figure width and size.

    Examples:
        >>> import typhon.plots
        >>> typhon.plots.figsize(1)
        (1, 0.61803398874989479)

        >>> typhon.plots.figsize(1, portrait=True)
        (1, 1.6180339887498949)
    """
    phi = constants.golden_ratio
    return (w, w * phi) if portrait else (w, w / phi)


def get_subplot_arrangement(n):
    """Get efficient (nrow, ncol) for n subplots

    If we want to put `n` subplots in a square-ish/rectangular
    arrangement, how should we arrange them?

    Returns (⌈√n⌉, ‖√n‖)
    """
    return (int(np.ceil(np.sqrt(n))),
            int(np.round(np.sqrt(n))))


def styles(name):
    """Return absolute path to typhon stylesheet.

    Matplotlib stylesheets can be passed via their full path. This function
    takes a style name and returns the absolute path to the typhon stylesheet.

    Parameters:
        name (str): Style name.

    Returns:
        str: Absolute path to stylesheet.

    Example:
        Use typhon style for matplotlib plots.

        >>> import matplotlib.pyplot as plt
        >>> plt.style.use(styles('typhon'))

    """
    stylelib_dir = os.path.join(os.path.dirname(__file__), 'stylelib')

    return os.path.join(stylelib_dir, name + '.mplstyle')


def get_available_styles():
    """Return list of names of all styles shipped with typhon.

    Returns:
        list[str]: List of available styles.

    """
    stylelib_dir = os.path.join(os.path.dirname(__file__), 'stylelib')
    pattern = os.path.join(stylelib_dir, '*.mplstyle')

    return [os.path.splitext(os.path.basename(s))[0]
            for s in glob.glob(pattern)]


def label_axes(axes=None, labels=None, loc=(.02, .9), **kwargs):
    """Walks through axes and labels each.

    Parameters:
        axes (iterable): An iterable container of :class:`AxesSubplot`.
        labels (iterable): Iterable of strings to use to label the axes.
            If ``None``, first upper and then lower case letters are used.
        loc (tuple of floats): Where to put the label in axes-fraction units.
        **kwargs: Additional keyword arguments are collected and
            passed to :func:`~matplotlib.pyplot.annotate`.

    Examples:
        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            from typhon.plots import label_axes, styles


            plt.style.use(styles('typhon'))

            # Automatic labeling of axes.
            fig, axes = plt.subplots(ncols=2, nrows=2)
            label_axes()

            # Manually specify the axes to label.
            fig, axes = plt.subplots(ncols=2, nrows=2)
            label_axes(axes[:, 0])  # label each row.

            # Pass explicit labels (and additional arguments).
            fig, axes = plt.subplots(ncols=2, nrows=2)
            label_axes(labels=map(str, range(axes.size)), weight='bold')

    .. Based on https://stackoverflow.com/a/22509497
    """

    # This code, or an earlier version, was posted by user 'tacaswell' on Stack
    # https://stackoverflow.com/a/22509497/974555 and is licensed under
    # CC-BY-SA 3.0.  This notice may not be removed.
    if axes is None:
        axes = plt.gcf().axes

    if labels is None:
        labels = string.ascii_uppercase + string.ascii_lowercase

    labels = itertools.cycle(labels)  # re-use labels rather than stop labeling

    for ax, lab in zip(axes, labels):
        ax.annotate(lab, xy=loc, xycoords='axes fraction', **kwargs)


def supcolorbar(mappable, fig=None, right=0.8, rect=(0.85, 0.15, 0.05, 0.7),
                **kwargs):
    """Create a common colorbar for all subplots in a figure.

    Parameters:
        mappable: A scalar mappable like to which the colorbar applies
            (e.g. :class:`~matplotlib.collections.QuadMesh`,
            :class:`~matplotlib.contour.ContourSet`, etc.).
        fig (:class:`~matplotlib.figure.Figure`):
            Figure to add the colorbar into.
        right (float): Fraction of figure to use for the colorbar (0-1).
        rect (array-like): Add an axes at postion
            ``rect = [left, bottom, width, height]`` where all quantities are
            in fraction of figure.
        **kwargs: Additional keyword arguments are passed to
            :func:`matplotlib.figure.Figure.add_axes`.

    Returns:
        matplotlib.colorbar.Colorbar: Colorbar.

    Note:
        In order to properly scale the value range of the colorbar the ``vmin``
        and ``vmax`` property should be set manually.

    Examples:
        .. plot::
            :include-source:

            import matplotlib.pyplot as plt
            import numpy as np
            from typhon.plots import supcolorbar


            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            for ax in axes.flat:
                sm = ax.pcolormesh(np.random.random((10,10)), vmin=0, vmax=1)

            supcolorbar(sm, label='Test label')

    """
    if fig is None:
        fig = plt.gcf()

    fig.subplots_adjust(right=right)
    cbar_ax = fig.add_axes(rect)

    return fig.colorbar(mappable, cax=cbar_ax, **kwargs)
