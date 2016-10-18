# -*- coding: utf-8 -*-
"""
This module provides functions related to plot or to plot data.

"""

import collections
import glob
import math
import os

import numpy as np

import typhon.cm
from typhon.math import stats as tpstats
from typhon import constants

__all__ = ['figsize',
           'styles',
           'get_available_styles',
           'get_subplot_arrangement',
           ]


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


def plot_distribution_as_percentiles(ax, x, y,
                                     nbins=10, bins=None,
                                     ptiles=(5, 25, 50, 75, 95),
                                     linestyles=(":", "--", "-", "--", ":"),
                                     ptile_to_legend=True,
                                     label=None,
                                     **kwargs):
    """Plot the distribution of y vs. x as percentiles.

    Bin y-data according to x-data (using :func:`typhon.math.stats.bin`).
    Then, within each bin, show the distribution by plotting percentiles.

    Arguments:

        ax (AxesSubplot): matplotlib axes to plot into
        x (ndarray): data for x-axis
        y (ndarray): data for y-axis
        nbins (int): Number of bins to use for dividing the x-data.
        bins (ndarray): Specific bins to use for dividing the x-data.
            Supersedes nbins.
        ptiles (ndarray): Percentiles to visualise.
        linestyles (List[str]): List of linestyles corresponding to percentiles
        ptile_to_legend (bool): True: Add a label to each percentile.
            False: Only add a legend to the median.
        label (str or None): Label to use in legend.
        **kwargs: Remaining arguments passed to :func:`AxesSubplot.plot`.
            You probably want to pass at least color.
    """

    if bins is None:
        bins = np.linspace(x.min(), x.max(), nbins)
    scores = tpstats.get_distribution_as_percentiles(x, y, bins, ptiles)

    # Collect where same linestyle is used, so it can be combined in
    # legend
    d_ls = collections.defaultdict(list)
    locallab = None
    for (ls, pt) in zip(linestyles, ptiles):
        d_ls[ls].append(pt)
    for i in range(len(ptiles)):
        if label is not None:
            if math.isclose(ptiles[i], 50):
                locallab = label + " (median)"
            else:
                if ptile_to_legend and linestyles[i] in d_ls:
                    locallab = label + " (p-{:s})".format(
                        "/".join("{:d}".format(x)
                                 for x in d_ls.pop(linestyles[i])))
                else:
                    locallab = None
        else:
            label = None

        ax.plot(bins, scores[:, i], linestyle=linestyles[i], label=locallab, **kwargs)


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
