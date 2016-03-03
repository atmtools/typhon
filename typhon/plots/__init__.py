# -*- coding: utf-8 -*-
"""
This module provides functions related to plot or to plot data.

"""

import math
import collections

import numpy as np

from ..math import stats as tpstats

__all__ = ['figsize']

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
        >>> typhon.cm.figsize(1)
        (1, 0.61803398874989479)

        >>> typhon.cm.figsize(1, portrait=True)
        (1, 1.6180339887498949)
    """
    phi = 0.5 * (np.sqrt(5) + 1)
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
    D_ls = collections.defaultdict(list)
    for (ls, pt) in zip(linestyles, ptiles):
        D_ls[ls].append(pt)
    for i in range(len(ptiles)):
        if label is not None:
            if math.isclose(ptiles[i], 50):
                locallab = label + " (median)"
            elif ptile_to_legend and linestyles[i] in D_ls:
                locallab = label + " (p-{:s})".format("/".join("{:d}".format(x)
                                        for x in D_ls.pop(linestyles[i])))
            else:
                locallab = None
            
        ax.plot(bins, scores[:, i], ls=linestyles[i], label=locallab, **kwargs)
