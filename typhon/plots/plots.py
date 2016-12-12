# -*- coding: utf-8 -*-

"""Functions to create plots using matplotlib.
"""
import collections
import math
import itertools
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from typhon.math import stats as tpstats


__all__ = [
    'plot_distribution_as_percentiles',
    'heatmap',
    'scatter_density_plot_matrix',
]


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

        ax.plot(bins, scores[:, i], linestyle=linestyles[i], label=locallab,
                **kwargs)


def heatmap(x, y, bins=20, bisectrix=True, ax=None, **kwargs):
    """Plot a heatmap of two data arrays.

    This function is a simple wrapper for :func:`plt.hist2d`.

    Parameters:
        x (np.ndarray): x data.
        y (np.ndarray): y data.
        bins (int | [int, int] | array_like | [array, array]):
            The bin specification:

            - If int, the number of bins for the two dimensions
              (nx=ny=bins).

            - If [int, int], the number of bins in each dimension
              (nx, ny = bins).

            - If array_like, the bin edges for the two dimensions
              (x_edges=y_edges=bins).

            - If [array, array], the bin edges in each dimension
              (x_edges, y_edges = bins).

            The default value is 20.

        bisectrix (bool): Toggle drawing of the bisectrix.
        ax (AxesSubplot, optional): Axes to plot in.
        **kwargs: Additional keyword arguments passed to
            :func:`matplotlib.pyplot.hist2d`.

    Returns:
        AxesImage.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon.plots import heatmap


        x = np.random.randn(500)
        y = x + np.random.randn(x.size)

        fig, ax = plt.subplots()
        heatmap(x, y, ax=ax)

        plt.show()

    """
    if ax is None:
        ax = plt.gca()

    # Default keyword arguments to pass to hist2d().
    kwargs_defaults = {
        'cmap': plt.get_cmap('Greys', 8),
        'rasterized': True,
        }

    kwargs_defaults.update(kwargs)

    # Plot the heatmap.
    N, xedges, yedges, img = ax.hist2d(x, y, bins, **kwargs_defaults)

    # Plot the bisectrix.
    if bisectrix:
        ax.plot((x.min(), x.max()), (x.min(), x.max()),
                color='red', linestyle='--', linewidth=2)

    return img


def scatter_density_plot_matrix(M,
        hist_kw={},
        hist2d_kw={"cmin": 1, "cmap": "viridis"},
        hexbin_kw={"mincnt": 1, "cmap": "viridis"},
        plot_dist_kw={"color": "tan", "ptiles": [5, 25, 50, 75, 95],
            "linestyles": [":", "--", "-", "--", ":"],
            "linewidth": 1.5},
        ranges={}):
    """Plot a scatter density plot matrix

    Like a scatter plot matrix but rather than every axes containing a
    scatter plot of M[i] vs. M[j], it contains a 2-D histogram along with
    the distribution as percentiles.  The 2-D histogram is created using
    hexbin rather than hist2d, because hexagonal bins are a more
    appropriate.  For example, see
    http://www.meccanismocomplesso.org/en/hexagonal-binning/

    On top of the 2-D histogram, shows the distribution using
    `func:plot_distribution_as_percentiles`.

    Plots regular 1-D histograms in the diagonals.

    Parameters:
        M (np.ndarray): Structured ndarray.  The fieldnames will be used
            as the variables to be plotted against each other.  Each field
            in the structured array shall be single-dimensional and
            of a numerical dtype.
        hist_kw (Mapping): Keyword arguments to pass to hist for diagonals.
        hist2d_kw (Mapping): Keyword arguments to pass to each call of
            hist2d.
        plot_dist_kw (Mapping): Keyword arguments to pass to each call of
            `func:plot_distribution_as_percentiles`.
        ranges (Mapping[str, Tuple[Real, Real]]): 
            For each field in M, can pass a range.  If provided, this
            range will be passed on to hist and hexbin.

    Returns:
        f (matplotlib.figure.Figure): Figure object created.
            You will still want to use subplots_adjust, suptitle, perhaps
            add a colourbar, and other things.
    """

    N = len(M.dtype.names)
    (f, ax_all) = plt.subplots(N, N, figsize=(4+3*N, 4+3*N))

    for ((x_i, x_f), (y_i, y_f)) in itertools.product(
            enumerate(M.dtype.names), repeat=2):
        # all with the same y-coordinate should have the same x-variable, and
        # 〃  〃   〃  〃   x-〃         〃     〃   〃  〃   y-〃.
        #
        # so at (0, 1) we want (x1, y0), at (0, 2) we want (x2, y0), etc.
        # hence turn-around y_i and x_i.
        a = ax_all[y_i, x_i]

        x = M[x_f]
        y = M[y_f]

        if x_i == y_i:
            rng = ranges.get(x_f, (x.min(), x.max()))
            a.hist(x, range=rng, **hist_kw)
        else:
            rng = (ranges.get(x_f, (x.min(), x.max())),
                   ranges.get(y_f, (y.min(), y.max())))
            # NB: hexbin may be better than hist2d
            a.hexbin(x, y,
                extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
                **hexbin_kw)
            inrange = ((x>rng[0][0])&(x<rng[0][1])&
                       (y>rng[1][0])&(y<rng[1][1]))
            plot_distribution_as_percentiles(
                a,
                x[inrange], y[inrange],
                **plot_dist_kw)
            a.set_xlim(rng[0])
            a.set_ylim(rng[1])

        if x_i == 0:
            a.set_ylabel(y_f)

        if y_i == N-1: # NB: 0 is top row, N-1 is bottom row
            a.set_xlabel(x_f)

    return f
