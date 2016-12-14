# -*- coding: utf-8 -*-

"""Functions to create plots using matplotlib.
"""

import warnings
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


def scatter_density_plot_matrix(M=None,
        hist_kw={},
        hexbin_kw={"mincnt": 1, "cmap": "viridis"},
        plot_dist_kw={"color": "tan", "ptiles": [5, 25, 50, 75, 95],
            "linestyles": [":", "--", "-", "--", ":"],
            "linewidth": 1.5},
        ranges={},
        units=None,
        **kwargs):
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

    There are three ways to pass the data:

    1. As a structured ndarray.  This is the preferred way.  The
       fieldnames will be used for the axes and order is preserved.
       Each field in the dtype must be scalar (0-d) and numerical.

    2. As keyword arguments.  All extra keyword arguments will be taken as
       variables.  They must all be 1-D ndarrays of numerical dtype, and
       they must all have the same size.  Order is preserve from Python
       3.6 only.

    3. As a regular 2-D numerical ndarray of shape [N × p].  In this case,
       the innermost dimension will be taken as the quantities to be
       plotted.  There is no axes labelling.

    Parameters:
        M (np.ndarray): ndarray.  If structured, the fieldnames will be used
            as the variables to be plotted against each other.  Each field
            in the structured array shall be 0-dimensional and
            of a numerical dtype.  If not structured, interpreted as 2-D
            array and the axes will be unlabelled.  You should pass either
            this argument, or additional keyworad arguments (see below).
        hist_kw (Mapping): Keyword arguments to pass to hist for diagonals.
        hexbin_kw (Mapping): Keyword arguments to pass to each call of
            hexbin.
        plot_dist_kw (Mapping): Keyword arguments to pass to each call of
            `func:plot_distribution_as_percentiles`.
        ranges (Mapping[str, Tuple[Real, Real]]): 
            For each field in M, can pass a range.  If provided, this
            range will be passed on to hist and hexbin.
        units (Mapping[str, str]): Unit strings for each of the
            quantities.  Optional.  If not passed, no unit is shown in the
            graph, unless the quantities to be plotted are pint quantity
            objects.

    If not passing `M`, you can instead pass keyword arguments
    referring to the different fields to be plotted.  In this case,
    each keyword argument should be a 1-dimensional ndarray with a
    numeric dtype.  If you use Python 3.6 or later, the order of the
    keyword arguments should be preserved.

    Returns:
        f (matplotlib.figure.Figure): Figure object created.
            You will still want to use subplots_adjust, suptitle, perhaps
            add a colourbar, and other things.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon.plots import scatter_density_plot_matrix

        x = 5*np.random.randn(5000)
        y = x + 10*np.random.randn(x.size)
        z = y**2 + x**2 + 20*np.random.randn(x.size)

        scatter_density_plot_matrix(
            x=x, y=y, z=z,
            hexbin_kw={"mincnt": 1, "cmap": "viridis", "gridsize": 20},
            units=dict(x="romans", y="knights", z="rabbits"))

        plt.show()

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon.plots import scatter_density_plot_matrix

        M = np.zeros(shape=(10000,),
                     dtype="f,f,f,f")
        M["f0"] = np.random.randn(M.size)
        M["f1"] = np.random.randn(M.size) + M["f0"]
        M["f2"] = 2*np.random.randn(M.size) + M["f0"]*M["f1"]
        M["f3"] = M["f0"] + M["f1"] + M["f2"] + 0.5*np.random.randn(M.size)

        scatter_density_plot_matrix(M,
            hexbin_kw={"mincnt": 1, "cmap": "viridis", "gridsize": 20})

        plt.show()
    """

    if M is None:
        M = np.empty(
            dtype=[(k, v.dtype) for (k, v) in kwargs.items()],
            shape=kwargs.copy().popitem()[1].shape)
        for (k, v) in kwargs.items():
            M[k] = v
    elif not M.dtype.fields:
        MM = np.empty(
            dtype=",".join([M.dtype.descr[0][1]]*M.shape[1]),
            shape=M.shape[0])
        for i in range(M.shape[1]):
            MM["f{:d}".format(i)] = M[:, i]
        M = MM
    if len(M.dtype.fields) > 20:
        raise ValueError("You've given me {:d} fields to plot. "
            "That would result in {:d} subplots.  I refuse to take "
            "more than 20 fields.".format(len(M.dtype.fields),
                len(M.dtype.fields)**2))
    if units is None:
        units = {}
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
            inrange = ((x>=rng[0][0])&(x<=rng[0][1])&
                       (y>=rng[1][0])&(y<=rng[1][1]))
            if not inrange.any():
                warnings.warn(
                    "Combination {:s}/{:s} has no valid values".format(
                        x_f, y_f),
                    RuntimeWarning)
                continue
            x = x[inrange]
            y = y[inrange]
            # NB: hexbin may be better than hist2d
            a.hexbin(x, y,
                extent=[rng[0][0], rng[0][1], rng[1][0], rng[1][1]],
                **hexbin_kw)
            plot_distribution_as_percentiles(
                a,
                x, y,
                **plot_dist_kw)
            a.set_xlim(rng[0])
            a.set_ylim(rng[1])

        if x_i == 0:
            a.set_ylabel(
                "{:s} [{:s}]".format(y_f, units[y_f]) if y_f in units else
                "{:s} [{:~}]".format(y_f, y.u) if hasattr(y, "u") else
                y_f)

        if y_i == N-1: # NB: 0 is top row, N-1 is bottom row
            a.set_xlabel(
                "{:s} [{:s}]".format(x_f, units[x_f]) if x_f in units else
                "{:s} [{:~}]".format(x_f, x.u) if hasattr(x, "u") else
                x_f)

    return f
