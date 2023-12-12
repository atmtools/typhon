# -*- coding: utf-8 -*-

"""Functions to create plots using matplotlib.
"""

import collections
from datetime import datetime
import itertools
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import get_cmap
import scipy.stats as stats

from typhon.plots import formatter
from typhon.math import stats as tpstats


__all__ = [
    'binned_statistic',
    'plot_distribution_as_percentiles',
    'heatmap',
    'histogram',
    'scatter_density_plot_matrix',
    'diff_histogram',
    'profile_p',
    'profile_p_log',
    'profile_z',
    'channels',
    'colored_bars',
    'plot_bitfield',
]


def binned_statistic(
            x, y, bins=20, ax=None, ptype=None, pargs=None, **kwargs
        ):
    """Bin the data and plot their statistics

    Per default, this calculates the median for each bin. If you need another
    statistic (e.g. mean or std) use the keyword `statistic`.

    Args:
        x: The values which should be binned and plotted on the x-axis.
        y: The values on which the statistics should be applied. The results
            will be plotted on the y-axis.
        bins: Number of bins. Default is 20.
        ax (AxesSubplot, optional): Axes to plot in.
        ptype: Plot type. Can be *scatter* or *boxplot*.
        pargs: Plotting keyword arguments that are allowed for *ptype*.
        **kwargs: Additional key word arguments for
            `scipy.stats.binned_statistic`.

    Returns:
        The plot object.

    Examples:

    .. :code-block:: python


    """

    if ax is None:
        ax = plt.gca()

    if pargs is None:
        pargs = {}

    if ptype is None or ptype == "scatter":
        default = {
            "statistic": "median",
            "bins": bins,
            **kwargs,
        }

        statistics, bin_edges, bin_ind = stats.binned_statistic(
            x, values=y, **default
        )
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2

        plot = ax.plot(bin_centers, statistics, **pargs)
    elif ptype == "boxplot":
        bin_lefts = np.linspace(x.min(), x.max(), bins)
        bins_indices = np.digitize(x, bin_lefts)

        plot = ax.boxplot(
            [y[bins_indices == i] for i in range(bins)],
            **pargs
        )

        bin_width = (bin_lefts[1] - bin_lefts[0])
        bin_centers = bin_lefts[1:] + bin_width / 2
        ax.set_xticklabels([f"{center:.1f}" for center in bin_centers])
    else:
        raise ValueError(f"Unknown plot type {ptype}!")

    return plot


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


def histogram(data, kind=None, ax=None, **kwargs):
    """Plot a histogram of a data array

    Args:
        data: A np.ndarray or xr.DataArray with the data that should be binned.
        kind: Kind of histogram that should be plotted. Can be *standard*
            (plot with bars, is default), *points* (scatter plot) or *line*
            (line plot).
        ax: Axes to plot in.
        **kwargs: Additional keyword arguments passed to
            :func:`matplotlib.pyplot.hist`.

    Returns:
        AxesImage.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon.plots import histogram


        x = np.random.randn(500)

        fig, axes = plt.subplots(nrows=3)
        kinds = ["standard", "points", "line"]
        for i, kind in enumerate(kinds):
            histogram(x, ax=axes[i], kind=kind)

        plt.tight_layout()
        plt.show()
    """
    if ax is None:
        ax = plt.gca()

    if kind == "points" or kind == "line":
        hist_keys = {"bins", "range", "normed", "weights", "density"}
        hist_kwargs = {key: value for key, value in kwargs.items() if
                       key in hist_keys}
        y, bin_edges = np.histogram(data, **hist_kwargs)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        plot_kwargs = {key: value for key, value in kwargs.items() if
                       key not in hist_keys}
        if kind == "points":
            ax.scatter(bin_centers, y, **plot_kwargs)
        elif kind == "line":
            ax.plot(bin_centers, y, '-', **plot_kwargs)
    elif kind is None or kind == "standard":
        ax.hist(data, **kwargs)
    else:
        raise ValueError(f"Unknown kind of histogram: {kind}!")


# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822.  This specifically applies to the functions
# scatter_density_plot_matrix and plot_bitfield.
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

def scatter_density_plot_matrix(
        M=None,
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
        raise ValueError(
            "You've given me {:d} fields to plot. "
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
            inrange = ((x >= rng[0][0]) & (x <= rng[0][1]) &
                       (y >= rng[1][0]) & (y <= rng[1][1]))
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

        if y_i == N-1:  # NB: 0 is top row, N-1 is bottom row
            a.set_xlabel(
                "{:s} [{:s}]".format(x_f, units[x_f]) if x_f in units else
                "{:s} [{:~}]".format(x_f, x.u) if hasattr(x, "u") else
                x_f)

    return f


def profile_p(p, x, ax=None, **kwargs):
    """Plot atmospheric profile against pressure in linear space.

    Parameters:
        p (ndarray): Pressure [Pa].
        x (ndarray): Atmospheric property.
        ax (AxesSubplot): Axes to plot in.
        **kwargs: Additional keyword arguments passed to `plt.plot`.

    See also:
            :func:`~typhon.plots.profile_p_log`
                Plot profile against pressure in log space.
            :func:`~typhon.plots.profile_z`
                Plot profile against height.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import typhon.plots

        p = typhon.math.nlogspace(1000e2, 0.1e2, 50)  # pressure in Pa.
        x = np.exp(p / p[0])

        fig, ax = plt.subplots()
        typhon.plots.profile_p(p, x, ax=ax)

        plt.show()

    """
    if ax is None:
        ax = plt.gca()

    # Label and format for yaxis.
    formatter.set_yaxis_formatter(formatter.HectoPascalFormatter(), ax=ax)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Pressure [hPa]')

    # Actual plot.
    ret = ax.plot(x, p, **kwargs)

    if hasattr(ax.yaxis, 'set_inverted'):
        # Matplotlib >=3.1.0
        # In matplotlib==3.1.0 the yaxis has to be inverted after actual plot
        # (https://github.com/matplotlib/matplotlib/issues/14615)
        ax.yaxis.set_inverted(True)
    elif not ax.yaxis_inverted():
        # Matplotlib <=3.0.3
        ax.invert_yaxis()

    return ret


def profile_p_log(p, x, ax=None, **kwargs):
    """Plot atmospheric profile against pressure in log space.

    This function is a wrapper for :func:`~typhon.plots.profile_p`:
    The input values as well as additional keyword arguments are passed through
    and the yscale is set to "log".

    Parameters:
        p (ndarray): Pressure [Pa].
        x (ndarray): Atmospheric property.
        ax (AxesSubplot): Axes to plot in.
        **kwargs: Additional keyword arguments passed to `plt.plot`.

    See also:
            :func:`~typhon.plots.profile_p`
                Plot profile against pressure in linear space.
            :func:`~typhon.plots.profile_z`
                Plot profile against height.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import typhon.plots

        p = typhon.math.nlogspace(1000e2, 0.1e2, 50)
        x = np.exp(p / p[0])

        fig, ax = plt.subplots()
        typhon.plots.profile_p_log(p, x, ax=ax)

        plt.show()

    """
    if ax is None:
        ax = plt.gca()

    # Call ``set_yscale`` before plot to prevent reset of axis inverting in
    # matplotlib 3.1.0 (https://github.com/matplotlib/matplotlib/issues/14620)
    ax.set_yscale('log')
    ret = profile_p(p, x, ax=ax, **kwargs)

    # Set logarithmic scale.
    formatter.set_yaxis_formatter(formatter.HectoPascalLogFormatter(), ax=ax)

    return ret


def profile_z(z, x, ax=None, **kwargs):
    """Plot atmospheric profile of arbitrary property against height (in km).

    Parameters:
        z (ndarray): Height [m].
        x (ndarray): Atmospheric property.
        ax (AxesSubplot): Axes to plot in.
        **kwargs: Additional keyword arguments passed to `plt.plot`.

    See also:
            :func:`~typhon.plots.profile_p`
                Plot profile against pressure in linear space.
            :func:`~typhon.plots.profile_p_log`
                Plot profile against pressure in log space.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import typhon.plots

        z = np.linspace(0, 80e3, 50)
        x = np.sin(z / 5e3)

        fig, ax = plt.subplots()
        typhon.plots.profile_z(z, x, ax=ax)

        plt.show()
    """
    if ax is None:
        ax = plt.gca()

    # Determine min/max pressure of data in plot. The current axis limits are
    # also taken into account. This ensures complete data coverage when using
    # the function iteratively on the same axis.
    zmin = np.min((np.min(z), *ax.get_ylim()))
    zmax = np.max((np.max(z), *ax.get_ylim()))
    ax.set_ylim(zmin, zmax)

    # Label and format for yaxis.
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('Height [km]')

    # Actual plot.
    ret = ax.plot(x, z, **kwargs)

    formatter.set_yaxis_formatter(formatter.ScalingFormatter('kilo', '{x:g}'),
                                  ax=ax)

    return ret


def diff_histogram(array1, array2, ax=None,
                    plot_args=None, **hist_args):
    """Bin two arrays and plot their differences as histogram.

    Args:
        array1: One-dimensional numpy.array.
        array2: One-dimensional numpy.array.
        ax: Axes to plot in.
        plot_args: Keyowrd arguments of matplotllib.pyplot.step as dictionary.
        **hist_args: Additional keyword arguments for numpy.histogram.

    Returns:
        A tuple of the plot object, the bin edges and the differences.
    """

    if ax is None:
        ax = plt.gca()

    # numpy.histogram cannot work with numpy.datetime64 objects, therefore we
    # have to convert them before:
    bins_are_timestamps = False
    if array1.dtype.type == np.datetime64 or array1.dtype.type == datetime:
        array1 = array1.astype("M8[ns]").astype("float")
        bins_are_timestamps = True
    if array2.dtype.type == np.datetime64 or array2.dtype.type == datetime:
        array2 = array2.astype("M8[ns]").astype("float")
        bins_are_timestamps = True

    if "range" in hist_args:
        range_array = np.array(hist_args["range"])

        # If a range for the histogram is set, we have to convert if it is a
        # datetime data type:
        if range_array.dtype.type == np.datetime64 \
                or isinstance(range_array.item(0), datetime):
            range_array = range_array.astype("M8[ns]").astype("float")
            hist_args["range"] = range_array.tolist()
    elif "bins" not in hist_args and "range" not in hist_args:
        # Both data arrays must be binned with the same bins. If the user did
        # not care, we do.
        start = min(array1.min(), array2.min())
        end = max(array1.max(), array2.max())
        hist_args["range"] = [start, end]

    y1, bins1 = np.histogram(array1, **hist_args)
    y2, bins2 = np.histogram(array2, **hist_args)

    # The two bins should be the same
    if not np.allclose(bins1, bins2):
        raise ValueError("Arrays could not be grouped into the same bins!")

    if bins_are_timestamps:
        bins1 = bins1.astype("M8[ns]")

    diff = y1 - y2

    if plot_args is None:
        plot_args = {}

    bar_plot = ax.step(bins1[:-1], diff, **plot_args)

    return bar_plot, bins1, diff


def channels(met_mm_backend, ylim=None, ax=None, **kwargs):
    """Plot instrument channels for passband-type sensors.

    Parameters:
        met_mm_backend (ndarray): Backend description for meteorological
            millimeter sensors with passbands (:arts:`met_mm_backend`).
        ylim (tuple): A tuple-like container with the lower and upper y-limits
            for the rectangles.
        ax (AxesSubplot): Axes to plot in.
        **kwargs: Additional keyword arguments passed
            to :class:`~matplotlib.patches.Rectangle`.

    Returns:
        list[matplotlib.patches.Rectangle]: List of all rectangles drawn.

    Note:
        :class:`matplotlib.patches.Patch` do not set the axis limits.
        If this function is used without other data in the plot, you have
        to set the axis limits to an appropriate value. Otherwise the drawn
        channels might not be seen.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import typhon.plots

        met_mm_backend = np.array([
            [89.000e9, 0., 0., 2800e6],
            [157.000e9, 0., 0., 2800e6],
            [183.311e9, 1.00e9, 0., 500e6],
            [183.311e9, 3.00e9, 0., 1000e6],
            [190.311e9, 0., 0., 2200e6],
            ])

        fig, ax = plt.subplots(figsize=(8, 6))
        typhon.plots.channels(met_mm_backend / 1e9, ylim=(0.2, 0.8))
        ax.set_xlim(80, 200)  # mpl.patches do **not** autoscale axis limits!
        ax.set_xlabel('Frequency [GHz]')

        plt.show()
    """
    # TODO: mpl.patch.Patch do not set the axis limits properly. Consider
    # adding a small convenience code to adjust limits automatically.

    if ax is None:
        ax = plt.gca()

    if ylim is None:
        ylim = ax.get_ylim()
    ymin, ymax = ylim

    def plot_band(center, width, ymin, ymax):
        """Plot a single instrument band."""
        xy = (center - 0.5 * width, ymin)
        height = ymax - ymin
        return ax.add_patch(Rectangle(xy, width, height, **kwargs))

    band_centers = []
    band_widths = []
    for center, off1, off2, width in met_mm_backend:
        if off1 == 0:
            band_centers += [center]
            band_widths += [width]
        else:
            band_centers += [center - off1, center + off1]
            band_widths += [width, width]

            # If second passband is given, add offset to the last two passbands
            if off2 != 0:
                for bc in band_centers[-2:]:
                    band_centers += [bc - off2, bc + off2]
                    band_widths += [width, width]

    patches = []
    for center, width in zip(band_centers, band_widths):
        patches.append(plot_band(center, width, ymin, ymax))

    return patches


def colored_bars(x, y, c=None, cmap=None, vmin=None, vmax=None, ax=None,
                 **kwargs):
    """Plot a colorized series of bars.

    Note:
        If the x-values are floats (smaller than ``1``) the ``width`` should
        be adjusted to prevent bars from overlapping.

    Parameters:
        x (ndarray): Abscissa values.
        y (ndarray): Ordinate values.
        c (ndarray): (Optional) values for color scaling.
        cmap (str or matplotlib.colors.Colormap):
            The colormap used to map normalized data values to RGBA colors.
        vmin (float): Set lower color limit.
        vmax (float): Set upper color limit.
        ax (AxesSubplot): Axes to plot into.
        **kwargs: Additional keyword arguments are passed to
            :func:`matplotlib.pyplot.bar`.

    Returns:
        matplotlib.cm.ScalarMappable, matplotlib.container.BarContainer:
            Mappable, Artists corresponding to each bar.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from typhon.plots import colored_bars


        N = 50
        x = np.arange(N)
        y = np.sin(np.linspace(0, 3 * np.pi, N)) + 0.5 * np.random.randn(N)

        # Basic series with bars colored according to y-value.
        fig, ax = plt.subplots()
        colored_bars(x, y, cmap='seismic')

        # Add a colorbar to the figure.
        fig, ax = plt.subplots()
        sm, bars = colored_bars(x, y, cmap='seismic')
        cb = fig.colorbar(sm, ax=ax)

        # Pass different values for coloring (here, x-values).
        fig, ax = plt.subplots()
        colored_bars(x, y, c=x)

        plt.show()
    """
    if ax is None:
        ax = plt.gca()

    # If no values for the color mapping are passed, use the y-values.
    if c is None:
        c = y

    # Create colormap instance. This works for strings and Colormap instances.
    cmap = plt.get_cmap(cmap)

    # Find limits for color mapping (values exceeding this threshold are set
    # to the color of the corresponding limit). If no explicit values are
    # passed, use the absolute value range; this results in a colormap
    # centered around zero.
    absmax = np.max(np.abs(c))
    vmin = -absmax if vmin is None else vmin
    vmax = absmax if vmax is None else vmax

    # Create a Normalize function for given color limits. This function is
    # used to map the data values to the range [0, 1].
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Combine the norm and colormap into a ScalarMappable. The ScalarMappable
    #  is returned and can be used for creating a colorbar.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(c.flat)  # Pass data range to ScalarMappable.

    # Create the actual bar plot. The color values are normalized and
    # converted to RGB values which are passed to ``color``.
    ret = ax.bar(x, y, color=cmap(norm(c)), **kwargs)

    # Return the ScalarMappable as well as all return values of ``plt.bar``.
    return sm, ret

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822.  This specifically applies to the functions
# scatter_density_plot_matrix and plot_bitfield.
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

def plot_bitfield(ax, X, Y, bitfield, flag_dict,
        cmap,
        cax=None,
        pcolor_args={},
        colorbar_args={},
        unflagged="unflagged",
        joiner=", "):
    """Plot a bitfield of categories with pcolor

    The numeric values in a bitfield are not directly meaningful.  Rather,
    the relevant information is whether a particular bit is set.  This
    function plots a 2-D bitfield using pcolor, then displays each unique
    combination of flags (or the absence of any flags) as a distinct
    category, and shows the corresponding labels in the colourbar.

    This assumes that, even when there are many possible flags, only a
    small subset of combinations of flags actually occurs within the data.
    Should this exceed a dozen or so, then the colorbar/legend will become
    particularly crowded.

    Note that a colorbar may not be the optimal legend to show alongside
    categorical data but as this function already exists it is more
    convenient to exploit than others.  Currently this function only works
    with pcolor, not with pcolormesh, scatter, or other plotting
    functions for 3-D data.

    See https://gist.github.com/jakevdp/8a992f606899ac24b711 for an
    illustration of what the result may look like, although that is for
    the case of a scatter rather than pcolor plot.

    Parameters:
        ax (Axes):
            Axes (or subclass thereof, such as GeoAxes) to plot in.

        X (ndarray):
            X-values for bitfield.  Interpretation as for pcolor.

        Y (ndarray):
            Y-values for bitfield.  Interpretation as for pcolor.

        bitfield (ndarray):
            Bitfield to be plotted.

        flag_dict (Mapping[int, str]):
            Mapping of flag values to their meanings.  Keys should be
            powers of 2.  For example, {1: "DO_NOT_USE", 2:
            "BAD_GEOLOCATION", 4: "BAD_TIME"}.

        cmap (str):
            Colourmap to use.  This needs to be passed here because it
            needs to be converted to be discrete corresponding to the
            number of unique values.  I recommend to choose a qualitative
            colourmap such as Set1, Set2, or Set3.

        cax (Axes):
            Optional.  If given, put colorbar here.

        pcolor_args (Mapping):
            Extra arguments to be passed to pcolor.

        colorbar_args (Mapping):
            Extra arguments to be passed to colorbar.

        unflagged (str):
            Label to use for unflagged values.  Defaults to "unflagged".

        joiner (str):
            How to join different flags.

    Returns:

        (AxesImage, Colorbar) that were generated
    """

    # 'unique' may give 'masked' as one of the values if bitfield is a
    # masked array; ignore the mask here.  This will still be used for
    # plotting.
    unique_values = np.unique(
        bitfield.data if isinstance(bitfield, np.ma.MaskedArray) else
        bitfield)

    # ensure 0 is always explicitly considered; we want unflagged to occur
    # in the legend always, even if it does not occur.  This ensures that
    # when there are different subplots, unflagged has the same colour in
    # each
    if not 0 in unique_values:
        unique_values = np.concatenate([[0], unique_values])

#    flagdefs = dict(zip(ds["quality_scanline_bitmask"].flag_masks,
#                        ds["quality_scanline_bitmask"].flag_meanings.split(",")))

    # each unique value corresponds to a label that consists of one or
    # more flags, except value 0, which is unflagged
    labels = {v: joiner.join(flag_dict[x] for x in flag_dict.keys() if v&x)
                  or unflagged
                  for v in unique_values}

    # translate all values to integers close to 0, as we are only
    # interested in categories
    #trans = {v:k for (k,v) in enumerate(unique_values)}
    trans = dict(enumerate(unique_values))

    new = bitfield.copy()
    for (to, fr) in trans.items():
        if isinstance(new, np.ma.MaskedArray):
            new.data[new.data==fr] = to
        else:
            new[new==fr] = to

    formatter = FuncFormatter(
        lambda val, loc: labels[trans[val]])
    img = ax.pcolor(X, Y, new,
        cmap=get_cmap(cmap, unique_values.size),
        **pcolor_args)
    cb = ax.figure.colorbar(img, cax=cax,
        ticks=list(trans.keys()),
        format=formatter,
        **colorbar_args)
    img.set_clim(min(trans.keys())-0.5, max(trans.keys())+0.5)

    return (img, cb)
