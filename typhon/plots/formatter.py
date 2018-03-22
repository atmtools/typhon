# -*- coding: utf-8 -*-
"""Custom tick formatter. """
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, FuncFormatter

from typhon import constants


__all__ = [
    'set_xaxis_formatter',
    'set_yaxis_formatter',
    'HectoPascalFormatter',
    'HectoPascalLogFormatter',
    'ScalingFormatter',
]


def set_xaxis_formatter(formatter, ax=None):
    """Set given formatter for major and minor xticks."""
    if ax is None:
        ax = plt.gca()

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)


def set_yaxis_formatter(formatter, ax=None):
    """Set given formatter for major and minor yticks."""
    if ax is None:
        ax = plt.gca()

    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)


def ScalingFormatter(scaling=1, fmtstr='{x:g}'):
    """Provide a ticklabel formatter that applies scaling.

    Parameters:
        scaling (float or string): Scaling that is applied to all labels.
            If `str`, try to find corresponding scale in `typhon.constants`.
        fmtstr (str): Format string used to create label text.
            The field `x` is replaced with the scaled label value.

    Returns:
        matplotlib.ticker.FuncFormatter: Formatter.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon.plots import (set_yaxis_formatter, ScalingFormatter)


        y = 1e6 * np.random.randn(100)

        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_ylabel('y')
        ax.set_title('default')

        fig, ax = plt.subplots()
        ax.plot(y)
        set_yaxis_formatter(ScalingFormatter(scaling=1e6))
        ax.set_ylabel('y in millions')
        ax.set_title('float scaling')

        fig, ax = plt.subplots()
        ax.plot(y)
        set_yaxis_formatter(ScalingFormatter(scaling=1e6, fmtstr='{x:g}M'))
        ax.set_ylabel('y')
        ax.set_title('float scaling and custom label')

        fig, ax = plt.subplots()
        ax.plot(y)
        set_yaxis_formatter(ScalingFormatter(scaling='kilo', fmtstr='{x:g}k'))
        ax.set_ylabel('y')
        ax.set_title('string scaling and custom label')

        plt.show()

    """
    # Try to find string scaling as attributes in `typhon.constants`.
    if isinstance(scaling, str):
        scaling = getattr(constants, scaling)

    @FuncFormatter
    def formatter(x, pos):
        return fmtstr.format(x=x / scaling)

    return formatter


def HectoPascalFormatter():
    """Creates hectopascal labels for pascal input.

    Note:
        Simple wrapper for :func:`~typhon.plots.ScalingFormatter`.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import typhon
        from typhon.plots import (set_yaxis_formatter, HectoPascalFormatter)


        p = typhon.math.nlogspace(1000e2, 0.1e2, 50)

        fig, ax = plt.subplots()
        ax.plot(np.exp(p / p[0]), p)
        ax.invert_yaxis()
        set_yaxis_formatter(HectoPascalFormatter())

        plt.show()

    See also:
            :func:`~typhon.plots.HectoPascalLogFormatter`
                Creates logarithmic hectopascal labels for pascal input.
    """
    return ScalingFormatter('hecto')


class HectoPascalLogFormatter(LogFormatter):
    """Creates logarithmic hectopascal labels for pascal input.

    This class can be used to create axis labels on the hectopascal scale for
    values plotted in pascals. It is handy in combination with plotting in
    logscale.

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import typhon
        from typhon.plots import (set_yaxis_formatter, HectoPascalLogFormatter)


        p = typhon.math.nlogspace(1000e2, 0.1e2, 50)  # pressue in Pa

        fig, ax = plt.subplots()
        ax.semilogy(np.exp(p / p[0]), p)
        ax.invert_yaxis()
        set_yaxis_formatter(typhon.plots.HectoPascalLogFormatter())

        plt.show()

    See also:
            :func:`~typhon.plots.HectoPascalFormatter`
                Creates hectopascal labels for pascal input.
    """
    # TODO (lkluft): Hack to preserve automatic toggling to minor ticks for
    # log-scales introduced in matplotlib 2.0.
    # Could be replaced with a decent Formatter subclass in the future.
    def _num_to_string(self, x, vmin, vmax):
        return '{:g}'.format(x / 100)


