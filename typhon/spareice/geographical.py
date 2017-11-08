# -*- coding: utf-8 -*-

"""General functions and classes for manipulating geographical data.
"""

from .array import ArrayGroup
import typhon.plots

__all__ = [
    'GeoData',
]


class GeoData(ArrayGroup):
    """A specialised ArrayGroup for geographical indexed data (with
    longitude, latitude and time field).

    Still under development. TODO.
    """

    def __len__(self):
        if "time" not in self:
            return 0

        return len(self["time"])

    def plot_worldmap(self, var, fig=None, ax=None, **kwargs):
        """Create a scatter plot projected on a world map.

        Args:
            var: Variable name that should be plotted.
            fig: (optional) A matplotlib figure object.
            ax: (optional) A matplotlib axis object.
            **kwargs: Additional arguments for :func:`typhon.plots.worldmap`.

        Returns:
            Axis on which the data was plotted.
        """

        ax, scatter = typhon.plots.worldmap(
            self["lat"],
            self["lon"],
            self[var],
            fig, ax, **kwargs
        )

        return ax
