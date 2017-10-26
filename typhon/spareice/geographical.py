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

    def plot_worldmap(self, field, fig=None, ax=None, **kwargs):
        """

        Args:
            plot_type:
            fields:
            fig:
            ax:
            **kwargs:

        Returns:

        """

        ax, scatter = typhon.plots.worldmap(
            self["lat"],
            self["lon"],
            self[field],
            fig, ax, **kwargs
        )

        return ax
