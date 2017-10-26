# -*- coding: utf-8 -*-

"""General functions and classes for manipulating geographical data.
"""

from .array import ArrayGroup

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

    def plot(self, fields, plot_type="worldmap", fig=None, ax=None, **kwargs):
        """

        Args:
            plot_type:
            fields:
            fig:
            ax:
            **kwargs:

        Returns:

        """

        if plot_type == "worldmap":
            ax, scatter = typhon.plots.worldmap(
                self["lat"],
                self["lon"],
                self[fields[0]],
                fig, ax, **kwargs
            )
        else:
            raise ValueError("Unknown plot type: '{}'".format(plot_type))

        return ax
