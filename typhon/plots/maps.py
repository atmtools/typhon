# -*- coding: utf-8 -*-
"""Functions related to plotting maps. """
import cartopy.crs as ccrs
from matplotlib import pyplot as plt


__all__ = [
    'worldmap',
]


def worldmap(lat, lon, var=None, fig=None, ax=None, projection=None,
             bg=False, **kwargs):
    """Plots the track of a variable on a worldmap.

    Args:
        lat: Array of latitudes.
        lon: Array of longitudes.
        var: Additional array for the variable to plot. If given,
            the track changes the color according to a color map.
        fig: A matplotlib figure object. If not given, the current
            figure is used.
        ax: A matplotlib axis object. If not given, a new axis
            object will be created in the current figure.
        projection: If no axis is given, specify here the cartopy projection.
        bg: If true, a background image will be drawn.
        **kwargs:

    Returns:
        Axis and scatter plot objects.
    """
    # Default keyword arguments to pass to hist2d().
    kwargs_defaults = {
        "cmap": "qualitative1",
        "s": 1,
    }
    kwargs_defaults.update(kwargs)

    if fig is None:
        fig = plt.gcf()

    if projection is None:
        if ax is None:

            projection = ccrs.PlateCarree()
        else:
            projection = ax.projection

    if ax is None:
        ax = fig.add_subplot(111, projection=projection)

    if bg:
        ax.stock_img()

    # It is counter-intuitive but if we want to plot our data with normal
    # latitudes and longitudes, we always have to set the transform to
    # PlateCarree (see https://github.com/SciTools/cartopy/issues/911)
    if var is None:
        scatter_plot = ax.scatter(
            lon, lat, transform=ccrs.PlateCarree(), **kwargs_defaults)
    else:
        scatter_plot = ax.scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(), **kwargs_defaults)
        ax.colorbar(scatter_plot)

    return ax, scatter_plot