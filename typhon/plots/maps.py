# -*- coding: utf-8 -*-
"""Functions related to plotting maps. """
from collections import namedtuple

try:
    import cartopy.crs as ccrs
    from cartopy.feature import NaturalEarthFeature, COLORS
except ImportError:
    raise ImportError('You have to install `cartopy` to use functions '
                      'located in `typhon.plots.maps`.')
else:
    from matplotlib import pyplot as plt


__all__ = [
    'worldmap',
    'get_cfeatures_at_scale',
]


def worldmap(lat, lon, var=None, fig=None, ax=None, projection=None,
             bg=False, draw_grid=False, draw_coastlines=False,
             interpolation=False, **kwargs):
    """Plots the track of a variable on a worldmap.

    Args:
        lat: Array of latitudes.
        lon: Array of longitudes.
        var: Additional array for the variable to plot. If 1-dimensional, the
            variable is plotted as track changing the color according to a
            color map. If 2-dimensional, variable is plotted as contour plot.

        fig: A matplotlib figure object. If not given, the current
            figure is used.
        ax: A matplotlib axis object. If not given, a new axis
            object will be created in the current figure.
        projection: If no axis is given, specify here the cartopy projection.
        bg: If true, a background image will be drawn.
        draw_grid:
        draw_coastlines:
        **kwargs:

    Returns:
        Scatter plot objects.
    """
    # Default keyword arguments to pass to hist2d().
    kwargs_defaults = {
        "cmap": "qualitative1",
        "s": 1,
        # This accelerates the drawing of many points:
        "rasterized": lat.size > 100_000,
        **kwargs
    }

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

    if draw_grid:
        ax.gridlines(draw_labels=True)

    if draw_coastlines:
        ax.coastlines()

    # It is counter-intuitive but if we want to plot our data with normal
    # latitudes and longitudes, we always have to set the transform to
    # PlateCarree (see https://github.com/SciTools/cartopy/issues/911)
    if var is None or len(var.shape) == 1:
        kwargs_defaults = {
            "cmap": "qualitative1",
            "s": 1,
            # This accelerates the drawing of many points:
            "rasterized": lat.size > 100_000,
            **kwargs
        }
        plot = ax.scatter(
            lon, lat, c=var, transform=ccrs.PlateCarree(), **kwargs_defaults
        )
    elif interpolation:
        kwargs_defaults = {
            **kwargs
        }
        plot = ax.contourf(
            lon, lat, var, transform=ccrs.PlateCarree(), **kwargs_defaults
        )
    else:
        kwargs_defaults = {
            **kwargs
        }
        plot = ax.pcolormesh(
            lon, lat, var, transform=ccrs.PlateCarree(), **kwargs_defaults
        )

    return plot


def get_cfeatures_at_scale(scale='110m'):
    """Return a collection of `NaturalEarthFeature` at given scale.

    Parameters:
        scale (str): The dataset scale, i.e. one of ‘10m’, ‘50m’,
            or ‘110m’. Corresponding to 1:10,000,000, 1:50,000,000,
            and 1:110,000,000 respectively.

    Returns:
        collections.namedtuple:
            Collection of :class:`~cartopy.feature.NaturalEarthFeature`

    Examples:
        >>> features = get_cfeatures_at_scale('50m')
        >>> print(features.COASTLINE.scale)
        '50m'

    """
    d = {}

    d['BORDERS'] = NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale=scale,
        edgecolor='black',
        facecolor='none',
    )

    d['STATES'] = NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale=scale,
        edgecolor='black',
        facecolor='none',
    )

    d['COASTLINE'] = NaturalEarthFeature(
        category='physical',
        name='coastline',
        scale=scale,
        edgecolor='black',
        facecolor='none',
    )

    d['LAKES'] = NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale=scale,
        edgecolor='face',
        facecolor=COLORS['water'],
    )

    d['LAND'] = NaturalEarthFeature(
        category='physical',
        name='land',
        scale=scale,
        edgecolor='face',
        facecolor=COLORS['land'],
        zorder=-1,
    )

    d['OCEAN'] = NaturalEarthFeature(
        category='physical',
        name='ocean',
        scale=scale,
        edgecolor='face',
        facecolor=COLORS['water'],
        zorder=-1,
    )

    d['RIVERS'] = NaturalEarthFeature(
        category='physical',
        name='rivers_lake_centerlines',
        scale=scale,
        edgecolor=COLORS['water'],
        facecolor='none',
    )

    NaturalEarthFeatures = namedtuple(
        typename='NaturalEarthFeatures',
        field_names=(
            'BORDERS',
            'STATES',
            'COASTLINE',
            'LAKES',
            'LAND',
            'OCEAN',
            'RIVERS',
        ),
    )

    return NaturalEarthFeatures(**d)
