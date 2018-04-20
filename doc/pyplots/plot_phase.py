# -*- coding: utf-8 -*-
"""Plot to demonstrate the phase colormap. """

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER, LATITUDE_FORMATTER)
from matplotlib.ticker import FuncFormatter

from typhon.plots.maps import get_cfeatures_at_scale


@FuncFormatter
def degree_formatter(x, pos):
    """Create degree ticklabels for radian data."""
    return '{:.0f}\N{DEGREE SIGN}'.format(np.rad2deg(x))


# Read wind data.
with netCDF4.Dataset('_data/test_data.nc') as nc:
    nth = 5
    lon, lat = np.meshgrid(nc.variables['lon'][::nth],
                           nc.variables['lat'][::nth])
    u, v = nc.variables['u'][::nth, ::nth], nc.variables['v'][::nth, ::nth]

wdir = np.arctan2(u, v) + np.pi

# Create plot with PlateCarree projection.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent([3, 16, 47, 56])

# Add map "features".
features = get_cfeatures_at_scale(scale='50m')
ax.add_feature(features.BORDERS)
ax.add_feature(features.COASTLINE)
ax.add_feature(features.OCEAN)

# Plot the actual data.
sm = ax.quiver(lon, lat, u, v, wdir,
               cmap=plt.get_cmap('phase', 8),
               transform=ccrs.PlateCarree(),
               )

# Add custom colorbar for wind directions (e.g. tick format).
cb = fig.colorbar(sm, label='Wind direction', format=degree_formatter,
                  fraction=0.0328, pad=0.02)
cb.set_ticks(np.linspace(0, 2 * np.pi, 9))

# Add coordinate system without drawing gridlines.
gl = ax.gridlines(draw_labels=True, color='none')
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabels_top = gl.ylabels_right = False

fig.tight_layout()
plt.show()
