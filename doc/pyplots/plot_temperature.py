# -*- coding: utf-8 -*-
"""Plot to demonstrate the temperature colormap. """

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import (LONGITUDE_FORMATTER, LATITUDE_FORMATTER)

from typhon.plots.maps import get_cfeatures_at_scale


# Read air temperature data.
with netCDF4.Dataset('_data/test_data.nc') as nc:
    lon, lat = np.meshgrid(nc.variables['lon'][:], nc.variables['lat'][:])
    temp = nc.variables['temp'][:]

# Create plot with PlateCarree projection.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.set_extent([3, 16, 47, 56])

# Add map "features".
features = get_cfeatures_at_scale(scale='50m')
ax.add_feature(features.BORDERS)
ax.add_feature(features.COASTLINE)
ax.add_feature(features.LAND)
ax.add_feature(features.OCEAN)

# Plot the actual data.
sm = ax.pcolormesh(lon, lat, temp,
                   cmap='temperature',
                   rasterized=True,
                   transform=ccrs.PlateCarree(),
                   )
fig.colorbar(sm, label='Temperature [K]', fraction=0.0328, pad=0.02)

# Add grids and coordinate system.
gl = ax.gridlines(draw_labels=True, color='black')
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabels_top = gl.ylabels_right = False

fig.tight_layout()
plt.show()
