# -*- coding: utf-8 -*-
"""Plot to demonstrate the velocity colormap. """

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import cartopy.crs as ccrs

from typhon.plots import (center_colorbar, get_cfeatures_at_scale)


# Read wind speed data.
with netCDF4.Dataset('_data/test_data.nc') as nc:
    lon, lat = np.meshgrid(nc.variables['lon'][:], nc.variables['lat'][:])
    v = nc.variables['v'][:]

# Create plot with PlateCarree projection.
fig, ax = plt.subplots(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([3, 16, 47, 56])

# Add map "features".
features = get_cfeatures_at_scale(scale='50m')
ax.add_feature(features.COASTLINE)
ax.add_feature(features.BORDERS)
ax.add_feature(features.LAND)

# Plot the actual data.
sm = ax.pcolormesh(lon, lat, v,
                   cmap='velocity',
                   rasterized=True,
                   transform=ccrs.PlateCarree(),
                   )

cb = fig.colorbar(sm, label='Meridional wind [m/s]', fraction=0.0328, pad=0.02)
center_colorbar(cb)

fig.tight_layout()
plt.show()
