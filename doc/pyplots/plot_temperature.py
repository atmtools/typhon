# -*- coding: utf-8 -*-

"""Plot to demonstrate the temperature colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

import typhon


nc = Dataset('_data/test_data.nc')
lon, lat = np.meshgrid(nc.variables['lon'][:], nc.variables['lat'][:])
temp = nc.variables['temp'][:]

fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat=47, llcrnrlon=3,
            urcrnrlat=56, urcrnrlon=16)
m.drawcoastlines()
m.drawcountries()

# Add colors for to land and sea (just to show off).
m.drawmapboundary(fill_color='lightblue', zorder=-1)
m.fillcontinents(color='lightgrey', zorder=0)

m.pcolormesh(lon, lat, temp, latlon=True, cmap='temperature', rasterized=True)
m.drawmeridians(np.arange(0, 20, 2), labels=[0, 0, 0, 1])
m.drawparallels(np.arange(45, 60, 2), labels=[1, 0, 0, 0])
m.colorbar(label='Temperature [K]')

fig.tight_layout()
plt.show()
