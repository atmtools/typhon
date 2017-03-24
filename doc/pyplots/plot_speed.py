# -*- coding: utf-8 -*-

"""Plot to demonstrate the speed colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

import typhon.plots


nc = Dataset('_data/test_data.nc')
lon, lat = np.meshgrid(nc.variables['lon'][:], nc.variables['lat'][:])
u, v = nc.variables['u'][:], nc.variables['v'][:]

wspeed = np.hypot(u, v)

fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat=47, llcrnrlon=3,
            urcrnrlat=56, urcrnrlon=16)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0, 20, 2), labels=[0, 0, 0, 1])
m.drawparallels(np.arange(45, 60, 2), labels=[1, 0, 0, 0])
m.pcolormesh(lon, lat, wspeed, latlon=True, cmap=plt.get_cmap('speed', lut=10))
m.colorbar(label='Wind speed [m/s]')

fig.tight_layout()
plt.show()
