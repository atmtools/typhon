# -*- coding: utf-8 -*-

"""Plot to demonstrate the temperature colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

import typhon.cm


nc = Dataset('_data/test_data.nc')
lon, lat = np.meshgrid(nc.variables['lon'][:], nc.variables['lat'][:])
temp = nc.variables['temp'][:]

fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat=47, llcrnrlon=4,
            urcrnrlat=56, urcrnrlon=16)
m.drawcoastlines()
m.drawcountries()
m.pcolormesh(lon, lat, temp, latlon=True, cmap='temperature')
m.colorbar()
ax.set_xlabel('Longitude', size=16)
ax.set_ylabel('Latitude', size=16)
ax.set_title('Temperature', size=20)

fig.tight_layout()
plt.show()
