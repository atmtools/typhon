# -*- coding: utf-8 -*-

"""Plot to demonstrate the speed colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

import typhon


nc = Dataset('_data/test_data.nc')
lon, lat = np.meshgrid(nc.variables['lon'][:], nc.variables['lat'][:])
u, v = nc.variables['u'][:], nc.variables['v'][:]

u = u.filled(fill_value=np.nan)
v = v.filled(fill_value=np.nan)

wspeed = np.sqrt(u**2 + v**2)

fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=47, llcrnrlon=4,
            urcrnrlat=56, urcrnrlon=16)
m.drawcoastlines()
m.drawcountries()
m.contourf(lon, lat, wspeed, 10, latlon=True, cmap='speed')
m.colorbar()
ax.set_xlabel('Longitude', size=16)
ax.set_ylabel('Latitude', size=16)
ax.set_title('Wind Speed', size=20)

fig.tight_layout()
plt.show()
