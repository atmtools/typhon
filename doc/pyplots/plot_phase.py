# -*- coding: utf-8 -*-

"""Plot to demonstrate the phase colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

import typhon.cm

nc = Dataset('_data/test_data.nc')
nth = 3
lon, lat = np.meshgrid(nc.variables['lon'][::nth], nc.variables['lat'][::nth])
u, v = nc.variables['u'][::nth, ::nth], nc.variables['v'][::nth, ::nth]

wdir = np.arctan2(u, v)

fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat=47, llcrnrlon=4,
            urcrnrlat=56, urcrnrlon=16)
m.drawcoastlines()
m.drawcountries()
m.quiver(lon, lat, u, v, wdir, cmap='phase', latlon=True)
ax.set_xlabel('Longitude', size=16)
ax.set_ylabel('Latitude', size=16)
ax.set_title('Wind Direction', size=20)

fig.tight_layout()
plt.show()
