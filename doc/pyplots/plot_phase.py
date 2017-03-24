# -*- coding: utf-8 -*-

"""Plot to demonstrate the phase colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import FuncFormatter

import typhon.plots


@FuncFormatter
def degree_formatter(x, pos):
    """Create degree ticklabels for radian data."""
    return '{:.0f}\N{DEGREE SIGN}'.format(np.rad2deg(x))


nc = Dataset('_data/test_data.nc')
nth = 3
lon, lat = np.meshgrid(nc.variables['lon'][::nth], nc.variables['lat'][::nth])
u, v = nc.variables['u'][::nth, ::nth], nc.variables['v'][::nth, ::nth]

wdir = np.arctan2(u, v)

fig, ax = plt.subplots(figsize=(10, 8))
m = Basemap(projection='cyl', resolution='i',
            llcrnrlat=47, llcrnrlon=3,
            urcrnrlat=56, urcrnrlon=16)
m.drawcoastlines()
m.drawcountries()
m.drawmeridians(np.arange(0, 20, 2), labels=[0, 0, 0, 1])
m.drawparallels(np.arange(45, 60, 2), labels=[1, 0, 0, 0])
m.quiver(lon, lat, u, v, wdir, cmap='phase', latlon=True)
# Use our own ticklabel formatter.
m.colorbar(label='Wind direction', format=degree_formatter)

fig.tight_layout()
plt.show()
