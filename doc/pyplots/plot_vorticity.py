# -*- coding: utf-8 -*-

"""Plot to demonstrate the vorticity colormap.
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from typhon.plots import figsize

fig = plt.figure(figsize=figsize(10))
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
surf = ax.plot_surface(X, Y, Z, cmap='vorticity', vmin=-80, vmax=80,
                       rstride=1, cstride=1, linewidth=0)
fig.colorbar(surf)
fig.tight_layout()
plt.show()
