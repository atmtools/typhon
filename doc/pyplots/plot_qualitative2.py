# -*- coding: utf-8 -*-

"""Plot to demonstrate the qualitative2 colormap.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from typhon.plots import (figsize, cmap2rgba)


# Create an iterator to conveniently change the marker in the following plot.
markers = (m for m in Line2D.filled_markers)

fig, ax = plt.subplots(figsize=figsize(10))
ax.set_prop_cycle(color=cmap2rgba('qualitative2', 7))
for c in np.arange(7):
    X = np.random.randn(100) / 2
    Y = np.random.randn(100) / 2
    ax.plot(X+c, Y+c, linestyle='none', marker=next(markers), markersize=10)

fig.tight_layout()
plt.show()
