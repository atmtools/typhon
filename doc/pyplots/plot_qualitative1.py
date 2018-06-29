# -*- coding: utf-8 -*-

"""Plot to demonstrate the qualitative1 colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from typhon.plots import (figsize, cmap2rgba)


x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=figsize(10))
ax.set_prop_cycle(color=cmap2rgba('qualitative1', 7))
for c in np.arange(1, 8):
    ax.plot(x, (15 + x) * c, linewidth=3)
ax.set_xlim(x.min(), x.max())

fig.tight_layout()
plt.show()
