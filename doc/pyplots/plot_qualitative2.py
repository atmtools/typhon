# -*- coding: utf-8 -*-

"""Plot to demonstrate the qualitative2 colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from typhon.cm import mpl_colors


fig, ax = plt.subplots(figsize=(10, 6))
ax.set_prop_cycle(color=mpl_colors('qualitative2', 5))
for c in np.arange(5):
    X = np.random.randn(100)/2
    Y = np.random.randn(100)/2
    ax.plot(X+c, Y+c, linestyle='none', marker='.', markersize=20)

fig.tight_layout()
plt.show()
