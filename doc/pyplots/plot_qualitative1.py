# -*- coding: utf-8 -*-

"""Plot to demonstrate the qualitative1 colormap.
"""

import numpy as np
import matplotlib.pyplot as plt

from typhon.cm import mpl_colors


x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_prop_cycle(color=mpl_colors('qualitative1', 20))
for c in np.arange(20):
    ax.plot(x, (15 + x) * c, linewidth=3)

fig.tight_layout()
plt.show()
