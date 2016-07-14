# -*- coding: utf-8 -*-

"""Plot to demonstrate the phase colormap
"""

import numpy as np
import matplotlib.pyplot as plt

from typhon.cm import mpl_colors
from typhon.plots import figsize


x = np.linspace(0, 4 * np.pi, 200)
phase_shifts = np.linspace(0, 2 * np.pi, 10)

fig, ax = plt.subplots(figsize=figsize(10))
ax.set_prop_cycle(color=mpl_colors('phase', len(phase_shifts)))
for p in phase_shifts:
    ax.plot(x, np.sin(x + p), lw=2)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.2, 1.2)

fig.tight_layout()
plt.show()
