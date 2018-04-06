# -*- coding: utf-8 -*-
"""Plot to demonstrate the Max Planck color palette (Pantone 328).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from typhon.plots import mpl_colors


x = np.linspace(-2, 7, 100)

fig, ax = plt.subplots()
ax.set_prop_cycle(color=mpl_colors('max_planck'))
for i in range(1, 5):
    ax.fill_between(x, norm.pdf(x, loc=i) / i, edgecolor='black')
ax.set_ylim(bottom=0)

fig.tight_layout()
plt.show()
