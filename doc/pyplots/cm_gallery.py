# -*- coding: utf-8 -*-
"""
Reference for colormaps included with typhon.

The original version of this script [0] is provided by Matplotlib.

This reference example shows all colormaps included with typhon. Note that
any colormap listed here can be reversed by appending "_r" (e.g., "pink_r").
These colormaps are divided into the following categories:

[0] http://matplotlib.org/mpl_examples/color/colormaps_reference.py

"""
import numpy as np
import matplotlib.pyplot as plt
import typhon.cm


cmaps = sorted([x for x in typhon.cm.cmaps.keys() if not x.endswith('_r')])

nrows = len(cmaps)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

fig, axes = plt.subplots(nrows=nrows*2, figsize=(10, len(cmaps)))
fig.subplots_adjust(top=0.99, bottom=0.01, left=0.13, right=0.99)

for ax, name in zip(zip(axes[::2], axes[1::2]), cmaps):
    ax[0].imshow(gradient, aspect='auto',  cmap=plt.get_cmap(name))
    ax[1].imshow(gradient, aspect='auto', interpolation='nearest',
                 cmap=plt.get_cmap(name, lut=11))
    pos = list(ax[0].get_position().bounds)
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3]/2.
    fig.text(x_text, y_text, name, va='center', ha='right', fontsize=12)

# Turn off *all* ticks & spines, not just the ones with colormaps.
for ax in axes:
    ax.set_axis_off()

plt.show()
