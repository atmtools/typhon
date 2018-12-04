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
import typhon.plots  # enable usage of typhon colormaps


cmaps = [
    ('Sequential', True, [
        'density', 'speed', 'temperature', 'phase']),
    ('Diverging', True, [
        'difference', 'velocity', 'vorticity']),
    ('Qualitative', False, [
        'material', 'max_planck', 'qualitative1', 'qualitative2', 'uhh']),
]


def plot_color_gradients(cmap_category, cmaps, add_discrete=False):
    nrows = len(cmaps)
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    if add_discrete:
        fig, axes = plt.subplots(nrows=nrows * 2, figsize=(10, len(cmaps)))
        fig.subplots_adjust(top=0.99, bottom=0.01, left=0.13, right=0.99)

        for ax, name in zip(zip(axes[::2], axes[1::2]), cmaps):
            ax[0].imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            ax[1].imshow(gradient, aspect='auto', interpolation='nearest',
                         cmap=plt.get_cmap(name, lut=11))
            pos = list(ax[0].get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.
            fig.text(x_text, y_text, name, va='center', ha='right',
                     fontsize=12)
    else:
        fig, axes = plt.subplots(nrows=nrows, figsize=(10, len(cmaps) / 2))
        fig.subplots_adjust(top=0.99, bottom=0.01, left=0.13, right=0.99)

        for ax, name in zip(axes, cmaps):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.
            fig.text(x_text, y_text, name, va='center', ha='right',
                     fontsize=12)

    axes[0].set_title(cmap_category, fontsize='large')

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()


for cmap_category, dflag, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list, add_discrete=dflag)

plt.show()
