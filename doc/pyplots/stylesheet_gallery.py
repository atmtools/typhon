# -*- coding: utf-8 -*-
"""Generate a gallery to compare all available typhon styles.
"""
import numpy as np
import matplotlib.pyplot as plt
from typhon.plots import styles


def simple_plot(stylename):
    """Generate a simple plot using a given matplotlib style."""
    x = np.linspace(0, np.pi, 20)

    fig, ax = plt.subplots()
    for s in np.linspace(0, np.pi / 2, 12):
        ax.plot(x, np.sin(x+s),
                label=r'$\Delta\omega = {:.2f}$'.format(s),
                marker='.',
                )
    ax.set_ylabel('y-axis')
    ax.set_xlabel('x-axis')
    ax.set_title(stylename)
    ax.grid()
    ax.legend()


# Create plot using default styles.
simple_plot('matplotlib 2.0')

# Create a plot for each available typhon style.
for style_name in styles.available:
    with plt.style.context(styles(style_name)):
        simple_plot(style_name)

plt.show()
