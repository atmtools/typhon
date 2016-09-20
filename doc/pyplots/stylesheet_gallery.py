# -*- coding: utf-8 -*-
"""Generate a gallery to compare all available typhon styles.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import typhon.plots


def simple_plot(style, stylename, **kwargs):
    """Generate a simple plot using a given matplotlib style."""
    plt.style.use('classic')
    plt.style.use(style)

    x = np.linspace(0, 2*np.pi, 20)

    fig, ax = plt.subplots()
    for s in np.linspace(0, np.pi, 10):
        ax.plot(x, np.sin(x+s),
                label=r'$\Delta\omega = {:.2f}$'.format(s),
                marker='.',
                **kwargs
                )
    ax.set_ylabel('y-axis')
    ax.set_xlabel('x-axis')
    ax.set_title(stylename)
    ax.legend()

# Create plot using default styles.
simple_plot('classic', 'classic')

# Create a plot for each available typhon style.
for style in typhon.plots.get_available_styles():
    simple_plot(typhon.plots.styles(style), style)

plt.show()
