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
plt.show()

# Create a plot for each typhon style.
style_dir = os.path.dirname(typhon.plots.__file__)
typhon_styles = os.listdir(os.path.join(style_dir, 'stylelib'))
styles = [s.split('.')[0] for s in typhon_styles if s.endswith('.mplstyle')]

for stylename in styles:
    simple_plot(typhon.plots.styles(stylename), stylename)
    plt.show()
