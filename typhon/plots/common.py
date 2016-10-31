# -*- coding: utf-8 -*-

"""Utility functions related to plotting.
"""
import glob
import numpy as np
import os

from typhon import constants


__all__ = ['figsize',
           'styles',
           'get_available_styles',
           'get_subplot_arrangement',
           ]


def figsize(w, portrait=False):
    """Return a figure size matching the golden ratio.

    This function takes a figure width and returns a tuple
    representing width and height in the golden ratio.
    Results can be returned for portrait orientation.

    Parameters:
        w (float): Figure width.
        portrait (bool): Return size for portrait format.

    Return:
        tuple: Figure width and size.

    Examples:
        >>> import typhon.plots
        >>> typhon.plots.figsize(1)
        (1, 0.61803398874989479)

        >>> typhon.plots.figsize(1, portrait=True)
        (1, 1.6180339887498949)
    """
    phi = constants.golden_ratio
    return (w, w * phi) if portrait else (w, w / phi)


def get_subplot_arrangement(n):
    """Get efficient (nrow, ncol) for n subplots

    If we want to put `n` subplots in a square-ish/rectangular
    arrangement, how should we arrange them?

    Returns (⌈√n⌉, ‖√n‖)
    """
    return (int(np.ceil(np.sqrt(n))),
            int(np.round(np.sqrt(n))))


def styles(name):
    """Return absolute path to typhon stylesheet.

    Matplotlib stylesheets can be passed via their full path. This function
    takes a style name and returns the absolute path to the typhon stylesheet.

    Parameters:
        name (str): Style name.

    Returns:
        str: Absolute path to stylesheet.

    Example:
        Use typhon style for matplotlib plots.

        >>> import matplotlib.pyplot as plt
        >>> plt.style.use(styles('typhon'))

    """
    stylelib_dir = os.path.join(os.path.dirname(__file__), 'stylelib')

    return os.path.join(stylelib_dir, name + '.mplstyle')


def get_available_styles():
    """Return list of names of all styles shipped with typhon.

    Returns:
        list[str]: List of available styles.

    """
    stylelib_dir = os.path.join(os.path.dirname(__file__), 'stylelib')
    pattern = os.path.join(stylelib_dir, '*.mplstyle')

    return [os.path.splitext(os.path.basename(s))[0]
            for s in glob.glob(pattern)]
