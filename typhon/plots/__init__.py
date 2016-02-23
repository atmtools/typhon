# -*- coding: utf-8 -*-
"""
This module provides functions related to plot or to plot data.

"""

import numpy as np


__all__ = ['figsize']

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
        >>> typhon.cm.figsize(1)
        (1, 0.61803398874989479)

        >>> typhon.cm.figsize(1, portrait=True)
        (1, 1.6180339887498949)
    """
    phi = 0.5 * (np.sqrt(5) + 1)
    return (w, w * phi) if portrait else (w, w / phi)
