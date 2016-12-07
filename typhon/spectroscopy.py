# -*- coding: utf-8 -*-

"""Functions directly related to spectroscopy.
"""
import numpy as np
from scipy import interpolate

__all__ = [
    'linewidth',
]


def linewidth(f, a):
    """Calculate the FWHM of an absorption line.

    Parameters:
        f (ndarray): Frequency grid.
        a (ndarray): Absorption cross-sections.

    Returns:
        float: Linewidth.
    """
    s = interpolate.UnivariateSpline(f, a - np.max(a)/2, s=0)
    return float(np.diff(s.roots()))
