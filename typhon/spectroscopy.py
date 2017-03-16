# -*- coding: utf-8 -*-

"""Functions directly related to spectroscopy.
"""
import numpy as np
from scipy import interpolate

__all__ = [
    'linewidth',
]


def linewidth(f, a):
    """Calculate the full-width at half maximum (FWHM) of an absorption line.

    Parameters:
        f (ndarray): Frequency grid.
        a (ndarray): Line properties
            (e.g. absorption coefficients or cross-sections).

    Returns:
        float: Linewidth.

    Examples:
        >>> f = np.linspace(0, np.pi, 100)
        >>> a = np.sin(f)**2
        >>> linewidth(f, a)
        1.571048056449009
    """
    s = interpolate.UnivariateSpline(f, a - np.max(a)/2, s=0)
    return float(np.diff(s.roots()))
