# -*- coding: utf-8 -*-

"""Various physics-related modules."""

import numpy as np

from . import (units, em)
from typhon import constants


__all__ = ['units',
           'em',
           'planck',
           'planck_wavelength',
           ]


def planck(f, T):
    """Calculate black body radiation for given frequency and temperature.

    Parameters:
        f : Frquencies [Hz].
        T : Temperature [K].

    Returns:
        Radiances.

    """
    c = constants.speed_of_light
    h = constants.planck
    k = constants.boltzmann

    return 2 * h * f**3 / (c**2 * (np.exp(h*f/(k*T)) - 1))


def planck_wavelength(l, T):
    """Calculate black body radiation for given wavelength and temperature.

    Parameters:
        l: Wavelength [m].
        T: Temperature [K].

    Returns:
        Radiances.

    """
    c = constants.speed_of_light
    h = constants.planck
    k = constants.boltzmann

    return 2 * h * c**2 / (l**5 * (np.exp(h*c/(l*k*T)) - 1))
