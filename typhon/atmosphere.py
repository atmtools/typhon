# -*- coding: utf-8 -*-

"""Functions directly related to atmospheric sciences.
"""
from . import constants
from . import math
from .physics import thermodynamics


__all__ = [
    'iwv',
]


def iwv(vmr, p, T, z):
    """Calculate the integrated water vapor (IWV).

    Parameters:
        vmr (ndarray): Volume mixing ratio,
        p (float or ndarray): Pressue [Pa].
            If type is ndarray, size must match vmr.
        T (float or ndarray): Temperature [K].
            If type is ndarray, size must match vmr.
        z (ndarray): Height [m]. Size must match vmr.

    Returns:
        float: Integrated water vapor [kg/m**2].

    """
    R_v = constants.gas_constant_water_vapor
    rho = thermodynamics.density(p, T, R=R_v)

    return math.integrate_column(vmr * rho, z)
