# -*- coding: utf-8 -*-

"""Functions directly related to atmospheric sciences.
"""
from . import constants
from . import math
from .physics import thermodynamics


__all__ = [
    'iwv',
    'relative_humidity',
]


def iwv(vmr, p, T, z):
    """Calculate the integrated water vapor (IWV).

    Parameters:
        vmr (ndarray): Volume mixing ratio,
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].
        z (ndarray): Height [m]. Size must match vmr.

    Returns:
        float: Integrated water vapor [kg/m**2].
    """
    R_v = constants.gas_constant_water_vapor
    rho = thermodynamics.density(p, T, R=R_v)

    return math.integrate_column(vmr * rho, z)


def relative_humidity(vmr, p, T):
    r"""Calculate relative humidity (RH).

    .. math::
        RH = \frac{VMR \cdot p}{e_s(T)}

    Parameters:
        vmr (ndarray): Volume mixing ratio,
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].

    Returns:
        float: Integrated water vapor [unitless].
    """
    return vmr * p / thermodynamics.e_eq_water_mk(T)
