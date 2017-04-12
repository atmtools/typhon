# -*- coding: utf-8 -*-

"""Functions directly related to atmospheric sciences.
"""
from . import constants
from . import math
from .physics import thermodynamics


__all__ = [
    'iwv',
    'relative_humidity',
    'vmr',
]


def iwv(vmr, p, T, z):
    """Calculate the integrated water vapor (IWV).

    Parameters:
        vmr (float or ndarray): Volume mixing ratio,
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
        vmr (float or ndarray): Volume mixing ratio,
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Relative humiditity [unitless].

    See also:
        :func:`~typhon.atmosphere.vmr`
            Complement function (returns VMR for given RH).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> relative_humidity(0.025, 1013e2, 300)
        0.71604995533615401
    """
    return 2*vmr * p / thermodynamics.e_eq_water_mk(T)


def vmr(RH, p, T):
    r"""Calculate the volume mixing ratio (VMR).

    .. math::
        VMR = \frac{RH \cdot e_s(T)}{p}

    Parameters:
        RH (float or ndarray): Relative humidity.
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Volume mixing ratio [unitless].

    See also:
        :func:`~typhon.atmosphere.relative_humidity`
            Complement function (returns RH for given VMR).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> vmr(0.75, 101300, 300)
        0.026185323887350429
    """
    return RH * thermodynamics.e_eq_water_mk(T) / p
