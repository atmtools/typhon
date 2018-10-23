# -*- coding: utf-8 -*-

"""Functions directly related to atmospheric sciences.
"""
import numpy as np
from scipy.interpolate import interp1d

from typhon import constants
from typhon import math
from typhon.physics import thermodynamics


__all__ = [
    'relative_humidity2vmr',
    'vmr2relative_humidity',
    'integrate_water_vapor',
    'moist_lapse_rate',
    'standard_atmosphere',
    'pressure2height',
]


def relative_humidity2vmr(RH, p, T, e_eq=None):
    r"""Convert relative humidity into water vapor VMR.

    .. math::
        VMR = \frac{RH \cdot e_s(T)}{p}

    Parameters:
        RH (float or ndarray): Relative humidity.
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Volume mixing ratio [unitless].

    See also:
        :func:`~typhon.physics.vmr2relative_humidity`
            Complement function (returns RH for given VMR).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> relative_humidity2vmr(0.75, 101300, 300)
        0.026185323887350429
    """
    if e_eq is None:
        e_eq = thermodynamics.e_eq_water_mk

    return RH * e_eq(T) / p


def vmr2relative_humidity(vmr, p, T, e_eq=None):
    r"""Convert water vapor VMR into relative humidity.

    .. math::
        RH = \frac{VMR \cdot p}{e_s(T)}

    Parameters:
        vmr (float or ndarray): Volume mixing ratio,
        p (float or ndarray): Pressure [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Relative humidity [unitless].

    See also:
        :func:`~typhon.physics.relative_humidity2vmr`
            Complement function (returns VMR for given RH).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> vmr2relative_humidity(0.025, 1013e2, 300)
        0.71604995533615401
    """
    if e_eq is None:
        e_eq = thermodynamics.e_eq_water_mk

    return vmr * p / e_eq(T)


def integrate_water_vapor(vmr, p, T, z, axis=0):
    """Calculate the integrated water vapor (IWV).

    Parameters:
        vmr (float or ndarray): Volume mixing ratio,
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].
        z (ndarray): Height [m]. Size must match vmr.
        axis (int): Axis to intergrate along.

    Returns:
        float: Integrated water vapor [kg/m**2].
    """
    R_v = constants.gas_constant_water_vapor
    rho = thermodynamics.density(p, T, R=R_v)  # Water vapor density.

    return math.integrate_column(vmr * rho, z, axis=axis)


def moist_lapse_rate(p, T, e_eq=None):
    r"""Calculate the moist-adiabatic temperature lapse rate.

    Bohren and Albrecht (Equation 6.111, note the **sign change**):

    .. math::
        \frac{dT}{dz} =
            \frac{g}{c_p} \frac{1 + l_v w_s / RT}{1 + l_v^2 w_s/c_p R_v T^2}

    Parameters:
        p (float or ndarray): Pressure [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Moist-adiabatic lapse rate [K/m].

    Examples:
        >>> moist_lapse_rate(1013.25e2, 288.15)
        0.004728194612232855

    References:
        Bohren C. and Albrecht B., Atmospheric Thermodynamics, p. 287-92
    """
    if e_eq is None:
        e_eq = thermodynamics.e_eq_water_mk

    # Use short formula symbols for physical constants.
    g = constants.earth_standard_gravity
    Lv = constants.heat_of_vaporization
    Rd = constants.gas_constant_dry_air
    Rv = constants.gas_constant_water_vapor
    Cp = constants.isobaric_mass_heat_capacity

    gamma_d = g / Cp  # dry lapse rate
    w_saturated = thermodynamics.vmr2mixing_ratio(e_eq(T) / p)

    lapse = (
        gamma_d * (
            (1 + (Lv * w_saturated) / (Rd * T)) /
            (1 + (Lv**2 * w_saturated) / (Cp * Rv * T**2))
        )
    )

    return lapse


def standard_atmosphere(z, coordinates='height'):
    """International Standard Atmosphere (ISA).

    The temperature profile is defined between 0-85 km (1089 h-0.004 hPa).
    Values exceeding this range are linearly interpolated.

    Parameters:
        z (float or ndarray): Geopotential height above MSL [m]
            or pressure [Pa] (see ``coordinates``).
        coordinates (str): Either 'height' or 'pressure'.

    Returns:
        ndarray: Atmospheric temperature [K].

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        from typhon.plots import (profile_p_log, profile_z)
        from typhon.physics import standard_atmosphere
        from typhon.math import nlogspace


        z = np.linspace(0, 84e3, 100)
        fig, ax = plt.subplots()
        profile_z(z, standard_atmosphere(z), ax=ax)

        p = nlogspace(1000e2, 0.4, 100)
        fig, ax = plt.subplots()
        profile_p_log(p, standard_atmosphere(p, coordinates='pressure'))

        plt.show()

    """
    h = np.array([-610, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
    p = np.array(
        [108_900, 22_632, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734]
    )
    temp = np.array([+19.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.28])

    if coordinates == 'height':
        z_ref = h
    elif coordinates == 'pressure':
        z_ref = np.log(p)
        z = np.log(z)
    else:
        raise ValueError(
            f'"{coordinates}" coordinate is unsupported. '
            'Use "height" or "pressure".')

    return interp1d(z_ref, temp + constants.K, fill_value='extrapolate')(z)


def pressure2height(p, T=None):
    r"""Convert pressure to height based on the hydrostatic equilibrium.

    .. math::
       z = \int -\frac{\mathrm{d}p}{\rho g}

    Parameters:
        p (ndarray): Pressure [Pa].
        T (ndarray): Temperature [K].
            If ``None`` the standard atmosphere is assumed.

    See also:
        :func:`~typhon.physics.standard_atmosphere`:
            Standard atmospheric temperature profile as function of pressure.

    Returns:
        ndarray: Height [m].
    """
    if T is None:
        T = standard_atmosphere(p, coordinates='pressure')

    dp = np.diff(p)
    dp = np.hstack([dp, dp[-1]])
    rho = thermodynamics.density(p, T)
    g = constants.earth_standard_gravity

    return np.cumsum(-dp / (rho * g))
