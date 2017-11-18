# -*- coding: utf-8 -*-

"""Functions related to water vapor and its thermodynamic effects
"""
import numpy as np

from typhon import constants


__all__ = [
    'e_eq_ice_mk',
    'e_eq_water_mk',
    'density',
    'mixing_ratio2specific_humidity',
    'mixing_ratio2vmr',
    'specific_humidity2mixing_ratio',
    'specific_humidity2vmr',
    'vmr2mixing_ratio',
    'vmr2specific_humidity',
]


def e_eq_ice_mk(T):
    r"""Calculate the equilibrium water vapor pressure over ice.

    Equilibrium water vapor pressure over ice using Murphy and Koop 2005
    parameterization formula.

    .. math::
        \ln(e_i) = 9.550426
                   - \frac{5723.265}{T}
                   + 3.53068 \cdot \ln(T)
                   - 0.00728332 \cdot T

    Parameters:
        T (float or ndarray): Temperature in [K].

    Returns:
        float or ndarray: Equilibrium water vapor pressure over ice in [Pa].

    References:
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
        ice and supercooled water for atmospheric applications,
        Quarterly Journal of the Royal Meteorological Society 131(608):
        1539–1565. doi:10.1256/qj.04.94

    """
    if np.any(T <= 0):
        raise Exception('Temperatures must be larger than 0 Kelvin.')

    # Give the natural log of saturation vapor pressure over ice in Pa
    e = 9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T

    return np.exp(e)


def e_eq_water_mk(T):
    r"""Calculate the equilibrium water vapor pressure over water.

    Equilibrium water vapor pressure over water using Murphy and
    Koop 2005 parameterization formula.

    .. math::
        \ln(e_w) &= 54.842763 - \frac{6763.22}{T} - 4.21 \cdot \ln(T) \\
                    &+ 0.000367 \cdot T
                    + \tanh \left(0.0415 \cdot (T - 218.8)\right) \\
                    &\cdot \left(53.878 - \frac{1331.22}{T}
                                 - 9.44523 \cdot \ln(T)
                                 + 0.014025 \cdot T \right)

    Parameters:
        T (float or ndarray): Temperature in [K].

    Returns:
        float or ndarray: Equilibrium water vapor pressure over water in [Pa].

    References:
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
        ice and supercooled water for atmospheric applications,
        Quarterly Journal of the Royal Meteorological Society 131(608):
        1539–1565. doi:10.1256/qj.04.94

    """
    if np.any(T <= 0):
        raise Exception('Temperatures must be larger than 0 Kelvin.')

    # Give the natural log of saturation vapor pressure over water in Pa

    e = (54.842763
         - 6763.22 / T
         - 4.21 * np.log(T)
         + 0.000367 * T
         + np.tanh(0.0415 * (T - 218.8))
         * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))

    return np.exp(e)


def density(p, T, R=constants.gas_constant_dry_air):
    r"""Calculates gas density by ideal gas law.

    .. math::
        \rho = \frac{p}{R \cdot T}

    Parameters:
        p (float or ndarray): Pressure [Pa.]
        T (float or ndarray): Temperature [K].
            If type of T and p is ndarray, size must match p.
        R (float): Gas constant [J K^-1 kg^-1].
            Default is gas constant for dry air.

    Returns:
        float or ndarray: Density [kg/m**3].

    See also:
        :mod:`typhon.constants`
            Module containing universal gas constant as well
            as gas constants for dry air and water vapor.

    Examples:
        >>> density(1013e2, 300)
        1.1763056653021122
    """
    return p / (R * T)


def mixing_ratio2specific_humidity(w):
    r"""Convert mass mixing ratio to specific humidity.

    .. math::
        q = \frac{w}{1 + w}

    Parameters:
        w (float or ndarray): Mass mixing ratio.

    Returns:
        float or ndarray: Specific humidity.

    Examples:
        >>> mixing_ratio2specific_humidity(0.02)
        0.0196078431372549
    """
    return w / (1 + w)


def mixing_ratio2vmr(w):
    r"""Convert mass mixing ratio to volume mixing ratio.

    .. math::
        x = \frac{w}{w + \frac{M_w}{M_d}}

    Parameters:
        w (float or ndarray): Mass mixing ratio.

    Returns:
        float or ndarray: Volume mixing ratio.

    Examples:
        >>> mixing_ratio2vmr(0.02)
        0.03115371853180794
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return w / (w + Mw / Md)


def specific_humidity2mixing_ratio(q):
    r"""Convert specific humidity to mass mixing ratio.

    .. math::
        w = \frac{q}{1 - q}

    Parameters:
        q (float or ndarray): Specific humidity.

    Returns:
        float or ndarray: Mass mixing ratio.

    Examples:
        >>> specific_humidity2mixing_ratio(0.02)
        0.020408163265306124
    """
    return q / (1 - q)


def specific_humidity2vmr(q):
    r"""Convert specific humidity to volume mixing ratio.

    .. math::
        x = \frac{q}{(1 - q) \frac{M_w}{M_d} + q}

    Parameters:
        q (float or ndarray): Specific humidity.

    Returns:
        float or ndarray: Volume mixing ratio.

    Examples:
        >>> specific_humidity2vmr(0.02)
        0.03176931009073226
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return q / ((1 - q) * Mw / Md + q)


def vmr2mixing_ratio(x):
    r"""Convert volume mixing ratio to mass mixing ratio.

    .. math::
        w = \frac{x}{1 - x} \frac{M_w}{M_d}

    Parameters:
        x (float or ndarray): Volume mixing ratio.

    Returns:
        float or ndarray: Mass mixing ratio.

    Examples:
        >>> vmr2mixing_ratio(0.04)
        0.025915747437955664
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return x / (1 - x) * Mw / Md


def vmr2specific_humidity(x):
    r"""Convert volume mixing ratio to specific humidity.

    .. math::
        q = \frac{x}{(1 - x) \frac{M_d}{M_w} + x}

    Parameters:
        x (float or ndarray): Volume mixing ratio.

    Returns:
        float or ndarray: Specific humidity.

    Examples:
        >>> vmr2specific_humidity(0.04)
        0.025261087474946833
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return x / ((1 - x) * Md / Mw + x)
