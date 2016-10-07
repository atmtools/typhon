# -*- coding: utf-8 -*-

"""Functions related to water vapor and its thermodynamic effects
"""
import numpy as np

from typhon import constants


__all__ = [
    'e_eq_ice_mk',
    'e_eq_water_mk',
    'density',
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
        raise Exception('Temperatures must be greater than 0K.')

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
        raise Exception('Temperatures must be greater than 0K.')

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

    """
    return p / (R * T)
