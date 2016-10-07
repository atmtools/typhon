# -*- coding: utf-8 -*-

"""Functions related to water vapor and its thermodynamic effects

This module contains wrapper functions to perform calculations with pint
quantities.

"""
from typhon import physics

from typhon.physics.units import constants
from typhon.physics.units.common import ureg


__all__ = [
    'density',
]


def density(p, T, R=None):
    """Wrapper around :func:`typhon.physics.thermodynamics.density`.

    Parameters:
        p (Quantity): Pressure.
        T (Quantity): Temperature.
            If magnitude of T and p is ndarray, sizes must match.
        R (Quantity): Gas constant.

    Returns:
        Quantity: Density [kg/m**3].

    """
    if R is None:
        R = constants.gas_constant_dry_air

    # SI conversion
    p = p.to('pascal')
    T = T.to('kelvin')
    R = R.to('joule / kelvin / kilogram')

    ret = physics.thermodynamics.density(
              p.magnitude, T.magnitude, R.magnitude)

    return ret * ureg('kilogram / meter**3')
