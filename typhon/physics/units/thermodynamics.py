# -*- coding: utf-8 -*-

"""Functions related to water vapor and its thermodynamic effects

This module contains wrapper functions to perform calculations with pint
quantities.
"""
from typhon import physics

from .common import ureg


__all__ = [
    'density',
]


def density(p, T, R):
    """Wrapper around :func:`typhon.physics.thermodynamics.density`.

    Parameters:
        p (Quantity): Pressure.
        T (Quantity): Temperature.
            If magnitude of T and p is ndarray, sizes must match.
        R (Quantity): Gas constant.

    Returns:
        Quantity: Density [kg/m**3].

    """
    p = p.to('pascal').magnitude
    T = T.to('kelvin').magnitude
    R = R.to('joule / kilogram / kelvin').magnitude

    ret = physics.thermodynamics.density(p, T, R)

    return ret * ureg('kilogram / meter**3')
