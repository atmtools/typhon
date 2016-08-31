# -*- coding: utf-8 -*-

"""Functions directly related to atmospheric sciences.
"""
from . import constants
from . import math
from . import thermodynamics


__all__ = [
    'iwp',
]


def iwp(vmr, p, T, z):
    """Calculate the integrated water vapor path.

    Parameters:
        vmr: Volume mixing ratio,
        p: Pressue [Pa].
        T: Temperature [K].
        z: Height [m].

    Returns:
        Integrated water vapor path [kg/m**2].

    """
    R_v = constants.gas_constant_water_vapor
    rho = thermodynamics.density(p, T, R=R_v)

    return math.integrate_column(vmr * rho, z)
