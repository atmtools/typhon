# -*- coding: utf-8 -*-

"""Functions for handling geographical coordinate systems
and reference ellipsoids.
"""
import numpy as np
from numpy.lib import scimath

from . import constants


__all__ = [
    'constants',
    'ellipsoidmodels',
    'sind',
    'cosd',
    'tand',
    'asind'
]


def sind(x):
    """Sine of argument in degrees."""
    return np.sin(np.deg2rad(x))


def cosd(x):
    """Cosine of argument in degrees."""
    return np.cos(np.deg2rad(x))


def tand(x):
    """Tangent of argument in degrees."""
    return np.tan(np.deg2rad(x))


def asind(x):
    """Inverse sine in degrees."""
    return np.arcsin(np.deg2rad(x))


# TODO: Convert to class?
def ellipsoidmodels(model="WGS84"):
    """Return data for different reference ellipsoids.

    The following models are covered:

        * SphericalEarth     (radius set as constants.earth_radius)
        * WGS84
        * SphericalVenus     (radius same as used in ARTS)
        * SphericalMars      (radius same as used in ARTS)
        * EllipsoidMars
        * SphericalJupiter   (radius same as used in ARTS)

    Parameters:
        model (str): Model ellipsoid.

    Returns:
         tuple: Equatorial radius (r), eccentricity (e)
    """
    data = {
        "SphericalEarth": (constants.earth_radius, 0),
        "WGS84": (6378137, 0.0818191908426),
        "SphericalVenus": (6051800.0, 0),
        "SphericalMars": (3389500.0, 0),
        "EllipsoidMars": (3396190.0, 0.1083),
        "SphericalJupiter": (69911000.0, 0),
        }

    if model in data:
        return data[model]
    else:
        raise Exception('Unknown ellipsoid model "{}".'.format(model))
