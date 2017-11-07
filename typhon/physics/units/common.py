# -*- coding: utf-8 -*-
""" Common variables to use within the units subpackage.
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

from pint import (UnitRegistry, Context)

__all__ = [
    'ureg',
    'radiance_units',
        ]

ureg = UnitRegistry()
ureg.define("micro- = 1e-6 = µ-")

# aid conversion between different radiance units
sp2 = Context("radiance")
# specrad per frequency → specrad per wavenumber
sp2.add_transformation(
    "[length] * [mass] / [time] ** 3",
    "[mass] / [time] ** 2",
    lambda ureg, x, **kwargs: x / ureg.speed_of_light)
# specrad per wavenumber → specrad per frequency
sp2.add_transformation(
    "[mass] / [time] ** 2",
    "[length] * [mass] / [time] ** 3",
    lambda ureg, x, **kwargs: x * ureg.speed_of_light)
#sp2.add_transformation(
#    "[mass] / ([length] * [time] ** 3)",
#    "[length] * [mass] / [time] ** 3",
#    lambda ureg, x, **kwargs: )
def _R_to_bt(ureg, R, srf):
    """For use by pint, do not call directly, use q.to or SRF class."""
    if srf.lookup_table is None:
        srf.make_lookup_table()
    return ureg.Quantity(srf.L_to_T(R), 'K')
def _bt_to_R(ureg, T, srf):
    return ureg.Quantity(srf.blackbody_radiance(T, spectral=True))
_bt_to_R.__doc__ = _R_to_bt.__doc__
sp2.add_transformation(
    "[mass] / [time] ** 2",
    "[temperature]",
    _R_to_bt)
sp2.add_transformation(
    "[temperature]",
    "[mass] / [time] ** 2",
    _bt_to_R)
ureg.add_context(sp2)

radiance_units = {
    "si": ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz),
    "ir": ureg.mW / (ureg.m**2 * ureg.sr * ureg.cm**-1)}

# add wavenumber/kayser to spectroscopy contexts for use with pint<0.8
ureg._contexts["spectroscopy"].add_transformation("[length]", "1/[length]",
    lambda ureg, x, **kwargs: 1/x)
ureg._contexts["spectroscopy"].add_transformation("1/[length]", "[length]",
    lambda ureg, x, **kwargs: 1/x)
