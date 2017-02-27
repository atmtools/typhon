# -*- coding: utf-8 -*-
""" Common variables to use within the units subpackage.
"""

from pint import (UnitRegistry, Context)

__all__ = [
    'ureg',
    'radiance_units',
        ]

ureg = UnitRegistry()
ureg.define("micro- = 1e-6 = µ-")

# aid conversion between different radiance units
sp2 = Context("radiance")
sp2.add_transformation(
    "[length] * [mass] / [time] ** 3",
    "[mass] / [time] ** 2",
    lambda ureg, x: x / ureg.speed_of_light)
sp2.add_transformation(
    "[mass] / [time] ** 2",
    "[length] * [mass] / [time] ** 3",
    lambda ureg, x: x * ureg.speed_of_light)
ureg.add_context(sp2)

radiance_units = {
    "si": ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz),
    "ir": ureg.mW / (ureg.m**2 * ureg.sr * ureg.cm**-1)}

# add wavenumber/kayser to spectroscopy contexts for use with pint<0.8
ureg._contexts["spectroscopy"].add_transformation("[length]", "1/[length]",
    lambda ureg, x, **kwargs: 1/x)
ureg._contexts["spectroscopy"].add_transformation("1/[length]", "[length]",
    lambda ureg, x, **kwargs: 1/x)
