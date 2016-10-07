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
    "ir": ureg.mW / (ureg.m**2 * ureg.sr * (1 / ureg.cm))}
