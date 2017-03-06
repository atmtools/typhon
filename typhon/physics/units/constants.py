# -*- coding: utf-8 -*-

"""Collection of physical constants and conversion factors.

The magnitudes of the defined constants are taken from
:mod:`typhon.constants`.

This module adds units defined with pint's UnitRegistry..

Physical constants
==================

============================  ============================
``g``                         Earth standard gravity
``h``                         Planck constant
``k``                         Boltzmann constant
``c``                         Speed of light
``N_A``                       Avogadro constant
``R``                         Universal gas constant
``molar_mass_dry_air``        Molar mass for dry air
``molar_mass_water``          Molar mass for water vapor
``gas_constant_dry_air``      Gas constant for dry air
``gas_constant_water_vapor``  Gas constant for water vapor
============================  ============================

Mathematical constants
======================

==========  ============
``golden``  Golden ratio
==========  ============

SI prefixes
===========

=========  ================
``yotta``  :math:`10^{24}`
``zetta``  :math:`10^{21}`
``exa``    :math:`10^{18}`
``peta``   :math:`10^{15}`
``tera``   :math:`10^{12}`
``giga``   :math:`10^{9}`
``mega``   :math:`10^{6}`
``kilo``   :math:`10^{3}`
``hecto``  :math:`10^{2}`
``deka``   :math:`10^{1}`
``deci``   :math:`10^{-1}`
``centi``  :math:`10^{-2}`
``milli``  :math:`10^{-3}`
``micro``  :math:`10^{-6}`
``nano``   :math:`10^{-9}`
``pico``   :math:`10^{-12}`
``femto``  :math:`10^{-15}`
``atto``   :math:`10^{-18}`
``zepto``  :math:`10^{-21}`
=========  ================

Non-SI ratios
=============

=======  =====================================
``ppm``  :math:`10^{-6}` `parts per million`
``ppb``  :math:`10^{-9}` `parts per billion`
``ppt``  :math:`10^{-12}` `parts per trillion`
=======  =====================================

Binary prefixes
===============

=================  ==============
``kibi``, ``KiB``  :math:`2^{10}`
``mebi``, ``MiB``  :math:`2^{20}`
``gibi``           :math:`2^{30}`
``tebi``           :math:`2^{40}`
``pebi``           :math:`2^{50}`
``exbi``           :math:`2^{60}`
``zebi``           :math:`2^{70}`
``yobi``           :math:`2^{80}`
=================  ==============

=================  ==============
``KB``             :math:`10^3`
``MB``             :math:`10^6`
=================  ==============

Earth characteristics
=====================

================  ===================
``earth_mass``    Earth mass
``earth_radius``  Earth radius
``atm``           Standard atmosphere
================  ===================

"""
import numpy as np

from typhon import constants
from typhon.physics.units.common import ureg


# Physcial constants
g = earth_standard_gravity = constants.g * ureg('m / s**2')
h = planck = constants.planck * ureg.joule
k = boltzmann = constants.boltzmann * ureg('J / K')
c = speed_of_light = constants.speed_of_light * ureg('m / s')
N_A = avogadro = N = constants.avogadro * ureg('1 / mol')
R = gas_constant = constants.gas_constant * ureg('J * mol**-1 * K**-1')
molar_mass_dry_air = 28.9645e-3 * ureg('kg / mol')
molar_mass_water = 18.01528e-3 * ureg('kg / mol')
gas_constant_dry_air = R / molar_mass_dry_air  # J K^-1 kg^-1
gas_constant_water_vapor = R / molar_mass_water  # J K^-1 kg^-1

# Mathematical constants
golden = golden_ratio = (1 + np.sqrt(5)) / 2

# SI prefixes
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21

# Non-SI ratios
ppm = 1e-6  # parts per million
ppb = 1e-9  # parts per billion
ppt = 1e-12  # parts per trillion

# Binary prefixes
kibi = KiB = 2**10
mebi = MiB = 2**20
gibi = 2**30
tebi = 2**40
pebi = 2**50
exbi = 2**60
zebi = 2**70
yobi = 2**80

KB = 10**3
MB = 10**6

# Earth characteristics
earth_mass = constants.earth_mass * ureg.kg
earth_radius = constants.earth_radius * ureg.m
atm = atmosphere = constants.atm * ureg.pascal
