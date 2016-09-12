# -*- coding: utf-8 -*-

"""Collection of physical constants and conversion factors.
"""
import numpy as np
import scipy.constants as spc


# Physcial constants
h = planck = spc.Planck  # J s
k = boltzmann = spc.Boltzmann  # J/K
c = speed_of_light = spc.speed_of_light  # m/s
R = gas_constant = spc.gas_constant
N_A = Avogadro = N = spc.Avogadro  # mol^-1 Avogadro constant

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

# Constants from Wallace and Hobbs
# Gas constants for dry air (Eq. 3.11)
R_d = gas_constant_dry_air = 287.0  # J K^-1 kg^-1
# Gas constant for water vapor (Eq. 3.13)
R_v = gas_constant_water_vapor = 461.51  # J K^-1 kg^-1
# Effective molecular weight of dry air
M_d = 28.97  # kg kmol^-1
# Molecular weight of H2O
M_w = 18.016  # kg kmol^-1

# Miscellaneous
K = zero_Celsius = 273.15  # offset °C ←→ K
atm = atmosphere = 101325  # Pa
earth_radius = 6.3781e6
