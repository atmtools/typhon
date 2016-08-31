# -*- coding: utf-8 -*-

"""Collection of physical constants and conversion factors.
"""
# TODO: Consider 'from scipy.constants import *' to have a wide range of
# pre-defined constants to build on.
import scipy.constants as spc


# Gas constants for dry air and water vapor are taken from
# Wallace and Hobbs Eq. 3.11 and 3.13.
R_d = gas_constant_dry_air = 287.0  # J K^-1 kg^-1
R_v = gas_constant_water_vapor = 461.51  # J K^-1 kg^-1
R = 8.3143  # J K^-1 mol^-1; Wallace and Hobbs page 467
h = planck = spc.Planck  # J s
k = boltzmann = spc.Boltzmann  # J/K
c = speed_of_light = spc.speed_of_light  # m/s
N = 6.02214129e23  # mol^-1 Avogadro constant
ppm = 1e-6  # parts per million
ppb = 1e-9  # parts per billion
ppt = 1e-12  # parts per trillion
nano = 1e-9
micro = 1e-6
milli = 1e-3
centi = 1e-2
hecto = 1e2
kilo = 1e3
giga = 1e9
tera = 1e12
M_d = 28.97  # kg kmol^-1; effective molecular weight of dry air; Wallace and Hobbs
M_w = 18.016  # kg kmol^-1; molecular weight of H2O (Wallace and Hobbs)
KiB = 2**10
KB = 10**3
MiB = 2**20
MB = 10**6
atm = 101325  # Pa
K = 273.15  # offset °C ←→ K
earth_radius = 6.3781e6
