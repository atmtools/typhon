# -*- coding: utf-8 -*-

"""Functions directly related to atmospheric sciences.
"""
from numbers import Number
import numpy as np
from scipy.interpolate import interp1d

from typhon import constants
from typhon import math


__all__ = [
    'relative_humidity2vmr',
    'vmr2relative_humidity',
    'column_relative_humidity',
    'integrate_water_vapor',
    'moist_lapse_rate',
    'standard_atmosphere',
    'pressure2height',
    'e_eq_ice_mk',
    'e_eq_water_mk',
    'e_eq_mixed_mk',
    'density',
    'mixing_ratio2specific_humidity',
    'mixing_ratio2vmr',
    'specific_humidity2mixing_ratio',
    'specific_humidity2vmr',
    'vmr2mixing_ratio',
    'vmr2specific_humidity',
    'water_vapor_pressure2specific_humidity',
]


def relative_humidity2vmr(RH, p, T, e_eq=None):
    r"""Convert relative humidity into water vapor VMR.

    .. math::
        x = \frac{\mathrm{RH} \cdot e_s(T)}{p}

    Note:
        By default, the relative humidity is calculated with respect to
        saturation over liquid water in accordance to the WMO standard for
        radiosonde observations.
        You can use :func:`~typhon.physics.e_eq_mixed_mk` to calculate
        relative humidity with respect to saturation over the mixed-phase
        following the IFS model documentation.

    Parameters:
        RH (float or ndarray): Relative humidity.
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Volume mixing ratio [unitless].

    See also:
        :func:`~typhon.physics.vmr2relative_humidity`
            Complement function (returns RH for given VMR).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> relative_humidity2vmr(0.75, 101300, 300)
        0.026185323887350429
    """
    if e_eq is None:
        e_eq = e_eq_water_mk

    return RH * e_eq(T) / p


def vmr2relative_humidity(vmr, p, T, e_eq=None):
    r"""Convert water vapor VMR into relative humidity.

    .. math::
        \mathrm{RH} = \frac{x \cdot p}{e_s(T)}

    Note:
        By default, the relative humidity is calculated with respect to
        saturation over liquid water in accordance to the WMO standard for
        radiosonde observations.
        You can use :func:`~typhon.physics.e_eq_mixed_mk` to calculate
        relative humidity with respect to saturation over the mixed-phase
        following the IFS model documentation.

    Parameters:
        vmr (float or ndarray): Volume mixing ratio,
        p (float or ndarray): Pressure [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Relative humidity [unitless].

    See also:
        :func:`~typhon.physics.relative_humidity2vmr`
            Complement function (returns VMR for given RH).
        :func:`~typhon.physics.e_eq_water_mk`
            Used to calculate the equilibrium water vapor pressure.

    Examples:
        >>> vmr2relative_humidity(0.025, 1013e2, 300)
        0.71604995533615401
    """
    if e_eq is None:
        e_eq = e_eq_water_mk

    return vmr * p / e_eq(T)



def column_relative_humidity(q, p, t, axis=0):
        r"""Convert specific humidity, pressure and temperature into column relative humidity.

    .. math::
        \mathrm{CRH} = \frac{IVW}{IVW_{saturated}}

    The integrated water vapour (IVW) is caluculated using ty.physics.integrate_water_vapor(vmr, p).
    vmr is calulated via ty.physics.specific_humidity2vmr(q).
    To calculate the saturated intergrated water vapour (IVWS) the saturated mixing (qs) ratio is needed which is calculated
    using ty.physics.specific_humidity(es,p). Doing that the saturated water vapour pressure (es) needs to be determined
    via ty.physics.e_eq_mixed_mk(t).

    Parameters:
        q (float or ndarray): Specific humidity,
        p (float or ndarray): Pressure [Pa],
        t (float or ndarray): Temperature [K].
        

    Returns:
        float or ndarray: Column relative humidity.
    """
        # ivw
        dim = list(q.shape)
        # q to vmr
        vmr = specific_humidity2vmr(q)

        # vmr to integrated water vapour content
        ivw = integrate_water_vapor(vmr, p, axis=axis)

        # ivws
        # t to es
        es = e_eq_mixed_mk(t)
        es.shape = (dim)
        # es to qs 
        if len(dim) == 1:
            l = len(es)
        else:
            l = len(es[axis])
        # qs = specific_humidity(es, ps)
        qs = np.zeros(dim)
        es = es.swapaxes(0,axis)
        qs = qs.swapaxes(0,axis)
        for i in range(0,l):
            qs[i] = water_vapor_pressure2specific_humidity(es[i], p[i])
        es = es.swapaxes(axis,0)
        qs = qs.swapaxes(axis,0)
        # qs to vmrs
        vmrs = specific_humidity2vmr(qs)

        # vmrs to integrated saturated water vapour content
        ivws = integrate_water_vapor(vmrs, p, axis=axis)

        crh = ivw/ivws
        return crh

def integrate_water_vapor(vmr, p, T=None, z=None, axis=0):
    r"""Calculate the integrated water vapor (IWV).

    The basic implementation of the function assumes the atmosphere
    to be in hydrostatic equilibrium.  The IWV is calculated as follows:

    .. math::
        \mathrm{IWV} = -\frac{1}{g} \int q(p)\,\mathrm{d}p

    For non-hydrostatic atmospheres, additional information
    on temperature and height are needed:

    .. math::
        \mathrm{IWV} = \int \rho_v(z)\,\mathrm{d}z

    Parameters:
        vmr (float or ndarray): Volume mixing ratio,
        p (float or ndarray): Pressue [Pa].
        T (float or ndarray): Temperature [K] (see ``z``).
        z (float or ndarray): Height [m]. For non-hydrostatic calculation
            both ``T`` and ``z`` have to be passed.
        axis (int): Axis to integrate along.

    Returns:
        float: Integrated water vapor [kg/m**2].
    """
    if T is None and z is None:
        # Calculate IWV assuming hydrostatic equilibrium.
        q = vmr2specific_humidity(vmr)
        g = constants.earth_standard_gravity

        return -math.integrate_column(q, p, axis=axis) / g
    elif T is None or z is None:
        raise ValueError(
            'Pass both `T` and `z` for non-hydrostatic calculation of the IWV.'
        )
    else:
        # Integrate the water vapor mass density for non-hydrostatic cases.
        R_v = constants.gas_constant_water_vapor
        rho = density(p, T, R=R_v)  # Water vapor density.

        return math.integrate_column(vmr * rho, z, axis=axis)


def moist_lapse_rate(p, T, e_eq=None):
    r"""Calculate the moist-adiabatic temperature lapse rate.

    Bohren and Albrecht (Equation 6.111, note the **sign change**):

    .. math::
        \frac{dT}{dz} =
            \frac{g}{c_p} \frac{1 + l_v w_s / RT}{1 + l_v^2 w_s/c_p R_v T^2}

    Parameters:
        p (float or ndarray): Pressure [Pa].
        T (float or ndarray): Temperature [K].
        e_eq (callable): Function to calculate the equilibrium vapor
            pressure of water in Pa. The function must implement the
            signature ``e_eq = f(T)`` where ``T`` is temperature in Kelvin.
            If ``None`` the function :func:`~typhon.physics.e_eq_water_mk` is
            used.

    Returns:
        float or ndarray: Moist-adiabatic lapse rate [K/m].

    Examples:
        >>> moist_lapse_rate(1013.25e2, 288.15)
        0.004728194612232855

    References:
        Bohren C. and Albrecht B., Atmospheric Thermodynamics, p. 287-92
    """
    if e_eq is None:
        e_eq = e_eq_water_mk

    # Use short formula symbols for physical constants.
    g = constants.earth_standard_gravity
    Lv = constants.heat_of_vaporization
    Rd = constants.gas_constant_dry_air
    Rv = constants.gas_constant_water_vapor
    Cp = constants.isobaric_mass_heat_capacity

    gamma_d = g / Cp  # dry lapse rate
    w_saturated = vmr2mixing_ratio(e_eq(T) / p)

    lapse = (
        gamma_d * (
            (1 + (Lv * w_saturated) / (Rd * T)) /
            (1 + (Lv**2 * w_saturated) / (Cp * Rv * T**2))
        )
    )

    return lapse


def standard_atmosphere(z, coordinates='height'):
    """International Standard Atmosphere (ISA).

    The temperature profile is defined between 0-85 km (1089 h-0.004 hPa).
    Values exceeding this range are linearly interpolated.

    Parameters:
        z (float or ndarray): Geopotential height above MSL [m]
            or pressure [Pa] (see ``coordinates``).
        coordinates (str): Either 'height' or 'pressure'.

    Returns:
        ndarray: Atmospheric temperature [K].

    Examples:

    .. plot::
        :include-source:

        import numpy as np
        from typhon.plots import (profile_p_log, profile_z)
        from typhon.physics import standard_atmosphere
        from typhon.math import nlogspace


        z = np.linspace(0, 84e3, 100)
        fig, ax = plt.subplots()
        profile_z(z, standard_atmosphere(z), ax=ax)

        p = nlogspace(1000e2, 0.4, 100)
        fig, ax = plt.subplots()
        profile_p_log(p, standard_atmosphere(p, coordinates='pressure'))

        plt.show()

    """
    h = np.array([-610, 11000, 20000, 32000, 47000, 51000, 71000, 84852])
    p = np.array(
        [108_900, 22_632, 5474.9, 868.02, 110.91, 66.939, 3.9564, 0.3734]
    )
    temp = np.array([+19.0, -56.5, -56.5, -44.5, -2.5, -2.5, -58.5, -86.28])

    if coordinates == 'height':
        z_ref = h
    elif coordinates == 'pressure':
        z_ref = np.log(p)
        z = np.log(z)
    else:
        raise ValueError(
            f'"{coordinates}" coordinate is unsupported. '
            'Use "height" or "pressure".')

    return interp1d(z_ref, temp + constants.K, fill_value='extrapolate')(z)


def pressure2height(p, T=None):
    r"""Convert pressure to height based on the hydrostatic equilibrium.

    .. math::
       z = \int -\frac{\mathrm{d}p}{\rho g}

    Parameters:
        p (ndarray): Pressure [Pa].
        T (ndarray): Temperature [K].
            If ``None`` the standard atmosphere is assumed.

    See also:
        .. autosummary::
            :nosignatures:

            standard_atmosphere

    Returns:
        ndarray: Relative height above lowest pressure level [m].
    """
    if T is None:
        T = standard_atmosphere(p, coordinates='pressure')

    layer_depth = np.diff(p)
    rho = density(p, T)
    rho_layer = 0.5 * (rho[:-1] + rho[1:])

    z = np.cumsum(-layer_depth / (rho_layer * constants.g))

    return np.hstack([0, z])


def e_eq_ice_mk(T):
    r"""Calculate the equilibrium vapor pressure of water over ice.

    .. math::
        \ln(e_\mathrm{ice}) = 9.550426
                   - \frac{5723.265}{T}
                   + 3.53068 \cdot \ln(T)
                   - 0.00728332 \cdot T

    Parameters:
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Equilibrium vapor pressure [Pa].

    See also:
        :func:`~typhon.physics.e_eq_water_mk`
            Calculate the equilibrium vapor pressure over liquid water.
        :func:`~typhon.physics.e_eq_mixed_mk`
            Calculate the vapor pressure of water over the mixed phase.

    References:
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
        ice and supercooled water for atmospheric applications,
        Quarterly Journal of the Royal Meteorological Society 131(608):
        1539–1565. doi:10.1256/qj.04.94

    """
    if np.any(T <= 0):
        raise ValueError('Temperatures must be larger than 0 Kelvin.')

    # Give the natural log of saturation vapor pressure over ice in Pa
    e = 9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T

    return np.exp(e)


def e_eq_water_mk(T):
    r"""Calculate the equilibrium vapor pressure of water over liquid water.

    .. math::
        \ln(e_\mathrm{liq}) &=
                    54.842763 - \frac{6763.22}{T} - 4.21 \cdot \ln(T) \\
                    &+ 0.000367 \cdot T
                    + \tanh \left(0.0415 \cdot (T - 218.8)\right) \\
                    &\cdot \left(53.878 - \frac{1331.22}{T}
                                 - 9.44523 \cdot \ln(T)
                                 + 0.014025 \cdot T \right)

    Parameters:
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Equilibrium vapor pressure [Pa].

    See also:
        :func:`~typhon.physics.e_eq_ice_mk`
            Calculate the equilibrium vapor pressure of water over ice.
        :func:`~typhon.physics.e_eq_mixed_mk`
            Calculate the vapor pressure of water over the mixed phase.

    References:
        Murphy, D. M. and Koop, T. (2005): Review of the vapour pressures of
        ice and supercooled water for atmospheric applications,
        Quarterly Journal of the Royal Meteorological Society 131(608):
        1539–1565. doi:10.1256/qj.04.94

    """
    if np.any(T <= 0):
        raise ValueError('Temperatures must be larger than 0 Kelvin.')

    # Give the natural log of saturation vapor pressure over water in Pa

    e = (54.842763
         - 6763.22 / T
         - 4.21 * np.log(T)
         + 0.000367 * T
         + np.tanh(0.0415 * (T - 218.8))
         * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))

    return np.exp(e)


def e_eq_mixed_mk(T):
    r"""Return equilibrium pressure of water with respect to the mixed-phase.

    The equilibrium pressure over water is taken for temperatures above the
    triple point :math:`T_t` the value over ice is taken for temperatures
    below :math:`T_t–23\,\mathrm{K}`.  For intermediate temperatures the
    equilibrium pressure is computed as a combination
    of the values over water and ice according to the IFS documentation:

    .. math::
        e_\mathrm{s} = \begin{cases}
            T > T_t, & e_\mathrm{liq} \\
            T < T_t - 23\,\mathrm{K}, & e_\mathrm{ice} \\
            else, & e_\mathrm{ice}
                + (e_\mathrm{liq} - e_\mathrm{ice})
                \cdot \left(\frac{T - T_t - 23}{23}\right)^2
        \end{cases}

    References:
        IFS Documentation – Cy45r1,
        Operational implementation 5 June 2018,
        Part IV: Physical Processes, Chapter 12, Eq. 12.13,
        https://www.ecmwf.int/node/18714

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from typhon import physics

        T = np.linspace(245, 285)
        fig, ax = plt.subplots()
        ax.semilogy(T, physics.e_eq_mixed_mk(T), lw=3, c='k', label='Mixed')
        ax.semilogy(T, physics.e_eq_ice_mk(T), ls='dashed', label='Ice')
        ax.semilogy(T, physics.e_eq_water_mk(T), ls='dashed', label='Water')
        ax.set_ylabel('Vapor pressure [Pa]')
        ax.set_xlabel('Temperature [K]')
        ax.legend()

        plt.show()

    Parameters:
        T (float or ndarray): Temperature [K].

    See also:
        :func:`~typhon.physics.e_eq_ice_mk`
            Equilibrium pressure of water over ice.
        :func:`~typhon.physics.e_eq_water_mk`
            Equilibrium pressure of water over liquid water.

    Returns:
        float or ndarray: Equilibrium pressure [Pa].
    """
    # Keep track of input type to match the return type.
    is_float_input = isinstance(T, Number)
    if is_float_input:
        # Convert float input to ndarray to allow indexing.
        T = np.asarray([T])

    e_eq_water = e_eq_water_mk(T)
    e_eq_ice = e_eq_ice_mk(T)

    is_water = T > constants.triple_point_water

    is_ice = T < (constants.triple_point_water - 23.)

    e_eq = (e_eq_ice + (e_eq_water - e_eq_ice)
            * ((T - constants.triple_point_water + 23) / 23)**2
            )
    e_eq[is_ice] = e_eq_ice[is_ice]
    e_eq[is_water] = e_eq_water[is_water]

    return e_eq[0] if is_float_input else e_eq


def density(p, T, R=constants.gas_constant_dry_air):
    r"""Calculates gas density by ideal gas law.

    .. math::
        \rho = \frac{p}{R \cdot T}

    Parameters:
        p (float or ndarray): Pressure [Pa.]
        T (float or ndarray): Temperature [K].
            If type of T and p is ndarray, size must match p.
        R (float): Gas constant [J K^-1 kg^-1].
            Default is gas constant for dry air.

    Returns:
        float or ndarray: Density [kg/m**3].

    See also:
        :mod:`typhon.constants`
            Module containing universal gas constant as well
            as gas constants for dry air and water vapor.

    Examples:
        >>> density(1013e2, 300)
        1.1763056653021122
    """
    return p / (R * T)


def mixing_ratio2specific_humidity(w):
    r"""Convert mass mixing ratio to specific humidity.

    .. math::
        q = \frac{w}{1 + w}

    Parameters:
        w (float or ndarray): Mass mixing ratio.

    Returns:
        float or ndarray: Specific humidity.

    Examples:
        >>> mixing_ratio2specific_humidity(0.02)
        0.0196078431372549
    """
    return w / (1 + w)


def mixing_ratio2vmr(w):
    r"""Convert mass mixing ratio to volume mixing ratio.

    .. math::
        x = \frac{w}{w + \frac{M_w}{M_d}}

    Parameters:
        w (float or ndarray): Mass mixing ratio.

    Returns:
        float or ndarray: Volume mixing ratio.

    Examples:
        >>> mixing_ratio2vmr(0.02)
        0.03115371853180794
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return w / (w + Mw / Md)


def specific_humidity2mixing_ratio(q):
    r"""Convert specific humidity to mass mixing ratio.

    .. math::
        w = \frac{q}{1 - q}

    Parameters:
        q (float or ndarray): Specific humidity.

    Returns:
        float or ndarray: Mass mixing ratio.

    Examples:
        >>> specific_humidity2mixing_ratio(0.02)
        0.020408163265306124
    """
    return q / (1 - q)


def specific_humidity2vmr(q):
    r"""Convert specific humidity to volume mixing ratio.

    .. math::
        x = \frac{q}{(1 - q) \frac{M_w}{M_d} + q}

    Parameters:
        q (float or ndarray): Specific humidity.

    Returns:
        float or ndarray: Volume mixing ratio.

    Examples:
        >>> specific_humidity2vmr(0.02)
        0.03176931009073226
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return q / ((1 - q) * Mw / Md + q)


def vmr2mixing_ratio(x):
    r"""Convert volume mixing ratio to mass mixing ratio.

    .. math::
        w = \frac{x}{1 - x} \frac{M_w}{M_d}

    Parameters:
        x (float or ndarray): Volume mixing ratio.

    Returns:
        float or ndarray: Mass mixing ratio.

    Examples:
        >>> vmr2mixing_ratio(0.04)
        0.025915747437955664
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return x / (1 - x) * Mw / Md


def vmr2specific_humidity(x):
    r"""Convert volume mixing ratio to specific humidity.

    .. math::
        q = \frac{x}{(1 - x) \frac{M_d}{M_w} + x}

    Parameters:
        x (float or ndarray): Volume mixing ratio.

    Returns:
        float or ndarray: Specific humidity.

    Examples:
        >>> vmr2specific_humidity(0.04)
        0.025261087474946833
    """
    Md = constants.molar_mass_dry_air
    Mw = constants.molar_mass_water

    return x / ((1 - x) * Md / Mw + x)

def water_vapor_pressure2specific_humidity(e, p):
        r"""Convert water vapor pressure into specific humidity.

    .. math::
        \mathrm{q} = \frac{0.622 \cdot e}{p - 0.378 \cdot e}

    Parameters:
        e (float or ndarray): Water vapour pressure [Pa],
        p (float or ndarray): Pressure [Pa].
        

    Returns:
        float or ndarray: specific humidity.


    Examples:
        >>> water_vapor_pressure2specific_humidity(2338, 101300)
        0.01448208036795962
    """
        return 0.622*e/(p-0.378*e)
