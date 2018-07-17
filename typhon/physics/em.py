# -*- coding: utf-8 -*-

"""Module for anything related to the electromagnetic spectrum.
"""
import numpy as np

from typhon import constants


__all__ = [
    'planck',
    'planck_wavelength',
    'planck_wavenumber',
    'rayleighjeans',
    'rayleighjeans_wavelength',
    'radiance2planckTb',
    'radiance2rayleighjeansTb',
    'snell',
    'fresnel',
    'frequency2wavelength',
    'frequency2wavenumber',
    'wavelength2frequency',
    'wavelength2wavenumber',
    'wavenumber2frequency',
    'wavenumber2wavelength',
    'stefan_boltzmann_law',
    'zeeman_splitting',
    'zeeman_strength',
    'zeeman_transitions',
    ]


def planck(f, T):
    """Calculate black body radiation for given frequency and temperature.

    Parameters:
        f (float or ndarray): Frquencies [Hz].
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Radiances.

    """
    c = constants.speed_of_light
    h = constants.planck
    k = constants.boltzmann

    return 2 * h * f**3 / (c**2 * (np.exp(np.divide(h * f, (k * T))) - 1))


def planck_wavelength(l, T):
    """Calculate black body radiation for given wavelength and temperature.

    Parameters:
        l (float or ndarray): Wavelength [m].
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Radiances.

    """
    c = constants.speed_of_light
    h = constants.planck
    k = constants.boltzmann

    return 2 * h * c**2 / (l**5 * (np.exp(np.divide(h * c, (l * k * T))) - 1))


def planck_wavenumber(n, T):
    """Calculate black body radiation for given wavenumber and temperature.

    Parameters:
        n (float or ndarray): Wavenumber.
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Radiances.

    """
    c = constants.speed_of_light
    h = constants.planck
    k = constants.boltzmann

    return 2 * h * c**2 * n**3 / (np.exp(np.divide(h * c * n, (k * T))) - 1)


def rayleighjeans(f, T):
    """Calculates the Rayleigh-Jeans approximation of the Planck function.

     Calculates the approximation of the Planck function for given
     frequency and temperature.

     Parameters:
        f (float or ndarray): Frequency [Hz].
        T (float or ndarray): Temperature [K].

     Returns:
        float or ndarray: Radiance [W/(m2*Hz*sr)].

    """
    c = constants.speed_of_light
    k = constants.boltzmann

    return 2 * f**2 * k * T / c**2


def rayleighjeans_wavelength(l, T):
    """Calculates the Rayleigh-Jeans approximation of the Planck function.

     Calculates the approximation of the Planck function for given
     wavelength and temperature.

     Parameters:
        l (float or ndarray): Wavelength [m].
        T (float or ndarray): Temperature [K].

     Returns:
        float or ndarray: Radiance [W/(m2*Hz*sr)].

    """
    c = constants.speed_of_light
    k = constants.boltzmann

    return np.divide(2 * c * k * T, l**4)


def radiance2planckTb(f, r):
    """Convert spectral radiance to Planck brightness temperture.

    Parameters:
        f (float or ndarray): Frequency [Hz].
        r (float or ndarray): Spectral radiance [W/m**-2/sr].

    Returns:
        float or ndarray: Planck brightness temperature [K].
    """
    c = constants.speed_of_light
    k = constants.boltzmann
    h = constants.planck

    return h / k * f / np.log(np.divide((2 * h / c**2) * f**3, r) + 1)


def radiance2rayleighjeansTb(f, r):
    """Convert spectral radiance to Rayleight-Jeans brightness temperture.

    Parameters:
        f (float or ndarray): Frequency [Hz].
        r (float or ndarray): Spectral radiance [W/m**-2/sr].

    Returns:
        float or ndarray: Rayleigh-Jeans brightness temperature [K].
    """
    c = constants.speed_of_light
    k = constants.boltzmann

    return np.divide(c**2, (2 * f**2 * k)) * r


def snell(n1, n2, theta1):
    """Calculates the angle of the transmitted wave, according to Snell's law.

    Snell's law for the case when both *n1* and *n2* have no imaginary part
    is found in all physics handbooks.

    The expression for complex *n2* is taken from "An introduction to
    atmospheric radiation" by K.N. Liou (Sec. 5.4.1.3).

    No expression that allows *n1* to be complex has been found.

    If theta2 is found to be complex, it is returned as NaN. This can happen
    when n1 > n2, and corresponds to a total reflection and there is no
    transmitted part.

    The refractive index and the dielectric constant, epsilon, are releated
    as

    .. math::
        n = \sqrt{\epsilon}

    Parameters:
        n1 (float or ndarray): Refractive index for medium of incoming
            radiation.
        n2 (float or ndarray): Refractive index for reflecting medium.
        theta1 (float): Angle between surface normal
            and incoming radiation [degree].

    Returns:
        float or ndarray: Angle for transmitted part [degree].

    .. Ported from atmlab. Original author: Patrick Eriksson
    """

    if np.any(np.real(n1) <= 0) or np.any(np.real(n2) <= 0):
        raise Exception('The real part of *n1* and *n2* can not be <= 0.')

    if np.all(np.isreal(n1)) and np.all(np.isreal(n2)):
        theta2 = np.arcsin(n1 * np.sin(np.deg2rad(theta1)) / n2)

    elif np.all(np.isreal(n1)):
        mr2 = (np.real(n2) / n1)**2
        mi2 = (np.imag(n2) / n1)**2
        sin1 = np.sin(np.deg2rad(theta1))
        s2 = sin1 * sin1
        Nr = np.sqrt((mr2 - mi2 + s2
                      + np.sqrt((mr2 - mi2 - s2)**2 + 4 * mr2 * mi2)
                      ) / 2)
        theta2 = np.arcsin(sin1 / Nr)
    else:
        raise Exception('No expression implemented for imaginary *n1*.')

    if not np.all(np.isreal(theta2)):
        theta2 = np.nan

    return np.rad2deg(theta2)


def fresnel(n1, n2, theta1):
    r"""Fresnel formulas for surface reflection.

    The amplitude reflection coefficients for a flat surface can also be
    calculated (Rv and Rh). Note that these are the coefficients for the
    amplitude of the wave. The power reflection coefficients are
    obtained as

    .. math::
        r = \lvert R \rvert^2

    The expressions used are taken from Eq. 3.31 in "Physical principles of
    remote sensing", by W.G. Rees, with the simplification that that relative
    magnetic permeability is 1 for both involved media. The theta2 angle is
    taken from snell.m.

    The refractive index of medium 2  (n2) can be complex. The refractive
    index and the dielectric constant, epsilon, are releated as

    .. math::
        n = \sqrt{\epsilon}

    No expression for theta2 that allows *n1* to be complex has been found.

    If theta2 is found to be complex, it is returned as NaN. This can happen
    when n1 > n2, and corresponds to a total reflection and there is no
    transmitted part. Rv and Rh are here set to 1.

    Parameters:
        n1 (float or ndarray): Refractive index for medium of incoming
            radiation.
        n2 (float or ndarray): Refractive index for reflecting medium.
        theta1 (float or ndarray): Angle between surface normal
            and incoming radiation [degree].

    Returns:
        float or ndarray, float or ndarray:
            Reflection coefficient for vertical polarisation,
            reflection coefficient for horisontal polarisation.

    .. Ported from atmlab. Original author: Patrick Eriksson
    """
    if np.any(np.imag(n1) < 0) or np.any(np.imag(n2) < 0):
        raise Exception(
            'The imaginary part of *n1* and *n2* can not be negative.')

    theta2 = snell(n1, n2, theta1)

    costheta1 = np.cos(np.deg2rad(theta1))
    costheta2 = np.cos(np.deg2rad(theta2))

    Rv = (n2 * costheta1 - n1 * costheta2) / (n2 * costheta1 + n1 * costheta2)
    Rh = (n1 * costheta1 - n2 * costheta2) / (n1 * costheta1 + n2 * costheta2)

    return Rv, Rh


def frequency2wavelength(frequency):
    """Convert frequency to wavelength.

    Parameters:
        frequency (float or ndarray): Frequency [Hz].

    Returns:
        float or ndarray: Wavelength [m].

    """
    return np.divide(constants.speed_of_light, frequency)


def frequency2wavenumber(frequency):
    """Convert frequency to wavenumber.

    Parameters:
        frequency (float or ndarray): Frequency [Hz].

    Returns:
        float or ndarray: Wavenumber [m^-1].

    """
    return frequency / constants.speed_of_light


def wavelength2frequency(wavelength):
    """Convert wavelength to frequency.

    Parameters:
        wavelength (float or ndarray): Wavelength [m].

    Returns:
        float or ndarray: Frequency [Hz].

    """
    return np.divide(constants.speed_of_light, wavelength)


def wavelength2wavenumber(wavelength):
    """Convert wavelength to wavenumber.

    Parameters:
        wavelength (float or ndarray): Wavelength [m].

    Returns:
        float or ndarray: Wavenumber [m^-1].

    """
    return np.divide(1, wavelength)


def wavenumber2frequency(wavenumber):
    """Convert wavenumber to frequency.

    Parameters:
        wavenumber (float or ndarray): Wavenumber [m^-1].
    Returns:
        float or ndarray: Frequency [Hz].

    """
    return constants.speed_of_light * wavenumber


def wavenumber2wavelength(wavenumber):
    """Convert wavenumber to wavelength.

    Parameters:
        wavenumber (float or ndarray): Wavenumber [m^-1].

    Returns:
        float or ndarray: Wavelength [m].

    """
    return np.divide(1, wavenumber)


def stefan_boltzmann_law(T):
    """Compute Stefan Boltzmann law for given temperature

    .. math::
        j = \\sigma T^4

    Parameters:
        T (float or ndarray): Physical temperature  of object [K]

    Returns:
        float or ndarray: Energy per surface area [W m^-2]
    """
    return constants.stefan_boltzmann_constant * T**4


def hund_case_a_landau_g_factor(o, j, s, l, gs, gl):
    """ Hund case A Landau g-factor

    .. math::
        g =  \\frac{\\Omega}{J(J+1)} \\left( g_sS + g_l\\Lambda \\right)

    Parameters:
        o (float): Omega of level

        j (float): J of level

        l (float): Lambda of level

        s (float): Sigma of level

        gs (float): relativistic spin factor of molecule

        gl (float): relativistic orbital factor of molecule

    Returns:
        g (float): Landau g-factor for Hund case A
    """
    return gs * o * s / (j * (j + 1)) + gl * o * l / (j * (j + 1))


def hund_case_b_landau_g_factor(n, j, s, l, gs, gl):
    """ Hund case B Landau g-factor

    .. math::
        g = g_s \\frac{J(J+1) + S(S+1) - N(N+1)}{2J(J+1)} +
        g_l \\Lambda \\frac{J(J+1) - S(S+1) + N(N+1)}{2JN(J+1)(N+1)}

    Parameters:
        n (float): N of level

        j (float): J of level

        l (float): Lambda of level

        s (float): S of level

        gs (float): relativistic spin factor of molecule

        gl (float): relativistic orbital factor of molecule

    Returns:
        g (float): Landau g-factor for Hund case B
    """
    if n != 0:
        return gs * (j * (j+1) + s * (s+1) - n * (n+1)) / (2 * j * (j + 1)) + \
            l * gl * (j * (j+1) - s * (s+1) + n * (n+1)) / (2 * j * n * (j + 1) * (n + 1))
    else:
        return gs * (j * (j+1) + s * (s+1) - n * (n+1)) / (2 * j * (j + 1))


def landau_g_factor(x, j, s, l=None, gs=2, gl=1, case=None):
    """ The Landau g-factor

    Parameters:
        x (float): N or Omega of level

        j (float): J of level

        s (float): S of level

        l (float): Lambda of level

        gs (float): relativistic spin factor of molecule

        gl (float): relativistic orbital factor of molecule

        case (str): Case type, 'a' or 'b'

    Returns:
        g (float): Landau g-factor for the Hund case or None for unknown case
    """
    if j == 0:
        return 0
    elif l is None:
        l = 0

    assert type(case) == str, "Case must be str for J not 0"

    if case.lower() == 'a':
        return hund_case_a_landau_g_factor(x, j, s, l, gs, gl)
    elif case.lower() == 'b':
        return hund_case_b_landau_g_factor(x, j, s, l, gs, gl)
    else:
        raise RuntimeError("Unknown case-type: " + str(case))


def zeeman_splitting(gu, gl, mu, ml, H=1):
    """ Zeeman splitting

    .. math::
        \Delta f = \\frac{H\mu_b}{h}(g_um_u - g_lm_l),

    where :math:`\mu_b` is the Bohr magneton and :math:`h` is the Planck
    constant.

    Parameters:
        gu (scalar or ndarray): Upper g

        gl (scalar or ndarray): Lower g

        mu (scalar or ndarray): Upper projection of j

        ml (scalar or ndarray): Lower projection of j

        H (scalar or ndarray): Absolute strength of magnetic field in Teslas

    Returns:
        scalar or ndarray: Splitting in Hertz
    """
    h = constants.planck
    mu_B = constants.mu_B

    frac = mu_B / h
    return frac * (gu * mu - gl * ml) * H


def zeeman_strength(ju, jl, mu, ml):
    """ Zeeman line strength

    .. math:: \Delta S_{M_u,M_l} = C \\left(\\begin{array}{ccc} J_l & 1 & J_u
                       \\\\ M_l & M_u-M_l&-M_u \\end{array}\\right)^2,

    where C is either 3/2 or 3/4 depending on in mu-ml is 0 or not.  In case
    the intent is to return many split lines, the size of either (jl, ju) or
    (ml, mu) must be the same.  Regardless, these pairs have to be of the same
    type

    Parameters:
        ju (scalar or 1darray): Upper level J.  Must be same size as jl

        jl (scalar or 1darray): Lower level J.  Must be same size as ju

        mu (scalar or ndarray): Upper level M.  Must be same size as ml unless
        J is vector-type, then it is ignored

        ml (scalar or ndarray): Lower level M.  Must be same size as mu unless
        J is vector-type, then it is ignored

    Returns:
        scalar or ndarray: Relative line strength of component normalized to 1
        or array(array(S+, Pi, S-))
    """
    assert type(jl) == type(ju), "Must have same type: J"
    assert type(ml) == type(mu), "Must have same type: M"

    try:
        import sympy.physics.wigner as wig
    except ModuleNotFoundError:
        raise RuntimeError("Must have sympy installed to use")

    if np.isscalar(mu) and np.isscalar(ju):
        dm = mu - ml
        w = wig.wigner_3j(jl, 1, ju, ml, dm, -mu)
        w = float(w)
        if dm == 0:
            r = w**2 * 1.5
        else:
            r = w**2 * 0.75
    elif np.isscalar(ju):
        r = []
        for i in range(len(mu)):
            r.append(zeeman_strength(ju, jl, mu[i], ml[i]))
        r = np.array(r)
    else:
        r = []
        for i in range(len(ju)):
            JU = ju[i]
            JL = jl[i]
            if np.isscalar(JU):
                t = []
                MU, ML = zeeman_transitions(JU, JL, "S-")
                t.append(zeeman_strength(JU, JL, MU, ML))
                MU, ML = zeeman_transitions(JU, JL, "Pi")
                t.append(zeeman_strength(JU, JL, MU, ML))
                MU, ML = zeeman_transitions(JU, JL, "S+")
                t.append(zeeman_strength(JU, JL, MU, ML))
                r.append(t)
            else:
                r.append(zeeman_strength(JU, JL, mu, ml))
        r = np.array(r)
    return r


def zeeman_transitions(ju, jl, type):
    """ Find possible mu and ml for valid ju and jl for a given transistion
    polarization

    Parameters:
        ju (scalar):  Upper level J

        jl (scalar):  Lower level J

        type (string): "Pi", "S+", or "S-" for relevant polarization type

    Returns:
        tuple: MU, ML arrays for given Js and polarization type
    """
    assert np.isscalar(ju) and np.isscalar(jl), "non-scalar J non supported"
    assert type.lower() in ["pi", "s+", "s-"], "unknown transition type"
    assert ju - jl in [-1, 0, 1], "delta-J should belong to {-1, 0, 1}"
    assert ju > 0 and jl >= 0, "only for positive ju and non-negative for jl"

    if type.lower() == "pi":
        J = min(ju, jl)
        return np.arange(-J, J + 1), np.arange(-J, J + 1)
    elif type.lower() == "s+":
        if ju < jl:
            return np.arange(-ju, ju+1), np.arange(-ju+1, ju+2)
        elif ju == jl:
            return np.arange(-ju, ju), np.arange(-ju+1, ju+1)
        else:
            return np.arange(-ju, jl), np.arange(-ju+1, jl+1)
    elif type.lower() == "s-":
        if ju < jl:
            return np.arange(-ju, ju+1), np.arange(-jl, ju)
        elif ju == jl:
            return np.arange(-ju+1, ju+1), np.arange(-ju, ju)
        else:
            return np.arange(-ju+2, ju+1), np.arange(-ju+1, ju)


def zeeman_theta(u, v, w, z=0, a=0):
    """ Find Zeeman angle along the magnetic field
    """

    try:
        import sympy as sp
    except ModuleNotFoundError:
        raise RuntimeError("Must have sympy installed to use")

    U, V, W, Z, A = np.meshgrid(u, v, w, z, a, copy=False)
    N = len(U.flatten())

    if type(sp.symbols('u')) == type(u):
        sin = sp.sin
        cos = sp.cos
        acos = sp.acos
        sqrt = sp.sqrt
        d = np.empty((N), type(u))
    else:
        sin = np.sin
        cos = np.cos
        acos = np.arccos
        sqrt = np.sqrt
        d = np.empty((N), float)

    for i in range(N):
        H = np.array([U.flat[i], V.flat[i], W.flat[i]])
        L = np.array([sin(Z.flat[i])*cos(A.flat[i]),
                      sin(Z.flat[i])*sin(A.flat[i]), cos(Z.flat[i])])
        d[i] = acos(H.dot(L) / sqrt(H.dot(H)))

    if type(sp.symbols('u')) == type(u):
        return d[0]

    shape = []
    for input in [u, v, w, z, a]:
        if np.isscalar(input):
            continue
        shape.append(len(input))

    if shape:
        d = d.reshape(shape)
    else:
        d = d[0]

    return d


def zeeman_eta(u, v, w, z=0, a=0):

    """ Find Zeeman angle along the magnetic field
    """

    try:
        import sympy as sp
    except ModuleNotFoundError:
        raise RuntimeError("Must have sympy installed to use")

    U, V, W, Z, A = np.meshgrid(u, v, w, z, a, copy=False)
    N = len(U.flatten())

    if type(sp.symbols('u')) == type(u):
        sin = sp.sin
        cos = sp.cos
        atan2 = sp.atan2
        d = np.empty((N), dtype=type(u))
    else:
        sin = np.sin
        cos = np.cos
        atan2 = np.arctan2
        d = np.empty((N), dtype=float)

    for i in range(N):
        H = np.array([U.flat[i], V.flat[i], W.flat[i]])
        L = np.array([sin(Z.flat[i])*cos(A.flat[i]),
                      sin(Z.flat[i])*sin(A.flat[i]), cos(Z.flat[i])])
        R = np.array([-sin(A.flat[i]), cos(A.flat[i]), 0])
        p = H - L.dot(H) * H
        d[i] = atan2(np.cross(R, p).dot(L), p.dot(R))

    if type(sp.symbols('u')) == type(u):
        return d[0]

    shape = []
    for input in [u, v, w, z, a]:
        if np.isscalar(input):
            continue
        shape.append(len(input))

    if shape:
        d = d.reshape(shape)
    else:
        d = d[0]

    return d
