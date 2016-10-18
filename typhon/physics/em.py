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
    'snell',
    'fresnel',
    'frequency2wavelength',
    'frequency2wavenumber',
    'wavelength2frequency',
    'wavelength2wavenumber',
    'wavenumber2frequency',
    'wavenumber2wavelength',
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

    return 2 * h * f**3 / (c**2 * (np.exp(h * f / (k * T)) - 1))


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

    return 2 * h * c**2 / (l**5 * (np.exp(h * c / (l * k * T)) - 1))


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

    return 2 * h * c**2 * n**3 / (np.exp(h * c * n / (k * T) - 1))


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

    return 2 * c * k * T / l**4


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
    return constants.speed_of_light / frequency


def frequency2wavenumber(frequency):
    """Convert frequency to wavelength.

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
    return constants.speed_of_light / wavelength


def wavelength2wavenumber(wavelength):
    """Convert wavelength to wavenumber.

    Parameters:
        wavelength (float or ndarray): Wavelength [m].

    Returns:
        float or ndarray: Wavenumber [m^-1].

    """
    return 1 / wavelength


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
    return 1 / wavenumber
