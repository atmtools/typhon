# -*- coding: utf-8 -*-

"""Module for anything related to the electromagnetic spectrum.

This module imports typhon.physics.units.common and therefore
has a soft dependency on the pint units library.
"""


import logging

import numpy
import scipy.interpolate

import numexpr
import pint


from typhon import config
from typhon.arts import xml
from typhon.constants import (h, k, c)
from typhon.physics.units.common import (ureg, radiance_units)


__all__ = [
    'FwmuMixin',
    'SRF',
    'planck_f',
    'specrad_wavenumber2frequency',
    'specrad_frequency_to_planck_bt',
    ]


class FwmuMixin:
    """Mixing for frequency/wavelength/wavenumber neutrality

    Best to use pint ureg quantities at all times.
    """
    _frequency = None
    _wavenumber = None
    _wavelength = None

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        try:
            self._frequency = value.to(ureg.Hz, "sp")
        except AttributeError:
            value = value * ureg.Hz
            self._frequency = value

        self._wavenumber = value.to(1 / ureg.centimeter, "sp")
        self._wavelength = value.to(ureg.metre, "sp")

    @property
    def wavenumber(self):
        return self._wavenumber

    @wavenumber.setter
    def wavenumber(self, value):
        try:
            self._wavenumber = value.to(1 / ureg.centimeter, "sp")
        except AttributeError:
            value = value * 1 / ureg.centimeter
            self._wavenumber = value

        self._frequency = value.to(ureg.Hz, "sp")
        self._wavelength = value.to(ureg.metre, "sp")

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        try:
            self._wavelength = value.to(ureg.metre, "sp")
        except AttributeError:
            value = value * ureg.meter
            self._wavelength = value

        self._frequency = value.to(ureg.hertz, "sp")
        self._wavenumber = value.to(1 / ureg.centimeter, "sp")


class SRF(FwmuMixin):
    """Respresents a spectral response function
    """

    T_lookup_table = numpy.arange(100, 370.01, 0.05) * ureg.K
    lookup_table = None
    L_to_T = None

    def __init__(self, f, W):
        """Initialise SRF object

        :param ndarray f: Array of frequencies
        :param ndarray W: Array of associated weights
        """

        self.frequency = f
        self.W = W

    def __repr__(self):
        """Get string representation.

        Uses centroid.
        """
        cf = self.centroid()
        if cf.to("Hz", "sp").m > 3e12:
            s = cf.to(ureg.um, "sp")
        else:
            s = cf.to(ureg.GHz, "sp")
        return "<{:s}: {:.4~}>".format(self.__class__.__name__, s)

    @classmethod
    def fromArtsXML(cls, sat, instr, ch):
        """Read SRF from ArtsXML files.

        Requires that in the TYPHONRC configuration file, the fields
        `srf_backend_f` and `srf_backend_response` in the section
        corresponding to instrument `instr` are defined to point to the
        respective files in ArtsXML format.  Within those definitions,
        {sat:s} will be substituted with the satellite name.

        Arguments:

            sat [str]

                Satellite name, such as 'NOAA15'

            instr [str]

                Instrument name, such as 'hirs'.

            ch [int]

                Channel number (start counting at 1).
        """
        cf = config.conf[instr]
        centres = xml.load(
            cf["srf_backend_f"].format(sat=sat))
        gfs = xml.load(
            cf["srf_backend_response"].format(sat=sat))
        freq = gfs[ch - 1].grids[0] + centres[ch - 1]
        response = gfs[ch - 1].data
        return cls(freq, response)

    def centroid(self):
        """Calculate centre frequency
        """
        return numpy.average(
            self.frequency, weights=self.W) * self.frequency.units

    def blackbody_radiance(self, T):
        """Calculate integrated radiance for blackbody at temperature T

        :param T: Temperature [K]
        """
        return self.integrate_radiances(
            self.frequency, planck_f(
                self.frequency[numpy.newaxis, :],
                T[:, numpy.newaxis]))

    def make_lookup_table(self):
        """Construct lookup table radiance <-> BT

        To convert a channel radiance to a brightness temperature,
        applying the (inverse) Planck function is not correct, because the
        Planck function applies to monochromatic radiances only.  Instead,
        to convert from radiance (in W m^-2 sr^-1 Hz^-1) to brightness
        temperature, we use a lookup table.  This lookup table is
        constructed by considering blackbodies at a range of temperatures,
        then calculating the channel radiance.  This table can then be
        used to get a mapping from radiance to brightness temperature.

        This method does not return anything, but fill self.lookup_table.
        """
        self.lookup_table = numpy.zeros(
            shape=(2, self.T_lookup_table.size), dtype=numpy.float64)
        self.lookup_table[0, :] = self.T_lookup_table
        self.lookup_table[1, :] = self.blackbody_radiance(self.T_lookup_table)
        self.L_to_T = scipy.interpolate.interp1d(self.lookup_table[1, :],
                                                 self.lookup_table[0, :],
                                                 kind='linear',
                                                 bounds_error=False,
                                                 fill_value=0)

    def integrate_radiances(self, f, L):
        """From a spectrum of radiances and a SRF, calculate channel radiance

        The spectral response function may not be specified on the same grid
        as the spectrum of radiances.  Therefore, this function interpolates
        the spectral response function onto the grid of the radiances.  This
        is less bad than the reverse, because a spectral response function
        tends to be more smooth than a spectrum.

        **Approximations:**

        * Interpolation of spectral response function onto frequency grid on
          which radiances are specified.

        :param ndarray f: Frequencies for spectral radiances [Hz]
        :param ndarray L: Spectral radiances [W m^-2 sr^-1 Hz^-1].
            Can be in radiance units of various kinds.  Make sure this is
            consistent with the spectral response function.
            Innermost dimension must correspond to frequencies.
        :returns: Channel radiance [W m^-2 sr^-1 Hz^-1]
        """
        # Interpolate onto common frequency grid.  The spectral response
        # function is more smooth so less harmed by interpolation, so I
        # interpolate the SRF.
        fnc = scipy.interpolate.interp1d(
            self.frequency, self.W, bounds_error=False, fill_value=0.0)
        w_on_L_grid = fnc(f) * (1 / ureg.Hz)
        # ch_BT = (w_on_L_grid * L_f).sum(-1) / (w_on_L_grid.sum())
        # due to numexpr limitation, do sum seperately
        # and due to numexpr bug, explicitly consider zero-dim case
        #     see https://github.com/pydata/numexpr/issues/229
        if L.shape[0] == 0:
            ch_rad_tot = numpy.empty(dtype="f8", shape=L.shape[:-1])
        else:
            ch_rad_tot = numexpr.evaluate("sum(w_on_L_grid * L, {:d})".format(
                L.ndim - 1))
        ch_rad_tot = ch_rad_tot * w_on_L_grid.u * L.u
        ch_rad = ch_rad_tot / w_on_L_grid.sum()
        return ch_rad

    def channel_radiance2bt(self, L):
        """Convert channel radiance to brightness temperature

        Using the lookup table, convert channel radiance to brightness
        temperature.  Will construct lookup table on first call.

        :param L: Radiance [W m^-2 sr^-1 Hz^-1] or compatible
        """
        if self.lookup_table is None:
            self.make_lookup_table()
        return self.L_to_T(
            L.to(ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz),
                 "radiance")) * ureg.K

    # Methods returning new SRFs with some changes
    def shift(self, amount):
        """Get new SRF, shifted by <amount>

        Return a new SRF, shifted by <amount> Hz.  The shape of the SRF is
        retained.

        Arguments:

            Quantity amount: Distance to shift SRF
        """
        return self.__class__(self.frequency.to(
            amount.u, "sp") + amount, self.W)

_specrad_freq = ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz)


def planck_f(f, T):
    """Planck law expressed in frequency.

    If more than 10⁵ resulting radiances, uses numexpr.

    :param f: Frequency.  Quantity in [Hz]
    :param T: Temperature.  Quantity in [K]
    """
#    try:
#        f = f.astype(numpy.float64)
#    except AttributeError:
#        pass
    if (f.size * T.size) > 1e5:
        return numexpr.evaluate("(2 * h * f**3) / (c**2) * "
                                "1 / (exp((h*f)/(k*T)) - 1)") * (
                                    radiance_units["si"])
    return ((2 * ureg.h * f**3) / (ureg.c ** 2) *
            1 / (numpy.exp(((ureg.h * f) / (ureg.k * T)).to("1")) - 1)).to(
                ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz))


def specrad_wavenumber2frequency(specrad_wavenum):
    """Convert spectral radiance from per wavenumber to per frequency

    :param specrad_wavenum: Spectral radiance per wavenumber
         [W·sr^{-1}·m^{-2}·{m^{-1}}^{-1}]
    :returns: Spectral radiance per frequency [W⋅sr−1⋅m−2⋅Hz−1]
    """

    if not isinstance(specrad_wavenum, pint.quantity._Quantity):
        specrad_wavenum = specrad_wavenum * ureg.W / (
            ureg.m**2 * ureg.sr * (1 / ureg.m))

    return (specrad_wavenum / ureg.c).to(ureg.W / (
        ureg.m**2 * ureg.sr * ureg.Hz))


def specrad_frequency_to_planck_bt(L, f):
    """Convert spectral radiance per frequency to brightness temperature

    This function converts monochromatic spectral radiance per frequency
    to Planck brightness temperature.  This is calculated by inverting the
    Planck function.

    Note that this function is NOT correct to estimate polychromatic
    brightness temperatures such as channel brightness temperatures.  For
    this, you need the spectral response function — see the SRF class.

    :param L: Spectral radiance [W m^-2 sr^-1 Hz^-1]
    :param f: Corresponding frequency [Hz]
    :returns: Planck brightness temperature [K].
    """

    # f needs to be double to prevent overflow
    f = numpy.asarray(f).astype(numpy.float64)
    if L.size > 1500000:
        logging.debug("Doing actual BT conversion: {:,} spectra * {:,} "
                      "frequencies = {:,} radiances".format(
                          L.size // L.shape[-1], f.size, L.size))
    # BT = (h * f) / (k * numpy.log((2*h*f**3)/(L * c**2) + 1))
    BT = numexpr.evaluate("(h * f) / (k * log((2*h*f**3)/(L * c**2) + 1))")
    BT = numpy.ma.masked_invalid(BT)
    if L.size > 1500000:
        logging.debug("(done)")
    return ureg.Quantity(BT, ureg.K)
