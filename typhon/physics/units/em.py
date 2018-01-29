# -*- coding: utf-8 -*-

"""Module for anything related to the electromagnetic spectrum.

This module imports typhon.physics.units.common and therefore
has a soft dependency on the pint units library.
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import warnings
import logging

import numpy
import scipy.interpolate

import numexpr
import pint
import xarray


from typhon import config
from typhon.arts import xml
from typhon.constants import (h, k, c)
from typhon.physics.units.common import (ureg, radiance_units)
from typhon.physics.units.tools import UnitsAwareDataArray as UADA


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

    TODO: representation of uncertainties
    """

    T_lookup_table = numpy.arange(0, 500.01, 0.05) * ureg.K
    lookup_table = None
    L_to_T = None

    def __init__(self, f, W):
        """Initialise SRF object.

        You can either initialise an SRF from scratch, or use the
        classmethod `fromArtsXML` to read it from a file.

        A toy example on initiating it from scratch:

        >>> from typhon.physics.units.common import ureg
        >>> from typhon.physics.units.em import SRF
        >>> srf = SRF(ureg.Quantity(numpy.array([200, 200.1, 200.2, 200.3, 200.4, 200.5]), 'GHz'), numpy.array([0, 0.5, 1, 1, 0.5, 0]))
        >>> R_300 = srf.blackbody_radiance(ureg.Quantity(300, 'K'))
        >>> print(R_300)
        [  3.63716781e-15] watt / hertz / meter ** 2 / steradian
        >>> print(R_300.to("K", "radiance", srf=srf))
        [ 300.] kelvin

        You can also pass in other spectroscopic units (wavenumber,
        wavelength) that will be converted internally to frequency:

        >>> srf = SRF(ureg.Quantity(numpy.array([10.8, 10.9, 11.0, 11.1, 11.2, 11.3]), 'um'), numpy.array([0, 0.5, 1, 1, 0.5, 0]))
        >>> R_300 = srf.blackbody_radiance(ureg.Quantity(numpy.atleast_1d(250), 'K'))
        >>> print(R_300)
        [  1.61922509e-12] watt / hertz / meter ** 2 / steradian
        >>> print(R_300.to("cm * mW / m**2 / sr", "radiance"))
        [ 48.54314703] centimeter * milliwatt / meter ** 2 / steradian
        >>> print(R_300.to("K", "radiance", srf=srf))
        [ 300.] kelvin

        :param ndarray f: Array of frequencies.  Can be either a pure
            ndarray, which will be assumed to be in Hz, or a ureg
            quantity.
        :param ndarray W: Array of associated weights.
        """

        try:
            self.frequency = f.to("Hz", "sp")
        except AttributeError:
            self.frequency = ureg.Quantity(f, "Hz")
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
        {sat:s} will be substituted with the satellite name.  For example,
        in typhonrc, one might have:

        [hirs]
        srf_backend_response = /path/to/{sat}_HIRS.backend_channel_response.xml
        srf_backend_f = /path/to/{sat}_HIRS.f_backend.xml

        so that we can do:

        >>> srf = SRF.fromArtsXML("NOAA15", "hirs", 12)
        >>> R_300 = srf.blackbody_radiance(ureg.Quantity(atleast_1d(250), 'K')) 
        >>> print(R_300)
        [  2.13002925e-13] watt / hertz / meter ** 2 / steradian
        >>> print(R_300.to("cm * mW / m**2 / sr", "radiance")) 
        [ 6.38566704] centimeter * milliwatt / meter ** 2 / steradian

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

    def blackbody_radiance(self, T, spectral=True):
        """Calculate integrated radiance for blackbody at temperature T

        :param T: Temperature [K].  This can be either a python number, or
            a numpy ndarray, on a ureg quantity encompassing either.
        :param spectral: Parameter to control whether to return spectral
            radiance or radiance.  See self.integrate_radiances for
            details.

        Returns quantity ndarray with blackbody radiance in desired unit.
        Note that this is an ndarray with dimension (1,) even if you
        passin a scalar.
        """
        try:
            T = T.to("K")
        except AttributeError:
            T = ureg.Quantity(T, "K")
        T = ureg.Quantity(numpy.atleast_1d(T), T.u)
        # fails if T is multidimensional
        shp = T.shape
        return self.integrate_radiances(
            self.frequency, planck_f(
                self.frequency[numpy.newaxis, :],
                T.reshape((-1,))[:, numpy.newaxis]),
                spectral=spectral).reshape(shp)

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
                                                 fill_value=(0, 2000))

    def integrate_radiances(self, f, L, spectral=True):
        """From a spectrum of radiances and a SRF, calculate channel (spectral) radiance

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
        :param bool spectral: If true, return spectral radiance
            [W m^-2 sr^-1 Hz^-1].  If false, return radiance [W m^-2
            sr^-1].  Defaults to True.
        :returns: Channel (spectral) radiance according to 'spectral'
        """
        # Interpolate onto common frequency grid.  The spectral response
        # function is more smooth so less harmed by interpolation, so I
        # interpolate the SRF.
        fnc = scipy.interpolate.interp1d(
            self.frequency, self.W, bounds_error=False, fill_value=0.0)
        #w_on_L_grid = fnc(f) * (1 / ureg.Hz)
        w_on_L_grid = ureg.Quantity(fnc(f), ureg.dimensionless)# * (1 / ureg.Hz)

        df = ureg.Quantity(numpy.diff(f), f.u)
        w1p = w_on_L_grid[1:]
        L1p = L[:, 1:]
        # ch_BT = (w_on_L_grid * L_f).sum(-1) / (w_on_L_grid.sum())
        # due to numexpr limitation, do sum seperately
        # and due to numexpr bug, explicitly consider zero-dim case
        #     see https://github.com/pydata/numexpr/issues/229
        if L.shape[0] == 0:
            ch_rad = numpy.empty(dtype="f8", shape=L.shape[:-1])
        else:
#            ch_rad = numexpr.evaluate("sum(w_on_L_grid * L, {:d})".format(
            ch_rad = numexpr.evaluate("sum(w1p * L1p * df, {:d})".format(
                L.ndim - 1))
        ch_rad = ureg.Quantity(ch_rad, w1p.u * L1p.u * df.u)
        if spectral:
            return ch_rad / (w1p*df).sum()
        else:
            return ch_rad

    def channel_radiance2bt(self, L):
        """Convert channel radiance to brightness temperature

        Using the lookup table, convert channel radiance to brightness
        temperature.  Will construct lookup table on first call.

        Typhon also registers a pint context “radiance” which can be used
        to convert between radiance units and brightness temperature (even
        though this is a different quantity), for example, by using
        L.to("K", "radiance", srf=srf)

        :param L: Radiance [W m^-2 sr^-1 Hz^-1] or compatible
        """
        if self.lookup_table is None:
            self.make_lookup_table()
        return self.L_to_T(
            L.to(ureg.W / (ureg.m**2 * ureg.sr * ureg.Hz),
                 "radiance")) * ureg.K

    def estimate_band_coefficients(self, sat=None, instr=None, ch=None):
        """Estimate band coefficients for fast/explicit BT calculations

        In some circumstances, a fully integrated SRF may be more
        expensive than needed.  We can then choose an effective wavelength lambda_c
        along with coefficients alpha, beta such that instead of integrating, we
        estimate R = B(lambda*, T*), with T* = alpha + beta · T_B and lambda* a wavelength
        which may be close to the centroid lambda_c (but there is no
        guarantee).  Such an approximation eliminates the explicit use of
        an integral which can make analysis easier.

        Returns:

            alpha (float): Offset in approximation for T*
            beta (float): Slope in approximation for T*
            lambda_eff (float): Effective wavelength
            delta_alpha (float): Uncertainty in alpha
            delta_beta (float): Uncertainty in beta
            delta_lambda_eff (float): Uncertainty in lambda_eff
        """

        warnings.warn("Obtaining band coefficients from file", UserWarning)
        srcfile = config.conf[instr]["band_file"].format(sat=sat)
        rxp = r"(.{5,6})_ch(\d\d?)_shift([+-]\d+)pm\.nc\s+([\d.]+)\s+(-?[\de\-.]+)\s+([\d.]+)"
        dtp = [("satname", "S6"), ("channel", "u1"), ("shift", "i2"),
               ("centre", "f4"), ("alpha", "f4"), ("beta", "f4")]
        M = numpy.fromregex(srcfile, rxp, dtp).reshape(19, 7)
        dims = ("channel", "shiftno")
        ds = xarray.Dataset(
            {"centre": (dims, M["centre"]),
             "alpha": (dims, M["alpha"]),
             "beta": (dims, M["beta"]),
             "shift": (dims, M["shift"])},
            coords = {"channel": M["channel"][:, 0]})

        ds = ds.sel(channel=ch)

        ds0 = ds.sel(shiftno=0) # varies 1.1 – 15.2 nm depending on channel
        lambda_c = UADA(ds0["centre"], attrs={"units": "1/cm"})
        alpha = UADA(ds0["alpha"], attrs={"units": "K"})
        beta = UADA(ds0["beta"], attrs={"units": "1"})

        delta_ds = ds.sel(shiftno=1) - ds0
        delta_lambda_c = abs(UADA(delta_ds["centre"], attrs={"units": "1/cm"}))
        delta_alpha = abs(UADA(delta_ds["alpha"], attrs={"units": "K"}))
        delta_beta = abs(UADA(delta_ds["beta"], attrs={"units": "1"}))

        return (alpha, beta, lambda_c, delta_alpha, delta_beta, delta_lambda_c)

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

    def as_dataarray(self, coordinate):
        """Return xarray.DataArray object.

        Coordinate can be "wavelength" (which will be in m), "frequency"
        (which will be in Hz), or "wavenumber" (which will be in 1/cm).
        """

        return xarray.DataArray(self.W, dims=(coordinate,),
            coords={coordinate: getattr(self, coordinate)}, name="SRF")

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
