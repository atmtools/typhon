# -*- coding: utf-8 -*-
"""Functions to work with ASTER L1B satellite data.
"""
import datetime
import logging
import os
import warnings
from collections import namedtuple

import gdal
import numpy as np
from scipy import interpolate
from skimage.measure import block_reduce

from typhon.math import multiple_logical


__all__ = [
    "ASTERimage",
    "lapserate_modis",
    "lapserate_moist_adiabate",
    "cloudtopheight_IR",
]

logger = logging.getLogger(__name__)

CONFIDENTLY_CLOUDY = 5
PROBABLY_CLOUDY = 4
PROBABLY_CLEAR = 3
CONFIDENTLY_CLEAR = 2
CLOUDY = 1
CLEAR = 0


class ASTERimage:
    """ASTER L1B image object."""

    subsensors = {
        "VNIR": ("1", "2", "3N", "3B"),
        "SWIR": ("4", "5", "6", "7", "8", "9"),
        "TIR": ("10", "11", "12", "13", "14"),
    }
    # list of ASTER channels
    channels = [c for sc in subsensors.values() for c in sc]
    # dictionary with keys=channels and values=subsensors
    channel2sensor = {c: s for s, sc in subsensors.items() for c in sc}

    wavelength_range = {
        # VNIR
        "1": (0.52, 0.60),
        "2": (0.63, 0.69),
        "3N": (0.78, 0.86),
        "3B": (0.78, 0.86),
        # SWIR
        "4": (1.600, 1.700),
        "5": (2.145, 2.185),
        "6": (2.185, 2.225),
        "7": (2.235, 2.285),
        "8": (2.295, 2.365),
        "9": (2.360, 2.430),
        # TIR
        "10": (8.125, 8.475),
        "11": (8.475, 8.825),
        "12": (8.925, 9.275),
        "13": (10.25, 10.95),
        "14": (10.95, 11.65)
    }

    # Unit conversion coefficients ucc [W m-2 sr-1 um-1] for different gain
    # settings, high, normal, low1, and low2.
    ucc = {
        **{k: dict(zip(["HGH", "NOR", "LOW", "LO2"], v)) for k, v in {
            "1": (0.676, 1.688, 2.25),
            "2": (0.708, 1.415, 1.89),
            "3N": (0.423, 0.862, 1.15),
            "3B": (0.423, 0.862, 1.15),
            "4": (0.1087, 0.2174, 0.290, 0.290),
            "5": (0.0348, 0.0696, 0.0925, 0.409),
            "6": (0.0313, 0.0625, 0.0830, 0.390),
            "7": (0.0299, 0.0597, 0.0795, 0.332),
            "8": (0.0209, 0.0417, 0.0556, 0.245),
            "9": (0.0159, 0.0318, 0.0424, 0.265)}.items()},
        **{k: {None: v} for k, v in {
            "10": 6.882e-3,
            "11": 6.780e-3,
            "12": 6.590e-3,
            "13": 5.693e-3,
            "14": 5.225e-3}.items()},
    }

    def __init__(self, filename):
        """Initialize ASTER image object.

        Parameters:
            filename (str): Path to ASTER L1B HDF file.
        """
        self.filename = filename

        self.meta = self.get_metadata()
        self.gain = dict(v.split(", ") for k, v in self.meta.items() if k.startswith("GAIN"))
        SolarDirection = namedtuple(
            "SolarDirection", ["azimuth", "elevation"]
        )  # SolarDirection = (0< az <360, -90< el <90)
        self.solardirection = SolarDirection(
            *self._convert_metastr(self.meta["SOLARDIRECTION"], dtype=tuple)
        )
        self.sunzenith = 90 - self.solardirection.elevation

        self.datetime = datetime.datetime.strptime(
            self.meta["CALENDARDATE"] + self.meta["TIMEOFDAY"][:13],
            "%Y-%m-%d%H:%M:%S.%f0",
        )
        SceneCenter = namedtuple("SceneCenter", ["latitude", "longitude"])
        self.scenecenter = SceneCenter(
            *self._convert_metastr(self.meta["SCENECENTER"], dtype=tuple)
        )
        CornerCoordinates = namedtuple(
            "CornerCoordinates", ["LOWERLEFT", "LOWERRIGHT", "UPPERRIGHT", "UPPERLEFT"]
        )
        self.cornercoordinates = CornerCoordinates(
            self._convert_metastr(self.meta["LOWERLEFT"], dtype=tuple),
            self._convert_metastr(self.meta["LOWERRIGHT"], dtype=tuple),
            self._convert_metastr(self.meta["UPPERRIGHT"], dtype=tuple),
            self._convert_metastr(self.meta["UPPERLEFT"], dtype=tuple),
        )

    @property
    def basename(self):
        """Filename without path."""
        return os.path.basename(self.filename)

    @staticmethod
    def _convert_metastr(metastr, dtype=None):
        """Convert metadata data type."""
        if dtype is None:
            dtype = str

        if issubclass(dtype, tuple):
            return tuple(float(f.strip()) for f in metastr.split(","))
        else:
            return dtype(metastr)

    def get_metadata(self):
        """Read full ASTER metadata information."""
        return gdal.Open(self.filename).GetMetadata()

    def get_gain(self, channel):
        """Get gain settings of specified channel."""
        gain = self.gain.get(channel, None)
        if gain == "LO1":
            gain = "LOW"  # both refer to column 3 in ucc table.
        return gain

    def get_ucc(self, channel):
        return self.ucc[channel][self.get_gain(channel)]

    def read_digitalnumbers(self, channel):
        """Read ASTER L1B raw digital numbers.

         Parameters:
             channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.

         Returns:
            ndarray: Digital numbers. Shape according to data resolution of
                the respective channel.
        """
        if channel in ("1", "2", "3N"):
            if self.meta["ASTEROBSERVATIONMODE.1"] == "VNIR1, ON":
                subsensor = "VNIR"
            else:
                raise ValueError("VNIR nadir observation mode OFF")
        elif channel == "3B":
            if self.meta["ASTEROBSERVATIONMODE.2"] == "VNIR2, ON":
                subsensor = "VNIR"
            else:
                raise ValueError("VNIR back observation mode OFF")
        elif channel in ("4", "5", "6", "7", "8", "9"):
            if self.meta["ASTEROBSERVATIONMODE.3"] == "SWIR, ON":
                subsensor = "SWIR"
            else:
                raise ValueError("SWIR observation mode OFF")
        elif channel in ("10", "11", "12", "13", "14"):
            if self.meta["ASTEROBSERVATIONMODE.4"] == "TIR, ON":
                subsensor = "TIR"
            else:
                raise ValueError("TIR observation mode OFF")
        else:
            raise ValueError("The chosen channel is not supported.")

        data_path = (
            f"HDF4_EOS:EOS_SWATH:{self.filename}:{subsensor}_Swath:"
            f"ImageData{channel}"
        )
        swath = gdal.Open(data_path)

        data = swath.ReadAsArray().astype("float")
        data[data == 0] = np.nan  # Set edge pixels to NaN.

        return data

    def get_radiance(self, channel):
        """Get ASTER radiance values.

        Read digital numbers from ASTER L1B file and convert them to spectral
        radiance values in [W m-2 sr-1 um-1] at TOA by means of unit conversion
        coefficients ucc [W m-2 sr-1 um-1] and the gain settings at the sensor.

        See also:
            :func:`read_digitalnumbers`: Reads the raw digital numbers from
                ASTER L1B HDF4 files.

        Parameters:
             channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.

         Returns:
            ndarray: Radiance values at TOA. Shape according to data
                resolution of the respective channel.

        References:
            ﻿M. Abrams, S. H., & Ramachandran, B. (2002). Aster user handbook
            version 2, p.25 [Computer software manual]. Retrieved 2017-05-25,
            from https://asterweb.jpl.nasa.gov/content/03 data/04 Documents/
            aster user guide v2.pdf.
        """
        dn = self.read_digitalnumbers(channel)
        return (dn - 1) * self.get_ucc(channel)


    def get_reflectance(self, channel):
        """Get ASTER L1B reflectance values at TOA.

        Spectral radiance values are converted to reflectance values for
        ASTER's near-infrared and short-wave channels (1-9).

        See also:
            :func:`get_radiance`: Reads the raw digital numbers from
                ASTER L1B HDF4 files and converts them to radiance values at
                TOA.

        Parameters:
             channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                '5', '6', '7', '8', '9'.

        Returns:
            ndarray: Reflectance values at TOA. Shape according to data
                resolution of the respective channel.

        References:
        Thome, K.; Biggar, S.; Slater, P (2001). Effects of assumed solar
            spectral irradiance on intercomparisons of earth-observing sensors.
            In Sensors, Systems, and Next-Generation Satellites; Proceedings of
            SPIE; December 2001; Fujisada, H., Lurie, J., Weber, K., Eds.;
            Vol. 4540, pp. 260–269.
        http://www.pancroma.com/downloads/ASTER%20Temperature%20and%20Reflectance.pdf
        http://landsathandbook.gsfc.nasa.gov/data_prod/prog_sect11_3.html
        """
        # Mean solar exo-atmospheric irradiances [W m-2 um-1] at TOA according
        # to Thome et al. 2001.
        E_sun = (
            1848,
            1549,
            1114,
            1114,
            225.4,
            86.63,
            81.85,
            74.85,
            66.49,
            59.85,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

        doy = self.datetime.timetuple().tm_yday  # day of the year (doy)

        distance = (
            0.016790956760352183
            * np.sin(-0.017024566555637135 * doy + 4.735251041365579)
            + 0.9999651354429786
        )  # acc. to Landsat Handbook

        return (
            np.pi
            * self.get_radiance(channel)
            * distance ** 2
            / E_sun[self.channels.index(channel)]
            / np.cos(np.deg2rad(self.sunzenith))
        )

    def get_brightnesstemperature(self, channel):
        """Get ASTER L1B reflectances values at TOA.

        Spectral radiance values are converted to brightness temperature values
        in [K] for ASTER's thermal channels (10-14).

        See also:
            :func:`get_radiance`: Reads the raw digital numbers from
                ASTER L1B HDF4 files and converts them to radiance values at
                TOA.

        Parameters:
             channel (str): ASTER channel number. '10', '11', '12', '13', '14'.

        Returns:
            ndarray: Brightness temperature values in [K] at TOA. Shape
                according to data resolution of the respective channel.

        References:
        http://www.pancroma.com/downloads/ASTER%20Temperature%20and%20Reflectan
            ce.pdf
        http://landsathandbook.gsfc.nasa.gov/data_prod/prog_sect11_3.html
        """
        K1 = {
            "10": 3040.136402,  # Constant K1 [W m-2 um-1].
            "11": 2482.375199,
            "12": 1935.060183,
            "13": 866.468575,
            "14": 641.326517,
        }

        K2 = {
            "10": 1735.337945,  # Constant K2 [K].
            "11": 1666.398761,
            "12": 1585.420044,
            "13": 1350.069147,
            "14": 1271.221673,
        }

        return K2[channel] / np.log((K1[channel] / self.get_radiance(channel)) + 1)

    def retrieve_cloudmask(
        self, output_binary=True, include_thermal_test=True, include_channel_r5=True
    ):
        """ASTER cloud mask.

        Four thresholding test based on visual bands distringuish between the
        dark ocean surface and bright clouds. An additional test corrects
        uncertain labeled pixels during broken cloud conditions and pixels with
        sun glint. A detailed description can be found in Werner et al., 2016.

        See also:
            :func:`get_reflectance`: Get reflectance values from ASTER's
                near-infrared and short-wave channels.
            :func:`get_brightnesstemperature`: Get brightness temperature
                values from ASTER's thermal channels.
            :func:multiple_logical`: Apply logical function to multiple
                arguments.

        Parameters:
            output_binary (bool): Switch between binary and full cloud mask
                flags.
                binary: 0 - clear (flag 2 & flag 3)
                        1 - cloudy (flag 4 & flag 5)
                full:   2 - confidently clear
                        3 - probably clear
                        4 - probably cloudy
                        5 - confidently cloudy
            include_thermal_test (bool): Switch for including test 5, which
                uses the thermal channel 14 at 11mu with 90m pixel resolution.
                The reduced resolution can introduce artificial straight cloud
                boundaries.
            include_channel_r5 (bool): Switch for including channel 5 in the
                thresholding tests. The SWIR sensor, including channel 5,
                suffered from temperature problems after May 2007. Per default,
                later recorded images are set to a value that has no influence
                on the thresholding tests.

        Returns:
            ndarray[float]: Cloud mask.

        References:
            Werner, F., Wind, G., Zhang, Z., Platnick, S., Di Girolamo, L.,
            Zhao, G., Amarasinghe, N., and Meyer, K.: Marine boundary layer
            cloud property retrievals from high-resolution ASTER observations:
            case studies and comparison with Terra MODIS, Atmos. Meas.
            Techannel., 9, 5869-5894, https://doi.org/10.5194/amt-9-5869-2016,
            2016.
        """

        # Read visual near infrared (VNIR) channels at 15m resolution.
        r1 = self.get_reflectance(channel="1")
        r2 = self.get_reflectance(channel="2")
        r3N = self.get_reflectance(channel="3N")

        # Read short-wave infrared (SWIR) channels at 30m resolution and match
        # VNIR resolution.
        r5 = self.get_reflectance(channel="5")
        if self.datetime > datetime.datetime(2007, 5, 1) or not include_channel_r5:
            # The SWIR sensor suffered from temperature problems after May
            # 2007. Images later on are set to a dummy value "1", which won't
            # influence the following thresholding tests. Swath edge NaN pixels
            # stay NaN.
            r5[~np.isnan(r5)] = 1
        r5 = np.repeat(np.repeat(r5, 2, axis=0), 2, axis=1)

        # Ratios for clear-cloudy-tests.
        r3N2 = r3N / r2
        r12 = r1 / r2

        ### TEST 1-4 ###
        # Set cloud mask to default "confidently clear".
        clmask = np.full_like(r1, CONFIDENTLY_CLEAR, dtype=np.float)

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings("ignore", r"invalid value encountered")
            # NaN values at swath edge cause warnings that can be ignored
            probably_clear = ((r3N > 0.03) & (r5 > 0.01) & (0.7 < r3N2)
                              & (r3N2 < 1.75) & (r12 < 1.45))
            probably_cloudy = ((r3N > 0.03) & (r5 > 0.015) & (0.75 < r3N2)
                               & (r3N2 < 1.75) & (r12 < 1.35))
            confidently_cloudy = ((r3N > 0.065) & (r5 > 0.02) & (0.8 < r3N2)
                                 & (r3N2 < 1.75) & (r12 < 1.2))
        
        mask_swathedge = (np.isnan(r1) | np.isnan(r2) | np.isnan(r3N) | np.isnan(r5))
        
        clmask[probably_clear] = PROBABLY_CLEAR
        clmask[probably_cloudy] = PROBABLY_CLOUDY
        clmask[confidently_cloudy] = CONFIDENTLY_CLOUDY
        clmask[mask_swathedge] = np.nan

        if include_thermal_test:
            ### TEST 5 ###
            # Uncertain warm ocean pixels, higher than the 5th percentile of
            # brightness temperature values from all "confidently clear"
            # labeled pixels, are overwritten with "confidently clear".
            # Read thermal (TIR) channel at 90m resolution and match VNIR
            # resolution.
            bt14 = self.get_brightnesstemperature(channel="14")
            bt14 = np.repeat(np.repeat(bt14, 6, axis=0), 6, axis=1)

            # Check for available "confidently clear" pixels.
            nc = np.sum(clmask == CONFIDENTLY_CLEAR) / np.sum(~np.isnan(clmask))
            if nc > 0.03:
                bt14_p05 = np.nanpercentile(bt14[clmask == CONFIDENTLY_CLEAR], 5)
            else:
                # If less than 3% of pixels are "confidently clear", test 5
                # cannot be applied according to Werner et al., 2016. However,
                # a sensitivity study showed that combining "probably clear"
                # and "confidently clear" pixels in such cases leads to
                # plausible results and we derive a threshold correspondingly.
                bt14_p05 = np.nanpercentile(
                    bt14[(clmask == CONFIDENTLY_CLEAR) | (clmask == PROBABLY_CLEAR)], 5)

            with np.warnings.catch_warnings():
                np.warnings.filterwarnings("ignore", r"invalid value encountered")
                # Pixels with brightness temperature values above the 5th
                # percentile of clear ocean pixels are overwritten with
                # "confidently clear".
                clmask[(bt14 > bt14_p05) & ~np.isnan(clmask)] = CONFIDENTLY_CLEAR

            # Add swath edge pixels of thermal channel.
            clmask[np.isnan(bt14)] = np.nan

        if output_binary:
            warnings.warn("output_binary argument is deprecated. Use static method "
                          "cloudmask_to_binary instead", DeprecationWarning)
            return self.cloudmask_to_binary(clmask)

        return clmask
    
    @staticmethod
    def cloudmask_to_binary(clmask):
        clmask = clmask.copy()
        clmask[(clmask == CONFIDENTLY_CLEAR) | (clmask == PROBABLY_CLEAR)] = CLEAR
        clmask[(clmask == CONFIDENTLY_CLOUDY) | (clmask == PROBABLY_CLOUDY)] = CLOUDY
        return clmask

    def read_coordinates(self, channel="1"):
        """Read reduced latitude and longitude grid.

        Extract geolocation table containing latitude and longitude values at
        11 x 11 lattice points. Latitudes are provided in geocentric coordinates
        and are maped to geodetic values.

        Parameters:
            channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                    '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.

        Returns:
            ndarray, ndarray: latitude, longitude
        """

        sdsidx = {"VNIR": (0, 1), "SWIR": (6, 7), "TIR": (14, 15)}

        latstr = ":".join(
            (
                "HDF4_SDS",
                "UNKNOWN",
                f"{self.filename}",
                f"{sdsidx[self.channel2sensor[channel]][0]}",
            )
        )
        lat = gdal.Open(latstr)
        lat = geocentric2geodetic(lat.ReadAsArray().astype("float"))

        lonstr = ":".join(
            (
                "HDF4_SDS",
                "UNKNOWN",
                f"{self.filename}",
                f"{sdsidx[self.channel2sensor[channel]][1]}",
            )
        )
        lon = gdal.Open(lonstr)
        lon = lon.ReadAsArray().astype("float")

        return lat, lon

    def get_latlon_grid(self, channel="1"):
        """Create latitude-longitude-grid for specified channel data.

        A latitude-logitude grid is created from geolocation information from
        11 x 11 boxes corresponding to the image data. The resolution and dimension
        of the image and latitude-logitude grid depend on the specified channel.

        Parameters:
            channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.

        Returns:
            ndarray[tuple]: Latitude longitude grid.

        References:
            M. Abrams, S. H., & Ramachandran, B. (2002). Aster user handbook
            version 2, p.25 [Computer software manual]. Retrieved 2017-05-25,
            from https://asterweb.jpl.nasa.gov/content/03 data/04 Documents/
            aster user guide v2.pdf.
        """
        ImageDimension = namedtuple("ImageDimension", ["lon", "lat"])
        imagedimension = ImageDimension(
            *self._convert_metastr(self.meta[f"IMAGEDATAINFORMATION{channel}"], tuple)[
                :2
            ]
        )

        lat, lon = self.read_coordinates(channel=channel)

        latidx = np.arange(11) * imagedimension.lat / 10
        lonidx = np.arange(11) * imagedimension.lon / 10

        # Function for interpolation to full domain grid.
        flat = interpolate.interp2d(lonidx, latidx, lat, kind="linear")
        flon = interpolate.interp2d(lonidx, latidx, lon, kind="linear")

        # Grid matching shifted corner coordinates (see above).
        latgrid = np.arange(0, int(imagedimension.lat) + 1, 1)
        longrid = np.arange(0, int(imagedimension.lon) + 1, 1)

        # Apply lat-lon-function to grid and cut off last row and column to get
        # the non-shifted grid.
        lats = flat(longrid, latgrid)[:-1, :-1]
        lons = flon(longrid, latgrid)[:-1, :-1]

        return lats, lons

    def get_lapserate(self):
        """Estimate lapse rate from MODIS climatology using :func:`lapserate`."""
        return lapserate_modis(self.datetime.month, self.scenecenter[0])

    def get_cloudtopheight(self):
        """Estimate the cloud top height according to Baum et al., 2012 
        using :func: `cloudtopheight_IR`."""
        bt = self.get_brightnesstemperature("14")
        cloudmask = self.retrieve_cloudmask()
        latitude = self.scenecenter.latitude
        month = self.datetime.month

        return cloudtopheight_IR(bt, cloudmask, latitude, month, method="modis")

    def dt_estimate_scanlines(self, sensor="vnir"):
        """Estimate the date time per scanline.

        Based on the approximate recording time for one ASTER image a date time
        array is constructed along the flight direction and depending on the
        sensor resolution.

        Parameters:
            sensor (str): ASTER sensor ("vnir", "swir", or "tir").

        Returns:
            (ndarray[datetime]): date time information per scanline.
        """
        dtdelta = datetime.timedelta(seconds=8, milliseconds=849)
        scanlines = {"vnir": 4200, "swir": 2100, "tir": 700}

        return np.linspace(-0.5, 0.5, scanlines[sensor]) * dtdelta + self.datetime

    def sensor_angles(self, channel="1"):
        """Calculate sensor zenith and azimuth angles.
        
        Angles are derived for each pixel depending on the channel and the
        corresponding ASTER subsensor ("VNIR", "SWIR", "TIR"), as well as on the
        subsensor geometry and settings.

        Note:
            All angular values are given in degree.

        Parameters:
            channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                    '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.

        Returns:
            ndarray, ndarray: sensor zenith, sensor azimuth angles in degree.

        References:
             Kang Yang, Huaguo Zhang, Bin Fu, Gang Zheng, Weibing Guan, Aiqin Shi
             & Dongling Li (2015) Observation of submarine sand waves using ASTER
             stereo sun glitter imagery, International Journal of Remote Sensing,
             36:22, 5576-5592, DOI: 10.1080/01431161.2015.1101652
             Algorithm Theoretical Basis Document for ASTER Level-1 Data 
             Processing (Ver. 3.0).1996.
        """
        if channel != "3B":
            sensor = self.channel2sensor[channel]
        else:
            sensor = "VNIRB"

        # Angular data from ASTER metadata data.
        S = float(self.meta["MAPORIENTATIONANGLE"])

        FOV = {"VNIR": 6.09, "VNIRB": 5.19, "SWIR": 4.9, "TIR": 4.9}

        P = {
            "VNIR": float(self.meta["POINTINGANGLE.1"]),
            "VNIRB": float(self.meta["POINTINGANGLE.1"]),
            "SWIR": float(self.meta["POINTINGANGLE.2"]),
            "TIR": float(self.meta["POINTINGANGLE.3"]),
        }

        # cut overlap area of backward pointing telescope
        if channel != "3B":
            field = self.read_digitalnumbers(channel)
        elif channel == "3B" and self.meta["FLYINGDIRECTION"] == "DE":
            field = self.read_digitalnumbers(channel)[400:]
        elif channel == "3B" and self.meta["FLYINGDIRECTION"] == "AE":
            field = self.read_digitalnumbers(channel)[:400]

        # design n field
        sidx = np.arange(np.shape(field)[1])

        mid0 = sidx[np.isfinite(field[5, :])][[0, -1]].mean()
        mid1 = sidx[np.isfinite(field[-5, :])][[0, -1]].mean()

        f = interpolate.interp1d(
            np.array([5, np.shape(field)[0] - 5]),
            np.array([mid0, mid1]),
            kind="linear",
            fill_value="extrapolate",
        )

        mids = f(np.arange(np.shape(field)[0]))
        # costructing an n-array indexing the pixels symmetric to the center of the
        # swath. If pointing angle is zero, the sensor zenith angle is zero in the
        # swath center.
        n = sidx - mids[:, np.newaxis]

        # left and right side of nadir are defined such that the sign follows the
        # roll angle sign, which is negative on the right and positive on the left
        # side of the sensor in flying direction (!), NOT in projected image. The
        # sides therefore depend on the ascending / decending mode defined in the
        # meta data.
        flyingdir = self.meta["FLYINGDIRECTION"]
        if flyingdir is "DE":
            n *= -1

        swath_widths = np.sum(np.isfinite(field), axis=1)
        # average swath width, but exluding possible NaN-scanlines at beginning and
        # end of the image.
        swath_width = np.mean(swath_widths[swath_widths > 4200])

        n_angles = n * FOV[sensor] / swath_width + P[sensor]
        azimuth = np.full_like(field, np.nan)

        if channel != "3B":
            zenith = abs(n_angles)

            if flyingdir is "DE":
                azimuth[n_angles > 0] = 90 + S
                azimuth[n_angles <= 0] = 270 + S
            else:
                azimuth[n_angles < 0] = 90 + S
                azimuth[n_angles >= 0] = 270 + S
        else:
            h = 705000  # in km above the equator
            zenith = np.rad2deg(
                np.arctan(
                    np.sqrt(
                        (h * np.tan(np.deg2rad(P[sensor])) + 15 * n) ** 2
                        + (h * np.tan(np.deg2rad(27.6)) / np.cos(np.deg2rad(P[sensor])))
                        ** 2
                    )
                    / h
                )
            )

            x = np.rad2deg(np.arctan(0.6 / np.tan(np.deg2rad(n_angles))))
            if flyingdir is "DE":
                azimuth[n_angles > 0] = np.array(90 - x + S)[n_angles > 0]
                azimuth[n_angles <= 0] = np.array(270 - x + S)[n_angles <= 0]
            else:
                azimuth[n_angles < 0] = np.array(90 - x + S)[n_angles < 0]
                azimuth[n_angles >= 0] = np.array(270 - x + S)[n_angles >= 0]

        zenith[np.isnan(field)] = np.nan
        azimuth[np.isnan(field)] = np.nan

        return zenith, azimuth

    def reflection_angles(self):
        """Calculate the reflected sun angle, theta_r, of specular reflection
        of sunlight into an instrument sensor.

        Returns:
            (ndarray): 2d field of size `cloudmask` of reflection
            angles in [°] for eachannel pixel.

        References:
             Kang Yang, Huaguo Zhang, Bin Fu, Gang Zheng, Weibing Guan, Aiqin Shi
             & Dongling Li (2015) Observation of submarine sand waves using ASTER
             stereo sun glitter imagery, International Journal of Remote Sensing,
             36:22, 5576-5592, DOI: 10.1080/01431161.2015.1101652
        """
        sun_azimuth = (
            self.solardirection.azimuth + 180
        )  # +180° corresponds to "anti-/reflected" sun
        sun_zenith = (
            90 - self.solardirection.elevation
        )  # sun zenith = 90 - sun elevation

        sensor_zenith, sensor_azimuth = self.sensor_angles()

        return np.degrees(
            np.arccos(
                np.cos(np.deg2rad(sensor_zenith)) * np.cos(np.deg2rad(sun_zenith))
                + np.sin(np.deg2rad(sensor_zenith))
                * np.sin(np.deg2rad(sun_zenith))
                * np.cos(np.deg2rad(sensor_azimuth) - np.deg2rad(sun_azimuth))
            )
        )


def lapserate_modis(month, latitude):
    """Estimate of the apparent lapse rate in [K/km].
    Typical lapse rates are assumed for each month and depending on the
    location on Earth, i.e. southern hemisphere, tropics, or northern
    hemisphere. For a specific case the lapse rate is estimated by a 4th order
    polynomial and polynomial coefficients given in the look-up table.
    This approach is based on the MODIS cloud top height retrieval and applies
    to data recorded at 11 microns. According to Baum et al., 2012, the
    predicted lapse rates are restricted to a maximum and minimum of
    10 and 2 K/km, respectively.


    Parameters:
        month (int): Month of the year.
        latitude (float): Latitude of center coordinate.

    Returns:
        float: Lapse rate estimate.

    References:
        Baum, B.A., W.P. Menzel, R.A. Frey, D.C. Tobin, R.E. Holz, S.A.
        Ackerman, A.K. Heidinger, and P. Yang, 2012: MODIS Cloud-Top Property
        Refinements for Collection 6. J. Appl. Meteor. Climatol., 51,
        1145–1163, https://doi.org/10.1175/JAMC-D-11-0203.1
    """
    lapserate_lut = {
        "month": np.arange(1, 13),
        "SH_transition": np.array(
            [-3.8, -21.5, -2.8, -23.4, -12.3, -7.0, -10.5, -7.8, -8.6, -7.0, -9.2, -3.7]
        ),
        "NH_transition": np.array(
            [22.1, 12.8, 10.7, 29.4, 14.9, 16.8, 15.0, 19.5, 17.4, 27.0, 22.0, 19.0]
        ),
        "a0": np.array(
            [
                (2.9769801, 2.9426577, 1.9009563),  # Jan
                (3.3483239, 2.6499606, 2.4878736),  # Feb
                (2.4060296, 2.3652047, 3.1251275),  # Mar
                (2.6522387, 2.5433158, 13.3931707),  # Apr
                (1.9578263, 2.4994028, 1.6432070),  # May
                (2.7659754, 2.7641496, -5.2366360),  # Jun
                (2.1106812, 3.1202043, -4.7396481),  # Jul
                (3.0982174, 3.4331195, -1.4424843),  # Aug
                (3.0760552, 3.4539390, -3.7140186),  # Sep
                (3.6377215, 3.6013337, 8.2237401),  # Oct
                (3.3206165, 3.1947419, -0.4502047),  # Nov
                (3.0526633, 3.1276377, 9.3930897),
            ]
        ),  # Dec
        "a1": np.array(
            [
                (-0.0515871, -0.0510674, 0.0236905),
                (0.1372575, -0.0105152, -0.0076514),
                (0.0372002, 0.0141129, -0.1214572),
                (0.0325729, -0.0046876, -1.2206948),
                (-0.2112029, -0.0364706, 0.1151207),
                (-0.1186501, -0.0728625, 1.0105575),
                (-0.3073666, -0.1002375, 0.9625734),
                (-0.1629588, -0.1021766, 0.4769307),
                (-0.2043463, -0.1158262, 0.6720954),
                (-0.0857784, -0.0775800, -0.5127533),
                (-0.1411094, -0.1045316, 0.2629680),
                (-0.1121522, -0.0707628, -0.8836682),
            ]
        ),
        "a2": np.array(
            [
                (0.0027409, 0.0052420, 0.0086504),
                (0.0133259, 0.0042896, 0.0079444),
                (0.0096473, 0.0059242, 0.0146488),
                (0.0100893, 0.0059325, 0.0560381),
                (-0.0057944, 0.0082002, 0.0033131),
                (0.0011627, 0.0088878, -0.0355440),
                (-0.0090862, 0.0064054, -0.0355847),
                (-0.0020384, 0.0010499, -0.0139027),
                (-0.0053970, 0.0015450, -0.0210550),
                (0.0024313, 0.0041940, 0.0205285),
                (-0.0026068, 0.0049986, -0.0018419),
                (-0.0009913, 0.0055533, 0.0460453),
            ]
        ),
        "a3": np.array(
            [
                (0.0001136, 0.0001097, -0.0002167),
                (0.0003043, 0.0000720, -0.0001774),
                (0.0002334, -0.0000159, -0.0003188),
                (0.0002601, 0.0000144, -0.0009874),
                (-0.0001050, 0.0000844, -0.0001458),
                (0.0000937, 0.0001768, 0.0005188),
                (-0.0000890, 0.0002620, 0.0005522),
                (0.0000286, 0.0001616, 0.0001759),
                (-0.0000541, 0.00017117, 0.0002974),
                (0.0001495, 0.0000941, -0.0003016),
                (0.0000058, 0.0001911, -0.0000369),
                (0.0000180, 0.0001550, -0.0008450),
            ]
        ),
        "a4": np.array(
            [
                (0.00000113, -0.00000372, 0.00000151),
                (0.00000219, -0.0000067, 0.00000115),
                (0.00000165, -0.00000266, 0.00000210),
                (0.00000199, -0.00000346, 0.00000598),
                (-0.00000074, -0.00000769, 0.00000129),
                (0.00000101, -0.00001168, -0.00000262),
                (0.00000004, -0.00001079, -0.00000300),
                (0.00000060, 0.00000510, -0.00000080),
                (-0.00000002, 0.00000248, -0.00000150),
                (0.00000171, -0.0000041, 0.00000158),
                (0.00000042, -0.00000506, 0.00000048),
                (0.00000027, -0.00000571, 0.00000518),
            ]
        ),
    }

    ind_month = month - 1

    if latitude < lapserate_lut["SH_transition"][ind_month]:
        region_flag = 0  # Southern hemisphere
    elif (
        latitude >= lapserate_lut["SH_transition"][ind_month]
        and latitude <= lapserate_lut["NH_transition"][ind_month]
    ):
        region_flag = 1  # Tropics
    elif latitude > lapserate_lut["NH_transition"][ind_month]:
        region_flag = 2  # Northern hemisphere
    else:
        raise ValueError("Latitude of center coordinate cannot be read.")

    lapserate = (
        lapserate_lut["a0"][ind_month][region_flag]
        + lapserate_lut["a1"][ind_month][region_flag] * latitude
        + lapserate_lut["a2"][ind_month][region_flag] * latitude ** 2
        + lapserate_lut["a3"][ind_month][region_flag] * latitude ** 3
        + lapserate_lut["a4"][ind_month][region_flag] * latitude ** 4
    )

    return lapserate


def lapserate_moist_adiabate():
    """Moist adiabatic lapse rate in [K/km].
    """
    return 6.5


def cloudtopheight_IR(bt, cloudmask, latitude, month, method="modis"):
    """Cloud Top Height (CTH) from 11 micron channel.
    
    Brightness temperatures (bt) are converted to CTHs using the IR window approach: 
    (bt_clear - bt_cloudy) / lapse_rate.

    See also:
        :func:`skimage.measure.block_reduce`
            Down-sample image by applying function to local blocks.
        :func:`lapserate_moist_adiabate`
            Constant value 6.5 [K/km]
        :func:`lapserate_modis`
            Estimate of the apparent lapse rate in [K/km]
            depending on month and latitude acc. to Baum et al., 2012.

    Parameters:
        bt (ndarray): brightness temperatures form 11 micron channel.
        cloudmask (ndarray): binary cloud mask.
        month (int): month of the year.
        latitude (ndarray): latitudes in [°], positive North, negative South.
        method (str): approach used to derive CTH: 'modis' see Baum et al., 2012,
            'simple' uses the moist adiabatic lapse rate.

    Returns:
        ndarray: cloud top height.
    
    References:
        Baum, B.A., W.P. Menzel, R.A. Frey, D.C. Tobin, R.E. Holz, S.A.
        Ackerman, A.K. Heidinger, and P. Yang, 2012: MODIS Cloud-Top Property
        Refinements for Collection 6. J. Appl. Meteor. Climatol., 51,
        1145–1163, https://doi.org/10.1175/JAMC-D-11-0203.1
    """
    # Lapse rate
    if method == "simple":
        lapserate = lapserate_moist_adiabate()

    elif method == "modis":
        lapserate = lapserate_modis(month, latitude)
    else:
        raise ValueError("Method is not supported.")

    resolution_ratio = np.shape(cloudmask)[0] // np.shape(bt)[0]

    cloudmask_inverted = cloudmask.copy()
    cloudmask_inverted[np.isnan(cloudmask_inverted)] = 1
    cloudmask_inverted = np.asarray(
        np.invert(np.asarray(cloudmask_inverted, dtype=bool)), dtype=int
    )

    cloudmask[np.isnan(cloudmask)] = 0
    cloudmask = np.asarray(cloudmask, dtype=int)

    # Match resolutions of cloud mask and brightness temperature (bt) arrays.
    if resolution_ratio > 1:
        # On bt resolution, flag pixels as cloudy only if all subgrid pixels
        # are cloudy in the original cloud mask.
        mask_cloudy = block_reduce(
            cloudmask, (resolution_ratio, resolution_ratio), func=np.alltrue
        )
        # Search for only clear pixels to derive a bt clearsky/ocean value.
        mask_clear = block_reduce(
            cloudmask_inverted, (resolution_ratio, resolution_ratio), func=np.alltrue
        )
    elif resolution_ratio < 1:
        try:
            mask_cloudy = np.repeat(
                np.repeat(cloudmask, resolution_ratio, axis=0), resolution_ratio, axis=1
            )
            mask_clear = np.repeat(
                np.repeat(cloudmask_inverted, resolution_ratio, axis=0),
                resolution_ratio,
                axis=1,
            )
        except ValueError:
            raise ValueError(
                "Problems matching the shapes of provided cloud mask and bt arrays."
            )
    else:
        mask_cloudy = cloudmask.copy()
        mask_clear = cloudmask_inverted.copy()

    bt_cloudy = np.ones(np.shape(bt)) * np.nan
    bt_cloudy[mask_cloudy] = bt[mask_cloudy]
    bt_clear_avg = np.nanmean(bt[mask_clear])

    return (bt_clear_avg - bt_cloudy) / lapserate


def geocentric2geodetic(latitude):
    """Translate geocentric to geodetic latitudes.

    Parameters:
        latitude (ndarray): latitude values in degree.

    Returns:
        (ndarray): geodetic latitudes.
    """

    return np.rad2deg(np.arctan(1.0067395 * np.tan(np.deg2rad(latitude))))
