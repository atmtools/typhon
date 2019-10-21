# -*- coding: utf-8 -*-
"""Functions to work with ASTER L1B satellite data.
"""
import datetime
import logging
import os
from collections import namedtuple

import gdal
import numpy as np
from skimage.measure import block_reduce

from typhon.math import multiple_logical


__all__ = [
    'ASTERimage',
    'cloudtopheight_IR',
    'lapserate_modis',
    'lapserate_moist_adiabate',
    'get_reflection_angle',
    'theta_r',
]

logger = logging.getLogger(__name__)


class ASTERimage:
    """ASTER L1B image object."""
    channels = (
        '1', '2', '3N', '3B',  # VNIR
        '4', '5', '6', '7', '8', '9',  # SWIR
        '10', '11', '12', '13', '14'  # TIR
    )

    def __init__(self, filename):
        """Initialize ASTER image object.

        Parameters:
            filename (str): Path to ASTER L1B HDF file.
        """
        self.filename = filename

        meta = self.get_metadata()
        SolarDirection = namedtuple('SolarDirection', ['azimuth', 'elevation'])
        self.solardirection = SolarDirection(
            *self._convert_metastr(meta['SOLARDIRECTION'], dtype=tuple)
        )
        self.sunelevation = self.solardirection[1]
        self.datetime = datetime.datetime.strptime(
            meta['CALENDARDATE'] + meta['TIMEOFDAY'], '%Y-%m-%d%H:%M:%S.%f0')
        self.scenecenter = self._convert_metastr(meta['SCENECENTER'],
                                                 dtype=tuple)


    @property
    def basename(self):
        """Filename without path."""
        return os.path.basename(self.filename)

    @staticmethod
    def _convert_metastr(metastr, dtype=str):
        """Convert metadata data type."""
        if dtype == tuple:
            return tuple(float(f.strip()) for f in metastr.split(','))
        else:
            return dtype(metastr)

    def get_metadata(self):
        """Read full ASTER metadata information."""
        return gdal.Open(self.filename).GetMetadata()

    def read_digitalnumbers(self, channel):
        """Read ASTER L1B raw digital numbers.

         Parameters:
             channel (str): ASTER channel number. '1', '2', '3N', '3B', '4',
                '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'.

         Returns:
            ndarray: Digital numbers. Shape according to data resolution of
                the respective channel.
        """
        if channel in ('1', '2', '3N', '3B'):
            subsensor = 'VNIR'
        elif channel in ('4', '5', '6', '7', '8', '9'):
            subsensor = 'SWIR'
        elif channel in ('10', '11', '12', '13', '14'):
            subsensor = 'TIR'

        data_path = (f'HDF4_EOS:EOS_SWATH:{self.filename}:{subsensor}_Swath:'
                     f'ImageData{channel}')
        swath = gdal.Open(data_path)

        data = swath.ReadAsArray().astype('float')
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

        # Unit conversion coefficients ucc [W m-2 sr-1 um-1] for different gain
        # settings, high, normal, low1, and low2.
        ucc = {'1': (0.676, 1.688, 2.25, np.nan),
               '2': (0.708, 1.415, 1.89, np.nan),
               '3N': (0.423, 0.862, 1.15, np.nan),
               '3B': (0.423, 0.862, 1.15, np.nan),
               '4': (0.1087, 0.2174, 0.290, 0.290),
               '5': (0.0348, 0.0696, 0.0925, 0.409),
               '6': (0.0313, 0.0625, 0.0830, 0.390),
               '7': (0.0299, 0.0597, 0.0795, 0.332),
               '8': (0.0209, 0.0417, 0.0556, 0.245),
               '9': (0.0159, 0.0318, 0.0424, 0.265),
               '10': 6.882e-3,
               '11': 6.780e-3,
               '12': 6.590e-3,
               '13': 5.693e-3,
               '14': 5.225e-3,
               }

        if channel in ['1', '2', '3N', '3B', '4', '5', '6', '7', '8', '8',
                       '9']:
            meta = self.get_metadata()
            gain = meta[f'GAIN.{channel[0]}'].split(',')[1].strip()
            if gain == 'LO1':
                gain = 'LOW' # both refer to column 3 in ucc table.
            radiance = (dn - 1) * ucc[channel][
                                    ['HGH', 'NOR', 'LOW', 'LO2'].index(gain)]
        elif channel in ['10', '11', '12', '13', '14']:
            radiance = (dn - 1) * ucc[channel]
        else:
            raise ValueError('Invalid channel "{channel}".')

        return radiance

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
        E_sun = (1848, 1549, 1114, 1114,
                 225.4, 86.63, 81.85, 74.85, 66.49, 59.85,
                 np.nan, np.nan, np.nan, np.nan, np.nan)

        sunzenith = 90 - self.solardirection.elevation

        doy = self.datetime.timetuple().tm_yday # day of the year (doy)

        distance = (0.016790956760352183
                    * np.sin(-0.017024566555637135 * doy
                             + 4.735251041365579)
                    + 0.9999651354429786) # acc. to Landsat Handbook

        return (np.pi * self.get_radiance(channel) * distance**2 /
                E_sun[self.channels.index(channel)] /
                np.cos(np.radians(sunzenith))
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
        K1 = {'10': 3040.136402, # Constant K1 [W m-2 um-1].
              '11': 2482.375199,
              '12': 1935.060183,
              '13': 866.468575,
              '14': 641.326517}

        K2 = {'10': 1735.337945, # Constant K2 [K].
              '11': 1666.398761,
              '12': 1585.420044,
              '13': 1350.069147,
              '14': 1271.221673}

        return (K2[channel] /
                np.log((K1[channel] / self.get_radiance(channel)) + 1)
                )

    def retrieve_cloudmask(self, output_binary=True, include_thermal_test=True,
                           include_channel_r5=True):
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
        r1 = self.get_reflectance(channel='1')
        r2 = self.get_reflectance(channel='2')
        r3N = self.get_reflectance(channel='3N')

        # Read short-wave infrared (SWIR) channels at 30m resolution and match
        # VNIR resolution.
        r5 = self.get_reflectance(channel='5')
        if (self.datetime > datetime.datetime(2007, 5, 1)
            or not include_channel_r5):
            # The SWIR sensor suffered from temperature problems after May
            # 2007. Images later on are set to a dummy value "1", which won't
            # influence the following thresholding tests. Swath edge NaN pixels
            # stay NaN.
            r5[~np.isnan(r5)] = 1
        r5 = np.repeat(np.repeat(r5, 2, axis=0), 2, axis=1)

        # Read thermal (TIR) channel at 90m resolution and match VNIR
        # resolution.
        bt14 = self.get_brightnesstemperature(channel='14')
        bt14 = np.repeat(np.repeat(bt14, 6, axis=0), 6, axis=1)

        # Ratios for clear-cloudy-tests.
        r3N2 = r3N / r2
        r12 = r1 / r2

        ### TEST 1-4 ###
        # Set cloud mask to default "confidently clear".
        clmask = np.ones(r1.shape, dtype=np.float) * 2

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', r'invalid value encountered')

            # Set "probably clear" pixels.
            clmask[multiple_logical(r3N > 0.03, r5 > 0.01, 0.7 < r3N2,
                                    r3N2 < 1.75, r12 < 1.45,
                                    func=np.logical_and)] = 3


            # Set "probably cloudy" pixels.
            clmask[multiple_logical(r3N > 0.03, r5 > 0.015, 0.75 < r3N2,
                                    r3N2 < 1.75, r12 < 1.35,
                                    func=np.logical_and)] = 4

            # Set "confidently cloudy" pixels
            clmask[multiple_logical(r3N > 0.065, r5 > 0.02, 0.8 < r3N2,
                                    r3N2 < 1.75, r12 < 1.2,
                                    func=np.logical_and)] = 5

            # Combine swath edge pixels.
            clmask[multiple_logical(np.isnan(r1), np.isnan(r2), np.isnan(r3N),
                                    np.isnan(r5), func=np.logical_or)] = np.nan

        if include_thermal_test:
            ### TEST 5 ###
            # Uncertain warm ocean pixels, higher than the 5th percentile of
            # brightness temperature values from all "confidently clear"
            # labeled pixels, are overwritten with "confidently clear".

            # Check for available "confidently clear" pixels.
            nc = np.sum(clmask == 2) / np.sum(~np.isnan(clmask))
            if (nc > 0.03):
                bt14_p05 = np.nanpercentile(bt14[clmask == 2], 5)
            else:
                # If less than 3% of pixels are "confidently clear", test 5
                # cannot be applied according to Werner et al., 2016. However,
                # a sensitivity study showed that combining "probably clear"
                # and "confidently clear" pixels in such cases leads to
                # plausible results and we derive a threshold correspondingly.
                bt14_p05 = np.nanpercentile(bt14[np.logical_or(clmask == 2,
                                                               clmask == 3)],5)

            with np.warnings.catch_warnings():
                np.warnings.filterwarnings('ignore',
                                           r'invalid value encountered')
                # Pixels with brightness temperature values above the 5th
                # percentile of clear ocean pixels are overwritten with
                # "confidently clear".
                clmask[np.logical_and(bt14 > bt14_p05, ~np.isnan(clmask))] = 2

            # Combine swath edge pixels.
            clmask[np.logical_or(np.isnan(clmask), np.isnan(bt14))] = np.nan

        if output_binary:
            clmask[np.logical_or(clmask == 2, clmask == 3)] = 0 # clear
            clmask[np.logical_or(clmask == 4, clmask == 5)] = 1 # cloudy

        return clmask


def cloudtopheight_IR(cloudmask, Tb14, method='simple', latitudes=None,
                      month=None):
    """Cloud Top Height (CTH) from 11 micron channel.
    ASTER brightness temperatures from channel 14 are converted to CTHs using
    the IR window approachannel: (Tb_clear - Tb_cloudy) / lapse_rate.

    Note:
        `cloudmask` shape must match that of the thermal channel, (700,830).

    See also:
        :func:`skimage.measure.block_reduce`: Down-sample image by applying
            function to local blocks.
        :func:`lapserate_moist_adiabate`: Constant value 6.5 [K/km]
        :func:`lapserate_modis`: Estimate of the apparent lapse rate in [K/km]
            depending on month and latitude acc. to Baum et al., 2012.

    Parameters:
        cloudmask (ndarray): binary ASTER cloud mask image.
        Tb14 (ndarray): brightness temperatures form ASTER channel 14.
        method (str): approachannel used to derive CTH: 'simple' or 'modis'.
        latitudes (ndarray): latitudes in [°], positive North, negative South.
        month (int): month of the year.

    Returns:
        ndarray: cloud top height.
    """
    # Lapse rate
    if method=='simple':
        lapserate = lapserate_moist_adiabate()

    elif method=='modis':
        latitude = latitudes[np.shape(latitudes)[0] // 2,
                             np.shape(latitudes)[1] // 2]
        lapserate = lapserate_modis(month=month, latitude=latitude)

    resolution_ratio = np.shape(cloudmask)[0] // np.shape(Tb14)[0]

    cloudmask_inverted = cloudmask.copy()
    cloudmask_inverted[np.isnan(cloudmask_inverted)] = 1
    cloudmask_inverted = np.asarray(np.invert(np.asarray(
                                cloudmask_inverted, dtype=bool)), dtype=int)

    cloudmask[np.isnan(cloudmask)] = 0
    cloudmask = np.asarray(cloudmask, dtype=int)

    # Match cloudmask resolution and Tb14 resolution.
    if resolution_ratio > 1:
        # On Tb14 resolution, flag pixels as cloudy only if all subgrid pixels
        # are cloudy in the original cloud mask.
        mask_cloudy = block_reduce(cloudmask,
                                        (resolution_ratio, resolution_ratio),
                                        func=np.alltrue)
        # Search for 90m only clear pixels to derive a Tb clearsky/ocean value.
        mask_clear = block_reduce(cloudmask_inverted,
                                        (resolution_ratio, resolution_ratio),
                                        func=np.alltrue)
    elif resolution_ratio < 1:
        try:
            mask_cloudy = np.repeat(np.repeat(cloudmask, resolution_ratio,
                                        axis=0), resolution_ratio, axis=1)
            mask_clear = np.repeat(np.repeat(cloudmask_inverted,
                                                  resolution_ratio, axis=0),
                                                  resolution_ratio, axis=1)
        except ValueError:
            logger.warning(
                'Problems matching the shapes of cloudmask and Tb14.')
    else:
        mask_cloudy = cloudmask.copy()
        mask_clear = cloudmask_inverted.copy()

    Tb_cloudy = np.ones(np.shape(Tb14)) * np.nan
    Tb_cloudy[mask_cloudy] = Tb14[mask_cloudy]

    Tb_clear_avg = np.nanmean(Tb14[mask_clear])

    return (Tb_clear_avg - Tb_cloudy) / lapserate


def lapserate_modis(month, latitude):
    """Estimate of the apparent lapse rate in [K/km].
    Typical lapse rates are assumed for eachannel month and depending on the
    location on Earth, i.e. southern hemisphere, tropics, or northern
    hemisphere. For a specific case the lapse rate is estimated by a 4th order
    polynomial and polynomial coefficients given in the look-up table.
    This approachannel is based on the MODIS cloud top height retrieval and applies
    to data recorded at 11 microns.

    Note:
        Northern hemisphere only!

    Parameters:
        month (int): month of the year.
        latitude (float): latitude.

    Returns:
        float: lapse rate.

    References:
        Baum, B.A., W.P. Menzel, R.A. Frey, D.C. Tobin, R.E. Holz, S.A.
        Ackerman, A.K. Heidinger, and P. Yang, 2012: MODIS Cloud-Top Property
        Refinements for Collection 6. J. Appl. Meteor. Climatol., 51,
        1145–1163, https://doi.org/10.1175/JAMC-D-11-0203.1
    """
    lapserate_lut = {'month': np.arange(1,13),
        'lat_split': np.array([22.1, 12.8, 10.7, 29.4, 14.9, 16.8,
                               15.0, 19.5, 17.4, 27.0, 22.0, 19.0]),
        'a0': np.array([(2.9426577,1.9009563), (2.6499606,2.4878736),
                        (2.3652047,3.1251275), (2.5433158,13.3931707),
                        (2.4994028,1.6432070), (2.7641496, -5.2366360),
                        (3.1202043,-4.7396481), (3.4331195,-1.4424843),
                        (3.4539390,-3.7140186), (3.6013337,8.2237401),
                        (3.1947419,-0.4502047), (3.1276377,9.3930897) ]),
        'a1': np.array([(-0.0510674,0.0236905), (-0.0105152, -0.0076514),
                        (0.0141129,-0.1214572), (-0.0046876,-1.2206948),
                        (-0.0364706,0.1151207), (-0.0728625, 1.0105575),
                        (-0.1002375,0.9625734), (-0.1021766,0.4769307),
                        (-0.1158262,0.6720954), (-0.0775800,-0.5127533),
                        (-0.1045316,0.2629680), (-0.0707628,-0.8836682) ]),
        'a2': np.array([(0.0052420,0.0086504), (0.0042896,0.0079444),
                        (0.0059242,0.0146488), (0.0059325,0.0560381),
                        (0.0082002,0.0033131), (0.0088878,-0.0355440),
                        (0.0064054,-0.0355847), (0.0010499,-0.0139027),
                        (0.0015450,-0.0210550), (0.0041940,0.0205285),
                        (0.0049986,-0.0018419), (0.0055533,0.0460453) ]),
        'a3': np.array([(0.0001097,-0.0002167), (0.0000720,-0.0001774),
                        (-0.0000159,-0.0003188), (0.0000144,-0.0009874),
                        (0.0000844,-0.0001458), (0.0001768,0.0005188),
                        (0.0002620,0.0005522), (0.0001616,0.0001759),
                        (0.00017117,0.0002974),(0.0000941,-0.0003016),
                        (0.0001911,-0.0000369), (0.0001550,-0.0008450) ]),
        'a4': np.array([(-0.00000372,0.00000151), (-0.0000067,0.00000115),
                        (-0.00000266,0.00000210), (-0.00000346,0.00000598),
                        (-0.00000769,0.00000129), (-0.00001168,-0.00000262),
                        (-0.00001079,-0.00000300), (0.00000510,-0.00000080),
                        (0.00000248,-0.00000150), (-0.0000041,0.00000158),
                        (-0.00000506,0.00000048), (-0.00000571,0.00000518) ])}

    month -= 1

    if latitude < lapserate_lut['lat_split'][month]:
        region_flag = 0
    else:
        region_flag = 1

    lapserate = (lapserate_lut['a0'][month][region_flag]
            + lapserate_lut['a1'][month][region_flag] * latitude
            + lapserate_lut['a2'][month][region_flag] * latitude**2
            + lapserate_lut['a3'][month][region_flag] * latitude**3
            + lapserate_lut['a4'][month][region_flag] * latitude**4)

    return lapserate


def lapserate_moist_adiabate():
    """Moist adiabatic lapse rate in [K/km].
    """
    return 6.5


def get_reflection_angle(cloudmask, metadata):
    """Calculate a sun reflection angle for a given ASTER image depending on
    the sensor-sun geometry and the sensor settings.

    Note:
        All angular values are given in [°].

    Parameters:
        cloudmask (ndarray): binary ASTER cloud mask image.
        metadata (dict): ASTER  HDF meta data information.

    Returns:
        reflection_angle (ndarray): 2d field of size `cloudmask` of reflection
        angles for eachannel pixel.

    References:
         Kang Yang, Huaguo Zhang, Bin Fu, Gang Zheng, Weibing Guan, Aiqin Shi
         & Dongling Li (2015) Observation of submarine sand waves using ASTER
         stereo sun glitter imagery, International Journal of Remote Sensing,
         36:22, 5576-5592, DOI: 10.1080/01431161.2015.1101652
    """

    # Angular data from ASTER metadata data.
    S = float(metadata['MAPORIENTATIONANGLE'])
    P = float(metadata['POINTINGANGLE.1'])
    # SOLARDIRECTION = (0< az <360, -90< el <90)
    sun_azimuth = float(metadata['SOLARDIRECTION'].split(',')[0].strip())
    # sun zenith = 90 - sun elevation
    sun_zenith = 90 - float(metadata['SOLARDIRECTION'].split(',')[1].strip())

    # Instrument view angles of the Visual Near Infrared (VNIR) sensor:
    # Field Of View (FOV)
    FOV_vnir = 6.09
    # Instantaneous FOV (IVOF)
    IFOV = FOV_vnir / cloudmask.shape[1]

    # Construct n-array indexing pixels n=- right and n=+ left from the image
    # central in flight direction.
    n = np.zeros(np.shape(cloudmask))

    for i in range(cloudmask.shape[0]):
        if np.sum(~np.isnan(cloudmask[i,:])) > 0:
            # get index of swath edge pixels and calculate swath mid pixel
            ind1 = next(x[0] for x in
                        enumerate(cloudmask[i]) if ~np.isnan(x[1]) )
            ind2 = (cloudmask.shape[1] - next(x[0] for x in
                    enumerate(cloudmask[i][::-1]) if ~np.isnan(x[1]) ) )
            ind_mid = (ind1 + ind2) / 2
            # Assign n-values correspondingly to left and right sides
            right_arr = np.arange(start=-round(ind_mid), stop=0, step=1,
                                  dtype=int)
            left_arr = np.arange(start=1,
                                 stop=cloudmask.shape[1] - round(ind_mid) + 1,
                                 step=1, dtype=int)
            n[i] = np.asarray(list(right_arr) + list(left_arr))
        else:
            # Put NaN if ONLY NaN values are in cloudmask row.
            n[i] = np.nan

    n[np.isnan(cloudmask)] = np.nan

    # Thresholding index for separating left and right from nadir in
    # azimuth calculation.
    n_az = n * IFOV + P

    # ASTER zenith angle
    ast_zenith = abs(np.asarray(n) * IFOV + P)

    # ASTER azimuth angle
    ast_azimuth = np.ones(cloudmask.shape) * np.nan
    # swath right side
    ast_azimuth[np.logical_and(n_az<0, ~np.isnan(n))] = 90 + S
    # swath left side
    ast_azimuth[np.logical_and(n_az>=0, ~np.isnan(n))] = 270 + S

    # Reflection angle, i.e. the angle between the sensor and the reflected sun
    # in the sensor-sun-plane.
    reflection_angle = theta_r(sun_zenith=sun_zenith,
                               # +180° corresponds to "anti-/reflected" sun
                               sun_azimuth=sun_azimuth + 180,
                               sensor_zenith=ast_zenith,
                               sensor_azimuth=ast_azimuth)

    return reflection_angle


def theta_r(sun_zenith, sun_azimuth, sensor_zenith, sensor_azimuth):
    """Calculate the reflected sun angle, theta_r, of specular reflection
    of sunlight into an instrument sensor.

    Parameters:
        sun_zenith (float): sun zenith angle in [°].
        sun_azimuth (float): sun azimuth angle in [°].
        sensor_zenith (ndarray): 2D sensor zenith angle in [°] for eachannel pixel
                                in the image.
        sensor_azimuth (ndarray): 2D sensor azimuth angle in [°] for eachannel
                                pixel in the image.

    Returns:
        (ndarray): reflection angle in [°] for eachannel pixel in the image.

    References:
         Kang Yang, Huaguo Zhang, Bin Fu, Gang Zheng, Weibing Guan, Aiqin Shi
         & Dongling Li (2015) Observation of submarine sand waves using ASTER
         stereo sun glitter imagery, International Journal of Remote Sensing,
         36:22, 5576-5592, DOI: 10.1080/01431161.2015.1101652
    """
    return np.degrees(np.arccos(np.cos(np.deg2rad(sensor_zenith))
                                * np.cos(np.deg2rad(sun_zenith))
                                + np.sin(np.deg2rad(sensor_zenith))
                                * np.sin(np.deg2rad(sun_zenith))
                                * np.cos( np.deg2rad(sensor_azimuth)
                                - np.deg2rad(sun_azimuth)) ) )
