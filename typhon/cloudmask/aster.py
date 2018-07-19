# -*- coding: utf-8 -*-
"""Functions to work with ASTER L1B satellite data.
"""
import gdal
import numpy as np

from datetime import datetime
from skimage.measure import block_reduce


__all__ = [
    'cloudmask_ASTER',
    'read_ASTER_channel',
    'convert_ASTERdn2ref',
    'convert_ASTERdn2bt',
    'lapserate_modis',
    'cloudtopheight',
]


def multiple_logical(*args, func=None):
    if func is None:
        func = np.logical_or
    mask = args[0]
    for arg in args[1:]:
        mask = func(mask, arg)
    return mask


def cloudmask_ASTER(filename):
    """Cloud mask derivation from ASTER channels. 
    Four thresholding test based on visual bands distringuish between the 
    dark ocean surface and bright clouds. An additional test five corrects 
    uncertain labeled pixels during broken cloud conditions and pixels with 
    sun glint. A detailed description can be found in Werner et al., 2016.
    Original cloud mask flags:  0 - confidently cloudy
                                1 - probably cloudy
                                2 - probably clear
                                3 - confidently clear
    For the output binary cloud mask flag 0, 1 are merged to 'cloudy - 1' and
    flag 2, 3 are merged to 'clear - 0'.
    
    See also:
        :func:`read_ASTER_channel`: Used to convert ASTER L1B digital
        numbers to physical values, either reflectances in the short wave
        channels, or brightness temperatures from the thermal channels.
        :func:`convert_dn2ref`: Used to convert instrument digital numbers from
        ASTER L1B near-infrared and short-wave channels to reflectance values.
        :func:`convert_dn2bt`: Used to convert instrument digital numbers from
        ASTER L1B thermal channels to brightness temperature values.
    
    Parameters: 
        filename (str): filename of ASTER L1B HDF file. 
    
    Returns:
        cloudmask (ndarray): binary cloud mask (clear - 0, cloudy - 1)
        
    References:
        Werner, F., Wind, G., Zhang, Z., Platnick, S., Di Girolamo, L., 
        Zhao, G., Amarasinghe, N., and Meyer, K.: Marine boundary layer 
        cloud property retrievals from high-resolution ASTER observations: 
        case studies and comparison with Terra MODIS, Atmos. Meas. Tech., 9, 
        5869-5894, https://doi.org/10.5194/amt-9-5869-2016, 2016. 
    """
    # Read band 1.
    r1 = read_ASTER_channel(filename, ch='1')
    
    # Read band 2.
    r2 = read_ASTER_channel(filename, ch='2')
    
    # Read band 3N.
    r3N = read_ASTER_channel(filename, ch='3N')
    
    # Read band 5.
    r5 = read_ASTER_channel(filename, ch='5')
    
    # Remap band 5 reflectances from 30m to 90m resolution.
    r5_hres = np.repeat(np.repeat(r5, 2, axis=0), 2, axis=1)
    
    # Read band 14.
    Tb14 = read_ASTER_channel(filename, ch='14')
    
    # Remap band 14 brightness temperatures at 90m to 15m resolution.
    Tb14_hres = np.repeat(np.repeat(Tb14, 6, axis=0), 6, axis=1)

    # Ratios for clear-cloudy-tests.
    r3N2 = r3N / r2
    r12 = r1 / r2

    # Test 1-4: distinguishing between ocean and cloud pixels
    
    # Set cloudmask to default "confidently clear".
    clmask = np.ones(r1.shape, dtype=np.float) * 3  

    # combined edge values of swath ch1, ch2, ch3N and ch5.
    clmask[np.logical_or(np.isnan(r1), 
                np.logical_or(np.isnan(r2),
                np.logical_or(np.isnan(r3N), np.isnan(r5_hres))))] = np.nan
    
    # Set "probably clear" pixels.
    clmask[np.logical_and(r3N > 0.03, 
                np.logical_and(r5_hres > 0.01,
                np.logical_and(0.7 < r3N2, 
                np.logical_and(r3N2 < 1.75, r12 < 1.45))))] = 2
    
    # Set "probably cloudy" pixels.
    clmask[np.logical_and(r3N > 0.03, 
                np.logical_and(r5_hres > 0.015,
                np.logical_and(0.75 < r3N2, 
                np.logical_and(r3N2 < 1.75, r12 < 1.35))))] = 1
    
    # Set "confidently cloudy" pixels
    clmask[np.logical_and(r3N > 0.065, 
                np.logical_and(r5_hres > 0.02,
                np.logical_and(0.8 < r3N2, 
                np.logical_and(r3N2 < 1.75, r12 < 1.2))))] = 0

    # Test 5: correct broken cloud conditions and pixels with sun glint.
    
    # threshold for Tb14: 5th percentil sampled over all clear pixels.
    Tb14_p05 = np.nanpercentile(Tb14_hres[clmask == 3], 5)
    
    # fraction of clear pixels with flag "3"
    nc = np.sum(clmask == 3) / np.sum(~np.isnan(clmask))
    
    # ADDITIONAL CONDITION (not included in Wernet et al., 2016):
    # if fraction of clear pixels nc < 0.03 use also "probably clear"-pixels 
    # to derive Tb threshold.
    if (nc < 0.03):
        Tb14_p05 = np.nanpercentile(
            Tb14_hres[np.logical_or(clmask == 3, clmask == 2)], 5)

    # Cloud mask including Test 5
    clmask_Tb14 = clmask.copy()
    clmask_Tb14[Tb14_hres > Tb14_p05] = 3
    clmask_Tb14[np.logical_or(np.isnan(clmask), np.isnan(Tb14_hres))] = np.nan

    # Final binary cloud mask
    cloudmask = clmask_Tb14.copy()
    cloudmask[cloudmask == 0] = 1
    cloudmask[np.logical_or(cloudmask == 2, cloudmask == 3)] = 0  # clear

    return cloudmask


def read_ASTER_channel(fname, channel):
    """
    read original ASTER hdf file
    convert digital numbers (DN) to physical variables reflectance (r) or 
    brightness temperature (Tb) depending on the channel selected

    input: fname    e.g. '/path/to/file/AST_L1B_00301012002170346_20160927074451_27963.hdf'
            channel      available aster channels: 1, 2, 3N, 4, 5, 6, 7, 8, 9, 10, 
                    11, 12, 13, 14
    output: var     variable r or Tb with dimension (4200,4980) or (700,830)

    date: Thu Mar 31 2016
    author: Theresa Mieslinger, theresa.mieslinger@mpimet.mpg.de
    """

    # extract date and time from ASTER file name
    im_name = fname.split(sep="/")[-1]
    im_date = datetime(int(im_name[15:19]), int(im_name[11:13]), int(im_name[13:15]),
                       int(im_name[19:21]), int(im_name[21:23]), int(im_name[23:25]))

    # open ASTER image
    g = gdal.Open(fname)
    meta = g.GetMetadata()
    #data_all = g.ReadAsArray()

    # convert DN to r or Tb
    if channel in ('1', '2', '3N', '4', '5', '6', '7', '8', '9'):
        # check metadata if solar elevation angle is >0°, i.e. daytime
        if float(meta['SOLARDIRECTION'].split(',')[1][1:-1]) > 0.:
            # read swath
            if channel in ('1', '2', '3N'):
                swath = gdal.Open('HDF4_EOS:EOS_SWATH:"' +
                                  fname+'":VNIR_Swath:ImageData'+channel)
            elif channel in ('4', '5', '6', '7', '8', '9'):
                swath = gdal.Open('HDF4_EOS:EOS_SWATH:"' +
                                  fname+'":SWIR_Swath:ImageData'+channel)
            data = swath.ReadAsArray()
            data = data.astype('float')
            data[data == 0] = np.nan
            # calculate reflectance r
            var = convert_ASTERdn2ref(dn=data,
                         gain=meta['GAIN.'+channel[0]].split(',')[1].strip(),
                         sun_el=float(
                             meta['SOLARDIRECTION'].split(',')[1].strip()),
                         im_date=im_date,
                         channel=channel,
                         )
        else:
            print('Night: no reflectance is calculated from visible channels')
            return
    elif channel in ('10', '11', '12', '13', '14'):
        # read swath
        swath = gdal.Open('HDF4_EOS:EOS_SWATH:"'+fname +
                          '":TIR_Swath:ImageData'+channel)
        data = swath.ReadAsArray()
        data = data.astype('float')
        data[data == 0] = np.nan
        # calculate brightness temperature Tb
        var = convert_ASTERdn2bt(dn=data, channel=channel)
    else:
        print('Choose one of the available ASTER channels: 1, 2, 3N, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14')

    # close Dataset
    g = None

    return var


def convert_ASTERdn2ref(dn_aster, gain, sun_el, image_date, channel):
    """Converts re-calibrated digital numbers (dn) from ASTER L1B data to 
    spectral radiances [W m-2 sr-1 um-1] and further to reflectance at TOA.
    
    See also: 
        
    Parameters: 
        
    Returns: 
        
    References:
        
    input:  dn          data array of digital numbers
            gain        instrument gain settings
            sun_el      sun elevation angle
            im_date     image date
    output: r           reflectance at TOA

    """
    # Constants / Definitions
    channels = ['1', '2', '3N', '3B',
                '4', '5', '6', '7', '8', '8', '9',
                '10', '11', '12', '13', '14']
    if gain == 'LO1':
        gain = 'LOW'
    gain_list = ['HGH', 'NOR', 'LOW', 'LO2']

    # unit conversion coefficients ucc [W m-2 sr-1 um-1] depending on gain settings
    # source: ASTER user handbook, chapter 5.0 ASTER Radiometry (p.25)
    # syntax: ucc_table={'ch':(ucc_gain_high, ucc_gain_normal, ucc_gain_low1, ucc_gain_low2)}
    ucc_table = {'1': (0.676, 1.688, 2.25, np.nan),
                 '2': (0.708, 1.415, 1.89, np.nan),
                 '3N': (0.423, 0.862, 1.15, np.nan),
                 '3B': (0.423, 0.862, 1.15, np.nan),
                 '4': (0.1087, 0.2174, 0.290, 0.290),
                 '5': (0.0348, 0.0696, 0.0925, 0.409),
                 '6': (0.0313, 0.0625, 0.0830, 0.390),
                 '7': (0.0299, 0.0597, 0.0795, 0.332),
                 '8': (0.0209, 0.0417, 0.0556, 0.245),
                 '9': (0.0159, 0.0318, 0.0424, 0.265),
                 '10': (np.nan, 6.822e-3, np.nan),
                 '11': (np.nan, 6.780e-3, np.nan)}

    # Mean solar exoatmospheric irradiances[W m-2 um-1] at TOA according to Thome et al.
    # Calculated using spectral irradiance values dervied with MODTRAN
    # literature: http://www.pancroma.com/downloads/ASTER%20Temperature%20and%20Reflectance.pdf
    E_sun = (1848, 1549, 1114,
             1114, 225.4, 86.63,
             81.85, 74.85, 66.49,
             59.85, np.nan, np.nan,
             np.nan, np.nan, np.nan)

    # solar zenith angle theta_z [°]
    theta_z = 90 - sun_el

    # day of year (DOY) ASTER scene
    doy = image_date.timetuple().tm_yday
    
    # sun-earth-distance d for corresponding day of the year (doy)
    # source: http://landsathandbook.gsfc.nasa.gov/data_prod/prog_sect11_3.html
    # fitting sinus curve to values results in the following parameters: 
    # distance = A * np.sin(w*doy + p) + c [AU]
    distance = (0.016790956760352183 
                * np.sin(-0.017024566555637135 * doy
                         + 4.735251041365579) 
                + 0.9999651354429786)

    # radiance values in [W m-2 sr-1 um-1] at TOA
    radiance = (dn_aster - 1) * ucc_table[channel][gain_list.index(gain)]

    return (np.pi * radiance * distance**2 /
        (E_sun[channels.index(channel)] * np.cos(np.radians(theta_z))))


def convert_ASTERdn2bt(dn, channel):
    """
    ASTER L1B data gives re-calibrated digital numbers (DN). This function converts DN to spectral radiances [W m-2 sr-1 um-1] 
    and additionally calculate the brightness temperature from channel 14. 
    source: http://landsathandbook.gsfc.nasa.gov/data_prod/prog_sect11_3.html 
        input:  dn      data array of digital numbers  
        output: Tb      brightness temperature

    date: Thu Mar 31 14:40:45 2016
    author: Theresa Mieslinger, theresa.mieslinger@mpimet.mpg.de
    """
    # unit conversion coefficients ucc [W m-2 sr-1 um-1]
    # https://lpdaac.usgs.gov/dataset_discovery/aster/aster_products_table/ast_l1t
    ucc = {'10': 6.882e-3,
           '11': 6.780e-3,
           '12': 6.590e-3,
           '13': 5.693e-3,
           '14': 5.225e-3}

    K1 = {'10': 3040.136402,
          '11': 2482.375199,
          '12': 1935.060183,
          '13': 866.468575,
          '14': 641.326517}  # [W m-2 um-1]

    K2 = {'10': 1735.337945,
          '11': 1666.398761,
          '12': 1585.420044,
          '13': 1350.069147,
          '14': 1271.221673}  # [K]

    # calculation of radiance values rad [W m-2 sr-1 um-1]
    rad = (dn - 1) * ucc[channel]

    # calculation of brightness temperature Tb [K] from inverse Planck function
    Tb = K2[channel] / np.log((K1[channel] / rad) + 1)

    return Tb


def lapserate_modis(month, latitude):
    """Estimate of the apparent lapse rate in [K/km].
    Typical lapse rates are assumed for each month and depending on the
    location on Earth, i.e. southern hemisphere, tropics, or northern
    hemisphere. For a specific case the lapse rate is estimated by a 4th order
    polynomial and polynomial coefficients given in the look-up table.
    This approach is based on the MODIS cloud top height retrieval and applies
    to data recorded at 11 microns.
    
    Note:
        Southern hemisphere is not yet included!
    
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
    
    
def cloudtopheight_IR(cloudmask, Tb14, method='simple', latitudes=None,
                      month=None):
    """Cloud Top Height (CTH) from 11 micron channel. 
    ASTER brightness temperatures from channel 14 are converted to CTHs using 
    the IR window approach: (Tb_clear - Tb_cloudy) / lapse_rate. 
    
    Note:
        `cloudmask` argument has to have the same resolution as the thermal
        channel, i.e. (700,830) at 90 m resolution.
    
    See also:
        :func:`block_reduce`:
        :func:`lapserate_modis`
        
    Parameters:
        cloudmask (ndarray): binary ASTER cloud mask image.
        Tb14 (ndarray): brightness temperatures form ASTER channel 14.
        method (str): approach used to derive CTH.
        latitudes (ndarray):
        month (int): moneth of the year.
    
    Returns:
        ndarray: cloud top height.
    """    
    # Lapse rate
    if method=='simple':
        lapserate = lapserate_moist_adiabate()
        
    elif method=='modis':
        latitude = latitudes[np.shape(latitudes)[0] // 2,
                             np.shape(latitudes)[1] // 2] #[2100, 2490]
        lapserate = lapserate_modis(month=month, latitude=latitude)
    
    # 
    #Tb_clear_avg = np.nanmean(Tb14[np.asarray(cloudmask, dtype=bool) == False])
    #Tb_cloudy = Tb14[np.asarray(cloudmask, dtype=bool) == True]
    #cloudmask[np.isnan(cloudmask)] = 0
    
    resolution_ratio = np.shape(cloudmask)[0] // np.shape(Tb14)[0]

    cloudmask_inverted = cloudmask.copy()
    cloudmask_inverted[np.isnan(cloudmask_inverted)] = 1
    cloudmask_inverted = np.asarray(np.invert(np.asarray(
                                cloudmask_inverted, dtype=bool)), dtype=int)
    
    cloudmask[np.isnan(cloudmask)] = 0
    cloudmask = np.asarray(cloudmask, dtype=int)
    
    # match cloudmask resolution and Tb14 resolution.
    if resolution_ratio > 1:
        # on Tb14 resolution, flag pixels as cloudy only if all subgrid pixels
        # are cloudy in the original cloud mask.
        mask_cloudy = block_reduce(cloudmask,
                                        (resolution_ratio, resolution_ratio),
                                        func=np.alltrue)
        # search for 90 m only clear pixels to derive a Tb clearsky/ocean value
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
            print('Problems matching the shapes of cloudmask and Tb14.')
    else:
        mask_cloudy = cloudmask.copy()
        mask_clear = cloudmask_inverted.copy()
        
    Tb_cloudy = np.ones(np.shape(Tb14)) * np.nan
    Tb_cloudy[mask_cloudy] = Tb14[mask_cloudy]
    
    Tb_clear_avg = np.nanmean(Tb14[mask_clear])

    return (Tb_clear_avg - Tb_cloudy) / lapserate