# -*- coding: utf-8 -*-
"""Functions to work with ASTER L1B satellite data.
"""
import gdal
import numpy as np

from datetime import datetime

__all__ = [
    'cloudmask_ASTER',
    'read_ASTER_channel',
    'convert_ASTERdn2ref',
    'convert_ASTERdn2bt',
]


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

    # Cloud mask including Test 5.
    clmask_Tb14 = clmask.copy()
    clmask_Tb14[Tb14_hres > Tb14_p05] = 3
    clmask_Tb14[np.logical_or(np.isnan(clmask), np.isnan(Tb14_hres))] = np.nan

    # Final binary cloud mask.
    cloudmask = clmask_Tb14.copy()
    cloudmask[cloudmask == 0] = 1
    cloudmask[np.logical_or(cloudmask == 2, cloudmask == 3)] = 0

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