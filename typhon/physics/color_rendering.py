# -*- coding: utf-8 -*-

"""Functions related to color rendering of spectral data in the visible range
"""

import numpy as np
from scipy.interpolate import interp1d
from typhon.constants import c

__all__ = [
    "cie_color_matching_kernels",
    "xyz2rgb_cv2",
    "xyz2rgb_walker_hdtv",
    "convert_xyz2rgb",
    "match_color",
]


def cie_color_matching_kernels(wavelength):
    """
    Calculates CIE color matching kernels for desired wavelengths
    between 380nm and 780nm. Outside of these range values are nan.

    Parameters:
        wavelength (ndarray): wavelength in m.

    Returns:
        ndarray: color matching kernels (dim:len(wavelengths)x3).

     References:
        https://scipython.com/static/media/blog/colours/cie-cmf.txt
        https://www.fourmilab.ch/documents/specrend/specrend.c


    """

    data = np.array(
        [
            [1.4000e-03, 0.0000e00, 6.5000e-03],
            [2.2000e-03, 1.0000e-04, 1.0500e-02],
            [4.2000e-03, 1.0000e-04, 2.0100e-02],
            [7.6000e-03, 2.0000e-04, 3.6200e-02],
            [1.4300e-02, 4.0000e-04, 6.7900e-02],
            [2.3200e-02, 6.0000e-04, 1.1020e-01],
            [4.3500e-02, 1.2000e-03, 2.0740e-01],
            [7.7600e-02, 2.2000e-03, 3.7130e-01],
            [1.3440e-01, 4.0000e-03, 6.4560e-01],
            [2.1480e-01, 7.3000e-03, 1.0391e00],
            [2.8390e-01, 1.1600e-02, 1.3856e00],
            [3.2850e-01, 1.6800e-02, 1.6230e00],
            [3.4830e-01, 2.3000e-02, 1.7471e00],
            [3.4810e-01, 2.9800e-02, 1.7826e00],
            [3.3620e-01, 3.8000e-02, 1.7721e00],
            [3.1870e-01, 4.8000e-02, 1.7441e00],
            [2.9080e-01, 6.0000e-02, 1.6692e00],
            [2.5110e-01, 7.3900e-02, 1.5281e00],
            [1.9540e-01, 9.1000e-02, 1.2876e00],
            [1.4210e-01, 1.1260e-01, 1.0419e00],
            [9.5600e-02, 1.3900e-01, 8.1300e-01],
            [5.8000e-02, 1.6930e-01, 6.1620e-01],
            [3.2000e-02, 2.0800e-01, 4.6520e-01],
            [1.4700e-02, 2.5860e-01, 3.5330e-01],
            [4.9000e-03, 3.2300e-01, 2.7200e-01],
            [2.4000e-03, 4.0730e-01, 2.1230e-01],
            [9.3000e-03, 5.0300e-01, 1.5820e-01],
            [2.9100e-02, 6.0820e-01, 1.1170e-01],
            [6.3300e-02, 7.1000e-01, 7.8200e-02],
            [1.0960e-01, 7.9320e-01, 5.7300e-02],
            [1.6550e-01, 8.6200e-01, 4.2200e-02],
            [2.2570e-01, 9.1490e-01, 2.9800e-02],
            [2.9040e-01, 9.5400e-01, 2.0300e-02],
            [3.5970e-01, 9.8030e-01, 1.3400e-02],
            [4.3340e-01, 9.9500e-01, 8.7000e-03],
            [5.1210e-01, 1.0000e00, 5.7000e-03],
            [5.9450e-01, 9.9500e-01, 3.9000e-03],
            [6.7840e-01, 9.7860e-01, 2.7000e-03],
            [7.6210e-01, 9.5200e-01, 2.1000e-03],
            [8.4250e-01, 9.1540e-01, 1.8000e-03],
            [9.1630e-01, 8.7000e-01, 1.7000e-03],
            [9.7860e-01, 8.1630e-01, 1.4000e-03],
            [1.0263e00, 7.5700e-01, 1.1000e-03],
            [1.0567e00, 6.9490e-01, 1.0000e-03],
            [1.0622e00, 6.3100e-01, 8.0000e-04],
            [1.0456e00, 5.6680e-01, 6.0000e-04],
            [1.0026e00, 5.0300e-01, 3.0000e-04],
            [9.3840e-01, 4.4120e-01, 2.0000e-04],
            [8.5440e-01, 3.8100e-01, 2.0000e-04],
            [7.5140e-01, 3.2100e-01, 1.0000e-04],
            [6.4240e-01, 2.6500e-01, 0.0000e00],
            [5.4190e-01, 2.1700e-01, 0.0000e00],
            [4.4790e-01, 1.7500e-01, 0.0000e00],
            [3.6080e-01, 1.3820e-01, 0.0000e00],
            [2.8350e-01, 1.0700e-01, 0.0000e00],
            [2.1870e-01, 8.1600e-02, 0.0000e00],
            [1.6490e-01, 6.1000e-02, 0.0000e00],
            [1.2120e-01, 4.4600e-02, 0.0000e00],
            [8.7400e-02, 3.2000e-02, 0.0000e00],
            [6.3600e-02, 2.3200e-02, 0.0000e00],
            [4.6800e-02, 1.7000e-02, 0.0000e00],
            [3.2900e-02, 1.1900e-02, 0.0000e00],
            [2.2700e-02, 8.2000e-03, 0.0000e00],
            [1.5800e-02, 5.7000e-03, 0.0000e00],
            [1.1400e-02, 4.1000e-03, 0.0000e00],
            [8.1000e-03, 2.9000e-03, 0.0000e00],
            [5.8000e-03, 2.1000e-03, 0.0000e00],
            [4.1000e-03, 1.5000e-03, 0.0000e00],
            [2.9000e-03, 1.0000e-03, 0.0000e00],
            [2.0000e-03, 7.0000e-04, 0.0000e00],
            [1.4000e-03, 5.0000e-04, 0.0000e00],
            [1.0000e-03, 4.0000e-04, 0.0000e00],
            [7.0000e-04, 2.0000e-04, 0.0000e00],
            [5.0000e-04, 2.0000e-04, 0.0000e00],
            [3.0000e-04, 1.0000e-04, 0.0000e00],
            [2.0000e-04, 1.0000e-04, 0.0000e00],
            [2.0000e-04, 1.0000e-04, 0.0000e00],
            [1.0000e-04, 0.0000e00, 0.0000e00],
            [1.0000e-04, 0.0000e00, 0.0000e00],
            [1.0000e-04, 0.0000e00, 0.0000e00],
            [0.0000e00, 0.0000e00, 0.0000e00],
        ]
    )

    wavelength_data = (
        np.array(
            [
                380.0,
                385.0,
                390.0,
                395.0,
                400.0,
                405.0,
                410.0,
                415.0,
                420.0,
                425.0,
                430.0,
                435.0,
                440.0,
                445.0,
                450.0,
                455.0,
                460.0,
                465.0,
                470.0,
                475.0,
                480.0,
                485.0,
                490.0,
                495.0,
                500.0,
                505.0,
                510.0,
                515.0,
                520.0,
                525.0,
                530.0,
                535.0,
                540.0,
                545.0,
                550.0,
                555.0,
                560.0,
                565.0,
                570.0,
                575.0,
                580.0,
                585.0,
                590.0,
                595.0,
                600.0,
                605.0,
                610.0,
                615.0,
                620.0,
                625.0,
                630.0,
                635.0,
                640.0,
                645.0,
                650.0,
                655.0,
                660.0,
                665.0,
                670.0,
                675.0,
                680.0,
                685.0,
                690.0,
                695.0,
                700.0,
                705.0,
                710.0,
                715.0,
                720.0,
                725.0,
                730.0,
                735.0,
                740.0,
                745.0,
                750.0,
                755.0,
                760.0,
                765.0,
                770.0,
                775.0,
                780.0,
            ]
        )
        * 1e-9
    )

    f_int = interp1d(wavelength_data, data, kind="linear", axis=0, bounds_error=False)

    kernels = f_int(wavelength)

    return kernels


def xyz2rgb_cv2():
    """
    Create transformation matrix from xyz to RGB
    Values taken from reverse engineering open cv's
    cv2.cvtColor(I, cv2.COLOR_XYZ2RGB).

    Returns:
        m_xyz2rgb (2darray): transformation matrix (3x3).

    References:
        https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_xyz
    """

    m_xyz2rgb = np.array(
        [
            [3.24047899, -1.53715003, -0.49853501],
            [-0.96925598, 1.87599099, 0.041556],
            [0.055648, -0.204043, 1.05731106],
        ]
    )

    return m_xyz2rgb


def xyz2rgb_walker_hdtv():
    """
    Create transformation matrix from xyz to HDTV RGB according to

    John Walker "Colour Rendering of Spectra ".


    Returns:
        m_xyz2rgb (2darray): transformation matrix (3x3).

    References:
        https://www.fourmilab.ch/documents/specrend/specrend.c


    """

    CS = {}
    CS["xRed"] = 0.670
    CS["yRed"] = 0.330
    CS["xGreen"] = 0.210
    CS["yGreen"] = 0.710
    CS["xBlue"] = 0.150
    CS["yBlue"] = 0.060
    CS["xWhite"] = 0.3127
    CS["yWhite"] = 0.3291

    xr = CS["xRed"]
    xg = CS["xGreen"]
    xb = CS["xBlue"]

    yr = CS["yRed"]
    yg = CS["yGreen"]
    yb = CS["yBlue"]

    zr = 1 - (xr + yr)
    zg = 1 - (xg + yg)
    zb = 1 - (xb + yb)

    xw = CS["xWhite"]
    yw = CS["yWhite"]
    zw = 1 - (xw + yw)

    # xyz -> rgb matrix, before scaling to white.
    rx = (yg * zb) - (yb * zg)
    gx = (yb * zr) - (yr * zb)
    bx = (yr * zg) - (yg * zr)

    ry = (xb * zg) - (xg * zb)
    gy = (xr * zb) - (xb * zr)
    by = (xg * zr) - (xr * zg)

    rz = (xg * yb) - (xb * yg)
    gz = (xb * yr) - (xr * yb)
    bz = (xr * yg) - (xg * yr)

    # White scaling factors.
    # Dividing by yw scales the white luminance to unity, as conventional. */
    rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw
    gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw
    bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw

    # xyz -> rgb matrix, correctly scaled to white.
    rx = rx / rw
    gx = gx / gw
    bx = bx / bw

    ry = ry / rw
    gy = gy / gw
    by = by / bw

    rz = rz / rw
    gz = gz / gw
    bz = bz / bw

    m_xyz2rgb = np.array([[rx, ry, rz], [gx, gy, gz], [bx, by, bz]])

    return m_xyz2rgb


def convert_xyz2rgb(xyz, xyz2rgb="opencv"):
    """
    Convert image in xyz color system to image in rgb color system.

    Parameters:
        xyz (3darray): Image in xyz color system [height,width, xyz]
        xyz2rgb (str, optional):
            Selects the transformation matrix. There
            are two possible matrices.

            - "opencv": according to the open cv package.
            - "hdtv": according to John Walker "Colour
              Rendering of Spectra" for HDTV.

            Defaults to "opencv".
    Returns:
        rgb (3darray): Image in rgb color system [height,width, rgb].

    References:
        https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_xyz
        https://www.fourmilab.ch/documents/specrend/specrend.c

    """

    if xyz2rgb == "opencv":
        m_xyz2rgb = xyz2rgb_cv2()
    elif xyz2rgb == "hdtv":
        m_xyz2rgb = xyz2rgb_walker_hdtv()

    rgb = np.zeros(np.shape(xyz))
    for i in range(np.size(rgb, axis=2)):
        for j in range(np.size(rgb, axis=2)):
            rgb[:, :, i] += m_xyz2rgb[i, j] * xyz[:, :, j]

    return rgb


def match_color(spc_image, f_grid, normalization_constant=2.5, gamma=1 / 2.2):
    """
    Transforms spectral image to RGB image by integrating the spectrum of each
    pixel with the CIE color matching kernels. The resulting image in xyz color system
    is then transformed to the rgb system.

    For details see,
    John Walker "Colour Rendering of Spectra"
    https://www.fourmilab.ch/documents/specrend/


    Parameters:
        spc_image (3darray): Spectral image [height,width, frequency] in [W/(m^2 sr Hz)].
        f_grid (array): frequency in ascending order [Hz].
        normalization_constant (float, optional): Luminosity normalization constant.
                                                  Defaults to 2.5 shows reasonable
                                                  results. Greater value lead to
                                                  darker images and vice versa.
        gamma (float, optional): Gamma correction. Defaults to 1/2.2.

    Returns:
        image (ndarray): RGB image [height,width,color].
        Luminosity (ndarray): Luminosity [height,width].

    """

    # Convert image from frequency units to wavelength units
    i_image = spc_image * f_grid[np.newaxis, np.newaxis, :] ** 2 / c

    wavelength = c / f_grid

    # get CIE kernels
    cie_kernels = cie_color_matching_kernels(wavelength)

    # Convert radiance to human vision perceived color in XYZ space
    XYZ = np.zeros((np.shape(i_image)[0:-1] + (3,)))

    for color_idx in range(np.size(cie_kernels, axis=1)):
        cie_temp = cie_kernels[:, color_idx]
        int_kernel = i_image * cie_temp[np.newaxis, np.newaxis, :]
        XYZ[:, :, color_idx] = -np.trapz(int_kernel, wavelength, axis=2)

    # normalized luminosity
    normed_luminosity = XYZ[:, :, 1] / normalization_constant

    # normalize XYZ
    norm_xyz = np.sum(XYZ, axis=2)
    xyz = np.zeros(np.shape(XYZ))
    for i in range(np.size(xyz, axis=2)):
        temp = XYZ[:, :, i] * 1.0
        logic = temp > 0
        temp[logic] = temp[logic] / norm_xyz[logic]
        xyz[:, :, i] = temp

    # Transform to RGB space
    image = convert_xyz2rgb(xyz)

    # Remove negatives by adding "white"
    w = image.min(axis=2)
    w[w > 0] = 0
    image += w[:, :, np.newaxis]

    image = image * normed_luminosity[:, :, np.newaxis]
    image[image > 1] = 1
    image[image < 0] = 0
    image = image ** (1 / 2.2)

    return image, XYZ[:, :, 1]
