"""
This module includes general purpose math functions.  This module was developed before
scipy was included as a prerequisite, so there will be some functions remaining that
duplicate scipy functionality. 
"""
import numpy as np
import scipy as sp


def integrate_phasemat(angles, Z):
    """Integrates the phase-matrix over all angles

    Parameters
    ~~~~~~~~~~

    angles : 1D-array
        Angles to integrate over [degrees]

    Z : nD-array
        quantity to integrate. The last dimension must match the size of
        angles.

    Returns
    ~~~~~~~

    Z_int : (n-1)D-array
        Array with integrated Z. Has all but the last dimension of Z.
    """

    angles = np.deg2rad(angles)
    integrand = 2 * np.pi * Z * np.sin(angles)
    return sp.integrate.trapz(integrand, angles)
