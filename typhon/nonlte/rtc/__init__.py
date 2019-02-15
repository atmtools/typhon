
import numba

import numpy as np
from ..spectra.source_function import Bv_T, PopuSource_AB
from ..spectra.abscoeff import basic

@numba.jit
def FOSC(tau, Sb, Sm, Ib):
    """ First Order Short Characteristics 

    See "Theory of stellar atmospheres: An introduction to astrophysical 
    non-equilibrium quantitative spectroscopic analysis" by Ivan Hubeny
    and Dimitri Mihalas, ISBN 978-0-691-16328-4

    Parameters:
        tau: optical depth between two layers 
        Sb: Source function at adjacent grid points 
        Sm: Source function at target point 
        Ib: Intensity adjacent grid points

    Returns: 
        Inwards-directed intensity, lambda used in equation
    """
    yd = tau - 1. + np.exp(-tau)  # (12.120)

    dev_cond = tau * 1.
    dev_cond[tau == 0] = np.nan

    lambda_m = yd/dev_cond  # (12.117)
    lambda_b = - (yd / dev_cond) + 1. - np.exp(-tau)  # (12.118, 116)
    Im = Ib * np.exp(-tau) + lambda_m * Sm + lambda_b * Sb  # (12.114)
    Im[tau == 0] = Ib[tau == 0]
    return Im, lambda_m


def SOSC(tau1, tau3, S1, S2, S3, I1):
    """
    Calculate intensity by second order short characteristic method
    written in Kunasz and Auer, 1988, J. Quanr Spectrosc. Radiar Transfer

    Parameters:
        tau1: optical depth at grid 1 
        tau3: optical depth at grid 3
        S1: Source function at grid 1
        S2: Source function at grid 2
        S3: Source function at grid 3
        I1: Intensity at grid 1

    Returns: 
        intensity at grid 2, lambda used in equation
    """
    lambda_1 = np.empty_like(tau1)
    lambda_2 = np.empty_like(tau1)
    lambda_3 = np.empty_like(tau1)

    tau1[tau1 != tau1] = 0
    tau3[tau3 != tau3] = 0

    linear = np.logical_or(tau1 < 1.e-4,
                           tau3 < 1.e-10)  # Computational error region
    exp_tau = np.exp(-tau1)
    w0 = 1 - exp_tau

    lambda_1[linear] = w0[linear]
    lambda_2[linear] = 0
    lambda_3[linear] = 0

    linear = linear == False  # Second order calculation for not too small tau

    w1 = tau1[linear] - w0[linear]
    w2 = tau1[linear]**2 - 2 * w1  # w2 < 1.e-16 is 0
    lambda_1[linear] = w0[linear] + (w2 - (tau3[linear] + 2 * tau1[linear])
                                     * w1) / (tau1[linear] *
                                              (tau1[linear] + tau3[linear]))
    lambda_2[linear] = (w1 * (tau1[linear] + tau3[linear])
                        - w2) / (tau1[linear] * tau3[linear])
    lambda_3[linear] = (w2 - w1 * tau1[linear]) / \
                       (tau3[linear] * (tau1[linear] + tau3[linear]))

    source_function = lambda_3 * S3 + lambda_2 * S2 + lambda_1 * S1

    I2 = I1 * exp_tau + source_function

    # return I2, lambda_2+I1 * exp_tau
    return I2, lambda_2 + lambda_1 * exp_tau


def SOSCdamy(tau, tau3, Sb, Sm, S3, Ib):
    """
    Calculate intensity by second order short characteristic method
    written in Kunasz and Auer, 1988, J. Quanr Spectrosc. Radiar Transfer

    Parameters:
        tau1: optical depth at grid 1
        tau3: optical depth at grid 3
        S1: Source function at grid 1
        S2: Source function at grid 2
        S3: Source function at grid 3
        I1: Intensity at grid 1

    Returns:
        intensity at grid 2, lambda used in equation
    """
    yd = tau - 1. + np.exp(-tau)  # (12.120)

    dev_cond = tau * 1.

    lambda_m = yd/dev_cond  # (12.117)
    lambda_b = - (yd / dev_cond) + 1. - np.exp(-tau)  # (12.118, 116)
    Im = Ib * np.exp(-tau) + lambda_m * Sm + lambda_b * Sb  # (12.114)
    return Im, lambda_m
