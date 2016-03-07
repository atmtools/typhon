# -*- coding: utf-8 -*-
"""
Collection of functions concerning the Optimal Estimation Method (OEM).

"""

import numpy as np
from scipy.linalg import inv


__all__ =  ['error_covariance_matrix',
            'averaging_kernel_matrix',
           ]

def error_covariance_matrix(K, S_a, S_y):
    """Calculate the error covariance matrix.

    Parameters:
        K (np.array): Simulated Jacobians.
        S_a (np.array): A priori error covariance matrix.
        S_y (np.array): Measurement covariance matrix.

    Returns:
        np.array: Measurement error covariance matrix.
    """
    return inv(K.T @ inv(S_y) @ K + inv(S_a))

def averaging_kernel_matrix(K, S_a, S_y):
    """Calculate the averaging kernel matrix.

    Parameters:
        K (np.array): Simulated Jacobians.
        S_a (np.array): A priori error covariance matrix.
        S_y (np.array): Measurement covariance matrix.

    Returns:
        np.array: Averaging kernel matrix.
    """
    return inv(inv(S_a) + K.T @ inv(S_y) @ K) @ K.T @ inv(S_y) @ K
