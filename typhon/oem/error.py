# -*- coding: utf-8 -*-
"""Functions to estimate the different sources of retrieval error.
"""

from typhon.oem import common


__all__ = [
    'smoothing_error',
    'retrieval_noise',
]


def smoothing_error(x, x_a, A):
    """Return the smoothing error through the averaging kernel.

    Parameters:
        x (ndarray): Atmospherice profile.
        x_a (ndarray): A priori profile.
        A (ndarray): Averaging kernel matrix.

    Returns:
        ndarray: Smoothing error due to correlation between layers.
    """
    return A @ (x - x_a)


def retrieval_noise(K, S_a, S_y, e_y):
    """Return the retrieval noise.

    Parameters:
        K (np.array): Simulated Jacobians.
        S_a (np.array): A priori error covariance matrix.
        S_y (np.array): Measurement covariance matrix.
        e_y (ndarray): Total measurement error.

    Returns:
        ndarray: Retrieval noise.
    """
    return common.retrieval_gain_matrix(K, S_a, S_y) @ e_y
