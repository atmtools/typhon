# -*- coding: utf-8 -*-

"""General functions for manipulating geographical data.
"""
import numpy as np


__all__ = [
    'area_weighted_mean',
        ]


def area_weighted_mean(lon, lat, data):
    """Calculate the mean of gridded data on a sphere.

    Data points on the Earth's surface are often represented as a grid. As the
    grid cells do not have a constant area they have to be weighted when
    calculating statistical properties (e.g. mean).

    This function returns the weighted mean assuming a perfectly spherical
    globe.

    Parameters:
        lon (ndarray): Longitude (M) angles [degree].
        lat (ndarray): Latitude (N) angles [degree].
        data ()ndarray): Data array (N x M).

    Returns:
        float: Area weighted mean.

    """
    # Calculate coordinates and steradian (in rad).
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    dlon = np.diff(lon)
    dlat = np.diff(lat)

    # Longitudal mean
    middle_points = (data[:, 1:] + data[:, :-1]) / 2
    norm = np.sum(dlon)
    lon_integral = np.sum(middle_points * dlon, axis=1) / norm

    # Latitudal mean
    lon_integral *= np.cos(lat)  # Consider varying grid area (N-S).
    middle_points = (lon_integral[1:] + lon_integral[:-1]) / 2
    norm = np.sum(np.cos((lat[1:] + lat[:-1]) / 2) * dlat)

    return np.sum(middle_points * dlat) / norm
