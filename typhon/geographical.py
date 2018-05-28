# -*- coding: utf-8 -*-

"""General functions for manipulating geographical data.
"""
from numbers import Number

import numpy as np
from sklearn.neighbors import BallTree, KDTree
from typhon.constants import earth_radius
from typhon.geodesy import geocentric2cart
from typhon.utils import split_units


__all__ = [
    'area_weighted_mean',
    'GeoIndex',
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


# Factor to convert a length unit to kilometers
UNITS_CONVERSION_FACTORS = [
    [{"cm", "centimeter", "centimeters"}, 1e-6],
    [{"m", "meter", "meters"}, 1e-3],
    [{"km", "kilometer", "kilometers"}, 1],
    [{"mi", "mile", "miles"}, 1.609344],  # english statute mile
    [{"yd", "yds", "yard", "yards"}, 0.9144e-3],
    [{"ft", "foot", "feet"}, 0.3048e-3],
]


def to_kilometers(distance):
    """Convert different length units to kilometers

    Args:
        distance: A string or number.

    Returns:
        A distance as float in kilometers
    """
    if isinstance(distance, Number):
        return distance
    elif not isinstance(distance, str):
        raise ValueError("Distance must be a number or a string!")

    length, unit = split_units(distance)

    if length == 0:
        raise ValueError("A valid distance length must be given!")

    if not unit:
        return length

    for units, factor in UNITS_CONVERSION_FACTORS:
        if unit in units:
            return length * factor

    raise ValueError(f"Unknown distance unit: {unit}!")


class GeoIndex:

    def __init__(self, lat, lon, metric=None, tree_class=None,
                 shuffle=True, **tree_kwargs):
        """
        
        Args:
            lat: 
            lon: 
            metric: We can use different metrics for the indexing. The default
                *minkowski* metric is the fastest but produces a big error for
                large distances. The *haversine* is more correct but is slower
                to compute. Rule of thumb: When searching for radii over
                1000km, the error is ca. 1.4 kilometers.
            tree_class: Say what tree do you want to use for building the
                spatial index. Either *Ball* or *KD* is allowed. Default is
                *Ball*.
            shuffle: The trees have a terrible performance for sorted data
                as discussed in this issue:
                https://github.com/scikit-learn/scikit-learn/issues/7687. For
                sorted, almost-gridded data (such as from SEVIRI) you should
                set this to *True*. Default is False.
        """
        if metric is None:
            self.metric = "minkowski"
        else:
            self.metric = metric

        # Convert the lat and lon to the chosen metric:
        points = self._to_metric(lat, lon)

        # Save the latitudes and longitudes:
        self.lat = lat
        self.lon = lon

        if tree_class is None or tree_class == "Ball":
            tree_class = BallTree
        elif tree_class == "KD":
            tree_class = KDTree

        # KD- or ball trees have a very poor building performance for sorted
        # data (such as from SEVIRI) as discussed in this issue:
        # https://github.com/scikit-learn/scikit-learn/issues/7687
        # Hence, we shuffle the data points before inserting them.
        if shuffle:
            self.shuffler = np.arange(points.shape[0])
            np.random.shuffle(self.shuffler)

            points = points[self.shuffler]
        else:
            # The user does not want to shuffle
            self.shuffler = None

        self.tree = tree_class(
            points, **{**tree_kwargs, "metric": self.metric}
        )

    def __getitem__(self, points):
        """Get nearest indices for given points.

        Args:
            points:

        Returns:

        """
        pass

    def _to_metric(self, lat, lon):
        if not (isinstance(lat, np.ndarray) or isinstance(lon, np.ndarray)):
            raise ValueError("lat and lon must be numpy.ndarray objects (no "
                             "pandas.Series or xarray.DataArray)!")

        if self.metric == "minkowski":
            return np.column_stack(
                geocentric2cart(earth_radius, lat, lon)
            )
        elif self.metric == "haversine":
            return np.radians(
                np.column_stack([lat, lon])
            )
        else:
            raise ValueError(f"Unknown metric '{self.metric}!'")

    def query(self, lat, lon, r):
        """Find all neighbours within a radius of query points

        Args:
            lat:
            lon:
            r: Radius in kilometers (if number is given). You can also use
                another unit if you pass this as string (e.g. *'10 miles'*).

        Returns:
            Two numpy arrays. The first has a *2xN* shape, where N is the
            number of found collocations. In the first
        """
        points = self._to_metric(lat, lon)

        # The user passes the radius in kilometers but we calculate with meters
        # internally
        r = to_kilometers(r)
        if self.metric == "minkowski":
            r *= 1000.
        elif self.metric == "haversine":
            r *= 1000. / earth_radius

        jagged_pairs, jagged_distances = \
            self.tree.query_radius(points, r, return_distance=True)

        # Build the list of the collocation pairs:
        pairs = np.array([
            [build_point, query_point]
            for query_point, build_points in enumerate(jagged_pairs)
            for build_point in build_points
        ]).T

        if not pairs.any():
            return pairs, pairs

        distances = np.hstack([
            distances_to_query
            for distances_to_query in jagged_distances
        ])

        # Return the distances in kilometers
        distances /= 1000.

        if self.shuffler is None:
            return pairs, distances
        else:

            # We shuffled the build points in the beginning, so the current
            # indices in the second row (the collocation indices from the build
            # points) are not correct
            pairs[0, :] = self.shuffler[pairs[0, :]]

            return pairs, distances

