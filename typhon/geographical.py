# -*- coding: utf-8 -*-

"""General functions for manipulating geographical data.
"""
from numbers import Number

import imageio
import numpy as np
from sklearn.neighbors import BallTree, KDTree
from typhon.constants import earth_radius
from typhon.geodesy import geocentric2cart
from typhon.utils import split_units


__all__ = [
    'area_weighted_mean',
    'GeoIndex',
    'gridded_mean',
    'sea_mask'
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
    """Indexer that allows fast range queries with geographical coordinates"""

    def __init__(self, lat, lon, metric=None, tree_class=None,
                 shuffle=True, **tree_kwargs):
        """Initialize a GeoIndex
        
        Args:
            lat: Latitudes between -90 and 90 degrees as 1-dimensional numpy
                array.
            lon: Longitudes between -180 and 180 degrees as 1-dimensional numpy
                array. Must have the same length as `lat`.
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
                set this to *True*. Default is True.

        Examples:

            We have two tracks (e.g. of satellites) and we want to know whether
            and where they have been in a close distance.

            .. code-block:: python

                import matplotlib.pyplot as plt
                import numpy as np
                from typhon.geographical import GeoIndex
                from typhon.plots import worldmap

                track1 = {
                    "lat": 90*np.sin(np.linspace(-2*np.pi, 2*np.pi, 90)),
                    "lon": np.linspace(-180, 180, 90),
                }
                track2 = {
                    "lat": 90*np.cos(np.linspace(-2*np.pi, 2*np.pi, 90)),
                    "lon": np.linspace(-180, 180, 90),
                }

                # Build the index with the data of the first track:
                index = GeoIndex(track1["lat"], track1["lon"])

                # Find all points from the second track that are within a
                # radius of 500 miles to the first track
                pairs, distances = index.query(
                    track2["lat"], track2["lon"], r="500 miles"
                )

                # Plot the points
                worldmap(
                    track1["lat"], track1["lon"],
                    s=10, bg=True, label="Track 1")
                worldmap(
                    track2["lat"], track2["lon"], s=10, label="Track 2")
                worldmap(
                    track1["lat"][pairs[0]], track1["lon"][pairs[0]],
                    s=10, label="Track 1 (matched)"
                )
                worldmap(
                    track2["lat"][pairs[1]], track2["lon"][pairs[1]],
                    s=10, label="Track 2 (matched)"
                )
                plt.legend()
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

        Warnings:
            Not yet implemented!

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

    def query(self, lat, lon, r, return_distance=True):
        """Find all neighbours within a radius of query points

        Args:
            lat: Latitudes between -90 and 90 degrees as 1-dimensional numpy
                array, list or single number.
            lon: Longitudes between -180 and 180 degrees as 1-dimensional numpy
                array, list or single number. Must have the same length as
                `lat`.
            r: Radius in kilometers (if number is given). You can also use
                another unit if you pass this as string (e.g. *'10 miles'*).
                Despite of the unit which is given here, the returned distances
                will always be in kilometers.
            return_distance: If True, the distances will be returned. Otherwise
                not and the query will be faster. Default is true.

        Returns:
            Two numpy arrays: *pairs* and *distances*. The first has a
            *2xN* shape, where N is the number of found collocations. The first
            row contains the indices of the points with which the GeoIndex was
            built. The second row contains the matches in the query points.
            *distances* is a numpy array with distances in kilometers.
        """
        points = self._to_metric(lat, lon)

        # The user passes the radius in kilometers but we calculate with meters
        # internally
        r = to_kilometers(r)
        if self.metric == "minkowski":
            r *= 1000.
        elif self.metric == "haversine":
            r *= 1000. / earth_radius

        results = self.tree.query_radius(
            points, r, return_distance=return_distance
        )

        if return_distance:
            jagged_pairs, jagged_distances = results
        else:
            jagged_pairs = results

        # Build the list of the collocation pairs:
        pairs = np.array([
            [build_point, query_point]
            for query_point, build_points in enumerate(jagged_pairs)
            for build_point in build_points
        ]).T

        if not return_distance:
            return pairs

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


def gridded_mean(lat, lon, data, grid):
    """Grid data along latitudes and longitudes

    Args:
        lat: Grid points along latitudes as 1xN dimensional array.
        lon: Grid points along longitudes as 1xM dimensional array.
        data: The data as NxM numpy array.
        grid: A tuple with two numpy arrays, consisting of latitude and
            longitude grid points.

    Returns:
        Two matrices in grid form: the mean and the number of points of `data`.
    """
    grid_sum, _, _ = np.histogram2d(lat, lon, grid, weights=data)
    grid_number, _, _ = np.histogram2d(lat, lon, grid)

    return grid_sum / grid_number, grid_number


def sea_mask(lat, lon, mask):
    """Check whether geographical coordinates are over sea

    Notes:
        This uses per default a land sea mask with a grid size of 5 minutes.
        You have to decide by yourself whether it is sufficient for your data.

    Args:
        lat: Latitudes between -90 and 90 degrees as 1-dimensional numpy
                array.
        lon: Longitudes between -180 and 180 degrees as 1-dimensional numpy
            array. Must have the same length as `lat`.
        mask: Your own land-sea mask as a 2-dimensional boolean matrix or
            a path to a monochromatic PNG file. Must be sea pixels must be True
            or white, respectively.

    Returns:
        Returns a boolean array with the same dimensions as `lat`. It is True
        where the coordinate is over the sea.

    Examples:

        .. code-block:: python

            lat = np.array([60])
            lon = np.array([-0])

            is_over_sea(lat, lon, "land_water_mask_5min.png")
    """
    def _to_array(item):
        if isinstance(item, np.ndarray):
            return item

        if isinstance(item, Number):
            return np.array([item])
        else:
            return np.array(item)

    lat = _to_array(lat)
    lon = _to_array(lon)

    if not lat.size or not lon.size:
        return np.array([])

    if lon.min() < -180 or lon.max() > 180:
        raise ValueError("Longitudes out of bounds!")

    if lat.min() < -90 or lat.max() > 90:
        raise ValueError("Latitudes out of bounds!")

    if isinstance(mask, str):
        mask = np.flip(np.array(imageio.imread(mask) == 255), axis=0)

    mask_lat_step = 180 / (mask.shape[0] - 1)
    mask_lon_step = 360 / (mask.shape[1] - 1)

    lat_cell = (90 - lat) / mask_lat_step
    lon_cell = lon / mask_lon_step

    return mask[lat_cell.astype(int), lon_cell.astype(int)]


