import abc
import logging
import time

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree as SklearnBallTree
import typhon.constants
from typhon.geodesy import geocentric2cart

__all__ = [
        "CollocatorAlgorithm",
        "BallTree",
        "BruteForce",
    ]


class CollocatorAlgorithm(metaclass=abc.ABCMeta):
    # Multiple threads in Python only speed things up when the code releases
    # the GIL. This flag can be used to indicate that this finder algorithm
    # should be used in multiple threads because it can handle such GIL issues.
    loves_threading = True

    # What fields are required for performing the collocation?
    required_fields = ("time", "lat", "lon")

    @abc.abstractmethod
    def run(
            self, primary_data, secondary_data, max_interval, max_distance,
            **kwargs):
        """Find collocations between two pandas.DataFrames

        Args:
            primary_data: The data from the primary dataset as xarray.Dataset
                object. This object must provide the special fields "lat",
                "lon" and "time" as 1-dimensional arrays.
            secondary_data: The data from the secondary dataset as
                xarray.Dataset object. Needs to meet the same conditions as
                primary_data ("lat", "lon" and "time" fields).
            max_interval: The maximum interval of time between two data points
                in seconds. If this is None, the data will be searched for
                spatial collocations only.
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria.

        Returns:
            One numpy.array with two rows (for the primary and the secondary
            data) and N columns. Each column represents one collocation and the
            two values indicate the index in the primary and secondary data.
        """
        pass


class BallTree(CollocatorAlgorithm):
    def __init__(self, tunnel_limit=None, magnitude_factor=100, **kwargs):
        """Initialize a CollocationsFinder with a BallTree algorithm

        Args:
            tunnel_limit: Maximum distance in kilometers at which to switch
                from tunnel to haversine distance metric. Per default this
                algorithm uses the tunnel metric, which simply transform all
                latitudes and longitudes to 3D-cartesian space and calculate
                their euclidean distance. This produces an error that grows
                with larger distances. When searching for distances exceeding
                this limit (`max_distance` is greater than this parameter), the
                haversine metric is used, which is more accurate but takes more
                time. Default is 1000 kilometers.
            magnitude_factor: Since building new trees is expensive, this
                algorithm tries to use the last tree when possible (e.g. for
                data with fixed grid). However, building the tree with the
                larger dataset and query it with the smaller dataset is faster
                than vice versa. Depending on which premise to follow, there
                might have a different performance in the end. This parameter
                is the factor that... TODO
            **kwargs: Additional keyword arguments that are allowed for the
                scikit-learn BallTree class (
                http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html)
                such as `leaf_size`.
        """
        super(BallTree, self).__init__()

        if tunnel_limit is None:
            self.tunnel_limit = 1000
        else:
            self.tunnel_limit = tunnel_limit

        # Building a ball tree is expensive. Let's cache the points from last
        # time, so maybe it is possible to reuse the last tree?
        self.tree = None
        self.tree_points = None
        self.used_primary = None

        # Additional BallTree key word arguments:
        kwargs = {}
        self.tree_kwargs = kwargs

        # New tree factor:
        self.magnitude_factor = magnitude_factor

    def run(
        self, primary_data, secondary_data, max_interval, max_distance,
        **kwargs,
    ):
        """Find collocations between two pandas.DataFrame objects

        Args:
            primary_data: The data from the primary dataset as xarray.Dataset
                object. This object must provide the special fields "lat",
                "lon" and "time" as 1-dimensional arrays.
            secondary_data: The data from the secondary dataset as
                xarray.Dataset object. Needs to meet the same conditions as
                primary_data ("lat", "lon" and "time" fields).
            max_interval: The maximum interval of time between two data points
                in seconds. If this is None, the data will be searched for
                spatial collocations only.
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria.

        Returns:
            Three numpy.arrays: the pairs of collocations (as indices in the
            original data), the interval for the time dimension and the
            distance for the spatial dimension. The pairs are a 2xN numpy.array
            where N is the number of found collocations. The first row contains
            the indices of the collocations in `data1`, the second row the
            indices in `data2`.
        """

        # Convert the data, set radius and choose metric
        primary_points, secondary_points, max_radius, metric \
            = self._prepare_tree_data(
                primary_data, secondary_data, max_interval, max_distance
            )

        # Finding collocations is expensive, therefore we want to optimize it
        # and have to decide which points to use for tree building.
        tree_with_primary = self._choose_points_to_build_tree(
            primary_points, secondary_points
        )

        # Build the tree:
        self.tree = self._build_tree(
            primary_points if tree_with_primary else secondary_points,
            metric
        )

        # Query the tree:
        jagged_pairs, jagged_distances = self.tree.query_radius(
            secondary_points if tree_with_primary else primary_points,
            r=max_radius, return_distance=True,
        )

        # Build the list of the collocation pairs:
        if tree_with_primary:
            pairs = np.array([
                [primary_index, secondary_index]
                for secondary_index, primary_indices in enumerate(jagged_pairs)
                for primary_index in primary_indices
            ]).T
        else:
            pairs = np.array([
                [primary_index, secondary_index]
                for primary_index, secondary_indices in enumerate(jagged_pairs)
                for secondary_index in secondary_indices
            ]).T

        # No collocations were found.
        if not pairs.any():
            # We return empty arrays to have consistent return values:
            return np.array([[], []]), \
                   np.array([], dtype='timedelta64[ns]'), \
                   np.array([])

        distances = np.hstack([
            distances_with_primary
            for distances_with_primary in jagged_distances
        ])

        intervals = np.abs(
            primary_data.index[pairs[0]]
            - secondary_data.index[pairs[1]]
        )

        # Check here for temporal collocations:
        if max_distance is not None and max_interval is not None:
            # Check whether the time differences between the spatial
            # collocations are less than the temporal boundary:

            passed_time_check = intervals < max_interval

            # Just keep all indices which satisfy the temporal condition.
            pairs = pairs[:, passed_time_check]
            intervals = intervals[passed_time_check]
            distances = distances[passed_time_check]

        return pairs, intervals, distances

    def _build_tree(self, points, metric):
        # Find out whether the cached tree still works with the new points:
        if self._is_cached(points):
            return self.tree

        # Or create a new one:
        self.tree_points = points

        return SklearnBallTree(
            points, **{**self.tree_kwargs, "metric": metric}
        )

    def _prepare_tree_data(self, primary_data, secondary_data, max_interval,
                           max_distance):
        # We can use different metrics for the BallTree. The default euclidean
        # metric is the fastest but produces a big error for large distances.
        # We have to convert also all data to 3D-cartesian coordinates.
        # The users can define a spatial limit for the euclidean metric. If
        # they set a maximum distance greater than this limit, the haversine
        # metric will be used, which is more correct.

        metric = "minkowski"
        if max_distance is None:
            # Search for temporal collocations only
            primary_time = \
                primary_data.index.to_datetime().astype(int) / 10e8
            secondary_time = \
                secondary_data.index.to_datetime().astype(int) / 10e8

            # The BallTree implementation only allows 2-dimensional data, hence
            # we need to add an empty second dimension
            # TODO: Use a more adequate algorithm here? Maybe a range tree?
            primary_points = np.column_stack(
                [primary_time, np.zeros_like(primary_time)]
            )
            secondary_points = np.column_stack(
                [secondary_time, np.zeros_like(secondary_time)]
            )

            max_radius = max_interval.total_seconds()
        elif self.tunnel_limit is not None \
                and max_distance > self.tunnel_limit:
            # Use the more expensive and accurate haversine metric
            metric = "haversine"

            # We need the latitudes and longitudes in radians:
            primary_points = np.radians(
                np.column_stack([primary_data["lat"], primary_data["lon"]])
            )
            secondary_points = np.radians(
                np.column_stack([secondary_data["lat"], secondary_data["lon"]])
            )

            # The parameter max_distance is in kilometers, so we have to
            # convert it to meters. The calculated distances by the haversine
            # metrics are on an unity sphere:
            max_radius = max_distance * 1000 / typhon.constants.earth_radius
        else:
            # Use the default and cheap tunnel metric

            # We try to find collocations by building one 3-d Ball tree
            # (see https://en.wikipedia.org/wiki/K-d_tree) and searching for
            # the nearest neighbours. Since a ball tree cannot handle latitude/
            # longitude data, we have to convert them to 3D-cartesian
            # coordinates. This introduces an error of the distance calculation
            # since it is now the distance through a tunnel and not the
            # distance along the sphere any longer. However, when having two
            # points with a distance of 10 degrees in longitude (ca. 1113 km)
            # at the equator, the error is roughly 1.5 kilometers. So it is
            # okay for small distances.
            cart_points = geocentric2cart(
                typhon.constants.earth_radius,
                primary_data["lat"],
                primary_data["lon"]
            )
            primary_points = np.column_stack(cart_points)

            # We need to convert the secondary data as well:
            cart_points = geocentric2cart(
                typhon.constants.earth_radius,
                secondary_data["lat"],
                secondary_data["lon"]
            )
            secondary_points = np.column_stack(cart_points)

            # The parameter max_distance is in kilometers, so we have to
            # convert it to meters.
            max_radius = max_distance * 1000

        return primary_points, secondary_points, max_radius, metric

    def _choose_points_to_build_tree(self, primary, secondary):
        """Choose which points should be used for tree building

        This method helps to optimize the performance.

        Args:
            primary: Converted primary points
            secondary: Converted secondary points

        Returns:
            True if primary points should be used for tree building. False
            otherwise.
        """
        # There are two options to optimize the performance:
        # A) Cache the tree and reuse it if either the primary or the secondary
        # points have not changed (that is the case for data with a fixed
        # grid). Building the tree is normally very expensive, so it should
        # never be done without a reason.
        # B) Build the tree with the larger set of points and query it with the
        # smaller set.
        # Which option should be used if A and B cannot be applied at the same
        # time? If the magnitude of one point set is much larger (by
        # `magnitude factor` larger) than the other point set, we strictly
        # follow B. Otherwise, we prioritize A.

        if primary.size > secondary.size * self.magnitude_factor:
            # Use primary points
            return True
        elif secondary.size > primary.size * self.magnitude_factor:
            # Use secondary points
            return False

        # Apparently, none of the datasets is much larger than the others. So
        # just check whether we still have a cached tree. If we used the
        # primary points last time and they still fit, use them again:
        if self.used_primary and self._is_cached(primary):
            return True

        # Check the same for the secondary data:
        if not self.used_primary and self._is_cached(secondary):
            return False

        # Otherwise, just use the larger dataset:
        return primary.size > secondary.size

    def _is_cached(self, points):
        if self.tree_points is None:
            return False

        try:
            return np.allclose(points, self.tree_points)
        except ValueError:
            # The shapes are different
            return False


class BruteForce(CollocatorAlgorithm):
    def __init__(self):
        super(BruteForce, self).__init__()

    def run(
            self, primary_data, secondary_data, max_interval, max_distance,
            **kwargs
    ):

        timer = time.time()

        # Find all spatial collocations by brute-force:
        primary_points = geocentric2cart(
            typhon.constants.earth_radius,
            primary_data["lat"], primary_data["lon"])
        secondary_points = geocentric2cart(
            typhon.constants.earth_radius,
            secondary_data["lat"], secondary_data["lon"])

        distances = distance_matrix(
            np.column_stack(primary_points),
            np.column_stack(secondary_points)
        )
        primary_indices, secondary_indices = np.nonzero(
            distances < max_distance*1000)

        logging.debug(
            "\tFound {} primary and {} secondary spatial collocations in "
            "{:.2f}s.".format(
                primary_indices.size,
                secondary_indices.size,
                time.time() - timer
            )
        )

        # Check the temporal condition:
        if max_interval is not None:
            intervals = \
                np.abs(primary_data.index[primary_indices]
                       - secondary_data.index[secondary_indices])
            passed_time_check = intervals < max_interval

            primary_indices = primary_indices[passed_time_check]
            secondary_indices = secondary_indices[passed_time_check]

        return primary_indices, secondary_indices



