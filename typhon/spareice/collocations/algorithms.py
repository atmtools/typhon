import abc
import logging
import time


import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree as SklearnBallTree
import typhon.constants
from typhon.geodesy import geocentric2cart
from typhon.spareice.array import Array
from typhon.utils.time import to_timedelta

__all__ = [
        "Finder",
        "BallTree",
        "BruteForce",
    ]


class Finder(metaclass=abc.ABCMeta):
    # Multiple threads in Python only speed things up when the code releases
    # the GIL. This flag can be used to indicate that this finder algorithm
    # should be used in multiple threads because it can handle such GIL issues.
    loves_threading = True

    @abc.abstractmethod
    def find_collocations(
            self, primary_data, secondary_data, max_interval, max_distance,
            **kwargs):
        """Finds collocations between two ArrayGroups / xarray.Datasets.

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
            Four lists:
            (1) primary indices - The indices of the data in *primary_data*
                that have collocations.
            (2) secondary indices - The indices of the data in *secondary_data*
                that have collocations.
            (3) primary collocations - This and (4) define the collocation
                pairs. It is a list containing indices of the corresponding
                data points. The first element of this list is collocated with
                the first element of the *secondary collocations* list (4),
                etc. Can contain duplicates if one dataset point has
                multiple collocations from the another dataset.
            (4) secondary collocations - Same as (3) but containing the
                collocations of the secondary data.
        """
        pass


class BallTree(Finder):
    def __init__(self, primary_leafsize=None, secondary_leafsize=None):
        super(BallTree, self).__init__()
        
        if primary_leafsize is None:
            self.primary_leafsize = 15
        else:
            self.primary_leafsize = primary_leafsize
        self.secondary_leafsize = secondary_leafsize

    def find_collocations(
        self, primary_data, secondary_data, max_interval, max_distance,
        **kwargs
    ):

        max_interval = to_timedelta(max_interval)

        if max_distance is None:
            # Search for temporal collocations only
            primary_time = \
                primary_data["time"].astype("M8[s]").astype("int")
            secondary_time = \
                secondary_data["time"].astype("M8[s]").astype("int")

            # The BallTree implementation only allows 2-dimensional data, hence
            # we need to add an empty second dimension
            # TODO: Change this
            primary_points = np.column_stack(
                [primary_time, np.zeros_like(primary_time)]
            )
            secondary_points = np.column_stack(
                [secondary_time, np.zeros_like(secondary_time)]
            )

            max_radius = max_interval.total_seconds()
        else:
            # We try to find collocations by building one 3-d Ball tree
            # (see https://en.wikipedia.org/wiki/K-d_tree) and searching for
            # the nearest neighbours. Since a k-d tree cannot handle latitude /
            # longitude data, we have to convert them to 3D-cartesian
            # coordinates. This introduces an error of the distance calculation
            # since it is now the distance in a 3D euclidean space and not the
            # distance along the sphere any longer. When having two points with
            # a distance of 5 degrees in longitude, the error is smaller than
            # 177 meters.
            cart_points = geocentric2cart(
                6371000.0, # typhon.constants.earth_radius,
                primary_data["lat"],
                primary_data["lon"]
            )
            primary_points = np.column_stack(cart_points)

            # We need to convert the secondary data as well:
            cart_points = geocentric2cart(
                6371000.0, # typhon.constants.earth_radius,
                secondary_data["lat"],
                secondary_data["lon"]
            )
            secondary_points = np.column_stack(cart_points)

            max_radius = max_distance*1000

        tree = SklearnBallTree(
            secondary_points, leaf_size=self.primary_leafsize)

        # Search for all collocations. This returns a list of lists. The index
        # of each element is the index of the primary data and each element
        # is a list of the indices of the collocated secondary data.
        # The parameter max_distance is in kilometers, so we have to convert it
        # to meters.
        results = tree.query_radius(primary_points, r=max_radius)

        # Build the list of the collocation pairs:
        pairs = np.array([
            [primary_index, secondary_index]
            for primary_index, secondary_indices in enumerate(results)
            for secondary_index in secondary_indices
        ]).T

        # No collocations were found.
        if not pairs.any():
            return pairs

        # Check here for temporal collocations:
        if max_distance is not None and max_interval is not None:

            # Check whether the time differences between the spatial
            # collocations are less than the temporal boundary:
            passed_time_check = np.abs(
                primary_data["time"][pairs[0]]
                - secondary_data["time"][pairs[1]]
            ) < np.timedelta64(max_interval)

            # Just keep all indices which satisfy the temporal condition.
            pairs = pairs[:, passed_time_check]

        return pairs


class BruteForce(Finder):
    def __init__(self):
        super(BruteForce, self).__init__()

    def find_collocations(
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
                np.abs(primary_data["time"][primary_indices]
                       - secondary_data["time"][secondary_indices])
            passed_time_check = intervals < max_interval

            primary_indices = primary_indices[passed_time_check]
            secondary_indices = secondary_indices[passed_time_check]

        return Array(primary_indices), Array(secondary_indices)



