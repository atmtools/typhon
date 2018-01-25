import abc
import logging
import time


import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree as SklearnBallTree
import typhon.constants
from typhon.geodesy import geocentric2cart
from typhon.spareice.array import Array

__all__ = [
        "Finder",
        "BallTree",
        "BruteForce",
    ]


class Finder(metaclass=abc.ABCMeta):
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
                in seconds. As well you can give here a datetime.timedelta
                object. If this is None, the data will be searched for
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

        timer = time.time()

        # We try to find collocations by building one 3-d Ball tree
        # (see https://en.wikipedia.org/wiki/K-d_tree) and searching for the
        # nearest neighbours. Since a k-d tree cannot handle latitude /
        # longitude data, we have to convert them to 3D-cartesian
        # coordinates. This introduces an error of the distance calculation
        # since it is now the distance in a 3D euclidean space and not the
        # distance along the sphere any longer. When having two points with a
        # distance of 5 degrees in longitude, the error is smaller than 177
        # meters.
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

        # TODO: Using multiple processes is deprecated so far. How to implement
        # TODO: them?

        # Results is a list of arrays. The first array contains the indices of
        # the found collocations of the secondary with the first point of the
        # primary dataset (and so on).
        sequence_id, results = self._find_spatial_collocations(
            0, primary_points, secondary_points, max_distance
        )

        # Build the list of the collocation indices. The index of the list
        # element denotes the collocation id. The element is the original
        # index of the primary / secondary data to the corresponding
        # collocation. For example, if there is at least one primary point
        # that collocates with multiple secondary points, the primary list
        # contains duplicates.
        primary_collocation_indices = Array([
            primary_index
            for primary_index, found in enumerate(results)
            for _ in range(len(found))
        ])

        # No collocations were found.
        if not primary_collocation_indices.any():
            return Array([]), Array([])

        secondary_collocation_indices = Array([
            secondary_index
            for found in results
            for secondary_index in found
        ])

        logging.debug(
            "\tFound {} primary and {} secondary spatial collocations in "
            "{:.2f}s.".format(
                primary_collocation_indices.size,
                secondary_collocation_indices.size,
                time.time() - timer
            )
        )

        # Check here for temporal collocations:
        if max_interval is not None:

            # Convert them from datetime objects to timestamps (seconds since
            # 1970-1-1) to make them faster for temporal distance checking.
            primary_time = primary_data["time"]
            secondary_time = secondary_data["time"]

            # Check whether the time differences between the spatial
            # collocations are less than the temporal boundary:
            passed_time_check = np.abs(
                primary_time[primary_collocation_indices]
                - secondary_time[secondary_collocation_indices]
            ) < max_interval

            # Just keep all indices which satisfy the temporal condition.
            primary_collocation_indices = \
                primary_collocation_indices[passed_time_check]
            secondary_collocation_indices = \
                secondary_collocation_indices[passed_time_check]

        return primary_collocation_indices, secondary_collocation_indices

    def _find_spatial_collocations(self,
            sequence_id, primary_points, secondary_points, max_distance):
        """ This finds spatial collocations between the primary and secondary
        data points and it does not regard the temporal dimension.

        One should not call this function directly but
        :meth:`find_collocations` which offers a performance boost by using
        multiple processes.

        Args:
            sequence_id: The process/worker id. Since this function is designed
                to called by multiple processes, this parameter is used to
                assign the results to the corresponding process.
            primary_points: The primary data points, an array of arrays with
                three elements: [x, y, z].
            secondary_points: The secondary data points, an array of arrays
                with three elements: [x, y, z].
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria.

        Returns:
            A list with the indices of the primary and secondary data that
            collocate with each other.
        """

        timer = time.time()
        tree = SklearnBallTree(
            secondary_points, leaf_size=self.primary_leafsize)
        logging.debug("\tNeeded {:.2f}s: for building the tree.".format(
            time.time() - timer))

        # Search for all collocations. This returns a list of lists. The index
        # of each element is the index of the primary data and each element
        # is a list of the indices of the collocated secondary data.
        # The parameter max_distance is in kilometers, so we have to convert it
        # to meters.
        timer = time.time()
        collocation_indices = tree.query_radius(
            primary_points, r=max_distance*1000)
        logging.debug("\tNeeded {:.2f}s for finding spatial "
                      "collocations".format(time.time() - timer))

        return sequence_id, collocation_indices

    @staticmethod
    def _find_temporal_collocations(
            sequence_id, primary_points, secondary_points, max_interval):
        """This finds temporal collocations between the primary and secondary
        data points.

        TODO:
            Fill this function.

        Args:
            sequence_id:
            primary_points:
            secondary_points:
            max_interval:

        Returns:

        """
        ...


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



