"""
This module contains classes to find collocations between datasets. They are
inspired by the implemented CollocatedDataset classes in atmlab written by
Gerrit Holl.

TODO: I would like to have this package as typhon.collocations.

Created by John Mrziglod, June 2017
"""

from collections import OrderedDict
from datetime import datetime, timedelta
import time

try:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
except:
    pass

import numpy as np
import scipy.spatial
import scipy.stats
import typhon.geodesy
from typhon.spareice.geographical import GeoData

from .datasets import Dataset

__all__ = [
    "CollocatedDataset"
]


class CollocatedDataset(Dataset):
    """Still under development.

    A dataset that finds and stores collocations amongst different datasets
    with geographical data.

    Collocations are match-ups between two datasets, i.e. a data point from
    a dataset that is located close to another data point from a secondary
    dataset in space and time.
    """

    def __init__(self, *args, **kwargs):
        """Opens existing files with collocated data as a CollocatedDataset
        object.

        If you have already collocated some datasets, you can open the stored
        collocations with this command.

        If you want to collocate two datasets, you should use the
        :meth:`from_datasets` method.

        Args:
            *args: Same positional arguments that the
                :class:`typhon.spareice.datasets.Dataset` base class accepts.
            **kwargs: Same key word arguments that the
                :class:`typhon.spareice.datasets.Dataset` base class accepts.

        Returns:
            A CollocatedDataset object.

        Examples:
            >>> CollocatedDataset(
            >>>     "/path/to/{year}/{month}/{day}.nc",
            >>>     handler=NetCDF4(),
            >>> )
        """
        super(CollocatedDataset, self).__init__(*args, **kwargs)

        # Which dataset should be taken when we collocate this dataset with
        # other datasets?
        self.primary_dataset = None

    def accumulate(self, start, end, concat_func=None, concat_args=None,
                   reading_args=None):
        """Accumulate all data between two dates in one GeoData object.

        Args:
            start: Start date either as datetime.datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            concat_func: Function that concatenates the read data to
                another. The first expected argument must be a list of
                objects to concatenate. Default is GeoData.concatenate.
            concat_args: A dictionary with additional arguments for
                *concat_func*.
            reading_args: A dictionary with additional arguments for reading
                the data (specified by the used file handler).

        Returns:
            Concatenated object.

        Examples:
            .. :code-block:: python

            dataset = CollocatedDataset(
                files="path/to/files.nc", handler=handlers.common.NetCDF4()
            )
            data = dataset.accumulate("2016-1-1", "2016-1-2",
                read_args={"fields" : ("temperature", )})

            # do something with data["temperature"]
            ...
        """

        if concat_func is None:
            concat_func = GeoData.concatenate

        return super(CollocatedDataset, self).accumulate(
            start, end, concat_func, concat_args, reading_args)

    def collapse(self, start, end, collapse_to,
                 include_stats=None, **collapsing_args):
        """ Accumulates the data between two dates but collapses multiple
        collocations from one dataset to a single data point.

        During searching for collocations, one might find multiple collocation
        points of the secondary dataset for one single point of the primary
        dataset. For example, the MHS instrument has a bigger footprint than
        the AVHRR instrument, hence one will find several AVHRR colloocation
        points for each MHS data point.

        Args:
            start: Starting date as datetime object.
            end: Ending date as datetime object.
            collapse_to: Name of dataset which has the coarsest footprint. All
                other datasets will be collapsed to its data points.
            include_stats: Set this to a name of a variable and in the return
                object will be statistical parameters included about the built
                data bins of the variable before collapsing. The variable
                should be one-dimensional.
            **collapsing_args: Additional keyword arguments for the
                GeoData.collapser method (including collapser function, etc.).

        Returns:
            A GeoData object with the collapsed data.

        Examples:

        """

        # Exclude all bins where the inhomogeneity (variation) is too high
        # passed = np.ones_like(bins).astype("bool")
        # if isinstance(variation_filter, tuple):
        #     if len(variation_filter) >= 2:
        #         if len(self[variation_filter[0]].shape) > 1:
        #             raise ValueError(
        #                 "The variation filter can only be used for "
        #                 "1-dimensional data! I.e. the field '{}' must be "
        #                 "1-dimensional!".format(variation_filter[0])
        #             )
        #
        #         # Bin only one field for testing of inhomogeneities:
        #         binned_data = self[variation_filter[0]].bin(bins)
        #
        #         # The user can define a different variation function (
        #         # default is the standard deviation).
        #         if len(variation_filter) == 2:
        #             variation_values = variation(binned_data, 1)
        #         else:
        #             variation_values = variation_filter[2](binned_data, 1)
        #         passed = variation_values < variation_filter[1]
        #     else:
        #         raise ValueError("The inhomogeneity filter must be a tuple "
        #                          "of a field name, a threshold and (optional)"
        #                          "a variation function.")

        collapsed_data_list = []
        for file, _ in self.find_files(start, end, sort=True):
            collocated_data = self.read(file)

            # Get the bin indices by the main dataset to which all other
            # shall be collapsed:
            bins = list(
                collocated_data[collapse_to]["__collocations"].group().values()
            )

            collapsed_data = GeoData()

            # Add additional statistics about one binned variable:
            if include_stats is not None:
                statistic_functions = {
                    "variation": scipy.stats.variation,
                    "mean": np.nanmean,
                    "number": lambda x, _: x.shape[0],
                    "std": np.nanstd,
                }

                collapsed_data["__statistics"] = \
                    collocated_data[include_stats].apply_on_bins(
                        bins, statistic_functions
                    )
                collapsed_data["__statistics"].attrs["description"] = \
                    "Statistics about the collapsed bins of '{}'.".format(
                        include_stats
                    )

            for dataset in collocated_data.groups():
                if dataset.startswith("__"):
                    collapsed_data[dataset] = collocated_data[dataset]

                collocations = collocated_data[dataset]["__collocations"]

                # We do not need the original and collocation indices any
                # longer because they will soon become useless. Moreover,
                # they could have a different dimension length than the
                # other variables and lead to errors in the selecting process:
                del collocated_data[dataset]["__original_indices"]
                del collocated_data[dataset]["__collocations"]

                if dataset == collapse_to:
                    # This is the main dataset to which all other will be
                    # collapsed. Therefore, we do not need explicit
                    # collapsing here.
                    collapsed_data[dataset] = \
                        collocated_data[dataset][np.unique(collocations)]
                else:
                    collapsed_data[dataset] = \
                        collocated_data[dataset].collapse(
                            bins, **collapsing_args
                        )

            collapsed_data_list.append(collapsed_data)

        return GeoData.concatenate(collapsed_data_list, dimension=0)

    @staticmethod
    def collocate(
            primary_data, secondary_data,
            max_interval=300, max_distance=10,
            processes=8, check_temporal_boundaries=True):
        """Finds collocations between two ArrayGroups / xarray.Datasets.

        TODO: Rename this function. Improve this doc string.

        Args:
            primary_data: The data from the primary dataset as xarray.Dataset
                object. This object must provide the special fields "lat",
                "lon" and "time" as 1-dimensional arrays.
            secondary_data: The data from the secondary dataset as
                xarray.Dataset object. Needs to meet the same conditions as
                primary_data ("lat", "lon" and "time" fields).
            max_interval: The maximum interval of time between two data points
                in seconds. As well you can give here a datetime.timedelta
                object.
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria.
            processes: The number of processes that should be used to boost the
                collocation process. The optimal number of processes heavily
                depends on your machine where you are working. I recommend to
                start with 8 processes and to in/decrease this parameter
                when lacking performance.
            check_temporal_boundaries: Whether to check the temporal interval
                as well. If this is false, the data will be searched for
                spatial collocations only.

        Returns:
            Four lists:
            (1) primary indices - The indices of the data in primary_data that
                meet the collocation conditions.
            (2) primary collocations - All collocation points of the primary
                data. Each element is an index of an element in the primary
                indices list (1).
            (3) secondary indices - The indices of the data in secondary_data
                that meet the collocation conditions.
            (4) secondary collocations - TODO

        """

        timer = time.time()

        if not isinstance(max_interval, timedelta):
            max_interval = timedelta(seconds=max_interval)

        # We try to find collocations by building one 3-d tree for each dataset
        # (see https://en.wikipedia.org/wiki/K-d_tree) and searching for the
        # nearest neighbours. Since a k-d tree cannot handle latitude /
        # longitude data, we have to convert them to 3D-cartesian
        # coordinates.
        x, y, z = typhon.geodesy.geocentric2cart(
            typhon.constants.earth_radius,
            primary_data["lat"],
            primary_data["lon"]
        )

        primary_points = np.asarray(list(zip(x, y, z)))

        # We need to convert the secondary data as well:
        x, y, z = typhon.geodesy.geocentric2cart(
            typhon.constants.earth_radius,
            secondary_data["lat"],
            secondary_data["lon"]
        )
        secondary_points = np.asarray(list(zip(x, y, z)))

        print("\tAfter {:.2f}s: finished the conversion from longitude"
              "/latitudes to cartesian.".format(
            time.time() - timer))

        # We want to use parallel processes to find spatial collocations since
        # it can take a lot of time. Hence, split the secondary data into N
        # chunks, where N is the number of parallel processes to use.
        # if secondary_points.shape[0] > 500:
        #    secondary_points_chunks = \
        #       np.array_split(secondary_points, processes)
        # else:
        #    secondary_points_chunks = [secondary_points]

        # We build the offset indices that indicate at which index a chunk
        # starts originally. We need those later to
        # identify the collocations in the original data.
        # offset_indices = [len(chunk) for chunk in secondary_points_chunks]
        # offset_indices.insert(0, 0)
        # offset_indices.pop()
        # offset_indices = [
        #   sum(offset_indices[:i+1]) for i, _ in enumerate(offset_indices)]

        # Prepare the k-d tree for the priary dataset:
        primary_tree = scipy.spatial.cKDTree(primary_points, leafsize=5)
        print("\tAfter {:.2f}s: finished building primary tree.".format(
            time.time() - timer))

        # To avoid defining complicated distance metrics in the k-d tree, we
        # use the tree for checking for spatial collocations only. All found
        # spatial collocations will be checked by using temporal boundaries
        # later.
        # data_pool = [
        #     (Dataset._find_spatial_collocations, None,
        #     (i, primary_tree, chunk, max_distance), None)
        #     for i, chunk in enumerate(secondary_points_chunks)
        # ]
        # pool = Pool(processes=processes)
        # results = pool.map(Dataset._call_map_function, data_pool)

        sequence_id, result = CollocatedDataset._find_spatial_collocations(
            0, primary_tree, secondary_points, max_distance
        )
        print("\tAfter {:.2f}s: finished finding spatial collocations.".format(
            time.time() - timer))

        primary_collocation_indices, secondary_collocation_indices = [], []

        # Merge all result lists.
        # for sequence_id, result in sorted(results, key=lambda x: x[0]):
        #
        #     # Build the list of the collocation indices. The index of the
        #     # list element denotes the collocation id. The element is the
        #     # original index of the primary / secondary data to the
        #     # corresponding collocation. For example, if there is at least
        #     # one primary point that collocates with multiple secondary
        #     # points, the primary list contains duplicates.
        #     primary_collocation_indices.extend([
        #         primary_index
        #         for primary_index, found in enumerate(result)
        #         for _ in range(len(found))
        #     ])
        #     secondary_collocation_indices.extend([
        #         secondary_index + offset_indices[sequence_id]
        #         for found in result
        #         for secondary_index in found
        #     ])

        # Build the list of the collocation indices. The index of the list
        # element denotes the collocation id. The element is the original
        # index of the primary / secondary data to the corresponding
        # collocation. For example, if there is at least one primary point
        # that collocates with multiple secondary points, the primary list
        # contains duplicates.
        primary_collocation_indices.extend([
            primary_index
            for primary_index, found in enumerate(result)
            for _ in range(len(found))
        ])
        secondary_collocation_indices.extend([
            secondary_index  # + offset_indices[sequence_id]
            for found in result
            for secondary_index in found
        ])

        # No collocations were found.
        if not primary_collocation_indices:
            print("\tNo collocations were found.")
            return None, None, None, None

        # Convert them to arrays to make them better sliceable.
        primary_collocation_indices = \
            np.asarray(primary_collocation_indices)
        secondary_collocation_indices = \
            np.asarray(secondary_collocation_indices)

        print("\tAfter {:.2f}s: finished retrieving the indices.".format(
            time.time() - timer))

        # Check here for temporal collocations:
        if check_temporal_boundaries:

            # Convert them from datetime objects to timestamps (seconds since
            # 1970-1-1) to make them faster for temporal distance checking.
            primary_time = primary_data["time"].astype('M8[s]')
            secondary_time = secondary_data["time"].astype('M8[s]')

            # Check whether the time differences between the spatial
            # collocations are less than the temporal boundary:
            collocation_indices = np.abs(
                primary_time[primary_collocation_indices]
                - secondary_time[secondary_collocation_indices]
            ) < np.timedelta64(max_interval)

            # Just keep all indices which satisfy the temporal condition.
            primary_collocation_indices = \
                primary_collocation_indices[collocation_indices]
            secondary_collocation_indices = \
                secondary_collocation_indices[collocation_indices]

        # No collocations were found.
        if not primary_collocation_indices.any():
            print("\tNo collocations were found. Took {:.2f}s.".format(
                time.time() - timer))
            return None, None, None, None

        # We are going to return four lists, two for each dataset. primary_
        # indices and secondary_indices are the lists with the indices of
        # the data in the original data arrays. So if someone wants to retrieve
        # other data from the original files one day, he can use those
        # indices. primary_collocations and secondary_collocations contain the
        # collocation pairs. The first element of the primary_collocations list
        # corresponds to the first element of the secondary_collocations and
        # so on. Their elements are the indices of the primary_indices and
        # secondary_indices elements. For example, if someone wants to
        # compare the collocations between primary and secondary, one could
        # use these expressions:
        #       primary_data[primary_indices[primary_collocations]] and
        #       secondary_data[secondary_indices[secondary_collocations]]
        primary_indices, primary_collocations, \
        secondary_indices, secondary_collocations = [], [], [], []

        # Simply remove all duplicates from the collocation_indices lists while
        # keeping the order:
        primary_indices = list(
            OrderedDict.fromkeys(primary_collocation_indices))
        secondary_indices = list(
            OrderedDict.fromkeys(secondary_collocation_indices))

        # print(primary_collocation_indices, secondary_collocation_indices)

        # We want the collocation lists (primary_collocations and secondary_
        # collocations) to show the indices of the indices lists (
        # primary_indices and secondary_indices). For example, primary_
        # collocation_indices shows [44, 44, 44, 103, 103, 109] at the
        # moment, but we want it to convert to [0, 0, 0, 1, 1, 2] where the
        # numbers correspond to the indices of the elements in
        # primary_indices which are [44, 103, 109].
        primary_index_ids = \
            {primary_index: xid
             for xid, primary_index in enumerate(primary_indices)}
        secondary_index_ids = \
            {secondary_index: xid
             for xid, secondary_index in enumerate(secondary_indices)}

        primary_collocations = \
            [primary_index_ids[primary_index]
             for primary_index in primary_collocation_indices]
        secondary_collocations = \
            [secondary_index_ids[secondary_index]
             for secondary_index in secondary_collocation_indices]

        print("\tFound {} primary and {} secondary collocations"
              " in {:.2f}s.".format(len(primary_indices),
                                    len(secondary_indices),
                                    time.time() - timer))

        return primary_indices, primary_collocations, \
               secondary_indices, secondary_collocations

    @staticmethod
    def _find_spatial_collocations(
            sequence_id, primary_tree, secondary_points, max_distance):
        """ This finds spatial collocations between the primary and secondary
        data points and it does not regard the temporal dimension.

        One should not call this function directly but .collocate()
        which offers a performance boost by using multiple processes.

        Args:
            sequence_id: The process/worker id. Since this function is designed
                to called by multiple processes, this parameter is used to
                assign the results to the corresponding process.
            primary_tree: The primary data points, as scipy.spatial.cKDTree
                object.
            secondary_points: The secondary data points, an array of arrays
                with three elements: [x, y, z]
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria.

        Returns:
            A list with the indices of the primary and secondary data that
            collocate with each other.
        """

        secondary_tree = scipy.spatial.cKDTree(secondary_points, leafsize=10)

        # Search for all collocations. This returns a list of lists. The index
        # of each element is the index of the primary data and each element
        # is a list of the indices of the collocated secondary data.
        # The parameter max_distance is in kilometers, so we have to convert it
        # to meters.
        collocation_indices = primary_tree.query_ball_tree(
            secondary_tree, max_distance * 1000)

        return sequence_id, collocation_indices

    @classmethod
    def collocate_from_datasets(
            cls, primary, secondary, start, end,
            fields, max_interval=None, max_distance=None, optimizer=None,
            processes=4, check_temporal_boundaries=True,
            *args, **kwargs):
        """Finds all collocations between two datasets and store them in files.

        This takes all files from the datasets between two dates and find
        collocations of their data points. Where and how the collocated data is
        stored is controlled by the *files* and *handler* parameter.

        If you want to collocate multiple datasets (more than two), you can
        give as a primary dataset also a CollocatedDataset.

        Each collocation output file provides these standard fields for each
        dataset:

        dataset_name/lat - Latitudes of the collocations.
        dataset_name/lon - Longitude of the collocations.
        dataset_name/time - Timestamp of the collocations.
        dataset_name/indices - Indices of the collocation data in the original
            files.
        dataset_name/collocations - Tells you which data points collocate with
            each other by giving their indices.

        Since this class inherits from the
        :class:`typhon.spareice.datasets.Dataset` class, please look also at
        its documentation.

        TODO: Revise and extend documentation. Maybe rename this method?
        TODO: Split this method into smaller methods.

        Args:
            primary: A Dataset or CollocatedDataset object.
            secondary: A Dataset or CollocatedDataset object.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as *start*.
            fields: The fields that should be extracted from the datasets and
                copied to the new collocation files. This should be a list
                of two lists containing the field names for each dataset. They
                will be accessible via "dataset_name.field_name" in the new
                collocation files. If you want to learn more about how to
                extract specific dimension from those fields, please read
                the documentation of the typhon.handlers classes.
            max_interval: (optional) The maximum interval of time between two
                data points in seconds. Default is 300 seconds.
            max_distance: (optional) The maximum distance between two data
                points in kilometers to meet the collocation criteria. Default
                is 10km.
            optimizer: (optional) Finding of collocations can sometimes be
                optimized.
            processes: The number of processes that should be used to boost the
                collocation process. The optimal number of processes heavily
                depends on your machine where you are working. I recommend to
                start with 8 processes and to in/decrease this parameter
                when lacking performance.
            check_temporal_boundaries: Whether to check the temporal interval
                as well. If this is false, the data will be searched for
                spatial collocations only.
            *args: Positional arguments that will passed to the
                :class:`typhon.spareice.datasets.Dataset` base class.
            **kwargs: Additional keyword arguments that will passed to the
                datasets.Dataset base class.

        Returns:
            A CollocatedDataset object.

        Examples:

        .. :code-block:: python

            from typhon.spareice.datasets import Dataset
            from typhon.spareice.collocations import CollocatedDataset

            # Define the two dataset amongst them you want to find
            # collocations.
            dataset1 = Dataset(
                "/path/to/files/noaa18_mhs_{year}/{month}/{day}/*.h5",
                name="MHS",
                handler=tovs.MHSAAPP(), # file_handler
                timestamp_retrieving_method="content"
            )
            dataset2 = Dataset(
                "/path2/to/files/{year}/{doy}/{year}{doy}{hour}{minute}{second}_*.hdf.zip",
                handler=cloudsat.C2CICE(),  # file_handler
                name="2C-ICE",
            )

            # Create the collocated dataset. The parameters "files" and
            # "file_handler" tells the object where the collocations should
            be stored and which format should be used.
            collocated_dataset = CollocatedDataset.from_datasets(
                dataset1, dataset2,
                # Define the period where you want to collocate.
                start="2007-01-01", end="2007-01-02",
                # Find the collocations with a maximum distance of 200km and
                # temporal interval of 300 seconds. Extract the
                # field "brightness_temperature" from the primary and
                # "ice_water_path" from the secondary dataset.
                max_distance=200, max_interval=300,
                fields=[["brightness_temperature"], ["ice_water_path"]],
                name="CollocatedDataset",
                # The found collocations will be stored as NetCDF4 files in
                # this path:
                files="CollocatedDataset/{year}/{month}/{day}/{hour}{minute}{second}.nc",
                # If you need another data format than NetCDF4 then pass
                # another file handler object:
                handler=common.NetCDF4(),
            )

        """
        dataset = cls(*args, **kwargs)
        dataset.primary_dataset = primary.name

        # The user wants sometimes to collocate already collocated data with an
        # additional dataset. That is totally fine, but we have make some
        # adjustments:
        if isinstance(primary, CollocatedDataset):
            pass

        # start and end could have been string objects:
        start = dataset._to_datetime(start)
        end = dataset._to_datetime(end)

        # All additional fields given by the user and the standard fields that
        # we need for finding collocations.
        primary_fields = list(set(fields[0]) | {"time", "lat", "lon"})
        secondary_fields = list(set(fields[1]) | {"time", "lat", "lon"})

        # The data of secondary files can overlap multiple primary files.
        last_secondary = None
        secondary_data_cache = None

        # Use this variable to store the point in time to which the were
        # checked. This avoids checking duplicates.
        last_timestamp = None

        total_primaries_points, total_secondaries_points = 0, 0

        # Use a timer for profiling.
        timer = time.time()

        # Go through all primary files and find the secondaries to them:
        for primary_file, secondary_files in primary.find_overlapping_files(
                start, end, secondary, max_interval):

            print("Primary:", primary_file)

            # Skip this primary file if there are no secondaries found.
            if not secondary_files:
                print("\tNo files of the secondary were found!")
                continue

            # Read the data of the primary file
            if isinstance(primary, CollocatedDataset):
                # The data comes from a collocated dataset, i.e. we have to
                # pick the primary dataset from that collocated dataset and
                # collocate the secondary data with those data points. After
                # collocating we copy only the collocated points from the
                # secondary dataset of the collocated dataset to the new
                # created files.
                primary_full_data_cache = GeoData.from_xarray(
                    primary.read(primary_file))

                if primary.primary_dataset is not None:
                    primary_dataset = primary.primary_dataset
                elif "primary_dataset" in primary_full_data_cache.attrs:
                    primary_dataset = \
                        primary_full_data_cache.attrs["primary_dataset"]
                else:
                    raise AttributeError(
                        "There is no primary dataset set in '{0}' (the "
                        "primary dataset)! I do not know which dataset from "
                        "'%s' to use for collocating. You can set this via "
                        "changing the primary_dataset attribute from "
                        "'{0}'.".format(primary.name))

                # Select only the dataset from the collocated data that we want
                # to use for finding the collocations.
                primary_data_cache = primary_full_data_cache[primary_dataset]
            else:
                primary_data_cache = primary.read(
                    primary_file,
                    fields=primary_fields
                )

            primary_start, primary_end = \
                primary_data_cache.get_range("time")

            # Go through all potential collocated secondary files:
            for secondary_file in secondary_files:
                print("Secondary:", secondary_file)

                # Only read in this secondary if you have not read it already
                # the last time!
                if last_secondary is None or last_secondary != secondary:
                    secondary_data_cache = secondary.read(
                        secondary_file,
                        fields=secondary_fields
                    )

                    # Save this secondary as last cached secondary
                    last_secondary = secondary_file
                else:
                    print("This secondary is still in cache.")

                # Select only the time period in the secondary data where also
                # primary data is available.
                if last_timestamp is None:
                    indices = \
                        (secondary_data_cache["time"] >= primary_start) \
                        & (secondary_data_cache["time"] <= primary_end)
                else:
                    # May be we checked a part of this time period already?
                    # Hence, start there where we stopped last time.
                    indices = \
                        (secondary_data_cache["time"] >= last_timestamp) \
                        & (secondary_data_cache["time"] <= primary_end)

                secondary_data = secondary_data_cache[indices]

                if not secondary_data:
                    print("Skip this file.")
                    continue

                # The "new" start and end times of the secondary data (the time
                # conversion is a little bit tricky due to a bug in numpy).
                secondary_start, secondary_end = \
                    secondary_data.get_range("time")

                # Select only the primary data that is in the same time range
                # as the secondary data.
                indices = (primary_data_cache["time"] >= secondary_start) \
                          & (primary_data_cache["time"] <= secondary_end)
                primary_data = primary_data_cache[indices]

                if isinstance(primary, CollocatedDataset):
                    primary_full_data = primary_full_data_cache[indices]

                # We are going to check the data until secondary_end. Hence,
                # we save this timestamp to avoid duplicates the next time.
                last_timestamp = secondary_end

                # No collocations have been found.
                if not primary_data:
                    print("Skip this file.")
                    continue

                print("Search for collocations between %s and %s." % (
                    secondary_start, secondary_end))

                print("\tTaking {} {} and {} {} points for collocation "
                      "search.".format(
                        len(primary_data), primary.name,
                        len(secondary_data), secondary.name,
                ))

                # Find the collocations
                primary_indices, primary_collocations, \
                    secondary_indices, secondary_collocations = \
                    dataset.collocate(
                        primary_data, secondary_data,
                        max_interval, max_distance, processes,
                        check_temporal_boundaries
                    )

                # Found no collocations
                if not primary_indices:
                    continue

                # Store the collocated data into a file:
                dataset._store_collocated_data(
                    datasets=[primary, secondary],
                    data_list=[primary_data, secondary_data],
                    original_indices=[
                        primary_indices,
                        secondary_indices
                    ],
                    collocations=[
                        primary_collocations,
                        secondary_collocations],
                    original_files=[
                        primary_file, secondary_file
                    ],
                    max_interval=max_interval, max_distance=max_distance
                )

                total_primaries_points += len(primary_indices)
                total_secondaries_points += len(secondary_indices)

        print("Needed {0:.2f} seconds to find {1} ({2}) and {3} ({4}) "
              "collocation points for a period of {5} hours.".format(
                time.time() - timer, total_primaries_points,
                primary.name, total_secondaries_points,
                secondary.name,
                end - start))

        return dataset

    def _store_collocated_data(
            self, datasets, data_list, original_indices, collocations,
            original_files, max_interval, max_distance):
        """Merge the data, original indices, collocation indices and
        additional information of the datasets to one GeoData object.

        Args:
            data_list:
            collocations:
            original_files:
            original_indices:
            max_interval:
            max_distance:

        Returns:
            A GeoData object.
        """
        collocated_data = GeoData(name="CollocatedData")
        collocated_data.attrs["max_interval"] = \
            "Max. interval in seconds: %d" % max_interval
        collocated_data.attrs["max_interval"] = \
            "Max. distance in kilometers: %d" % max_distance
        collocated_data.attrs["primary_dataset"] = datasets[0].name
        for i, all_data in enumerate(data_list):
            data = all_data[original_indices[i]]
            data["__collocations"] = collocations[i]
            data["__original_indices"] = original_indices[i]
            data.attrs["original_file"] = original_files[i]
            collocated_data[datasets[i].name] = data

        collocations_start, collocations_end = \
            collocated_data.get_range("time", deep=True)
        collocated_data.attrs["start_time"] = \
            collocations_start.strftime("%Y-%m-%dT%H:%M:%S.%f")
        collocated_data.attrs["end_time"] = \
            collocations_end.strftime("%Y-%m-%dT%H:%M:%S.%f")

        # Prepare the name for the output file:
        filename = self.generate_filename(
            self.files,
            collocations_start,
            collocations_end
        )

        # Write the data to the file.
        print("\tWrite collocations to '{0}'".format(filename))
        self.write(filename, collocated_data)
