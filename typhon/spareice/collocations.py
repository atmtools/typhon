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
except ImportError:
    pass

import numpy as np
import scipy.stats
from sklearn.neighbors import BallTree
import typhon.geodesy
from typhon.spareice.array import ArrayGroup
from typhon.spareice.datasets import Dataset
from typhon.spareice.geographical import GeoData

__all__ = [
    "CollocatedDataset",
    "NotCollapsedError",
]


class NotCollapsedError(Exception):
    """Should be raised if a file from a CollocatedDataset object is not yet
    collapsed but it is required.
    """
    def __init__(self, *args):
        Exception.__init__(self, *args)


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
        :meth:`create_from` method.

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

        # The data of secondary files can overlap multiple primary files.
        self._last_file1 = None
        self._last_file2 = None
        self._file1_data = None
        self._file2_data = None

        # Use this variable to store the point in time to which the were
        # checked. This avoids checking duplicates.
        self._last_timestamp = None

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

    def collapse(self, start, end, output, reference,
                 collapser=None, include_stats=None, **mapping_args):
        """Collapses all multiple collocation points (collocations that refer
        to the same point from another dataset) to a single data point.

        During searching for collocations, one might find multiple collocation
        points from one dataset for one single point of the other dataset. For
        example, the MHS instrument has a larger footprint than the AVHRR
        instrument, hence one will find several AVHRR colloocation points for
        each MHS data point. This method performs a function on the multiple
        collocation points to merge them to one single point (e.g. the mean
        function).

        Args:
            start: Starting date as datetime object.
            end: Ending date as datetime object.
            output: Dataset object where the collapsed data should be stored.
            reference: Name of dataset which has the largest footprint. All
                other datasets will be collapsed to its data points.
            collapser: Function that should be applied on each bin (
                numpy.nanmean is the default).
            include_stats: Set this to a name of a variable (or list of
                names) and statistical parameters will be stored about the
                built data bins of the variable before collapsing. The variable
                must be one-dimensional.
            **mapping_args: Additional keyword arguments that are allowed
                for :meth:`Dataset.map_content` method (except *output*).

        Returns:
            None

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

        if not isinstance(output, Dataset):
            raise ValueError("The argument output must be a Dataset object!")
        mapping_args["output"] = output

        func_args = {
            "reference": reference,
            "include_stats": include_stats,
            "collapser": collapser,
        }

        self.map_content(
            start, end, CollocatedDataset.collapse_data,
            func_args, **mapping_args,
        )

    @staticmethod
    def collapse_data(
            collocated_data, reference, include_stats, collapser):
        """TODO: Write documentation."""

        # Get the bin indices by the main dataset to which all other
        # shall be collapsed:
        reference_bins = list(
            collocated_data[reference]["__collocations"].group().values()
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

            # Create the bins for the varaible from which you want to have
            # the statistics:
            group, _ = ArrayGroup.parse(include_stats)
            bins = collocated_data[group]["__collocations"].bin(
                reference_bins
            )
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

            if (dataset == reference
                or collocated_data[dataset].attrs.get("COLLAPSED_TO", None)
                    == reference):
                # This is the main dataset to which all other will be
                # collapsed. Therefore, we do not need explicitly
                # collapse here.
                collapsed_data[dataset] = \
                    collocated_data[dataset][np.unique(collocations)]
            else:
                bins = collocations.bin(reference_bins)
                collapsed_data[dataset] = \
                    collocated_data[dataset].collapse(
                        bins, collapser=collapser,
                    )

                collapsed_data[dataset].attrs["COLLAPSED_TO"] = reference

        # Set the collapsed flag:
        collapsed_data.attrs["COLLAPSED"] = 1

        # Overwrite the content of the old file:
        return collapsed_data

    @staticmethod
    def _collapse_file(
            collocated_dataset, filename, _,
            reference, include_stats, **collapsing_args):

        collocated_data = collocated_dataset.read(filename)
        collapsed_data = CollocatedDataset.collapse_data(
            collocated_data, reference, include_stats, **collapsing_args
        )

        # Overwrite the content of the old file:
        collocated_dataset.write(filename, collapsed_data)

    def collocate(
            self, primary, secondary, start, end,
            fields, max_interval=None, max_distance=None, optimizer=None,
            processes=4):
        """Finds all collocations between two datasets and store them in files.

        This takes all files from the datasets between two dates and find
        collocations of their data points. Where and how the collocated data is
        stored is controlled by the *files* and *handler* parameter.

        If you want to collocate multiple datasets (more than two), you can
        give as primary or secondary dataset also a CollocatedDataset. Please
        note, that the data in the CollocatedDataset have to be collapsed
        before (via the :meth:`collapse` method).

        Each collocation output file provides these standard fields:

        *dataset_name/lat* - Latitudes of the collocations.
        *dataset_name/lon* - Longitude of the collocations.
        *dataset_name/time* - Timestamp of the collocations.
        *dataset_name/__original_indices* - Indices of the collocation data in
            the original files.
        *dataset_name/__collocations* - Tells you which data points collocate
            with each other by giving their indices.

        Since this class inherits from the
        :class:`typhon.spareice.datasets.Dataset` class, please look also at
        its documentation.

        TODO: Revise and extend documentation. Maybe rename this method?

        Args:
            primary: A Dataset or CollocatedDataset object.
            secondary: A Dataset or CollocatedDataset object.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as *start*.
            fields: The fields that should be extracted from the datasets and
                copied to the new collocation files. This should be a list
                of two lists containing the field names for the primary and
                secondary dataset. Please note if one dataset is a
                CollocatedDataset object, its field selection will be ignored.
                If you want to learn more about how to extract specific
                dimension from those fields, please read the documentation of
                the typhon.handlers classes.
            max_interval: (optional) The maximum interval of time between two
                data points in seconds. Default is 300 seconds. If this is
                None, the data will be searched for spatial collocations only.
            max_distance: (optional) The maximum distance between two data
                points in kilometers to meet the collocation criteria. Default
                is 10km.
            optimizer: (optional) Finding of collocations can sometimes be
                optimized by giving here a Optimizer object.
            processes: The number of processes that should be used to boost the
                collocation process. The optimal number of processes heavily
                depends on your machine where you are working. I recommend to
                start with 8 processes and to in/decrease this parameter
                when lacking performance.

        Returns:
            None

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
                "dir/{year}/{doy}/{year}{doy}{hour}{minute}{second}_*.hdf.zip",
                handler=cloudsat.C2CICE(),
                name="2C-ICE",
            )

            # Create the collocated dataset. The parameters "files" and
            # "file_handler" tells the object where the collocations should
            be stored and which format should be used.
            collocated_dataset = CollocatedDataset.create_from(
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
                # another file handler object here:
                handler=common.NetCDF4(),
            )

            # You can treat the collocated_dataset as a normal Dataset object.
            found_files = collocated_dataset.find_files(
                "2007-01-01", "2007-01-02", sort=True
            )
            for file, time_coverage in found_files:
                print("File:", file)
        """
        self.primary_dataset = primary.name

        # start and end can be string objects:
        start = self._to_datetime(start)
        end = self._to_datetime(end)

        if max_interval is not None \
                and not isinstance(max_interval, timedelta):
            max_interval = timedelta(seconds=max_interval)

        total_primaries_points, total_secondaries_points = 0, 0

        # Use a timer for profiling.
        timer = time.time()

        # Go through all primary files and find the secondaries to them:
        for primary_file, secondary_files in primary.find_overlapping_files(
                start, end, secondary, max_interval):
            print("Primary:", primary_file)

            # Skip this primary file if there are no secondaries found.
            if not secondary_files:
                print("\tNo secondaries were found for this primary file!")
                continue

            # Go through all potential collocated secondary files:
            for secondary_file in secondary_files:
                print("Secondary:", secondary_file)

                # Find the collocations in those files and save the results:
                primary_points, secondary_points = self._collocate_files(
                    primary, secondary, primary_file, secondary_file,
                    start, end, fields,
                    max_interval, max_distance, processes
                )

                total_primaries_points += primary_points
                total_secondaries_points += secondary_points

        print("Needed {0:.2f} seconds to find {1} ({2}) and {3} ({4}) "
              "collocation points for a period of {5} hours.".format(
                time.time() - timer, total_primaries_points,
                primary.name, total_secondaries_points,
                secondary.name,
                end - start)
        )

    def _collocate_files(
            self, dataset1, dataset2, file1, file2, start_user, end_user,
            fields, max_interval, max_distance, processes
    ):
        """Find the collocations between two files and store them in a file of
        this dataset.

        Args:
            dataset1:
            dataset2:
            file1:
            file2:
            max_interval:
            max_distance:
            processes:

        Returns:
            The number of the found collocations of the first and second
            dataset.
        """

        # All additional fields given by the user and the standard fields that
        # we need for finding collocations.
        fields1 = list(set(fields[0]) | {"time", "lat", "lon"})
        fields2 = list(set(fields[1]) | {"time", "lat", "lon"})

        # TODO: This is duplicated code for each dataset. Fix this.
        # We do want to open the same file twice. Hence, have a look whether we
        # have the needed file still in our cache:
        if self._last_file1 is None or self._last_file1 != file1:
            if isinstance(dataset1, CollocatedDataset):
                # The data comes from a collocated dataset, i.e. we pick its
                # internal primary dataset for finding collocations. After
                # collocating we use the original indices and copy also
                # its other datasets to the newly created files.
                self._file1_data_all = dataset1.read(file1).as_type(GeoData)

                # The whole collocation routine does not work with non
                # collapsed data.
                if "COLLAPSED" not in self._file1_data_all.attrs:
                    raise NotCollapsedError(
                        "I cannot proceed since the dataset '{}' is not "
                        "collapsed. Use the method 'collapse' on that dataset "
                        "first.".format(dataset1.name)
                    )

                if dataset1.primary_dataset is not None:
                    # Very good, the user set a primary dataset by themselves.
                    pass
                elif "primary_dataset" in self._file1_data_all.attrs:
                    # There is a primary dataset flag in the file.
                    dataset1.primary_dataset = \
                        self._file1_data_all.attrs["primary_dataset"]
                else:
                    raise AttributeError(
                        "There is no primary dataset set in '{0}'! I do not "
                        "know which dataset to use for collocating. You can"
                        "set this via the primary_dataset attribute from "
                        "'{0}'.".format(dataset1.name))

                # Select only the dataset from the collocated data that we want
                # to use for finding the collocations.
                self._file1_data = \
                    self._file1_data_all[dataset1.primary_dataset]
            else:
                self._file1_data = dataset1.read(
                    file1, fields=fields1).as_type(GeoData)
                self._file1_data_all = None

            self._last_file1 = file1

        if self._last_file2 is None or self._last_file2 != file2:
            if isinstance(dataset2, CollocatedDataset):
                # The data comes from a collocated dataset, i.e. we pick its
                # internal primary dataset for finding collocations. After
                # collocating we use the original indices and copy also
                # its other datasets to the newly created files.
                self._file2_data_all = dataset2.read(file1).as_type(GeoData)

                # The whole collocation routine does not work with non
                # collapsed data.
                if "COLLAPSED" not in self._file2_data_all.attrs:
                    raise NotCollapsedError(
                        "I cannot proceed since the dataset '{}' is not "
                        "collapsed. Use the method 'collapse' on that dataset "
                        "first.".format(dataset2.name)
                    )

                if dataset2.primary_dataset is not None:
                    # Very good, the user set a primary dataset by themselves.
                    pass
                elif "primary_dataset" in self._file2_data_all.attrs:
                    # There is a primary dataset flag in the file.
                    dataset2.primary_dataset = \
                        self._file2_data_all.attrs["primary_dataset"]
                else:
                    raise AttributeError(
                        "There is no primary dataset set in '{0}'! I do not "
                        "know which dataset to use for collocating. You can"
                        "set this via the primary_dataset attribute from "
                        "'{0}'.".format(dataset1.name))

                # Select only the dataset from the collocated data that we want
                # to use for finding the collocations.
                self._file2_data = \
                    self._file2_data_all[dataset2.primary_dataset]
            else:
                self._file2_data = dataset2.read(
                    file2, fields=fields2).as_type(GeoData)
                self._file2_data_all = None
            self._last_file2 = file2

        # Firstly, we start by selecting only the time period where both
        # datasets have data and that lies in the time period requested by the
        # user.
        time_indices1, time_indices2 = \
            self._select_common_time_period(start_user, end_user, max_interval)

        if time_indices1 is None:
            # There was no common time window found
            return 0, 0

        print("\tTaking {} {} and {} {} points for collocation search.".format(
                np.sum(time_indices1), dataset1.name,
                np.sum(time_indices2), dataset2.name,
            )
        )

        # Select the relevant data:
        data1 = self._file1_data[time_indices1]
        data2 = self._file2_data[time_indices2]

        # Get the offset between the original data and the selected data.
        offset1 = np.where(time_indices1)[0][0]
        offset2 = np.where(time_indices2)[0][0]

        # Secondly, find the collocations.
        indices1, indices2, collocations1, collocations2 = \
            self.find_collocations(
                data1, data2,
                max_interval, max_distance, processes
            )

        # Found no collocations
        if not indices1:
            return 0, 0

        # These are the indices of the points in the original data that have
        # collocations.
        original_indices1 = np.asarray(indices1) + offset1
        original_indices2 = np.asarray(indices2) + offset2

        # Store the collocated data into a file:
        self._store_collocated_data(
            datasets=[dataset1, dataset2],
            original_indices=[original_indices1, original_indices2],
            collocations=[collocations1, collocations2],
            original_files=[file1, file2],
            max_interval=max_interval, max_distance=max_distance
        )

        return len(indices1), len(indices2)

    @staticmethod
    def find_collocations(
            primary_data, secondary_data,
            max_interval=300, max_distance=10,
            processes=8):
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
                object. If this is None, the data will be searched for
                spatial collocations only.
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria.
            processes: The number of processes that should be used to boost the
                collocation process. The optimal number of processes heavily
                depends on your machine where you are working. I recommend to
                start with 8 processes and to in/decrease this parameter
                when lacking performance.

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
                etc.
            (4) secondary collocations - Same as (2) but containing the
                collocations of the secondary data.
        """

        timer = time.time()

        max_interval = Dataset._to_timedelta(max_interval)

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

        primary_points = np.vstack([x, y, z]).T

        # We need to convert the secondary data as well:
        x, y, z = typhon.geodesy.geocentric2cart(
            typhon.constants.earth_radius,
            secondary_data["lat"],
            secondary_data["lon"]
        )
        secondary_points = np.vstack([x, y, z]).T

        print("\tAfter {:.2f}s: finished the conversion from longitude"
              "/latitudes to cartesian.".format(
            time.time() - timer))

        # TODO: Using multiple processes is deprecated so far. How to implement
        # TODO: them?
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

        # Prepare the k-d tree for the primary dataset already here, to save
        # the building time (in case of using multiple processes).


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
            0, primary_points, secondary_points, max_distance
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
        if max_interval is not None:
            # TODO: Put this into a function to make this replaceable by an
            # TODO: optimizer.

            # Convert them from datetime objects to timestamps (seconds since
            # 1970-1-1) to make them faster for temporal distance checking.
            primary_time = primary_data["time"]
            secondary_time = secondary_data["time"]

            # Check whether the time differences between the spatial
            # collocations are less than the temporal boundary:
            collocation_indices = np.abs(
                primary_time[primary_collocation_indices]
                - secondary_time[secondary_collocation_indices]
            ) < max_interval

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

        return primary_indices, secondary_indices, \
            primary_collocations, secondary_collocations

    @staticmethod
    def _find_spatial_collocations(
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

        timer = time.time()
        tree = BallTree(secondary_points, leaf_size=15)
        print("\tNeeded {:.2f}s: for building the tree.".format(
            time.time() - timer))

        # Search for all collocations. This returns a list of lists. The index
        # of each element is the index of the primary data and each element
        # is a list of the indices of the collocated secondary data.
        # The parameter max_distance is in kilometers, so we have to convert it
        # to meters.
        timer = time.time()
        collocation_indices = tree.query_radius(
            primary_points, r=max_distance*1000)
        print("\tNeeded {:.2f}s for finding spatial collocations".format(
            time.time() - timer))

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

    def get_original_files(self, start, end):
        for data in self.read_period(start, end):
            files = []
            for dataset in data.groups():
                files.append(data[dataset].attrs["original_file"])
            yield files

    def _select_common_time_period(self, start_user, end_user, max_interval):
        """Selects only the time window where both datasets have data. Sets
        also the last timestamp marker.

        Returns:
            Two lists. Each contains the selected indices for the primary or
            second dataset file, respectively.
        """

        # TODO: Using max_interval we might have duplicates in the data.
        # TODO: Filter them out.

        start1, end1 = \
            self._file1_data.get_range("time")

        start1 -= max_interval
        end1 += max_interval

        # Find no collocations outside from the time period given by the user.
        if start1 < start_user:
            start1 = start_user
        if end1 > end_user:
            end1 = end_user

        if self._last_timestamp is None:
            indices2 = \
                (self._file2_data["time"] >= start1) \
                & (self._file2_data["time"] <= end1)
        else:
            # May be we checked a part of this time period already?
            # Hence, start there where we stopped last time.
            indices2 = (
                (self._file2_data["time"] >= self._last_timestamp-max_interval)
                & (self._file2_data["time"] <= end1))

        # Maybe there is no overlapping between those files?
        if not indices2.any():
            return None, None

        # The "new" start and end times of the secondary data.
        start2, end2 = \
            self._file2_data[indices2].get_range("time")

        # Select only the primary data that is in the same time range
        # as the secondary data.
        indices1 = \
            (self._file1_data["time"] >= start2) \
            & (self._file1_data["time"] <= end2)

        # Maybe there is no overlapping between those files?
        if not indices1.any():
            return None, None

        # We save this timestamp to avoid duplicates the next time.
        self._last_timestamp = end2

        print("Search for collocations between %s and %s." % (
            start2, end2))

        return indices1, indices2

    def _store_collocated_data(
            self, datasets, original_indices, collocations,
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
            "Max. interval in seconds: %d" % max_interval.total_seconds()
        collocated_data.attrs["max_distance"] = \
            "Max. distance in kilometers: %d" % max_distance
        collocated_data.attrs["primary_dataset"] = datasets[0].name

        data_list = [self._file1_data, self._file2_data]
        # If a dataset is a CollocatedDataset, then we have to store also its
        # other collocated datasets:
        data_all_list = [self._file1_data_all, self._file2_data_all]

        for i, dataset in enumerate(datasets):
            if isinstance(dataset, CollocatedDataset):
                # This dataset contains multiple already-collocated datasets.
                for group in data_all_list[i].groups(exclude_prefix="__"):
                    data = self._prepare_data(
                        data_all_list[i][group][original_indices[i]],
                        collocations[i],
                        original_indices[i],
                        data_all_list[i][group].attrs["original_file"]
                    )

                    collocated_data[group] = data
            else:
                data = self._prepare_data(
                    data_list[i][original_indices[i]],
                    collocations[i],
                    original_indices[i],
                    original_files[i]
                )
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

    @staticmethod
    def _prepare_data(data, collocations, original_indices, original_file):
        data["__collocations"] = collocations
        data["__original_indices"] = original_indices
        data.attrs["original_file"] = original_file
        return data