"""
This module contains classes to find collocations between datasets. They are
inspired by the CollocatedDataset classes in atmlab implemented by Gerrit Holl.

TODO: I would like to have this package as typhon.collocations.

Created by John Mrziglod, June 2017
"""

from datetime import datetime, timedelta
import logging
import time
import traceback

try:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
except ImportError:
    pass

import numpy as np
import pandas as pd
import scipy.stats
from typhon.spareice.array import Array, ArrayGroup
from typhon.spareice.datasets import Dataset
from typhon.spareice.geographical import GeoData
from typhon.utils.time import to_datetime, to_timedelta

from .algorithms import BallTree, BruteForce

__all__ = [
    "CollocatedDataset",
    "CollocationsFinder",
    "NotCollapsedError",
]


COLLOCATION_FIELD = "__collocation_ids"


class NotCollapsedError(Exception):
    """Should be raised if a file from a CollocatedDataset object is not yet
    collapsed but it is required.
    """
    def __init__(self, *args):
        Exception.__init__(self, *args)


class CollocatedDataset(Dataset):
    """Still under development.

    A dataset that stores collocations that were found by :class:`Collocator`
    amongst different datasets with geographical data.
    """

    def __init__(self, *args, primary_dataset=None, **kwargs):
        """Opens existing files with collocated data as a CollocatedDataset
        object.

        If you have already collocated some datasets, you can open the stored
        collocations with this command.

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
        if primary_dataset is None:
            self.primary_dataset = "/"
        else:
            self.primary_dataset = primary_dataset

    @staticmethod
    def _add_fields_to_data(data, original_dataset, group, fields):
        try:
            original_file = data[group].attrs["original_file"]
        except KeyError:
            raise KeyError(
                "The collocation files does not contain information about "
                "their original files.")
        original_data = original_dataset.read(original_file, fields=fields)
        original_indices = data[group]["__original_indices"]
        data[group] = ArrayGroup.merge(
            [data[group], original_data[original_indices]],
            overwrite_error=False
        )

        return data

    def add_fields(self, start, end, original_dataset, group, fields):
        """

        Args:
            start:
            end:
            original_dataset:
            group
            fields:

        Returns:
            None
        """
        self.map_content(
            start, end, self._add_fields_to_data, {
                "group": group,
                "original_dataset": original_dataset,
                "fields": fields,
            }, output=self
        )

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
            collocated_data[reference][COLLOCATION_FIELD].group().values()
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
            bins = collocated_data[group][COLLOCATION_FIELD].bin(
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

            collocations = collocated_data[dataset][COLLOCATION_FIELD]

            # We do not need the original and collocation indices any
            # longer because they will soon become useless. Moreover,
            # they could have a different dimension length than the
            # other variables and lead to errors in the selecting process:
            del collocated_data[dataset]["__original_indices"]
            del collocated_data[dataset][COLLOCATION_FIELD]

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

    @classmethod
    def from_dataset(cls, dataset):
        """Transform a Dataset into a CollocatedDataset

        Args:
            dataset: A Dataset object.

        Returns:
            A CollocatedDataset object.
        """
        obj = cls()
        obj.__dict__.update(dataset.__dict__)
        return obj


class CollocationsFinder:
    """Find collocations between datasets or data arrays.

    Collocations are two or more data points that are located close to each
    other in space and/or time.

    """

    _algorithm = {
        "BallTree": BallTree(),
        "BruteForce": BruteForce(),
    }

    def __init__(self, start=None, end=None, max_interval=None,
                 max_distance=None, algorithm=None, processes=None,
                 verbose=False):
        """Initialise a CollocationsFinder object.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as *start*.
            max_interval: The maximum interval of time between two data points
                in seconds. Default is 300 seconds. If this is None, the data
                will be searched for spatial collocations only.
            max_distance: The maximum distance between two data points in
                kilometers to meet the collocation criteria. If this is None,
                the data will be searched for spatial collocations only (not
                yet implemented!).
            algorithm: Defines which algorithm should be used to find the
                collocations. Must be either a Finder object (a subclass from
                :class:`~typhon.spareice.collocations.algorithms.Algorithm`) or
                a string with the name of an algorithm. Default is the
                *BallTree* algorithm. See below for a table of available
                algorithms.
            processes: The number of processes that should be used to boost the
                collocation process. The optimal number of processes heavily
                depends on your machine where you are working. I recommend to
                start with 8 processes and to in/decrease this parameter
                when lacking performance.
            verbose: Prints debug messages if true.

        How the collocations are going to be found is specified by the used
        algorithm. The following algorithms are possible (you can use your
        own algorithm by subclassing the
        :class:`~typhon.spareice.collocations.algorithms.Algorithm` class):

        +--------------+------------------------------------------------------+
        | Algorithm    | Description                                          |
        +==============+======================================================+
        | BallTree     | (default) Uses the highly optimized Ball Tree class  |
        |              |                                                      |
        |              | from sklearn [1]_.                                   |
        +--------------+------------------------------------------------------+
        | BruteForce   | Finds the collocation by comparing each point of the |
        |              |                                                      |
        |              | dataset with each other. Should be only used for     |
        |              |                                                      |
        |              | testing purposes since it is inefficient and very    |
        |              |                                                      |
        |              | memory- and time consuming for big datasets.         |
        +--------------+------------------------------------------------------+

        .. [1] http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
        """

        if start is None:
            self.start = datetime.min
        else:
            self.start = to_datetime(start)

        if end is None:
            self.end = datetime.max
        else:
            self.end = to_datetime(end)

        if max_interval is not None:
            self.max_interval = to_timedelta(
                max_interval, numbers_as="seconds")

        if max_distance is None:
            raise NotImplementedError(
                "To find only temporal collocations is not yet implemented!"
            )

        self.max_distance = max_distance

        if algorithm is None:
            self.algorithm = BallTree()
        else:
            if isinstance(algorithm, str):
                try:
                    self.algorithm = self._algorithm[algorithm]
                except KeyError:
                    raise ValueError("Unknown algorithm: %s" % algorithm)
            else:
                self.algorithm = algorithm

        self.processes = processes
        self.verbose = verbose

    def _debug(self, msg):
        if self.verbose:
            print(msg)

    def match_datasets(self, datasets, output, fields=None,):
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

        TODO: Revise and extend documentation.

        Args:
            datasets: A list of Dataset or CollocatedDataset objects. If this
                list contains more than two elements, it starts to collocate
                the first two datasets, collapse them to the first dataset and
                then collocates the result with next the dataset, etc.
            output: Either a path as string containing placeholders or a
                Dataset-like object.
            fields: The fields that should be extracted from the datasets and
                copied to the new collocation files. This should be a list
                of lists, one for each dataset in *datasets*. The list should
                containing the field names. Please note if one dataset is a
                CollocatedDataset object, its field selection will be ignored.
                If you want to learn more about how to extract specific
                dimension from those fields, please read the documentation of
                the typhon.handlers classes.

        Returns:
            A :class:`CollocatedDataset` object holding the collocated data.

        Examples:

        .. :code-block:: python

            # TODO
        """
        # Make sure that our output dataset is a CollocatedDataset
        if isinstance(output, str):
            name = "-".join([ds.name for ds in datasets])
            output = CollocatedDataset(path=output, name=name)
        elif isinstance(output, CollocatedDataset):
            # everything ok
            pass
        elif isinstance(output, Dataset):
            output = CollocatedDataset.from_dataset(output)
        else:
            raise ValueError("The parameter output must be a string, Dataset "
                             "or CollocatedDataset object!")

        # Go through all datasets and find collocations between them
        for index, dataset in enumerate(datasets, 1):
            if index == len(datasets):
                break

            if index == 1:
                # The first two datasets will be collocated...
                self._match_two_datasets(
                    datasets[0], datasets[1],
                    output, fields[0], fields[1],
                )
            else:
                # Before we can collocate the next dataset, we need to collapse
                # the collocations from the last ones...
                output.primary_dataset = datasets[0]
                output.collapse(
                    self.start, self.end, output, reference=datasets[0],
                    #include_stats="2C-ICE/IO_RO_ice_water_path",
                )

                self._match_two_datasets(
                    output, dataset, output, fields[index], fields[index+1]
                )

    def _match_two_datasets(
            self, primary_ds, secondary_ds, output_ds,
            primary_fields, secondary_fields):
        """

        Args:
            primary_ds:
            secondary_ds:
            output_ds:
            fields:

        Returns:
            None
        """
        # Use a timer for profiling.
        timer = time.time()

        self._debug(
                "Find collocations between {} and {} from {} to {}".format(
                    primary_ds.name, secondary_ds.name, self.start, self.end,
                ))

        self._debug("Retrieve time coverages from files...")

        # Go through all primary files and find the secondaries to them:
        overlaps = primary_ds.overlaps_with(secondary_ds, self.start, self.end)

        # We can flush this into one list because Dataset did this already by
        # itself
        file_pairs = [
            (primary, secondary)
            for primary, secondaries in overlaps
            for secondary in secondaries
        ]

        total_primaries_points, total_secondaries_points = 0, 0
        last_primary, last_primary_end_time = None, None
        last_secondary, last_secondary_end_time = None, None

        for i, file_pair in enumerate(file_pairs):
            self._debug_collocation_status(
                primary_ds, secondary_ds, timer, file_pairs, i
            )

            primary, secondary = file_pair

            # To avoid multiple reading of the same file, we cache their
            # content.
            try:
                self._debug("Load next primary from:")
                if last_primary is None or last_primary != primary:
                    self._debug("  %s" % primary)
                    primary_cache, primary_data = self._read_input_file(
                        primary_ds, primary, primary_fields
                    )
                    last_primary = primary
                else:
                    self._debug("  Cache")

                self._debug("Load next secondary from:")
                if last_secondary is None or last_secondary != secondary:
                    self._debug("  %s" % secondary)
                    secondary_cache, secondary_data = self._read_input_file(
                        secondary_ds, secondary, secondary_fields
                    )
                    last_secondary = secondary
                else:
                    self._debug("  Cache")
            except Exception as err:
                self._debug(
                    "The search in this time period failed due to an error!")
                traceback.print_exc()
                self._debug("-" * 79)

            # TODO: Filter out duplicates (overlapping between files from the
            # TODO: same dataset)

            # Find the collocations in those data arrays:
            collocations = self.match_arrays(primary_data, secondary_data)

            if not collocations.any():
                self._debug("Found no collocations!")

                continue

            # Store the collocated data to the output dataset:
            number_of_collocations = self._store_collocations(
                output_ds,
                datasets=[primary_ds, secondary_ds],
                data_list=[primary_cache, secondary_cache],
                collocations=collocations,
                original_files=[primary, secondary],
            )

            total_primaries_points += number_of_collocations[0]
            total_secondaries_points += number_of_collocations[1]

        self._debug(
            "Needed {0:.2f} seconds to find {1} ({2}) and {3} ({4}) "
            "collocation points for a period of {5} hours.".format(
                time.time() - timer, total_primaries_points,
                primary_ds.name, total_secondaries_points,
                secondary_ds.name, self.end - self.start)
        )

    def _debug_collocation_status(
            self, primary_ds, secondary_ds, timer, file_pairs, i):
        if i == 0:
            expected_time = "unknown"
        else:
            elapsed_time = time.time()-timer
            expected_time = timedelta(
                seconds=int(elapsed_time/i * len(file_pairs) - elapsed_time)
            )

        self._debug("-" * 79)
        self._debug(
            f"Collocating {primary_ds.name} to {secondary_ds.name}: "
            f"{100*i/len(file_pairs):.2f}% processed "
            f"({expected_time} hours remaining)"
        )

    def _read_input_file(self, dataset, file, fields):
        """Read a file from a dataset and extract fields

        Args:
            dataset:
            file:
            fields:

        Returns:
            Two ArrayGroups: The first contains all fields extracted from the
            file, the second contains only the time, lat and lon information.
        """

        # All additional fields given by the user and the standard fields that
        # we need for finding collocations.
        fields = list(set(fields) | {"time", "lat", "lon"})

        if isinstance(dataset, CollocatedDataset):
            # The data comes from a collocated dataset, i.e. we pick its
            # internal primary dataset for finding collocations. After
            # collocating we use the original indices and copy also
            # its other datasets to the newly created files.
            data = dataset.read(file).as_type(GeoData)

            # The whole collocation routine does not work with non
            # collapsed data.
            if "COLLAPSED" not in data.attrs:
                raise NotCollapsedError(
                    "I cannot proceed since the dataset '{}' is not "
                    "collapsed. Use the method 'collapse' on that dataset "
                    "first.".format(dataset.name)
                )

            if dataset.primary_dataset is not None:
                # Very good, the user set a primary dataset by themselves.
                pass
            elif "primary_dataset" in data.attrs:
                # There is a primary dataset flag in the file.
                dataset.primary_dataset = \
                    data.attrs["primary_dataset"]
            else:
                raise AttributeError(
                    "There is no primary dataset set in '{0}'! I do not "
                    "know which dataset to use for collocating. You can"
                    "set this via the primary_dataset attribute from "
                    "'{0}'.".format(dataset.name))

            return data, data[dataset.primary_dataset][("time", "lat", "lon")]
        else:
            data = dataset.read(file, fields=fields).as_type(GeoData)
            return data, data[("time", "lat", "lon")]

    def _store_collocations(
            self, output, datasets, data_list, collocations, original_files,):
        """Merge the data, original indices, collocation indices and
        additional information of the datasets to one GeoData object.

        Args:
            output:
            datasets:
            data:
            collocations:
            original_files:

        Returns:
            List with number of collocations
        """
        collocated_data = GeoData(name="CollocatedData")
        collocated_data.attrs["max_interval"] = \
            "Max. interval in secs: {}".format(
                "None" if self.max_interval is None
                else self.max_interval.total_seconds()
            )
        collocated_data.attrs["max_distance"] = \
            "Max. distance in kilometers: %d" % self.max_distance
        collocated_data.attrs["primary_dataset"] = datasets[0].name

        number_of_collocations = []

        # If a dataset is a CollocatedDataset, then we have to store also its
        # other collocated datasets:
        for i, dataset in enumerate(datasets):
            # These are the indices of the points in the original data that
            # have collocations. Remove the duplicates since we want to copy
            # the required data only once:
            original_indices = pd.unique(collocations[i])

            # After selecting the collocated data, the original indices cannot
            # be applied any longer. We need new indices that indicate the
            # pairs in the collocated data.
            indices_in_collocated_data = {
                original_index: new_index
                for new_index, original_index in enumerate(original_indices)
            }
            collocation_indices = [
                indices_in_collocated_data[index]
                for index in collocations[i]
            ]

            number_of_collocations.append(len(original_indices))

            if isinstance(dataset, CollocatedDataset):
                # This dataset contains multiple already-collocated datasets.
                for group in data_list[i].groups(exclude_prefix="__"):
                    data = data_list[i][group][original_indices]
                    # Change the collocation ids
                    data["__collocation_ids"] = collocation_indices
                    collocated_data[group] = data
            else:
                data = data_list[i][original_indices]
                data["__collocation_ids"] = collocation_indices
                data.attrs["original_file"] = original_files[i].path

                # We want to give each point a specific id, which we can use
                # for retrieving additional information about this point later.
                # It is simply the starting timestamp of the file (seconds
                # since 1970) and the original index.
                timestamp = original_files[i].times[0].timestamp()
                data["__file_ids"] = Array(
                    [timestamp] * number_of_collocations[i], dims=["time_id"],
                    attrs={
                        "long_name": "Starting timestamp of original file",
                        "units": "seconds since 1970-01-01T00:00:00"
                    }
                )
                data["__original_indices"] = Array(
                    original_indices, dims=["time_id", ],
                    attrs={
                        "long_name": "Index in the original file",
                    }
                )

                collocated_data[datasets[i].name] = data

        time_coverage = collocated_data.get_range("time", deep=True)
        collocated_data.attrs["start_time"] = \
            time_coverage[0].strftime("%Y-%m-%dT%H:%M:%S.%f")
        collocated_data.attrs["end_time"] = \
            time_coverage[1].strftime("%Y-%m-%dT%H:%M:%S.%f")

        # Prepare the name for the output file:
        filename = output.generate_filename(time_coverage)

        self._debug("Store {} ({}) and {} ({}) collocations in\n  {}".format(
            number_of_collocations[0], datasets[0].name,
            number_of_collocations[1], datasets[1].name,
            filename
        ))

        # Write the data to the file.
        output.write(filename, collocated_data)

        return number_of_collocations

    def match_arrays(self, data1, data2,):
        """Find the collocations between two data arrays

        A data array must be dictionary-like providing the fields *time*,
        *lat*, *lon*. Its values must be 1-dimensional numpy.array-like
        objects and share the same length. See below for examples.

        Args:
            data1: A data array that fulfills the specifications from above.
            data2: A data array that fulfills the specifications from above.

        Returns:
            A dictionary with following keys and values:
            TODO

        Examples:
            TODO
        """
        # TODO: We could bin all data along the time coordinates into periods
        # TODO: that have the duration of max_interval. Then we only find for
        # TODO: collocations that are in neighbour bins (we could parallelise
        # TODO: this).

        # Firstly, we start by selecting only the time period where both data
        # arrays have data and that lies in the time period requested by the
        # user.
        time_indices = self._select_common_time_period(
            data1["time"], data2["time"],
        )

        if time_indices is None:
            # There was no common time window found
            return np.array([[], []])

        # Get the offsets between the original data and the selected data.
        offset1 = np.where(time_indices[0])[0][0]
        offset2 = np.where(time_indices[1])[0][0]

        # Select the relevant data:
        data1 = data1[time_indices[0]]
        data2 = data2[time_indices[1]]

        # Secondly, find the collocations.
        pairs = self.algorithm.find_collocations(
            data1, data2, self.max_interval, self.max_distance,
            processes=self.processes
        )

        # No collocations were found.
        if not pairs.any():
            return pairs

        # We selected a common time window and cut off a part in the beginning,
        # do you remember? Now we correct the indices so that they point again
        # to the real original data.
        pairs[0] += offset1
        pairs[1] += offset2

        return pairs

    def _select_common_time_period(self, times1, times2,):
        """Selects only the time window where both time series have data

        Returns:
            Two lists. Each contains the selected indices for the primary or
            second dataset file, respectively.
        """
        # numpy arrays do not work with native python timedelta objects
        max_interval = np.timedelta64(self.max_interval)

        common_start = max(self.start,
                           times1.min() - max_interval,
                           times2.min() - max_interval)
        common_end = min(self.end,
                         times1.max() + max_interval,
                         times2.max() + max_interval)
        indices1 = (times1 >= common_start) & (times1 <= common_end)

        if not indices1.any():
            return None

        indices2 = (times2 >= common_start) & (times2 <= common_end)

        if not indices2.any():
            return None

        return indices1, indices2
