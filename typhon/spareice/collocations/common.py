"""
This module contains classes to find collocations between datasets. They are
inspired by the CollocatedDataset classes in atmlab implemented by Gerrit Holl.

TODO: I would like to have this package as typhon.collocations.

Created by John Mrziglod, June 2017
"""

from datetime import datetime, timedelta
import logging
import time
from multiprocessing.pool import ThreadPool
import traceback
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from typhon.math import cantor_pairing
from typhon.spareice.array import Array, GroupedArrays
from typhon.spareice.datasets import Dataset, DataSlider
from typhon.utils.time import to_datetime, to_timedelta

from .algorithms import BallTree, BruteForce

__all__ = [
    "collocate",
    "collocate_datasets",
    "CollocatedDataset",
    "NotCollapsedError",
]

# Finder algorithms for collocations:
ALGORITHM = {
    "BallTree": BallTree,
    "BruteForce": BruteForce,
}

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

    def __init__(self, *args, primary=None, **kwargs):
        """Opens existing files with collocated data as a CollocatedDataset
        object.

        If you have already collocated some datasets, you can open the stored
        collocations with this command.

        Args:
            primary: Name of the primary group.
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
        if primary is None:
            self.primary = "/"
        else:
            self.primary = primary

    @staticmethod
    def _add_fields_to_data(data, original_dataset, group, fields):
        try:
            original_file = data[group].attrs["__original_file"]
        except KeyError:
            raise KeyError(
                "The collocation files does not contain information about "
                "their original files.")
        original_data = original_dataset.read(original_file)[fields]
        original_indices = data[group]["__original_indices"]
        data[group] = GroupedArrays.merge(
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
        self.map(
            start, end, func=self._add_fields_to_data,
            kwargs={
                "group": group,
                "original_dataset": original_dataset,
                "fields": fields,
            }, on_content=True, output=self
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
                for :meth:`Dataset.map` method (except *output*).

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

        self.map(
            start, end, CollocatedDataset.collapse_data,
            kwargs=func_args, on_content=True, **mapping_args,
        )

    @staticmethod
    def collapse_data(
            collocated_data, file_info, reference, include_stats, collapser):
        """TODO: Write documentation."""

        # Get the bin indices by the main dataset to which all other
        # shall be collapsed:
        reference_bins = list(
            collocated_data[reference][COLLOCATION_FIELD].group().values()
        )

        collapsed_data = GroupedArrays()

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
            group, _ = GroupedArrays.parse(include_stats)
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

            if (dataset == reference
                or collocated_data[dataset].attrs.get("COLLAPSED_TO", None)
                    == reference):
                # The collocation indices will become useless
                del collocated_data[dataset][COLLOCATION_FIELD]

                # This is the main dataset to which all other will be
                # collapsed. Therefore, we do not need explicitly
                # collapse here.
                collapsed_data[dataset] = \
                    collocated_data[dataset][np.unique(collocations)]
            else:
                # We do not need the original and collocation indices from the
                # dataset that will be collapsed because they will soon become
                # useless. Moreover, they could have a different dimension
                # length than the other variables and lead to errors in the
                # selecting process.

                del collocated_data[dataset]["__original_indices"]
                del collocated_data[dataset][COLLOCATION_FIELD]

                bins = collocations.bin(reference_bins)

                # We ignore some warnings rather than fixing them
                # TODO: Maybe fix them?
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="invalid value encountered in double_scalars")
                    collapsed_data[dataset] = \
                        collocated_data[dataset].collapse(
                            bins, collapser=collapser,
                        )

                collapsed_data[dataset].attrs["COLLAPSED_TO"] = reference

        # Set the collapsed flag:
        collapsed_data.attrs["COLLAPSED"] = 1

        # Overwrite the content of the old file:
        return collapsed_data

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


def collocate(arrays, max_interval=None, max_distance=None,
              algorithm=None, threads=None,):
    """Find collocations between two data arrays

    Collocations are two or more data points that are located close to each
    other in space and/or time.

    A data array must be dictionary-like object (such as xarray.Dataset or
    GroupedArrays) providing the fields *time*, *lat*, *lon*. Its values must
    be 1-dimensional numpy.array-like objects and share the same length. The
    field *time* must have the data type *numpy.datetime64*, *lat* must be
    latitudes between *-90* (south) and *90* (north) and *lon* must be
    longitudes between *-180* (west) and *180* (east) degrees. See below for
    examples.

    Args:
        arrays: A list of data arrays that fulfill the specifications from
            above. So far, only collocating two arrays is implemented.
        max_interval: The maximum interval of time between two data points
            in seconds. If this is None, the data will be searched for spatial
            collocations only.
        max_distance: The maximum distance between two data points in
            kilometers to meet the collocation criteria. If this is None,
            the data will be searched for temporal collocations only. Either
            *max_interval* or *max_distance* must be given.
        algorithm: Defines which algorithm should be used to find the
            collocations. Must be either a Finder object (a subclass from
            :class:`~typhon.spareice.collocations.algorithms.Algorithm`) or
            a string with the name of an algorithm. Default is the
            *BallTree* algorithm. See below for a table of available
            algorithms.
        threads:

    Returns:
        A 2xN numpy array where N is the number of found collocations. The
        first row contains the indices of the collocations in *data1*, the
        second row the indices in *data2*.

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

    Examples:

        .. code-block: python

            import numpy as np
            from typhon.spareice import collocate

            # Create the data
            primary = {
                "time": np.arange(
                    "2018-01-01", "2018-01-02", dtype="datetime64[h]"
                ),
                "lat": 30.*np.sin(np.linspace(-3.14, 3.14, 24))+20,
                "lon": np.linspace(0, 90, 24),
            }
            secondary = {
                "time": np.arange(
                    "2018-01-01", "2018-01-02", dtype="datetime64[h]"
                ),
                "lat": 30.*np.sin(np.linspace(-3.14, 3.14, 24)+1.)+20,
                "lon": np.linspace(0, 90, 24),
            }

            # Find collocations with a maximum distance of 300 kilometers and
            # a maximum interval of 1 hour
            indices = collocate(
                [primary, secondary], max_distance=300, max_interval="1 hour")

            print(indices)  # prints [[4], [4]]


    """

    if max_distance is None and max_interval is None:
        raise ValueError("Either max_distance or max_interval must be given!")

    if len(arrays) != 2:
        raise ValueError("So far, only collocating of two arrays is allowed.")

    if max_interval is not None:
        max_interval = to_timedelta(max_interval, numbers_as="seconds")

    if algorithm is None:
        algorithm = BallTree()
    else:
        if isinstance(algorithm, str):
            try:
                algorithm = ALGORITHM[algorithm]()
            except KeyError:
                raise ValueError("Unknown algorithm: %s" % algorithm)
        else:
            algorithm = algorithm

    threads = 2 if threads is None else threads

    # If the time matters (i.e. max_interval is not None), we split the data
    # into temporal bins. This produces an overhead that is only negligible if
    # we have a lot of data:
    data_magnitude = arrays[0]["time"].shape[0] * arrays[1]["time"].shape[0]

    # We can search for spatial collocations (max_interval=None), temporal
    # collocations (max_distance=None) or both.
    if max_interval is not None and data_magnitude > 100_0000:
        # Search for temporal and temporal-spatial collocations #

        # We start by selecting only the time period where both data
        # arrays have data and that lies in the time period requested by the
        # user.
        common_time = _select_common_time(
            arrays[0]["time"], arrays[1]["time"], max_interval
        )

        if common_time is None:
            # There was no common time window found
            return np.array([[], []])

        start, end, *time_indices = common_time

        # Select the relevant data:
        # TODO: We should convert the data arrays first to an GroupedArrays or
        # TODO: xarray explicitly
        arrays[0] = arrays[0][time_indices[0]]
        arrays[1] = arrays[1][time_indices[1]]

        # We need this frequency as pandas.Timestamp because we use
        # pandas.period_range later.
        bin_size = pd.Timedelta(
            (pd.Timestamp(end) - pd.Timestamp(start)) / (4 * threads - 1)
        )

        # Now let's split the two data arrays along their time coordinate so we
        # avoid searching for spatial collocations that do not fulfill the
        # temporal condition in the first place. However, the overhead of the
        # finding algorithm must be considered too (for example the BallTree
        # creation time). We choose therefore a bin size of roughly 10'000
        # elements and minimum bin duration of max_interval.
        # The collocations that we will miss at the bin edges are going to be
        # found later.
        # TODO: Unfortunately, a first attempt parallelizing this using threads
        # TODO: worsened the performance.
        pairs_without_overlaps = np.hstack([
            _collocate_period(arrays[0], arrays[1], period)
            for period in pd.period_range(start, end, freq=bin_size)
        ])

        # Now, imagine our situation like this:
        #
        # [ PRIMARY BIN 1       ][ PRIMARY BIN 2        ]
        # ---------------------TIME--------------------->
        #   ... [ -max_interval ][ +max_interval ] ...
        # ---------------------TIME--------------------->
        # [ SECONDARY BIN 1     ][ SECONDARY BIN 2      ]
        #
        # We have already found the collocations between PRIMARY BIN 1 &
        # SECONDARY BIN 1 and PRIMARY BIN 2 & SECONDARY BIN 2. However, the
        # [-max_interval] part of PRIMARY BIN 2 might be collocated with the
        # [+max_interval] part of SECONDARY BIN 1 (the same applies to
        # [+max_interval] of the PRIMARY BIN 1 and [-max_interval] of the
        # SECONDARY BIN 2). Let's find them here:
        pairs_of_overlaps = np.hstack([
            _collocate_period(
                *arrays,
                algorithm, (max_interval, max_distance),
                pd.Period(date - max_interval, max_interval)
                if prev1_with_next2 else pd.Period(date, max_interval),
                pd.Period(date, max_interval) if prev1_with_next2
                else pd.Period(date - max_interval, max_interval),
            )
            for date in pd.date_range(start, end, freq=bin_size)[1:-1]
            for prev1_with_next2 in [True, False]
        ])

        # Put all collocations together then. Note that they are not sorted:
        pairs = np.hstack([pairs_without_overlaps, pairs_of_overlaps])

        # No collocations were found.
        if not pairs.any():
            return pairs

        # We selected a common time window and cut off a part in the beginning,
        # do you remember? Now we shift the indices so that they point again
        # to the real original data.
        pairs[0] += np.where(time_indices[0])[0][0]
        pairs[1] += np.where(time_indices[1])[0][0]
    else:
        # Search for spatial or temporal-spatial collocations but do not do any
        # pre-binning:
        pairs = algorithm.find_collocations(
            *arrays, max_distance=max_distance, max_interval=max_interval
        )

    return pairs


def _select_common_time(data1, data2, max_interval):
    common_start = np.max(
        [data1["time"].min(), data1["time"].min()]
    ) - max_interval
    common_end = np.min(
        [data1["time"].max(), data1["time"].max()]
    ) + max_interval

    # Return the indices from the data in the common time window
    indices1 = (common_start <= data1["time"]) & (data1["time"] <= common_end)
    if not indices1.any():
        return None

    indices2 = (common_start <= data2["time"]) & (data2["time"] <= common_end)
    if not indices2.any():
        return None

    return common_start, common_end, indices1, indices2


def _collocate_period(data1, data2, algorithm, algorithm_args,
                      period1, period2=None, ):
    if period2 is None:
        period2 = period1

    # Select the period
    indices1 = np.where(
        (period1.start_time < data1["time"])
        & (data1["time"] < period1.end_time)
    )[0]
    if not indices1.any():
        return np.array([[], []])

    indices2 = np.where(
        (period2.start_time < data2["time"])
        & (data2["time"] < period2.end_time)
    )[0]
    if not indices2.any():
        return np.array([[], []])

    pair_indices = algorithm.find_collocations(
        data1[indices1], data2[indices2],
        *algorithm_args,
    )

    if not pair_indices.any():
        return np.array([[], []])

    # We selected a time period, hence we must correct the found indices
    # to let them point to the original data1 and data2
    pair_indices[0] = indices1[pair_indices[0]]
    pair_indices[1] = indices2[pair_indices[1]]

    # Get also the indices where we searched in overlapping periods
    # overlapping_with_before = \
    #     np.where(
    #         (period.start_time - max_interval < data1["time"])
    #         & (data1["time"] < period.start_time)
    #     )
    # overlapping_with_after = \
    #     np.where(
    #         (period.end_time - max_interval < data1["time"])
    #         & (data1["time"] < period.end_time)
    #     )

    # Return also unique collocation ids to detect duplicates later
    return pair_indices

# TODO: Parallelizing collocate() does not work properly since threading and
# the GIL introduces a significant overhead. Maybe one should give
# multiprocessing a try but this would require pickling many (possibly huge)
# data arrays. Hence, this is so far deprecated:
# def _parallelizing():
#     # We need to decide whether we should parallelize everything by using
#     # threads or not.
#     if (time_indices[0].size > 10000 and time_indices[1].size > 10000) or
#             algorithm.loves_threading:
#         # Oh yes, let's parallelize it and create a pool of threads! Why
#         # threads instead of processes? We do not want to pickle the arrays
#         # (because they could be huge) and we trust our finding algorithm
#         # when it says it loves threading.
#         pool = ThreadPool(threads, )
#
#         # Get all overlaps (time periods where two threads search for
#         # collocations):
#         # overlap_indicess = [
#         #     [[period.start_time-max_interval, period.start_time],
#         #      [period.end_time + max_interval, period.end_time]]
#         #     for i, period in enumerate(periods)
#         #     if i > 0
#         # ]
#
#         overlapping_pairs = \
#             pool.map(_collocate_thread_period, periods)
#
#         # The search periods had overlaps. Hence the collocations contain
#         # duplicates.
#         pairs = np.hstack([
#             pairs_of_thread[0]
#             for i, pairs_of_thread in enumerate(overlapping_pairs)
#             if pairs_of_thread is not None
#         ])


def collocate_datasets(
        datasets, start=None, end=None, output=None, verbose=True,
        **collocate_args,):
    """Finds all collocations between two datasets and store them in files.

    Collocations are two or more data points that are located close to each
    other in space and/or time.

    This takes all files from the datasets between two dates and find
    collocations of their data points. Afterwards they will be stored in
    *output*.

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
        datasets: A list of Dataset or CollocatedDataset objects.
        start:
        end:
        output: Either a path as string containing placeholders or a
            Dataset-like object.
        verbose: If true, it prints logging messages.
        **collocate_args: Additional keyword arguments that are allowed for
            :func:`collocate` except *arrays*.

    Returns:
        A :class:`CollocatedDataset` object holding the collocated data.

    Examples:

    .. :code-block:: python

        # TODO
    """

    if len(datasets) != 2:
        raise ValueError("Only collocating two datasets at once is allowed"
                         "at the moment!")
    primary, secondary = datasets

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
        raise ValueError("The parameter 'output' must be a string, Dataset"
                         " or CollocatedDataset object!")

    start = datetime.min if start is None else to_datetime(start)
    end = datetime.max if end is None else to_datetime(end)

    # Use a timer for profiling.
    timer = time.time()

    if verbose:
        print(
            f"Find collocations between {primary.name} and {secondary.name} "
            f"from {start} to {end}"
        )

    total_collocations = [0, 0]

    if verbose:
        print("Retrieve time coverages from files...")

    for data, files in DataSlider(start, end, *datasets):

        primary_start, primary_end = data[primary.name].get_range("time")


        if verbose:
            _collocating_status(
                primary, secondary, timer, start, end,
                np.min([primary_end,
                        data[secondary.name]["time"].max()]).astype("O"),
            )

        # Find the collocations in those data arrays:
        collocations = collocate(
            [data[primary.name], data[secondary.name]],
            **collocate_args,
        )

        if not collocations.any():
            if verbose:
                print("Found no collocations!")
            continue

        # Store the collocated data to the output dataset:
        filename, n_collocations = _store_collocations(
            output, datasets=[primary, secondary], raw_data=data,
            collocations=collocations, files=files, **collocate_args
        )

        if verbose:
            print(
                f"Store {n_collocations[0]} ({datasets[0].name}) and "
                f"{n_collocations[1]} ({datasets[1].name}) collocations in"
                f"{filename}"
            )

        total_collocations[0] += n_collocations[0]
        total_collocations[1] += n_collocations[1]

    if verbose:
        print("-" * 79)
        print(
            f"Took {time.time()-timer:.2f} s to find {total_collocations[0]}"
            f" ({primary.name}) and {total_collocations[1]} ({secondary.name})"
            f" collocations.\nProcessed {end-start} hours of data."
        )


def _collocating_status(primary, secondary, timer, start, end, current_end):
    current = (current_end - start).total_seconds()
    progress = current / (end - start).total_seconds()

    elapsed_time = time.time()-timer
    expected_time = timedelta(
        seconds=int(elapsed_time * (1/progress - 1))
    )

    print("-" * 79)
    print(
        f"Collocating {primary.name} to {secondary.name}: {100*progress:d}% "
        f"done ({expected_time} hours remaining)"
    )


def _store_collocations(
        output, datasets, raw_data, collocations,
        files, **collocate_args):
    """Merge the data, original indices, collocation indices and
    additional information of the datasets to one GroupedArrays object.

    Args:
        output:
        datasets:
        raw_data:
        collocations:
        files:

    Returns:
        List with number of collocations
    """

    # The data that will be stored to a file:
    output_data = GroupedArrays(name="CollocatedData")

    # We need this name to store the collocation metadata in an adequate
    # group
    collocations_name = datasets[0].name+"."+datasets[1].name
    output_data["__collocations/"+collocations_name] = GroupedArrays()
    metadata = output_data["__collocations/"+collocations_name]

    max_interval = collocate_args.get("max_interval", None)
    if max_interval is not None:
        max_interval = to_timedelta(max_interval).total_seconds()
    metadata.attrs["max_interval"] = f"Max. interval in secs: {max_interval}"

    max_distance = collocate_args.get("max_distance", None)
    metadata.attrs["max_distance"] = \
        f"Max. distance in kilometers: {max_distance}"
    metadata.attrs["primary"] = datasets[0].name
    metadata.attrs["secondary"] = datasets[1].name

    pairs = []
    number_of_collocations = []

    for i, dataset in enumerate(datasets):
        dataset_data = raw_data[dataset.name]

        if "__collocations" in dataset_data.groups():
            # This dataset contains already-collocated datasets,
            # therefore we do not select any data but copy all of them.
            # This keeps the indices valid, which point to the original
            # files and data:
            output_data = GroupedArrays.merge(
                [output_data, dataset_data]
            )

            # Add the collocation indices. We do not have to adjust them
            # since we do not change the original data.
            pairs.append(collocations[i])
            continue

        # These are the indices of the points in the original data that
        # have collocations. Remove the duplicates since we want to copy
        # the required data only once:
        original_indices = pd.unique(collocations[i])

        number_of_collocations.append(len(original_indices))

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

        # Save the collocation indices in the metadata group:
        pairs.append(collocation_indices)

        data = dataset_data[original_indices]
        data["__original_indices"] = Array(
            original_indices, dims=["time_id", ],
            attrs={
                "long_name": "Index in the original file",
            }
        )

        if "__original_files" not in data:
            # Set where the data came from:
            data.attrs["__original_files"] = \
                ";".join(file.path for file in files[datasets[i].name])
        output_data[datasets[i].name] = data

    metadata["pairs"] = pairs

    # Use only the times of the primary dataset as start and end time (makes it
    # easier to find corresponding files later):
    time_coverage = output_data[datasets[0].name].get_range(
        "time",
    )
    output_data.attrs["start_time"] = \
        time_coverage[0].strftime("%Y-%m-%dT%H:%M:%S.%f")
    output_data.attrs["end_time"] = \
        time_coverage[1].strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Prepare the name for the output file:
    filename = output.generate_filename(time_coverage)

    # Write the data to the file.
    output.write(output_data, filename)

    return filename, number_of_collocations
