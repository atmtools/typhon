"""
This module contains classes to find collocations between datasets. They are
inspired by the CollocatedDataset classes in atmlab implemented by Gerrit Holl.

TODO: Move this package to typhon.collocations.

Created by John Mrziglod, June 2017
"""

from datetime import datetime, timedelta
import logging
from numbers import Number
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
from typhon.utils import split_units
from typhon.utils.time import to_datetime, to_timedelta
import xarray as xr

from .algorithms import BallTree, BruteForce

__all__ = [
    "collapse",
    "collocate",
    "collocate_datasets",
    "expand",
]

# Finder algorithms for collocations:
ALGORITHM = {
    "BallTree": BallTree,
    "BruteForce": BruteForce,
}

COLLOCATION_FIELD = "__collocation_ids"

# Factor to convert a length unit to kilometers
UNITS_CONVERSION_FACTORS = [
    [{"cm", "centimeter", "centimeters"}, 1e-6],
    [{"m", "meter", "meters"}, 1e-3],
    [{"km", "kilometer", "kilometers"}, 1],
    [{"mi", "mile", "miles"}, 1.609344],  # english statute mile
    [{"yd", "yds", "yard", "yards"}, 0.9144e-3],
    [{"ft", "foot", "feet"}, 0.3048e-3],
]

# @staticmethod
# def _add_fields_to_data(data, original_dataset, group, fields):
#     try:
#         original_file = data[group].attrs["__original_file"]
#     except KeyError:
#         raise KeyError(
#             "The collocation files does not contain information about "
#             "their original files.")
#     original_data = original_dataset.read(original_file)[fields]
#     original_indices = data[group]["__original_indices"]
#     data[group] = GroupedArrays.merge(
#         [data[group], original_data[original_indices]],
#         overwrite_error=False
#     )
#
#     return data
#
# def add_fields(self, start, end, original_dataset, group, fields):
#     """
#
#     Args:
#         start:
#         end:
#         original_dataset:
#         group
#         fields:
#
#     Returns:
#         None
#     """
#     self.map(
#         start, end, func=self._add_fields_to_data,
#         kwargs={
#             "group": group,
#             "original_dataset": original_dataset,
#             "fields": fields,
#         }, on_content=True, output=self
#     )


def collapse(data, reference=None, collapser=None, include_stats=None, ):
    """Collapses all multiple collocation points to a single data point

    Warnings:
        Does not work yet!

    During searching for collocations, one might find multiple collocation
    points from one dataset for one single point of the other dataset. For
    example, the MHS instrument has a larger footprint than the AVHRR
    instrument, hence one will find several AVHRR colloocation points for
    each MHS data point. This method performs a function on the multiple
    collocation points to merge them to one single point (e.g. the mean
    function).

    Args:
        data: Data from collocations.
        reference: Name of dataset which has the largest footprint. All
            other datasets will be collapsed to its data points.
        collapser: Reference to a function that should be applied on each bin
            (numpy.nanmean is the default).
        include_stats: Set this to a name of a variable (or list of
            names) and statistical parameters will be stored about the
            built data bins of the variable before collapsing. The variable
            must be one-dimensional.

    Returns:
        A GroupedArrays object with the collapsed data

    Examples:
        .. code-block:: python

            # TODO: Add examples
    """

    raise NotImplementedError("Not yet implemented!")

    # Get the bin indices by the main dataset to which all other
    # shall be collapsed:
    reference_bins = list(
        data[reference][COLLOCATION_FIELD].group().values()
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
        bins = data[group][COLLOCATION_FIELD].bin(
            reference_bins
        )
        collapsed_data["__statistics"] = \
            data[include_stats].apply_on_bins(
                bins, statistic_functions
            )
        collapsed_data["__statistics"].attrs["description"] = \
            "Statistics about the collapsed bins of '{}'.".format(
                include_stats
            )

    for dataset in data.groups():
        if dataset.startswith("__"):
            collapsed_data[dataset] = data[dataset]

        collocations = data[dataset][COLLOCATION_FIELD]

        if (dataset == reference
            or data[dataset].attrs.get("COLLAPSED_TO", None)
                == reference):
            # The collocation indices will become useless
            del data[dataset][COLLOCATION_FIELD]

            # This is the main dataset to which all other will be
            # collapsed. Therefore, we do not need explicitly
            # collapse here.
            collapsed_data[dataset] = \
                data[dataset][np.unique(collocations)]
        else:
            # We do not need the original and collocation indices from the
            # dataset that will be collapsed because they will soon become
            # useless. Moreover, they could have a different dimension
            # length than the other variables and lead to errors in the
            # selecting process.

            del data[dataset]["__original_indices"]
            del data[dataset][COLLOCATION_FIELD]

            bins = collocations.bin(reference_bins)

            # We ignore some warnings rather than fixing them
            # TODO: Maybe fix them?
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in double_scalars")
                collapsed_data[dataset] = \
                    data[dataset].collapse(
                        bins, collapser=collapser,
                    )

            collapsed_data[dataset].attrs["COLLAPSED_TO"] = reference

    # Set the collapsed flag:
    collapsed_data.attrs["COLLAPSED"] = 1

    # Overwrite the content of the old file:
    return collapsed_data


def expand(data):
    """Repeat each data point to its multiple collocation points

    Warnings:
        Does not work yet!

    This is the inverse function of :func:`collapse`.

    Args:
        data:

    Returns:

    """

    raise NotImplementedError("Not yet implemented!")
    expanded_data = GroupedArrays()
    for group_name in data.groups():
        if group_name.startswith("__"):
            continue

        #indices = data["__collocations"][]
        expanded_data[group_name] = data[group_name][indices]

def _to_kilometers(distance):
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


def collocate(arrays, max_interval=None, max_distance=None,
              algorithm=None, threads=None,):
    """Find collocations between two data arrays

    Collocations are two or more data points that are located close to each
    other in space and/or time.

    A data array must be a dictionary, a xarray.Dataset or a GroupedArrays
    object with the keys *time*, *lat*, *lon*. Its values must
    be 1-dimensional numpy.array-like objects and share the same length. The
    field *time* must have the data type *numpy.datetime64*, *lat* must be
    latitudes between *-90* (south) and *90* (north) and *lon* must be
    longitudes between *-180* (west) and *180* (east) degrees. See below for
    examples.

    Args:
        arrays: A list of data arrays that fulfill the specifications from
            above. So far, only collocating two arrays is implemented.
        max_interval: Either a number as a time interval in seconds, a string
            containing a time with a unit (e.g. *100 minutes*) or a timedelta
            object. This is the maximum time interval between two data points
            If this is None, the data will be searched for spatial collocations
            only.
        max_distance: Either a number as a length in kilometers or a string
            containing a length with a unit (e.g. *100 meters*). This is the
            maximum distance between two data points in to meet the collocation
            criteria. If this is None, the data will be searched for temporal
            collocations only. Either *max_interval* or *max_distance* must be
            given.
        algorithm: Defines which algorithm should be used to find the
            collocations. Must be either an object that inherits from
            :class:`~typhon.spareice.collocations.algorithms.CollocationsFinder`
            or a string with the name of an algorithm. Default is the
            *BallTree* algorithm. See below for a table of available
            algorithms.
        threads: Finding collocations can be parallelised in threads. Give here
            the maximum number of threads that you want to use. This does not
            work so far.

    Returns:
        A 2xN numpy array where N is the number of found collocations. The
        first row contains the indices of the collocations in *data1*, the
        second row the indices in *data2*.

    How the collocations are going to be found is specified by the used
    algorithm. The following algorithms are possible (you can use your
    own algorithm by subclassing the
    :class:`~typhon.spareice.collocations.algorithms.CollocationsFinder`
    class):

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

            # Create the data. primary and secondary can also be
            # xarray.Dataset or a GroupedArray objects:
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
                [primary, secondary], max_distance="300km", max_interval="1h")

            print(indices)  # prints [[4], [4]]


    """

    if max_distance is None and max_interval is None:
        raise ValueError("Either max_distance or max_interval must be given!")

    if len(arrays) != 2:
        raise ValueError("So far, only collocating of two arrays is allowed.")

    if max_interval is not None:
        max_interval = to_timedelta(max_interval, numbers_as="seconds")

    if max_distance is not None:
        max_distance = _to_kilometers(max_distance)

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

    for i, array in enumerate(arrays):
        if isinstance(array, dict):
            arrays[i] = GroupedArrays.from_dict(array)
        elif isinstance(array, xr.Dataset):
            arrays[i] = GroupedArrays.from_xarray(array)

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
            print("Return empty!")
            # There was no common time window found
            return np.array([[], []])

        start, end, *time_indices = common_time

        # Select the relevant data:
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
            _collocate_period(
                arrays, algorithm, (max_interval, max_distance), period,
            )
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
                arrays, algorithm, (max_interval, max_distance),
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

        pairs = pairs.astype("int64")
    else:
        # Search for spatial or temporal-spatial collocations but do not do any
        # pre-binning:
        pairs = algorithm.find_collocations(
            *arrays, max_distance=max_distance, max_interval=max_interval
        )

    return pairs


def _select_common_time(time1, time2, max_interval):
    common_start = np.max([time1.min(), time2.min()]).item(0) - max_interval
    common_end = np.min([time1.max(), time2.max()]).item(0) + max_interval

    # Return the indices from the data in the common time window
    indices1 = (common_start <= time1) & (time1 <= common_end)
    if not indices1.any():
        return None

    indices2 = (common_start <= time2) & (time2 <= common_end)
    if not indices2.any():
        return None

    return common_start, common_end, indices1, indices2


def _collocate_period(data_arrays, algorithm, algorithm_args,
                      period1, period2=None, ):
    data1, data2 = data_arrays

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

    * *dataset_name/lat* - Latitudes of the collocations.
    * *dataset_name/lon* - Longitude of the collocations.
    * *dataset_name/time* - Timestamp of the collocations.
    * *dataset_name/__original_indices* - Indices of the collocation data in
        the original files.
    * *__collocations/{primary}-{secondary}/pairs* - Tells you which data
        points are collocated with each other by giving their indices.

    Args:
        datasets: A list of Dataset objects.
        start: Start date either as datetime object or as string
            ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
            Hours, minutes and seconds are optional. If no date is given, the
            *0000-01-01* wil be taken.
        end: End date. Same format as "start". If no date is given, the
            *9999-12-31* wil be taken.
        output: Either a path as string containing placeholders or a
            Dataset object.
        verbose: If true, it prints logging messages.
        **collocate_args: Additional keyword arguments that are allowed for
            :func:`collocate` except *arrays*.

    Returns:
        A :class:`Dataset` object holding the collocated data.

    Examples:

    .. :code-block:: python

        # TODO Add examples
    """

    if len(datasets) != 2:
        raise ValueError("Only collocating two datasets at once is allowed"
                         "at the moment!")
    primary, secondary = datasets

    # Make sure that our output dataset is a CollocatedDataset
    if isinstance(output, str):
        name = "-".join([ds.name for ds in datasets])
        output = Dataset(path=output, name=name)
    elif not isinstance(output, Dataset):
        raise ValueError("The parameter 'output' must be a string or Dataset "
                         "object!")

    # Set the defaults for start and end
    start = datetime.min if start is None else to_datetime(start)
    end = datetime.max if end is None else to_datetime(end)

    # Check the max_interval argument because we need it later
    max_interval = collocate_args.get("max_interval", None)
    if max_interval is None:
        raise ValueError("Collocating datasets without max_interval is"
                         " not yet implemented!")
    max_interval = to_timedelta(max_interval, numbers_as="seconds")

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

    # We may have collocations that overlap multiple files. Hence, we save in
    # this cache always the last max_interval minutes from each dataset
    cache = {}

    for files, data in DataSlider(start, end, *datasets):
        primary_start, primary_end = data[primary.name].get_range("time")

        if verbose:
            _collocating_status(
                primary, secondary, timer, start, end,
                min([primary_end, data[secondary.name]["time"].max().item(0)]),
            )

        print("Before:", data["SatelliteA"]["time"].shape, data["SatelliteB"]["time"].shape)
        if cache:
            # Data from this iteration might be collocated with data from
            # previous iterations in the cache. Hence, include the cache for
            # the collocation search and add it to data:
            collocations = _collocate_include_cache(
                data, cache, primary, secondary, max_interval, collocate_args
            )
        else:
            collocations = collocate(
                [data[primary.name], data[secondary.name]],
                **collocate_args,
            )

        print("After:", data["SatelliteA"]["time"].shape, data["SatelliteB"]["time"].shape)

        # Cache the last max_interval time period of each dataset:
        for name, dataset_data in data.items():
            interval_to_cache = dataset_data["time"] >= \
                                dataset_data["time"].max().item(
                                    0) - max_interval
            cache[name] = dataset_data[interval_to_cache]

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
                f"{n_collocations[1]} ({datasets[1].name}) collocations in\n"
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

    return output


def _collocate_include_cache(data, cache, primary, secondary,
                             max_interval, collocate_args):
    # We have to search for collocations twice. At first, we do a
    # cross-collocation with the cached content from previous iterations:
    check_with_cache = \
        data[secondary.name]["time"] <= \
        data[secondary.name]["time"].min().item(0) + max_interval
    pri_cache_sec_collocations = collocate(
        [cache[primary.name], data[secondary.name][check_with_cache]],
        **collocate_args,
    )
    check_with_cache = \
        data[primary.name]["time"] <= \
        data[primary.name]["time"].min().item(0) + max_interval
    pri_sec_cache_collocations = collocate(
        [data[primary.name][check_with_cache], cache[secondary.name]],
        **collocate_args,
    )

    # and afterwards we collocate the new data:
    collocations = collocate(
        [data[primary.name], data[secondary.name]],
        **collocate_args,
    )

    if not pri_cache_sec_collocations.any() and not\
            pri_sec_cache_collocations.any():
        return collocations

    # Add the cached data to all_data if we found something
    data[primary.name] = GroupedArrays.concat(
        [cache[primary.name], data[primary.name]])
    data[secondary.name] = GroupedArrays.concat(
        [cache[secondary.name], data[secondary.name]])

    print("Before:", collocations)

    # We have to shift the collocation indices because we added data at the
    # front
    collocations[0] += cache[primary.name]["time"].shape[0]
    collocations[1] += cache[secondary.name]["time"].shape[0]

    print("After:", collocations)

    return np.hstack([
        pri_cache_sec_collocations,
        pri_sec_cache_collocations,
        collocations,
    ]).astype(int)




def _collocating_status(primary, secondary, timer, start, end, current_end):
    print("-" * 79)

    if start == datetime.min and end == datetime.max:
        print(
            f"Collocating {primary.name} to {secondary.name}: "
            f"processing {current_end}"
        )
    else:
        current = (current_end - start).total_seconds()
        progress = current / (end - start).total_seconds()

        elapsed_time = time.time() - timer
        expected_time = timedelta(
            seconds=int(elapsed_time * (1 / progress - 1))
        )

        print(
            f"Collocating {primary.name} to {secondary.name}: "
            f"{100*progress:.0f}% done ({expected_time} hours remaining)"
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
        data["__indices"] = Array(
            original_indices, dims=["time_id", ],
            attrs={
                "long_name": "Index in the original file",
            }
        )

        if "__original_files" not in data.attrs:
            # Set where the data came from:
            data.attrs["__original_files"] = \
                ";".join(file.path for file in files[datasets[i].name])
        output_data[datasets[i].name] = data

    metadata["pairs"] = pairs

    time_coverage = output_data.get_range("time", deep=True)
    output_data.attrs["start_time"] = \
        time_coverage[0].strftime("%Y-%m-%dT%H:%M:%S.%f")
    output_data.attrs["end_time"] = \
        time_coverage[1].strftime("%Y-%m-%dT%H:%M:%S.%f")

    # Prepare the name for the output file:
    attributes = {
        p: v for file in files.values() for p, v in file[0].attr.items()
    }
    filename = output.generate_filename(time_coverage, fill=attributes)

    # Write the data to the file.
    output.write(output_data, filename)

    return filename, number_of_collocations
