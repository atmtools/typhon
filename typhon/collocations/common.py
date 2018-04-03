"""
This module contains classes to find collocations between datasets. They are
inspired by the CollocatedDataset classes in atmlab implemented by Gerrit Holl.

TODO: Move this package to typhon.collocations.

Created by John Mrziglod, June 2017
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import logging
from numbers import Number
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from typhon.files import FileSet
from typhon.math import cantor_pairing
from typhon.collections import DataGroup
from typhon.utils import split_units
from typhon.utils.time import to_datetime, to_timedelta
import xarray as xr

from .algorithms import BallTree, BruteForce

__all__ = [
    "collocate",
    "Collocations",
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


class Collocations(FileSet):
    """Class for finding and storing collocations between FileSet objects

    If you want to find collocations between Arrays, use :func:`collocate`
    instead.
    """
    def __init__(self, *args, **kwargs):
        # Call the base class initializer
        super().__init__(*args, **kwargs)

    def add_fields(self, original_fileset, fields, **map_args):
        """

        Args:
            start:
            end:
            original_fileset:
            group
            fields:

        Returns:
            None
        """
        map_args = {
            "on_content": True,
            "kwargs": {
                "original_fileset": original_fileset,
                "fields": fields,
            },
            **map_args,
        }

        return self.map(Collocations._add_fields, **map_args)

    @staticmethod
    def _add_fields(data, original_fileset, fields):
        try:
            original_file = data[group].attrs["__original_file"]
        except KeyError:
            raise KeyError(
                "The collocation files does not contain information about "
                "their original files.")
        original_data = original_fileset.read(original_file)[fields]
        original_indices = data[group]["__original_indices"]
        data[group] = GroupedArrays.merge(
            [data[group], original_data[original_indices]],
            overwrite_error=False
        )

        return data

    def collapse(self, primary, secondary, collapser=None, include_stats=None,
                 **map_args):
        """Collapses all multiple collocation points to a single data point

        During searching for collocations, one might find multiple collocation
        points from one dataset for one single point of the other dataset. For
        example, the MHS instrument has a larger footprint than the AVHRR
        instrument, hence one will find several AVHRR colloocation points for
        each MHS data point. This method performs a function on the multiple
        collocation points to merge them to one single point (e.g. the mean
        function).

        Args:
            primary: Name of fileset which has the largest footprint. All
                other filesets will be collapsed to its data points.
            secondary:
            collapser: Reference to a function that should be applied on each
                bin (numpy.nanmean is the default).
            include_stats: Set this to a name of a variable (or list of
                names) and statistical parameters will be stored about the
                built data bins of the variable before collapsing. The variable
                must be one-dimensional.
            **map_args

        Returns:
            A DataGroup object with the collapsed data

        Examples:
            .. code-block:: python

                # TODO: Add examples
        """
        map_args = {
            **map_args,
            "on_content": True,
            "worker_type":
                "thread" if map_args.get("output", None) is None else "process",  # noqa
            "kwargs": {
                "primary": primary,
                "secondary": secondary,
                "collapser": collapser,
                "include_stats": include_stats,
            },
        }

        return self.map(Collocations._collapse, **map_args)

    @staticmethod
    def _collapse(data, primary, secondary, collapser=None,
                  include_stats=None, ):
        timer = time.time()
        print(f"After {time.time()-timer:.2f}s: Starting collapse")
        pairs = data["__collocations"][primary + "." + secondary]["pairs"]

        # Get the bin indices by the main dataset to which all other
        # shall be collapsed:
        reference_bins = list(pairs[0].group().values())
        print(f"After {time.time()-timer:.2f}s: Got reference bins")

        collapsed_data = DataGroup()

        # Add additional statistics about one binned variable:
        if include_stats is not None:
            statistic_functions = {
                "variation": scipy.stats.variation,
                "mean": np.nanmean,
                "number": lambda x, _: x.shape[0],
                "std": np.nanstd,
            }

            # Create the bins for the variable from which you want to have
            # the statistics:
            group, _ = DataGroup.parse(include_stats)
            pair_index = 0 if group == primary else 1
            bins = pairs[pair_index].bin(reference_bins)
            collapsed_data["__statistics"] = \
                data[include_stats].apply_on_bins(
                    bins, statistic_functions
                )
            collapsed_data["__statistics"].attrs["description"] = \
                "Statistics about the collapsed bins of '{}'.".format(
                    include_stats
                )
            print(f"After {time.time()-timer:2.f}s: Got statistics")

        # This is the main dataset to which all other will be collapsed.
        # Therefore, we do not need explicitly collapse here.
        collapsed_data[primary] = data[primary][np.unique(pairs[0])]
        print(f"After {time.time()-timer:.2f}s: Got primary")

        # Collapse the secondary to the primary:
        bins = pairs[1].bin(reference_bins)
        print(f"After {time.time()-timer:.2f}s: collapsed secondary bins")

        # We ignore some warnings rather than fixing them
        # TODO: Maybe fix them?
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in double_scalars")
            collapsed_data[secondary] = \
                data[secondary].collapse(
                    bins, collapser=collapser,
                )
        print(f"After {time.time()-timer:.2f} s: Got secondary")

        print("Ending collapse")

        return collapsed_data

    def expand(self):
        """Repeat each data point to its multiple collocation points

        Warnings:
            Does not work yet!

        This is the inverse function of :func:`collapse`.

        Args:
            data:

        Returns:

        """

        raise NotImplementedError("Not yet implemented!")
        expanded_data = DataGroup()
        for group_name in data.groups():
            if group_name.startswith("__"):
                continue

            #indices = data["__collocations"][]
            expanded_data[group_name] = data[group_name][indices]

    def search(
            self, filesets, start=None, end=None, remove_overlaps=True,
            processes=None, verbose=True, **collocate_args, ):
        """Find all collocations between two filesets and store them in files

        Collocations are two or more data points that are located close to each
        other in space and/or time.

        This takes all files from the filesets between two dates and find
        collocations of their data points. Afterwards they will be stored in
        *output*.

        Each collocation output file provides these standard fields:

        * *fileset_name/lat* - Latitudes of the collocations.
        * *fileset_name/lon* - Longitude of the collocations.
        * *fileset_name/time* - Timestamp of the collocations.
        * *fileset_name/__original_indices* - Indices of the collocation data in
            the original files.
        * *__collocations/{primary}.{secondary}/pairs* - Tells you which data
            points are collocated with each other by giving their indices.

        Args:
            filesets: A list of FileSet objects.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional. If no date is given,
                the *0000-01-01* wil be taken.
            end: End date. Same format as "start". If no date is given, the
                *9999-12-31* wil be taken.
            remove_overlaps: If the files of the primary fileset overlap in
                time, the overlapping data is used only once for collocating.
            verbose: If true, it prints logging messages.
            **collocate_args: Additional keyword arguments that are allowed for
                :func:`collocate` except *arrays*.

        Returns:
            A :class:`FileSet` object holding the collocated data.

        Examples:

        .. :code-block:: python

            # TODO Add examples
        """

        if len(filesets) != 2:
            raise ValueError("Only collocating two filesets at once is allowed"
                             "at the moment!")
        primary, secondary = filesets

        # Set the defaults for start and end
        start = datetime.min if start is None else to_datetime(start)
        end = datetime.max if end is None else to_datetime(end)

        # Check the max_interval argument because we need it later
        max_interval = collocate_args.get("max_interval", None)
        if max_interval is None:
            raise ValueError("Collocating filesets without max_interval is"
                             " not yet implemented!")
        max_interval = to_timedelta(max_interval, numbers_as="seconds")

        # Check the algorithm argument because we need the required fields
        algorithm = _get_algorithm(collocate_args.get("algorithm", None))

        # Use a timer for profiling.
        timer = time.time()

        if verbose:
            print(f"Find collocations between {primary.name} and "
                  f"{secondary.name} from {start} to {end}")

        total_collocations = [0, 0]

        if verbose:
            print("Retrieve time coverages from files...")

        verbose_timer = time.time()

        # The primaries may overlap. So we use the end timestamp of the last
        # primary as starting point for this search:
        last_primary_end = None

        # Get all primary and secondary data that overlaps with each other
        for files, data in primary.align([secondary], start, end,
                                         max_interval):
            if verbose > 1:
                reading_time = time.time() - verbose_timer

            # This should make it easier to retrieve more data from the
            # original file in potential post-processing:
            data = self._add_identifiers(filesets, files, data)

            # Concatenate the data:
            raw_data = {
                primary.name:
                    DataGroup.concat(data[primary.name], dim="time"),
                secondary.name:
                    DataGroup.concat(data[secondary.name], dim="time"),
            }

            # The user may not want to collocate overlapping data twice since
            # it might contain duplicates
            if remove_overlaps and last_primary_end is not None:
                raw_data[primary.name] = \
                    raw_data[primary.name].sel(time=slice(last_primary_end))

            # We do not have to collocate everything, just the common time
            # period expanded by max_interval and limited by the global start
            # and end parameter:
            raw_data = self._limit_and_sort_data(
                start, end, raw_data, primary.name, max_interval
            )

            # We do not need all data for collocating, therefore we select only
            # the required fields and convert all to a DataFrame (makes a lot
            # of things easier):
            required_fields = list(algorithm.required_fields)
            # The time is the index of the dataframe and therefore no field any
            # longer:
            required_fields.remove("time")
            data = {
                name: value.select(required_fields).data.to_dataframe()  # noqa
                for name, value in raw_data.items()
            }

            if data[primary.name].empty:
                if verbose:
                    print("Skip overlapping data!")
                continue

            if verbose:
                self._print_collocating_status(
                    timer, start, end, data[primary.name].index
                )
            if verbose > 1:
                print(f"{reading_time:.2f}s for reading the data")

            # Maybe we have overlapping primary files? Let's save always the
            # end of our last search to use it as starting point for the next
            # iteration:
            last_primary_end = data[primary.name].index.max()

            verbose_timer = time.time()
            collocations = collocate(
                [data[primary.name], data[secondary.name]],
                **collocate_args,
            )

            if verbose > 1:
                print(f"{time.time()-verbose_timer:.2f}s for collocating the "
                      f"data")

            verbose_timer = time.time()

            if not collocations.any():
                if verbose:
                    print("Found no collocations!")
                continue

            # Store the collocated data to the output fileset:
            filename, n_collocations = self._store_collocations(
                filesets=[primary, secondary], raw_data=raw_data,
                collocations=collocations, files=files, **collocate_args
            )

            if verbose:
                print(
                    f"Store {n_collocations[0]} ({filesets[0].name}) and "
                    f"{n_collocations[1]} ({filesets[1].name}) collocations to"
                    f"\n{filename}"
                )
            if verbose > 1:
                print(f"{time.time()-verbose_timer:.2f}s for storing the data")

            total_collocations[0] += n_collocations[0]
            total_collocations[1] += n_collocations[1]

            verbose_timer = time.time()

        if verbose:
            print("-" * 79)
            print(
                f"Took {time.time()-timer:.2f} s to find "
                f"{total_collocations[0]} ({primary.name}) and "
                f"{total_collocations[1]} ({secondary.name}) "
                f"collocations.\nProcessed {end-start} hours of data."
            )

    @staticmethod
    def _add_identifiers(filesets, files, data):
        """Add file identifier (start and end time) to each data point
        """
        for fileset in filesets:
            ds_name = fileset.name
            # Add the file start and end time to each element and its index
            # in the original file
            for index, file in enumerate(files[ds_name]):
                if "__file_start" in data[ds_name][index]:
                    # Apparently, we collocated and tagged this dataset
                    # already.
                    # TODO: Nevertheless, should we add new identifiers now?
                    continue

                length = data[ds_name][index]["time"].shape[0]
                data[ds_name][index]["__file_start"] = \
                    "time", np.repeat(np.datetime64(file.times[0]), length)
                data[ds_name][index]["__file_end"] = \
                    "time", np.repeat(np.datetime64(file.times[1]), length)
                data[ds_name][index]["__index"] = "time", np.arange(length)

        return data

    @staticmethod
    def _limit_and_sort_data(
            global_start, global_end, raw_data, primary, max_interval):
        """Returns the search periods for the primary and secondary data
        """
        # Because xarray has a very annoying bug in time retrieving
        # (https://github.com/pydata/xarray/issues/1240), this is a little bit
        # cumbersome:
        start = max(
            global_start,
            pd.Timestamp(raw_data[primary][:].time.min().item(0))-max_interval
        )
        end = min(
            global_end,
            pd.Timestamp(raw_data[primary][:].time.max().item(0))+max_interval
        )

        print(start, end)

        # Select only the common time window from the data:
        raw_data = {
            name: value.sel(time=slice(start, end))
            for name, value in raw_data.items()
        }

        # We need monotonically increasing or decreasing time indices (well,
        # pandas needs that):
        return {
            name: value.apply(xr.Dataset.sortby, "time")
            for name, value in raw_data.items()
        }

    @staticmethod
    def _print_collocating_status(timer, start, end, data):
        current_start = data.min()
        current_end = data.max()
        print("-" * 79)
        print(f"Collocating from {current_start} to {current_end}")

        if start == datetime.min and end == datetime.max:
            return

        current = (current_end - start).total_seconds()
        progress = current / (end - start).total_seconds()

        elapsed_time = time.time() - timer
        expected_time = timedelta(
            seconds=int(elapsed_time * (1 / progress - 1))
        )

        print(f"Progress: {100*progress:.0f}% done ({expected_time} hours "
              f"remaining)")

    def _store_collocations(
            self, filesets, raw_data, collocations, files, **collocate_args):
        """Merge the data, original indices, collocation indices and
        additional information of the filesets to one DataGroup object.

        Args:
            output:
            filesets:
            raw_data:
            collocations:
            files:

        Returns:
            List with number of collocations
        """

        # The data that will be stored to a file:
        output_data = DataGroup(name="CollocatedData")

        # We need this name to store the collocation metadata in an adequate
        # group
        collocations_name = filesets[0].name + "." + filesets[1].name
        output_data["__collocations/" + collocations_name] = DataGroup()
        metadata = output_data["__collocations/" + collocations_name]

        max_interval = collocate_args.get("max_interval", None)
        if max_interval is not None:
            max_interval = to_timedelta(max_interval).total_seconds()
        metadata[:].attrs["max_interval"] = \
            f"Max. interval in secs: {max_interval}"

        max_distance = collocate_args.get("max_distance", None)
        metadata[:].attrs["max_distance"] = \
            f"Max. distance in kilometers: {max_distance}"
        metadata[:].attrs["primary"] = filesets[0].name
        metadata[:].attrs["secondary"] = filesets[1].name

        pairs = []
        number_of_collocations = []

        for i, fileset in enumerate(filesets):
            dataset_data = raw_data[fileset.name]

            if "__collocations" in dataset_data.groups():
                # This dataset contains already-collocated filesets,
                # therefore we do not select any data but copy all of them.
                # This keeps the indices valid, which point to the original
                # files and data:
                output_data = DataGroup.merge(
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

            output_data[filesets[i].name] = \
                dataset_data.isel(time=original_indices)

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
        filename = self.get_filename(time_coverage, fill=attributes)

        # Write the data to the file.
        self.write(output_data, filename)

        return filename, number_of_collocations


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
              algorithm=None, threads=None, bin_factor=2):
    """Find collocations between two data arrays

    Collocations are two or more data points that are located close to each
    other in space and/or time.

    A data array must be a dictionary, a xarray.Dataset or a pandas.DataFrame
    object with the keys *time*, *lat*, *lon*. Its values must
    be 1-dimensional numpy.array-like objects and share the same length. The
    field *time* must have the data type *numpy.datetime64*, *lat* must be
    latitudes between *-90* (south) and *90* (north) and *lon* must be
    longitudes between *-180* (west) and *180* (east) degrees. See below for
    examples.

    If you want to find collocations between FileSet objects, use
    :class:`Collocations` instead.

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
            the maximum number of threads that you want to use. Which number of
            threads is the best, may be machine-dependent. So this is a
            parameter that you can use to fine-tune the performance.
        bin_factor: When using a temporal criterion via `max_interval`, the
            data will be temporally binned to speed-up the search. The bin size
            is `bin_factor` * `max_interval`. Which bin factor is the best, may
            be dataset-dependent. So this is a parameter that you can use to
            fine-tune the performance.

    Returns:
        A 2xN numpy array where N is the number of found collocations. The
        first row contains the indices of the collocations in `data1`, the
        second row the indices in `data2`.

    How the collocations are going to be found is specified by the used
    algorithm. The following algorithms are possible (you can use your
    own algorithm by subclassing
    :class:`~typhon.spareice.collocations.algorithms.CollocationsFinder`):

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
    # Internally, we use pandas.Dateframe objects. There are simpler to use
    # than xarray.Dataset objects and are well designed for this purpose.
    # Furthermore, xarray.Dataset has a very annoying bug at the
    # moment that makes time selection more cumbersome
    # (https://github.com/pydata/xarray/issues/1240).

    for i, array in enumerate(arrays):
        if isinstance(array, pd.DataFrame):
            pass
        elif isinstance(array, dict):
            arrays[i] = pd.DataFrame(array)
        elif isinstance(array, xr.Dataset):
            arrays[i] = array.to_dataframe()
        else:
            raise ValueError("Unknown array object!")

        # We use the time coordinate for binning, therefore we set it as index:
        if arrays[i].index.name != "time":
            arrays[i] = arrays[i].set_index("time")

    if arrays[0].empty or arrays[1].empty:
        # At least one of the arrays is empty
        return np.array([[], []])

    if max_distance is None and max_interval is None:
        raise ValueError("Either max_distance or max_interval must be given!")

    if len(arrays) != 2:
        raise ValueError("So far, only collocating of two arrays is allowed.")

    if max_interval is not None:
        max_interval = to_timedelta(max_interval, numbers_as="seconds")

    if max_distance is not None:
        max_distance = _to_kilometers(max_distance)

    algorithm = _get_algorithm(algorithm)

    # Unfortunately, a first attempt parallelizing this using threads worsened
    # the performance. Hence, even it is ironically, it is better to use only
    # one thread.
    threads = 1 if threads is None else threads

    # If the time matters (i.e. max_interval is not None), we split the data
    # into temporal bins. This produces an overhead that is only negligible if
    # we have a lot of data:
    data_magnitude = len(arrays[0]) * len(arrays[1])

    # We can search for spatial collocations (max_interval=None), temporal
    # collocations (max_distance=None) or both.
    if max_interval is not None and data_magnitude > 100_0000:
        # Search for temporal and temporal-spatial collocations #

        # We start by selecting only the time period where both data
        # arrays have data and that lies in the time period requested by the
        # user.
        start = max([arrays[0].index.min(), arrays[1].index.min()]) \
            - max_interval
        end = min([arrays[0].index.max(), arrays[1].index.max()]) \
            + max_interval

        # Get the offset of the start date (so we can shift the indices later):
        offsets = [
            arrays[0].index.searchsorted(start),
            arrays[1].index.searchsorted(start)
        ]

        # Select the relevant data:
        arrays[0] = arrays[0].loc[start:end]
        arrays[1] = arrays[1].loc[start:end]

        # Now let's split the two data arrays along their time coordinate so we
        # avoid searching for spatial collocations that do not fulfill the
        # temporal condition in the first place. However, the overhead of the
        # finding algorithm must be considered too (for example the BallTree
        # creation time). We choose therefore a bin size of roughly 10'000
        # elements and minimum bin duration of max_interval.
        def get_chunk_pairs(chunk1_start, chunk1):
            chunk2_start = chunk1_start - max_interval
            chunk2_end = chunk2_start + max_interval
            offset1 = arrays[0].index.searchsorted(chunk1_start)
            offset2 = arrays[1].index.searchsorted(chunk2_start)
            chunk2 = arrays[1].loc[chunk2_start:chunk2_end]
            return offset1, chunk1, offset2, chunk2

        chunks_with_args = (
            [*get_chunk_pairs(chunk_start, chunk),
             algorithm, (max_interval, max_distance)]
            for chunk_start, chunk in arrays[0].groupby(
                pd.Grouper(freq=2*max_interval))
        )

        with ThreadPoolExecutor(threads) as pool:
            pairs_list = pool.map(_collocate_chunks, chunks_with_args)
        pairs = np.hstack(pairs_list)

        # No collocations were found.
        if not pairs.any():
            return pairs

        # We selected a common time window and cut off a part in the beginning,
        # do you remember? Now we shift the indices so that they point again
        # to the real original data.
        pairs[0] += offsets[0]
        pairs[1] += offsets[1]

        pairs = pairs.astype("int64")
    else:
        # Search for spatial or temporal-spatial collocations but do not do any
        # pre-binning:
        pairs = algorithm.find_collocations(
            *arrays, max_distance=max_distance, max_interval=max_interval
        )

    return pairs


def _get_algorithm(algorithm):
    if algorithm is None:
        return BallTree()
    else:
        if isinstance(algorithm, str):
            try:
                return ALGORITHM[algorithm]()
            except KeyError:
                raise ValueError("Unknown algorithm: %s" % algorithm)
        else:
            return algorithm


def _collocate_chunks(args):
    offset1, data1, offset2, data2, algorithm, algorithm_args = args

    if data1.empty or data2.empty:
        return np.array([[], []])

    pairs = algorithm.find_collocations(data1, data2, *algorithm_args)
    pairs[0] += offset1
    pairs[1] += offset2
    return pairs


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