"""
This module provides classes and functions to find collocations between
filesets and datasets. They are inspired by the CollocatedDataset classes in
atmlab implemented by Gerrit Holl.

Created by John Mrziglod, June 2017
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
import logging
from numbers import Number
import random
import time
import traceback
import warnings

import numba
import numpy as np
import pandas as pd
import scipy.stats
from typhon.files import FileSet, NoHandlerError
from typhon.math import cantor_pairing
from typhon.utils import reraise_with_stack, split_units
from typhon.utils.time import to_datetime, to_timedelta
import xarray as xr

from .algorithms import BallTree, BruteForce

__all__ = [
    "collapse",
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

# The names for the processes. This started as an easter egg, but it actually
# helps identify different processes during debugging.
PROCESS_NAMES = [
    'Newton', 'Einstein', 'Bohr', 'Darwin', 'Pasteur', 'Freud', 'Galilei',
    'Lavoisier', 'Kepler', 'Copernicus', 'Faraday', 'Maxwell', 'Bernard',
    'Boas', 'Heisenberg', 'Pauling', 'Virchow', 'Schrodinger', 'Rutherford',
    'Dirac', 'Vesalius', 'Brahe', 'Buffon', 'Boltzmann', 'Planck', 'Curie',
    'Herschel', 'Lyell', 'Laplace', 'Hubble', 'Thomson', 'Born', 'Crick',
    'Fermi', 'Euler', 'Liebig', 'Eddington', 'Harvey', 'Malpighi', 'Huygens',
    'Gauss', 'Haller', 'Kekule', 'Koch', 'Gell-Mann', 'Fischer', 'Mendeleev',
    'Glashow', 'Watson', 'Bardeen', 'Neumann', 'Feynman', 'Wegener', 'Hawking',
    'Leeuwenhoek', 'Laue', 'Kirchhoff', 'Bethe', 'Euclid', 'Mendel', 'Onnes',
    'Morgan', 'Helmholtz', 'Ehrlich', 'Mayr', 'Sherrington', 'Dobzhansky',
    'Delbruck', 'Lamarck', 'Bayliss', 'Chomsky', 'Sanger', 'Lucretius',
    'Dalton', 'Broglie', 'Linnaeus', 'Piaget', 'Simpson', 'Levi-Strauss',
    'Margulis', 'Landsteiner', 'Lorenz', 'Wilson', 'Hopkins', 'Elion', 'Selye',
    'Oppenheimer', 'Teller', 'Libby', 'Haeckel', 'Salk', 'Kraepelin',
    'Lysenko', 'Galton', 'Binet', 'Kinsey', 'Fleming', 'Skinner', 'Wundt',
    'Archimedes'
]
random.shuffle(PROCESS_NAMES)


class Collocations(FileSet):
    """Class for finding and storing collocations between FileSet objects

    If you want to find collocations between Arrays, use :func:`collocate`
    instead.
    """
    def __init__(
            self, *args, primary=None, secondary=None, mode=None,
            collapser=None, **kwargs):
        """Initialize a Collocation object

        This :class:`~typhon.files.fileset.FileSet`

        Args:
            *args:
            primary: Name of fileset which has the largest footprint. All
                other filesets will be collapsed to its data points.
            secondary: Name of fileset which has the smaller footprints.
            mode: The collocations can be collapsed or expanded after
                collecting. Set this either to *collapse* (default) or
                *expand*.
            collapser: If the mode is *collapse*, here you can give your
                dictionary with additional collapser functions.
            **kwargs:
        """
        # Call the base class initializer
        super().__init__(*args, **kwargs)

        self.primary = primary
        self.secondary = secondary
        self.mode = mode
        self.collapser = collapser

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
        pass

    def read(self, *args, **kwargs):
        """Read a file and apply a collapser / expand function to it

        Does the same as :meth:`~typhon.files.fileset.FileSet.read` from the
        base class :class:`~typhon.files.fileset.FileSet`, but can
        collapse or expand collocations after reading them.

        Args:
            *args: Positional arguments for
                :meth:`~typhon.files.fileset.FileSet.read`.
            **kwargs: Keyword arguments for
                :meth:`~typhon.files.fileset.FileSet.read`.

        Returns:
            The same as :meth:`~typhon.files.fileset.FileSet.read`, but with
            data that is either collapsed or expanded.
        """
        data = super(Collocations, self).read(*args, **kwargs)

        if self.mode == "fileset":
            return data

        if self.primary is None or self.secondary is None:
            raise ValueError(
                "You must set the primary and secondary properties before "
                "using advanced reading modes!"
            )

        if self.mode == "collapse" or self.mode is None:
            return collapse(data, self.primary, self.secondary, self.collapser)
        elif self.mode == "expand":
            return expand(data, self.primary, self.secondary)
        else:
            raise ValueError(
                f"Unknown reading mode for collocations: {self.mode}!\n"
                "Allowed modes are: 'collapse' (default), 'expand' or "
                "'fileset'."
            )

    def search(
            self, filesets, start=None, end=None, remove_overlaps=True,
            processes=None, log_dir=None, verbose=1, **collocate_args, ):
        """Find all collocations between two filesets and store them in files

        Collocations are two or more data points that are located close to each
        other in space and/or time.

        This takes all files from the filesets between two dates, find
        collocations of their data points and store them in output files.

        Each collocation output file provides these standard fields:

        * *fileset_name/lat* - Latitudes of the collocations.
        * *fileset_name/lon* - Longitude of the collocations.
        * *fileset_name/time* - Timestamp of the collocations.
        * *fileset_name/__index* - Indices of the collocation data in
            the original files.
        * *fileset_name/__file_start* - Start time of the original file.
        * *fileset_name/__file_end* - End time of the original file.
        * *__collocations/{primary}.{secondary}/pairs* - Tells you which data
            points are collocated with each other by giving their indices.

        And all additional fields that were loaded from the original files (you
        can control which fields should be loaded by the `read_args` parameter
        from the :class:`~typhon.files.fileset.FileSet` objects in `filesets`).
        Note that subgroups in the original data will be flattened by replacing
        */* with *_* since xarray is not yet able to handle grouped data
        properly.

        Args:
            filesets: A list of FileSet objects.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional. If no date is given,
                the *0000-01-01* wil be taken.
            end: End date. Same format as `start`. If no date is given, the
                *9999-12-31* wil be taken.
            remove_overlaps: If the files of the primary fileset overlap in
                time, the overlapping data is used only once for collocating.
            processes: The number of processes that should be used to
                parallelize the collocation process. Default is 1.
            log_dir: If given, the log files for the processes will be stored
                in this directory.
            verbose: If true, it prints logging messages.
            **collocate_args: Additional keyword arguments that are allowed for
                :func:`collocate` except `data`.

        Examples:

        .. :code-block:: python

            # TODO Add examples
        """

        if len(filesets) != 2:
            raise ValueError("Only collocating two filesets at once is allowed"
                             "at the moment!")

        # Set the primary and secondary names (will be needed when reading from
        # the collocation files):
        self.primary = filesets[0].name
        self.secondary = filesets[1].name

        # Check the max_interval argument because we need it later
        max_interval = collocate_args.get("max_interval", None)
        if max_interval is None:
            raise ValueError("Collocating filesets without max_interval is"
                             " not yet implemented!")
        # max_interval = to_timedelta(max_interval, numbers_as="seconds")

        # Check the algorithm argument because we need the required fields
        collocate_args["algorithm"] = \
            _get_algorithm(collocate_args.get("algorithm", None))

        # Encourage the user to set the start and end timestamps explicitly
        # (otherwise splitting onto multiple processes may not be very
        # efficient):
        if start is None or end is None:
            raise ValueError("The collocation search needs explicit start and "
                             "end timestamps")
        start = to_datetime(start)
        end = to_datetime(end)

        # Set the number of processes per default to 1, the user shall
        # explicitly ask for parallel processing:
        if processes is None:
            processes = 1

        # We split up the whole search period into chunks and search in each
        # chunk with an individual process.
        diff = (end - start) / processes
        periods = [
            [start + diff * i, start + diff * (i + 1)]
            for i in range(processes)
        ]

        if verbose:
            print(f"Start {processes} processes to collocate "
                  f"{filesets[0].name} and {filesets[1].name}\nfrom {start} to"
                  f" {end}")
            print("-" * 79)

        with ProcessPoolExecutor(processes) as pool:
            futures = [
                pool.submit(
                    Collocations._search, self, PROCESS_NAMES[i],
                    filesets, period[0], period[1], remove_overlaps,
                    verbose=min(verbose, 1), **collocate_args
                )
                for i, period in enumerate(periods)
            ]

            # Wait for all processes to end
            wait(futures)

            for future in futures:
                if future.exception() is not None:
                    print(future.exception())

    @reraise_with_stack
    def _search(self, pid, filesets, start, end, remove_overlaps,
                verbose, **collocate_args):
        primary, secondary = filesets

        # Check the max_interval argument because we need it later
        max_interval = to_timedelta(
            collocate_args["max_interval"], numbers_as="seconds"
        )

        # Use a timer for profiling.
        timer = time.time()

        if verbose:
            print(f"[{pid}] Collocate from {start} to {end}")

        total_collocations = [0, 0]

        verbose_timer = time.time()

        # The primaries may overlap. So we use the end timestamp of the last
        # primary as starting point for this search:
        last_primary_end = None

        # Get all primary and secondary data that overlaps with each other
        file_pairs = primary.align(
            [secondary], start, end, max_interval
        )
        for files, raw_data in file_pairs:
            if verbose > 1:
                reading_time = time.time() - verbose_timer

            raw_data, stacked_dims = self._prepare_data(
                filesets, files, raw_data.copy(), start, end, max_interval,
                remove_overlaps, last_primary_end,
            )

            if raw_data is None:
                if verbose > 1:
                    print("Skipping because data is empty!")
                continue

            # We do not need all data for collocating, therefore we select only
            # the required fields and convert all to a DataFrame (makes a lot
            # of things easier):
            required_fields = list(collocate_args["algorithm"].required_fields)
            data = {
                name: value[required_fields].to_dataframe()  # noqa
                for name, value in raw_data.items()
            }

            if data[primary.name].empty or data[secondary.name].empty:
                if verbose > 1:
                    print("Skipping because data is empty!")
                continue

            if verbose:
                self._print_collocating_status(
                    pid, timer, start, end, data[primary.name]["time"]
                )
            if verbose > 1:
                print(f"{reading_time:.2f}s for reading the data")

            # Maybe we have overlapping primary files? Let's save always the
            # end of our last search to use it as starting point for the next
            # iteration:
            last_primary_end = np.datetime64(data[primary.name]["time"].max())

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
                if verbose > 1:
                    print("Found no collocations!")
                continue

            # Store the collocated data to the output fileset:
            filename, n_collocations = self._store_collocations(
                filesets=[primary, secondary], raw_data=raw_data,
                collocations=collocations, files=files,
                stacked_dims=stacked_dims, **collocate_args
            )

            if verbose > 1:
                print(
                    f"Store {n_collocations[0]} ({filesets[0].name}) and "
                    f"{n_collocations[1]} ({filesets[1].name}) collocations to"
                    f"\n{filename}"
                )
                print(f"{time.time()-verbose_timer:.2f}s for storing the data")

            total_collocations[0] += n_collocations[0]
            total_collocations[1] += n_collocations[1]

            verbose_timer = time.time()

        if verbose:
            print(
                f"[{pid}] {time.time()-timer:.2f} s to find "
                f"{total_collocations[0]} ({primary.name}) and "
                f"{total_collocations[1]} ({secondary.name}) "
                f"collocations."
            )

    def _prepare_data(self, filesets, files, dataset,
                      global_start, global_end, max_interval,
                      remove_overlaps, last_primary_end, ):
        """Make the raw data almost ready for collocating

        Before we can collocate the data points, we need to flat them (if they
        are stored in a structured representation).
        """
        primary, secondary = filesets

        # We may have stack (flat) some datasets. The dimensions that have to
        # be stacked will be chosen automatically. We store there names in this
        # dictionary for later:
        stacked_dims = {}

        for fileset in filesets:
            name = fileset.name

            # Add identifiers: Maybe we want to add more data after
            # collocating? To make this possible, we add the start and end
            # timestamp of the original file and the index of each data point
            # to the data. We need the length of the main dimension for this.
            # What is the main dimension? Well, we simply use the first
            # dimension of the time variable.
            main_dimension = dataset[name][0].time.dims[0]
            dataset[name] = self._add_identifiers(
                files[name], dataset[name], main_dimension
            )

            # Concatenate all data: Since the data may come from multiple files
            # we have to concatenate them along their main dimension before
            # moving on.
            dataset[name] = xr.concat(dataset[name], dim=main_dimension)

            # Flat the data: For collocating, we need a flat data structure.
            # Fortunately, xarray provides the very convenient stack method
            # where we can flat multiple dimensions to one. Which dimensions do
            # we have to stack together? We need the fields *time*, *lat* and
            # *lon* to be flat. So we choose their dimensions to be stacked.
            dataset[name], stacked_dims[name] = \
                self._flat_to_main_coord(dataset[name])

            # The user may not want to collocate overlapping data twice since
            # it might contain duplicates
            # TODO: This checks only for overlaps in primaries. What about
            # TODO: overlapping secondaries?
            if name == primary.name:
                if remove_overlaps and last_primary_end is not None:
                    dataset[name] = dataset[name].isel(
                        __collocation=dataset[name]['time'] > last_primary_end
                    )

                # Check whether something is left:
                if not len(dataset[name]['time']):
                    return None, None

                # We want to select a common time window from both datasets,
                # aligned to the primary's time coverage. Because xarray has a
                # very annoying bug in time retrieving
                # (https://github.com/pydata/xarray/issues/1240), this is a
                # little bit cumbersome:
                start = max(
                    global_start,
                    pd.Timestamp(dataset[name]['time'].min().item(0))
                    - max_interval
                )
                end = min(
                    global_end,
                    pd.Timestamp(dataset[name]['time'].max().item(0))
                    + max_interval
                )

                if start > end:
                    raise ValueError(
                        "The time coverage of the content of the following "
                        "files is not identical with the time coverage given "
                        "by get_info. Actual time coverage:\n"
                        + "\n".join(repr(file) for file in files[name])
                    )

            # We do not have to collocate everything, just the common time
            # period expanded by max_interval and limited by the global start
            # and end parameter:
            dataset[name] = dataset[name].isel(
                __collocation=(dataset[name]['time'] >= np.datetime64(start))
                              & (dataset[name]['time'] <= np.datetime64(end))
            )

            # Filter out NaNs:
            not_nans = \
                dataset[name].time.notnull() \
                & dataset[name].lat.notnull() \
                & dataset[name].lon.notnull()
            dataset[name] = dataset[name].isel(__collocation=not_nans)

            # Check whether something is left:
            if not len(dataset[name].time):
                return None, None

            dataset[name] = dataset[name].sortby("time")

        return dataset, stacked_dims

    @staticmethod
    def _flat_to_main_coord(data):
        # Some field might be more deeply stacked than another. Choose the
        # dimensions of the most deeply stacked variable:
        dims = max(
            data["time"].dims, data["lat"].dims, data["lon"].dims,
            key=lambda x: len(x)
        )

        return data.stack(__collocation=dims), dims

    @staticmethod
    def _add_identifiers(files, data, dimension):
        """Add identifiers (file start and end time, original index) to
        each data point.
        """
        for index, file in enumerate(files):
            if "__file_start" in data[index]:
                # Apparently, we collocated and tagged this dataset
                # already.
                # TODO: Nevertheless, should we add new identifiers now?
                continue

            length = data[index][dimension].shape[0]
            data[index]["__file_start"] = \
                dimension, np.repeat(np.datetime64(file.times[0]), length)
            data[index]["__file_end"] = \
                dimension, np.repeat(np.datetime64(file.times[1]), length)

            # TODO: This gives us only the index per main dimension but what if
            # TODO: the data is more deeply stacked?
            data[index]["__index"] = dimension, np.arange(length)

        return data

    @staticmethod
    def _print_collocating_status(pid, timer, start, end, data):
        current_start = data.min()
        current_end = data.max()
        #print("-" * 79)
        #print(f"Collocating from {current_start} to {current_end}")

        if start == datetime.min and end == datetime.max:
            return

        current = (current_end - start).total_seconds()
        progress = current / (end - start).total_seconds()

        elapsed_time = time.time() - timer
        expected_time = timedelta(
            seconds=int(elapsed_time * (1 / progress - 1))
        )

        print(f"[{pid}] {100*progress:.0f}% done ({expected_time} "
              f"hours remaining)")

    def _store_collocations(
            self, filesets, raw_data, collocations, files, stacked_dims,
            **collocate_args):
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

        file_start, file_end = None, None
        pairs = []
        number_of_collocations = []
        output_data = {}

        for i, fileset in enumerate(filesets):
            # if "__collocations" in dataset_data.groups():
            #     # This dataset contains already-collocated filesets,
            #     # therefore we do not select any data but copy all of them.
            #     # This keeps the indices valid, which point to the original
            #     # files and data:
            #     output_data = DataGroup.merge(
            #         [output_data, dataset_data]
            #     )
            #
            #     # Add the collocation indices. We do not have to adjust them
            #     # since we do not change the original data.
            #     pairs.append(collocations[i])
            #     continue

            # These are the indices of the points in the original data that
            # have collocations. We remove the duplicates since we want to copy
            # the required data only once:
            original_indices = pd.unique(collocations[i])
            number_of_collocations.append(len(original_indices))

            # After selecting the collocated data, the original indices cannot
            # be applied any longer. We need new indices that indicate the
            # pairs in the collocated data.
            new_indices = pd.Series(
                np.arange(len(original_indices)),
                index=original_indices,
            )

            collocation_indices = new_indices.loc[collocations[i]].values

            # Save the collocation indices in the metadata group:
            pairs.append(collocation_indices)

            output_data[fileset.name] = \
                raw_data[fileset.name].isel(__collocation=original_indices)

            # xarrays does not really handle grouped data (actually, not at
            # all). Until this has changed, I do not want to have subgroups in
            # the output data (this makes things complicated when it comes to
            # coordinates). Therefore, we 'flat' each group before continuing:
            output_data[fileset.name].rename(
                {
                    old_name: old_name.replace("/", "_")
                    for old_name in output_data[fileset.name].variables
                    if "/" in old_name
                }, inplace=True
            )

            # We need the total time coverage of all datasets for the name of
            # the output file
            data_start = pd.Timestamp(
                output_data[fileset.name]["time"].min().item(0)
            )
            data_end = pd.Timestamp(
                output_data[fileset.name]["time"].max().item(0)
            )
            if file_start is None or file_start > data_start:
                file_start = data_start
            if file_end is None or file_end < data_end:
                file_end = data_end

            # We have to convert the MultiIndex to a normal index because we
            # cannot store it to a file otherwise. We can convert it by simply
            # setting it to new values, but we are losing the sub-level
            # coordinates (the dimenisons that we stacked to create the
            # multi-index in the first place) with that step. Hence, we store
            # the sub-level coordinates in additional dataset to preserve them.
            stacked_dims_data = xr.merge([
                xr.DataArray(
                    output_data[fileset.name][dim].values,
                    name=dim, dims=["__collocation"]
                )
                for dim in stacked_dims[fileset.name]
            ])
            output_data[fileset.name]["__collocation"] = \
                np.arange(output_data[fileset.name]["__collocation"].size)

            # Now, since we unstacked the multi-index, we can add thw
            # stacked dimensions back to the dataset:
            output_data[fileset.name] = xr.merge(
                [output_data[fileset.name], stacked_dims_data],
            )

            # We want to merge all datasets together (but as subgroups). Hence,
            # add the fileset name to each dataset as prefix:
            output_data[fileset.name].rename(
                {
                    name: "/".join([fileset.name, name])
                    for name in output_data[fileset.name].variables
                }, inplace=True
            )

        # Merge all datasets into one:
        output_data = xr.merge(
            [data for data in output_data.values()]
        )

        # Add the metadata information (collocation pairs, distance and
        # interval):
        max_interval = collocate_args.get("max_interval", None)
        if max_interval is not None:
            max_interval = to_timedelta(max_interval).total_seconds()
        max_distance = collocate_args.get("max_distance", None)

        # This holds the collocation information:
        metadata = xr.DataArray(
            np.array(pairs, dtype=int), dims=("fileset", "pair"),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": filesets[0].name,
                "secondary": filesets[1].name
            }
        )

        meta_group = _get_meta_group(filesets[0].name, filesets[1].name)
        output_data[meta_group + "/pairs"] = metadata

        # Prepare the name for the output file:
        attributes = {
            p: v
            for file in files.values()
            for p, v in file[0].attr.items()
        }
        filename = self.get_filename([file_start, file_end], fill=attributes)

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


def _get_meta_group(primary, secondary):
    return f"{primary}.{secondary}"


@numba.jit
def _rows_for_secondaries(primary):
    """Helper function for collapse"""
    current_row = np.zeros(primary.size, dtype=int)
    rows = np.zeros(primary.size, dtype=int)
    i = 0
    for p in primary:
        rows[i] = current_row[p]
        i += 1
        current_row[p] += 1
    return rows


def collapse(data, primary, secondary, collapser=None):
    """Collapse all multiple collocation points to a single data point

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
        collapser: Dictionary with names of collapser functions to apply and
            references to them. Defaults collpaser functions are *mean*, *std*
            and *number* (count of valid data points).

    Returns:
        A xr.Dataset object with the collapsed data

    Examples:
        .. code-block:: python

            # TODO: Add examples
    """
    pairs = _get_meta_group(primary, secondary) + "/pairs"
    primary_indices = data[pairs][0].values
    secondary_indices = data[pairs][1].values

    # THE GOAL: We want to bin the secondary data according to the
    # primary indices and apply a collapse function (e.g. mean) to it.
    # THE PROBLEM: We might to group the data in many (!) bins that might
    # not have the same size and we have to apply a function onto each of
    # these bins. How to make this efficient?
    # APPROACH 1: pandas and xarray provide the powerful groupby method
    # which allows grouping by bins and applying a function to it.
    # -> This does not scale well with big datasets (100k of primaries
    # takes ~15 seconds).
    # APPROACH 2: Applying functions onto matrix columns is very fast in
    # numpy even if the matrix is big. We could create a matrix where each
    # column acts as a primary bin, the number of its rows is the maximum
    # number of secondaries per bin. Then, we fill the secondary data into
    # the corresponding bins. Since they might be a different number of
    # secondaries for each bin, there will be unfilled slots. We fill these
    # slots with NaN values (so they won't affect the outcome of the
    # collapser function). Now, we can apply the collapser function on the
    # whole matrix along the columns.
    # -> This approach is very fast but might need more space.
    # We follow approach 2, since it may scale better than approach 1 and
    # we normally do not have to worry about memory space. Gerrit's
    # collocation toolkit in atmlab also follows a similar approach.
    # Ok, let's start!

    # The matrix has the shape of N(max. number of secondaries per primary)
    # x N(unique primaries). So the columns are the primary bins. We know at
    # which column to put the secondary data by using primary_indices. Now, we
    # have to find out at which row to put them:
    rows_in_bins = _rows_for_secondaries(primary_indices)

    # Create an empty matrix:
    binned_data = np.empty(
        (np.max(rows_in_bins)+1,
         np.unique(primary_indices).size)
    )

    # Fill all slots with NaNs:
    binned_data[:] = np.nan

    # The user may give his own collapser functions:
    if collapser is None:
        collapser = {}
    collapser = {
        "mean": lambda m: np.nanmean(m, axis=1),
        "std": lambda m: np.nanstd(m, axis=1),
        "number": lambda m: np.count_nonzero(~np.isnan(m), axis=1),
        **collapser,
    }

    collapsed = xr.Dataset()

    for var_name, var_data in data.variables.items():
        group, local_name = var_name.split("/")

        # We copy the primaries, collapse the secondaries and ignore the rest
        if group == primary:
            # We want to make the resulting dataset collocation-friendly (so
            # that we might use it for a collocation search with another
            # dataset)
            if local_name in ("time", "lat", "lon"):
                collapsed[local_name] = var_data
            else:
                collapsed[var_name] = var_data
            continue
        elif group != secondary:
            continue

        # The standard fields (time, lat, lon) and the special fields to
        # retrieve additional fields are useless after collapsing. Hence,
        # ignore them (won't be copied to the resulting dataset):
        if local_name in ("time", "lat", "lon") or local_name.startswith("__"):
            continue

        # Fill the data in the bins:
        binned_data[rows_in_bins, primary_indices] \
            = var_data[secondary_indices]

        for func_name, func in collapser.items():
            collapsed[f"{var_name}_{func_name}"] = func(binned_data)

    return collapsed


def collocate(data, max_interval=None, max_distance=None,
              algorithm=None, threads=None, bin_factor=2):
    """Find collocations between two data objects

    Collocations are two or more data points that are located close to each
    other in space and/or time.

    A data object must be a dictionary, a xarray.Dataset or a pandas.DataFrame
    object with the keys *time*, *lat*, *lon*. Its values must
    be 1-dimensional numpy.array-like objects and share the same length. The
    field *time* must have the data type *numpy.datetime64*, *lat* must be
    latitudes between *-90* (south) and *90* (north) and *lon* must be
    longitudes between *-180* (west) and *180* (east) degrees. See below for
    examples.

    If you want to find collocations between FileSet objects, use
    :class:`Collocations` instead.

    Args:
        data: A list of data objects that fulfill the specifications from
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
    for i, array in enumerate(data):
        if isinstance(array, pd.DataFrame):
            pass
        elif isinstance(array, dict):
            data[i] = pd.DataFrame(array)
        elif isinstance(array, xr.Dataset):
            data[i] = array.to_dataframe()
        else:
            raise ValueError("Unknown array object!")

        # We use the time coordinate for binning, therefore we set it as index:
        if data[i].index.name != "time":
            data[i] = data[i].set_index("time")

    if data[0].empty or data[1].empty:

        # At least one of the data is empty
        return np.array([[], []])

    if max_distance is None and max_interval is None:
        raise ValueError("Either max_distance or max_interval must be given!")

    if len(data) != 2:
        raise ValueError("So far, only collocating of two data is allowed.")

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
    data_magnitude = len(data[0]) * len(data[1])

    # We can search for spatial collocations (max_interval=None), temporal
    # collocations (max_distance=None) or both.
    if max_interval is not None and data_magnitude > 100_0000:
        # Search for temporal and temporal-spatial collocations #

        # We start by selecting only the time period where both data
        # data have data and that lies in the time period requested by the
        # user.
        start = max([data[0].index.min(), data[1].index.min()]) \
            - max_interval
        end = min([data[0].index.max(), data[1].index.max()]) \
            + max_interval

        # Get the offset of the start date (so we can shift the indices later):
        offsets = [
            data[0].index.searchsorted(start),
            data[1].index.searchsorted(start)
        ]

        # Select the relevant data:
        data[0] = data[0].loc[start:end]
        data[1] = data[1].loc[start:end]

        # Now let's split the two data data along their time coordinate so we
        # avoid searching for spatial collocations that do not fulfill the
        # temporal condition in the first place. However, the overhead of the
        # finding algorithm must be considered too (for example the BallTree
        # creation time). We choose therefore a bin size of roughly 10'000
        # elements and minimum bin duration of max_interval.
        def get_chunk_pairs(chunk1_start, chunk1):
            chunk2_start = chunk1_start - max_interval
            chunk2_end = chunk2_start + max_interval
            offset1 = data[0].index.searchsorted(chunk1_start)
            offset2 = data[1].index.searchsorted(chunk2_start)
            chunk2 = data[1].loc[chunk2_start:chunk2_end]
            return offset1, chunk1, offset2, chunk2

        chunks_with_args = (
            [*get_chunk_pairs(chunk_start, chunk),
             algorithm, (max_interval, max_distance)]
            for chunk_start, chunk in data[0].groupby(
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
            *data, max_distance=max_distance, max_interval=max_interval
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


def expand(data, primary, secondary):
    """Collect collocation data from files and pad

    This is the inverse function of :func:`collapse`.

    Args:
        data:
        primary:
        secondary:

    Returns:

    """

    pairs = _get_meta_group(primary, secondary) + "/pairs"
    primary_indices = data[pairs][0].values
    secondary_indices = data[pairs][1].values

    expanded_data = xr.Dataset()
    for group_name in data.groups():
        if group_name.startswith("__"):
            continue

        #indices = data["__collocations"][]
        expanded_data[group_name] = data[group_name][indices]