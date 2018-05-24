"""
This module provides classes and functions to find collocations between
filesets and datasets. They are inspired by the CollocatedDataset classes in
atmlab implemented by Gerrit Holl.

Created by John Mrziglod, June 2017
"""

from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime, timedelta
import logging

import random
import time

import numba
import numpy as np
import pandas as pd
from typhon.files import FileSet
from typhon.utils import reraise_with_stack
from typhon.utils.timeutils import to_datetime, to_timedelta
import xarray as xr

from .collocator import Collocator

__all__ = [
    "collapse",
    "Collocator",
    "Collocations",
    "expand",
]

# The names for the processes. This started as an easter egg, but it actually
# helps to identify different processes during debugging.
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


class InvalidCollocationData(Exception):
    """Error when trying to collapse / expand invalid collocation data

    """

    def __init__(self, message, *args):
        Exception.__init__(self, message, *args)


class Collocations(FileSet):
    """Class for finding and storing collocations between FileSet objects

    If you want to find collocations between Arrays, use :func:`collocate`
    instead.
    """
    def __init__(
            self, *args, reference=None, read_mode=None, collapser=None,
            **kwargs):
        """Initialize a Collocation object

        This :class:`~typhon.files.fileset.FileSet`

        Args:
            *args: Positional arguments for
                :class:`~typhon.files.fileset.FileSet`.
            read_mode: The collocations can be collapsed or expanded after
                collecting. Set this either to *collapse* (default) or
                *expand*.
            reference: If `read_mode` is *collapse*, here you can set the name
                of the dataset to that the others should be collapsed. Default
                is the primary dataset.
            collapser: If `read_mode` is *collapse*, here you can give your
                dictionary with additional collapser functions.
            **kwargs: Keyword arguments for
                :class:`~typhon.files.fileset.FileSet`.
        """
        # Call the base class initializer
        super().__init__(*args, **kwargs)

        self.read_mode = read_mode
        self.reference = reference
        self.collapser = collapser
        self.collocator = None

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
        """Read a file and apply a collapse / expand function to it

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

        if self.read_mode == "fileset":
            # Do nothing
            return data
        elif self.read_mode == "collapse" or self.read_mode is None:
            # Collapse the data (default)
            return collapse(data, self.reference, self.collapser)
        elif self.read_mode == "expand":
            # Expand the data
            return expand(data)
        else:
            raise ValueError(
                f"Unknown reading read_mode for collocations: "
                f"{self.read_mode}!\nAllowed read_modes are: 'collapse' "
                f"(default), 'expand' or 'fileset'."
            )

    def search(
            self, filesets, start=None, end=None, skip_overlaps=True,
            processes=None, collocator=None, log_dir=None,
            verbose=1, **collocate_args):
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
        * *collocations/{primary}.{secondary}/pairs* - Tells you which data
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
            skip_overlaps: If the files of the primary fileset overlap in
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

        # We create the collocator object now and use it multiple times,
        # since it may cache internally things to speed up the process.
        if collocator is None:
            self.collocator = Collocator()
        else:
            self.collocator = collocator

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

        if processes > 1:
            verbose = min(verbose, 1)

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
                    filesets, period[0], period[1], skip_overlaps,
                    verbose=verbose, **collocate_args
                )
                for i, period in enumerate(periods)
            ]

            # Wait for all processes to end
            wait(futures)

            for i, future in enumerate(futures):
                if future.exception() is not None:
                    print(f"[{PROCESS_NAMES[i]}] {future.exception()}")

    @reraise_with_stack
    def _search(self, pid, filesets, start, end, skip_overlaps,
                verbose, **collocate_args):

        # Check the max_interval argument because we need it later
        max_interval = to_timedelta(
            collocate_args["max_interval"], numbers_as="seconds"
        )

        if verbose:
            print(f"[{pid}] Collocate from {start} to {end}")

        total_collocations = [0, 0]

        # The primaries may overlap. So we use the end timestamp of the last
        # primary as starting point for this search:
        last_primary_end = None

        # Get all primary and secondary data that overlaps with each other
        file_pairs = filesets[0].align(filesets[1], start, end, max_interval)

        start, end = pd.Timestamp(start), pd.Timestamp(end)

        # Use a timer for profiling.
        timer = time.time()
        verbose_timer = time.time()
        for files, raw_data in file_pairs:
            if verbose:
                self._print_collocating_status(
                    pid, timer, start, end, last_primary_end
                )
            if verbose > 1:
                print(
                    f"[{pid}] {time.time() - verbose_timer:.2f}s for reading"
                )
                verbose_timer = time.time()

            primary, secondary = self._prepare_data(
                filesets, files, raw_data.copy(), start, end, max_interval,
                skip_overlaps, last_primary_end
            )

            if primary is None:
                if verbose > 1:
                    print(f"[{pid}] Skipping because data is empty!")
                continue

            current_start = np.datetime64(primary["time"].min().item(0), "ns")
            current_end = np.datetime64(primary["time"].max().item(0), "ns")
            print(f"[{pid}] Collocating {current_start} to {current_end}")

            # Maybe we have overlapping primary files? Let's save always the
            # end of our last search to use it as starting point for the next
            # iteration:
            last_primary_end = current_end

            if verbose > 1:
                print(
                    f"[{pid}] {time.time()-verbose_timer:.2f}s for preparing"
                )

            verbose_timer = time.time()
            collocations = self.collocator.run(
                (filesets[0].name, primary),
                (filesets[1].name, secondary), **collocate_args,
            )

            if verbose > 1:
                print(
                    f"[{pid}] {time.time()-verbose_timer:.2f}s for collocating"
                )

            verbose_timer = time.time()

            if not collocations.variables:
                if verbose > 1:
                    print(f"[{pid}] Found no collocations!")
                continue

            # Check whether the collocation data is compatible and was build
            # correctly
            check_collocation_data(collocations)

            # Prepare the name for the output file:
            attributes = {
                p: v
                for file in files.values()
                for p, v in file[0].attr.items()
            }
            filename = self.get_filename(
                [to_datetime(collocations.attrs["start_time"]),
                 to_datetime(collocations.attrs["end_time"])], fill=attributes
            )

            # Write the data to the file.
            self.write(collocations, filename)

            found = [
                collocations[f"{filesets[0].name}/lat"].size,
                collocations[f"{filesets[1].name}/lat"].size
            ]

            if verbose > 1:
                print(
                    f"[{pid}] Store {found[0]} ({filesets[0].name}) and "
                    f"{found[1]} ({filesets[1].name}) collocations to"
                    f"\n[{pid}] {filename}"
                )
                print(f"[{pid}] {time.time()-verbose_timer:.2f}s for storing")

            total_collocations[0] += found[0]
            total_collocations[1] += found[1]

            verbose_timer = time.time()

        if verbose:
            print(
                f"[{pid}] {time.time()-timer:.2f} s to find "
                f"{total_collocations[0]} ({filesets[0].name}) and "
                f"{total_collocations[1]} ({filesets[1].name}) "
                f"collocations."
            )

    def _prepare_data(self, filesets, files, dataset,
                      global_start, global_end, max_interval,
                      remove_overlaps, last_primary_end):
        """Make the raw data almost ready for collocating

        Before we can collocate the data points, we need to flat them (if they
        are stored in a structured representation).
        """
        primary, secondary = filesets

        for fileset in filesets:
            name = fileset.name

            # Add identifiers: Maybe we want to add more data after
            # collocating? To make this possible, we add the start and end
            # timestamp of the original file and the index of each data point
            # to the data. We need the length of the main dimension for this.
            # What is the main dimension? Well, we simply use the first
            # dimension of the time variable.
            main_dimension = dataset[name][0].time.dims[0]
            timer = time.time()
            dataset[name] = self._add_identifiers(
                files[name], dataset[name], main_dimension
            )
            print(f"{time.time()-timer:.2f} seconds for adding identifiers")

            # Concatenate all data: Since the data may come from multiple files
            # we have to concatenate them along their main dimension before
            # moving on.
            dataset[name] = xr.concat(dataset[name], dim=main_dimension)

            # Flat the data: For collocating, we need a flat data structure.
            # Fortunately, xarray provides the very convenient stack method
            # where we can flat multiple dimensions to one. Which dimensions do
            # we have to stack together? We need the fields *time*, *lat* and
            # *lon* to be flat. So we choose their dimensions to be stacked.
            timer = time.time()
            dataset[name] = Collocator.flat_to_main_coord(dataset[name])
            print(f"{time.time()-timer:.2f} seconds for flatting data")

            # The user may not want to collocate overlapping data twice since
            # it might contain duplicates
            # TODO: This checks only for overlaps in primaries. What about
            # TODO: overlapping secondaries?
            if name == primary.name:
                if remove_overlaps and last_primary_end is not None:
                    dataset[name] = dataset[name].isel(
                        collocation=dataset[name]['time'] > last_primary_end
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
                    print(f"Start: {start}, end: {end}.")
                    raise ValueError(
                        "The time coverage of the content of the following "
                        "files is not identical with the time coverage given "
                        "by get_info. Actual time coverage:\n"
                        + "\n".join(repr(file) for file in files[name])
                    )

            # We do not have to collocate everything, just the common time
            # period expanded by max_interval and limited by the global start
            # and end parameter:
            timer = time.time()
            dataset[name] = dataset[name].isel(
                collocation=(dataset[name]['time'] >= np.datetime64(start))
                             & (dataset[name]['time'] <= np.datetime64(end))
            )
            print(f"{time.time()-timer:.2f} seconds for selecting time")

            # Filter out NaNs:
            timer = time.time()
            not_nans = \
                dataset[name].time.notnull() \
                & dataset[name].lat.notnull() \
                & dataset[name].lon.notnull()
            dataset[name] = dataset[name].isel(collocation=not_nans)
            print(f"{time.time()-timer:.2f} seconds to filter nans")

            # Check whether something is left:
            if not len(dataset[name].time):
                return None, None

            timer = time.time()
            dataset[name] = dataset[name].sortby("time")
            print(f"{time.time()-timer:.2f} seconds to sort")

        return dataset[primary.name], dataset[secondary.name]

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
    def _print_collocating_status(pid, timer, start, end, current_end):
        if start == datetime.min and end == datetime.max:
            return

        if current_end is None:
            progress = 0
            expected_time = "unknown"
        else:
            # Annoying bug! The order of the calculation is important,
            # otherwise we get a TypeError!
            current = abs((start - current_end).total_seconds())
            progress = current / (end - start).total_seconds()

            elapsed_time = time.time() - timer
            expected_time = timedelta(
                seconds=int(elapsed_time * (1 / progress - 1))
            )

        print(f"[{pid}] {100*progress:.0f}% done ({expected_time} "
              f"hours remaining)")


@numba.jit
def _rows_for_secondaries_numba(primary):
    """Helper function for collapse - numba optimized"""
    current_row = np.zeros(primary.size, dtype=int)
    rows = np.zeros(primary.size, dtype=int)
    i = 0
    for p in primary:
        rows[i] = current_row[p]
        i += 1
        current_row[p] += 1
    return rows


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


def collapse(data, reference=None, collapser=None):
    """Collapse all multiple collocation points to a single data point

    During searching for collocations, one might find multiple collocation
    points from one dataset for one single point of the other dataset. For
    example, the MHS instrument has a larger footprint than the AVHRR
    instrument, hence one will find several AVHRR colloocation points for
    each MHS data point. This method performs a function on the multiple
    collocation points to merge them to one single point (e.g. the mean
    function).

    Args:
        data:
        reference: Normally the name of the dataset with the largest
            footprints. All other dataset will be collapsed to its data points.
        collapser: Dictionary with names of collapser functions to apply and
            references to them. Defaults collpaser functions are *mean*, *std*
            and *number* (count of valid data points).

    Returns:
        A xr.Dataset object with the collapsed data

    Examples:
        .. code-block:: python

            # TODO: Add examples
    """

    # Check whether the collocation data is compatible
    check_collocation_data(data)

    pairs = data["Collocations/pairs"].values
    groups = data["Collocations/group"].values.tolist()

    if reference is None:
        # Take automatically the first member of the groups (was the primary
        # while collocating)
        reference = groups[0]

    if reference not in groups:
        raise ValueError(
            f"The selected reference '{reference}' is not valid, because it "
            f"was not collocated! Collocated groups were: {groups}."
        )

    # Find the index of the reference group. If it is the primary, it is 0
    # otherwise 1.
    reference_index = groups[0] != reference

    primary_indices = pairs[int(reference_index)]
    secondary_indices = pairs[int(not reference_index)]

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
    # have to find out at which row to put them. For big datasets, this might
    # be a very expensive function. Therefore, we have have two version: One
    # (pure-python), which we use for small datasets, and one (numba-optimized)
    # for big datasets because Numba produces an overhead:
    #timer = time.time()
    if len(primary_indices) < 1000:
        rows_in_bins = _rows_for_secondaries(primary_indices)
    #    print(f"{time.time()-timer:.2f} seconds for pure-python")
    else:
        rows_in_bins = _rows_for_secondaries_numba(primary_indices)
    #    print(f"{time.time()-timer:.2f} seconds for numba")

    # The user may give his own collapser functions:
    if collapser is None:
        collapser = {}
    collapser = {
        "mean": lambda m, a: np.nanmean(m, axis=a),
        "std": lambda m, a: np.nanstd(m, axis=a),
        "number": lambda m, a: np.count_nonzero(~np.isnan(m), axis=a),
        **collapser,
    }

    collapsed = xr.Dataset()

    for var_name, var_data in data.variables.items():
        group, local_name = var_name.split("/", 1)

        # We copy the reference, collapse the rest and ignore the meta data of
        # the collocations (because they become useless now)
        if group == "Collocations":
            continue

        # This is the name of the dimension along which we collapse:
        collapse_dim = group + "/collocation"

        # Make sure that our collapsing dimension is the first dimension of the
        # array. Otherwise we get problems, when converting the DataArray to a
        # numpy array.
        new_ordered_dims = list(var_data.dims)
        new_ordered_dims.remove(collapse_dim)
        new_ordered_dims.insert(0, collapse_dim)
        var_data = var_data.transpose(*new_ordered_dims)

        # Rename the first dimension to the main collocation dimension
        dims = list(var_data.dims)
        dims[0] = "collocation"
        var_data.dims = dims

        if group == reference:
            # We want to make the resulting dataset collocation-friendly (so
            # that we might use it for a collocation search with another
            # dataset). So the content of the reference group moves "upward"
            # (group name vanishes from the path):
            collapsed[local_name] = var_data
            continue

        # The standard fields (time, lat, lon) and the special fields to
        # retrieve additional fields are useless after collapsing. Hence,
        # ignore them (won't be copied to the resulting dataset):
        if local_name in ("time", "lat", "lon") or local_name.startswith("__"):
            continue

        # The dimensions of the binner matrix:
        binner_dims = [
            np.max(rows_in_bins) + 1,
            np.unique(primary_indices).size
        ]

        # The data might have additional dimensions (e.g. brightness
        # temperatures from MHS have 5 channels). We must now how many
        # additional dimensions the data have and what their length is.
        add_dim_sizes = [
            var_data.shape[i]
            for i, dim in enumerate(var_data.dims)
            if dim != "collocation"
        ]
        binner_dims.extend(add_dim_sizes)

        # Fill the data in the bins:
        # Create an empty matrix:
        binned_data = np.empty(binner_dims)

        # Fill all slots with NaNs:
        binned_data[:] = np.nan
        binned_data[rows_in_bins, primary_indices] \
            = var_data.isel(collocation=secondary_indices).values

        for func_name, func in collapser.items():
            collapsed[f"{var_name}_{func_name}"] = \
                var_data.dims, func(binned_data, 0)

    return collapsed


def expand(dataset):
    """Repeat the primary data so they align with their secondary collocations

    During searching for collocations, one might find multiple collocation
    points from one dataset for one single point of the other dataset. For
    example, the MHS instrument has a larger footprint than the AVHRR
    instrument, hence one will find several AVHRR colloocation points for
    each MHS data point. To avoid needing more storage than required, no
    duplicated data values are stored even if they collocate multiple times.

    Args:
        dataset:

    Returns:
        A xarray.Dataset object.
    """
    # Check whether the collocation data is compatible
    check_collocation_data(dataset)

    pairs = dataset["Collocations/pairs"].values
    groups = dataset["Collocations/group"].values.tolist()

    expanded = dataset.isel(
        **{groups[0] + "/collocation": pairs[0]}
    )
    expanded = expanded.isel(
        **{groups[1] + "/collocation": pairs[1]}
    )

    # The collocation coordinate of all datasets are equal now:
    for i in range(2):
        expanded["collocation"] = groups[i] + "/collocation", \
                                  np.arange(pairs[i].size)
        expanded.swap_dims(
            {groups[i] + "/collocation": "collocation"}, inplace=True
        )

    # The variable pairs is useless now:
    expanded = expanded.drop("Collocations/pairs")

    # expanded.reset_coords([
    #     primary + "/collocation",
    #     secondary + "/collocation"], drop=True, inplace=True
    # )

    return expanded


def check_collocation_data(dataset):
    """Check whether the dataset fulfills the standard of collocated data

    Args:
        dataset: A xarray.Dataset object

    Raises:
        A InvalidCollocationData Error if the dataset did not pass the test.
    """
    mandatory_fields = ["Collocations/pairs", "Collocations/group"]

    for mandatory_field in mandatory_fields:
        if mandatory_field not in dataset.variables:
            raise InvalidCollocationData(
                f"Could not find the field '{mandatory_field}'!"
            )
