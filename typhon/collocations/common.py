"""
This module provides classes and functions to find collocations between
filesets and datasets. They are inspired by the CollocatedDataset classes in
atmlab implemented by Gerrit Holl.

Created by John Mrziglod, June 2017
"""

import logging

import numba
import numpy as np
from typhon.files import FileSet
from typhon.utils.timeutils import Timer
import xarray as xr

from .collocator import check_collocation_data, Collocator

__all__ = [
    "collapse",
    "Collocator",
    "Collocations",
    "expand",
]

logger = logging.getLogger(__name__)


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
                collecting. Set this either to *collapse* (default),
                *expand* or *compact*.
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

    def add_fields(self, original_fileset, fields, **kwargs):
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
            **kwargs,
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

        if self.read_mode == "compact":
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
                f"(default), 'expand' or 'compact'."
            )

    def search(self, filesets, collocator=None, **kwargs):
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
        * *collocations/pairs* - Tells you which data points are collocated
            with each other by giving their indices.

        And all additional fields that were loaded from the original files (you
        can control which fields should be loaded by the `read_args` parameter
        from the :class:`~typhon.files.fileset.FileSet` objects in `filesets`).
        Note that subgroups in the original data will be flattened by replacing
        */* with *_* since xarray is not yet able to handle grouped data
        properly.

        Args:
            filesets: A list of FileSet objects.
            collocator: If you want to use your own collocator, you can pass it
                here. It must behave like
                :class:`~typhon.collocations.collocator.Collocator`.
            **kwargs: Additional keyword arguments that are allowed for
                :meth:`~typhon.collocations.collocator.Collocator.collocate_filesets`.

        Examples:

        .. :code-block:: python

            from typhon.collocations import Collocations
            from typhon.files import FileSet, MHS_HDF, SEVIRI


            # Define a fileset with the files from MHS / NOAA18:
            mhs = FileSet(
                name="MHS",
                path="/path/to/files/{year}/{month}/{day}/*NSS.MHSX.*."
                     "S{hour}{minute}.E{end_hour}{end_minute}.*.h5",
                handler=MHS_HDF(),
                read_args={
                    "fields": ["Data/btemps"],
                }
            )

            # Define a fileset with files from SEVIRI
            seviri = FileSet(
                name="SEVIRI",
                path="/path/{year}/{month}/{day}/{hour}{minute}{second}.*.h5.gz",  # noqa
                handler=SEVIRI(),
                # Each file of CloudSat covers exactly 15 minutes.
                time_coverage="15 minutes",
            )

            # Define where the collocations should be stored:
            collocations = Collocations(
                path="/path/MHS-{satname}_SEVIRI/{year}/{month}/{day}/"
                     "{hour}{minute}{second}-{end_hour}{end_minute}{end_second}.nc"
            )

            # Search for collocations and store them
            collocations.search(
                [seviri, mhs], start="2013-12-10", end="20 Dec 2013",
                processes=10, max_interval="5 min", max_distance="5 km",
            )
        """
        if collocator is None:
            collocator = Collocator()

        collocated_files = collocator.collocate_filesets(
            filesets, output=self, **kwargs
        )

        timer = Timer().start()
        for _ in collocated_files:
            pass
        logger.info(f"{timer} for finding all collocations")


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
    collapsed = xr.Dataset(
        attrs=data.attrs.copy(),
    )

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

    for var_name, var_data in data.variables.items():
        group, local_name = var_name.split("/", 1)

        # We copy the reference, collapse the rest and ignore the meta data of
        # the collocations (because they become useless now)
        if group == "Collocations":
            continue

        # This is the name of the dimension along which we collapse:
        collapse_dim = group + "/collocation"

        if collapse_dim not in var_data.dims:
            # This variable does not depend on the collocation coordinate.
            # Hence, we cannot collapse it but we simply copy it. To make the
            # dataset collocation-friendly, the time, lat and lon fields of the
            # reference group will be copied to the root path:
            if group == reference and local_name in ("time", "lat", "lon"):
                collapsed[local_name] = var_data
            else:
                collapsed[var_name] = var_data

            continue

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
            if local_name in ("time", "lat", "lon"):
                collapsed[local_name] = var_data
            else:
                collapsed[var_name] = var_data
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
        dataset: A xarray.Dataset object with collocated data.

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
        expanded = expanded.swap_dims(
            {groups[i] + "/collocation": "collocation"}
        )

    # The variable pairs is useless now:
    expanded = expanded.drop("Collocations/pairs")

    expanded = expanded.rename({"Collocations/collocation": "collocation"})

    return expanded
