import abc
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging

import time

import numpy as np
import pandas as pd
from typhon.geodesy import great_circle_distance
from typhon.geographical import GeoIndex
from typhon.utils.timeutils import to_timedelta
import xarray as xr

__all__ = [
        "Collocator",
    ]


class Collocator:
    def __init__(
            self, threads=None, bin_factor=1, magnitude_factor=10,
            tunnel_limit=None,
    ):
        """Initialize a collocator object that can find collocations

        Args:
            tunnel_limit: Maximum distance in kilometers at which to switch
                from tunnel to haversine distance metric. Per default this
                algorithm uses the tunnel metric, which simply transform all
                latitudes and longitudes to 3D-cartesian space and calculate
                their euclidean distance. This produces an error that grows
                with larger distances. When searching for distances exceeding
                this limit (`max_distance` is greater than this parameter), the
                haversine metric is used, which is more accurate but takes more
                time. Default is 1000 kilometers.
            magnitude_factor: Since building new trees is expensive, this
                algorithm tries to use the last tree when possible (e.g. for
                data with fixed grid). However, building the tree with the
                larger dataset and query it with the smaller dataset is faster
                than vice versa. Depending on which premise to follow, there
                might have a different performance in the end. This parameter
                is the factor that... TODO
            threads: Finding collocations can be parallelised in threads. Give
                here the maximum number of threads that you want to use. Which
                number of threads is the best, may be machine-dependent. So
                this is a parameter that you can use to fine-tune the
                performance.
            bin_factor: When using a temporal criterion via `max_interval`, the
                data will be temporally binned to speed-up the search. The bin
                size is `bin_factor` * `max_interval`. Which bin factor is the
                best, may be dataset-dependent. So this is a parameter that you
                can use to fine-tune the performance.

        """

        # If no collocations are found, this will be returned. We need empty
        # arrays to concatenate the results without problems:
        self.empty = xr.Dataset()
        self.no_pairs = np.array([[], []])
        self.no_intervals = np.array([], dtype='timedelta64[ns]')
        self.no_distances = np.array([])

        self.index = None
        self.index_with_primary = False

        self.threads = threads
        self.bin_factor = bin_factor
        self.magnitude_factor = magnitude_factor
        self.tunnel_limit = tunnel_limit

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, primary, secondary, max_interval=None, max_distance=None):
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
            primary: Data object that fulfill the specifications from above.
            secondary: Data object that fulfill the specifications from above.
            max_interval: Either a number as a time interval in seconds, a
                string containing a time with a unit (e.g. *100 minutes*) or a
                timedelta object. This is the maximum time interval between two
                data points. If this is None, the data will be searched for
                spatial collocations only.
            max_distance: Either a number as a length in kilometers or a string
                containing a length with a unit (e.g. *100 meters*). This is
                the maximum distance between two data points in to meet the
                collocation criteria. If this is None, the data will be
                searched for temporal collocations only. Either `max_interval`
                or *max_distance* must be given.

        Returns:
            Three numpy.arrays: the pairs of collocations (as indices in the
            original data), the interval for the time dimension and the
            distance for the spatial dimension. The pairs are a 2xN numpy.array
            where N is the number of found collocations. The first row contains
            the indices of the collocations in `data1`, the second row the
            indices in `data2`.

        Examples:

            .. code-block: python

                import numpy as np
                from typhon.collocations import Collocator

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

                # Find collocations with a maximum distance of 300 kilometers
                # and a maximum interval of 1 hour
                collocator = Collocator()
                indices = collocator.run(
                    primary, secondary,
                    max_distance="300km", max_interval="1h"
                )

                print(indices)  # prints [[4], [4]]


        """
        primary_name, primary, secondary_name, secondary = self._get_names(
            primary, secondary
        )

        # Flat the data: For collocating, we need a flat data structure.
        # Fortunately, xarray provides the very convenient stack method
        # where we can flat multiple dimensions to one. Which dimensions do
        # we have to stack together? We need the fields *time*, *lat* and
        # *lon* to be flat. So we choose their dimensions to be stacked.
        timer = time.time()
        primary = self.flat_to_main_coord(primary)
        print(f"{time.time()-timer:.2f} seconds to flat {primary_name}")
        timer = time.time()
        secondary = self.flat_to_main_coord(secondary)
        print(f"{time.time()-timer:.2f} seconds to flat {secondary_name}")

        # Maybe one data object is empty?
        if not primary["time"].size or not secondary["time"].size:
            return self.empty

        # Retrieve the important fields from the data. To avoid any overhead by
        # xarray, we use the plain numpy.arrays:
        lat1 = primary["lat"].values
        lon1 = primary["lon"].values
        lat2 = secondary["lat"].values
        lon2 = secondary["lon"].values
        time1 = primary["time"].values
        time2 = secondary["time"].values

        if max_distance is None and max_interval is None:
            raise ValueError(
                "Either max_distance or max_interval must be given!"
            )

        if max_interval is not None:
            max_interval = to_timedelta(max_interval, numbers_as="seconds")

        # We can search for spatial collocations (max_interval=None), temporal
        # collocations (max_distance=None) or both.
        if max_interval is None:
            # Search for spatial collocations only:
            pairs, distances = self.spatial_search(
                lat1, lon1, lat2, lon2, max_distance,
            )

            intervals = self._get_intervals(
                time1[pairs[0]], time2[pairs[1]]
            )

            return self._create_return(
                primary, secondary, primary_name, secondary_name,
                pairs, intervals, distances,
                max_interval, max_distance
            )
        elif max_distance is None:
            # Search for temporal collocations only:
            pairs, intervals = self.temporal_search(
                time1, time2, max_interval
            )

            distances = self._get_distances(
                lat1[pairs[0]], lon1[pairs[0]],
                lat2[pairs[1]], lon2[pairs[1]],
            )

            return self._create_return(
                    primary, secondary, primary_name, secondary_name,
                    pairs, intervals, distances,
                    max_interval, max_distance
            )

        # The user wants to use both criteria and search for spatial and
        # temporal collocations. At first, we do a coarse temporal pre-binning
        # so that we only search for collocations between points that might
        # also be temporally collocated. Unfortunately, this also produces an
        # overhead that is only negligible if we have a lot of data:
        data_magnitude = time1.size * time2.size

        if data_magnitude > 100_0000:
            # We have enough data, do pre-binning!
            pairs, distances = self.spatial_search_with_temporal_binning(
                {"lat": lat1, "lon": lon1, "time": time1},
                {"lat": lat2, "lon": lon2, "time": time2},
                max_distance, max_interval
            )
        else:
            # We do not have enough data to justify that whole pre-binning.
            # Simply do it directly!
            pairs, distances = self.spatial_search(
                lat1, lon1, lat2, lon2, max_distance,
            )

        # Did we find any spatial collocations?
        if not pairs.any():
            return self.empty

        # Check now whether the spatial collocations really pass the temporal
        # condition:
        passed_temporal_check, intervals = self._temporal_check(
            time1[pairs[0]], time2[pairs[1]], max_interval
        )

        # Return only the values that passed the time check
        return self._create_return(
            primary, secondary, primary_name, secondary_name,
            pairs[:, passed_temporal_check],
            intervals,
            distances[passed_temporal_check],
            max_interval, max_distance
        )

    @staticmethod
    def _get_names(primary, secondary):
        # Check out if the user gave the primary and secondary any name:
        if isinstance(primary, (tuple, list)):
            primary_name, primary = primary
        else:
            primary_name = "primary"
        if isinstance(secondary, (tuple, list)):
            secondary_name, secondary = secondary
        else:
            secondary_name = "secondary"

        return primary_name, primary, secondary_name, secondary

    @staticmethod
    def flat_to_main_coord(data):
        """Make the dataset flat despite of its original structure

        We need a flat dataset structure for the collocation algorithms, i.e.
        time, lat and lon are not allowed to be gridded, they must be
        1-dimensional and share the same dimension. There are three groups of
        original data structures that this method can handle:

        * linear (e.g. ship track measurements): time, lat and lon have the
            same dimension and are all 1-dimensional. Fulfills all criteria
            from above. No action has to be taken.
        * gridded_coords (e.g. instruments on satellites with gridded swaths):
            time, lat or lon are gridded (they have multiple dimensions). Stack
            the coordinates of them together to a new shared dimension.
        * gridded_data (e.g. model output data): time, lat and lon are not
            gridded but they grid the data variables. Stack time, lat and lon
            to a new shared dimension.

        Args:
            data: xr.Dataset object
            is_gridded: Boolean value whether the data is gridded along time,
                lat and lon coordinates.

        Returns:
            A xr.Dataset where time, lat and lon are aligned on one shared
            dimension.
        """
        # Flat: time, lat, lon share the same dimension size and are
        # 1-dimensional
        flat = len(
            set(data["time"].shape)
            | set(data["lat"].shape)
            | set(data["lon"].shape)) == 1

        if flat:
            return data.rename({
                data.time.dims[0]: "collocation"
            })

        # The coordinates are gridded:
        # Some field might be more deeply stacked than another. Choose the
        # dimensions of the most deeply stacked variable:
        dims = max(
            data["time"].dims, data["lat"].dims, data["lon"].dims,
            key=lambda x: len(x)
        )

        # We might have a problems if the dimensions are also coordinates (i.e.
        # they have values). For example, we opened multiple MHS files and
        # concatenated them, then the coordinate scnline contains the same
        # values multiple times. xarray cannot stack them together and build a
        # multi index from them because the coordinate values are not unique.
        # Hence, we need to replace the former coordinates with new coordinates
        # that have unique values.
        new_dims = []
        for dim in dims:
            new_dim = f"{dim}_replacement"
            data[new_dim] = dim, np.arange(data.dims[dim])
            data.swap_dims({dim: new_dim}, inplace=True)
            new_dims.append(new_dim)

        return data.stack(collocation=new_dims)

    @staticmethod
    def get_meta_group():
        return f"Collocations"

    def spatial_search_with_temporal_binning(
            self, primary, secondary, max_distance, max_interval
    ):
        # For time-binning purposes, pandas Dataframe objects are a good choice
        primary = pd.DataFrame(primary).set_index("time")
        secondary = pd.DataFrame(secondary).set_index("time")

        # For pre-binning
        primary, secondary, offsets = self._select_common_time(
            primary, secondary, max_interval
        )

        # Now let's split the two data data along their time coordinate so
        # we avoid searching for spatial collocations that do not fulfill
        # the temporal condition in the first place. However, the overhead
        # of the finding algorithm must be considered too (for example the
        # BallTree creation time). This can be adjusted by the parameter
        # bin_factor:
        bin_duration = self.bin_factor * max_interval

        # The binning is more efficient if we use the largest dataset as
        # primary:
        swapped_datasets = secondary.size > primary.size
        if swapped_datasets:
            print("Swap datasets!")
            primary, secondary = secondary, primary

        # Let's bin the primaries along their time axis and search for the
        # corresponding secondary bins:
        bin_pairs = (
            self._bin_pairs(start, chunk, primary, secondary, max_interval)
            for start, chunk in primary.groupby(pd.Grouper(freq=bin_duration))
        )

        # Add arguments to the bins (we need them for the spatial search
        # function):
        bins_with_args = (
            [self, max_distance, *bin_pair]
            for bin_pair in bin_pairs
        )

        # Unfortunately, a first attempt parallelizing this using threads
        # worsened the performance. Update: The BallTree code from scikit-learn
        # does not release the GIL. Maybe there will be a new version coming
        # that solves this problem in the future? See the scikit-learn issue:
        # https://github.com/scikit-learn/scikit-learn/pull/4009 (even it is
        # closed and merged, they have not revised the query_radius method
        # yet).
        # threads = 1 if self.threads is None else self.threads
        #
        # with ThreadPoolExecutor(threads) as pool:
        #     results = pool.map(
        #         Collocator._spatial_search_bin, bins_with_args
        #     )
        results = list(map(
            Collocator._spatial_search_bin, bins_with_args
        ))
        pairs_list, distances_list = zip(*results)
        pairs = np.hstack(pairs_list)

        # No collocations were found.
        if not pairs.any():
            return self.no_pairs, self.no_distances

        # Stack the rest of the results together:
        distances = np.hstack(distances_list)

        if swapped_datasets:
            # Swap the rows of the results
            pairs[[0, 1]] = pairs[[1, 0]]
            distances[[0, 1]] = distances[[1, 0]]

        # We selected a common time window and cut off a part in the
        # beginning, do you remember? Now we shift the indices so that they
        # point to the real original data again.
        pairs[0] += offsets[0]
        pairs[1] += offsets[1]

        return pairs.astype("int64"), distances

    @staticmethod
    def _select_common_time(primary, secondary, max_interval):
        # We start by selecting only the time period where both data
        # data have data and that lies in the time period requested by the
        # user.
        common_start = max([primary.index.min(), secondary.index.min()]) \
            - max_interval
        common_end = min([primary.index.max(), secondary.index.max()]) \
            + max_interval

        # Get the offset of the start date (so we can shift the indices
        # later):
        offsets = [
            primary.index.searchsorted(common_start),
            secondary.index.searchsorted(common_start)
        ]

        # Select the relevant data:
        primary = primary.loc[common_start:common_end]
        secondary = secondary.loc[common_start:common_end]

        return primary, secondary, offsets

    @staticmethod
    def _bin_pairs(chunk1_start, chunk1, primary, secondary, max_interval):
        """"""
        chunk2_start = chunk1_start - max_interval
        chunk2_end = chunk1.index.max() + max_interval
        offset1 = primary.index.searchsorted(chunk1_start)
        offset2 = secondary.index.searchsorted(chunk2_start)
        chunk2 = secondary.loc[chunk2_start:chunk2_end]
        return offset1, chunk1, offset2, chunk2

    @staticmethod
    def _spatial_search_bin(args):
        self, max_distance, offset1, data1, offset2, data2 = args

        if data1.empty or data2.empty:
            return self.no_pairs, self.no_distances

        pairs, distances = self.spatial_search(
            data1["lat"].values, data1["lon"].values,
            data2["lat"].values, data2["lon"].values, max_distance
        )
        pairs[0] += offset1
        pairs[1] += offset2
        return pairs, distances

    def spatial_search(self, lat1, lon1, lat2, lon2, max_distance):
        # Finding collocations is expensive, therefore we want to optimize it
        # and have to decide which points to use for the index building.
        index_with_primary = self._choose_points_to_build_index(
            [lat1, lon1], [lat2, lon2],
        )

        self.index_with_primary = index_with_primary

        if index_with_primary:
            build_points = lat1, lon1
            query_points = lat2, lon2
        else:
            build_points = lat2, lon2
            query_points = lat1, lon1

        self.index = self._build_spatial_index(*build_points)
        pairs, distances = self.index.query(*query_points, r=max_distance)

        # No collocations were found.
        if not pairs.any():
            # We return empty arrays to have consistent return values:
            return self.no_pairs, self.no_distances

        if not index_with_primary:
            # The primary indices should be in the first row, the secondary
            # indices in the second:
            pairs[[0, 1]] = pairs[[1, 0]]

        return pairs, distances

    def _build_spatial_index(self, lat, lon):
        # Find out whether the cached index still works with the new points:
        if self._spatial_is_cached(lat, lon):
            print("Spatial index is cached and can be reused")
            return self.index

        return GeoIndex(lat, lon)

    def _spatial_is_cached(self, lat, lon):
        if self.index is None:
            return False

        try:
            return np.allclose(lat, self.index.lat) \
                   & np.allclose(lon, self.index.lon)
        except ValueError:
            # The shapes are different
            return False

    def _choose_points_to_build_index(self, primary, secondary):
        """Choose which points should be used for tree building

        This method helps to optimize the performance.

        Args:
            primary: Converted primary points
            secondary: Converted secondary points

        Returns:
            True if primary points should be used for tree building. False
            otherwise.
        """
        # There are two options to optimize the performance:
        # A) Cache the index and reuse it if either the primary or the
        # secondary points have not changed (that is the case for data with a
        # fixed grid). Building the tree is normally very expensive, so it
        # should never be done without a reason.
        # B) Build the tree with the larger set of points and query it with the
        # smaller set.
        # Which option should be used if A and B cannot be applied at the same
        # time? If the magnitude of one point set is much larger (by
        # `magnitude factor` larger) than the other point set, we strictly
        # follow B. Otherwise, we prioritize A.

        if primary[0].size > secondary[0].size * self.magnitude_factor:
            # Use primary points
            return True
        elif secondary[0].size > primary[0].size * self.magnitude_factor:
            # Use secondary points
            return False

        # Apparently, none of the datasets is much larger than the others. So
        # just check whether we still have a cached tree. If we used the
        # primary points last time and they still fit, use them again:
        if self.index_with_primary and self._spatial_is_cached(*primary):
            return True

        # Check the same for the secondary data:
        if not self.index_with_primary and self._spatial_is_cached(*secondary):
            return False

        # Otherwise, just use the larger dataset:
        return primary[0].size > secondary[0].size

    def temporal_search(self, primary, secondary, max_interval):
        raise NotImplementedError("Not yet implemented!")
        #return self.no_pairs, self.no_intervals

    def _temporal_check(
            self, primary_time, secondary_time, max_interval
    ):
        """Checks whether the current collocations fulfill temporal conditions

        Returns:

        """
        intervals = self._get_intervals(primary_time, secondary_time)

        # Check whether the time differences are less than the temporal
        # boundary:
        passed_time_check = intervals < max_interval

        return passed_time_check, intervals[passed_time_check]

    @staticmethod
    def _get_intervals(time1, time2):
        return np.abs((time1 - time2)).astype("timedelta64[s]")

    @staticmethod
    def _get_distances(lat1, lon1, lat2, lon2):
        return great_circle_distance(lat1, lon1, lat2, lon2)

    def _create_return(
            self, primary, secondary, primary_name, secondary_name,
            original_pairs, intervals, distances,
            max_interval, max_distance
    ):
        if not original_pairs.any():
            return self.empty

        pairs = []
        output = {}

        # We are going to save the time coverage of the data as attributes in
        # the output dataset
        start, end = None, None

        names = [primary_name, secondary_name]
        for i, dataset in enumerate([primary, secondary]):
            # name of the current dataset (primary or secondary)
            name = names[i]

            # These are the indices of the points in the original data that
            # have collocations. We remove the duplicates since we want to copy
            # the required data only once:
            original_indices = pd.unique(original_pairs[i])

            # After selecting the collocated data, the original indices cannot
            # be applied any longer. We need new indices that indicate the
            # pairs in the collocated data.
            new_indices = pd.Series(
                np.arange(len(original_indices)),
                index=original_indices,
            )

            collocation_indices = new_indices.loc[original_pairs[i]].values

            # Save the collocation indices in the metadata group:
            pairs.append(collocation_indices)

            output[names[i]] = dataset.isel(collocation=original_indices)

            # xarrays does not really handle grouped data (actually, not at
            # all). Until this has changed, I do not want to have subgroups in
            # the output data (this makes things complicated when it comes to
            # coordinates). Therefore, we 'flat' each group before continuing:
            output[names[i]].rename(
                {
                    old_name: old_name.replace("/", "_")
                    for old_name in output[name].variables
                    if "/" in old_name
                }, inplace=True
            )

            # We need the total time coverage of all datasets for the name of
            # the output file
            data_start = pd.Timestamp(
                output[name]["time"].min().item(0)
            )
            data_end = pd.Timestamp(
                output[name]["time"].max().item(0)
            )
            if start is None or start > data_start:
                start = data_start
            if end is None or end < data_end:
                end = data_end

            # We have to convert the MultiIndex to a normal index because we
            # cannot store it to a file otherwise. We can convert it by simply
            # setting it to new values, but we are losing the sub-level
            # coordinates (the dimenisons that we stacked to create the
            # multi-index in the first place) with that step. Hence, we store
            # the sub-level coordinates in additional dataset to preserve them.
            # main_coord_is_multiindex = isinstance(
            #     output[name].get_index("collocation"),
            #     pd.core.indexes.multi.MultiIndex
            # )
            # if main_coord_is_multiindex:
            #     stacked_dims_data = xr.merge([
            #         xr.DataArray(
            #             output[name][dim].values,
            #             name=dim, dims=["collocation"]
            #         )
            #         for dim in output[name].get_index("collocation").names
            #     ])

            # Okay, actually we want to get rid of the main coordinate. It
            # should stay as a dimension name but without own labels. I.e. we
            # want to drop it. Because it still may a MultiIndex, we cannot
            # drop it directly but we have to set it to something different.
            output[name]["collocation"] = \
                np.arange(output[name]["collocation"].size)

            # if main_coord_is_multiindex:
            #     # Now, since we unstacked the multi-index, we can add the
            #     # stacked dimensions back to the dataset:
            #     output[name] = xr.merge(
            #         [output[name], stacked_dims_data],
            #     )

            # Now, we can rename it (to make it to a member of this group) and
            # then we can drop it.
            output[name].rename(
                {"collocation": f"{name}/collocation"}, inplace=True
            )
            output[name] = output[name].drop(f"{name}/collocation")

            # We want to merge all datasets together (but as subgroups). Hence,
            # add the fileset name to each dataset as prefix:
            output[name].rename(
                {
                    var_name: "/".join([name, var_name])
                    for var_name in output[name].variables
                }, inplace=True
            )

        # Merge all datasets into one:
        output = xr.merge(
            [data for data in output.values()]
        )

        # This holds the collocation information (pairs, intervals and
        # distances):
        output["Collocations/pairs"] = xr.DataArray(
            np.array(pairs, dtype=int), dims=("group", "collocation"),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": primary_name,
                "secondary": secondary_name,
            }
        )
        output["Collocations/interval"] = xr.DataArray(
            intervals, dims=("collocation", ),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": primary_name,
                "secondary": secondary_name,
            }
        )
        output["Collocations/distance"] = xr.DataArray(
            distances, dims=("collocation",),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": primary_name,
                "secondary": secondary_name,
                "units": "kilometers",
            }
        )
        output["Collocations/group"] = xr.DataArray(
            [primary_name, secondary_name], dims=("group",),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
            }
        )

        output.attrs = {
            "start_time": str(start),
            "end_time": str(end),
        }

        return output

