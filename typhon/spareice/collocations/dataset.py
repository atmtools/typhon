from collections import OrderedDict
import datetime
from multiprocessing import Pool
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial
import typhon.geodesy
from .. import datasets
import xarray as xr

__all__ = [
    "Dataset"
    ]


class Dataset(datasets.Dataset):

    def __init__(self, datasets, *args, **kwargs):
        """ Class that can find collocations amongst different datasets and store them.

        Collocations are match-ups between two datasets, i.e. a data point from a dataset that is located close to
        another data point from a secondary dataset in space and time.

        Since this class inherits from the typhon.datasets.Dataset class please look also at its documentation.

        Args:
            datasets: A list of datasets.Dataset() objects that you want to be collocated with each other. At the
                moment, it is only possible to find collocations between two different datasets at the same time. This
                will be changed in the future.
            *args: Additional arguments that will passed to the datasets.Dataset base class.
            **kwargs: Additional keyword arguments that will passed to the datasets.Dataset base class.

        Examples:
            import typhon.spareice.datasets as datasets
            import typhon.spareice.collocations as collocations

            # Define the two dataset amongst them you want to find collocations.
            dataset1 = datasets.Dataset(
                name="MHS",
                files="/path/to/files/noaa18_mhs_{year}/{month}/{day}/*.h5",
                file_handler=tovs.MHSAAPPFile(), # file_handler
                timestamp_retrieving_method="content"
            )
            dataset2 = datasets.Dataset(
                name="2C-ICE",
                files="/path2/to/files/{year}/{doy}/{year}{doy}{hour}{minute}{second}_*.hdf.zip",
                file_handler=cloudsat.CPR2CICEFile(),  # file_handler
            )

            # Create the collocated dataset. The parameters "files" and "file_handler" tells the object where the
            # collocations should be stored and which format should be used.
            collocated_dataset = collocations.Dataset(
                datasets=[dataset1, dataset2],
                name="CollocatedDataset",
                # The found collocations will be stored as NetCDF4 files in this path:
                files="CollocatedDataset/{year}/{month}/{day}/{hour}{minute}{second}.nc",
                # If you need another data format than NetCDF4 then pass another file handler object:
                file_handler=common.NetCDF4File(),
            )

            # Define the period where you want to collocate.
            start, end = datetime.datetime(2007, 1, 1), datetime.datetime(2007, 1, 2)

            # Find the collocations with a maximum distance of 200km and temporal interval of 300 seconds. Extract the
            # field "brightness_temperature" from the primary and "ice_water_path" from the secondary dataset.
            collocated_dataset.collocate(
                start, end,
                fields=[["brightness_temperature"], ["ice_water_path"]],
                max_distance=200, max_interval=300,
            )
        """
        super().__init__(**kwargs)

        # Initialize member variables:
        if len(datasets) != 2:
            raise ValueError("Need two datasets to perform collocations!")
        self.datasets = datasets

    def collapse(self, start, end, field, collapser=np.nanmean, **kwargs):
        """ Accumulates the data as Dataset.accumulate() between two dates but collapses multiple collocations from one
        dataset to a single data point.

        During searching for collocations, one might find multiple collocation points of the secondary dataset for one
        single point of the primary dataset. For example, the MHS instrument has a bigger footprint than the AVHRR
        instrument, hence one will find several AVHRR colloocation points for each MHS data point.

        Args:
            start: Starting date as datetime.datetime object.
            end: Ending date as datetime.datetime object.
            fields: List or tuple of variables to select.
            collapser: Collapser function that should be applied on the multiple collocation points. Default is
                numpy.nanmean.
            **kwargs: All additional arguments that are valid for Dataset.accumulate as well.

        Yields:
            datasets.AccumulatedData objects with the maximum chunk size.

        Examples:
        """
        raise NotImplementedError("This method has not been implemented yet.")


    def collocate(self, start, end, fields, max_interval=300, max_distance=10, **kwargs):
        """This function finds all collocations between two dates and store them to files.

        Where and how this function stores the collocations is controlled by the Dataset.files and Dataset.file_handler
        parameters. Each collocation output file provides these standard fields for each dataset:

        dataset_name.lat - Latitudes of the collocations.
        dataset_name.lon - Longitude of the collocations.
        dataset_name.time - Timestamp of the collocations.
        dataset_name.indices - Indices of the collocation data in the original files.
        dataset_name.collocations - Tells you which data points collocate with each other by giving their indices to
            you.

        For example, if you collocated the datasets "MHS" and "CloudSat" then you will find the latitude information of
        their collocations in "MHS.lat" and "CloudSat.lat". If you want to copy additional fields from the original
        files, you can use the fields parameter.
        
        Args:
            start: Start date as datetime.datetime object with year, month and day. Hours, minutes and seconds are
                optional.
            end: End date. Same format as "start".
            fields: The fields that should be extracted from the datasets and copied to the new collocation files. This
                should be a list of two lists containing the field names for each dataset. They will be accessible via
                "dataset_name.field_name" in the new collocation files. If you want to learn more about how to extract
                specific dimension from those fields, please read the documentation of the typhon.handlers classes.
            max_interval: The maximum interval of time between two data points in seconds. As alternative you can give a
                datetime.timedelta object.
            max_distance: The maximum distance between two data points in kilometers to meet the collocation criteria.
            **kwargs: Additional arguments that will be passed to Dataset.find_collocations(..).

        Returns:
            None

        Examples:

        """

        # Check whether are some datasets to collocate:
        if self.datasets is None or len(self.datasets) != 2:
            raise ValueError("Need two datasets to perform collocations! "
                             "You can set them via the attribute 'datasets'.")

        # Find all files from the primary dataset in the given period and sort them.
        primaries = sorted(list(self.datasets[0].find_files(start, end)), key=lambda x: x[1])

        # All additional fields given by the user and the standard fields that we need for finding collocations.
        primary_fields = list(set(fields[0]) | {"time", "lat", "lon"})
        secondary_fields = list(set(fields[1]) | {"time", "lat", "lon"})

        # The data of secondary files can overlap multiple primary files.
        last_secondary = None
        secondary_data_cache = None

        # Use this variable to store the point in time to which the were checked. This avoids checking duplicates.
        last_timestamp = None

        total_primaries_points, total_secondaries_points = 0, 0

        # Convert the given time intervall if necessary.
        if not isinstance(max_interval, datetime.timedelta):
            max_interval = datetime.timedelta(seconds=max_interval)

        # Use a timer for profiling.
        timer = time.time()

        # Go through all primary files:
        for index, primary in enumerate(primaries):
            # primary is a tuple of name and time coverage of the primary file.
            print("Primary:", primary[0])

            # The starting time of the primary file is provided by .find_files(..). Since we assume a continuous
            # dataset, we can use the starting time of the next primary file as ending time for the first. For the last
            # primary file, we use the date which is given by the "end" parameter as ending time.
            primary_start = primary[1] - max_interval
            primary_end = primary[2] + max_interval if index != len(primaries) - 1 else end

            # Find all secondary files in this time range.
            secondaries = sorted(self.datasets[1].find_files(primary_start, primary_end), key=lambda x: x[1])

            # Skip this primary file if there are no secondaries found.
            if not secondaries:
                continue

            # Read the data of the primary file
            primary_data_cache = self.datasets[0].read(primary[0], primary_fields)

            # Go through all potential collocated secondary files:
            for secondary in secondaries:
                print("Secondary:", secondary[0], "\nLast secondary: ", last_secondary)

                # Only read in this secondary if you have not read it already the last time!
                if last_secondary is None or last_secondary != secondary:
                    secondary_data_cache = self.datasets[1].read(secondary[0], secondary_fields)
                else:
                    print("This secondary is still in cache.")

                # Select only the time period in the secondary data where also primary data is available.
                if last_timestamp is None:
                    secondary_data = secondary_data_cache.sel(
                        time=slice(primary_start, primary_end)
                    )
                else:
                    # May be we checked a part of this time period already? Hence, start there where we stopped last
                    # time.
                    secondary_data = secondary_data_cache.sel(
                        time=slice(last_timestamp, primary_end)
                    )

                # The "new" start and end times of the secondary data (the time conversion is a little bit tricky due to
                # a bug in numpy).
                secondary_start = pd.to_datetime(str(secondary_data["time"].min().data))
                secondary_end = pd.to_datetime(str(secondary_data["time"].max().data))

                print("Primary times", primary_start, primary_end)
                print("Secondary times", secondary_start, secondary_end)
                print(primary_data_cache["time"], secondary_data["time"])
                print(primary_data_cache["time"].min().data, primary_data_cache["time"].max().data)

                print(slice(secondary_start, secondary_end))

                # Select only the primary data that is in the same time range as the secondary data.
                primary_data = primary_data_cache.sel(
                    time=slice(secondary_start, secondary_end)
                )

                # We are going to check the data until secondary_end. Hence, we save this timestamp to avoid duplicates
                # the next time.
                last_secondary = secondary
                last_timestamp = secondary_end

                # No collocations have been found.
                if len(primary_data) == 0 or len(secondary_data) == 0:
                    continue

                print("\tTaking {} {} and {} {} points for collocation search.".format(
                    len(primary_data["lat"]), self.datasets[0].name,
                    len(secondary_data["lat"]), self.datasets[1].name,
                ))

                # Find the collocations
                primary_indices, primary_collocation_indices, secondary_indices, secondary_collocation_indices = \
                    self.find_collocations(primary_data, secondary_data, max_interval, max_distance, **kwargs)

                # Found no collocations
                if not primary_indices:
                    continue

                # Retrieve with the indices the collocation points from the data.
                primary_data = primary_data.isel(time=primary_indices)
                secondary_data = secondary_data.isel(time=secondary_indices)

                # Merge the two datasets to one collocation dataset:
                collocation_data = datasets.AccumulatedData.merge([primary_data, secondary_data])

                # Prepare the data array to write it to a file.
                collocation_data.attrs["files"] = [primary[0], secondary[0]]
                collocation_data.attrs["datasets"] = [self.datasets[0].name, self.datasets[1].name]
                collocation_data.attrs["max_interval"] = [max_interval.total_seconds(), "Max. time interval in seconds."]
                collocation_data.attrs["max_distance"] = [max_distance, "Max. spatial distance in kilometers."]

                # Add the indices
                collocation_data[self.datasets[0].name + ".indices"] = xr.DataArray(
                    primary_indices, dims=[self.datasets[0].name + ".time"],
                    attrs={"description": "Indices of the primary data in the original files."})
                collocation_data[self.datasets[1].name + ".indices"] = xr.DataArray(
                    secondary_indices, dims=[self.datasets[1].name + ".time"],
                    attrs={"description": "Indices of the secondary data in the original files."})
                collocation_data[self.datasets[0].name + ".collocations"] = xr.DataArray(
                    primary_collocation_indices, dims=["collocation_id"], attrs={
                        "description": "Collocation indices of the primary data. Each element indicates the index of "
                                       "the primary data of the corresponding collocation."})
                collocation_data[self.datasets[1].name + ".collocations"] = xr.DataArray(
                    secondary_collocation_indices, dims=["collocation_id"], attrs={
                        "description": "Collocation indices of the secondary data. Each element indicates the index of "
                                       "the secondary data of the corresponding collocation."})

                filename = self.generate_filename_from_time(
                    self.files,
                    datetime.datetime.utcfromtimestamp(primary_data["time"][0].astype('O')/1e9))

                # Write the data to the file.
                print("\tWrite collocations to '{0}'".format(filename))
                self.write(filename, collocation_data)

                # Plot the collocations
                """fig = plt.figure(figsize=(8, 6), dpi=80, )
                collocation_data.plot("collocations", fields=("time", ), s=10, cmap="qualitative1")
                plt.savefig("plots/overview%d.png" % index)
                plt.close(fig)

                fig = plt.figure(figsize=(8, 6), dpi=80, )
                collocation_data.plot("collocations", s=10, cmap="qualitative1")
                plt.savefig("plots/map%d.png" % index)
                plt.close(fig)"""

                total_primaries_points += len(primary_indices)
                total_secondaries_points += len(secondary_indices)

        print("Needed {0:.2f} seconds to find {1} ({2}) and {3} ({4}) collocation points for a period of {5}.".format(
            time.time()-timer, total_primaries_points,
            self.datasets[0].name, total_secondaries_points, self.datasets[1].name,
            end-start)
        )


    @staticmethod
    def find_collocations(
            primary_data, secondary_data,
            max_interval=300, max_distance=10,
            processes=8, check_temporal_boundaries=True):
        """Finds collocations between two xarray.Datasets.

        TODO: Rename this function. Improve this doc string.

        Args:
            primary_data: The data from the primary dataset as xarray.Dataset object. This object must provide the
                special fields "lat", "lon" and "time" as 1-dimensional arrays.
            secondary_data: The data from the secondary dataset as xarray.Dataset object. Needs to meet the same
                conditions as primary_data ("lat", "lon" and "time" fields).
            max_interval: The maximum interval of time between two data points in seconds. As well you can give here a
                datetime.timedelta object.
            max_distance: The maximum distance between two data points in kilometers to meet the collocation criteria.
            processes: The number of processes that should be used to boost the collocation process. The
                optimal number of processes heavily depends on your machine where you are working. I recommend to start
                with 8 processes and to in/decrease this parameter when lacking performance.
            check_temporal_boundaries: Whether to check the temporal interval as well. If this is false, the data will
                be searched for spatial collocations only.

        Returns:
            Four lists:
            (1) primary indices - The indices of the data in primary_data that meet the collocation conditions.
            (2) primary collocations - All collocation points of the primary data. Each element is an index of an
                element in the primary indices list (1).
            (3) secondary indices - The indices of the data in secondary_data that meet the collocation conditions.
            (4) secondary collocations -

        """

        timer = time.time()

        if not isinstance(max_interval, datetime.timedelta):
            max_interval = datetime.timedelta(seconds=max_interval)

        # We try to find collocations by building one 3-d tree for each dataset (see
        # https://en.wikipedia.org/wiki/K-d_tree) and searching for the nearest neighbours. Since a k-d tree cannot
        # handle latitude/longitude data, we have to convert them to 3D-cartesian coordinates.
        x, y, z = typhon.geodesy.geocentric2cart(
            typhon.constants.earth_radius, np.asarray(primary_data["lat"]), np.asarray(primary_data["lon"])
        )

        primary_points = np.asarray(list(zip(x, y, z)))

        # We need to convert the secondary data as well:
        x, y, z = typhon.geodesy.geocentric2cart(
            typhon.constants.earth_radius, np.asarray(secondary_data["lat"]), np.asarray(secondary_data["lon"])
        )
        secondary_points = np.asarray(list(zip(x, y, z)))

        print("\tAfter {:.2f}s: finished the conversion from longitude/latitudes to cartesian.".format(
            timer - time.time()))

        # We want to use parallel processes to find spatial collocations since it can take a lot of time. Hence, split
        # the secondary data into N chunks, where N is the number of parallel processes to use.
        #if secondary_points.shape[0] > 500:
        #    secondary_points_chunks = np.array_split(secondary_points, processes)
        #else:
        #    secondary_points_chunks = [secondary_points]

        # We build the offset indices that indicate at which index a chunk starts originally. We need those later to
        # identify the collocations in the original data.
        #offset_indices = [len(chunk) for chunk in secondary_points_chunks]
        #offset_indices.insert(0, 0)
        #offset_indices.pop()
        #offset_indices = [sum(offset_indices[:i+1]) for i, _ in enumerate(offset_indices)]

        # Prepare the k-d tree for the priary dataset:
        primary_tree = scipy.spatial.cKDTree(primary_points, leafsize=5)
        print("\tAfter {:.2f}s: finished building primary tree.".format(
            timer - time.time()))

        # To avoid defining complicated distance metrics in the k-d tree, we use the tree for checking for spatial
        # collocations only. All found spatial collocations will be checked by using temporal boundaries later.
        #pool = Pool(processes=processes)
        #results = pool.map(
        #    Dataset._call_map_function,
        #    [(Dataset._find_spatial_collocations, None,
        #      (i, primary_tree, secondary_points_chunk, max_distance), None)
        #     for i, secondary_points_chunk in enumerate(secondary_points_chunks)])

        sequence_id, result = Dataset._find_spatial_collocations(0, primary_tree, secondary_points, max_distance)
        print("\tAfter {:.2f}s: finished finding spatial collocations.".format(
            timer - time.time()))

        primary_collocation_indices, secondary_collocation_indices  = [], []

        # Merge all result lists.
        """for sequence_id, result in sorted(results, key=lambda x: x[0]):

            # Build the list of the collocation indices. The index of the list element denotes the collocation id.
            # The element is the original index of the primary / secondary data to the corresponding collocation. For
            # example, if there is at least one primary point that collocates with multiple secondary points,
            # the primary list contains duplicates.
            primary_collocation_indices.extend([
                primary_index
                for primary_index, found in enumerate(result)
                for _ in range(len(found))
            ])
            secondary_collocation_indices.extend([
                secondary_index + offset_indices[sequence_id]
                for found in result
                for secondary_index in found
            ])"""

        # Build the list of the collocation indices. The index of the list element denotes the collocation id.
        # The element is the original index of the primary / secondary data to the corresponding collocation. For
        # example, if there is at least one primary point that collocates with multiple secondary points,
        # the primary list contains duplicates.
        primary_collocation_indices.extend([
            primary_index
            for primary_index, found in enumerate(result)
            for _ in range(len(found))
        ])
        secondary_collocation_indices.extend([
            secondary_index# + offset_indices[sequence_id]
            for found in result
            for secondary_index in found
        ])

        # No collocations were found.
        if not primary_collocation_indices:
            print("\tNo collocations were found.")
            return None, None, None, None

        # Convert them to arrays to make them better sliceable.
        primary_collocation_indices = np.asarray(primary_collocation_indices)
        secondary_collocation_indices = np.asarray(secondary_collocation_indices)

        print("\tAfter {:.2f}s: finished retrieving the indices.".format(
            timer - time.time()))

        # Check here for temporal collocations:
        if check_temporal_boundaries:
            # Convert them from datetime objects to timestamps (seconds since 1970-1-1) to make them faster for
            # temporal distance checking.
            primary_time = np.asarray(primary_data["time"].astype('uint64')) / 1e9
            secondary_time = np.asarray(secondary_data["time"].astype('uint64')) / 1e9

            # Check whether the time differences between the spatial collocations are less than the temporal boundary:
            collocation_indices = \
                np.abs(primary_time[primary_collocation_indices] - secondary_time[secondary_collocation_indices]) \
                < max_interval.total_seconds()

            # Just keep all indices which satisfy the temporal condition.
            primary_collocation_indices = primary_collocation_indices[collocation_indices]
            secondary_collocation_indices = secondary_collocation_indices[collocation_indices]

        # No collocations were found.
        if not primary_collocation_indices.any():
            print("\tNo collocations were found. Took {:.2f}s.".format(time.time() - timer))
            return None, None, None, None

        # We are going to return four lists, two for each dataset. primary_indices and secondary_indices are the lists
        # with the indices of the data in the original data arrays. So if someone wants to retrieve other data from the
        # original files one day, he can use those indices. primary_collocations and secondary_collocations contain the
        # collocation pairs. The first element of the primary_collocations list corresponds to the first element of the
        # secondary_collocations and so on. Their elements are the indices of the primary_indices and secondary_indices
        # elements. For example, if someone wants to compare the collocations between primary and secondary, one could
        # use these expressions:
        #       primary_data[primary_indices[primary_collocations]] and
        #       secondary_data[secondary_indices[secondary_collocations]]
        primary_indices, primary_collocations, secondary_indices, secondary_collocations = [], [], [], []

        # Simply remove all duplicates from the collocation_indices lists while keeping the order:
        primary_indices = list(OrderedDict.fromkeys(primary_collocation_indices))
        secondary_indices = list(OrderedDict.fromkeys(secondary_collocation_indices))

        #print(primary_collocation_indices, secondary_collocation_indices)

        # We want the collocation lists (primary_collocations and secondary_collocations) to show the indices of the
        # indices lists (primary_indices and secondary_indices). For example, primary_collocation_indices shows
        # [44, 44, 44, 103, 103, 109] at the moment, but we want it to convert to [0, 0, 0, 1, 1, 2] where the numbers
        # correspond to the indices of the elements in primary_indices which are [44, 103, 109].
        primary_index_ids = {primary_index: xid for xid, primary_index in enumerate(primary_indices)}
        secondary_index_ids = {secondary_index: xid for xid, secondary_index in enumerate(secondary_indices)}

        primary_collocations = [primary_index_ids[primary_index] for primary_index in primary_collocation_indices]
        secondary_collocations = [secondary_index_ids[secondary_index] for secondary_index in secondary_collocation_indices]

        print("\tFound {} primary and {} secondary collocations in {:.2f}s.".format(
            len(primary_indices), len(secondary_indices), time.time() - timer))

        return primary_indices, primary_collocations, secondary_indices, secondary_collocations

    @staticmethod
    def _find_spatial_collocations(sequence_id, primary_tree, secondary_points, max_distance):
        """ This finds spatial collocations between the primary and secondary data points and it does not regard the
        temporal dimension.

        One should not call this function directly but .find_collocations() which offers a performance boost by using
        multiple processes.

        Args:
            sequence_id: The process/worker id. Since this function is designed to called by multiple processes, this
                parameter is used to assign the results to the corresponding process.
            primary_tree: The primary data points, as scipy.spatial.cKDTree object.
            secondary_data: The secondary data points, an array of arrays with three elements: [x, y, z]
            max_distance: The maximum distance between two data points in kilometers to meet the collocation criteria.

        Returns:
            A list with the indices of the primary and secondary data that collocate with each other.
        """

        secondary_tree = scipy.spatial.cKDTree(secondary_points, leafsize=2)

        # Search for all collocations. This returns a list of lists. The index of each element is the
        # index of the primary data and each element is a list of the indices of the collocated secondary data.
        # The parameter max_distance is in kilometers, so we have to convert it to meters.
        collocation_indices = primary_tree.query_ball_tree(secondary_tree, max_distance * 1000)

        return sequence_id, collocation_indices
