from datetime import datetime, timedelta

from netCDF4 import Dataset
import numpy as np
from typhon.spareice.geographical import GeoData
import xarray as xr

from .common import FileHandler, FileInfo, expects_file_info

__all__ = [
    'MHSAAPP',
    ]


class MHSAAPP(FileHandler):
    """File handler for MHS level 1C HDF files (convert with the AAPP tool.)
    """
    # This file handler always wants to return at least time, lat and lon
    # fields. These fields are required for this:
    standard_fields = {
        "Data/scnlintime",  # milliseconds since midnight
        "Data/scnlinyr",
        "Data/scnlindy",
        "Data/scnlin",
        "Geolocation/Latitude",
        "Geolocation/Longitude"
    }

    mapping = {
        "Geolocation/Latitude": "lat",
        "Geolocation/Longitude": "lon",
        "Data/scnlin": "scnline",
    }

    def __init__(self, mapping=None, apply_scaling=True, **kwargs):
        """

        Args:
            mapping: A dictionary of old and new names. The fields are going to
                be renamed according to it.
            apply_scaling: Apply the scaling parameters given in the
                variable's attributes to the data. Default is true.
            **kwargs: Additional key word arguments for base class.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        self.user_mapping = mapping
        self.apply_scaling = apply_scaling

    @expects_file_info
    def get_info(self, file_info, **kwargs):
        with Dataset(file_info.path, "r") as file:
            file_info.times[0] = \
                datetime(int(file.startdatayr[0]), 1, 1) \
                + timedelta(days=int(file.startdatady[0]) - 1) \
                + timedelta(
                    milliseconds=int(file.startdatatime_ms[0]))
            file_info.times[1] = \
                datetime(int(file.enddatayr), 1, 1) \
                + timedelta(days=int(file.enddatady) - 1) \
                + timedelta(milliseconds=int(file.enddatatime_ms))

            return file_info

    @expects_file_info
    def read(self, file_info, extra_fields=None, mapping=None):
        """"Read and parse HDF4 files and load them to an ArrayGroup.

        Args:
            file_info: Path and name of the file as string or FileInfo object.
            extra_fields: Additional field names that you want to extract from
                this file as a list.
            mapping: A dictionary that maps old field names to new field names.
                If given, *extra_fields* must contain the old field names.

        Returns:
            An ArrayGroup object.
        """

        if extra_fields is None:
            extra_fields = []

        fields = self.standard_fields | set(extra_fields)

        dataset = GeoData.from_netcdf(file_info.path, fields)
        dataset.name = "MHS"

        # We do the internal mapping first so we do not deal with difficult
        # names in the following loop.
        dataset.rename(self.mapping, inplace=True)

        # Handle the standard fields:
        dataset["time"] = self._get_time_field(dataset)

        # Flat the latitude and longitude vectors:
        dataset["lon"] = dataset["lon"].flatten()
        dataset["lat"] = dataset["lat"].flatten()

        # Repeat the scanline and create the scnpos:
        dataset["scnpos"] = np.tile(np.arange(1, 91), dataset["scnline"].size)
        dataset["scnline"] = np.repeat(dataset["scnline"], 90)

        # Remove fields that we do not need any longer (expect the user asked
        # for them explicitly)
        dataset.drop(
            {"Data/scnlinyr", "Data/scnlindy", "Data/scnlintime"}
            - set(extra_fields),
            inplace=True
        )

        # Some fields need special treatment
        for var in dataset.vars(deep=True):
            # Unfold the variable automatically if it is a swath variable.
            if len(dataset[var].shape) > 1 and dataset[var].shape[1] == 90:
                # Unfold the dimension of the variable
                # to the shapes of the time vector.
                dataset[var] = dataset[var].reshape(
                    -1, dataset[var].shape[-1]
                )

            # Some variables are scaled. If the user wants us to do
            # rescaling, we do it and delete the note in the attributes.
            if self.apply_scaling and "Scale" in dataset[var].attrs:
                dataset[var] = dataset[var] * dataset[var].attrs["Scale"]
                del dataset[var].attrs["Scale"]

            dataset[var].dims = ["time_id"]

        if mapping is not None:
            dataset.rename(mapping)

        return dataset

    @staticmethod
    def _get_time_field(dataset):
        swath_times = \
            dataset["Data/scnlinyr"].astype('datetime64[Y]') - 1970 \
            + dataset["Data/scnlindy"].astype('timedelta64[D]') - 1 \
            + dataset["Data/scnlintime"].astype("timedelta64[ms]")

        return np.repeat(swath_times, 90).flatten()
        # swath_times = \
        #
        #     dataset["Data/scnlintime"].astype("timedelta64[ms]")
        #
        # # Interpolate between the start times of each swath to retrieve the
        # # timestamp of each pixel.
        # swath_start_indices = np.arange(
        #     0, dataset["Data/scnlintime"].size * 90, 90)
        # pixel_times = np.interp(
        #     np.arange((dataset["Data/scnlintime"].size - 1) * 90),
        #     swath_start_indices, dataset["Data/scnlintime"])
        #
        # # Add the timestamps from the last swath here, we could not interpolate
        # # them in the step before because we do not have an ending timestamp.
        # # We simply extrapolate from the difference between the last two
        # # timestamps.
        # last_swath_pixels = \
        #     dataset["Data/scnlintime"][-1] \
        #     - dataset["Data/scnlintime"][-2] \
        #     + np.linspace(
        #         dataset["Data/scnlintime"][-2], dataset["Data/scnlintime"][-1],
        #         90, dtype="int32", endpoint=False)
        # pixel_times = np.concatenate([pixel_times, last_swath_pixels])
        #
        # # Convert the pixel times into timedelta objects
        # pixel_times = pixel_times.astype("timedelta64[ms]")
        #
        # # Convert the swath time variables (year, day of year, miliseconds
        # # since midnight) to numpy.datetime objects. These are the start times
        # # of each swath, we have to bring them together with the interpolated
        # # pixel times.
        # swath_times = \
        #     dataset["Data/scnlinyr"].astype('datetime64[Y]') - 1970 \
        #     + dataset["Data/scnlindy"].astype('timedelta64[D]') - 1
        #
        # swath_times = np.repeat(swath_times, 90)
        # times = swath_times + pixel_times
        #
        # return times.flatten()
