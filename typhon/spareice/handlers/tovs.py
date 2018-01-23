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
    standard_fields = [
        "Data/scnlintime",  # milliseconds since midnight
        "Data/scnlinyr",
        "Data/scnlindy",
        "Data/scnlin",
        "Geolocation/Latitude",
        "Geolocation/Longitude"
    ]

    mapping = {
        "Geolocation/Latitude": "lat",
        "Geolocation/Longitude": "lon",
        "Data/scnlin": "scnline",
    }
    inv_mapping = {v: k for k, v in mapping.items()}

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
    def read(self, file_info, fields=None):
        """Reads and parses NetCDF files and load them to a GeoData object.

        TODO: Extend documentation.
        """

        if fields is None:
            fields_to_extract = self.standard_fields
        else:
            mapped_fields = [
                self.inv_mapping.get(field, field)
                for field in fields
            ]
            fields_to_extract = mapped_fields + self.standard_fields
            fields_to_extract = set(fields_to_extract)
            for field in ["time", "scnpos"]:
                try:
                    fields_to_extract.remove(field)
                except KeyError:
                    pass
            fields_to_extract = list(fields_to_extract)

        dataset = GeoData.from_netcdf(file_info.path, fields_to_extract)
        dataset.name = "MHS"

        # We do the internal mapping first so we do not deal with difficult
        # names in the following loop.
        dataset.rename(self.mapping, inplace=True)

        # Add standard field "time":
        dataset["time"] = self._get_time_field(dataset)

        # Flat the latitude and longitude vectors:
        dataset["lon"] = dataset["lon"].flatten()
        dataset["lat"] = dataset["lat"].flatten()

        # Repeat the scanline and create the scnpos:
        dataset["scnpos"] = np.tile(np.arange(1, 91), dataset["scnline"].size)
        dataset["scnline"] = np.repeat(dataset["scnline"], 90)

        # Some fields need special treatment
        vars_to_drop = []
        for var in dataset.vars(deep=True):
            # Some variables have been loaded only for temporary reasons.
            if (var in self.standard_fields
                    or (fields is not None and var not in fields)):
                vars_to_drop.append(var)

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

        dataset.drop(vars_to_drop, inplace=True)

        if self.user_mapping is not None:
            dataset.rename(self.user_mapping)
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
