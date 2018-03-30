from datetime import datetime, timedelta

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from typhon.collections import DataGroup
import xarray as xr

from .common import FileHandler, expects_file_info

__all__ = ['MHSAAPP', ]


class MHSAAPP(FileHandler):
    """File handler for MHS level 1C HDF files (converted with the AAPP tool.)
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

    def __init__(self, mapping=None, apply_scaling=True,
                 **kwargs):
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

    @expects_file_info()
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

    @expects_file_info()
    def read(self, file_info, extra_fields=None, mapping=None):
        """"Read and parse HDF4 files and load them to an DataGroup.

        Args:
            file_info: Path and name of the file as string or FileInfo object.
            extra_fields: Additional field names that you want to extract from
                this file as a list.
            mapping: A dictionary that maps old field names to new field names.
                If given, *extra_fields* must contain the old field names.

        Returns:
            An DataGroup object.
        """

        dataset = DataGroup.from_netcdf(
            file_info.path, mask_and_scale=True
        )
        dataset.name = "MHS"
        dataset["Data"][:].rename({
            "phony_dim_0": "swath",
            "phony_dim_1": "pixel",
            "phony_dim_2": "channel",
        }, inplace=True)
        dataset["Geolocation"][:].rename({
            "phony_dim_3": "swath",
            "phony_dim_4": "pixel",
        }, inplace=True)

        if extra_fields is None:
            fields = self.standard_fields
        else:
            fields = self.standard_fields | set(extra_fields)
        dataset = dataset[fields]

        # We do the internal mapping first so we do not deal with difficult
        # names in the following loop.
        dataset.rename(self.mapping, inplace=True)

        # Create the time coordinate but do not add it to the xarray yet since
        # it may contain duplicated values and prevent stacking from other
        # variables (I do not know why);
        time_coords = self._get_time_field(dataset)

        # Repeat the scanline and create the scnpos:
        dataset["scnpos"] = \
            "time", np.tile(np.arange(1, 91), dataset["scnline"].size)
        dataset["scnline"] = "time", np.repeat(dataset["scnline"], 90)

        # Remove fields that we do not need any longer (expect the user asked
        # for them explicitly)
        dataset.drop(
            {"Data/scnlinyr", "Data/scnlindy", "Data/scnlintime"}
            - set(extra_fields),
            inplace=True
        )

        # Some fields need special treatment
        for var in dataset.deep():
            # Unfold the variable automatically if it is a swath variable.
            if "swath" in dataset[var].dims and "pixel" in dataset[var].dims:
                # Unfold the dimension of the variable
                # to the shapes of the time vector.
                dataset[var] = dataset[var].stack(time=("swath", "pixel"))

            scaling = dataset[var].attrs.pop("Scale", None)
            if scaling is not None:
                dataset[var] = dataset[var].astype(float) * scaling

        # Add the time coordinate. We cannot add it earlier since it may
        # contain duplicated values:
        dataset["time"] = time_coords

        if mapping is not None:
            dataset.rename(mapping)

        if not dataset["Geolocation"]:
            del dataset["Geolocation"]

        return dataset

    @staticmethod
    def _get_time_field(dataset):
        swath_times = \
            (dataset["Data/scnlinyr"]-1970).astype('datetime64[Y]') \
            + (dataset["Data/scnlindy"]-1).astype('timedelta64[D]') \
            + dataset["Data/scnlintime"].astype("timedelta64[ms]")

        return "time", np.repeat(swath_times, 90).values.ravel()
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
