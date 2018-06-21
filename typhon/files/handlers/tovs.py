from datetime import datetime, timedelta
import time

import numpy as np
import numexpr as ne
from netCDF4 import Dataset
import xarray as xr

from .common import NetCDF4, expects_file_info

__all__ = [
    'AAPP_HDF',
    'AVHRR_GAC_HDF',
    'MHS_HDF',
]


class AAPP_HDF(NetCDF4):
    """Base class for handling TOVS satellite data converted with AAPP tools

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

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs: Additional key word arguments for base class.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

    @expects_file_info()
    def get_info(self, file_info, **kwargs):
        with Dataset(file_info.path, "r") as file:
            file_info.times[0] = \
                datetime(int(file.startdatayr[0]), 1, 1) \
                + timedelta(days=int(file.startdatady[0]) - 1) \
                + timedelta(milliseconds=int(file.startdatatime_ms[0]))
            file_info.times[1] = \
                datetime(int(file.enddatayr), 1, 1) \
                + timedelta(days=int(file.enddatady) - 1) \
                + timedelta(milliseconds=int(file.enddatatime_ms))

            return file_info

    @staticmethod
    def _get_time_field(dataset, user_fields):
        time = \
            (dataset["Data/scnlinyr"].values - 1970).astype('datetime64[Y]') \
            + (dataset["Data/scnlindy"].values - 1).astype('timedelta64[D]') \
            + dataset["Data/scnlintime"].values.astype("timedelta64[ms]")

        dataset["time"] = "scnline", time

        # Remove the time fields that we do not need any longer (expect the
        # user asked for them explicitly)
        dataset = dataset.drop(
            {"Data/scnlinyr", "Data/scnlindy", "Data/scnlintime"}
            - set(user_fields),
        )

        return dataset

    @staticmethod
    def _mask_and_scale(dataset):
        # xarray.open_dataset can mask and scale automatically, but it does not
        # know the attribute *Scale* (which is specific for AAPP files):
        for var in dataset.variables:
            scaling = dataset[var].attrs.pop("Scale", None)
            if scaling is not None:
                dataset[var] = dataset[var].astype(float) * scaling

    def _test_coords(self, dataset, wanted=None):
        # Maximal these dimensions (or less) should be in the dataset:
        if wanted is None:
            wanted = {'channel', 'scnline', 'scnpos'}
        reality = set(dataset.dims.keys())

        if reality - wanted:
            raise ValueError(
                f"Unexpected dimension in AAPP file! {reality - wanted}"
            )


class MHS_HDF(AAPP_HDF):
    """File handler for MHS level 1C HDF files
    """
    def __init__(self, **kwargs):
        super(MHS_HDF, self).__init__(**kwargs)

        # Map the standard fields to standard names (make also the names of all
        # dimensions more meaningful):
        self.mapping = {
            "Geolocation/Latitude": "lat",
            "Geolocation/Longitude": "lon",
            "Data/scnlin": "scnline",
            "Data/phony_dim_0": "scnline",
            "Data/phony_dim_1": "scnpos",
            "Data/phony_dim_2": "channel",
            "Geolocation/phony_dim_3": "scnline",
            "Geolocation/phony_dim_4": "scnpos",
        }

    @expects_file_info()
    def read(self, paths, mask_and_scale=True, **kwargs):
        """Read and parse MHS AAPP HDF5 files and load them to xarray

        Args:
            paths: Path and name of the file as string or FileInfo object.
                This can also be a tuple/list of file names or a path with
                asterisk.
            mask_and_scale: Where the data contains missing values, it will be
                masked with NaNs. Furthermore, data with scaling attributes
                will be scaled with them.
            **kwargs: Additional keyword arguments that are valid for
                :class:`~typhon.files.handlers.common.NetCDF4`.

        Returns:
            A xrarray.Dataset object.
        """

        # Make sure that the standard fields are always gonna be imported:
        user_fields = kwargs.pop("fields", {})
        if user_fields:
            fields = self.standard_fields | set(user_fields)
        else:
            fields = None

        # We catch the user mapping here, since we do not want to deal with
        # user-defined names in the further processing. Instead, we use our own
        # mapping
        user_mapping = kwargs.pop("mapping", None)

        # Load the dataset from the file:
        dataset = super().read(
            paths, fields=fields, mapping=self.mapping,
            mask_and_scale=mask_and_scale, **kwargs
        )

        dataset = dataset.assign_coords(
            scnline=dataset["scnline"]
        )

        dataset["scnline"] = np.arange(1, dataset["scnline"].size + 1)
        dataset["scnpos"] = np.arange(1, 91)

        # Create the time variable (is built from several other variables):
        dataset = self._get_time_field(dataset, user_fields)

        if mask_and_scale:
            self._mask_and_scale(dataset)

        # Make a fast check whether everything is alright
        self._test_coords(dataset)

        if user_mapping is not None:
            dataset.rename(user_mapping, inplace=True)

        return dataset


class AVHRR_GAC_HDF(AAPP_HDF):
    """File handler for AVHRR GAC level 1C HDF files
    """

    def __init__(self, **kwargs):
        super(AVHRR_GAC_HDF, self).__init__(**kwargs)

        # Map the standard fields to standard names (make also the names of all
        # dimensions more meaningful):
        self.mapping = {
            "Geolocation/Latitude": "lat",
            "Geolocation/Longitude": "lon",
            "Data/scnlin": "scnline",
            "Data/phony_dim_0": "scnline",
            "Data/phony_dim_1": "scnpos",
            "Data/phony_dim_2": "channel",
            "Data/phony_dim_3": "calib",
            "Geolocation/phony_dim_4": "scnline",
            "Geolocation/phony_dim_5": "packed_pixels",
        }

    @expects_file_info()
    def read(self, paths, mask_and_scale=True, **kwargs):
        """Read and parse MHS AAPP HDF5 files and load them to xarray

        Args:
            paths: Path and name of the file as string or FileInfo object.
                This can also be a tuple/list of file names or a path with
                asterisk.
            mask_and_scale: Where the data contains missing values, it will be
                masked with NaNs. Furthermore, data with scaling attributes
                will be scaled with them.
            **kwargs: Additional keyword arguments that are valid for
                :class:`~typhon.files.handlers.common.NetCDF4`.

        Returns:
            A xrarray.Dataset object.
        """

        # Make sure that the standard fields are always gonna be imported:
        user_fields = kwargs.pop("fields", {})
        if user_fields:
            fields = self.standard_fields | set(user_fields)
        else:
            fields = None

        # We catch the user mapping here, since we do not want to deal with
        # user-defined names in the further processing. Instead, we use our own
        # mapping
        user_mapping = kwargs.pop("mapping", None)

        # Load the dataset from the file:
        dataset = super().read(
            paths, fields=fields, mapping=self.mapping,
            mask_and_scale=mask_and_scale, **kwargs
        )

        dataset = dataset.assign_coords(
            scnline=dataset["scnline"]
        )

        dataset["scnline"] = np.arange(1, dataset["scnline"].size+1)
        dataset["scnpos"] = np.arange(1, 2049)

        # Create the time variable (is built from several other variables):
        dataset = self._get_time_field(dataset, user_fields)

        if mask_and_scale:
            self._mask_and_scale(dataset)

        # All geolocation fields are packed in the AVHRR GAC files:
        self._interpolate_packed_pixels(dataset)

        # Make a fast check whether everything is alright
        self._test_coords(dataset, {'channel', 'calib', 'scnline', 'scnpos'})

        if user_mapping is not None:
            dataset.rename(user_mapping, inplace=True)

        return dataset

    @staticmethod
    def _interpolate_packed_pixels(dataset):
        # We have 51 lat/lon pairs for each scanline starting from the 25th and
        # ending at the 2025th pixel, i.e. we have to interpolate all pixels
        # between them and extrapolate the pixels 1-24 and 2026-2048. We could
        # apply scipy.interpolate.interp1d to each scanline but this would take
        # around 6-7 seconds. This approach is faster:

        # The pixel number of each grid point (including the boundaries)
        grid_pixel = np.zeros(53, dtype=int)
        grid_pixel[1:-1] = np.arange(1, 52) * 40 - 16
        grid_pixel[0] = 0
        grid_pixel[-1] = 2047

        # The pixels for that we want to have the interpolated grid values,
        # i.e. actually all pixels
        all_pixels = np.arange(2048)

        # The index of the closest grid point on the left side
        left = all_pixels // 40
        # The index of the closest grid point on the right side
        right = left + 1

        # And the pixel number for left and right
        all_pixels_ratio = (
            (all_pixels - grid_pixel[left])
            / (grid_pixel[right] - grid_pixel[left])
        )

        for var_name, var in dataset.data_vars.items():
            if "packed_pixels" not in var.dims:
                continue
            data = AVHRR_GAC_HDF._interpolate(
                var.values, left, right, all_pixels_ratio
            )
            dataset[var_name] = ("scnline", "scnpos"), data

    @staticmethod
    def _interpolate(grid, left, right, all_pixels_ratio):
        # Add the boundary values (we extrapolate here):
        left_boundary = grid[:, 0] + (grid[:, 0] - grid[:, 1]) / 40 * 25
        right_boundary = grid[:, -1] + (grid[:, -1] - grid[:, -2]) / 40 * 23
        grid = np.column_stack([left_boundary, grid, right_boundary])

        # And the values of the grid for the left and right point (we select
        # here the values once, so we do not have to do it multiple times):
        grid_left = grid[:, left]
        grid_right = grid[:, right]

        # Calculate the gradient between the closest left and right grid point,
        # multiply it with the ratio of each pixel and add it to the value of
        # the left grid point. The ratio is between 0 and 1 and describes
        # whether the pixel is closer to its left grid point or to its right
        # one.
        return ne.evaluate(
            "all_pixels_ratio * (grid_right - grid_left) + grid_left"
        )
