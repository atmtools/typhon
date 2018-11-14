from datetime import datetime, timedelta

import numpy as np
import numexpr as ne
from netCDF4 import Dataset
from scipy.interpolate import CubicSpline
from typhon.utils import Timer
import xarray as xr

from .common import NetCDF4, expects_file_info
from .testers import check_lat_lon

__all__ = [
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
            # We want to remove some attributes after applying them but
            # OrderedDict does not allow to pop the values:
            attrs = dict(dataset[var].attrs)

            mask = attrs.pop('FillValue', None)
            if mask is not None:
                dataset[var] = dataset[var].where(
                    # Also cover overflow errors as they are in
                    # NSS.MHSX.NN.D07045.S2234.E0021.B0896162.GC.h5
                    (dataset[var] != mask) & (dataset[var] != -2147483648.0)
                )

            scaling = attrs.pop('Scale', None)
            if scaling is not None:
                dataset[var] = dataset[var].astype(float) * scaling

            dataset[var].attrs = attrs

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
    def read(self, file_info, mask_and_scale=True, **kwargs):
        """Read and parse MHS AAPP HDF5 files and load them to xarray

        Args:
            file_info: Path and name of the file as string or FileInfo object.
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
            file_info, fields=fields, mapping=self.mapping,
            mask_and_scale=mask_and_scale, **kwargs
        )

        scnlines = dataset["scnline"].values
        dataset = dataset.assign_coords(
             scnline=dataset["scnline"]
        )

        dataset["scnline"] = np.arange(1, dataset.scnline.size+1)
        dataset["scnpos"] = np.arange(1, 91)
        dataset["channel"] = "channel", np.arange(1, 6)

        # Create the time variable (is built from several other variables):
        dataset = self._get_time_field(dataset, user_fields)

        if mask_and_scale:
            self._mask_and_scale(dataset)

        # Make a fast check whether everything is alright
        self._test_coords(dataset)

        # Check the latitudes and longitudes:
        check_lat_lon(dataset)

        if user_mapping is not None:
            dataset = dataset.rename(user_mapping)

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
    def read(self, file_info, mask_and_scale=True, interpolate_packed_pixels=True,
             max_nans_interpolation=10, **kwargs):
        """Read and parse MHS AAPP HDF5 files and load them to xarray

        Args:
            file_info: Path and name of the file as string or FileInfo object.
                This can also be a tuple/list of file names or a path with
                asterisk.
            mask_and_scale: Where the data contains missing values, it will be
                masked with NaNs. Furthermore, data with scaling attributes
                will be scaled with them.
            interpolate_packed_pixels: Geo-location data is packed and must be
                interpolated to use them as reference for each pixel.
            max_nans_interpolation: How many NaN values are allowed in latitude
                and longitudes before raising an error?
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
            file_info, fields=fields, mapping=self.mapping,
            mask_and_scale=mask_and_scale, **kwargs
        )

        # Keep the original scnlines
        scnlines = dataset["scnline"].values

        dataset = dataset.assign_coords(
            scnline=dataset["scnline"]
        )

        dataset["scnline"] = np.arange(1, dataset.scnline.size+1)
        dataset["scnpos"] = np.arange(1, 2049)
        dataset["channel"] = "channel", np.arange(1, 6)

        # Currently, the AAPP converting tool seems to have a bug. Instead of
        # retrieving 409 pixels per scanline, one gets 2048 pixels. The
        # additional values are simply duplicates (or rather quintuplicates):
        dataset = dataset.sel(scnpos=slice(4, None, 5))
        dataset["scnpos"] = np.arange(1, 410)

        # Create the time variable (is built from several other variables):
        dataset = self._get_time_field(dataset, user_fields)

        if mask_and_scale:
            self._mask_and_scale(dataset)

        # All geolocation fields are packed in the AVHRR GAC files:
        if interpolate_packed_pixels:
            self._interpolate_packed_pixels(dataset, max_nans_interpolation)
            allowed_coords = {'channel', 'calib', 'scnline', 'scnpos'}
        else:
            allowed_coords = {'channel', 'calib', 'scnline', 'scnpos',
                              'packed_pixels'}

        # Make a fast check whether everything is alright
        self._test_coords(dataset, allowed_coords)

        # Check the latitudes and longitudes:
        check_lat_lon(dataset)

        if user_mapping is not None:
            dataset = dataset.rename(user_mapping)

        return dataset

    @staticmethod
    def _interpolate_packed_pixels(dataset, max_nans_interpolation):
        given_pos = np.arange(5, 409, 8)
        new_pos = np.arange(1, 410)

        lat_in = np.deg2rad(dataset["lat"].values)
        lon_in = np.deg2rad(dataset["lon"].values)

        # We cannot define given positions for each scanline, but we have to
        # set them for all equally. Hence, we skip every scan position of all
        # scan lines even if only one contains a NaN value:
        nan_scnpos = \
            np.isnan(lat_in).sum(axis=0) + np.isnan(lon_in).sum(axis=0)
        valid_pos = nan_scnpos == 0

        if valid_pos.sum() < 52 - max_nans_interpolation:
            raise ValueError(
                "Too many NaNs in latitude and longitude of this AVHRR file. "
                "Cannot guarantee a good interpolation!"
            )

        # Filter NaNs because CubicSpline cannot handle it:
        lat_in = lat_in[:, valid_pos]
        lon_in = lon_in[:, valid_pos]
        given_pos = given_pos[valid_pos]

        x_in = np.cos(lon_in) * np.cos(lat_in)
        y_in = np.sin(lon_in) * np.cos(lat_in)
        z_in = np.sin(lat_in)

        xf = CubicSpline(given_pos, x_in, axis=1, extrapolate=True)(new_pos)
        yf = CubicSpline(given_pos, y_in, axis=1, extrapolate=True)(new_pos)
        zf = CubicSpline(given_pos, z_in, axis=1, extrapolate=True)(new_pos)
        lon = np.rad2deg(np.arctan2(yf, xf))
        lat = np.rad2deg(np.arctan2(zf, np.sqrt(xf ** 2 + yf ** 2)))

        dataset["lat"] = ("scnline", "scnpos"), lat
        dataset["lon"] = ("scnline", "scnpos"), lon

        # The other packed variables will be simply padded:
        for var_name, var in dataset.data_vars.items():
            if "packed_pixels" not in var.dims:
                continue

            nan_scnpos = np.isnan(var).sum(axis=0)
            valid_pos = nan_scnpos == 0
            given_pos = np.arange(5, 409, 8)[valid_pos]

            dataset[var_name] = xr.DataArray(
                CubicSpline(
                    given_pos, var.values[:, valid_pos], axis=1,
                    extrapolate=True)(new_pos),
                dims=("scnline", "scnpos")
            )
