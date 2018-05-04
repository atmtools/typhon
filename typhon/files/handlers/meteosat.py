from datetime import datetime, timedelta
import warnings

import h5py
import numpy as np
import xarray as xr

from .common import HDF5, expects_file_info

__all__ = ['SEVIRI', ]


class SEVIRI(HDF5):
    """File handler for SEVIRI level 1.5 HDF files
    """

    def __init__(self, **kwargs):
        """

        Args:
            **kwargs: Additional key word arguments for base class.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        self._grid = None

        # We are going to import those fields per default (not channel 12
        # because it is awkward):
        self.standard_fields = {
            f"U-MARF/MSG/Level1.5/DATA/Channel {ch:02d}/IMAGE_DATA"
            for ch in range(1, 12)
        }

        # Map the standard fields to standard names (make also the names of all
        # dimensions more meaningful):
        self.mapping = {
            f"U-MARF/MSG/Level1.5/DATA/Channel {ch:02d}/IMAGE_DATA":
                f"channel_{ch}"
            for ch in range(1, 12)
        }
        # self.mapping.update({
        #     f"U-MARF/MSG/Level1.5/DATA/Channel {ch:02d}/phony_dim00": f"line"
        #     for ch in range(1, 13)
        # })
        # self.mapping.update({
        #     f"U-MARF/MSG/Level1.5/DATA/Channel {ch:02d}/phony_dim01": f"column"
        #     for ch in range(1, 13)
        # })

    @expects_file_info()
    def get_info(self, file_info, **kwargs):
        with netCDF4.Dataset(file_info.path, "r") as file:
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
    def read(self, file_info, fields=None, **kwargs):
        """Read SEVIRI HDF5 files and load them to a xarray.Dataset

        Args:
            file_info: Path and name of the file as string or FileInfo object.
                This can also be a tuple/list of file names or a path with
                asterisk.
            fields: ...
            **kwargs: Additional keyword arguments that are valid for
                :class:`typhon.files.handlers.common.NetCDF4`.

        Returns:
            A xrarray.Dataset object.
        """

        # Here, the user fields overwrite the standard fields:
        if fields is None:
            fields = self.standard_fields

        # We catch the user mapping here, since we do not want to deal with
        # user-defined names in the further processing. Instead, we use our own
        # mapping at the moment, and apply the user mapping later.
        user_mapping = kwargs.pop("mapping", None)

        # Load the dataset from the file:
        with h5py.File(file_info.path, 'r') as file:
            dataset = xr.Dataset()

            for field in fields:
                dataset[field] = xr.DataArray(
                    file[field], dims=("line", "column"),
                    attrs=dict(file[field].attrs)
                )

            xr.decode_cf(dataset, **kwargs)
            dataset.load()

        dataset.rename(self.mapping, inplace=True)

        # Add the latitudes and longitudes of the grid points:
        dataset = xr.merge([dataset, self.grid])

        # Add the time variable (is derived from the filename normally):
        if file_info.times[0] is None:
            warnings.warn(
                "SEVIRI: The time field was not specified by the filename! Set"
                "it to 1970-01-01!"
            )
            dataset["time"] = ("time", np.array(["1970-01-01"], dtype="M8[s]"))
        else:
            dataset["time"] = ("time", [file_info.times[0]])

        # For collocating and other things, we always need the three dimensions
        # time, lat and lon to be "connected". Hence, add the time dimension to
        # each data variable as extra dimension.
        dataset = xr.concat([dataset], dim="time")

        if user_mapping is not None:
            dataset.rename(user_mapping, inplace=True)

        return dataset

    @property
    def grid(self):
        """Return the geo-grid of the SEVIRI images (lat and lon)

        Notes:
            When calling this method for the first time, the latitudes and
            longitudes are going to be calculated. This may take a while. The
            results is cached, so multiple calls are faster. The caching may
            need a lot of space (the grid is HUGE)!

        Returns:
            A xarray.Dataset with lat and lon fields
        """
        if self._grid is not None:
            return self._grid

        # The SEVIRI image has a size of 3712x3712 pixels
        height_width = np.arange(3712)

        # Get the intermediate coordinates
        OFF = 1856
        FAC = -781648343
        intermediate_coords = 2 ** 16 * (height_width + 1 - OFF) / FAC

        # And all their combinations
        x, y = np.meshgrid(intermediate_coords, intermediate_coords)

        # Calculate the geo locations (lat, lon) from them
        cos_x = np.cos(x)
        cos_y = np.cos(y)
        sin_y = np.sin(y)

        sa = (42164 * cos_x * cos_y) ** 2 \
            - (cos_y ** 2 + 1.006803 * sin_y ** 2) * 1737121856

        sd = np.sqrt(sa)
        sn = (42164 * cos_x * cos_y - sd) / (
                cos_y ** 2 + 1.006803 * sin_y ** 2)
        s1 = 42164 - sn * cos_x * cos_y
        s2 = sn * np.sin(x) * cos_y
        s3 = -1 * sn * sin_y
        sxy = np.sqrt(s1 ** 2 + s2 ** 2)

        # conversion to degree
        lon = np.arctan(s2 / s1) * 180 / np.pi
        lat = np.arctan(1.006803 * (s3 / sxy)) * 180 / np.pi

        # We cache the grid
        self._grid = xr.Dataset({
            "lat": (("line", "column"), lat),
            "lon": (("line", "column"), lon),
        })

        self._grid["lat"].attrs = {
            "long_name": "latitude",
            "units": "degrees [-90 - 180]",
        }
        self._grid["lon"].attrs = {
            "long_name": "longitude",
            "units": "degrees [-180 - 180]",
        }

        return self._grid
