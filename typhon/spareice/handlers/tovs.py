import datetime
import time
import warnings

import h5py
import numpy as np
import typhon.geographical
import xarray as xr

from .. import handlers

__all__ = [
    'MHSAAPP',
    ]


class MHSAAPP(handlers.FileHandler):

    field_names = {
        "time", "lat", "lon", "brightness_temperature"
    }
    internal_mapping = {
        "brightness_temperature" : "Data/btemps",
        "lat" : "Geolocation/Latitude",
        "lon": "Geolocation/Longitude",
    }

    def __init__(self, **kwargs):
        # Call the base class initializer
        super().__init__(**kwargs)

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        with h5py.File(filename, "r") as file:

            info = {
                "times" : [
                     datetime.datetime(int(file.attrs["startdatayr"][0]), 1, 1) \
                        + datetime.timedelta(days=int(file.attrs["startdatady"][0])-1) \
                        + datetime.timedelta(milliseconds=int(file.attrs["startdatatime_ms"][0])),
                     datetime.datetime(int(file.attrs["enddatayr"]), 1, 1) \
                        + datetime.timedelta(days=int(file.attrs["enddatady"]) - 1) \
                        + datetime.timedelta(milliseconds=int(file.attrs["enddatatime_ms"])),
                ],
            }

            return info

    def read(self, filename, fields=None):
        """ Reads and parses NetCDF files and load them to a xarray.

        See the parent class for further documentation.
        """

        timer = time.time()

        if fields is None:
            fields = ("time", "lat", "lon")

        with h5py.File(filename, "r") as file:
            dataset = typhon.geographical.GeoData(name="MHS")

            for field, dimensions in self.parse_fields(fields):
                data = None

                # Mapping for additional fields
                if field == "time":
                    # Add standard field "time":
                    data = self._get_time_field(file)
                elif field == "brightness_temperature":
                    bt = np.asarray(file[self.internal_mapping[field]])
                    data = xr.DataArray(
                        bt.reshape(file[self.internal_mapping[field]].shape[0] * 90, 5)
                        * file[self.internal_mapping[field]].attrs["Scale"],
                        dims=["time_id", "channel"]
                    )
                elif field == "lat" or field == "lon":
                    data = xr.DataArray(
                        np.asarray(file[self.internal_mapping[field]]).flatten()
                        * file[self.internal_mapping[field]].attrs["Scale"],
                        dims=["time_id"]
                    )

                # Add the field data to the dataset.
                try:
                    dataset[field] = self.select(data, dimensions)
                except ValueError as e:
                    print(dataset, field, data)
                    raise e

            return dataset

    def _get_time_field(self, file):
        # Interpolate between the start times of each swath to retrieve the timestamp of each pixel.
        swath_start_indices = np.arange(0, file["Data"]["scnlintime"].size * 90,
                                        90)
        pixel_times = np.interp(
            np.arange((file["Data"]["scnlintime"].size - 1) * 90),
            swath_start_indices, file["Data"]["scnlintime"])

        # Add the timestamps from the last swath here, we could not interpolate them in the step before
        # because we do not have an ending timestamp. We simply extrapolate from the difference between the
        # last two timestamps.
        last_swath_pixels = \
            file["Data"]["scnlintime"][-1] \
            - file["Data"]["scnlintime"][-2] \
            + np.linspace(
                file["Data"]["scnlintime"][-2], file["Data"]["scnlintime"][-1],
                90, dtype="int32", endpoint=False)
        pixel_times = np.concatenate([pixel_times, last_swath_pixels])

        """"[
            np.linspace(
                file["Data"]["scnlintime"][i],
                file["Data"]["scnlintime"][i+1], 90, dtype="int32", endpoint=False
            )
            for i in range(file["Data"]["scnlintime"].size-1)]"""

        # The timestamps of the last swath are estimated from the difference between the last.
        """pixel_times.append(
            file["Data"]["scnlintime"][-1] - file["Data"]["scnlintime"][-2]
            + np.linspace(
                file["Data"]["scnlintime"][-2], file["Data"]["scnlintime"][-1], 90, dtype="int32", endpoint=False)
        )"""

        # Convert the pixel times into timedelta objects
        pixel_times = pixel_times.astype("timedelta64[ms]")

        # Convert the swath time variables (year, day of year, miliseconds since midnight) to numpy.datetime
        # objects. These are the start times of each swath, we have to bring them together with the
        # interpolated pixel times.
        swath_times = (np.asarray(file["Data/scnlinyr"]).astype(
            'datetime64[Y]') - 1970) \
                      + (np.asarray(file["Data/scnlindy"]).astype(
            'timedelta64[D]') - 1)

        swath_times = np.repeat(swath_times, 90)
        times = swath_times + pixel_times
        data = xr.DataArray(np.asarray(times).flatten(), dims=["time_id"])

        return data