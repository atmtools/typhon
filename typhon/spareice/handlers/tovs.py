import datetime
import warnings

import h5py
import numpy as np
import xarray as xr

from .. import handlers
from . import common

__all__ = [
    'MHSAAPPFile',
    ]

class MHSAAPPFile():

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

        if fields is None:
            raise NotImplementedError("I need field names to extract the correct variables from the file! Native reading"
                                      " is not yet implemented! Allowed field names are:" + str(MHSAAPPFile.field_names))

        with h5py.File(filename, "r") as file:
            dataset = xr.Dataset()

            for field in fields:
                # Mapping for additional fields
                if field == "brightness_temperature":
                    data = np.asarray(file[MHSAAPPFile.internal_mapping[field]])
                    dataset[field] = xr.DataArray(
                        data.reshape(file[MHSAAPPFile.internal_mapping[field]].shape[0] * 90, 5)
                        * file[MHSAAPPFile.internal_mapping[field]].attrs["Scale"],
                        dims=["time", "channel"]
                    )
                elif field == "time":
                    # Mapping for standard field time

                    # Interpolate between the start times of each swath to retrieve the timestamp of each pixel.
                    pixel_times = [
                        np.linspace(
                            file["Data"]["scnlintime"][i],
                            file["Data"]["scnlintime"][i+1], 90, dtype="int32", endpoint=False
                        )
                        for i in range(file["Data"]["scnlintime"].size-1)]

                    # The timestamps of the last swath are estimated from the difference between the last.
                    pixel_times.append(
                        file["Data"]["scnlintime"][-1] - file["Data"]["scnlintime"][-2]
                        + np.linspace(
                            file["Data"]["scnlintime"][-2], file["Data"]["scnlintime"][-1], 90, dtype="int32", endpoint=False)
                    )

                    # Convert the pixel times into timedelta objects
                    pixel_times = np.asarray(pixel_times, dtype="timedelta64[ms]")

                    # Convert the swath time variables (year, day of year, miliseconds since midnight) to numpy.datetime
                    # objects. These are the start times of each swath, we have to bring them together with the
                    # interpolated pixel times.
                    swath_times = (np.asarray(file["Data/scnlinyr"]).astype('datetime64[Y]') - 1970) \
                                  + (np.asarray(file["Data/scnlindy"]).astype('timedelta64[D]') - 1)

                    times = [swath_times[i] + swath_pixel_times for i, swath_pixel_times in enumerate(pixel_times)]
                    dataset['time'] = xr.DataArray(np.asarray(times).flatten(), dims=["time"])
                elif field == "lat" or field == "lon":
                    dataset[field] = xr.DataArray(
                        np.asarray(file[MHSAAPPFile.internal_mapping[field]]).flatten()
                        * file[MHSAAPPFile.internal_mapping[field]].attrs["Scale"],
                        dims=["time"]
                    )

            return dataset

    def write(self, filename, data):
        """ Writes a xarray to a NetCDF file.

        See the base class for further documentation.
        """

        # Data must be a xarray object!
        data.to_netcdf(filename)