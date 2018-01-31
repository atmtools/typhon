from datetime import datetime, timedelta
from time import time
import warnings

from netCDF4 import Dataset
import numpy as np
import pandas as pd
from typhon.spareice.array import Array
from typhon.spareice.geographical import GeoData
import xarray as xr

from .common import FileHandler, expects_file_info

pyhdf_is_installed = False
try:
    from pyhdf import HDF, VS, V
    from pyhdf.SD import SD, SDC
    pyhdf_is_installed = True
except ImportError:
    pass

__all__ = [
    'CloudSat',
]


class CloudSat(FileHandler):
    """File handler for CloudSat data in HDF4 files.
    """

    def __init__(self, **kwargs):
        if not pyhdf_is_installed:
            raise ImportError("Could not import pyhdf, which is necessary for "
                              "reading CloudSat HDF files!")

        # Call the base class initializer
        super().__init__(**kwargs)

    @expects_file_info
    def get_info(self, file_info, **kwargs):
        """Return a :class:`FileInfo` object with parameters about the
        file content.

        Args:
            file_info: Path and name of the file of which to retrieve the info
                about.
            **kwargs: Additional keyword arguments.

        Returns:
            A FileInfo object.
        """

        file = SD(file_info.path, SDC.READ)
        file_info.times[0] = \
            datetime.strptime(getattr(file, 'start_time'), "%Y%m%d%H%M%S")
        file_info.times[1] = \
            datetime.strptime(getattr(file, 'end_time'), "%Y%m%d%H%M%S")

        return file_info

    @expects_file_info
    def read(self, file_info, extra_fields=None, mapping=None):
        """Read and parse HDF4 files and load them to an ArrayGroup.

        Args:
            file_info: Path and name of the file as string or FileInfo object.
            extra_fields: Additional field names that you want to extract from
                this file as a list.
            mapping: A dictionary that maps old field names to new field names.
                If given, *extra_fields* must contain the old field names.

        Returns:
            An ArrayGroup object.
        """

        dataset = GeoData(name="CloudSat")

        # The files are in HDF4 format therefore we cannot use the netCDF4
        # module. This code is taken from
        # http://hdfeos.org/zoo/OTHER/2010128055614_21420_CS_2B-GEOPROF_GRANULE_P_R04_E03.hdf.py
        # and adapted by John Mrziglod. A description about all variables in
        # CloudSat dataset can be found in
        # http://www.cloudsat.cira.colostate.edu/data-products/level-2c/2c-ice?term=53.

        file = HDF.HDF(file_info.path)

        try:
            vs = file.vstart()

            # Extract the standard fields:
            dataset["time"] = self._get_time_field(vs, file_info)
            dataset["lat"] = self._get_field(vs, "Latitude")
            dataset["lon"] = self._get_field(vs, "Longitude")
            dataset["scnline"] = Array(
                np.arange(dataset["time"].size), dims=["time_id"]
            )
            dataset["scnpos"] = Array(
                [1 for _ in range(dataset["time"].size)], dims=["time_id"]
            )

            # Get the extra fields:
            if extra_fields is not None:
                for field, dimensions in self.parse_fields(extra_fields):
                    data = self._get_field(vs, field)

                    # Add the field data to the dataset.
                    dataset[field] = self.select(data, dimensions)
        except Exception as e:
            raise e
        finally:
            file.close()

        return dataset

    def _get_field(self, vs, field):
        field_id = vs.find(field)

        if field_id == 0:
            # Field was not found.
            warnings.warn(
                "Field '{0}' was not found!".format(field),
                RuntimeWarning)

        field_id = vs.attach(field_id)
        nrecs, _, _, _, _ = field_id.inquire()
        raw_data = field_id.read(nRec=nrecs)
        data = Array(raw_data, dims=["time_id"]).ravel()
        field_id.detach()
        return data

    def _get_time_field(self, vs, file_info):
        # This gives us the starting time of the first profile in
        # seconds since midnight in UTC:
        first_profile_time = self._get_field(vs, 'UTC_start')[0][0]

        # This gives us the starting time of all other profiles in
        # seconds since the start of the first profile.
        profile_times = self._get_field(vs, 'Profile_time').ravel()

        # Convert the seconds to milliseconds
        profile_times *= 1000
        profile_times = profile_times.astype("int")

        try:
            date = file_info.times[0].date()
        except AttributeError:
            warnings.warn("No starting date in file_info set. Use "
                          "1970-01-01 as starting point.")
            date = datetime(1970, 1, 1)

        # Put all times together so we obtain one full timestamp
        # (date + time) for each data point. We are using the
        # starting date coming from parsing the filename.
        profile_times = \
            np.datetime64(date) \
            + np.timedelta64(round(first_profile_time), "s") \
            + profile_times.astype("timedelta64[ms]")

        return Array(profile_times, dims=["time_id"])