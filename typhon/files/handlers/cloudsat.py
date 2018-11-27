from datetime import datetime
import warnings

import numpy as np
import xarray as xr

from .common import HDF4, expects_file_info

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


class CloudSat(HDF4):
    """File handler for CloudSat data in HDF4 files.
    """

    # This file handler always wants to return at least time, lat and lon
    # fields. These fields are required for this:
    standard_fields = {
        "UTC_start",
        "Profile_time",
        "Latitude",
        "Longitude"
    }

    # Map the standard fields to standard names:
    mapping = {
        "Latitude": "lat",
        "Longitude": "lon",
        "dim_0": "scnline",
    }

    def __init__(self, **kwargs):

        # Call the base class initializer
        super().__init__(**kwargs)

    @expects_file_info()
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

    @expects_file_info()
    def read(self, file_info, **kwargs):
        """Read and parse HDF4 files and load them to a xarray.Dataset

        A description about all variables in CloudSat dataset can be found in
        http://www.cloudsat.cira.colostate.edu/data-products/level-2c/2c-ice?term=53.

        Args:
            file_info: Path and name of the file as string or FileInfo object.
            **kwargs: Additional keyword arguments that are valid for
                :class:`typhon.files.handlers.common.HDF4`.

        Returns:
            A xarray.Dataset object.
        """

        # We need to import at least the standard fields
        user_fields = kwargs.pop("fields", {})
        fields = self.standard_fields | set(user_fields)

        # We catch the user mapping here, since we do not want to deal with
        # user-defined names in the further processing. Instead, we use our own
        # mapping
        user_mapping = kwargs.pop("mapping", None)

        # Load the dataset from the file:
        dataset = super().read(
            file_info, fields=fields, mapping=self.mapping, **kwargs
        )

        dataset["time"] = self._get_time_field(dataset, file_info)

        # Remove fields that we do not need any longer (expect the user asked
        # for them explicitly)
        dataset = dataset.drop(
            {"UTC_start", "Profile_time"} - set(user_fields),
        )

        if user_mapping is not None:
            dataset = dataset.rename(user_mapping)

        return dataset

    def _get_time_field(self, dataset, file_info):
        # This gives us the starting time of the first profile in seconds
        # since midnight in UTC:
        first_profile_time = round(dataset['UTC_start'].item(0))

        # This gives us the starting time of all other profiles in seconds
        # since the start of the first profile.
        profile_times = dataset['Profile_time']

        # Convert the seconds to milliseconds
        profile_times *= 1000
        profile_times = profile_times.astype("int")

        try:
            date = file_info.times[0].date()
        except AttributeError:
            # We have to load the info by ourselves:
            date = self.get_info(file_info).times[0].date()

        # Put all times together so we obtain one full timestamp
        # (date + time) for each data point. We are using the
        # starting date coming from parsing the filename.
        profile_times = \
            np.datetime64(date) \
            + np.timedelta64(first_profile_time, "s") \
            + profile_times.astype("timedelta64[ms]")

        return profile_times
