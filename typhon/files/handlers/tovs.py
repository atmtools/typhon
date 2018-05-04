from datetime import datetime, timedelta

from netCDF4 import Dataset

from .common import NetCDF4, expects_file_info

__all__ = ['MHSAAPP', ]


class MHSAAPP(NetCDF4):
    """File handler for MHS level 1C HDF files (converted with the AAPP tool)
    """
    # Map the standard fields to standard names (make also the names of all
    # dimensions more meaningful):
    mapping = {
        "Geolocation/Latitude": "lat",
        "Geolocation/Longitude": "lon",
        "Data/scnlin": "scnline",
        "Data/phony_dim_0": "scnline",
        "Data/phony_dim_1": "scnpos",
        "Data/phony_dim_2": "channel",
        "Geolocation/phony_dim_3": "scnline",
        "Geolocation/phony_dim_4": "scnpos",
    }

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
                + timedelta(
                    milliseconds=int(file.startdatatime_ms[0]))
            file_info.times[1] = \
                datetime(int(file.enddatayr), 1, 1) \
                + timedelta(days=int(file.enddatady) - 1) \
                + timedelta(milliseconds=int(file.enddatatime_ms))

            return file_info

    @expects_file_info()
    def read(self, paths, mask_and_scale=True, **kwargs):
        """Read and parse MHS HDF5 files and load them to a xarray.Dataset

        Args:
            paths: Path and name of the file as string or FileInfo object.
                This can also be a tuple/list of file names or a path with
                asterisk.
            mask_and_scale: Where the data contains missing values, it will be
                masked with NaNs. Furthermore, data with scaling attributes
                will be scaled with them.
            **kwargs: Additional keyword arguments that are valid for
                :class:`typhon.files.handlers.common.NetCDF4`.

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

        # Create the time variable (is built from several other variables):
        dataset["time"] = self._get_time_field(dataset)

        # Remove fields that we do not need any longer (expect the user asked
        # for them explicitly)
        dataset = dataset.drop(
            {"Data/scnlinyr", "Data/scnlindy", "Data/scnlintime"}
            - set(user_fields),
        )

        # xarray.open_dataset can mask and scale automatically, but it does not
        # know the attribute *Scale* (which is specific for MHS):
        if mask_and_scale:
            for var in dataset.variables:
                scaling = dataset[var].attrs.pop("Scale", None)
                if scaling is not None:
                    dataset[var] = dataset[var].astype(float) * scaling

        # Make a fast check whether everything is alright
        self._test_dataset(dataset)

        if user_mapping is not None:
            dataset.rename(user_mapping, inplace=True)

        return dataset

    @staticmethod
    def _get_time_field(dataset):
        time = \
            (dataset["Data/scnlinyr"]-1970).astype('datetime64[Y]') \
            + (dataset["Data/scnlindy"]-1).astype('timedelta64[D]') \
            + dataset["Data/scnlintime"].astype("timedelta64[ms]")

        return time

    def _test_dataset(self, dataset):
        # Only these dimensions should be in the dataset:
        wanted = ['channel', 'scnline', 'scnpos']
        reality = list(sorted(dataset.dims.keys()))

        if wanted != reality:
            raise ValueError(
                f"Unexpected dimension in MHS file! {set(reality)-set(wanted)}"
            )