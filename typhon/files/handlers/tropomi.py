from satpy import Scene
import xarray as xr
import pandas as pd

from .common import expects_file_info, FileHandler

__all__ = [
    'TROPOMI',
]

class TROPOMI(FileHandler):
    """File handler for TROPOMI data using Satpy reader
    """

    # This file handler always wants to return at least time, lat and lon
    # fields. These fields are required for this:
    standard_fields = {
        'latitude',
        'longitude',
        'time_utc',
    }

    # Map the standard fields to standard names:
    mapping = { 
        "latitude": "lat",
        "longitude": "lon",
        "time_utc": "time",
    }   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.satpy_reader = 'tropomi_l2'

    @expects_file_info()
    def read(self, filename, **kwargs):
        scene = Scene(reader=self.satpy_reader, filenames=[filename.path])

        # We need to import at least the standard fields
        user_fields = kwargs.pop("fields", {}) 
        fields = self.standard_fields | set(user_fields)

        # If the user has not passed any fields to us, we load all per default.
        if fields is None:
            fields = scene.available_dataset_ids()

        # Load all selected fields
        scene.load(fields, **kwargs)

        # convert into dataset
        dataset = scene.to_xarray_dataset()

        # convert string array to datetime array
        dataset['time_utc'] = dataset['time_utc'].astype("datetime64[ns]")

        # delete useless coords
        dataset = dataset.drop_vars(['time', 'crs'])
        # rename standard variables
        dataset = dataset.rename(self.mapping)

        # We catch the user mapping here, since we do not want to deal with
        # user-defined names in the further processing. Instead, we use our own
        # mapping
        user_mapping = kwargs.pop("mapping", None)
        if user_mapping is not None:
            dataset = dataset.rename(user_mapping)

        # clean attributes
        for var in dataset.data_vars:
            dataset[var].attrs = []

        return dataset

