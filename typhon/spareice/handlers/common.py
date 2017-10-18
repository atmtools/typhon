import datetime
import pickle

import numpy as np
import pandas as pd
import typhon.arts.xml
from typhon.geographical import GeoData
import xarray

from .. import datasets
from .. import handlers

__all__ = [
    'GeoFile'
    'HDF5'
    'NetCDF4',
    'Numpy',
    'Pickle',
    'XML'
    ]


class GeoFile(handlers.FileHandler):
    def __init__(self, file_format="netcdf", output="GeoData", **kwargs):
        """File handler to read and write geographical data (stored in a typhon.geographical.GeoData or
        typhon.geographical.CollocatedData object).

        Args:
            file_format:
            output: Defines what the .read() method should return:
                * "GeoData" - a GeoData object.
                * "CollocatedData" - a CollocatedData object.
            **kwargs:
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        self.file_format = file_format
        self.output = output

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        ds = xarray.open_dataset(filename)

        info = {
            "times": [
                datetime.datetime.strptime(ds.attrs["start_time"], "%Y-%m-%dT%H:%M:%S.%f"),
                datetime.datetime.strptime(ds.attrs["end_time"], "%Y-%m-%dT%H:%M:%S.%f")
            ],
        }

        ds.close()

        return info

    def read(self, filename, fields=None, mapping=None):
        """ Reads and parses NetCDF files and load them to a xarray.

        See the base class for further documentation.
        """

        ds = xarray.open_dataset(filename)
        if fields is not None:
            ds = ds[fields]

        if mapping is not None:
            ds.rename(mapping, inplace=True)

        if self.output == "CollocatedData":
            return CollocatedData.from_xarray(ds)
        elif self.output == "GeoData":
            return GeoData.from_xarray(ds)
        else:
            raise ValueError("Unknown output type: %s" % self.output)

    def write(self, filename, data):
        """ Writes a GeoData or CollocatedData object to a file.

        See the base class for further documentation.
        """

        if self.file_format == "netcdf":
            if isinstance(data, xarray.Dataset):
                data.to_netcdf(filename)
            else:
                data.to_xarray().to_netcdf(filename)
        else:
            raise ValueError("Unknown output format: '%s'" % self.file_format)


class HDF5(handlers.FileHandler):
    def __init__(self, **kwargs):
        # Call the base class initializer
        super().__init__(**kwargs)

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        ...

    def read(self, filename, fields=None):
        """ Reads and parses NetCDF files and load them to a xarray.

        See the parent class for further documentation.
        """
        #
        ds = xarray.open_dataset(filename)
        if fields is not None:
            ds = ds[fields]
        return ds

    def write(self, filename, data):
        """ Writes a xarray to a NetCDF file.

        See the base class for further documentation.
        """

        # Data must be a xarray object!
        data.to_netcdf(filename)


class NetCDF4(handlers.FileHandler):
    def __init__(self, **kwargs):
        # Call the base class initializer
        super().__init__(**kwargs)

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        ...

    def read(self, filename, fields=None):
        """ Reads and parses NetCDF files and load them to a xarray.

        See the base class for further documentation.
        """
        #
        ds = xarray.open_dataset(filename)
        if fields is not None:
            ds = ds[fields]
        return ds

    def write(self, filename, data):
        """ Writes a xarray to a NetCDF file.

        See the base class for further documentation.
        """

        # Data must be a xarray object!
        data.to_netcdf(filename)




class Numpy(handlers.FileHandler):
    def __init__(self, **kwargs):
        # Call the base class initializer
        super().__init__(**kwargs)

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        ...

    def read(self, filename, fields=None):
        """ Reads and parses files with numpy arrays and load them to a xarray.

        See the base class for further documentation.
        """
        numpy_data = np.load(filename)
        print(numpy_data.keys())
        data = xarray.Dataset.from_dict(numpy_data)

        return data

    def write(self, filename, data):
        """ Writes a xarray to a NetCDF file.

        See the base class for further documentation.
        """

        # Data must be a xarray object!
        data_dict = data.to_dict()
        np.save(filename, data_dict)


class Pickle(handlers.FileHandler):
    def __init__(self, **kwargs):
        # Call the base class initializer
        super().__init__(**kwargs)

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        ...

    def read(self, filename, fields=None):
        """ Reads and parses files with numpy arrays and load them to a xarray.

        See the base class for further documentation.
        """

        with open(filename, 'rb') as file:
            return pickle.load(file)

    def write(self, filename, data):
        """ Writes a xarray to a NetCDF file.

        See the base class for further documentation.
        """

        with open(filename, 'wb') as file:
            pickle.dump(data, file)


class XML(handlers.FileHandler):
    def __init__(self, **kwargs):
        # Call the base class initializer
        super().__init__(**kwargs)

    def get_info(self, filename):
        # Get info parameters from a file (time coverage, etc)
        ...

    def read(self, filename, fields=None):
        """ Reads and parses NetCDF files and load them to a xarray.

        See the parent class for further documentation.
        """
        #
        return typhon.arts.xml.load(filename)

    def write(self, filename, data):
        """ Writes a xarray to a NetCDF file.

        See the base class for further documentation.
        """

        # Data must be a xarray object!
        typhon.arts.xml.save(data, filename)