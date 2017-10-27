import datetime
import pickle

import numpy as np
import pandas as pd
import typhon.arts.xml
from typhon.spareice.array import ArrayGroup
from typhon.spareice.geographical import GeoData
import xarray

from .. import datasets
from .. import handlers

__all__ = [
    'NetCDF4',
    'Numpy',
    'Pickle',
    'XML'
    ]


class NetCDF4(handlers.FileHandler):
    """File handler that can read / write data from / to a netCDF4 or HDF5
    file.
    """
    def __init__(self, file_loader=None, info_loader=None, **kwargs):
        """Initializes a NetCDF4 file handler class.

        Args:
            file_loader: (optional) Defines how to return the read data.
                Default is the *ArrayGroup.from_netcdf* class method but
                possible are others, e.g. *xarray.open_dataset* or
                *GeoData.from_netcdf*.
            info_loader: (optional) You cannot use the :meth:`get_info`
                without giving a function here that returns a FileInfo object.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        if file_loader is None:
            self.file_loader = ArrayGroup.from_netcdf
        else:
            self.file_loader = file_loader

        self.info_loader = info_loader

    def get_info(self, filename):
        """

        Args:
            filename:

        Returns:

        """
        if self.info_loader is None:
            raise NotImplementedError(
                "The NetCDF4 file handler does not have a native get_info "
                "support. You have to define one via 'info_loader' during "
                "initialization.")
        else:
            # Get info parameters from a file (time coverage, etc)
            return self.info_loader(filename)

    def read(self, filename, fields=None):
        """Reads and parses NetCDF files and load them to an ArrayGroup.

        If you need another return value, change it via the parameter
        *file_loader* of the :meth:`__init__` method.

        Args:
            filename: Path and name of the file to read.
            fields: (optional) List of field names that should be read. The
                other fields will be ignored.

        Returns:
            An ArrayGroup object.
        """
        ds = self.file_loader(filename)
        if fields is not None:
            ds = ds[fields]
        return ds

    def write(self, filename, data):
        """ Writes a data object to a NetCDF file.

        The data object must have a *to_netcdf* method, e.g. an ArrayGroup
        or xarray.Dataset object.
        """

        # Data can be a ArrayGroup or xarray.Dataset object!
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