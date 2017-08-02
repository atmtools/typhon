import numpy as np
import pickle
import typhon.arts.xml
import xarray

from .. import handlers

__all__ = [
    'HDF5File'
    'NetCDF4File',
    'NumpyFile',
    'PickleFile',
    'XMLFile'
    ]


class HDF5File(handlers.FileHandler):
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


class NetCDF4File(handlers.FileHandler):
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




class NumpyFile(handlers.FileHandler):
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


class PickleFile(handlers.FileHandler):
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


class XMLFile(handlers.FileHandler):
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