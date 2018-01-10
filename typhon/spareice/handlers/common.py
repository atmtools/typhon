import csv
from datetime import datetime
from inspect import signature, ismethod
import os
import pickle

import numpy as np
import typhon.arts.xml
from typhon.spareice.array import ArrayGroup
import xarray

__all__ = [
    'CSV',
    'FileHandler',
    'FileInfo',
    'Plot',
    'NetCDF4',
    # 'Numpy',
    # 'Pickle',
    # 'XML'
]


class FileHandler:
    """Base file handler class.

    This can be used alone or with the Dataset classes. You can
    either initialize specific *reader* ,*info_reader* or *writer* functions or
    you can inherit from this class and override its methods. If you need a
    very specialised and reusable file handler class, you should
    consider following the second approach.
    """

    # Flag whether this file handler supports reading from multiple files at
    # once.
    multifile_reader_support = False

    # Can handle zipped files (otherwise the Dataset class unzip them)
    handle_zipped_files = False

    def __init__(
            self, reader=None, info_reader=None, writer=None, **kwargs):
        """Initializes a filer handler object.

        Args:
            reader: (optional) Reference to a function that defines how to
                read a given file and returns an object with the read data. The
                function must accept a filename as first parameter.
            info_reader: (optional) You cannot use the :meth:`get_info`
                without giving a function here that returns a FileInfo object.
                The function must accept a filename as first parameter.
            writer: (optional) Reference to a function that defines how to
                write the data to a file. The function must accept the data
                object as first and a filename as second parameter.
        """

        self.reader = reader
        self.info_reader = info_reader
        self.writer = writer

    def get_info(self, filename, **kwargs):
        """Returns a :class:`FileInfo` object with parameters about the
        file content.

        It must contain the key "times" with a tuple of two datetime
        objects as value, indicating the start and end time of this file.

        Args:
            filename: Path and name of the file of which to retrieve the info
                about.
            **kwargs: Additional keyword arguments.

        Returns:
            A FileInfo object.
        """
        if self.info_reader is not None:
            # Some functions do not accept additional key word arguments (via
            # kwargs). And if they are methods, they accept an additional
            # "self" or "class" parameter.
            number_args = 1 + int(ismethod(self.info_reader))
            if len(signature(self.info_reader).parameters) > number_args:
                return self.info_reader(filename, **kwargs)
            else:
                return self.info_reader(filename)

        raise NotImplementedError(
            "This file handler does not support reading data from a file. You "
            "should use a different file handler.")

    @staticmethod
    def parse_fields(fields):
        """Checks whether the element of fields are strings or tuples.

        So far, this function does not do much. But I want it to be here as a
        central, static method to make it easier if we want to change the
        behaviour of the field selection in the future.

        Args:
            fields: An iterable object of strings or fields.

        Yields:
            A tuple of a field name and its selected dimensions.
        """
        for field in fields:
            if isinstance(field, str):
                yield field, None
            elif isinstance(field, tuple):
                yield field
            else:
                raise ValueError(
                    "Unknown field element: {}. The elements in fields must be"
                    "strings or tuples!".format(type(field)))

    def read(self, filename, **kwargs):
        """This method open a file by its name, read its content and return
        a object containing this content.

        Args:
            filename: Path and name of the file from which to read.
            **kwargs: Additional key word arguments.

        Returns:
            An object containing the file's content.
        """
        if self.reader is not None:
            # Some functions do not accept additional key word arguments (via
            # kwargs). And if they are methods, they accept an additional
            # "self" or "class" parameter.
            number_args = 1 + int(ismethod(self.reader))
            if len(signature(self.reader).parameters) > number_args:
                return self.reader(filename, **kwargs)
            else:
                return self.reader(filename)

        raise NotImplementedError(
            "This file handler does not support reading data from a file. You "
            "should use a different file handler.")

    @staticmethod
    def select(data, dimensions):
        """ Returns only the selected dimensions of the data.

        So far, this function does not do much. But I want it to be here as a
        central, static method to make it easier if we want to change the
        behaviour of the field selection in the future.

        Args:
            data: A sliceable object such as xarray.DataArray or numpy.array.
            dimensions: A list of the dimensions to select.

        Returns:
            The original data object but only with the selected dimensions.
        """

        if dimensions is not None:
            data = data[:, dimensions]

        return data

    def write(self, filename, data, **kwargs):
        """Stores a data object to a file.

        Args:
            filename: Path and name of the file to which to store the data.
                Existing files will be overwritten.
            data: Object with data (e.g. numpy array, etc.).

        Returns:
            None
        """
        if self.writer is not None:
            if len(signature(self.writer).parameters) > 2:
                self.writer(data, filename, **kwargs)
            else:
                self.writer(data, filename)

            return None

        raise NotImplementedError(
            "This file handler does not support writing data to a file. You "
            "should use a different file handler.")


class FileInfo(os.PathLike):
    """Contains information about a file (time coverage, etc.)
    """
    def __init__(self, path=None, times=None, attr=None):
        super(FileInfo, self).__init__()

        self.path = path

        if times is None:
            self.times = [None, None]
        else:
            self.times = times

        if attr is None:
            self.attr = {}
        else:
            self.attr = attr

    def __eq__(self, other):
        return self.path == other.path and self.times == other.times

    def __fspath__(self):
        return self.path

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.attr:
            attr_string = "\n  Attributes:\n"
            for k, v in self.attr.items():
                attr_string += "    %s: %s\n" % (k, v)

        return "{}\n  Start: {}\n  End: {}{}".format(
            self.path, *self.times,
            attr_string if self.attr else "",
        )

    @classmethod
    def from_json_dict(cls, json_dict):
        times = []
        for i in range(2):
            if json_dict["times"][i] is None:
                times.append([None])
            else:
                times.append(
                    datetime.strptime(
                        json_dict["times"][i], "%Y-%m-%dT%H:%M:%S.%f"),
                )

        return cls(json_dict["path"], times, json_dict["attr"])

    def update(self, other_info, ignore_none_time=True):
        """Update this object with another FileInfo object.

        Args:
            other_info: A FileInfo object.
            ignore_none_time: If the start time or end time of *other_info* is
                set to None, it does not overwrite the corresponding time of
                this object.

        Returns:
            None
        """
        self.attr.update(**other_info.attr)

        if other_info.times[0] is not None or not ignore_none_time:
            self.times[0] = other_info.times[0]
        if other_info.times[1] is not None or not ignore_none_time:
            self.times[1] = other_info.times[1]

    def to_json_dict(self):
        return {
            "path": self.path,
            "times": [
                self.times[0].strftime("%Y-%m-%dT%H:%M:%S.%f"),
                self.times[1].strftime("%Y-%m-%dT%H:%M:%S.%f")
            ],
            "attr": self.attr,
        }


class CSV(FileHandler):
    """File handler that can read / write data from / to a ASCII file with
    comma separated values (or by any other delimiter).
    """
    def __init__(
            self, delimiter=None, header=None,
            skip_column=None, skip_header=None,
            **kwargs):
        """Initializes a CSV file handler class.

        Args:
            info_reader: (optional) You cannot use the :meth:`get_info`
                without giving a function here that returns a FileInfo object.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        self.delimiter = delimiter
        self.header = header
        self.skip_column = skip_column
        self.skip_header = skip_header

    def read(self, filename, fields=None):
        """Reads a CSV file and returns an ArrayGroup object.

        Args:
            filename:
            fields:

        Returns:
            An ArrayGroup object.
        """
        with open(filename, "r") as file:
            column_names = None

            if self.skip_header is not None:
                # Skip the header by ourselves because we may need the names
                # of the columns.
                line_number = 0
                while line_number < self.skip_header:
                    line = next(file)
                    line_number += 1
                    if line_number == self.header:
                        column_names = line.rstrip('\n').split(self.delimiter)

            raw_data = np.genfromtxt(
                (line.encode('utf-8') for line in file),
                delimiter=self.delimiter,
                dtype=None,
            )

            column_data = list(zip(*raw_data))

            table = ArrayGroup()
            if column_names is None:
                column_names = [
                    "col_" + str(c)
                    for c in range(len(column_data))]

            for column, name in enumerate(column_names):
                #name = str(name, 'utf-8')
                table[name] = column_data[column]

            return table


class NetCDF4(FileHandler):
    """File handler that can read / write data from / to a netCDF4 or HDF5
    file.
    """

    def __init__(self, return_type=None, **kwargs):
        """Initializes a NetCDF4 file handler class.

        Args:
            return_type: (optional) Defines what object should be returned by
                :meth:`read`. Default is *ArrayGroup* but *xarray* is also
                possible.
            info_reader: (optional) You cannot use the :meth:`get_info`
                without giving a function here that returns a FileInfo object.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        if return_type is None:
            self.return_type = "ArrayGroup"
        else:
            self.return_type = return_type

        # ArrayGroup supports reading from multiple files.
        if self.return_type == "ArrayGroup":
            self.multifile_reader_support = True
            self.reader = ArrayGroup.from_netcdf
        elif self.return_type == "xarray":
            self.multifile_reader_support = False
            self.reader = xarray.open_dataset
        else:
            raise ValueError("Unknown return type '%s'!" % return_type)

    def get_info(self, filename, **kwargs):
        """

        Args:
            filename:

        Returns:

        """
        if self.info_reader is None:
            raise NotImplementedError(
                "The NetCDF4 file handler does not have a native get_info "
                "support. You have to define one via 'info_reader' during "
                "initialization.")
        else:
            # Get info parameters from a file (time coverage, etc)
            return super(NetCDF4, self).get_info(filename, **kwargs)

    def read(self, filename, fields=None, mapping=None):
        """Reads and parses NetCDF files and load them to an ArrayGroup.

        If you need another return value, change it via the parameter
        *return_type* of the :meth:`__init__` method.

        Args:
            filename: Path and name of the file to read. If *return_type* is
                *ArrayGroup*, this can also be a tuple/list of file names.
            fields: (optional) List of field names that should be read. The
                other fields will be ignored.
            mapping: (optional) A dictionary which is used for renaming the
                fields. The keys are the old and the values are the new names.

        Returns:
            An ArrayGroup object.
        """

        if "fields" in signature(self.reader).parameters:
            ds = self.reader(filename, fields)
        else:
            ds = self.reader(filename)
            if fields is not None:
                ds = ds[fields]

        if mapping is not None:
            ds.rename(mapping, inplace=True)

        return ds

    def write(self, filename, data, **kwargs):
        """ Writes a data object to a NetCDF file.

        The data object must have a *to_netcdf* method, e.g. as an ArrayGroup
        or xarray.Dataset object.
        """

        if len(signature(data.to_netcdf).parameters) == 2:
            data.to_netcdf(filename)
        else:
            data.to_netcdf(filename, **kwargs)


class Plot(FileHandler):
    """File handler that can save matplotlib.figure objects to a file.

    This is a specialised file handler object that can just store
    matplotlib.figure objects. It cannot read from a file nor get the time
    coverage from one. This is designed for having a simple plot dataset as
    output.
    """

    def __init__(self, fig_args=None, **kwargs):
        """Initializes a Plot file handler class.

        Args:
            fig_args: A dictionary of additional keyword arguments for the
                fig.savefig method.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        if fig_args is None:
            self.fig_args = {}
        else:
            self.fig_args = fig_args

    def write(self, filename, figure, fig_args=None):
        """ Saves a matplotlib.figure object to a file.

        Args:
            filename: Path and name of the file.
            figure: A matplotlib.figure object.
            fig_args: A dictionary of additional keyword arguments for the
                fig.savefig method. This updates the *fig_args* given during
                initialisation.
        """

        params = self.fig_args.copy()
        if fig_args is not None:
            params.update(**fig_args)

        return figure.savefig(filename, **params)


# class Numpy(handlers.FileHandler):
#     def __init__(self, **kwargs):
#         # Call the base class initializer
#         super().__init__(**kwargs)
#
#     def get_info(self, filename):
#         # Get info parameters from a file (time coverage, etc)
#         ...
#
#     def read(self, filename, fields=None):
#         """ Reads and parses files with numpy arrays and load them to a xarray.
#
#         See the base class for further documentation.
#         """
#         numpy_data = np.load(filename)
#         print(numpy_data.keys())
#         data = xarray.Dataset.from_dict(numpy_data)
#
#         return data
#
#     def write(self, filename, data):
#         """ Writes a xarray to a NetCDF file.
#
#         See the base class for further documentation.
#         """
#
#         # Data must be a xarray object!
#         data_dict = data.to_dict()
#         np.save(filename, data_dict)
#
#
# class Pickle(handlers.FileHandler):
#     def __init__(self, **kwargs):
#         # Call the base class initializer
#         super().__init__(**kwargs)
#
#     def get_info(self, filename):
#         # Get info parameters from a file (time coverage, etc)
#         ...
#
#     def read(self, filename, fields=None):
#         """ Reads and parses files with numpy arrays and load them to a xarray.
#
#         See the base class for further documentation.
#         """
#
#         with open(filename, 'rb') as file:
#             return pickle.load(file)
#
#     def write(self, filename, data):
#         """ Writes a xarray to a NetCDF file.
#
#         See the base class for further documentation.
#         """
#
#         with open(filename, 'wb') as file:
#             pickle.dump(data, file)
#
#
# class XML(handlers.FileHandler):
#     def __init__(self, **kwargs):
#         # Call the base class initializer
#         super().__init__(**kwargs)
#
#     def get_info(self, filename):
#         # Get info parameters from a file (time coverage, etc)
#         ...
#
#     def read(self, filename, fields=None):
#         """ Reads and parses NetCDF files and load them to a xarray.
#
#         See the parent class for further documentation.
#         """
#         #
#         return typhon.arts.xml.load(filename)
#
#     def write(self, filename, data):
#         """ Writes a xarray to a NetCDF file.
#
#         See the base class for further documentation.
#         """
#
#         # Data must be a xarray object!
#         typhon.arts.xml.save(data, filename)