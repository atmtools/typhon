from copy import copy
from datetime import datetime
from functools import wraps
from inspect import signature, ismethod
import os
import pickle

import pandas as pd
from typhon.spareice.array import ArrayGroup
import xarray as xr

__all__ = [
    'CSV',
    'FileHandler',
    'FileInfo',
    'Plotter',
    'NetCDF4',
    'expects_file_info',
    # 'Numpy',
    # 'Pickle',
    # 'XML'
]


def parametrized(dec):
    """A decorator for decorators that need parameters

    Do not think about this too long, it may cause headaches. Have a look at
    this instead: https://stackoverflow.com/a/26151604

    Args:
        dec: A decorator function

    Returns:
        The decoratored decorator function.
    """
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


@parametrized
def expects_file_info(meth, pos=None, key=None):
    """Convert a method argument to a :class:`FileInfo` object

    This is a decorator function that can take parameters.

    If the argument is already a FileInfo object, nothing happens.

    Args:
        meth: Method object that should be decorated.
        pos: The index of the file info in the positional argument list.
            Default is 1 (assumes to decorate a method).
        key: The key of the file info in the key word argument dict.

    Returns:
        The return value of the decorated method.

    Examples:

        .. code-block:: python

        @expects_file_info()(0)
        def read(file, *args, *kwargs):
            # file is a Fileinfo object now with the attribute path containing
            # "path/to/file.txt"

        read("path/to/file.txt")

    """

    if pos is None and key is None:
        pos = 1

    @wraps(meth)
    def wrapper(*args, **kwargs):
        args = list(args)
        if args and pos is not None:
            if not isinstance(args[pos], FileInfo):
                args[pos] = FileInfo(args[pos])
        else:
            if not isinstance(kwargs[key], FileInfo):
                kwargs[key] = FileInfo(kwargs[key])

        return meth(*args, **kwargs)
    return wrapper


class FileHandler:
    """Base file handler class.

    This can be used alone or with the Dataset classes. You can
    either initialize specific *reader* ,*info* or *writer* functions or
    you can inherit from this class and override its methods. If you need a
    very specialised and reusable file handler class, you should
    consider following the second approach.
    """

    # Flag whether this file handler supports reading from multiple files at
    # once.
    multifile_reader_support = False

    # If the file handler can handle compressed files, you can set this to a
    # list of formats otherwise the Dataset class decompress them first. The
    # formats are defined in typhon.files. For example, if zipped files can be
    # handled, set this to:
    # handle_compression_formats = ["zip", ]
    handle_compression_formats = []

    def __init__(
            self, reader=None, info=None, writer=None, **kwargs):
        """Initialize a filer handler object.

        Args:
            reader: Reference to a function that defines how to read a given
                file and returns an object with the read data. The function
                must accept a :class:`FileInfo` object as first parameter.
            info: Reference to a function that returns a :class:`FileInfo`
                object with information about the given file. You cannot use
                the :meth:`get_info` without setting this parameter. The
                function must accept a filename as string as first parameter.
            writer: Reference to a function that defines how to write the data
                to a file. The function must accept the data object as first
                and a :class:`FileInfo` object as second parameter.
        """

        self.reader = reader
        self.info = info
        self.writer = writer

    @expects_file_info()
    def get_info(self, filename, **kwargs):
        """Return a :class:`FileInfo` object with parameters about the
        file content.

        Notes:
            This is the base class method that does nothing per default.

        Args:
            filename: A string containing path and name or a :class:`FileInfo`
                object of the file of which to get the information about.
            **kwargs: Additional keyword arguments.

        Returns:
            A :class:`FileInfo` object.
        """
        if self.info is not None:
            # Some functions do not accept additional key word arguments (via
            # kwargs). And if they are methods, they accept an additional
            # "self" or "class" parameter.
            number_args = 1 + int(ismethod(self.info))
            if len(signature(self.info).parameters) > number_args:
                return self.info(filename, **kwargs)
            else:
                return self.info(filename)

        raise NotImplementedError(
            "This file handler does not support reading data from a file. You "
            "should use a different file handler.")

    @expects_file_info()
    def read(self, filename, **kwargs):
        """Open a file by its name, read its content and return it

        Notes:
            This is the base class method that does nothing per default.

        Args:
            filename: A string containing path and name or a :class:`FileInfo`
                object of the file from which to read.
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

    @expects_file_info(pos=2)
    def write(self, data, filename, **kwargs):
        """Store a data object to a file.

        Notes:
            This is the base class method that does nothing per default.

        Args:
            filename: A string containing path and name or a :class:`FileInfo`
                object to which to store the data. Existing files will be
                overwritten.
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
    """Container of information about a file (time coverage, etc.)

    This is a simple object that holds the path and name, time coverage and
    further attributes of a file. It fulfills the os.PathLike protocol, i.e.
    you can use it as filename argument for the most python functions.

    See this Example:

    .. code-block:: python

        # Initialise a FileInfo object that points to a file
        file_info = FileInfo(
            path="path/to/a/file.txt",
            # The time coverage of the file (needed by Dataset classes)
            times=[datetime(2018, 1, 1), datetime(2018, 1, 10)],
            # Additional attributes:
            attr={},
        )

        with open(file_info) as file:
            ...

        # If you need to access the path or other attributes directly, you can
        # do it like this:
        file_info.path
        file_info.times
        file_info.attr
    """
    def __init__(self, path=None, times=None, attr=None):
        """Initialise a FileInfo object.

        Args:
            path: Absolute path to a file.
            times: A list or tuple of two datetime objects indicating start and
                end time of the file.
            attr: A dictionary with further attributes.
        """
        super(FileInfo, self).__init__()

        self._path = None
        self.path = path

        self._times = None
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
        return self.path

    def __str__(self):
        if self.attr:
            attr_string = "\n  Attributes:"
            for k, v in self.attr.items():
                attr_string += "\n    %s: %s" % (k, v)

        return "{}\n  Start: {}\n  End: {}{}".format(
            self.path, *self.times,
            attr_string if self.attr else "",
        )
        # return self.path

    def copy(self):
        return copy(self)

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

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        if isinstance(value, FileInfo):
            raise ValueError("You cannot set path to a FileInfo object.")
        self._path = value

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, value):
        if value is None:
            self._times = [None, None]
        else:
            self._times = list(value)
            if len(self._times) != 2:
                raise ValueError("FileInfo.times can only be a list of two "
                                 "timestamps!")

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
            self, info=None, return_type=None,
            read_csv=None, write_csv=None):
        """Initializes a CSV file handler class.

        Args:
            info: A function that return a :class:`FileInfo object of a
                given file.
            return_type: Defines what object should be returned by
                :meth:`read`. Default is *ArrayGroup* but *xarray* is also
                possible.
            **read_csv: Additional keyword arguments for the pandas function
                `pandas.read_csv`. See for more details:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
            **write_csv: Additional keyword arguments for
                `pandas.Dataframe.to_csv`. See for more details:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
        """
        # Call the base class initializer
        super().__init__(info=info)

        if return_type is None:
            self.return_type = "ArrayGroup"
        else:
            self.return_type = return_type

        self.read_csv = {} if read_csv is None else read_csv

        self.write_csv = {}
        if write_csv is not None:
            self.write_csv.update(write_csv)

    @expects_file_info()
    def read(self, filename, fields=None, **read_csv):
        """Read a CSV file and return an ArrayGroup object with its content.

        Args:
            filename: Path and name of the file as string or FileInfo object.
            fields: Field that you want to extract from the file. If not given,
                all fields are going to be extracted.
            **read_csv: Additional keyword arguments for the pandas function
                `pandas.read_csv`. See for more details:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

        Returns:
            An ArrayGroup object.
        """

        kwargs = self.read_csv.copy()
        kwargs.update(read_csv)

        if self.return_type == "ArrayGroup":
            return ArrayGroup.from_csv(filename.path, fields, **kwargs)
        else:
            dataframe = pd.read_csv(filename.path, **kwargs)
            return xr.Dataset.from_dataframe(dataframe)

    @expects_file_info(pos=2)
    def write(self, data, filename, **write_csv):
        """Write an ArrayGroup object to a CSV file.

        Args:
            data: An ArrayGroup object that should be saved.
            filename: Path and name of the file as string or FileInfo object.
            **write_csv: Additional keyword arguments for
                `pandas.Dataframe.to_csv`. See for more details:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html

        Returns:
            An ArrayGroup object.
        """

        kwargs = self.write_csv.copy()
        kwargs.update(write_csv)

        if isinstance(data, xr.Dataset):
            data = data.to_dataframe()

        return data.to_csv(filename.path, **kwargs)


class NetCDF4(FileHandler):
    """File handler that can read / write data from / to a netCDF4 or HDF5
    file.
    """

    def __init__(self, return_type=None, **kwargs):
        """Initializes a NetCDF4 file handler class.

        Args:
            return_type: Defines what object should be returned by
                :meth:`read`. Default is *ArrayGroup* but *xarray* is also
                possible.
            info: You cannot use the :meth:`get_info` without giving a
                function here that returns a FileInfo object.
        """
        # Call the base class initializer
        super().__init__(**kwargs)

        if return_type is None:
            self.return_type = "ArrayGroup"
        else:
            self.return_type = return_type

    @expects_file_info()
    def read(self, filename, fields=None, mapping=None, main_group=None,
             **kwargs):
        """Reads and parses NetCDF files and load them to an ArrayGroup.

        If you need another return value, change it via the parameter
        *return_type* of the :meth:`__init__` method.

        Args:
            filename: Path and name of the file as string or FileInfo object.
                If *return_type* is *ArrayGroup*, this can also be a tuple/list
                of file names.
            fields: List of field names that should be read. The other fields
                will be ignored.
            mapping: A dictionary which is used for renaming the fields. The
                keys are the old and the values are the new names.
            main_group: If the file contains multiple groups, the main group
                will be linked to this one (only valid for ArrayGroup).

        Returns:
            An ArrayGroup object.
        """

        # ArrayGroup supports reading from multiple files.
        if self.return_type == "ArrayGroup":
            ds = ArrayGroup.from_netcdf(filename.path, fields, **kwargs)
            if not ds:
                return None
            if main_group is not None:
                ds.set_main_group(main_group)
        elif self.return_type == "xarray":
            ds = xr.open_dataset(filename.path, **kwargs)
            if not ds.variables:
                return None
        else:
            raise ValueError("Unknown return type '%s'!" % self.return_type)

        if fields is not None:
            ds = ds[fields]

        if mapping is not None:
            ds.rename(mapping, inplace=True)

        return ds

    @expects_file_info(pos=2)
    def write(self, data, filename, **kwargs):
        """ Writes a data object to a NetCDF file.

        The data object must have a *to_netcdf* method, e.g. as an ArrayGroup
        or xarray.Dataset object.
        """

        if len(signature(data.to_netcdf).parameters) == 2:
            data.to_netcdf(filename.path)
        else:
            data.to_netcdf(filename.path, **kwargs)


class Plotter(FileHandler):
    """File handler that can save matplotlib.figure objects to a file.

    This is a specialised file handler object that can just store
    matplotlib.figure objects. It cannot read from a file nor get the time
    coverage from one. This is designed for having a simple plot dataset as
    output.
    """

    def __init__(self, fig_args=None, **kwargs):
        """Initializes a Plotter file handler class.

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

    @expects_file_info(pos=1)
    def write(self, figure, filename, fig_args=None):
        """ Saves a matplotlib.figure object to a file.

        Args:
            figure: A matplotlib.figure object.
            filename: Path and name of the file as string or FileInfo object.
            fig_args: A dictionary of additional keyword arguments for the
                fig.savefig method. This updates the *fig_args* given during
                initialisation.
        """

        params = self.fig_args.copy()
        if fig_args is not None:
            params.update(**fig_args)

        return figure.savefig(filename.path, **params)


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