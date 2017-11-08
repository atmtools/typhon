# -*- coding: utf-8 -*-

"""This package provides file handler classes. The file handler classes provide
specialized reading (sometimes as well writing) methods for several data
formats."""

from inspect import ismethod, signature

__all__ = [
    'FileHandler'
]


class FileHandler:
    """Base file handler class.

    File handler classes that can be used with the Dataset classes. You can
    either initialize specific *reader* ,*info loader* or *writer* functions or
    you can inherit from this class and override its methods. If you need a
    very specialised and well-documented file handler class, you should
    consider following the second approach.
    """

    def __init__(
            self, reader=None, info_reader=None, writer=None, **kwargs):
        """Initializes a basic filer handler object.

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


class FileInfo(dict):
    """Contains information about a file (time coverage, etc.)
    """
    def __init__(self):
        super(FileInfo, self).__init__()
