# -*- coding: utf-8 -*-

"""This package provides file handler classes. The file handler classes provide specialized reading (sometimes as well
writing) methods for several data formats. All reading methods return a xarray.Dataset in a standardized format.
TODO: Find a standard format (e.g. arrays of time/lat/lon)."""

__all__ = [
    'FileHandler'
]

import abc

class FileHandler:
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_info(self, filename):
        """ Returns a dictionary with info parameters about the file content.

            It must contain the key "times" with a tuple of two datetime.datetime as value, indicating the start and end
            time of this file.

        Args:
            filename: Path and name of the file of which to retrieve the info about.

        Returns:
            A dictionary with info parameters.
        """
        pass


    @staticmethod
    def parse_fields(fields):
        """ Checks whether the element of fields are strings or tuples.

        So far, this function does not do much. But I want it to be here as a central, static method to make it easier
        if we want to change the behaviour of the field selection in the future.

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
                    "Unknown field element: {}. The elements in fields must be strings or tuples!".format(type(field)))


    @abc.abstractmethod
    def read(self, filename):
        """This method should open a file by its name, read its content and return a numpy.array (or xarray) with the
        following structure:

            data = [
                "times" : [...],
                "lats" : [...],
                "lons" : [...],
                "field[0]" : [...],
                "field[1]" : [...],
                ...
            ]

        This method is abstract therefore it has to be implemented in the specific file handler subclass.

        Args:
            filename:
            **kwargs:

        Returns:
            numpy.array
        """
        pass

    @staticmethod
    def select(data, dimensions):
        """ Returns only the selected dimensions of the data.

        So far, this function does not do much. But I want it to be here as a central, static method to make it easier
        if we want to change the behaviour of the field selection in the future.

        Args:
            data: A sliceable object such as xarray.DataArray or numpy.array.
            dimensions: A list of the dimensions to select.

        Returns:
            The original data object but only with the selected dimensions.
        """

        if dimensions is not None:
            data = data[:, dimensions]

        return data

    def write(self, filename, data):
        """This method should store data to a file.

        This method is not abstract and therefore it is optional whether a file handler subclass does support a
        writing-data-to-file feature.

        Args:
            filename:
            data:

        Returns:
            None
        """
        raise NotImplementedError("This file handler does not support writing data to a file. You should use a "
                                  "different file handler.")