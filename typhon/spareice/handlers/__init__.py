# -*- coding: utf-8 -*-

"""This package provides file handler classes. The file handler classes provide specialized reading (sometimes as well
writing) methods for several data formats. All reading methods return a xarray/numpy array in a standardized format.
TODO: Find a standard format (e.g. arrays of time/lat/lon)."""

__all__ = [
    'FileHandler'
]

import abc

class FileHandler:
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        ...

    @abc.abstractmethod
    def get_info(self, filename):
        """ Returns a dictionary with info parameters about the file content. It has following keys:

            start_time, end_time

        Args:
            filename:

        Returns:
            A dictionary with info parameters.
        """
        pass

    @abc.abstractmethod
    def read(self, filename, fields=()):
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
            fields:

        Returns:
            numpy.array
        """
        pass

    @abc.abstractmethod
    def write(self, filename, data):
        """This method should create a file and write data to its specific format.

        This method is abstract therefore it has to be implemented in the specific file handler subclass.

        Args:
            filename:

        Returns:
            None
        """
        pass