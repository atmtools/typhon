"""
This module contains classes to handle datasets consisting of many files. They
are inspired by the implemented dataset classes in atmlab written by Gerrit
Holl.

Created by John Mrziglod, June 2017
"""

import atexit
from collections import defaultdict
from datetime import datetime, timedelta
import glob
import json
from multiprocessing import Pool
import os.path
import re
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import typhon.files
import typhon.plots
from typhon.spareice.array import ArrayGroup
from typhon.trees import IntervalTree

__all__ = [
    "Dataset",
    "DatasetManager",
]


class NoFilesError(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class NoHandlerError(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class InhomogeneousFilesError(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)


class Dataset:
    """Class which provides methods to handle a set of multiple files
    (dataset).

    """
    placeholder = {
        # "placeholder_name" : [regex to find the placeholder]
        "year": "(\d{4})",
        "year2": "(\d{2})",
        "month": "(\d{2})",
        "day": "(\d{2})",
        "doy": "(\d{3})",
        "hour": "(\d{2})",
        "minute": "(\d{2})",
        "second": "(\d{2})",
        "millisecond": "(\d{3})",
        "end_year": "(\d{4})",
        "end_year2": "(\d{2})",
        "end_month": "(\d{2})",
        "end_day": "(\d{2})",
        "end_doy": "(\d{3})",
        "end_hour": "(\d{2})",
        "end_minute": "(\d{2})",
        "end_second": "(\d{2})",
        "end_millisecond": "(\d{3})",
        "name": "",
    }

    def __init__(
            self, files, name=None, handler=None,
            time_coverage_retrieving_method="filename", times_cache=None
    ):
        """Initializes a dataset object.

        Args:
            files: A string with the complete path to the dataset files. The
                string can contain placeholder such as {year}, {month},
                etc. See here for a complete list (TODO: insert link to
                placeholder documentation). For example, the Dataset class
                finds with the argument files="/path/to/files/{year}/{
                month}/{day}/{hour}{minute}{second}.nc.gz" files such as
                "/path/to/files/2016/11/12/051422.nc.gz". If this path is
                relative, it will be appended on the current working directory
                during the initialization of this object.
            name: (optional) The name of the dataset.
            handler: (optional) An object which can handle the dataset files.
                This dataset class does not care which format its files have
                when this file handler object is given. You can use a file
                handler class from typhon.handlers or write your own class.
                For example, if this dataset consists of NetCDF files, you can
                use typhon.spareice.handlers.common.NetCDF4() here.
            time_coverage_retrieving_method: (optional) Defines how the
                timestamp of the data should be retrieved. Default is
                *filename*. Look at Dataset.retrieve_timestamp() for more
                details.
            times_cache: (optional) Finding the correct files for a time period
                may take a while, especially when the time retrieving
                method is set to "content". Therefore, if the file names and
                their time coverage are cached, multiple calls of find_files
                (for time coverages that are close) are significantly faster.
                Specify a name to a file here (which need not exist) if you
                wish to save those time coverages to a file. When restarting
                your script, this cache is used.

        Examples:

        .. code-block:: python

            dataset = Dataset(
                files="/path/to/daily/files/{year}/{month}/{day}/*.nc",
                handler=handlers.common.NetCDF4(),
                name="TestData",
                # If the time coverage of the data cannot be retrieved from the
                # filename, you should set this to "content":
                timestamp_retrieving_method="filename"
            )
        """

        # Initialize member variables:
        self._name = None
        self.name = name
        self.handler = handler

        self.files = files
        self.files_placeholders = re.findall("\{(\w+)\}", self.files)

        # Multiple calls of .find_files() can be very slow when using a time
        # coverage retrieving method "content". Hence, we use a cache to
        # store the names and time coverages of already touched files in this
        # dictionary.
        self.times_cache_filename = times_cache
        self.time_coverages_cache = defaultdict(list)
        if self.times_cache_filename is not None:
            try:
                # Load the time coverages from a file:
                self.load_time_coverages(self.times_cache_filename)
            except Exception as e:
                raise e
            else:
                # Save the time coverages cache into a file before exiting.
                # This will be executed as well when the python code is
                # aborted due to an exception. This is normally okay, but what
                # happens if the error occurs during the loading of the time
                # coverages? We would overwrite the cache with nonsense.
                # Therefore, we need this code in this else block.
                atexit.register(
                    Dataset.save_time_coverages,
                    self, self.times_cache_filename)

        self.time_coverage_retrieving_method = time_coverage_retrieving_method

    """def __iter__(self):
        return self

    def __next__(self):
        # We split the path of the input files after the first appearance of 
        # {day} or {doy}.
        path_parts = re.split(r'({\w+})', self.files)

        for dir in self._find_subdirs(path_parts[0])
            print(path_parts)

            yield file
    """
    def __contains__(self, item):
        start = self._to_datetime(item)
        end = start + timedelta(seconds=5)
        for _, _ in self.find_files(start, end):
            return True

        return False

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(self.read_period(item.start, item.stop))
        elif isinstance(item, (datetime, str)):
            filename = self.find_file(item)
            if filename is not None:
                return self.read(filename)
            return None

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start = key.start
            end = key.stop
        else:
            start = end = key

        filename = self.generate_filename(self.files, start, end)
        self.write(filename, value)

    def __repr__(self):
        return self.name

    def __str__(self):
        info = "Name:\t" + self.name
        info += "\nFiles:\t" + self.files
        return info

    def accumulate(self, start, end, concat_func=None, concat_args=None,
                   reading_args=None):
        """Accumulate all data between two dates in one object.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            concat_func: Function that concatenates the read data to
                another. The first accepted argument must be a list of objects
                to concatenate. Default is ArrayGroup.concatenate.
            concat_args: A dictionary with additional arguments for
                *concat_func*.
            reading_args: A dictionary with additional arguments for reading
                the data (specified by the used file handler).

        Returns:
            Concatenated object.

        Examples:
            .. :code-block:: python

            import xarray

            dataset = Dataset(
                files="path/to/files.nc", handler=handlers.common.NetCDF4()
            )
            data = dataset.accumulate(
                datetime(2016, 1, 1), datetime(2016, 2, 1),
                xarray.concat, read_args={fields : ("temperature", )})

            # do something with data["temperature"]
            data["temperature"].plot()
            ...
        """

        if concat_func is None:
            concat_func = ArrayGroup.concatenate

        if reading_args is None:
            contents = list(self.read_period(start, end))
        else:
            contents = list(self.read_period(start, end, **reading_args))

        if concat_args is None:
            return concat_func(contents)
        else:
            return concat_func(contents, **concat_args)

    @staticmethod
    def _call_function_with_file_info(args):
        """ This is a small wrapper function to call the function that is
        called on dataset files via .map().

        Args:
            args: A tuple containing following elements:
                (Dataset object, (filename, time coverage), function,
                function_arguments)

        Returns:
            The return value of *function* called with the arguments *args*.
        """
        self, file_info, function, function_arguments, return_file_info = args
        filename, time_coverage = file_info

        return_value = None
        if function_arguments is None:
            return_value = function(self, filename, time_coverage)
        else:
            return_value = function(
                self, filename, time_coverage, **function_arguments)

        if return_file_info:
            return filename, time_coverage, return_value
        else:
            return return_value

    @staticmethod
    def _call_method_of_file_content(args):
        """ This is a small wrapper function to call a method on an object
        returned by reading a dataset file via Dataset.read().

        Args:
            args: A tuple containing following elements:
                (Dataset object, (filename, timestamp), read_arguments, method,
                 method_arguments, overwrite)

        Returns:
            The return value of *function* called with the arguments
            *method_arguments*.
        """
        self, file_info, read_arguments, method, method_arguments, \
            return_file_info, overwrite = args
        filename, time_coverage = file_info

        if read_arguments is None:
            data = self.read(filename)
        else:
            data = self.read(filename, **read_arguments)

        return_value = None

        if method_arguments is None:
            return_value = method(data)
        else:
            return_value = method(data, **method_arguments)

        # Overwrite with the (potentially) changed data the old data in the
        # file:
        if overwrite:
            self.handler.write(filename, data)

        if return_file_info:
            return filename, time_coverage, return_value
        else:
            return return_value

    @staticmethod
    def _copy_file(
            dataset, filename, time_coverage,
            path, converter, delete_originals):
        """This is a small wrapper function for copying files. Do not use it
        directly but :meth:`Dataset.copy` instead.

        Args:
            dataset:
            filename:
            time_coverage:
            path:
            converter:
            delete_originals:

        Returns:
            None
        """
        # Generate the new file name
        new_filename = dataset.generate_filename(
            path, *time_coverage)

        # Create the new directory if necessary.
        os.makedirs(os.path.dirname(new_filename), exist_ok=True)

        # Shall we simply copy or even convert the files?
        if converter is None:
            if delete_originals:
                print("\tDelete:", filename)
                shutil.move(filename, new_filename)
            else:
                shutil.copy(filename, new_filename)
        else:
            # Read the file with the current file handler
            data = dataset.read(filename)

            # Store the data of the file with the new file handler
            converter.write(new_filename, data)

            if delete_originals:
                print("\tDelete:", filename)
                os.remove(filename)

    def copy(
            self, start, end, destination,
            converter=None, delete_originals=False, verbose=False,
            new_name=None
    ):
        """ Copies all files from this dataset between two dates to another
        location.

        When passing a file handler via the argument converter, it also
        converts all matched files to a new format defined by the passed
        file handler.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            destination: The new path of the files. Must contain place holders
                (such as {year}, {month}, etc.).
            converter: If you want to convert the files during copying to a
                different format, you can pass a file handler object with
                writing-to-file support here.
            delete_originals: If true, then all original files will be deleted.
                Be careful, this cannot be undone!
            verbose:
            new_name: The name of the new dataset. If it is not given,
                the new name is the the old name followed by "_copy".

        Returns:
            New Dataset object with the new files.

        Examples:

        .. code-block:: python

            # Copy all the files between the 15th and 23rd September 2016:
            date1 = datetime(2017, 9, 15)
            date2 = datetime(2017, 9, 23)
            old_dataset = Dataset(
                files="old/path/to/files/{year}/{month}/{day}/{hour}{minute}{second}.jpg",
                handler=FileHandlerJPG()
            )
            new_dataset = old_dataset.copy(
                date1, date2,
                "new/path/to/files/{year}/{month}/{day}/{hour}{minute}{second}.jpg",
            )

        .. code-block:: python

            # When you want to convert the files during copying:
            old_dataset = Dataset(
                files="old/path/to/files/{year}/{month}/{day}/{hour}{minute}{second}.jpg",
                handler=FileHandlerJPG()
            )
            # Note that this only works if the converter file handler
            # (FileHandlerPNG in this example) supports
            # writing to a file.
            new_dataset = old_dataset.copy(
                date1, date2,
                "new/path/to/files/{year}/{month}/{day}/{hour}{minute}{second}.png",
                converter=FileHandlerPNG(),
            )
        """

        # If the new path contains place holders, fill them for each file.
        # Otherwise it is a path which does not describe each file
        # individually. So far, we cannot handle this.
        # TODO: What happens if this dataset consists of one file only?
        if "{" in destination:
            # Copy the files
            self.map(start, end, Dataset._copy_file,
                     {
                         "path" : destination,
                         "converter" : converter,
                         "delete_originals" : delete_originals
                     },
                     verbose=verbose
            )
        else:
            raise ValueError(
                "The new_path argument must describe each file individually by"
                " using place holders!")

        # Copy this dataset object but change the files parameter.
        new_dataset = Dataset(
            destination,
            new_name if new_name is not None else self.name + "_copy",
            self.handler, self.time_coverage_retrieving_method
        )

        if converter is not None:
            # The files are in a different format now. Hence, we need the new
            # file handler:
            new_dataset.handler = converter

        return new_dataset

    @staticmethod
    def _create_date_from_placeholders(placeholders, values, prefix=None,
                                       exclude=None, default=None):
        """Creates a dictionary with date and time keys from placeholders and
        their values.

        Args:
            placeholders:
            values:
            prefix:
            exclude:
            default:

        Returns:
            A dictionary with "year", "month", etc.
        """
        date_args = {}

        if prefix is None:
            prefix = ""

        if default is not None:
            date_args.update(**default)

        for index, placeholder in enumerate(placeholders):
            value = int(values[index])
            if placeholder == prefix + "year2":
                # TODO: What should be the threshold that decides whether the
                # year is 19xx or 20xx?
                if value < 65:
                    date_args["year"] = 2000 + value
                else:
                    date_args["year"] = 1900 + value
            elif placeholder == prefix + "millisecond":
                date_args[prefix + "microsecond"] = value * 1000
            elif (exclude is None or not placeholder.startswith(exclude)) \
                    and placeholder.startswith(prefix):
                if prefix is None:
                    date_args[placeholder] = value
                else:
                    # Cut off the prefix
                    date_args[placeholder[len(prefix):]] = value

        if prefix + "doy" in placeholders:
            date = datetime(date_args["year"], 1, 1) \
                   + timedelta(date_args["doy"] - 1)
            date_args["month"] = date.month
            date_args["day"] = date.day
            del date_args["doy"]

        return date_args

    def find_file(self, timestamp, max_interval=None):
        """Finds the closest file to a timestamp.

        Args:
            timestamp: date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            max_interval: Maximal time interval in seconds that should be
                between *timestamp* and the found closest file. If this
                argument is None (default value), then simply the closest file
                name is returned regardless of their interval in time.

        Returns:
            The found file name as a string. If no file was found, None is
            returned.
        """

        # Maybe there is a file with exact this timestamp?
        path = self.generate_filename(
            self.files, timestamp, timestamp
        )

        timestamp = self._to_datetime(timestamp)

        if os.path.exists(path):
            return path

        if max_interval is None:
            files = []
            while not files and path:
                path = path.rsplit("/", 1)[0]
                files = list(self._get_all_files(path))
                times = [self.retrieve_time_coverage(file) for file in files]
        else:
            interval = timedelta(seconds=max_interval)
            files, times = zip(
                *self.find_files(
                    timestamp-interval, timestamp+interval
                )
            )

        if not files:
            return None

        intervals = np.abs(np.asarray(times) - timestamp)
        closest_index = np.argmin(intervals) // 2

        return files[closest_index]

    def find_files(
            self, start, end,
            sort=False, verbose=False,
            no_files_error=True,
    ):
        """ Finds all files between the given start and end date.

        This method calculates the days between the start and end date. It uses
        those days to loop over the dataset
        files. This is a generator method if the 'sort' argument is false.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            sort: If this is set to true, all files will be returned sorted by
                their starting time.
            verbose: If this is set to true, debug messages will be printed.
            no_files_error: Raises an NoFilesError when no files are found.

        Yields:
            It will yield a tuple of one file name and its time coverage (a
            tuple of two datetime objects).
        """

        path_parts = self.files.split("/")

        #
        day_index = 0
        for index, part in enumerate(path_parts):
            # TODO: What about file path identifier which does not contain the
            # TODO: day parameter?
            if "{day}" in part or "{doy}" in part:
                day_index = index
                break

        # If the user wants sorted files we have to collect them first:
        found_files = []
        found_counter = 0

        # Prepare the regex for the file path
        regex = self.files.format(**Dataset.placeholder)
        regex = re.compile(regex.replace("*", ".*?"))

        # The user can give string instead of datetime objects:
        start = self._to_datetime(start)
        end = self._to_datetime(end)

        # At first, get all days between start and end date (including the last
        # day so we catch overlapping files as well).
        for date in pd.date_range(start-timedelta(days=1), end):

            # Generate the daily path string for the files
            path_name = '/'.join(path_parts[:day_index+1])
            day_indices = []
            if "{day}" in path_name:
                day_indices.append(path_name.index("{day}")+5)
            if "{doy}" in path_name:
                day_indices.append(path_name.index("{doy}")+5)

            daily_path = self.generate_filename(
                path_name[:min(day_indices)], date)
            if os.path.isdir(daily_path):
                daily_path += "/"

            if verbose:
                print("Daily path:", daily_path)

            # Find all files in that daily path
            for filename in self._get_all_files(daily_path):
                if verbose:
                    print(filename)

                # Test whether the file matches the files parameter.
                if regex.match(filename):
                    if verbose:
                        print("\tMatched files path pattern")
                    file_start, file_end = \
                        self.retrieve_time_coverage(filename)

                    # Test whether the file is overlapping the interval between
                    # start and end date.
                    if IntervalTree.overlaps(
                            (file_start, file_end), (start, end)):
                        if verbose:
                            print("\tPassed time check")

                        if sort:
                            found_files.append(
                                [filename, [file_start, file_end]])
                        else:
                            yield filename, [file_start, file_end]

                        found_counter += 1
        if sort:
            yield from sorted(found_files, key=lambda x: x[1][0])

        if found_counter == 0 and no_files_error:
            raise NoFilesError(
                "Found no files for %s between %s and %s!\nAre you sure you "
                "gave the correct files parameter? Or is there maybe no data "
                "for this time period?" % (self.name, start, end))

    def find_overlapping_files(
            self, start, end, other_dataset, max_interval=10):
        """Finds all files from this dataset and from another dataset that
        overlap in time between two dates.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            other_dataset: A Dataset object which holds the other files.
            max_interval: Maximal time interval between two overlapping files.

        Yields:
            A tuple with the names of two files which correspond to each other.
        """

        primary_files, primary_times = list(
            zip(*self.find_files(start, end, sort=True)))
        secondary_files, secondary_times = list(
            zip(*other_dataset.find_files(start, end, sort=True)))

        # Convert the times (datetime objects) to seconds (integer)
        primary_times = [[int(dt[0].timestamp()), int(dt[1].timestamp())]
                         for dt in primary_times]
        secondary_times = np.asarray([[dt[0].timestamp(), dt[1].timestamp()]
                                      for dt in secondary_times]).astype('int')

        # Expand the intervals of the secondary dataset to close-in-time
        # intervals.
        secondary_times[:, 0] -= max_interval
        secondary_times[:, 1] += max_interval

        tree = IntervalTree(secondary_times)

        # Search for all overlapping intervals:
        results = tree.query(primary_times)

        for i, overlapping_files in enumerate(results):
            yield primary_files[i], \
                  [secondary_files[oi] for oi in overlapping_files]

    def generate_filename(self, template, start_time, end_time=None):
        """ Generates the file name for a specific date by using the template
        argument.

        Allowed placeholders in the template are:

        ==================  ============================================  ============
        place holder        Description                                   Example
        ==================  ============================================  ============
        year                Four digits indicating the year.              1999
        year2               Two digits indicating the year.               99 (=1999)
        month               Two digits indicating the month.              09
        day                 Two digits indicating the day.                08
        doy                 Three digits indicating the day of the year.  002
        hour                Two digits indicating the hour.               22
        minute              Two digits indicating the minute.             58
        second              Two digits indicating the second.             58
        millisecond         Three digits indicating the millsecond.       999
        ==================  ============================================  ============

        Args:
            template: A string with format placeholders such as {year} or {day}.
            start_time: A datetime object with the needed date and time.
            end_time: (optional) A datetime object. If this is not given,
                it will be set to ´´start_time´´.

        Returns:
            A string containing the full path and name of the file.

        Example:

        .. code-block:: python

            Dataset.generate_filename(
                "{year2}/{month}/{day}.dat",
                datetime(2016, 1, 1))
            # Returns 16/01/01.dat

        """

        start_time = Dataset._to_datetime(start_time)

        if end_time is None:
            end_time = start_time
        else:
            end_time = Dataset._to_datetime(end_time)

        # Fill all placeholders variables with values
        template = template.format(
            year=start_time.year, year2=str(start_time.year)[-2:],
            month="{:02d}".format(start_time.month),
            day="{:02d}".format(start_time.day),
            doy="{:03d}".format(
                (start_time - datetime(start_time.year, 1, 1)).days
                + 1),
            hour="{:02d}".format(start_time.hour),
            minute="{:02d}".format(start_time.minute),
            second="{:02d}".format(start_time.second),
            millisecond="{:03d}".format(int(start_time.microsecond / 1000)),
            end_year=end_time.year, end_year2=str(end_time.year)[-2:],
            end_month="{:02d}".format(end_time.month),
            end_day="{:02d}".format(end_time.day),
            end_doy="{:03d}".format(
                (end_time - datetime(end_time.year, 1, 1)).days
                + 1),
            end_hour="{:02d}".format(end_time.hour),
            end_minute="{:02d}".format(end_time.minute),
            end_second="{:02d}".format(end_time.second),
            end_millisecond="{:03d}".format(int(end_time.microsecond / 1000)),
            name=self.name,
        )

        return template

    def _get_all_files(self, directory):
        if os.path.isdir(directory):
            for sub_directory in glob.iglob(directory + "/*", recursive=True):
                yield from self._get_all_files(sub_directory)
        else:
            if os.path.isfile(directory):
                yield directory
            else:
                for sub_directory in glob.iglob(directory + "*",
                                               recursive=True):
                    yield from self._get_all_files(sub_directory)

    def get_info(self, filename):
        """Gets info about a file by using the dataset's file handler.

        Notes:
            You need to specify a file handler for this dataset before you
            can use this method.

        Args:
            filename: Path and name of the file.

        Returns:
            Dictionary with information about the file.
        """
        if self.handler is None:
            raise NoHandlerError(
                "Could not get info from the file '{}'! No file handler is "
                "specified!".format(filename))

        with typhon.files.decompress(filename) as file:
            return self.handler.get_info(file)

    def load_time_coverages(self, filename):
        """ Loads the time coverages cache from a file.

        Returns:
            None
        """
        if filename is not None and os.path.exists(filename):
            print("Load time coverages of {} dataset from {}.".format(
                self.name, filename))
            with open(filename) as file:
                time_coverages = json.load(file)
                # Parse string times into datetime objects:
                time_coverages = {
                    file : (datetime.strptime(
                        times[0], "%Y-%m-%dT%H:%M:%S.%f"),
                            datetime.strptime(
                                times[1], "%Y-%m-%dT%H:%M:%S.%f"))
                    for file, times in time_coverages.items()
                }
                self.time_coverages_cache.update(time_coverages)

    def map(
            self, start, end,
            func, func_arguments=None,
            max_processes=4, include_file_info=False, verbose=False):
        """Applies a function on all files of this dataset between two dates.

        This method can work on two different ways: if the argument read_before
        processing is true, each file will be read before the function will
        be applied. The given function  must then be a method of the object
        that the Dataset.read() method returns (depends on the used file
        handler class).

        This method can use multiple processes to boost the procedure
        significantly. Depending on which system you work, you should try
        different numbers for max_processes.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            func: A reference to a function. The function should accept
                at least three arguments: the dataset object, the filename and
                the time coverage of the file (tuple of two datetime objects).
            func_arguments: Additional keyword arguments for the function.
            max_processes: Max. number of parallel processes to use. When
                lacking performance, you should change this number.
            include_file_info: Since the order of the returning results is
                arbitrary, you can include the name of the processed file
                and its time coverage in the results.
            verbose: If this is true, debug information will be printed.

        Returns:
            A list with one item for each processed file. The order is
            arbitrary. If include_file_info is true, the item is a tuple
            with the name of the file, its time coverage (a tuple of two
            datetime objects) and the return value of the applied function.
            If include_file_info is false, the item is simply the return
            value of the applied function.

        Examples:

        """

        if verbose:
            print("Process all files from %s to %s.\nThis may take a while..."
                  % (start, end))

        # Measure the time for profiling.
        start_time = time.time()

        # Create a pool of processes and process all the files with them.
        pool = Pool(processes=max_processes)

        results = pool.map(
            Dataset._call_function_with_file_info,
            [(self, x, func, func_arguments, include_file_info)
             for x in self.find_files(start, end, verbose=verbose)]
        )

        if results:
            if verbose:
                print("It took %.2f seconds using %d parallel processes to "
                      "process %d files." % (
                    time.time() - start_time, max_processes, len(results)))
        elif verbose:
            print("Warning: No files found!")

        return results

    def map_content(
            self, start, end,
            func, func_arguments=None,
            reading_arguments=None, overwrite=False,
            max_processes=4, include_file_info=False, verbose=False):
        """Applies a method on the content of each file of this dataset between
        two dates.

        This method is similar to Dataset.map() but each file will be read
        before the given function will be applied. Moreover, the given
        function must be a method of the object that the Dataset.read() method
        returns (depends on the used file handler class).

        This method can use multiple processes to boost the procedure
        significantly. Depending on which system (e.g. mistral or local) you
        work, you should try different numbers for max_processes.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            func: A reference to a function. The function should expect
                as first argument the content object which is returned by
                the file handler's *read* method.
            func_arguments: Additional keyword arguments for the function.
            reading_arguments: Additional keyword arguments that will be passed
                to the reading function (see Dataset.read() for more
                information).
            overwrite: Set this to true if you want to overwrite the old file
                content with the new one (assuming the
                used method changed something of the file content). The
                current file handler needs to support writing
                data to file. Otherwise you should consider to convert your
                dataset via Dataset.copy_to() first.
            max_processes: Max. number of parallel processes to use. When
                lacking performance, you should change this number.
            include_file_info: Since the order of the returning
                results is arbitrary, you can include the name of the
                processed file and its time coverage in the results.
            verbose: If this is true, debug information will be printed.

        Returns:
            A list with one item for each processed file. The order is
            arbitrary. If *include_file_info_in_results* is true, the item is
            a tuple with the name of the file, its time coverage (a tuple of
            two datetime objects) and the return value of the applied
            method. If *include_file_info_in_results* is false, it is simply
            the return value of the applied function.

        Examples:

        """

        if verbose:
            print("Process all files from %s to %s.\nThis may take a while..."
                  % (start, end))

        # Measure the time for profiling.
        start_time = time.time()

        # Create a pool of processes and process all the files with them.
        pool = Pool(processes=max_processes)

        results = pool.map(
            Dataset._call_method_of_file_content,
            [(self, x, reading_arguments, func, func_arguments,
              include_file_info, overwrite)
             for x in self.find_files(start, end, verbose=verbose)]
        )

        if results:
            if verbose:
                print("It took %.2f seconds using %d parallel processes to "
                      "process %d files." % (
                        time.time() - start_time, max_processes, len(results)))
        elif verbose:
            print("Warning: No files found!")

        return results

    @property
    def name(self):
        """Gets or sets the dataset's name.

        Returns:
            A string with the dataset's name.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            value = str(id(self))

        self._name = value
        self.placeholder["name"] = value

    def read(self, filename, **reading_arguments):
        """Opens and reads a file.

        Notes:
            You need to specify a file handler for this dataset before you
            can use this method.

        Args:
            filename: Path and name of the file to read.
            **reading_arguments: Additional key word arguments for the
                *read* method of the used file handler class.

        Returns:
            The content of the read file.
        """
        if self.handler is None:
            raise NoHandlerError(
                "Could not get info from the file '{}'! No file handler is "
                "specified!".format(filename))

        with typhon.files.decompress(filename) as file:
            data = self.handler.read(file, **reading_arguments)
            return data

    def read_period(self, start, end, **reading_arguments):
        """Reads all files between two dates and returns their content sorted
        by their starting time.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            **reading_arguments: Additional key word arguments for the
                *read* method of the used file handler class.

        Yields:
            The content of the read file.
        """
        #
        for filename, _ in self.find_files(start, end, sort=True):
            yield self.read(filename, **reading_arguments)

    def retrieve_time_coverage(self, filename, retrieve_method=None):
        """ Retrieves the timestamp from a given file. This is not the real
        timestamp of the file itself but of its content.

        Args:
            filename: Name of the file.
            retrieve_method: Defines how the timestamp of a dataset file (not
                the real timestamp of the file but of the content) should be
                retrieved. There are two options:
                    "filename" (default): The timestamp is parsed from the
                        filename by using the placeholders. This is fast but
                        could lead to errors due to ambiguity. This only works
                        if placeholders have been used in the "files"
                        argument of the Dataset object.
                    "content": The file is opened and its content is checked to
                        retrieve the timestamp. This could be time-intensive
                        but is more reliable.

        Returns:
            A tuple of two datetime objects, indicating the time coverage.
        """

        if filename in self.time_coverages_cache:
            return self.time_coverages_cache[filename]

        if retrieve_method is None:
            retrieve_method = self.time_coverage_retrieving_method

        time_coverage = None

        if retrieve_method == "filename":
            regex = self.files.format(**Dataset.placeholder)
            regex = re.compile(regex.replace("*", ".*?"))
            try:
                values = regex.findall(filename)
                values = values[0]
            except:
                raise ValueError(
                    "The filename does not match the given template (via "
                    "files). I could not retrieve the timestamp.")

            start_date_args = self._create_date_from_placeholders(
                self.files_placeholders, values, exclude="end_")

            # Default: if no end date is given then the starting date is also
            # the end date.
            end_date_args = self._create_date_from_placeholders(
                self.files_placeholders, values, prefix="end_",
                default=start_date_args)

            start_date = datetime(**start_date_args)
            end_date = datetime(**end_date_args)

            # Sometimes the filename does not explicitly provide the complete
            # end date. Imagine there is only hour and minute given, then day
            # change would not be noticed. Therefore, make sure that the end
            # date is always bigger (later) than the start date.
            if end_date < start_date:
                end_date += timedelta(days=1)

            time_coverage = (start_date, end_date)
        else:
            time_coverage = self.get_info(filename)["times"]

        # Cache the time coverage of this file:
        self.time_coverages_cache[filename] = time_coverage

        return time_coverage

    def save_time_coverages(self, filename):
        """ Saves time coverages cache to a file.

        Returns:
            None
        """
        if filename is not None:
            print("Save time coverages of {} dataset to {}.".format(
                self.name, filename))
            with open(filename, 'w') as file:
                # We cannot save datetime objects with json directly. We have
                # to convert them to strings first:
                time_coverages = {
                    file : (
                        times[0].strftime("%Y-%m-%dT%H:%M:%S.%f"),
                        times[1].strftime("%Y-%m-%dT%H:%M:%S.%f"))
                    for file, times in self.time_coverages_cache.items()
                }
                json.dump(time_coverages, file)

    @staticmethod
    def _to_datetime(obj):
        if isinstance(obj, str):
            return pd.Timestamp(obj).to_pydatetime()
        elif isinstance(obj, datetime):
            return obj
        else:
            raise KeyError("Cannot convert object of type '%s' to datetime "
                           "object! Allowed are only datetime or string "
                           "objects!" % type(obj))

    def write(self, filename, data, **writing_arguments):
        """Writes content to a file by using the Dataset's file handler.

        Notes:
            You need to specify a file handler for this dataset before you
            can use this method.

        Args:
            filename: Path and name of the file where to put the data.
            data: An object that can be stored by the used file handler class.
            **writing_arguments: Additional key word arguments for the
            *write* method of the used file handler class.

        Returns:
            None
        """
        if self.handler is None:
            raise NoHandlerError(
                "Could not get info from the file '{}'! No file handler is "
                "specified!".format(filename))

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return self.handler.write(filename, data, **writing_arguments)


class DatasetManager(dict):
    def __init__(self, *args, **kwargs):
        """ This manager can hold multiple Dataset objects. You can use it as a
        native dictionary.

        More functionality will be added in future.

        Example:

        .. code-block:: python

            datasets = DatasetManager()

            datasets += Dataset(
                name="dumbo.thermocam",
                files="path/to/files",
                handler=cloud.handler.dumbo.ThermoCamFile())

            ## do something with it
            for name, dataset in datasets.items():
                dataset.find_files(...)

        """
        super(DatasetManager, self).__init__(*args, **kwargs)

    def __iadd__(self, dataset):
        if dataset.name in self:
            warnings.warn(
                "DatasetManager: Overwrite dataset with name '%s'!"
                % dataset.name, RuntimeWarning)

        self[dataset.name] = dataset
        return self
