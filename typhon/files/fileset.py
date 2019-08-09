"""
This module contains classes to handle datasets consisting of many files.

Created by John Mrziglod, June 2017
"""

import atexit
from collections import Counter, defaultdict, deque, OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
import gc
import glob
from itertools import tee
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os.path
import re
import shutil
from sys import platform
import threading
import traceback
import warnings

import numpy as np
import pandas as pd
import typhon.files
from typhon.trees import IntervalTree
from typhon.utils import unique
from typhon.utils.timeutils import set_time_resolution, to_datetime, to_timedelta

from .handlers import expects_file_info, FileInfo
from .handlers import CSV, NetCDF4

__all__ = [
    "FileSet",
    "FileSetManager",
    "InhomogeneousFilesError",
    "NoFilesError",
    "NoHandlerError",
    "UnknownPlaceholderError",
    "PlaceholderRegexError",
]


logger = logging.getLogger(__name__)


class InhomogeneousFilesError(Exception):
    """Should be raised if the files of a fileset do not have the same internal
    structure but it is required.
    """
    def __init__(self, *args):
        Exception.__init__(self, *args)


class NoFilesError(Exception):
    """Should be raised if no files were found by the :meth:`find`
    method.

    """
    def __init__(self, fileset, start, end, *args):
        if start == datetime.min and end >= datetime.max-timedelta(seconds=1):
            message = f"Found no files for {fileset.name}!"
        else:
            message = f"Found no files for {fileset.name} between {start} " \
                      f"and {end}!"

        message += f"\nPath: {fileset.path}\nCheck the path for misspellings" \
                   f" and whether there are files in this time period."
        Exception.__init__(self, message, *args)


class NoHandlerError(Exception):
    """Should be raised if no file handler is specified in a fileset object but
    a handler is required.
    """
    def __init__(self, msg, *args):
        message = f"{msg} I do not know which file handler to " \
                  f"use. Set one by yourself."
        Exception.__init__(self, message, *args)


class UnfilledPlaceholderError(Exception):
    """Should be raised if a placeholder was found that cannot be filled.
    """
    def __init__(self, name, placeholder_name=None, *args):
        if placeholder_name is None:
            message = \
                "The path of '%s' contains a unfilled placeholder!" % (name,)
        else:
            message = \
                "The fileset '%s' could not fill the placeholder %s!" % (
                    name, placeholder_name)
        Exception.__init__(self, message, *args)


class UnknownPlaceholderError(Exception):
    """Should be raised if a placeholder was found that was not defined before.
    """
    def __init__(self, name, placeholder_name=None, *args):
        if placeholder_name is None:
            message = \
                "The path of '%s' contains a unknown placeholder!" % (name,)
        else:
            message = \
                "The fileset '%s' does not know the placeholder %s!" % (
                    name, placeholder_name)
        Exception.__init__(self, message, *args)


class PlaceholderRegexError(Exception):
    """Should be raised if the regex of a placeholder is broken.
    """
    def __init__(self, name, msg):
        Exception.__init__(
            self, f"The path of '{name}' contains syntax errors: {msg}"
        )


class AlignError(Exception):
    """Should be raised if two filesets could not be aligned to each other.
    """
    def __init__(self, msg):
        Exception.__init__(self, msg)


class FileSet:
    """Provide methods to handle a set of multiple files

    For more examples and an user guide, look at this tutorial_.

    .. _tutorial: http://radiativetransfer.org/misc/typhon/doc-trunk/tutorials/fileset.html

    Examples:

        FileSet with multiple files:

        .. code-block:: python

            from typhon.files import FileSet

            # Define a fileset consisting of multiple files:
            files = FileSet(
                path="/dir/{year}/{month}/{day}/{hour}{minute}{second}.nc",
                name="TestData",
                # If the time coverage of the data cannot be retrieved from the
                # filename, you should set this to "handler" and giving a file
                # handler to this object:
                info_via="filename"
            )

            # Find some files of the fileset:
            for file in files.find("2017-01-01", "2017-01-02"):
                # Should print the path of the file and its time coverage:
                print(file)

        FileSet with a single file:

        .. code-block:: python

            # Define a fileset consisting of a single file:
            file = FileSet(
                # Simply use the path without placeholders:
                path="/path/to/file.nc",
                name="TestData2",
                # The time coverage of the data cannot be retrieved from the
                # filename (because there are no placeholders). You can use the
                # file handler get_info() method with info_via="handler" or you
                # can define the time coverage here directly:
                time_coverage=("2007-01-01 13:00:00", "2007-01-14 13:00:00")
            )

        FileSet to open MHS files:

        .. code-block:: python

            from typhon.files import FileSet, MHS_HDF

            # Define a fileset consisting of multiple files:
            files = FileSet(
                path="/dir/{year}/{month}/{day}/{hour}{minute}{second}.nc",
                name="MHS",
                handler=MHS_HDF(),
            )

            # Find some files of the fileset:
            for file in files.find("2017-01-01", "2017-01-02"):
                # Should print the path of the file and its time coverage:
                print(file)

    References:
        The FileSet class is inspired by the implemented dataset classes in
        atmlab_ developed by Gerrit Holl.

        .. _atmlab: http://www.radiativetransfer.org/tools/

    """

    # Required temporal placeholders that can be overridden by the user but
    # not deleted:
    _time_placeholder = {
        # "placeholder_name": [regex to find the placeholder]
        "year": r"\d{4}",
        "year2": r"\d{2}",
        "month": r"\d{2}",
        "day": r"\d{2}",
        "doy": r"\d{3}",
        "hour": r"\d{2}",
        "minute": r"\d{2}",
        "second": r"\d{2}",
        "decisecond": r"\d{1}",
        "centisecond": r"\d{2}",
        "millisecond": r"\d{3}",
        "microsecond": r"\d{6}",
        #"nanosecond": r"\d{9}",
        "end_year": r"\d{4}",
        "end_year2": r"\d{2}",
        "end_month": r"\d{2}",
        "end_day": r"\d{2}",
        "end_doy": r"\d{3}",
        "end_hour": r"\d{2}",
        "end_minute": r"\d{2}",
        "end_second": r"\d{2}",
        "end_decisecond": r"\d{1}",
        "end_centisecond": r"\d{2}",
        "end_millisecond": r"\d{3}",
        "end_microsecond": r"\d{6}",
        #"end_nanosecond": r"\d{9}",
    }

    _temporal_resolution = OrderedDict({
        # time placeholder: [pandas frequency, resolution rank]
        "year": timedelta(days=366),
        "month": timedelta(days=31),
        "day": timedelta(days=1),
        "hour": timedelta(hours=1),
        "minute": timedelta(minutes=1),
        "second": timedelta(seconds=1),
        "decisecond": timedelta(microseconds=100000),
        "centisecond": timedelta(microseconds=10000),
        "millisecond": timedelta(microseconds=1000),
        "microsecond": timedelta(microseconds=1),
    })

    # If one has a year with two-digit representation, all years equal or
    # higher than this threshold are based onto 1900, all years below are based
    # onto 2000.
    year2_threshold = 65

    # Default handler
    default_handler = {
        "nc": NetCDF4,
        "h5": NetCDF4, #HDF5,
        "txt": CSV,
        "csv": CSV,
        "asc": CSV,
    }

    # Special characters that show whether a path contains a regex or
    # placeholder:
    _special_chars = ["{", "*", "[", "<", "(", "?", "!", "|"]

    # FIXME OLE: On windows we can't have backslash here because it is the
    #            directory separator. Not sure if we need the \\ on Unix
    #            in _special_chars, but left it here to not break anything
    if platform != "win32":
        _special_chars += "\\"

    def __init__(
            self, path, handler=None, name=None, info_via=None,
            time_coverage=None, info_cache=None, exclude=None,
            placeholder=None, max_threads=None, max_processes=None,
            worker_type=None, read_args=None, write_args=None,
            post_reader=None, compress=True, decompress=True, temp_dir=None,
    ):
        """Initialize a FileSet object.

        Args:
            path: A string with the complete path to the files. The
                string can contain placeholder such as {year}, {month},
                etc. See below for a complete list. The direct use of
                restricted regular expressions is also possible. Please note
                that instead of dots '.' the asterisk '\\*' is interpreted as
                wildcard. If no placeholders are given, the path must point to
                a file. This fileset is then seen as a single file set.
                You can also define your own placeholders by using the
                parameter *placeholder*.
            name: The name of the fileset.
            handler: An object which can handle the fileset files.
                This fileset class does not care which format its files have
                when this file handler object is given. You can use a file
                handler class from typhon.files, use
                :class:`~typhon.files.handlers.common.FileHandler` or write
                your own class. If no file handler is given, an adequate one is
                automatically selected for the most common filename suffixes.
                Please note that if no file handler is specified (and none
                could set automatically), this fileset's functionality is
                restricted.
            info_via: Defines how further information about the file will
                be retrieved (e.g. time coverage). Possible options are
                *filename*, *handler* or *both*. Default is *filename*. That
                means that the placeholders in the file's path will be parsed
                to obtain information. If this is *handler*, the
                :meth:`~typhon.files.handlers.common.FileInfo.get_info` method
                is used. If this is *both*, both options will be executed but
                the information from the file handler overwrites conflicting
                information from the filename.
            info_cache: Retrieving further information (such as time coverage)
                about a file may take a while, especially when *get_info* is
                set to *handler*. Therefore, if the file information is cached,
                multiple calls of :meth:`find` (for time periods that
                are close) are significantly faster. Specify a name to a file
                here (which need not exist) if you wish to save the information
                data to a file. When restarting your script, this cache is
                used.
            time_coverage: If this fileset consists of multiple files, this
                parameter is the relative time coverage (i.e. a timedelta, e.g.
                "1 hour") of each file. If the ending time of a file cannot be
                retrieved by its file handler or filename, it is then its
                starting time + *time_coverage*. Can be a timedelta object or
                a string with time information (e.g. "2 seconds"). Otherwise
                the missing ending time of each file will be set to its
                starting time. If this fileset consists of a single file, then
                this is its absolute time coverage. Set this to a tuple of
                timestamps (datetime objects or strings). Otherwise the period
                between year 1 and 9999 will be used as a default time
                coverage.
            exclude: A list of time periods (tuples of two timestamps) or
                filenames (strings) that will be excluded when searching for
                files of this fileset.
            placeholder: A dictionary with pairs of placeholder name and a
                regular expression matching its content. These are user-defined
                placeholders, the standard temporal placeholders do not have to
                be defined.
            max_threads: Maximal number of threads that will be used to
                parallelise some methods (e.g. writing in background). This
                sets also the default for
                :meth:`~typhon.files.fileset.FileSet.map`-like methods
                (default is 3).
            max_processes: Maximal number of processes that will be used to
                parallelise some methods. This sets also the default for
                :meth:`~typhon.files.fileset.FileSet.map`-like methods
                (default is 8).
            worker_type: The type of the workers that will be used to
                parallelise some methods. Can be *process* (default) or
                *thread*.
            read_args: Additional keyword arguments in a dictionary that should
                always be passed to :meth:`read`.
            write_args: Additional keyword arguments in a dictionary that
                should always be passed to :meth:`write`.
            post_reader: A reference to a function that will be called *after*
                reading a file. Can be used for post-processing or field
                selection, etc. Its signature must be
                `callable(file_info, file_data)`.
            temp_dir: You can set here your own temporary directory that this
                FileSet object should use for compressing and decompressing
                files. Per default it uses the tempdir given by
                `tempfile.gettempdir` (see :func:`tempfile.gettempdir`).
            compress: If true and `path` ends with a compression
                suffix (such as *.zip*, *.gz*, *.b2z*, etc.), newly created
                files will be compressed after writing them to disk. Default
                value is true.
            decompress: If true and `path` ends with a compression
                suffix (such as *.zip*, *.gz*, *.b2z*, etc.), files will be
                decompressed before reading them. Default value is true.

        You can use regular expressions or placeholders in `path` to
        generalize the files path. Placeholders are going to be captured and
        returned by file-finding methods such as :meth:`find`. Temporal
        placeholders will be converted to datetime objects and represent a
        file's time coverage. Allowed temporal placeholders in the `path`
        argument are:

        +-------------+------------------------------------------+------------+
        | Placeholder | Description                              | Example    |
        +=============+==========================================+============+
        | year        | Four digits indicating the year.         | 1999       |
        +-------------+------------------------------------------+------------+
        | year2       | Two digits indicating the year. [1]_     | 58 (=2058) |
        +-------------+------------------------------------------+------------+
        | month       | Two digits indicating the month.         | 09         |
        +-------------+------------------------------------------+------------+
        | day         | Two digits indicating the day.           | 08         |
        +-------------+------------------------------------------+------------+
        | doy         | Three digits indicating the day of       | 002        |
        |             | the year.                                |            |
        +-------------+------------------------------------------+------------+
        | hour        | Two digits indicating the hour.          | 22         |
        +-------------+------------------------------------------+------------+
        | minute      | Two digits indicating the minute.        | 58         |
        +-------------+------------------------------------------+------------+
        | second      | Two digits indicating the second.        | 58         |
        +-------------+------------------------------------------+------------+
        | millisecond | Three digits indicating the millisecond. | 999        |
        +-------------+------------------------------------------+------------+

        .. [1] Numbers lower than 65 are interpreted as 20XX while numbers
            equal or greater are interpreted as 19XX (e.g. 65 = 1965,
            99 = 1999)

        All those place holders are also allowed to have the prefix *end*
        (e.g. *end_year*). They represent the end of the time coverage.

        Moreover, you are allowed do define your own placeholders by using the
        parameter `placeholder` or :meth:`set_placeholders`. Their names
        must consist of alphanumeric signs (underscores are also allowed).
        """

        # Initialize member variables:
        self._name = None
        self.name = name

        # Flag whether this is a single file fileset (will be derived in the
        # path setter method automatically):
        self.single_file = None

        # Complete the standard time placeholders. This must be done before
        # setting the path to the fileset's files.
        self._time_placeholder = self._complete_placeholders_regex(
            self._time_placeholder
        )

        # Placeholders that can be changed by the user:
        self._user_placeholder = {}

        # The path parameters (will be set and documented in the path setter
        # method):
        self._path = None
        self._path_placeholders = None
        self._end_time_superior = None
        self._path_extension = None
        self._filled_path = None
        self._base_dir = None
        self._sub_dir = ""
        self._sub_dir_chunks = []
        self._sub_dir_time_resolution = None
        self.path = path

        # Add user-defined placeholders:
        if placeholder is not None:
            self.set_placeholders(**placeholder)

        if handler is None:
            # Try to derive the file handler from the files extension but
            # before we might remove potential compression suffixes:
            basename, extension = os.path.splitext(self.path)
            if typhon.files.is_compression_format(extension.lstrip(".")):
                _, extension = os.path.splitext(basename)

            extension = extension.lstrip(".")

            self.handler = self.default_handler.get(extension, None)
            if self.handler is not None:
                self.handler = self.handler()
        else:
            self.handler = handler

        # Defines which method will be used by .get_info():
        if info_via is None or info_via == "filename":
            self.info_via = "filename"
        else:
            if self.handler is None:
                raise NoHandlerError(f"Cannot set 'info_via' to '{info_via}'!")
            else:
                self.info_via = info_via

        # A list of time periods that will be excluded when searching files:
        self._exclude_times = None
        self._exclude_files = {}
        if exclude is not None:
            self.exclude_files(
                [file for file in exclude if isinstance(file, str)]
            )
            self.exclude_times(
                [times for times in exclude if isinstance(times, tuple)]
            )

        # The default worker settings for map-like functions
        self.max_threads = 3 if max_threads is None else max_threads
        self.max_processes = 4 if max_processes is None else max_processes
        self.worker_type = "process" if worker_type is None else worker_type

        # The default settings for read and write methods
        self.read_args = {} if read_args is None else read_args
        self.write_args = {} if write_args is None else write_args
        self.post_reader = post_reader

        self.compress = compress
        self.decompress = decompress
        self.temp_dir = temp_dir

        self._time_coverage = None
        self.time_coverage = time_coverage

        # Multiple calls of .find() can be very slow when using the handler as
        # as information retrieving method. Hence, we use a cache to store the
        # names and time coverages of already touched files in this dictionary.
        self.info_cache_filename = info_cache
        self.info_cache = {}
        if self.info_cache_filename is not None:
            try:
                # Load the time coverages from a file:
                self.load_cache(self.info_cache_filename)
            except Exception as e:
                raise e
            else:
                # Save the time coverages cache into a file before exiting.
                # This will be executed as well when the python code is
                # aborted due to an exception. This is normally okay, but what
                # happens if the error occurs during the loading of the time
                # coverages? We would overwrite the cache with nonsense.
                # Therefore, we need this code in this else block.
                atexit.register(FileSet.save_cache,
                                self, self.info_cache_filename)

        # Writing processes can be moved to background threads. But we do want
        # to have too many backgrounds threads running at the same time, so we
        # create FIFO queue. The queue limits the number of parallel threads
        # to a maximum. The users can also make sure that all writing threads
        # are finished before they move on in the code.
        # TODO: We cannot use queues as attributes for FileSet because they
        # TODO: cannot be pickled.
        # self._write_queue = Queue(max_threads)

        # Dictionary for holding links to other filesets:
        self._link = {}

    def __iter__(self):
        return iter(self.find())

    def __contains__(self, item):
        """Checks whether a timestamp is covered by this fileset.

        Notes:
            This only gives proper results if the fileset consists of
            continuous data (files that covers a time span instead of only one
            timestamp).

        Args:
            item: Either a string with time information or datetime object.
                Can be also a tuple or list of strings / datetime objects that
                will be checked.

        Returns:
            True if timestamp is covered.
        """
        if isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise ValueError("Can only test single timestamps or time "
                                 "periods consisting of two timestamps")

            start = to_datetime(item[0])
            end = to_datetime(item[1])
        else:
            start = to_datetime(item)
            end = start + timedelta(microseconds=1)

        try:
            next(self.find(start, end, no_files_error=False, sort=False,))
            return True
        except StopIteration:
            return False

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            time_args = item[0]
            filters = item[1]
        else:
            time_args = item
            filters = None

        if isinstance(time_args, slice):
            return self.collect(
                time_args.start, time_args.stop, filters=filters,
            )
        elif isinstance(time_args, (datetime, str)):
            filename = self.find_closest(time_args, filters=filters)
            if filename is None:
                return None

            return self.read(filename)

    def __len__(self):
        return sum(1 for _ in self.find())

    def __setitem__(self, key, value):
        if isinstance(key, (tuple, list)):
            time_args = key[0]
            fill = key[1]
        else:
            time_args = key
            fill = None

        if isinstance(time_args, slice):
            start = time_args.start
            end = time_args.stop
        else:
            start = end = time_args

        filename = self.get_filename((start, end), fill=fill)
        self.write(value, filename)

    def __repr__(self):
        return str(self)

    def __str__(self):
        dtype = "Single-File" if self.single_file else "Multi-Files"

        info = "Name:\t" + self.name
        info += "\nType:\t" + dtype
        info += "\nFiles path:\t" + self.path
        if self._user_placeholder:
            info += "\nUser placeholder:\t" + str(self._user_placeholder)
        return info

    def align(self, other, start=None, end=None, matches=None,
              max_interval=None, return_info=True, compact=False,
              skip_errors=False):
        """Collect files from this fileset and a matching other fileset

        Warnings:
            The name of this method may change in future.

        This generator finds the matches between two filesets (the files that
        overlap each other in time), reads their content and yields them. The
        reading is done in parallel threads to enhance the performance.

        Args:
            other: Another fileset object.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional. If not given, it is
                datetime.min per default.
            end: End date. Same format as "start". If not given, it is
                datetime.max per default.
            matches: A list of matches between this fileset and `other`.
                Normally, this is just the return value of :meth:`match`. These
                matches will be used for aligning. If `matches` is given,
                `start`, `end` and `max_interval` will be ignored.
            max_interval: A time interval (as string, number or timedelta
                object) that expands the search time period for secondaries.
            return_info: Additionally, to the file content also its info object
                will be returned.
            compact: Not yet implemented. Decides how the collected data will
                be returned.
            skip_errors: Normally, when an exception is raised during
                reading a file, the program stops. But if this parameter is
                True, the error will be only printed as warning and this and
                its matching file are skipped.

        Yields:
            If `return_info` False, it yields two objects: the primary and
            secondary file content.

        .. code-block:: python

            fileset = FileSet(
                "old/path/{year}/{month}/{day}/{hour}{minute}{second}.nc",
            )

            # Delete all files in this fileset:
            fileset.delete()
        """
        start = None if start is None else to_datetime(start)
        end = None if end is None else to_datetime(end)

        # Find all overlapping files. Or has the user already given some
        # matches to us?
        if matches is None:
            matches = list(
                self.match(other, start, end, max_interval=max_interval)
            )
        primaries, secondaries = zip(*matches)

        # We have to consider the following to make the align method work
        # properly:
        # A) Load the files in parallel loading queues (done by icollect) and
        # read them only once even if it might be used by multiple primaries.
        # B) Secondaries that are going to be used by multiple primaries must
        # be cached until we are sure that they won't be needed any longer.

        # This deals with part A:
        # We need the list flattened and without duplicates.
        unique_secondaries = unique(
            secondary for match in secondaries for secondary in match
        )

        # Prepare the loader for the primaries and secondaries:
        primary_loader = self.icollect(
            files=primaries, error_to_warning=skip_errors
        )
        secondary_loader = other.icollect(
            files=unique_secondaries, return_info=True,
            error_to_warning=skip_errors
        )

        # Here we prepare part B:
        # We count how often we need the secondaries (some primaries may need
        # the same secondary). Later, we decrease the counter for the
        # secondary for each use. If the counter reaches zero, we can delete it
        # from the cache.
        secondary_usage = Counter(
            secondary for match in secondaries for secondary in match
        )

        # We will need this cache for secondaries that are used by multiple
        # primaries:
        cache = {}

        for match_id, primary_data in enumerate(primary_loader):

            if return_info:
                primary = [primaries[match_id], primary_data]
            else:
                primary = primary_data

            # We only need to load the secondary files that have not been
            # cached earlier:
            for secondary_file in matches[match_id][1]:

                if secondary_file not in cache:
                    secondary_loaded, secondary_data = next(secondary_loader)
                    if secondary_file != secondary_loaded:
                        raise AlignError(
                            f"Expected '{secondary_file}'\nbut "
                            f"'{secondary_loaded}' was loaded!\nDoes your "
                            f"fileset '{self.name}' contain files between that"
                            f"are completely overlapped by other files? "
                            f"Please exclude them via `exclude`."
                        )

                    # Add the loaded secondary to the cache
                    cache[secondary_file] = secondary_data
                else:
                    secondary_data = cache[secondary_file]

                # Decrease the counter for this secondary:
                secondary_usage[secondary_file] -= 1

                # Apparently, this secondary won't be needed any longer. Delete
                # it from the cache:
                if not secondary_usage[secondary_file]:
                    del cache[secondary_file]

                    # Tell the python interpreter explicitly to free up memory
                    # to improve performance (see
                    # https://stackoverflow.com/q/1316767/9144990):
                    gc.collect()

                # Check whether something went wrong:
                if primary_data is None or secondary_data is None \
                        and skip_errors:
                    # There was an exception during reading the primary or
                    # secondary file, therefore we skip this match:
                    continue

                if return_info:
                    secondary = [secondary_file, secondary_data]
                else:
                    secondary = secondary_data

                # Yield the primary and secondary to the user:
                yield primary, secondary



    @staticmethod
    def _pseudo_passer(*args):
        return args[0]

    def collect(self, start=None, end=None, files=None, return_info=False,
                **kwargs):
        """Load all files between two dates sorted by their starting time

        Notes
            This does not constrain the loaded data to the time period given by
            `start` and `end`. This fully loads all files that contain data in
            that time period, i.e. it returns also data that may exceed the
            time period.

        This parallelizes the reading of the files by using threads. This
        should give a speed up if the file handler's read function internally
        uses CPython code that releases the GIL. Note that this method is
        faster than :meth:`icollect` but also more memory consuming.

        Use this if you need all files at once but if want to use a for-loop
        consider using :meth:`icollect` instead.

        Args:
            start: The same as in :meth:`find`.
            end: The same as in :meth:`find`.
            files: If you have already a list of files that you want to
                process, pass it here. The list can contain filenames or lists
                (bundles) of filenames. If this parameter is given, it is not
                allowed to set `start` and `end` then.
            return_info: If true, return a FileInfo object with each content
                value indicating to which file the function was applied.
            **kwargs: Additional keyword arguments that are allowed
                for :meth:`map`. Some might be overwritten by this method.

        Returns:
            If `return_info` is True, two list are going to be returned:
            one with FileInfo objects of the files and one with the read
            content objects. Otherwise, the list with the read content objects
            only. The lists are sorted by the starting times of the files.

        Examples:

        .. code-block:: python

            ## Load all files between two dates:
            # Note: data may contain timestamps exceeding the given time period
            data = fileset.collect("2018-01-01", "2018-01-02")

            # The above is equivalent to this magic slicing:
            data = fileset["2018-01-01":"2018-01-02"]

            ## If you want to iterate through the files in a for loop, e.g.:
            for content in fileset.collect("2018-01-01", "2018-01-02"):
                # do something with file and content...

            # Then you should rather use icollect, which uses less memory:
            for content in fileset.icollect("2018-01-01", "2018-01-02"):
                # do something with file and content...

        """

        # Actually, this method is nothing else than a customized alias for the
        # map method:
        map_args = {
            **kwargs,
            "files": files,
            "start": start,
            "end": end,
            "worker_type": "thread",
            "on_content": True,
            "return_info": True,
        }

        if "func" not in map_args:
            map_args["func"] = self._pseudo_passer

        # If we used map with processes, it would need to pickle the data
        # coming from all workers. This would be very inefficient. Threads
        # are better because sharing data does not cost much and a file
        # reading function is typically IO-bound. However, if the reading
        # function consists mainly of pure python code that does not
        # release the GIL, this will slow down the performance.
        results = self.map(**map_args)

        # Tell the python interpreter explicitly to free up memory to improve
        # performance (see https://stackoverflow.com/q/1316767/9144990):
        gc.collect()

        # We do not want to have any None as data
        files, data = zip(*[
            [info, content]
            for info, content in results
            if content is not None
        ])

        if return_info:
            return list(files), list(data)
        else:
            return list(data)

    def icollect(self, start=None, end=None, files=None,
                 **kwargs):
        """Load all files between two dates sorted by their starting time

        Does the same as :meth:`collect` but works as a generator. Instead of
        loading all files at the same time, it loads them in chunks (the chunk
        size is defined by `max_workers`). Hence, this method is less memory
        space consuming but slower than :meth:`collect`. Simple hint: use this
        in for-loops but if you need all files at once, use :meth:`collect`
        instead.

        Args:
            start: The same as in :meth:`find`.
            end: The same as in :meth:`find`.
            files: If you have already a list of files that you want to
                process, pass it here. The list can contain filenames or lists
                (bundles) of filenames. If this parameter is given, it is not
                allowed to set *start* and *end* then.
            **kwargs: Additional keyword arguments that are allowed
                for :meth:`imap`. Some might be overwritten by this method.

        Yields:
            A tuple of the FileInfo object of a file and its content. These
            tuples are yielded sorted by its file starting time.

        Examples:

        .. code-block:: python

            ## Perfect for iterating over many files.
            for content in fileset.icollect("2018-01-01", "2018-01-02"):
                # do something with file and content...

            ## If you want to have all files at once, do not use this:
            data_list = list(fileset.icollect("2018-01-01", "2018-01-02"))

            # This version is faster:
            data_list = fileset.collect("2018-01-01", "2018-01-02")
        """

        # Actually, this method is nothing else than a customized alias for the
        # imap method:
        map_args = {
            **kwargs,
            "files": files,
            "start": start,
            "end": end,
            "worker_type": "thread",
            "on_content": True,
        }

        if "func" not in map_args:
            map_args["func"] = self._pseudo_passer

        yield from self.imap(**map_args)

    def copy(self):
        """Create a so-called deep-copy of this fileset object

        Notes
            This method does not copy any files. If you want to do so, use
            :meth:`move` with the parameter `copy=True`.

        Returns:
            The copied fileset object
        """
        return deepcopy(self)

    def delete(self, dry_run=False, **kwargs):
        """Remove files in this fileset from the disk

        Warnings:
            The deleting of the files cannot be undone! There is no prompt
            before deleting the files. Use this function with caution!

        Args:
            dry_run: If true, all files that would be deleted are printed.
            **kwargs: Additional keyword arguments that are allowed
                for :meth:`find` such as `start`, `end` or `files`.

        Returns:
            Nothing

        Examples:

        .. code-block:: python

            fileset = FileSet(
                "old/path/{year}/{month}/{day}/{hour}{minute}{second}.nc",
            )

            # Delete all files in this fileset:
            fileset.delete()
        """

        if dry_run:
            self.map(FileSet._dry_delete, **kwargs)
        else:
            # Delete the files
            self.map(
                FileSet._delete_single_file, **kwargs
            )

    @staticmethod
    def _delete_single_file(file):
        logger.info(f"Delete '{file}'!")

        os.remove(file)

    @staticmethod
    def _dry_delete(file):
        print(f"[Dry] Delete '{file}'!")

    def detect(self, test, *args, **kwargs):
        """Search for anomalies in fileset

        Args:
            test: Can be *duplicates*, *overlaps*, *enclosed* or a reference
                to your own function that expects two :class:`FileInfo` objects
                as parameters.
            *args: Positional arguments for :meth:`find`.
            **kwargs: Keyword arguments for :meth:`find`.

        Yields:
            A :class:`FileInfo` object of each file that fulfilled the test.
        """
        if not callable(test):
            if test == "enclosed":
                yield from self._detect_enclosed(*args, **kwargs)

        if test is None:
            raise ValueError("Need a valid test function or name!")

        previous = None
        for file in self.find(*args, **kwargs):
            if previous is None:
                continue
            if test(previous, file):
                yield file

    def _detect_enclosed(self, *args, **kwargs):
        yield None

    def exclude_times(self, periods):
        if periods is None or not periods:
            self._exclude_times = None
        else:
            self._exclude_times = IntervalTree(periods)

    def exclude_files(self, filenames):
        self._exclude_files = set(filenames)

    def is_excluded(self, file):
        """Checks whether a file is excluded from this FileSet.

        Args:
            file: A file info object.

        Returns:
            True or False
        """
        if file.path in self._exclude_files:
            return True

        if self._exclude_times is None:
            return False

        return file.times in self._exclude_times

    def find(
            self, start=None, end=None, sort=True, only_path=False,
            bundle=None, filters=None, no_files_error=True,
    ):
        """ Find all files of this fileset in a given time period.

        The *start* and *end* parameters build a semi-open interval: only the
        files that are equal or newer than *start* and older than *end* are
        going to be found.

        While searching this method checks whether the file lies in the time
        periods given by `exclude` while initializing.

        Args:
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional. If not given, it is
                datetime.min per default.
            end: End date. Same format as "start". If not given, it is
                datetime.max per default.
            sort: If true, all files will be yielded sorted by their starting
                and ending time. Default is true.
            only_path: If true, only the paths of the files will be returned
                not their :class:`~typhon.files.handlers.common.FileInfo`
                object.
            bundle: Instead of only yielding one file at a time, you can get a
                bundle of files. There are two possibilities: by setting this
                to an integer, you can define the size of the bundle directly
                or by setting this to a string (e.g. *1H*),
                you can define the time period of one bundle. See
                http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
                for allowed time specifications. Default value is 1. This
                argument will be ignored when having a single-file fileset.
                When using *bundle*, the returned files will always be sorted
                ignoring the state of the *sort* argument.
            filters: Limits user-defined placeholder to certain values.
                Must be a dictionary where the keys are the names of
                user-defined placeholders and the values either strings or
                lists of strings with allowed placeholder values (can be
                represented by regular expressions). If the key name starts
                with a *!* (exclamation mark), the value represent a black
                list (values that are not allowed).
            no_files_error: If true, raises an NoFilesError when no
                files are found.

        Yields:
            Either a :class:`~typhon.files.handlers.common.FileInfo` object for
            each found file or - if *bundle_size* is not None - a list of
            :class:`~typhon.files.handlers.common.FileInfo` objects.

        Examples:

        .. code-block:: python

            # Define a fileset consisting of multiple files:
            fileset = FileSet(
                "/dir/{year}/{month}/{day}/{hour}{minute}{second}.nc"
            )

            # Find some files of the fileset:
            for file in fileset.find("2017-01-01", "2017-01-02"):
                # file is a FileInfo object that has the attribute path
                # and times.
                print(file.path)  # e.g. "/dir/2017/01/01/120000.nc"
                print(file.times)  # list of two datetime objects
        """

        # The user can give strings instead of datetime objects:
        start = datetime.min if start is None else to_datetime(start)
        end = datetime.max if end is None else to_datetime(end)

        # We want to have a semi-open interval as explained in the doc string.
        end -= timedelta(microseconds=1)

        if end < start:
            raise ValueError(
                "The start must be smaller than the end parameter!")

        logger.info("Find files between %s and %s!" % (start, end))

        # Special case: the whole fileset consists of one file only.
        if self.single_file:
            if os.path.isfile(self.path):
                file_info = self.get_info(self.path)
                if IntervalTree.interval_overlaps(
                        file_info.times, (start, end)):
                    yield file_info
                elif no_files_error:
                    raise NoFilesError(self, start, end)
                return
            else:
                raise ValueError(
                    "The path of '%s' neither contains placeholders"
                    " nor is a path to an existing file!" % self.name)

        # Files may exceed the time coverage of their directories. For example,
        # a file located in the directory of 2018-01-13 contains data from
        # 2018-01-13 18:00:00 to 2018-01-14 02:00:00. In order to find them, we
        # must include the previous sub directory into the search range:
        if self._sub_dir_time_resolution is None or start == datetime.min:
            dir_start = start
        else:
            dir_start = start - self._sub_dir_time_resolution

        # Filter handling:
        if filters is None:
            # We can apply the standard path regex:
            regex = re.compile(self._filled_path)
            white_list = {}
            black_list = {}
        else:
            # Complete the regexes of the filters (simply adding curls around
            # them):
            white_list = self._complete_placeholders_regex(
                {f: v for f, v in filters.items() if not f.startswith("!")}
            )

            # The new regex for all files:
            regex = self._fill_placeholders(
                self.path,
                extra_placeholder=white_list,
                compile=True,
            )

            def convert(value):
                if value is None:
                    return None
                elif isinstance(value, (tuple, list)):
                    return re.compile(f"{'|'.join(value)}")
                else:
                    return re.compile(f"{value}")

            black_list = {
                f.lstrip("!"): convert(v)
                for f, v in filters.items()
                if f.startswith("!")
            }

        if filters is not None:
            logger.info(f"Loaded filters:\nWhitelist: {white_list}"
                  f"\nBlacklist: {black_list}")

        # Find all files by iterating over all searching paths and check
        # whether they match the path regex and the time period.
        file_finder = (
            file_info
            for path, _ in self._get_search_dirs(dir_start, end, white_list)
            for file_info in self._get_matching_files(path, regex, start, end,)
            if not black_list or self._check_file(black_list, file_info.attr)
        )

        # Even if no files were found, the user does not want to know.
        if not no_files_error:
            yield from self._prepare_find_return(
                file_finder, sort, only_path, bundle
            )
            return

        # The users wants an error to be raised if no files were found. Since
        # the file_finder is an iterator, we have to check whether it is empty.
        # I do not know whether there is a more pythonic way but Matthew
        # Flaschen shows how to do it with itertools.tee:
        # https://stackoverflow.com/a/3114423
        return_files, check_files = tee(file_finder)
        try:
            next(check_files)

            # We have found some files and can return them
            yield from self._prepare_find_return(
                return_files, sort, only_path, bundle
            )
        except StopIteration as err:
            raise NoFilesError(self, start, end)

    def _get_search_dirs(self, start, end, white_list):
        """Yields all searching directories for a time period.

        Args:
            start: Datetime that defines the start of a time interval.
            end: Datetime that defines the end of a time interval. The time
                coverage of the files should overlap with this interval.
            white_list: A dictionary that limits placeholders to certain
                values.

        Returns:
            A tuple of path as string and parsed placeholders as dictionary.
        """

        # Goal: Search for all directories that match the path regex and is
        # between start and end date.
        # Strategy: Go through each folder in the hierarchy and find the ones
        # that match the regex so far. Filter out folders that does not overlap
        # with the given time interval.
        search_dirs = [(self._base_dir, {}), ]

        # If the directory does not contain regex or placeholders, we simply
        # return the base directory
        if not self._sub_dir:
            return search_dirs

        for subdir_chunk in self._sub_dir_chunks:
            # Sometimes there is a sub directory part that has no
            # regex/placeholders:
            if not any(True for ch in subdir_chunk
                       if ch in self._special_chars):
                # We can add this sub directory part because it will always
                # match to our path
                search_dirs = [
                    (os.path.join(old_dir, subdir_chunk), attr)
                    for old_dir, attr in search_dirs
                ]
                continue

            # The sub directory covers a certain time coverage, we make
            # sure that it is included into the search range.
            start_check = set_time_resolution(
                start, self._get_time_resolution(subdir_chunk)[0]
            )
            end_check = set_time_resolution(
                end, self._get_time_resolution(subdir_chunk)[0]
            )

            # compile the regex for this sub directory:
            regex = self._fill_placeholders(
                subdir_chunk, extra_placeholder=white_list, compile=True
            )
            search_dirs = [
                (new_dir, attr)
                for search_dir in search_dirs
                for new_dir, attr in self._get_matching_dirs(search_dir, regex)
                if self._check_placeholders(attr, start_check, end_check)
            ]

        return search_dirs

    def _get_matching_dirs(self, dir_with_attrs, regex):
        base_dir, dir_attr = dir_with_attrs
        for new_dir in glob.iglob(os.path.join(base_dir + "*", "")):
            # The glob function yields full paths, but we want only to check
            # the new pattern that was added:
            basename = new_dir[len(base_dir):].rstrip(os.sep)
            try:
                new_attr = {
                    **dir_attr,
                    **self.parse_filename(basename, regex)
                }
                yield new_dir, new_attr
            except ValueError:
                pass

    def _check_placeholders(self, attr, start, end):
        attr_start, attr_end = self._to_datetime_args(attr)
        attr_end = {**attr_start, **attr_end}
        year = attr_start.get("year", None)
        if year is not None:
            try:
                return datetime(**attr_start) >= start \
                    and datetime(**attr_end) <= end
            except:
                return year >= start.year and attr_end["year"] <= end.year

        return True

    def _get_matching_files(self, path, regex, start, end,):
        """Yield files that matches the search conditions.

        Args:
            path: Path to the directory that contains the files that should be
                checked.
            regex: A regular expression that should match the filename.
            start: Datetime that defines the start of a time interval.
            end: Datetime that defines the end of a time interval. The time
                coverage of the file should overlap with this interval.

        Yields:
            A FileInfo object with the file path and time coverage
        """

        for filename in glob.iglob(os.path.join(path, "*")):
            if regex.match(filename):
                file_info = self.get_info(filename)

                # Test whether the file is overlapping the interval between
                # start and end date.
                if IntervalTree.interval_overlaps(
                        file_info.times, (start, end))\
                        and not self.is_excluded(file_info):
                    yield file_info

    @staticmethod
    def _check_file(black_list, placeholders):
        """Check whether placeholders are filled with something forbidden

        Args:
            black_list: A dictionary with placeholder name and content that
                should be filtered out.
            placeholders: A dictionary with placeholders and their fillings.

        Returns:
            False if the placeholders are filled with something that is
            forbidden due to the filters. True otherwise.
        """
        for placeholder, forbidden in black_list.items():
            value = placeholders.get(placeholder, None)
            if value is None:
                continue

            if forbidden.match(value):
                return False

        return True

    @staticmethod
    def _prepare_find_return(file_iterator, sort, only_path, bundle_size):
        """Prepares the return value of the find method.

        Args:
            file_iterator: Generator function that yields the found files.
            sort: If true, all found files will be sorted according to their
                starting and ending times.
            only_path:
            bundle_size: See the documentation of the *bundle* argument in
                :meth:`find` method.

        Yields:
            Either one FileInfo object or - if bundle_size is set - a list of
            FileInfo objects.
        """
        # We always want to have sorted files if we want to bundle them.
        if sort or isinstance(bundle_size, int):
            # Sort the files by starting and ending time:
            file_iterator = sorted(
                file_iterator, key=lambda x: (x.times[0], x.times[1])
            )

        if bundle_size is None:
            yield from file_iterator
            return

        # The argument bundle was defined. Either it sets the bundle size
        # directly via a number or indirectly by setting time periods.
        if isinstance(bundle_size, int):
            files = list(file_iterator)

            yield from (
                files[i:i + bundle_size]
                for i in range(0, len(files), bundle_size)
            )
        elif isinstance(bundle_size, str):
            files = list(file_iterator)

            # We want to split the files into hourly (or daily, etc.) bundles.
            # pandas provides a practical grouping function.
            time_series = pd.Series(
                files,
                [file.times[0] for file in files]
            )
            yield from (
                bundle[1].values.tolist()
                for bundle in time_series.groupby(
                    pd.Grouper(freq=bundle_size))
                if bundle[1].any()
            )
        else:
            raise ValueError(
                "The parameter bundle must be a integer or string!")

    def find_closest(self, timestamp, filters=None):
        """Find the closest file to a timestamp

        Args:
            timestamp: date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            filters: The same filter argument that is allowed for
                :meth:`find`.

        Returns:
            The FileInfo object of the found file. If no file was found, a
            NoFilesError is raised.
        """

        # Special case: the whole fileset consists of one file only.
        if self.single_file:
            if os.path.isfile(self.path):
                # We do not have to check the time coverage since there this is
                # automatically the closest file to the timestamp.
                return self.path
            else:
                raise ValueError(
                    "The path parameter of '%s' does not contain placeholders"
                    " and is not a path to an existing file!" % self.name)

        timestamp = to_datetime(timestamp)

        # We might need some more fillings than given by the user therefore
        # we need the error catching:
        try:
            # Maybe there is a file with exact this timestamp?
            path = self.get_filename(timestamp, )
            if os.path.isfile(path):
                return self.get_info(path)
        except (UnknownPlaceholderError, UnfilledPlaceholderError):
            pass

        # We need to find all files that are around the given timestamp. Hence,
        # we use the sub directory time resolution to specify a time period
        # within the file should possibly be:
        if self._sub_dir_time_resolution is None:
            start = datetime.min
            end = datetime.max
        else:
            start = timestamp - self._sub_dir_time_resolution
            end = timestamp + self._sub_dir_time_resolution

        files = list(self.find(start, end, sort=False, filters=filters))

        if not files:
            return None

        times = [file.times for file in files]

        # Either we find a file that covers the certain timestamp:
        for index, time_coverage in enumerate(times):
            if IntervalTree.interval_contains(time_coverage, timestamp):
                return files[index]

        # Or we find the closest file.
        intervals = np.min(np.abs(np.asarray(times) - timestamp), axis=1)
        return files[np.argmin(intervals)]

    def get_filename(
            self, times, template=None, fill=None):
        """Generate the full path and name of a file for a time period

        Use :meth:`parse_filename` if you want retrieve information from the
        filename instead.

        Args:
            times: Either a tuple of two datetime objects representing start
                and end time or simply one datetime object (for discrete
                files).
            template: A string with format placeholders such as {year} or
                {day}. If not given, the template in `FileSet.path` is used.
            fill: A dictionary with fillings for user-defined placeholder.

        Returns:
            A string containing the full path and name of the file.

        Example:

        .. code-block:: python

            fileset.get_filename(
                datetime(2016, 1, 1),
                "{year2}/{month}/{day}.dat",
            )
            # Returns "16/01/01.dat"

            fileset.get_filename(
                ("2016-01-01", "2016-12-31"),
                "{year}{month}{day}-{end_year}{end_month}{end_day}.dat",
            )
            # Returns "20160101-20161231.dat"

        """
        if isinstance(times, (tuple, list)):
            start_time = to_datetime(times[0])
            end_time = to_datetime(times[1])
        else:
            start_time = to_datetime(times)
            end_time = start_time

        if template is None:
            template = self.path

        # Remove the automatic regex completion from the user placeholders and
        # use them as default fillings
        default_fill = {
            p: self._remove_group_capturing(p, v)
            for p, v in self._user_placeholder.items()
        }
        if fill is None:
            fill = default_fill
        else:
            fill = {**default_fill, **fill}

        try:
            # Fill all placeholders variables with values
            filename = template.format(
                year=start_time.year, year2=str(start_time.year)[-2:],
                month="{:02d}".format(start_time.month),
                day="{:02d}".format(start_time.day),
                doy="{:03d}".format(
                    (start_time - datetime(start_time.year, 1, 1)).days
                    + 1),
                hour="{:02d}".format(start_time.hour),
                minute="{:02d}".format(start_time.minute),
                second="{:02d}".format(start_time.second),
                millisecond="{:03d}".format(
                    int(start_time.microsecond / 1000)),
                end_year=end_time.year, end_year2=str(end_time.year)[-2:],
                end_month="{:02d}".format(end_time.month),
                end_day="{:02d}".format(end_time.day),
                end_doy="{:03d}".format(
                    (end_time - datetime(end_time.year, 1, 1)).days
                    + 1),
                end_hour="{:02d}".format(end_time.hour),
                end_minute="{:02d}".format(end_time.minute),
                end_second="{:02d}".format(end_time.second),
                end_millisecond="{:03d}".format(
                    int(end_time.microsecond/1000)),
                **fill,
            )

            # Some placeholders might be unfilled:
            if any((c in self._special_chars) for c in filename):
                raise UnfilledPlaceholderError(self.name, filename)

            return filename

        except KeyError:
            raise UnknownPlaceholderError(self.name)

    @expects_file_info()
    def get_info(self, file_info, retrieve_via=None):
        """Get information about a file.

        How the information will be retrieved is defined by

        Args:
            file_info: A string, path-alike object or a
                :class:`~typhon.files.handlers.common.FileInfo` object.
            retrieve_via: Defines how further information about the file will
                be retrieved (e.g. time coverage). Possible options are
                *filename*, *handler* or *both*. Default is the value of the
                *info_via* parameter during initialization of this FileSet
                object. If this is *filename*, the placeholders in the file's
                path will be parsed to obtain information. If this is
                *handler*, the
                :meth:`~typhon.files.handlers.common.FileInfo.get_info` method
                is used. If this is *both*, both options will be executed but
                the information from the file handler overwrites conflicting
                information from the filename.

        Returns:
            A :meth:`~typhon.files.handlers.common.FileInfo` object.
        """
        # We want to save time in this routine, therefore we first check
        # whether we cached this file already.

        if file_info.path in self.info_cache:
            return self.info_cache[file_info.path]

        # We have not processed this file before.

        info = file_info.copy()
        if self.single_file:
            info.times = self.time_coverage

        if retrieve_via is None:
            retrieve_via = self.info_via

        # Parsing the filename
        if retrieve_via in ("filename", "both"):
            filled_placeholder = self.parse_filename(info.path)

            filename_info = FileInfo(
                info.path, self._retrieve_time_coverage(filled_placeholder),
                # Filter out all placeholder that are not coming from the user
                {k: v for k, v in filled_placeholder.items()
                 if k in self._user_placeholder}
            )
            info.update(filename_info)

        # Using the handler for getting more information
        if retrieve_via in ("handler", "both"):
            with typhon.files.decompress(info.path, tmpdir=self.temp_dir) as \
                    decompressed_path:
                decompressed_file = info.copy()
                decompressed_file.path = decompressed_path
                handler_info = self.handler.get_info(decompressed_file)
                info.update(handler_info)

        if info.times[0] is None:
            if info.times[1] is None:
                # This is obviously a non-temporal fileset, set the times to
                # minimum and maximum so we have no problem to find it
                info.times = [datetime.min, datetime.max]
            else:
                # Something went wrong, we need a starting time if we have an
                # ending time.
                raise ValueError(
                    "Could not retrieve the starting time information from "
                    "the file '%s' from the %s fileset!"
                    % (info.path, self.name)
                )
        elif info.times[1] is None:
            # Sometimes the files have only a starting time. But if the user
            # has defined a timedelta for the coverage, the ending time can be
            # calculated from this. Otherwise this is a FileSet that has only
            # files that are discrete in time
            if isinstance(self.time_coverage, timedelta):
                info.times[1] = info.times[0] + self.time_coverage
            else:
                info.times[1] = info.times[0]

        self.info_cache[info.path] = info
        return info

    def dislink(self, name_or_fileset):
        """Remove the link between this and another fileset

        Args:
            name_or_fileset: Name of a fileset or the FileSet object itself. It
                must be linked to this fileset. Otherwise a KeyError will be
                raised.

        Returns:
            None
        """
        if isinstance(name_or_fileset, FileSet):
            del self._link[name_or_fileset.name]
        else:
            del self._link[name_or_fileset]

    def link(self, other_fileset, linker=None):
        """Link this fileset with another FileSet

        If one file is read from this fileset, its corresponding file from
        `other_fileset` will be read, too. Their content will then be merged by
        using the file handler's data merging function. If it is not
        implemented, it tries to derive a standard merging function from known
        data types.

        Args:
            other_fileset: Other FileSet-like object.
            linker: Reference to a function that searches for the corresponding
                file in *other_fileset* for a given file from this fileset.
                Must accept *other_fileset* as first and a
                :class:`~typhon.files.handlers.common.FileInfo` object as
                parameters. It must return a FileInfo of the corresponding
                file. If none is given,
                :meth:`~typhon.files.fileset.FileSet.get_filename`
                will be used as default.

        Returns:
            None
        """

        self._link[other_fileset.name] = {
            "target": other_fileset,
            "linker": linker,
        }

    def load_cache(self, filename):
        """Load the information cache from a JSON file

        Returns:
            None
        """
        if filename is not None and os.path.exists(filename):
            try:
                with open(filename) as file:
                    json_info_cache = json.load(file)
                    # Create FileInfo objects from json dictionaries:
                    info_cache = {
                        json_dict["path"]: FileInfo.from_json_dict(json_dict)
                        for json_dict in json_info_cache
                    }
                    self.info_cache.update(info_cache)
            except Exception as err:
                warnings.warn(
                    f"Could not load the file information from cache file "
                    "'{filename}':\n{err}."
                )

    def make_dirs(self, filename):
        os.makedirs(
            os.path.dirname(filename),
            exist_ok=True
        )

    def map(
            self, func, args=None, kwargs=None, files=None, on_content=False,
            pass_info=None, read_args=None, output=None,
            max_workers=None, worker_type=None,
            return_info=False, error_to_warning=False, **find_kwargs
    ):
        """Apply a function on files of this fileset with parallel workers

        This method can use multiple workers processes / threads to boost the
        procedure significantly. Depending on which system you work, you should
        try different numbers for `max_workers`.

        Use this if you need to process the files as fast as possible without
        needing to retrieve the results immediately. Otherwise you should
        consider using :meth:`imap` in a for-loop.

        Notes:
            This method sorts the results after the starting time of the files
            unless *sort* is False.

        Args:
            func: A reference to a function that should be applied.
            args: A list/tuple with positional arguments that should be passed
                to `func`. It will be extended with the file arguments, i.e.
                a FileInfo object if `on_content` is false or - if `on_content`
                and `pass_info` are true - the read content of a file and its
                corresponding FileInfo object.
            kwargs: A dictionary with keyword arguments that should be passed
                to `func`.
            files: If you have already a list of files that you want to
                process, pass it here. The list can contain filenames or lists
                (bundles) of filenames. If this parameter is given, it is not
                allowed to set `start` and `end` then.
            on_content: If true, the file will be read before `func` will be
                applied. The content will then be passed to `func`.
            pass_info: If `on_content` is true, this decides whether also the
                `FileInfo` object of the read file should be passed to `func`.
                Default is false.
            read_args: Additional keyword arguments that will be passed
                to the reading function (see :meth:`read` for more
                information). Will be ignored if `on_content` is False.
            output: Set this to a path containing placeholders or a FileSet
                object and the return value of `func` will be copied there if
                it is not None.
            max_workers: Max. number of parallel workers to use. When
                lacking performance, you should change this number.
            worker_type: The type of the workers that will be used to
                parallelize `func`. Can be `process` or `thread`. If `func` is
                a function that needs to share a lot of data with its
                parallelized copies, you should set this to `thread`. Note that
                this may reduce the performance due to Python's Global
                Interpreter Lock (`GIL <https://stackoverflow.com/q/1294382>`).
            worker_initializer: DEPRECATED! Must be a reference to a function
                that is called once when initialising a new worker. Can be used
                to preload variables into a worker's workspace. See also
                https://docs.python.org/3.1/library/multiprocessing.html#module-multiprocessing.pool
                for more information.
            worker_initargs: DEPRECATED! A tuple with arguments for
                `worker_initializer`.
            return_info: If true, return a FileInfo object with each return
                value indicating to which file the function was applied.
            error_to_warning: Normally, if an exception is raised during
                reading of a file, this method is aborted. However, if you set
                this to *true*, only a warning is given and None is returned.
                This parameter will be ignored if `on_content=True`.
            **find_kwargs: Additional keyword arguments that are allowed
                for :meth:`find` such as `start` or `end`.

        Returns:
            A list with tuples of a FileInfo object and the return value of the
            function applied to this file. If *output* is set, the second
            element is not the return value but a boolean values indicating
            whether the return value was not None.

        Examples:

            .. code-block:: python

                ## Imaging you want to calculate some statistical values from the
                ## data of the files
                def calc_statistics(content, file_info):
                    # return the mean and maximum value
                    return content["data"].mean(), content["data"].max()

                results = fileset.map(
                    calc_statistics, start="2018-01-01", end="2018-01-02",
                    on_content=True, return_info=True,
                )

                # This will be run after processing all files...
                for file, result in results
                    print(file) # prints the FileInfo object
                    print(result) # prints the mean and maximum value

                ## If you need the results directly, you can use imap instead:
                results = fileset.imap(
                    calc_statistics, start="2018-01-01", end="2018-01-02",
                    on_content=True,
                )

                for result in results
                    # After the first file has been processed, this will be run
                    # immediately ...
                    print(result) # prints the mean and maximum value

            If you need to pass some args to the function, use the parameters
            *args* and *kwargs*:

            .. code-block:: python

                def calc_statistics(arg1, content, kwarg1=None):
                    # return the mean and maximum value
                    return content["data"].mean(), content["data"].max()

                # Note: If you do not use the start or the end parameter, all
                # files in the fileset are going to be processed:
                results = fileset.map(
                    calc_statistics, args=("value1",),
                    kwargs={"kwarg1": "value2"}, on_content=True,
                )
        """

        pool_class, pool_args, worker_args = \
            self._configure_pool_and_worker_args(
                func, args, kwargs, files, on_content, pass_info, read_args,
                output, max_workers, worker_type,
                #worker_initializer, worker_initargs,
                return_info, error_to_warning, **find_kwargs
            )

        with pool_class(**pool_args) as pool:
            # Process all found files with the arguments:
            return list(pool.map(
                self._call_map_function, worker_args,
            ))

    def imap(self, *args, **kwargs):
        """Apply a function on files and return the result immediately

        This method does exact the same as :meth:`map` but works as a generator
        and is therefore less memory space consuming.

        Args:
            *args: The same positional arguments as for :meth:`map`.
            **kwargs: The same keyword arguments as for :meth:`map`.

        Yields:
            A tuple with the FileInfo object of the processed file and the
            return value of the applied function. If `output` is set, the
            second element is not the return value but a boolean values
            indicating whether the return value was not None.

        """

        pool_class, pool_args, worker_args = \
            self._configure_pool_and_worker_args(*args, **kwargs)

        worker_queue = deque()

        with pool_class(**pool_args) as pool:
            workers = pool_args["max_workers"]
            if workers is None:
                workers = 1

            for func_args in worker_args:
                wait = len(worker_queue) >= workers

                if wait:
                    yield worker_queue.popleft().result()

                worker_queue.append(
                    pool.submit(
                        self._call_map_function,
                        func_args,
                    )
                )

            # Flush the rest:
            while worker_queue:
                yield worker_queue.popleft().result()

    def _configure_pool_and_worker_args(
            self, func, args=None, kwargs=None, files=None,
            on_content=False, pass_info=None, read_args=None, output=None,
            max_workers=None, worker_type=None,
            #worker_initializer=None, worker_initargs=None,
            return_info=False, error_to_warning=False, **find_args
    ):
        if func is None:
            raise ValueError("The parameter `func` must be given!")

        if files is not None \
                and (find_args.get("start", None) is not None
                     or find_args.get("end", None) is not None):
            raise ValueError(
                "Either `files` or `start` and `end` must be given. Not all of"
                " them!")

        # Convert the path to a FileSet object:
        if isinstance(output, str):
            output_path = output
            output = self.copy()
            output.path = output_path

        if worker_type is None:
            # If the output is directly stored to a new FileSet, it is always
            # better to use processes
            if output is None:
                worker_type = self.worker_type
            else:
                worker_type = "process"

        if worker_type == "process":
            pool_class = ProcessPoolExecutor
            if max_workers is None:
                max_workers = self.max_processes
        elif worker_type == "thread":
            pool_class = ThreadPoolExecutor
            if max_workers is None:
                max_workers = self.max_threads
        else:
            raise ValueError(f"Unknown worker type '{worker_type}!")

        pool_args = {
            "max_workers": max_workers,
            #"initializer": worker_initializer,
            #"initargs": worker_initargs,
        }

        if kwargs is None:
            kwargs = {}

        if read_args is None:
            read_args = {}

        if files is None:
            files = self.find(**find_args)

        worker_args = (
            (self, file, func, args, kwargs, pass_info, output,
             on_content, read_args, return_info, error_to_warning)
            for file in files
        )

        return pool_class, pool_args, worker_args

    @staticmethod
    def _call_map_function(all_args):
        """ This is a small wrapper function to call the function that is
        called on fileset files via .map().

        Args:
            all_args: A tuple containing following elements:
                (FileSet object, file_info, function,
                args, kwargs, output, on_content, read_args, return_info)

        Returns:
            The return value of *function* called with the arguments *args* and
            *kwargs*. This arguments have been extended by file info (and file
            content).
        """
        fileset, file_info, func, args, kwargs, pass_info, output, \
            on_content, read_args, return_info, error_to_warning = all_args

        args = [] if args is None else list(args)

        def _return(file_info, return_value):
            """Small helper for return / not return the file info object."""

            if return_info:
                return file_info, return_value
            else:
                return return_value

        if on_content:
            try:
                # file_info could be a bundle of files
                if isinstance(file_info, FileInfo):
                    file_content = fileset.read(file_info, **read_args)
                else:
                    file_content = \
                        fileset.collect(files=file_info, read_args=read_args)
                args.append(file_content)
            except Exception as e:
                if error_to_warning:
                    msg = f"[ERROR] Could not read the file(s):\n{file_info}\n"
                    msg += str(e) + "\n"
                    warnings.warn(
                        msg + "".join(traceback.format_tb(e.__traceback__)),
                        RuntimeWarning
                    )
                    return _return(file_info, None)

                raise e

        if not on_content or pass_info:
            args.append(file_info)

        # Call the function:
        return_value = func(*args, **kwargs)

        if output is None:
            # No output is needed, simply return the file info and the
            # function's return value
            return _return(file_info, return_value)

        if return_value is None:
            # We cannot write a file with the content None, hence simply return
            # the file info and False indicating that we did not write a file.
            return _return(file_info, False)

        # file_info could be a bundle of files
        if isinstance(file_info, FileInfo):
            new_filename = output.get_filename(
                file_info.times, fill=file_info.attr
            )
        else:
            start_times, end_times = zip(
                *(file.times for file in file_info)
            )
            new_filename = output.get_filename(
                (min(start_times), max(end_times)), fill=file_info[0].attr
            )

        output.write(return_value, new_filename, in_background=False)

        return _return(file_info, True)

    def match(
            self, other, start=None, end=None, max_interval=None,
            filters=None, other_filters=None):
        """Find matching files between two filesets

        Matching files are files that overlap each in their time coverage.

        Args:
            other: Another FileSet object.
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            max_interval: Maximal time interval between two overlapping files.
                If it is an integer or float, it will be interpreted as
                seconds.
            filters: The same filter argument that is allowed for
                :meth:`find`.
            other_filters: The same as `filter` but it will be applied to
                `other`.

        Yields:
            A tuple with the :class:`FileInfo` object from this fileset and a
            list with matching files from the other fileset.

        Examples:
            TODO: Add example
        """
        if max_interval is not None:
            max_interval = to_timedelta(max_interval, numbers_as="seconds")
            start = to_datetime(start) - max_interval
            end = to_datetime(end) + max_interval

        files1 = list(
            self.find(start, end, filters=filters)
        )
        files2 = list(
            other.find(start, end, filters=other_filters)
        )

        # Convert the times (datetime objects) to seconds (integer)
        times1 = np.asarray([
            file.times
            for file in files1
        ]).astype("M8[s]").astype(int).tolist()
        times2 = np.asarray([
            file.times
            for file in files2
        ]).astype("M8[s]").astype(int)

        if max_interval is not None:
            # Expand the intervals of the secondary fileset to close-in-time
            # intervals.
            times2[:, 0] -= int(max_interval.total_seconds())
            times2[:, 1] += int(max_interval.total_seconds())

        # Search for all overlapping intervals:
        tree = IntervalTree(times2)
        results = tree.query(times1)

        for i, overlapping_files in enumerate(results):
            matches = [files2[oi] for oi in sorted(overlapping_files)]
            if matches:
                yield files1[i], matches

    def move(
            self, target=None, convert=None, copy=False, **kwargs,
    ):
        """Move (or copy) files from this fileset to another location

        Args:
            target: Either a FileSet object or the path (containing
                placeholders) where the files should be moved to.
            convert: If true, the files will be read by the old fileset's file
                handler and written to their new location by using the new file
                handler from `target`. Both file handlers must be compatible,
                i.e. the object that the old file handler's read method returns
                must handable for the new file handler's write method. You
                can also set this to a function that converts the return value
                of the read method into something else before it will be passed
                to the write method. Default is false, i.e. the file will be
                simply moved without converting.
            copy: If true, then the original files will be copied instead of
                moved.
            **kwargs: Additional keyword arguments that are allowed
                for :meth:`find` such as `start`, `end` or `files`.

        Returns:
            New FileSet object with the new files.

        Examples:

        .. code-block:: python

            ## Copy all files between two dates to another location

            old_fileset = FileSet(
                "old/path/{year}/{month}/{day}/{hour}{minute}{second}.nc",
            )

            # New fileset with other path
            new_fileset = FileSet(
                "new/path/{year}/{doy}/{hour}{minute}{second}.nc",
            )

            old_fileset.move(
                new_fileset, start="2017-09-15", end="2017-09-23",
            )

        .. code-block:: python

            ## Copy all files between two dates to another location and convert
            ## them to a different format

            from typhon.files import CSV, NetCDF4

            old_fileset = FileSet(
                "old/path/{year}/{month}/{day}/{hour}{minute}{second}.nc"
            )
            new_fileset = FileSet(
                "new/path/{year}/{doy}/{hour}{minute}{second}.csv"
            )

            # Note that this only works if both file handlers are compatible
            new_fileset = old_fileset.move(
                new_fileset, convert=True, start="2017-09-15",
                end="2017-09-23",
            )

            # It is also possible to set the new path directly:
            new_fileset = old_fileset.move(
                "new/path/{year}/{doy}/{hour}{minute}{second}.csv",
                convert=True, start="2017-09-15", end="2017-09-23",
            )
        """

        # Convert the path to a FileSet object:
        if not isinstance(target, FileSet):
            destination = self.copy()
            destination.path = target
        else:
            destination = target

        if convert is None:
            convert = False

        if self.single_file:
            file_info = self.get_info(self.path)

            FileSet._move_single_file(
                file_info, self, destination, convert, copy
            )
        else:
            if destination.single_file:
                raise ValueError(
                    "Cannot move files from multi-file to single-file set!")

            move_args = {
                "fileset": self,
                "destination": destination,
                "convert": convert,
                "copy": copy
            }

            # Copy the files
            self.map(FileSet._move_single_file, kwargs=move_args, **kwargs)

        return destination

    @staticmethod
    def _move_single_file(
            file_info, fileset, destination, convert, copy):
        """This is a small wrapper function for moving files. It is better to
        use :meth:`FileSet.move` directly.

        Args:
            fileset:
            file_info: FileInfo object of the file that should be to copied.
            destination:
            convert:
            copy:

        Returns:
            None
        """

        # Generate the new file name
        new_filename = destination.get_filename(
            file_info.times, fill=file_info.attr
        )

        # Shall we simply move or even convert the files?
        if convert:
            # Read the file with the current file handler
            data = fileset.read(file_info)

            # Maybe the user has given us a converting function?
            if callable(convert):
                data = convert(data)

            # Store the data of the file with the new file handler
            destination.write(data, new_filename)

            if not copy:
                os.remove(file_info.path)
        else:
            # Create the new directory if necessary.
            os.makedirs(os.path.dirname(new_filename), exist_ok=True)

            if copy:
                shutil.copy(file_info.path, new_filename)
            else:
                shutil.move(file_info.path, new_filename)

    @property
    def name(self):
        """Get or set the fileset's name.

        Returns:
            A string with the fileset's name.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            value = str(id(self))

        self._name = value

    def _to_datetime_args(self, placeholder):
        """Get datetime args from placeholders for start and end date.

        Args:
            placeholder: A dictionary containing time placeholders.

        Returns:
            A tuple of two dictionaries
        """
        start_args = {
            p: int(value)
            for p, value in placeholder.items()
            if not p.startswith("end_") and p in self._time_placeholder
        }

        end_args = {
            p[len("end_"):]: int(value)
            for p, value in placeholder.items()
            if p.startswith("end_") and p in self._time_placeholder
        }

        start_datetime_args = self._standardise_datetime_args(start_args)

        # If temporal placeholders are set, we need at least year, month and day
        if (start_datetime_args
            and not {'year', 'month', 'day'} & set(start_datetime_args.keys())):
            raise ValueError('Cannot use temporal placeholders if year, month '
                             'and day are not set.')

        end_datetime_args = self._standardise_datetime_args(end_args)

        return start_datetime_args, end_datetime_args

    def _standardise_datetime_args(self, args):
        """Replace some placeholders to datetime-conform placeholder.

        Args:
            args: A dictionary of placeholders.

        Returns:
            The standardised placeholder dictionary.
        """
        year2 = args.pop("year2", None)
        if year2 is not None:
            if year2 < self.year2_threshold:
                args["year"] = 2000 + year2
            else:
                args["year"] = 1900 + year2

        # There is only microseconds as parameter for datetime for sub-seconds.
        # So if one of them is set, we replace them and setting microsecond
        # instead.
        if {'decisecond', 'centisecond', 'millisecond', 'microsecond'} & set(args.keys()): # noqa
            args["microsecond"] = \
                100000*args.pop("decisecond", 0) \
                + 10000*args.pop("centisecond", 0) \
                + 1000*args.pop("millisecond", 0) \
                + args.pop("microsecond", 0)

        doy = args.pop("doy", None)
        if doy is not None:
            date = datetime(args["year"], 1, 1) + timedelta(doy - 1)
            args["month"] = date.month
            args["day"] = date.day

        return args

    def parse_filename(self, filename, template=None,):
        """Parse the filename with temporal and additional regular expressions.

        This method uses the standard temporal placeholders which might be
        overwritten by the user-defined placeholders.

        Args:
            filename: Path and name of the file.
            template: Template with regex/placeholders that should be used.
                Default is *FileSet.path*.

        Returns:
            A dictionary with filled placeholders.
        """

        if template is None:
            regex = re.compile(self._filled_path)
        else:
            if isinstance(template, str):
                regex = self._fill_placeholders(template, compile=True)
            else:
                regex = template

        results = regex.match(filename)

        if not results:
            raise ValueError(
                "Could not parse the filename; it does not match the given "
                "template.")
        else:
            return results.groupdict()

    @property
    def path(self):
        """Gets or sets the path to the fileset's files.

        Returns:
            A string with the path (can contain placeholders or wildcards.)
        """

        # We need always the absolute path:
        return os.path.abspath(self._path)

    @path.setter
    def path(self, value):
        if value is None:
            raise ValueError("The path parameter cannot be None!")

        self._path = value

        # The path consists of three parts: the base directory, the sub
        # directory and the filename. The sub directory and filename may
        # contain regex/placeholder, the base directory not. We need to split
        # the path into these three parts to enable file finding.
        directory = os.path.dirname(self.path)
        index_of_sub_directory = \
            next(
                (i for i, ch in enumerate(directory)
                 if ch in self._special_chars), None
            )

        if index_of_sub_directory is None:
            # There is no sub directory
            self._base_dir = directory
        else:
            self._base_dir = directory[:index_of_sub_directory]
            self._sub_dir = directory[index_of_sub_directory:]

            # Later, we iterate over all possible sub directories and find
            # those that match the regex / placeholders. Hence, we split the
            # sub directory into chunks for each hierarchy level:
            self._sub_dir_chunks = self._sub_dir.split(os.path.sep)

            # The sub directory time resolution is needed for find_closest:
            self._sub_dir_time_resolution = self._get_time_resolution(
                self._sub_dir
            )[1]

        # Retrieve the used placeholder names from the path:
        self._path_placeholders = set(re.findall(r"{(\w+)}", self.path))

        # Set additional user-defined placeholders to default values (
        # non-greedy wildcards).
        self.set_placeholders(**{
            p: ".+?"
            for p in self._path_placeholders.difference(self._time_placeholder)
        })

        # Get all temporal placeholders from the path (for ending time):
        end_time_placeholders = {
            p[len("end_"):] for p in self._path_placeholders
            if p.startswith("end") and p in self._time_placeholder
        }

        # If the end time retrieved from the path is younger than the start
        # time, the end time will be incremented by this value:
        self._end_time_superior = \
            self._get_superior_time_resolution(end_time_placeholders)

        # Flag whether this is a single file fileset or not. We simply check
        # whether the path contains special characters:
        self.single_file = not any(
            True for ch in self.path
            if ch in self._special_chars
        )

        self._path_extension = os.path.splitext(self.path)[0].lstrip(".")

    @staticmethod
    def _get_superior_time_resolution(placeholders, ):
        """Get the superior time resolution of all placeholders.

        Examples:
            The superior time resolution of seconds are minutes, of hours are
            days, etc.

        Args:
            placeholders: A list or dictionary with placeholders.

        Returns:
            A pandas compatible frequency string or None if the superior time
            resolution is higher than a year.
        """
        # All placeholders from which we know the resolution:
        placeholders = set(placeholders).intersection(
            FileSet._temporal_resolution
        )

        if not placeholders:
            return None

        highest_resolution = max(
            (FileSet._temporal_resolution[tp] for tp in placeholders),
        )

        highest_resolution_index = list(
            FileSet._temporal_resolution.values()).index(highest_resolution)

        if highest_resolution_index == 0:
            return None

        resolutions = list(FileSet._temporal_resolution.values())
        superior_resolution = resolutions[highest_resolution_index - 1]

        return pd.Timedelta(superior_resolution).to_pytimedelta()

    @staticmethod
    def _get_time_resolution(path_or_dict, highest=True):
        """Get the lowest/highest time resolution of all placeholders

        Seconds have a higher time resolution than minutes, etc. If our path
        contains seconds, minutes and hours, this will return a timedelta
        object representing 1 second if *highest* is True otherwise 1 hour.

        Args:
            path_or_dict: A path or dictionary with placeholders.
            highest: If true, search for the highest time resolution instead of
                the lowest.

        Returns:
            The placeholder name with the lowest / highest resolution and
            the representing timedelta object.
        """
        if isinstance(path_or_dict, str):
            placeholders = set(re.findall(r"{(\w+)}", path_or_dict))
            if "doy" in placeholders:
                placeholders.remove("doy")
                placeholders.add("day")
            if "year2" in placeholders:
                placeholders.remove("year2")
                placeholders.add("year")
        else:
            placeholders = set(path_or_dict.keys())

        # All placeholders from which we know the resolution:
        placeholders = set(placeholders).intersection(
            FileSet._temporal_resolution
        )

        if not placeholders:
            # There are no placeholders in the path, therefore we return the
            # highest time resolution automatically
            return "year", FileSet._temporal_resolution["year"]

        # E.g. if we want to find the temporal placeholder with the lowest
        # resolution, we have to search for the maximum of their values because
        # they are represented as timedelta objects, i.e. month > day > hour,
        # etc. expect
        if highest:
            placeholder = min(
                placeholders, key=lambda k: FileSet._temporal_resolution[k]
            )
        else:
            placeholder = max(
                placeholders, key=lambda k: FileSet._temporal_resolution[k]
            )

        return placeholder, FileSet._temporal_resolution[placeholder]

    def _fill_placeholders(self, path, extra_placeholder=None, compile=False):
        """Fill all placeholders in a path with its RegExes and compile it.

        Args:
            path:
            extra_placeholder:
            compile: Compile as regex before return
        Returns:

        """
        if extra_placeholder is None:
            extra_placeholder = {}

        placeholder = {
            **self._time_placeholder,
            **self._user_placeholder,
            **extra_placeholder,
        }

        # Mask all dots and convert the asterisk to regular expression syntax:
        path = path.replace("\\", "\\\\").replace(".", r"\.").replace("*", ".*?")

        # Python's standard regex module (re) cannot handle multiple groups
        # with the same name. Hence, we need to cover duplicated placeholders
        # so that only the first of them does group capturing.
        path_placeholders = re.findall(r"{(\w+)}", path)
        duplicated_placeholders = {
            p: self._remove_group_capturing(p, placeholder[p])
            for p in path_placeholders if path_placeholders.count(p) > 1
        }

        if duplicated_placeholders:
            for p, v in duplicated_placeholders.items():
                split_index = path.index("{"+p+"}") + len(p) + 2

                # The value of the placeholder might contain a { or } as regex.
                # We have to escape them because we use the formatting function
                # later.
                v = v.replace("{", "{{").replace("}", "}}")

                changed_part = path[split_index:].replace("{" + p + "}", v)
                path = path[:split_index] + changed_part
        try:
            # Prepare the regex for the template, convert it to an exact match:
            regex_string = "^" + path.format(**placeholder) + "$"
        except KeyError as err:
            raise UnknownPlaceholderError(self.name, err.args[0])
        except ValueError as err:
            raise PlaceholderRegexError(self.name, str(err))

        if compile:
            return re.compile(regex_string)
        else:
            return regex_string

    @staticmethod
    def _complete_placeholders_regex(placeholder):
        """Complete placeholders' regexes to capture groups.

        Args:
            placeholder: A dictionary of placeholders and their matching
            regular expressions

        Returns:

        """

        return {
            name: FileSet._add_group_capturing(name, value)
            for name, value in placeholder.items()
        }

    @staticmethod
    def _add_group_capturing(placeholder, value):
        """Complete placeholder's regex to capture groups.

        Args:
            placeholder: A dictionary of placeholders and their matching
            regular expressions

        Returns:

        """
        if value is None:
            return None
        elif isinstance(value, (tuple, list)):
            return f"(?P<{placeholder}>{'|'.join(value)})"
        else:
            return f"(?P<{placeholder}>{value})"

    @staticmethod
    def _remove_group_capturing(placeholder, value):
        if f"(?P<{placeholder}>" not in value:
            return value
        else:
            # The last character is the closing parenthesis:
            return value[len(f"(?P<{placeholder}>"):-1]

    @expects_file_info()
    def read(self, file_info, **read_args):
        """Open and read a file

        Notes:
            You need to specify a file handler for this fileset before you
            can use this method.

        Args:
            file_info: A string, path-alike object or a
                :class:`~typhon.files.handlers.common.FileInfo` object.
            **read_args: Additional key word arguments for the
                *read* method of the used file handler class.

        Returns:
            The content of the read file.
        """
        if self.handler is None:
            raise NoHandlerError(f"Could not read '{file_info.path}'!")

        read_args = {**self.read_args, **read_args}

        if self.decompress:
            with typhon.files.decompress(file_info.path, tmpdir=self.temp_dir)\
                    as decompressed_path:
                decompressed_file = file_info.copy()
                decompressed_file.path = decompressed_path
                data = self.handler.read(decompressed_file, **read_args)
        else:
            data = self.handler.read(file_info, **read_args)

        # Maybe the user wants to do some post-processing?
        if self.post_reader is not None:
            data = self.post_reader(file_info, data)

        return data

    def _retrieve_time_coverage(self, filled_placeholder,):
        """Retrieve the time coverage from a dictionary of placeholders.

        Args:
            filled_placeholder: A dictionary with placeholders and their
                fillings.

        Returns:
            A tuple of two datetime objects.
        """
        if not filled_placeholder:
            return None

        start_args, end_args = self._to_datetime_args(filled_placeholder)

        if start_args:
            start_date = datetime(**start_args)
        else:
            start_date = None

        if end_args:
            end_args = {**start_args, **end_args}
            end_date = datetime(**end_args)

            # Sometimes the filename does not explicitly provide the complete
            # end date. Imagine there is only hour and minute given, then day
            # change would not be noticed. Therefore, make sure that the end
            # date is always bigger (later) than the start date.
            if end_date < start_date:
                end_date += self._end_time_superior
        else:
            end_date = None

        return start_date, end_date

    def reset_cache(self):
        """Reset the information cache

        Returns:
            None
        """
        self.info_cache = {}

    def save_cache(self, filename):
        """Save the information cache to a JSON file

        Returns:
            None
        """
        if filename is not None:
            # First write all to a backup file. If something happens, only the
            # backup file will be overwritten.
            with open(filename+".backup", 'w') as file:
                # We cannot save datetime objects with json directly. We have
                # to convert them to strings first:
                info_cache = [
                    info.to_json_dict()
                    for info in self.info_cache.values()
                ]
                json.dump(info_cache, file)

            # Then rename the backup file
            shutil.move(filename+".backup", filename)

    def get_placeholders(self):
        """Get placeholders for this FileSet.

        Returns:
            A dictionary of placeholder names set by the user (i.e. excluding
            the temporal placeholders) and their regexes.
        """
        return self._user_placeholder.copy()

    def set_placeholders(self, **placeholders):
        """Set placeholders for this FileSet.

        Args:
            **placeholders: Placeholders as keyword arguments.

        Returns:
            None
        """

        self._user_placeholder.update(
            self._complete_placeholders_regex(placeholders)
        )

        # Update the path regex (uses automatically the user-defined
        # placeholders):
        self._filled_path = self._fill_placeholders(self.path)

    @property
    def time_coverage(self):
        """Get and set the time coverage of the files of this fileset

        Setting the time coverage after initialisation resets the info cache of
        the fileset object.

        Returns:
            The time coverage of the whole fileset (if it is a single file) as
            tuple of datetime objects or (if it is a multi file fileset) the
            fixed time duration of each file as timedelta object.

        """
        return self._time_coverage

    @time_coverage.setter
    def time_coverage(self, value):
        if self.single_file:
            if value is None:
                # The default for single file filesets:
                self._time_coverage = [
                    datetime.min,
                    datetime.max
                ]
            else:
                self._time_coverage = [
                    to_datetime(value[0]),
                    to_datetime(value[1]),
                ]
        elif value is not None:
            self._time_coverage = to_timedelta(value)
        else:
            self._time_coverage = None

        # Reset the info cache because some file information may have changed
        # now
        self.info_cache = {}


    def to_dataframe(self, include_times=False, **kwargs):
        """Create a pandas.Dataframe from this FileSet

        This method creates a pandas.DataFrame containing all the filenames in this
        FileSet as row indices and the placeholders as columns.

        Args:
            include_times: If True, also the start and end time of each file are
                included. Default: False.
            **kwargs: Additional keyword arguments which are allowed for
                :meth:`~typhon.files.filset.FileSet.find`.

        Returns:
            A pandas.DataFrame with the filenames as row indices and the
            placeholders as columns.

        Examples:
        .. code-block:: python

            # Example directory:
            # dir/
            #   Satellite-A/
            #       20190101-20190201.nc
            #       20190201-20190301.nc
            #   Satellite-B/
            #       20190101-20190201.nc
            #       20190201-20190301.nc

            from typhon.files import FileSet

            files = FileSet(
                'dir/{satellite}/{year}{month}{day}'
                '-{end_year}{end_month}{end_day}.nc'
            )
            df = files.to_dataframe()

            # Content of df:
            # 	                                    satellite
            # /dir/Satellite-B/20190101-20190201.nc Satellite-B
            # /dir/Satellite-A/20190101-20190201.nc Satellite-A
            # /dir/Satellite-B/20190201-20190301.nc Satellite-B
            # /dir/Satellite-A/20190201-20190301.nc Satellite-A
        """
        if include_times:
            data = [
                (file.path, *file.times, *file.attr.values())
                for file in self.find(**kwargs)
            ]
            columns = ['start_time', 'end_time']
        else:
            data = [
                (file.path, *file.attr.values())
                for file in self.find(**kwargs)
            ]
            columns = []
        df = pd.DataFrame(data).set_index(0)
        columns += list(self.get_placeholders().keys())
        df.columns = columns
        del df.index.name
        return df

    def write(self, data, file_info, in_background=False, **write_args):
        """Write content to a file by using the FileSet's file handler.

        If the filename extension is a compression format (such as *zip*,
        etc.) and *FileSet.compress* is set to true, the file will be
        compressed afterwards.

        Notes:
            You need to specify a file handler for this fileset before you
            can use this method.

        Args:
            data: An object that can be stored by the used file handler class.
            file_info: A string, path-alike object or a
                :class:`~typhon.files.handlers.common.FileInfo` object.
            in_background: If true (default), this runs the writing process in
                a background thread so it does not pause the main process.
            **write_args: Additional key word arguments for the *write* method
                of the used file handler object.

        Returns:
            None

        Examples:

        .. code-block:: python

            import matplotlib.pyplot as plt
            from typhon.files import FileSet, Plotter

            # Define a fileset consisting of multiple files:
            plots = FileSet(
                path="/dir/{year}/{month}/{day}/{hour}{minute}{second}.png",
                handler=Plotter,
            )

            # Let's create a plot example
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            ax.set_title("Data from 2018-01-01")

            ## To save the plot as a file of the fileset, you have two options:
            # Use this simple expression:
            plots["2018-01-01"] = fig

            # OR use write in combination with get_filename
            filename = plots.get_filename("2018-01-01")
            plots.write(fig, filename)

            # Hint: If saving the plot takes a lot of time but you want to
            # continue with the program in the meanwhile, you can use the
            # *in_background* option. This saves the plot in a background
            # thread.
            plots.write(fig, filename, in_background=True)

            # continue with other stuff immediately and do not wait until the
            # plot is saved...
            do_other_stuff(...)

        """
        if isinstance(file_info, str):
            file_info = FileInfo(file_info)

        if self.handler is None:
            raise NoHandlerError(
                f"Could not write data to '{file_info.path}'!"
            )

        if in_background:
            warnings.warn("in_background option is deprecated!")

            # Run this function again but as a background thread in a queue:
            #threading.Thread(
            #    target=FileSet.write, args=(self, data, file_info),
            #    kwargs=write_args.update(in_background=False),
            #).start()
            #return

        write_args = {**self.write_args, **write_args}

        # The users should not be bothered with creating directories by
        # themselves.
        self.make_dirs(file_info.path)

        if self.compress:
            with typhon.files.compress(file_info.path, tmpdir=self.temp_dir) \
                    as compressed_path:
                compressed_file = file_info.copy()
                compressed_file.path = compressed_path
                self.handler.write(data, compressed_file, **write_args)
        else:
            self.handler.write(data, file_info, **write_args)


class FileSetManager(dict):
    def __init__(self, *args, **kwargs):
        """Simple container for multiple FileSet objects.

        You can use it as a native dictionary.

        More functionality will be added in future.

        Example:

        .. code-block:: python

            filesets = FileSetManager()

            filesets += FileSet(
                name="images",
                files="path/to/files.png",
            )

            # do something with it
            for name, fileset in filesets.items():
                fileset.find(...)

        """
        super(FileSetManager, self).__init__(*args, **kwargs)

    def __iadd__(self, fileset):
        if fileset.name in self:
            warnings.warn(
                "FileSetManager: Overwrite fileset with name '%s'!"
                % fileset.name, RuntimeWarning)

        self[fileset.name] = fileset
        return self
