from collections import defaultdict
import datetime
import glob
import json
from multiprocessing import Pool
import os.path
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import typhon.datasets
from .accumulated_data import *
import typhon.files
import typhon.plots
import xarray

__all__ = [
    'Dataset',
    'AccumulatedData',
]


class Dataset:
    placeholder = {
        # "placeholder_name" : [regex to find the placeholder]
        "year" : "(\d{4})",
        "year2" : "(\d{2})",
        "month" : "(\d{2})",
        "day" : "(\d{2})",
        "doy" : "(\d{3})",
        "hour" : "(\d{2})",
        "minute" : "(\d{2})",
        "second" : "(\d{2})",
        "millisecond": "(\d{3})",
    }

    def __init__(
            self, name,
            files, file_handler,
            time_coverage_retrieving_method="filename",
            cache_time_coverages=True, times_cache_filename=None
    ):
        """ Class which provides methods to handle a set of multiple files (dataset).

        Args:
            name: The name of the dataset.
            files: A string with the complete path to the dataset files. The string can contain placeholder such as
                {year}, {month}, etc. See here for a complete list (TODO: insert link to placeholder documentation).
                For example, the Dataset class finds with the argument
                files="/path/to/files/{year}/{month}/{day}/{hour}{minute}{second}.nc.gz" files such as
                "/path/to/files/2016/11/12/051422.nc.gz"
            file_handler: An object which can handle the dataset files. This dataset class does not care which format
                its files have when this file handler object is given. You can use a file handler class from
                typhon.handlers or write your own class. For example, if this dataset consists of NetCDF files, you can
                use typhon.spareice.handlers.common.NetCDF4File() here.
            time_coverage_retrieving_method: Defines how the timestamp of the data should be retrieved. Default is
                "filename". Look at Dataset.retrieve_timestamp() for more details.
            times_caching: Finding the correct files for a period may take a while, especially when the time
                retrieving method is set to "content". Therefore, if this flag is true, the file names and their time
                coverage is cached and multiple calls of find_files (for time coverages that are close) are faster.
            times_cache_filename: Specify a name to a file here (which need not exist) if you wish to save those time
                coverages to a file. When restarting your script, this cache is used.

        Examples:
            import typhon.spareice.datasets as datasets
            from typhon.spareice.handlers import common

            dataset = datasets.Dataset(
                name="TestData",
                files="/path/to/daily/files/{year}/{month}/{day}/*.nc",
                file_handler=common.NetCDF4File(),
                # If the time coverage of the data cannot be retrieved from the filename, you should
                # set this to "content":
                timestamp_retrieving_method="filename"
            )
        """

        # Initialize member variables:
        self.file_handler = file_handler
        self.files = files
        self.file_extension = os.path.splitext(files)[-1]

        # Multiple calls of .find_files() can be very slow when using a time coverage retrieving method "content".
        # Hence, we use a cache to store the names and time coverages of already touched files in this dictionary. The
        # keys are datetime objects
        if cache_time_coverages:
            self.time_coverages_cache = defaultdict(list)
            if times_cache_filename is not None and os.path.exists(times_cache_filename):
                print("Load time coverages from cache file.")
                with open(times_cache_filename) as file:
                    self.time_coverages_cache.update(json.load(file))

        self.cache_time_coverages = cache_time_coverages
        self.times_cache_filename = times_cache_filename

        self.name = name
        self.time_coverage_retrieving_method = time_coverage_retrieving_method

    def __del__(self):
        """ Desconstructor: Saves time coverages cache to a file if necessary.

        Returns:
            None
        """
        if self.cache_time_coverages \
                and self.times_cache_filename is not None \
                and os.path.exists(self.times_cache_filename):
            print("Save time coverages to cache file.")
            with open(self.times_cache_filename) as file:
                json.dump(self.time_coverages_cache, file)

    """def __iter__(self):
        return self

    def __next__(self):
        # We split the path of the input files after the first appearance of {day} or {doy}.
        path_parts = re.split(r'({\w+})', self.files)

        for dir in self._find_subdirs(path_parts[0])
            print(path_parts)

            yield file
    """

    def __str__(self):
        info = "Dataset:\t" + self.name
        info += "\nFiles:\t" + self.files
        return info

    def accumulate(self, start, end, fields=None, chunk_size=100, daily_bundle=False, gridded=False):
        """ Accumulate all data between two dates in one xarray.Dataset.

        Args:
            start: Starting date as datetime.datetime object.
            end: Ending date as datetime.datetime object.
            fields: List or tuple of variables to select.
            chunk_size: The maximum size of the accumulated data array before it gets split into chunks.
            daily_bundle: If true, this function returns one xarray.Dataset() per day. If the chunk size is exceeded
                during accumulating the data of one day, one day will be split into multiple chunks.
            gridded: If true, the data will be gridded along the lat, lon and time axis.

        Yields:
            xarray.Dataset() objects with the maximum chunk size.

        Examples:
            dataset = typhon.dataset.Dataset(
                files="path/to/files.nc", file_handler=typhon.handlers.common.NetCDFFile()
            )
            data = dataset.accumulate(
                datetime.datetime(2016, 1, 1), datetime.datetime(2016, 2, 1), fields=("temperature"))

            # do something with data["temperature"]
            data["temperature"].plot()
            ...
        """

        data = None

        for filename, _ in sorted(list(self.find_files(start, end)), key=lambda x: x[1]):
            file_data = self.read(filename, fields)
            if data is None:
                data = file_data
            else:
                data = xarray.concat([data, file_data], dim="time")

        data.attrs["name"] = self.name

        return AccumulatedData.from_xarray(data)

    @staticmethod
    def _call_map_function(args):
        """ This is a small wrapper function to call the function that is called on dataset files via .map().

        Args:
            args: A tuple containing following elements:
                (function_name, Dataset object, *args,  **kwargs)

        Returns:
            The return value of function_name called with the arguments in args.
        """

        if args[1] is None:
            if args[3] is None:
                return args[0](*args[2])
            else:
                return args[0](*args[2], **args[3])
        else:
            if args[3] is None:
                return args[0](args[1], *args[2])
            else:
                return args[0](args[1], *args[2], **args[3])

    def find_files(self, start, end, bundle_one_day=False):
        """ Finds all files between the given start and end date.

        This method calculates the days between the start and end date. It uses those days to loop over the dataset
        files. This is a generator method.

        Args:
            start: Start date as datetime.datetime object with year, month and day. Hours, minutes and seconds are
                optional.
            end: End date. Same format as "start".
            bunde_one_day: If this is set to true, all files of one day will be returned at once.

        Yields:
            If bunde_one_day is set to true, this yields one tuple of file names and timestamps for each day.
            Otherwise it will yield a tuple of one file name and timestamp only.
        """

        # We We split the path of the input files after the first appearance of {day} or {doy}.
        path_parts = re.split(r'({day}|{doy})', self.files)

        if bundle_one_day:
            day_bundle = []

        # At first, get all days between start and end date
        for date in pd.date_range(start, end):

            # Generate the daily path string for the files
            daily_path = self.generate_filename_from_time(''.join(path_parts[:2]), date)
            if os.path.isdir(daily_path):
                daily_path += "/"

            # Find all files in that daily path
            for filename in glob.iglob(daily_path + "*", recursive=True):

                # Test all files whether they are between the start and end date.
                if filename.endswith(self.file_extension):
                    file_start, file_end = self.retrieve_time_coverage(filename)

                    if file_start < end and file_end > start:
                        if bundle_one_day:
                            day_bundle.append([filename, file_start, file_end])
                        else:
                            yield filename, file_start, file_end

            if bundle_one_day:
                yield day_bundle.copy()
                day_bundle.clear()

    def find_file(self, timestamp):
        """ This method still does not work correctly!

        Args:
            timestamp:

        Returns:

        """
        raise NotImplementedError("Dataset.find_file(): This method still does not work correctly!")

        filename = self.generate_filename_from_time(self.files, timestamp)

        if os.path.exists(filename):
            return filename

        # If this file does not exist, we have to find the best file (by checking the time).
        best_file = None
        template = self.files
        while template:
            last_placeholder_index = template.rfind("{")
            if last_placeholder_index == -1:
                return None
            template = template[:last_placeholder_index]

            # Find all files in that path
            searching_path = "{}*.{}".format(template, self.file_extension)
            for filename in glob.iglob(self.generate_filename_from_time(searching_path, timestamp)):
                if best_file is None:
                    time_difference = self.retrieve_time_coverage(filename) - timestamp
                    best_file = (filename, abs(time_difference.total_seconds()))
                else:
                    time_difference = self.retrieve_time_coverage(filename) - timestamp
                    if abs(time_difference.total_seconds) < best_file[1]:
                        best_file = (filename, abs(time_difference.total_seconds()))

            if best_file is not None:
                return best_file

        return None

    def retrieve_all_time_coverages(self, path, file_extension=None, retrieve_method=None):
        """

        Args:
            path:
            file_extension:

        Returns:

        """
        if self.cache_time_coverages and path in self.time_coverages_cache:
            yield from self.time_coverages_cache[path]



    def retrieve_time_coverage(self, filename, retrieve_method=None):
        """ Retrieves the timestamp from a given file. This is not the real timestamp of the file itself but
        of its content.

        TODO: This function cannot retrieve the ending of the time coverage by using the filename so far. When using the
        retrieve method "filename", the same timestamp is returned twice.

        Args:
            filename: Name of the file.
            retrieve_method: Defines how the timestamp of a dataset file (not the real timestamp of the file but
                of the content) should be retrieved. There are two options:
                    "filename" (default): The timestamp is parsed from the filename by using the placeholders. This is
                        fast but could lead to errors due to ambiguity. This only works if placeholders have been used
                        in the "files" argument of the Dataset object.
                    "content": The file is opened and its content is checked to retrieve the timestamp. This could be
                        time-intensive but is more reliable.

        Returns:
            A tuple of two datetime objects, indicating the time coverage.
        """

        if self.cache_time_coverages and filename in self.time_coverages_cache:
            return self.time_coverages_cache[filename]

        if retrieve_method is None:
            retrieve_method = self.time_coverage_retrieving_method

        time_coverage = None

        if retrieve_method == "filename":
            pattern = self.files.format(**Dataset.placeholder)
            pattern = pattern.replace("*", ".+?")
            try:
                values = re.findall(pattern, filename)
                values = values[0]
                keys = re.findall("\{(\w+)\}", self.files)
            except:
                raise ValueError("The filename does not match the given template (via files). I could not "
                                 "retrieve the timestamp.")

            date_args = {}
            for index, key in enumerate(keys):
                value = int(values[index])
                if key == "year2":
                    # TODO: What should be the threshold that decides whether the year is 19xx or 20xx?
                    if value < 65:
                        date_args["year"] = 2000 + value
                    else:
                        date_args["year"] = 1900 + value
                elif key == "millisecond":
                    date_args["microsecond"] = value * 1000
                else:
                    date_args[key] = value

            if "doy" in keys:
                date = datetime.datetime(date_args["year"], 1, 1) + datetime.timedelta(date_args["doy"] - 1)
                date_args["month"] = date.month
                date_args["day"] = date.day
                del date_args["doy"]

            # Actually we do not know the time coverage by looking at the filename, therefore give back the same
            # timestamp for start and end timestamp. TODO: Change this to the real time coverage.
            time_coverage = (datetime.datetime(**date_args), datetime.datetime(**date_args))
        else:
            time_coverage = self.get_info(filename)["times"]

        if self.cache_time_coverages:
            self.time_coverages_cache[filename] = time_coverage

        return time_coverage

    @staticmethod
    def generate_filename_from_time(template, start_time, end_time=None):
        """ Generates the file name for a specific date by using the template argument.

            Allowed placeholders in the template are:
            year - Four digits indicating the year, e.g. 1999
            year2 - Two digits indicating the year, e.g. 99 (=1999).
            month - Two digits indicating the month, e.g. 09 (september).
            day - Two digits indicating the day, e.g. 09.
            doy - Three digits indicating the day of the year, e.g. 001.
            hour - Two digits indicating the hour, e.g. 12.
            minute - Two digits indicating the minute, e.g. 09.
            second - Two digits indicating the second, e.g. 59.
            millisecond - Three digits indicating the millsecond, e.g. 439.

            Args:
                template: A string with format placeholders such as {year} or {day}.
                timestamp: A datetime object with the needed date and time.

            Returns:
                A string containing the full path and name of the file.

            Example:
                >>> Dataset.generate_filename_from_time("{year2}/{month}/{day}.dat", datetime.datetime(2016, 1, 1))
                16/01/01.dat

            """

        # Fill all placeholders variables with values
        template = template.format(
            year=start_time.year, year2=str(start_time.year)[-2:],
            month="{:02d}".format(start_time.month), day="{:02d}".format(start_time.day),
            doy="{:03d}".format((start_time - datetime.datetime(start_time.year, 1, 1)).days + 1),
            hour="{:02d}".format(start_time.hour), minute="{:02d}".format(start_time.minute), second="{:02d}".format(start_time.second)
        )

        return template

    # def load_meta_data(self, filename, file_handler):
    #     """ Loads the meta data of the datasets from a file.
    #
    #     Args:
    #         filename:
    #         file_handler:
    #
    #     Returns:
    #         None
    #     """
    #     fields = ["timestamp", "position latitude", "position longitude"]
    #     meta_data = file_handler().read(filename, fields)
    #     self.meta_data = {
    #         "times": None,
    #         "lat": None,
    #         "lon": None,
    #     }
    #     self.meta_data["times"], self.meta_data["lat"], self.meta_data["lon"] = zip(*meta_data)

    def map(self, start, end, func, kwargs=None, processes=12):
        """ Processes a function on all files between start and end date on parallel processes.

        This method can use multiple processes to boost the procedure significantly. Depending on which system
        you work, you should try different numbers for processes.

        Args:
            start: Start date as datetime object.
            end: End date. Same format as "start".
            func: Function that should be applied on the data of each file.
            kwargs: A dictionary of additional keyword arguments that should be given over to the function in func.
            processes: Max. number of parallel processes to use. When lacking performance, you should change this
                number.

        Returns:
            A list of the return values of the applied function.
        """
        print("Process all files from %s to %s.\nThis may take a while..." % (start, end))

        # Measure the time for profiling.
        start_time = time.time()

        # Create a pool of processes and process all the files with them.
        pool = Pool(processes=processes)
        results = pool.map(Dataset._call_map_function,
                           [(func, self, x, kwargs) for x in
                            self.find_files(start, end)])

        if results:
            print("It took %.2f seconds using %d parallel processes to process %d files." % (
                time.time() - start_time, processes, len(results)))
        else:
            print("Warning: No files found!")

        return results

    def get_info(self, filename):
        with typhon.files.decompress(filename) as file:
            return self.file_handler.get_info(file)

    def read(self, filename, fields="all"):
        with typhon.files.decompress(filename) as file:
            data = self.file_handler.read(file, fields)
            data.attrs["name"] = self.name
            return data

    def write(self, filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return self.file_handler.write(filename, data)