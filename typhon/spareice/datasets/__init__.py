import datetime
import glob
from multiprocessing import Pool
import os.path
import re
import time

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import typhon.datasets
import typhon.files
import xarray

__all__ = [
    'Dataset'
]


class Dataset:
    """ Simple temporary dataset wrapper. Should be unnecessary when the base Dataset class is updated.
    """

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
    }

    def __init__(self, name, files, file_handler, timestamp_retrieving_method="filename"):
        """ Class which provides methods to handle a set of multiple files (dataset).

        Args:
            name: The name of the dataset.
            files: A string with the complete path to the dataset files. The string can contain placeholder such as
                {year}, {month}, etc. See here for a complete list (TODO: insert link to placeholder documentation).
                For example, the Dataset class finds with the argument
                files="/path/to/files/{year}/{month}/{day}/{hour}{minute}{second}.nc.gz" files such as
                "/path/to/files/2016/11/12/051422.nc.gz"
            file_handler: An object which can handle the dataset files. You can use a file handler class from
                typhon.handlers or write your own class. For example, if this dataset consists of NetCDF files, you can
                use the typhon.handlers.netcdf.NetCDF4File class.
            timestamp_retrieving_method: Defines how the timestamp of the data should be retrieved. Default is "filename".
                Look at Dataset.retrieve_timestamp_from_file() for more details.
        """

        # Initialize member variables:
        self.file_handler = file_handler
        self.files = files
        self.file_extension = os.path.splitext(files)[-1]
        self.name = name
        self.timestamp_retrieving_method = timestamp_retrieving_method

        #super().__init__(**kwargs)

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

    def accumulate(self, start, end, fields, chunk_size=100):
        """ Accumulate all data between two dates in one xarray.DataArray.

        Args:
            start: Starting date as datetime.datetime object.
            end: Ending date as datetime.datetime object.
            fields: List or tuple of variables to select.
            chunk_size:

        Returns:
            One xarray.DataArray with all data.

        Examples:
            dataset = typhon.dataset.Dataset(
                files="path/to/files.nc", file_handler=typhon.handlers.common.NetCDFFile()
            )
            data = dataset.accumulate(
                datetime.datetime(2016, 1, 1), datetime.datetime(2016, 2, 1), fields=("temperature"))

            # do something with data["temperature"] ...
        """

        data = None

        for filename, _ in sorted(list(self.find_files(start, end)), key=lambda x: x[1]):
            print("read file", filename, fields)
            file_data = self.read(filename, fields)
            print("fertig!")
            if data is None:
                data = file_data
            else:
                data = xarray.concat([data, file_data], dim="time")

        return data

    @staticmethod
    def _call_process_function(args):
        """ This is a small wrapper function to call the function that is called on dataset files via .process().

        Args:
            args: A tuple containing following elements:
                (function_name, Dataset object, (filename, timestamp),  kwargs_dictionary)

        Returns:
            The return value of function_name called with the arguments in args.
        """
        if args[4] is None:
            return args[0](args[1], *args[2])
        else:
            return args[0](args[1], *args[2], **args[3])

    @staticmethod
    def _find_subdirs(dir):
        ...

    def find_files(self, start, end, bundle_one_day=False):
        """ Finds all files for the given start and end date.

        This method calculates the days between the start and end date. It uses those days to loop over the dataset
        files. This is a generator method.

        Args:
            start: Start date as tuple, i.e. (YYYY, MM, DD, ?hh, mm, ss?). Hours, minutes and seconds are optional.
            end: End date. Same format as "start".
            bunde_one_day: If this is set to true, all files of one day will be returned at once.

        Yields:
            If bunde_one_day is set to true, this returns one tuple of file names and timestamps for each day.
            Otherwise it will return a tuple of one file name and timestamp only.
        """

        # We We split the path of the input files after the first appearance of {day} or {doy}.
        path_parts = re.split(r'({day}|{doy})', self.files)

        if bundle_one_day:
            day_bundle = []

        # At first, get all days between start and end date
        for date in pd.date_range(start, end):

            # Generate the daily path string for the files
            daily_path = self.generate_filename_from_timestamp(''.join(path_parts[:2]), date)
            if os.path.isdir(daily_path):
                daily_path += "/"

            # Find all files in that daily path
            for filename in glob.iglob(daily_path + "*", recursive=True):

                # Test all files whether they are between the start and end date.
                if filename.endswith(self.file_extension):
                    timestamp = self.retrieve_timestamp_from_file(filename)
                    if timestamp > start and timestamp < end:
                        if bundle_one_day:
                            day_bundle.append([filename, timestamp])
                        else:
                            yield filename, timestamp

            if bundle_one_day:
                yield day_bundle.copy()
                day_bundle.clear()

    def find_file(self, timestamp):
        filename = self.generate_filename_from_timestamp(self.files, timestamp)

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
            for filename in glob.iglob(self.generate_filename_from_timestamp(searching_path, timestamp)):
                if best_file is None:
                    time_difference = self.retrieve_timestamp_from_file(filename) - timestamp
                    best_file = (filename, abs(time_difference.total_seconds()))
                else:
                    time_difference = self.retrieve_timestamp_from_file(filename) - timestamp
                    if abs(time_difference.total_seconds) < best_file[1]:
                        best_file = (filename, abs(time_difference.total_seconds()))

            if best_file is not None:
                return best_file

        return None

    def make_plots(self, start, end, *plot_dicts):
        """ Make plots for all data between two dates.

        Args:
            start: Starting date as datetime.datetime object.
            end: Ending date as datetime.datetime object.
            *plot_dicts: A dictionary of plot attributes.

        Returns:
            List of matplotlib.figures objects.
        """

        vars = []
        for plot in plot_dicts:
            vars.extend(plot["vars"])

        data = self.accumulate(start, end, vars)

        for plot in plot_dicts:
            # TODO: Implement native typhon.plots functions and use them here.
            if plot["type"] == "worldmap":
                plt.title('contour lines over filled continent background')
                ax = plt.axes(projection=ccrs.PlateCarree())

                plt.contourf(data[vars[0]], data[vars[1]], data[vars[2]], 60,
                             transform=ccrs.PlateCarree())
                plt.savefig("test.png")
            else:
                raise ValueError("Unknown plot type '{}'!".format(plot["type"]))

    def retrieve_timestamp_from_file(self, filename, retrieve_method=None):
        """ Retrieves the timestamp from a given file. This is not the real timestamp of the file itself but
        of its content.

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
            A datetime object.
        """

        if retrieve_method is None:
            retrieve_method = self.timestamp_retrieving_method

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
                else:
                    date_args[key] = value

            if "doy" in keys:
                date = datetime.datetime(date_args["year"], 1, 1) + datetime.timedelta(date_args["doy"] - 1)
                date_args["month"] = date.month
                date_args["day"] = date.day
                del date_args["doy"]

            return datetime.datetime(**date_args)
        else:
            return self.file_handler.get_info(filename)["times"][0]

    @staticmethod
    def generate_filename_from_timestamp(template, timestamp):
        """ Generates the file name for a specific date by using the template argument.

            Allowed placeholders in the template are:
                year, year2, month, day, hour, minute, second

            Args:
                template: A string with format placeholders such as {year} or {day}.
                timestamp: A datetime object with the needed date and time.

            Returns:
                A string containing the full path and name of the file.

            Example:
                >>> Dataset.generate_filename_from_timestamp("{year2}/{month}/{day}.dat", datetime.datetime(2016, 1, 1))
                16/01/01.dat

            """

        # Fill all placeholders variables with values
        template = template.format(
            year=timestamp.year, year2=str(timestamp.year)[-2:],
            month="{:02d}".format(timestamp.month), day="{:02d}".format(timestamp.day),
            doy="{:03d}".format((timestamp - datetime.datetime(timestamp.year, 1, 1)).days + 1),
            hour="{:02d}".format(timestamp.hour), minute="{:02d}".format(timestamp.minute), second="{:02d}".format(timestamp.second)
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

    def process(self, start, end, func, kwargs=None, max_processes=12):
        """ Processes a function on all files between start and end date on parallel processes.

        This method can use multiple processes to boost the procedure significantly. Depending on which system
        you work, you should try different numbers for max_processes.

        Args:
            start: Start date as datetime object.
            end: End date. Same format as "start".
            func: Function that should be applied on the data of each file.
            kwargs: A dictionary of additional keyword arguments that should be given over to the function in func.
            max_processes: Max. number of parallel processes to use. When lacking performance, you should change this
                number.

        Returns:
            A list of the return values of the applied function.
        """
        print("Process all files from %s to %s.\nThis may take a while..." % (start, end))

        # Measure the time for profiling.
        start_time = time.time()

        # Create a pool of processes and process all the files with them.
        pool = Pool(processes=max_processes)
        results = pool.map(Dataset._call_process_function,
                           [(func, self, x, kwargs) for x in
                            self.find_files(start, end)])

        if results:
            print("It took %.2f seconds using %d parallel processes to process %d files." % (
                time.time() - start_time, max_processes, len(results)))
        else:
            print("Warning: No files found!")

        return results

    def get_info(self, filename):
        with typhon.files.decompress(filename) as file:
            return self.file_handler.get_info(file)

    def read(self, filename, fields="all"):
        with typhon.files.decompress(filename) as file:
            return self.file_handler.read(file, fields)

    def write(self, filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return self.file_handler.write(filename, data)