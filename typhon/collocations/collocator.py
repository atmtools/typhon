import gc
import logging
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import xarray as xr

from typhon.geodesy import great_circle_distance
from typhon.geographical import GeoIndex
from typhon.utils import add_xarray_groups, get_xarray_groups
from typhon.utils.timeutils import to_datetime, to_timedelta, Timer

__all__ = [
    "Collocator",
    "check_collocation_data"
]

logger = logging.getLogger(__name__)


# The names for the processes. This started as an easter egg, but it actually
# helps to identify different processes during debugging.
PROCESS_NAMES = [
    'Newton', 'Einstein', 'Bohr', 'Darwin', 'Pasteur', 'Freud', 'Galilei',
    'Lavoisier', 'Kepler', 'Copernicus', 'Faraday', 'Maxwell', 'Bernard',
    'Boas', 'Heisenberg', 'Pauling', 'Virchow', 'Schrodinger', 'Rutherford',
    'Dirac', 'Vesalius', 'Brahe', 'Buffon', 'Boltzmann', 'Planck', 'Curie',
    'Herschel', 'Lyell', 'Laplace', 'Hubble', 'Thomson', 'Born', 'Crick',
    'Fermi', 'Euler', 'Liebig', 'Eddington', 'Harvey', 'Malpighi', 'Huygens',
    'Gauss', 'Haller', 'Kekule', 'Koch', 'Gell-Mann', 'Fischer', 'Mendeleev',
    'Glashow', 'Watson', 'Bardeen', 'Neumann', 'Feynman', 'Wegener', 'Hawking',
    'Leeuwenhoek', 'Laue', 'Kirchhoff', 'Bethe', 'Euclid', 'Mendel', 'Onnes',
    'Morgan', 'Helmholtz', 'Ehrlich', 'Mayr', 'Sherrington', 'Dobzhansky',
    'Delbruck', 'Lamarck', 'Bayliss', 'Chomsky', 'Sanger', 'Lucretius',
    'Dalton', 'Broglie', 'Linnaeus', 'Piaget', 'Simpson', 'Levi-Strauss',
    'Margulis', 'Landsteiner', 'Lorenz', 'Wilson', 'Hopkins', 'Elion', 'Selye',
    'Oppenheimer', 'Teller', 'Libby', 'Haeckel', 'Salk', 'Kraepelin',
    'Lysenko', 'Galton', 'Binet', 'Kinsey', 'Fleming', 'Skinner', 'Wundt',
    'Archimedes'
]


class ProcessCrashed(Exception):
    """Helper exception for crashed processes"""
    pass


class Collocator:
    def __init__(
            self, threads=None, name=None, #log_dir=None
    ):
        """Initialize a collocator object that can find collocations

        Args:
            threads: Finding collocations can be parallelized in threads. Give
                here the maximum number of threads that you want to use. Which
                number of threads is the best, may be machine-dependent. So
                this is a parameter that you can use to fine-tune the
                performance. Note: Not yet implemented due to GIL usage of
                sklearn BallTree.
            name: The name of this collocator, will be used in log statements.
        """

        self.empty = None  # xr.Dataset()

        self.index = None
        self.index_with_primary = False

        self.threads = threads

        # These optimization parameters will be overwritten in collocate
        self.bin_factor = None
        self.magnitude_factor = None
        self.tunnel_limit = None
        self.leaf_size = None

        self.name = name if name is not None else "Collocator"

    # If no collocations are found, this will be returned. We need empty
    # arrays to concatenate the results without problems:
    @property
    def no_pairs(self):
        return np.array([[], []])

    @property
    def no_intervals(self):
        return np.array([], dtype='timedelta64[ns]')

    @property
    def no_distances(self):
        return np.array([])

    def __call__(self, *args, **kwargs):
        return self.collocate(*args, **kwargs)

    def _debug(self, msg):
        logger.debug(f"[{self.name}] {msg}")

    def _info(self, msg):
        logger.info(f"[{self.name}] {msg}")

    def _error(self, msg):
        logger.error(f"[{self.name}] {msg}")

    def collocate_filesets(
            self, filesets, start=None, end=None, processes=None, output=None,
            bundle=None, skip_file_errors=False, post_processor=None,
            post_processor_kwargs=None, **kwargs
    ):
        """Find collocation between the data of two filesets

        If you want to save the collocations directly to disk, it may be easier
        to use :meth:`~typhon.collocations.Collocations.search` directly.

        Args:
            filesets: A list of two :class:`FileSet` objects, the primary and
                the secondary fileset. Can be also
                :class:`~typhon.collocations.common.Collocations` objects with
                `read_mode=collapse`. The order of the filesets is irrelevant
                for the results of the collocation search but files from the
                secondary fileset might be read multiple times if  using
                parallel processing (`processes` is greater than one). The
                number of output files could be different (see also the option
                `bundle`).
            start: Start date either as datetime object or as string
                ("YYYY-MM-DD hh:mm:ss"). Year, month and day are required.
                Hours, minutes and seconds are optional. If not given, it is
                datetime.min per default.
            end: End date. Same format as "start". If not given, it is
                datetime.max per default.
            processes: Collocating can be parallelized which improves the
                performance significantly. Pass here the number of processes to
                use.
            output: Fileset object where the collocated data should be stored.
            bundle: Set this to *primary* if you want to bundle the output 
                files by their collocated primaries, i.e. there will be only 
                one output file per primary. *daily* is also possible, then all
                files from one day are bundled together. Per default, all
                collocations for each file match will be saved separately.
                This might lead to a high number of output files.
                Note: *daily* means one process bundles all collocations from
                one day into one output file. If using multiple processes, this
                could still produce several daily output files per day.
            skip_file_errors: If this is *True* and a file could not be read, 
                the file and its match will be skipped and a warning will be 
                printed. Otheriwse the program will stop (default).
            post_processor: A function for post-processing the collocated data
                before saving it to `output`. Must accept two parameters: a
                xarray.Dataset with the collocated data and a dictionary with
                the path attributes from the collocated files.
            post_processor_kwargs: A dictionary with keyword arguments that
                should be passed to `post_processor`.
            **kwargs: Further keyword arguments that are allowed for
                :meth:`collocate`.

        Yields:
            A xarray.Dataset with the collocated data if `output` is not set.
            If `output` is set to a FileSet-like object, only the filename of
            the stored collocations is yielded. The results are not ordered if
            you use more than one process. For more information about the 
            yielded xarray.Dataset have a look at :meth:`collocate`.

        Examples:

        .. code-block:: python

        """
        timer = Timer().start()

        if len(filesets) != 2:
            raise ValueError("Only collocating two filesets at once is allowed"
                             "at the moment!")

        # Check the max_interval argument because we need it later
        max_interval = kwargs.get("max_interval", None)
        if max_interval is None:
            raise ValueError("Collocating filesets without max_interval is"
                             " not yet implemented!")

        if start is None:
            start = datetime.min
        else:
            start = to_datetime(start)
        if end is None:
            end = datetime.max
        else:
            end = to_datetime(end)

        self._info(f"Collocate from {start} to {end}")

        # Find the files from both filesets which overlap tempoerally.
        matches = list(filesets[0].match(
            filesets[1], start=start, end=end, max_interval=max_interval,
        ))

        if processes is None:
            processes = 1

        # Make sure that there are never more processes than matches
        processes = min(processes, len(matches))

        total_matches = sum(len(match[1]) for match in matches)
        self._info(f"using {processes} process(es) on {total_matches} matches")

        # MAGIC with processes
        # Each process gets a list with matches. Important: the matches should
        # be continuous to guarantee a good performance. After finishing one
        # match, the process pushes its results to the result queue. If errors
        # are raised during collocating, the raised errors are pushed to the 
        # error queue,
        matches_chunks = np.array_split(matches, processes)

        # This queue collects all results:
        results = Queue(maxsize=processes)

        # This queue collects all error exceptions
        errors = Queue()

        # Extend the keyword arguments that we are going to pass to
        # _collocate_files:
        kwargs.update({
            "start": start,
            "end": end,
            "filesets": filesets,
            "output": output,
            "bundle": bundle,
            "skip_file_errors": skip_file_errors,
            "post_processor": post_processor,
            "post_processor_kwargs": post_processor_kwargs,
        })

        # This list contains all running processes
        process_list = [
            Process(
                target=Collocator._process_caller,
                args=(
                    self, results, errors, PROCESS_NAMES[i],
                ),
                kwargs={**kwargs, "matches": matches_chunk},
                daemon=True,
            )
            for i, matches_chunk in enumerate(matches_chunks)
        ]

        # We want to keep track of the progress of the collocation search since
        # it may take a while.
        process_progress = {
            name: 0.  # Each process is at 0 percent at the beginning
            for name in PROCESS_NAMES[:processes]
        }

        # Start all processes:
        for process in process_list:
            process.start()

        # As long as some processes are still running, wait for their results:
        running = process_list.copy()

        processed_matches = 0

        # The main process has two tasks during its child processes are
        # collocating. 
        # 1) Collect their results and yield them to the user
        # 2) Display the progress and estimate the remaining processing time
        while running:

            # Filter out all processes that are dead: they either crashed or
            # complete their task
            running = [
                process for process in running if process.is_alive()
            ]

            # Get all results from the result queue
            while not results.empty():
                process, progress, result = results.get()

                # The process might be crashed. To keep the remaining time
                # estimation useful, we exclude the crashed process from the
                # calculation.
                if result is ProcessCrashed:
                    del process_progress[process]
                else:
                    process_progress[process] = progress

                try:
                    nerrors = errors.qsize()
                except NotImplementedError:
                    nerrors = 'unknown'

                self._print_progress(
                    timer.elapsed, process_progress, len(running), nerrors)
                if result is not None:
                    yield result

                # Explicit free up memory:
                gc.collect()

        for process in process_list:
            process.join()

        if not errors.empty():
            self._error("Some processes terminated due to errors:")

        while not errors.empty():
            error = errors.get()
            msg = '\n'.join([
                "-"*79,
                error[2],
                "".join(traceback.format_tb(error[1])),
                "-" * 79 + "\n"
            ])
            self._error(msg)

    @staticmethod
    def _print_progress(elapsed_time, process_progress, processes, errors):

        elapsed_time -= timedelta(microseconds=elapsed_time.microseconds)
        if len(process_progress) == 0:
            msg = "-"*79 + "\n"
            msg += f"100% | {elapsed_time} hours elapsed | " \
                   f"{errors} processes failed\n"
            msg += "-"*79 + "\n"
            logger.error(msg)
            return

        progress = sum(process_progress.values()) / len(process_progress)

        try:
            expected_time = elapsed_time * (100 / progress - 1)
            expected_time -= timedelta(
                    microseconds=expected_time.microseconds)
        except ZeroDivisionError:
            expected_time = "unknown"

        msg = "-"*79 + "\n"
        msg += f"{progress:.0f}% | {elapsed_time} hours elapsed, " \
               f"{expected_time} hours left | {processes} proc running, " \
               f"{errors} failed\n"
        msg += "-"*79 + "\n"
        logger.error(msg)

    @staticmethod
    def _process_caller(
            self, results, errors, name, output, bundle, post_processor,
            post_processor_kwargs, **kwargs):
        """Wrapper around _collocate_matches

        This function is called for each process. It communicates with the main
        process via the result and error queue.

        Result Queue:
            Adds for each collocated file match the process name, its progress
            and the actual results.

        Error Queue:
            If an error is raised, the name of this proces and the error
            messages is put to this queue.
        """
        self.name = name

        # We keep track of how many file pairs we have already processed to
        # make the error debugging easier. We need the match in flat form:
        matches = [
            [match[0], secondary]
            for match in kwargs['matches']
            for secondary in match[1]
        ]

        # If we want to bundle the output, we need to collect some contents.
        # The current_bundle_tag stores a certain information for the current
        # bundle (e.g. filename of primary or day of the year). If it changes,
        # the bundle is stored to disk and a new bundle is created.
        cached_data = []
        cached_attributes = {}
        current_bundle_tag = None
        try:
            processed = 0
            collocated_matches = self._collocate_matches(**kwargs)
            for collocations, attributes in collocated_matches:
                match = matches[processed]
                processed += 1
                progress = 100 * processed / len(matches)

                if collocations is None:
                    results.put([name, progress, None])
                    continue

                # The user does not want to bundle anything therefore just save
                # the current collocations
                if bundle is None:
                    result = self._save_and_return(
                            collocations, attributes, output,
                            post_processor, post_processor_kwargs
                    )
                    results.put([name, progress, result])
                    continue

                # The user may want to bundle the collocations before writing
                # them to disk, e.g. by their primaries.
                save_cache = self._should_save_cache(
                        bundle, current_bundle_tag, match,
                        to_datetime(collocations.attrs["start_time"])
                )

                if save_cache:
                    result = self._save_and_return(
                        cached_data,
                        cached_attributes, output,
                        post_processor, post_processor_kwargs
                    )
                    results.put([name, progress, result])

                    cached_data = []
                    cached_attributes = {}

                # So far, we have not cached any collocations or we still need
                # to wait before saving them to disk.
                cached_data.append(collocations)
                cached_attributes.update(**attributes)

                if bundle == "primary":
                    current_bundle_tag = match[0].path
                elif bundle == "daily":
                    current_bundle_tag = \
                        to_datetime(collocations.attrs["start_time"]).date()

            # After all iterations, save last cached data to disk:
            if cached_data:
                result = self._save_and_return(
                    cached_data,
                    cached_attributes, output,
                    post_processor, post_processor_kwargs
                )
                results.put([name, progress, result])

        except Exception as exception:
            # Tell the main process to stop considering this process for the 
            # remaining processing time:
            results.put(
                [name, 100., ProcessCrashed]
            )

            self._error("ERROR: I got a problem and terminate!")

            # Build a message that contains all important information for
            # debugging:

            msg = f"Process {name} ({matches[0][0].times[0]} -" \
                f"{matches[-1][0].times[1]}) failed\n" \
                f"Failed to collocate {matches[processed]} with"\
                f"{matches[processed]}\n"

            # The main process needs to know about this exception!
            error = [
                name, exception.__traceback__,
                msg + "ERROR: " + str(exception)
            ]
            errors.put(error)

            self._error(exception)

            # Finally, raise the exception to terminate this process:
            raise exception

        self._info(f"Finished all {len(matches)} matches")

    def _save_and_return(self, collocations, attributes, output,
            post_processor, post_processor_kwargs):
        """Save collocations to disk or return them"""

        if isinstance(collocations, list):
            collocations = concat_collocations(
                    collocations
            )

        if output is None:
            return collocations, attributes
        else:
            filename = output.get_filename(
                [to_datetime(collocations.attrs["start_time"]),
                 to_datetime(collocations.attrs["end_time"])],
                fill=attributes
            )

            # Apply a post processor function from the user
            if post_processor is not None:
                if post_processor_kwargs is None:
                    post_processor_kwargs = {}

                collocations = post_processor(
                    collocations, attributes, **post_processor_kwargs
                )

            if collocations is None:
                return None

            self._info(f"Store collocations to\n{filename}")

            # Write the data to the file.
            output.write(collocations, filename)
            return filename

    @staticmethod
    def _should_save_cache(bundle, current_bundle_tag, match, start_time):
        """Return true if the cache should be saved otherwise false
        """
        if current_bundle_tag is None:
            return False
        elif bundle == "primary":
            # Check whether the primary has changed since the last time:
            return current_bundle_tag != match[0].path
        elif bundle == "daily":
            # Has the day changed since last time?
            return current_bundle_tag != start_time.date()

        # In all other cases, the bundle should not be saved yet:
        return False

    def _collocate_matches(
        self, filesets, matches, skip_file_errors, **kwargs
    ):
        """Load file matches and collocate their content

        Yields:
            A tuple of two items: the first is always the current percentage
            of progress. If output is True, the second is only the filename of
            the saved collocations. Otherwise, it is a tuple of collocations
            and their collected :class:`~typhon.files.handlers.common.FileInfo`
            attributes as a dictionary.
        """
        # Load all matches in a parallized queue:
        loaded_matches = filesets[0].align(
            filesets[1], matches=matches, return_info=True, compact=False,
            skip_errors=skip_file_errors,
        )


        for loaded_match in loaded_matches:
            # The FileInfo objects of the matched files:
            files = loaded_match[0][0], loaded_match[1][0]

            # We copy the data from the matches since they might be used for
            # other matches as well:
            primary, secondary = \
                loaded_match[0][1].copy(), loaded_match[1][1].copy()

            self._debug(f"Collocate {files[0].path}\nwith {files[1].path}")

            collocations = self.collocate(
                (filesets[0].name, primary),
                (filesets[1].name, secondary), **kwargs,
            )

            if collocations is None:
                self._debug("Found no collocations!")
                # At least, give the process caller a progress update:
                yield None, None
                continue

            # Check whether the collocation data is compatible and was build
            # correctly
            check_collocation_data(collocations)

            found = [
                collocations[f"{filesets[0].name}/time"].size,
                collocations[f"{filesets[1].name}/time"].size
            ]

            self._debug(
                f"Found {found[0]} ({filesets[0].name}) and "
                f"{found[1]} ({filesets[1].name}) collocations"
            )

            # Add the names of the processed files:
            for f in range(2):
                if f"{filesets[f].name}/__file" in collocations.variables:
                    continue

                collocations[f"{filesets[f].name}/__file"] = files[f].path

            # Collect the attributes of the input files. The attributes get a
            # prefix, primary or secondary, to allow not-unique names.
            attributes = {
                f"primary.{p}" if f == 0 else f"secondary.{p}": v
                for f, file in enumerate(files)
                for p, v in file.attr.items()
            }

            yield collocations, attributes


    def collocate(
            self, primary, secondary, max_interval=None, max_distance=None,
            bin_factor=1, magnitude_factor=10, tunnel_limit=None, start=None,
            end=None, leaf_size=40
    ):
        """Find collocations between two xarray.Dataset objects

        Collocations are two or more data points that are located close to each
        other in space and/or time.

        Each xarray.Dataset contain the variables *time*, *lat*, *lon*. They
        must be - if they are coordinates - unique. Otherwise, their
        coordinates must be unique, i.e. they cannot contain duplicated values.
        *time* must be a 1-dimensional array with a *numpy.datetime64*-like
        data type. *lat* and *lon* can be gridded, i.e. they can be multi-
        dimensional. However, they must always share the first dimension with
        *time*. *lat* must be latitudes between *-90* (south) and *90* (north)
        and *lon* must be longitudes between *-180* (west) and *180* (east)
        degrees. See below for examples.

        The collocation searched is performed with a fast ball tree
        implementation by scikit-learn. The ball tree is cached and reused
        whenever the data points from `primary` or `secondary` have not
        changed.

        If you want to find collocations between FileSet objects, use
        :class:`collocate_filesets` instead.

        Args:
            primary: A tuple of a string with the dataset name and a
                xarray.Dataset that fulfill the specifications from above. Can
                be also a xarray.Dataset only, the name is then automatically
                set to *primary*.
            secondary: A tuple of a string with the dataset name and a
                xarray.Dataset that fulfill the specifications from above. Can
                be also a xarray.Dataset only, the name is then automatically
                set to *secondary*.
            max_interval: Either a number as a time interval in seconds, a
                string containing a time with a unit (e.g. *100 minutes*) or a
                timedelta object. This is the maximum time interval between two
                data points. If this is None, the data will be searched for
                spatial collocations only.
            max_distance: Either a number as a length in kilometers or a string
                containing a length with a unit (e.g. *100 meters*). This is
                the maximum distance between two data points to meet the
                collocation criteria. If this is None, the data will be
                searched for temporal collocations only. Either `max_interval`
                or *max_distance* must be given.
            tunnel_limit: Maximum distance in kilometers at which to switch
                from tunnel to haversine distance metric. Per default this
                algorithm uses the tunnel metric, which simply transform all
                latitudes and longitudes to 3D-cartesian space and calculate
                their euclidean distance. This is faster than the haversine
                metric but produces an error that grows with larger distances.
                When searching for distances exceeding this limit
                (`max_distance` is greater than this parameter), the haversine
                metric is used, which is more accurate but takes more time.
                Default is 1000 kilometers.
            magnitude_factor: Since building new trees is expensive, this
                algorithm tries to use the last tree when possible (e.g. for
                data with fixed grid). However, building the tree with the
                larger dataset and query it with the smaller dataset is faster
                than vice versa. Depending on which premise to follow, there
                might have a different performance in the end. This parameter
                is the factor of that one dataset must be larger than the other
                to throw away an already-built ball tree and rebuild it with
                the larger dataset.
            leaf_size: The size of one leaf in the Ball Tree. The higher the
                leaf size the faster is the tree building but the slower is the
                tree query. The optimal leaf size is dataset-dependent. Default
                is 40.
            bin_factor: When using a temporal criterion via `max_interval`, the
                data will be temporally binned to speed-up the search. The bin
                size is `bin_factor` * `max_interval`. Which bin factor is the
                best, may be dataset-dependent. So this is a parameter that you
                can use to fine-tune the performance.
            start: Limit the collocated data from this start date. Can be
                either as datetime object or as string ("YYYY-MM-DD hh:mm:ss").
                Year, month and day are required. Hours, minutes and seconds
                are optional. If not given, it is datetime.min per default.
            end: End date. Same format as "start". If not given, it is
                datetime.max per default.

        Returns:
            None if no collocations were found. Otherwise,
            a xarray.Dataset with the collocated data in *compact* form. It
            consists of three groups (groups of variables containing */* in
            their name): the *primary*, *secondary* and the *Collocations*
            group. If you passed `primary` or `secondary` with own names,
            they will be used in the output. The *Collocations* group contains
            information about the found collocations. *Collocations/pairs* is
            a 2xN array where N is the number of found collocations. It
            contains the indices of the *primary* and *secondary* data points
            which are collocations. The indices refer to the data points stored
            in the *primary* or *secondary* group. *Collocations/interval* and
            *Collocations/distance* are the intervals and distances between the
            collocations in seconds and kilometers, respectively. Collocations
            in *compact* form are efficient when saving them to disk but it
            might be complicated to use them directly. Consider applying
            :func:`~typhon.collocations.common.collapse` or
            :func:`~typhon.collocations.common.expand` on them.

        Examples:

            .. code-block: python
                
                # TODO: Update this example!

                import numpy as np
                from typhon.collocations import Collocator

                # Create the data. primary and secondary can also be
                # xarray.Dataset objects:
                primary = {
                    "time": np.arange(
                        "2018-01-01", "2018-01-02", dtype="datetime64[h]"
                    ),
                    "lat": 30.*np.sin(np.linspace(-3.14, 3.14, 24))+20,
                    "lon": np.linspace(0, 90, 24),
                }
                secondary = {
                    "time": np.arange(
                        "2018-01-01", "2018-01-02", dtype="datetime64[h]"
                    ),
                    "lat": 30.*np.sin(np.linspace(-3.14, 3.14, 24)+1.)+20,
                    "lon": np.linspace(0, 90, 24),
                }

                # Find collocations with a maximum distance of 300 kilometers
                # and a maximum interval of 1 hour
                collocator = Collocator()
                collocated = collocator.collocate(
                    primary, secondary,
                    max_distance="300km", max_interval="1h"
                )

                print(collocated)


        """
        if max_distance is None and max_interval is None:
            raise ValueError(
                "Either max_distance or max_interval must be given!"
            )

        if max_interval is not None:
            max_interval = to_timedelta(max_interval, numbers_as="seconds")

        # The user can give strings instead of datetime objects:
        start = datetime.min if start is None else to_datetime(start)
        end = datetime.max if end is None else to_datetime(end)

        # Did the user give the datasets specific names?
        primary_name, primary, secondary_name, secondary = self._get_names(
            primary, secondary
        )

        # Select the common time period of both datasets and flat them.
        primary, secondary = self._prepare_data(
            primary, secondary, max_interval, start, end
        )

        # Maybe there is no data left after selection?
        if primary is None:
            return self.empty

        self.bin_factor = bin_factor
        self.magnitude_factor = magnitude_factor
        self.tunnel_limit = tunnel_limit
        self.leaf_size = leaf_size

        timer = Timer().start()
        
        # We cannot allow NaNs in the time, lat or lon fields
        not_nans1 = self._get_not_nans(primary)
        not_nans2 = self._get_not_nans(secondary)

        # Retrieve the important fields from the data. To avoid any overhead by
        # xarray, we use the plain numpy.arrays and do not use the isel method
        # (see https://github.com/pydata/xarray/issues/2227). We rather use
        # index arrays that we use later to select the rest of the data
        lat1 = primary.lat.values[not_nans1]
        lon1 = primary.lon.values[not_nans1]
        time1 = primary.time.values[not_nans1]
        lat2 = secondary.lat.values[not_nans2]
        lon2 = secondary.lon.values[not_nans2]
        time2 = secondary.time.values[not_nans2]
        original_indices = [
            np.arange(primary.time.size)[not_nans1],
            np.arange(secondary.time.size)[not_nans2]
        ]
        self._debug(f"{timer} for filtering NaNs")

        # We can search for spatial collocations (max_interval=None), temporal
        # collocations (max_distance=None) or both.
        if max_interval is None:
            # Search for spatial collocations only:
            pairs, distances = self.spatial_search(
                lat1, lon1, lat2, lon2, max_distance,
            )

            intervals = self._get_intervals(
                time1[pairs[0]], time2[pairs[1]]
            )

            return self._create_return(
                primary, secondary, primary_name, secondary_name,
                self._to_original(pairs, original_indices),
                intervals, distances,
                max_interval, max_distance
            )
        elif max_distance is None:
            # Search for temporal collocations only:
            pairs, intervals = self.temporal_search(
                time1, time2, max_interval
            )

            distances = self._get_distances(
                lat1[pairs[0]], lon1[pairs[0]],
                lat2[pairs[1]], lon2[pairs[1]],
            )

            return self._create_return(
                primary, secondary, primary_name, secondary_name,
                self._to_original(pairs, original_indices),
                intervals, distances,
                max_interval, max_distance
            )

        # The user wants to use both criteria and search for spatial and
        # temporal collocations. At first, we do a coarse temporal pre-binning
        # so that we only search for collocations between points that might
        # also be temporally collocated. Unfortunately, this also produces an
        # overhead that is only negligible if we have a lot of data:
        data_magnitude = time1.size * time2.size

        if data_magnitude > 100_0000:
            # We have enough data, do temporal pre-binning!
            pairs, distances = self.spatial_search_with_temporal_binning(
                {"lat": lat1, "lon": lon1, "time": time1},
                {"lat": lat2, "lon": lon2, "time": time2},
                max_distance, max_interval
            )
        else:
            # We do not have enough data to justify that whole pre-binning.
            # Simply do it directly!
            pairs, distances = self.spatial_search(
                lat1, lon1, lat2, lon2, max_distance,
            )

        # Did we find any spatial collocations?
        if not pairs.any():
            return self.empty

        # Check now whether the spatial collocations really pass the temporal
        # condition:
        passed_temporal_check, intervals = self._temporal_check(
            time1[pairs[0]], time2[pairs[1]], max_interval
        )

        # Return only the values that passed the time check
        return self._create_return(
            primary, secondary, primary_name, secondary_name,
            self._to_original(
                pairs[:, passed_temporal_check], original_indices),
            intervals, distances[passed_temporal_check],
            max_interval, max_distance
        )

    @staticmethod
    def _to_original(pairs, original_indices):
        return np.array([
            original_indices[i][pair_array]
            for i, pair_array in enumerate(pairs)
        ])

    @staticmethod
    def _get_names(primary, secondary):
        # Check out whether the user gave the primary and secondary any name:
        if isinstance(primary, (tuple, list)):
            primary_name, primary = primary
        else:
            primary_name = "primary"
        if isinstance(secondary, (tuple, list)):
            secondary_name, secondary = secondary
        else:
            secondary_name = "secondary"

        return primary_name, primary, secondary_name, secondary

    def _prepare_data(self, primary, secondary, max_interval, start, end):
        """Prepare the data for the collocation search
        
        This method selects the time period which should be searched for
        collocations and flats the input datasets if they have gridded
        variables.

        Returns:
            The datasets constraint to the common time period, sorted by time
            and flattened. If no common time period could be found, two None
            objects are returned.
        """
        if max_interval is not None:
            timer = Timer().start()
            # We do not have to collocate everything, just the common time
            # period expanded by max_interval and limited by the global start
            # and end parameter:
            primary_period, secondary_period = self._get_common_time_period(
                primary, secondary, max_interval, start, end
            )

            # Check whether something is left:
            if not primary_period.size or not secondary_period.size:
                return None, None

            # We need everything sorted by the time, otherwise xarray's stack
            # method makes problems:
            primary_period = primary_period.sortby(primary_period)
            primary_dim = primary_period.dims[0]
            secondary_period = secondary_period.sortby(secondary_period)
            secondary_dim = secondary_period.dims[0]

            # Select the common time period and while using sorted indices:
            primary = primary.sel(**{primary_dim: primary_period[primary_dim]})
            secondary = secondary.sel(
                **{secondary_dim: secondary_period[secondary_dim]}
            )

            # Check whether something is left:
            if not primary_period.size or not secondary_period.size:
                return None, None

            self._debug(f"{timer} for selecting common time period")

        # Flat the data: For collocating, we need a flat data structure.
        # Fortunately, xarray provides the very convenient stack method
        # where we can flat multiple dimensions to one. Which dimensions do
        # we have to stack together? We need the fields *time*, *lat* and
        # *lon* to be flat. So we choose their dimensions to be stacked.
        timer = Timer().start()
        primary = self._flat_to_main_coord(primary)
        secondary = self._flat_to_main_coord(secondary)
        self._debug(f"{timer} for flatting data")

        return primary, secondary

    @staticmethod
    def _get_common_time_period(
            primary, secondary, max_interval, start, end):
        max_interval = pd.Timedelta(max_interval)

        # We want to select a common time window from both datasets,
        # aligned to the primary's time coverage. Because xarray has a
        # very annoying bug in time retrieving
        # (https://github.com/pydata/xarray/issues/1240), this is a
        # little bit cumbersome:
        common_start = max(
            start,
            pd.Timestamp(primary.time.min().item(0)) - max_interval,
            pd.Timestamp(secondary.time.min().item(0)) - max_interval
        )
        common_end = min(
            end,
            pd.Timestamp(primary.time.max().item(0)) + max_interval,
            pd.Timestamp(secondary.time.max().item(0)) + max_interval
        )

        primary_period = primary.time.where(
            (primary.time.values >= np.datetime64(common_start))
            & (primary.time.values <= np.datetime64(common_end))
        ).dropna(primary.time.dims[0])

        secondary_period = secondary.time.where(
            (secondary.time.values >= np.datetime64(common_start))
            & (secondary.time.values <= np.datetime64(common_end))
        ).dropna(secondary.time.dims[0])

        return primary_period, secondary_period

    @staticmethod
    def _get_not_nans(dataset):
        return dataset.lat.notnull().values & dataset.lon.notnull().values

    @staticmethod
    def _flat_to_main_coord(data):
        """Make the dataset flat despite of its original structure

        We need a flat dataset structure for the collocation algorithms, i.e.
        time, lat and lon are not allowed to be gridded, they must be
        1-dimensional and share the same dimension (namely *collocation*).
        There are three groups of original data structures that this method
        can handle:

        * linear (e.g. ship track measurements): time, lat and lon have the
            same dimension and are all 1-dimensional. Fulfills all criteria
            from above. No action has to be taken.
        * gridded_coords (e.g. instruments on satellites with gridded swaths):
            lat or lon are gridded (they have multiple dimensions). Stack
            the coordinates of them together to a new shared dimension.

        Args:
            data: xr.Dataset object

        Returns:
            A xr.Dataset where time, lat and lon are aligned on one shared
            dimension.
        """
        # Flat:
        shared_dims = list(
            set(data.time.dims) | set(data.lat.dims) | set(data.lon.dims)
        )

        # Check whether the dataset is flat (time, lat and lon share the same
        # dimension size and are 1-dimensional)
        if len(shared_dims) == 1:

            if shared_dims[0] in ("time", "lat", "lon"):
                # One of the key variables is the main dimension! Change this:
                data["collocation"] = shared_dims[0], np.arange(
                    data[shared_dims[0]].size)
                data = data.swap_dims({shared_dims[0]: "collocation"})
                data = data.reset_coords(shared_dims[0])

                # So far, collocation is a coordinate. We want to make it to a
                # dimension,  so drop its values:
                return data.drop("collocation")

            return data.rename({
                shared_dims[0]: "collocation"
            })

        # The coordinates are gridded:
        # Some field might be more deeply stacked than another. Choose the
        # dimensions of the most deeply stacked variable:
        dims = max(
            data["time"].dims, data["lat"].dims, data["lon"].dims,
            key=lambda x: len(x)
        )
        # We want to be able to retrieve additional fields after collocating.
        # Therefore, we give each dimension that is no coordinate yet a value
        # to use them as indices later.
        for dim in dims:
            if dim not in data.coords:
                data[dim] = dim, np.arange(data.dims[dim])

        # We assume that coordinates must be unique! Otherwise, we would have
        # to use this ugly work-around:
        # Replace the former coordinates with new coordinates that have unique 
        # values.
        # new_dims = []
        # for dim in dims:
        #     new_dim = f"__replacement_{dim}"
        #     data[new_dim] = dim, np.arange(data.dims[dim])
        #     data = data.swap_dims({dim: new_dim})
        #     new_dims.append(new_dim)
        return data.stack(collocation=dims)

    def _create_return(
            self, primary, secondary, primary_name, secondary_name,
            original_pairs, intervals, distances,
            max_interval, max_distance
    ):
        if not original_pairs.any():
            return self.empty

        pairs = []
        output = {}

        names = [primary_name, secondary_name]
        for i, dataset in enumerate([primary, secondary]):
            # name of the current dataset (primary or secondary)
            name = names[i]

            # These are the indices of the points in the original data that
            # have collocations. We remove the duplicates since we want to copy
            # the required data only once. They are called original_indices
            # because they are the indices in the original data array:
            original_indices = pd.unique(original_pairs[i])

            # After selecting the collocated data, the original indices cannot
            # be applied any longer. We need new indices that indicate the
            # pairs in the collocated data.
            new_indices = np.empty(original_indices.max() + 1, dtype=int)
            new_indices[original_indices] = np.arange(
                original_indices.size
            )

            collocation_indices = new_indices[original_pairs[i]]

            # Save the collocation indices in the metadata group:
            pairs.append(collocation_indices)

            output[names[i]] = dataset.isel(collocation=original_indices)

            # We have to convert the MultiIndex to a normal index because we
            # cannot store it to a file otherwise. We can convert it by simply
            # setting it to new values, but we are losing the sub-level
            # coordinates (the dimenisons that we stacked to create the
            # multi-index in the first place) with that step. Hence, we store
            # the sub-level coordinates in additional dataset to preserve them.
            main_coord_is_multiindex = isinstance(
                output[name].get_index("collocation"),
                pd.core.indexes.multi.MultiIndex
            )
            if main_coord_is_multiindex:
                stacked_dims_data = xr.merge([
                    xr.DataArray(
                        output[name][dim].values,
                        name=dim, dims=["collocation"]
                    )
                    for dim in output[name].get_index("collocation").names
                ])

            # Okay, actually we want to get rid of the main coordinate. It
            # should stay as a dimension name but without own labels. I.e. we
            # want to drop it. Because it still may a MultiIndex, we cannot
            # drop it directly but we have to set it to something different.
            output[name]["collocation"] = \
                np.arange(output[name]["collocation"].size)

            if main_coord_is_multiindex:
                # Now, since we unstacked the multi-index, we can add the
                # stacked dimensions back to the dataset:
                output[name] = xr.merge(
                    [output[name], stacked_dims_data],
                )

            # For the flattening we might have created temporal variables,
            # also collect them to drop:
            vars_to_drop = [
                var for var in output[name].variables.keys()
                if var.startswith("__replacement_")
            ]

            output[name] = output[name].drop([
                f"collocation", *vars_to_drop
            ])

        # Merge all datasets into one:
        output = add_xarray_groups(
            xr.Dataset(), **output
        )

        # This holds the collocation information (pairs, intervals and
        # distances):
        metadata = xr.Dataset()
        metadata["pairs"] = xr.DataArray(
            np.array(pairs, dtype=int), dims=("group", "collocation"),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": primary_name,
                "secondary": secondary_name,
            }
        )
        metadata["interval"] = xr.DataArray(
            intervals, dims=("collocation", ),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": primary_name,
                "secondary": secondary_name,
            }
        )
        metadata["distance"] = xr.DataArray(
            distances, dims=("collocation",),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
                "primary": primary_name,
                "secondary": secondary_name,
                "units": "kilometers",
            }
        )
        metadata["group"] = xr.DataArray(
            [primary_name, secondary_name], dims=("group",),
            attrs={
                "max_interval": f"Max. interval in secs: {max_interval}",
                "max_distance": f"Max. distance in kilometers: {max_distance}",
            }
        )

        output = add_xarray_groups(
            output, Collocations=metadata
        )

        start = pd.Timestamp(
            output[primary_name+"/time"].min().item(0)
        )
        end = pd.Timestamp(
            output[primary_name+"/time"].max().item(0)
        )

        output.attrs = {
            "start_time": str(start),
            "end_time": str(end),
        }

        return output

    @staticmethod
    def get_meta_group():
        return f"Collocations"

    def spatial_search_with_temporal_binning(
            self, primary, secondary, max_distance, max_interval
    ):
        # For time-binning purposes, pandas Dataframe objects are a good choice
        primary = pd.DataFrame(primary).set_index("time")
        secondary = pd.DataFrame(secondary).set_index("time")

        # Now let's split the two data data along their time coordinate so
        # we avoid searching for spatial collocations that do not fulfill
        # the temporal condition in the first place. However, the overhead
        # of the finding algorithm must be considered too (for example the
        # BallTree creation time). This can be adjusted by the parameter
        # bin_factor:
        bin_duration = self.bin_factor * max_interval

        # The binning is more efficient if we use the largest dataset as
        # primary:
        swapped_datasets = secondary.size > primary.size
        if swapped_datasets:
            primary, secondary = secondary, primary

        # Let's bin the primaries along their time axis and search for the
        # corresponding secondary bins:
        bin_pairs = (
            self._bin_pairs(start, chunk, primary, secondary, max_interval)
            for start, chunk in primary.groupby(pd.Grouper(freq=bin_duration))
        )

        # Add arguments to the bins (we need them for the spatial search
        # function):
        bins_with_args = (
            [self, max_distance, *bin_pair]
            for bin_pair in bin_pairs
        )

        # Unfortunately, a first attempt parallelizing this using threads
        # worsened the performance. Update: The BallTree code from scikit-learn
        # does not release the GIL. But apparently there will be a new version
        # coming that solves this problem, see this scikit-learn issue:
        # https://github.com/scikit-learn/scikit-learn/pull/10887. So stay
        # tuned!
        # threads = 1 if self.threads is None else self.threads
        t = Timer(verbose=False).start()
        # with ThreadPoolExecutor(max_workers=2) as pool:
        #     results = list(pool.map(
        #         Collocator._spatial_search_bin, bins_with_args
        #     ))

        results = list(map(
            Collocator._spatial_search_bin, bins_with_args
        ))

        self._debug(f"Collocated {len(results)} bins in {t.stop()}")

        pairs_list, distances_list = zip(*results)
        pairs = np.hstack(pairs_list)

        # No collocations were found.
        if not pairs.any():
            return self.no_pairs, self.no_distances

        # Stack the rest of the results together:
        distances = np.hstack(distances_list)

        if swapped_datasets:
            # Swap the rows of the results
            pairs[[0, 1]] = pairs[[1, 0]]

        return pairs.astype("int64"), distances

    @staticmethod
    def _bin_pairs(chunk1_start, chunk1, primary, secondary, max_interval):
        """"""
        chunk2_start = chunk1_start - max_interval
        chunk2_end = chunk1.index.max() + max_interval
        offset1 = primary.index.searchsorted(chunk1_start)
        offset2 = secondary.index.searchsorted(chunk2_start)
        chunk2 = secondary.loc[chunk2_start:chunk2_end]
        return offset1, chunk1, offset2, chunk2

    @staticmethod
    def _spatial_search_bin(args):
        self, max_distance, offset1, data1, offset2, data2 = args

        if data1.empty or data2.empty:
            return self.no_pairs, self.no_distances

        pairs, distances = self.spatial_search(
            data1["lat"].values, data1["lon"].values,
            data2["lat"].values, data2["lon"].values, max_distance
        )
        pairs[0] += offset1
        pairs[1] += offset2
        return pairs, distances

    def spatial_search(self, lat1, lon1, lat2, lon2, max_distance):
        # Finding collocations is expensive, therefore we want to optimize it
        # and have to decide which points to use for the index building.
        index_with_primary = self._choose_points_to_build_index(
            [lat1, lon1], [lat2, lon2],
        )

        self.index_with_primary = index_with_primary

        if index_with_primary:
            build_points = lat1, lon1
            query_points = lat2, lon2
        else:
            build_points = lat2, lon2
            query_points = lat1, lon1

        self.index = self._build_spatial_index(*build_points)
        pairs, distances = self.index.query(*query_points, r=max_distance)

        # No collocations were found.
        if not pairs.any():
            # We return empty arrays to have consistent return values:
            return self.no_pairs, self.no_distances

        if not index_with_primary:
            # The primary indices should be in the first row, the secondary
            # indices in the second:
            pairs[[0, 1]] = pairs[[1, 0]]

        return pairs, distances

    def _build_spatial_index(self, lat, lon):
        # Find out whether the cached index still works with the new points:
        if self._spatial_is_cached(lat, lon):
            self._debug("Spatial index is cached and can be reused")
            return self.index

        return GeoIndex(lat, lon, leaf_size=self.leaf_size)

    def _spatial_is_cached(self, lat, lon):
        """Return True if the cached ball tree is still applicable to the new
        data"""
        if self.index is None:
            return False

        try:
            return np.allclose(lat, self.index.lat) \
                   & np.allclose(lon, self.index.lon)
        except ValueError:
            # The shapes are different
            return False

    def _choose_points_to_build_index(self, primary, secondary):
        """Choose which points should be used for tree building

        This method helps to optimize the performance.

        Args:
            primary: Converted primary points
            secondary: Converted secondary points

        Returns:
            True if primary points should be used for tree building. False
            otherwise.
        """
        # There are two options to optimize the performance:
        # A) Cache the index and reuse it if either the primary or the
        # secondary points have not changed (that is the case for data with a
        # fixed grid). Building the tree is normally very expensive, so it
        # should never be done without a reason.
        # B) Build the tree with the larger set of points and query it with the
        # smaller set.
        # Which option should be used if A and B cannot be applied at the same
        # time? If the magnitude of one point set is much larger (by
        # `magnitude factor` larger) than the other point set, we strictly
        # follow B. Otherwise, we prioritize A.

        if primary[0].size > secondary[0].size * self.magnitude_factor:
            # Use primary points
            return True
        elif secondary[0].size > primary[0].size * self.magnitude_factor:
            # Use secondary points
            return False

        # Apparently, none of the datasets is much larger than the others. So
        # just check whether we still have a cached tree. If we used the
        # primary points last time and they still fit, use them again:
        if self.index_with_primary and self._spatial_is_cached(*primary):
            return True

        # Check the same for the secondary data:
        if not self.index_with_primary and self._spatial_is_cached(*secondary):
            return False

        # Otherwise, just use the larger dataset:
        return primary[0].size > secondary[0].size

    def temporal_search(self, primary, secondary, max_interval):
        raise NotImplementedError("Not yet implemented!")
        #return self.no_pairs, self.no_intervals

    def _temporal_check(
            self, primary_time, secondary_time, max_interval
    ):
        """Checks whether the current collocations fulfill temporal conditions

        Returns:

        """
        intervals = self._get_intervals(primary_time, secondary_time)

        # Check whether the time differences are less than the temporal
        # boundary:
        passed_time_check = intervals < max_interval

        return passed_time_check, intervals[passed_time_check]

    @staticmethod
    def _get_intervals(time1, time2):
        return np.abs((time1 - time2)).astype("timedelta64[s]")

    @staticmethod
    def _get_distances(lat1, lon1, lat2, lon2):
        return great_circle_distance(lat1, lon1, lat2, lon2)


def concat_collocations(collocations):
    """Concat compact collocations

    Compact collocations cannot be concatenated directly because indices in
    *Collocations/pairs* won't be correct any longer afterwards. This
    concatenate function fixes this problem.

    Args:
        collocations: A list of xarray.Dataset objects
            with compact collocations.

    Returns:
        One xarray.Dataset object
    """

    # We need to increment the pair indices when concatening the datasets
    primary = collocations[0]["Collocations/group"].item(0)
    secondary = collocations[0]["Collocations/group"].item(1)
    primary_size = 0
    secondary_size = 0
    collocation_coord = {
        "Collocations": "Collocations/collocation",
        primary: f"{primary}/collocation",
        secondary: f"{secondary}/collocation",
    }

    # Collect all collocations for each single group:
    groups = defaultdict(list)
    for obj in collocations:
        for group, data in get_xarray_groups(obj).items():
            if group == "Collocations":
                # Correct the indices:
                data["Collocations/pairs"][0, :] += primary_size
                data["Collocations/pairs"][1, :] += secondary_size
                data = data.drop("Collocations/group")
            groups[group].append(data)

        primary_size += obj.dims[f"{primary}/collocation"]
        secondary_size += obj.dims[f"{secondary}/collocation"]

    starts = []
    ends = []
    for group, data_list in groups.items():
        groups[group] = xr.concat(
                data_list,
                dim=collocation_coord[group]
            )

    start = pd.Timestamp(groups[primary][primary+"/time"].min().item(0))
    end = pd.Timestamp(groups[primary][primary+"/time"].max().item(0))
    merged = xr.merge(groups.values())
    merged.attrs = {
        "start_time": str(start),
        "end_time": str(end),
    }
    merged["Collocations/group"] = collocations[0]["Collocations/group"]
    return merged


class InvalidCollocationData(Exception):
    """Error when trying to collapse / expand invalid collocation data

    """

    def __init__(self, message, *args):
        Exception.__init__(self, message, *args)


def check_collocation_data(dataset):
    """Check whether the dataset fulfills the standard of collocated data

    Args:
        dataset: A xarray.Dataset object

    Raises:
        A InvalidCollocationData Error if the dataset did not pass the test.
    """
    mandatory_fields = ["Collocations/pairs", "Collocations/group"]

    for mandatory_field in mandatory_fields:
        if mandatory_field not in dataset.variables:
            raise InvalidCollocationData(
                f"Could not find the field '{mandatory_field}'!"
            )
