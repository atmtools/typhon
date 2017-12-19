"""Module containing classes abstracting datasets
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import abc
import functools
import itertools
import logging
import pathlib
import re
import shelve
import string
import shutil
import tempfile
import datetime
import sys
import collections
import warnings
import math

import numpy
import numpy.lib.arraysetops
import numpy.lib.recfunctions

import netCDF4

import xarray

from .. import config
from .. import utils
from .. import math as tpmath
from ..physics.units import em
from ..constants import MiB

from . import filters

try:
    import progressbar
except ImportError:
    progressbar = None

class GranuleLocatorError(Exception):
    """Problem locating granules.
    """

class DataFileError(Exception):
    """Superclass for any datafile problems

    Upon reading a large amounts of data, some files will contain
    problems.  When processing a year of data, we don't want to fail
    entirely when one orbit file fails.  But if there is a real bug
    somewhere, we probably do want to fail.  Therefore, the typhon dataset
    framework is optionally resilient to datafile related errors, but only
    if those derive from DataFileError.

    Therefore, users implementing their own reading routine may wish to
    catch errors arising from corrupted data, and raise an exception
    derived from DataFileError instead.  That way, typhon knows that a
    problem is due to corrupted data and not due to a bug.
    """

class InvalidFileError(DataFileError, ValueError):
    """Raised when the requested information cannot be obtained from the file

    See DataFileError for more information.
    """

class InvalidDataError(DataFileError):
    """Raised when data is not how it should be.

    See DataFileError for more information.
    """

all_datasets = {}

class Dataset(metaclass=utils.metaclass.AbstractDocStringInheritor):
    """Represents a dataset.

    This is an abstract class.  More specific subclasses are
    SingleFileDataset and MultiFileDataset.
    
    To add a dataset, subclass one of the subclasses of Dataset, such as
    MultiFileDataset, and implement the abstract methods.

    Dataset objects have a limited number of attributes.  To limit the
    occurence of bugs, dynamically setting non-pre-existing attributes is
    limited.  Attributes can be set either by passing keyword arguments
    when creating the object, or by setting the appropriate field in your
    typhon configuration file (such as .typhonrc).  The configuration
    section will correspond to the object name, the key to the attribute,
    and the value to the value assigned to the attribute.  See also
    :mod:`typhon.config`.

    Attributes:

        start_date (datetime.datetime or numpy.datetime64)
            Starting date for dataset.  May be used to search through ALL
            granules.  WARNING!  If this is set at a time t_0 before the
            actual first measurement t_1, then the collocation algorith (see
            CollocatedDataset) will conclude that there are 0 collocations
            in [t_0, t_1], and will not realise if data in [t_0, t_1] are
            actually added later!
        end_date (datetime.datetime or numpy.datetime64)
            Similar to start_date, but for ending.
        name (str)
            Name for the dataset.  Used to make sure there is only a
            single dataset with the same name for any particular dataset.
            If a dataset is initiated with a pre-exisitng name, the
            previous product is called.
        aliases (Mapping[str, str])
            Aliases for field.  Dictionary can be useful if you want to
            programmatically loop through the same field for many different
            datasets, but they are named differently.  For example, an alias
            could be "ch4_profile".
        unique_fields (Container[str])
            Set of fields that make any individual measurement unique.  For
            example, the default value is {"time", "lat", "lon"}.
        related (Mapping[str, Dataset])
            Dictionary whose keys may refer to other datasets with related
            information, such as DMPs or flags.

    """

    _instances = None

    start_date = None
    end_date = None
    name = ""
    section = ""
    aliases = {}
    time_field = "time"
    unique_fields = {"time", "lat", "lon"}
    mandatory_fields = None # fields that reader relies upon
    related = {}
    maxsize = 10000*MiB
    my_pseudo_fields = None
    read_returns = "ndarray"
    default_orbit_filters = []

    # in case of xarray returns, if concatenation is NOT along time
    # coordinate, this should be set to something else
    concat_coor = None

    # Make singleton: there is no point in having multiple copies of a
    # dataset-object around when both relate to the same dataset and
    # class.
    def __new__(cls, name=None, **kwargs):
        name = name or cls.name or cls.__name__
        if cls._instances is None:
            cls._instances = {}
        # make sure subclasses aren't confused for their parents
        if not cls in cls._instances:
            cls._instances[cls] = {}
        if name in cls._instances[cls]:
            return cls._instances[cls][name]
        self = super().__new__(cls)
        cls._instances[cls][name] = self
        if not cls in all_datasets:
            all_datasets[cls] = {}
        all_datasets[cls][name] = self
        return self # __init__ takes care of the rest

    # through __new__, I make sure there's only a single dataset
    # object per class with the same name.  However, __init__ will get
    # called again despite __new__ returning an older self.  Make sure
    # we don't get hurt by that.
    def __init__(self, **kwargs):
        """Initialise a Dataset object.

        All keyword arguments will be translated into attributes.
        Does not take positional arguments.

        Note that if you create a dataset with a name that already exists,
        the existing object is returned, but __init__ is still called
        (Python does this, see
        https://docs.python.org/3.5/reference/datamodel.html#object.__new__).
        """
        self.mandatory_fields = set()
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        self.setlocal()
        if self.my_pseudo_fields is None:
            self.my_pseudo_fields = collections.OrderedDict()

    def __setattr__(self, k, v):
        if hasattr(self, k) or hasattr(type(self), k):
            super().__setattr__(k, v)
        else:
            raise AttributeError("Unknown attribute: {}. ".format(k))

    def setlocal(self):
        """Set local attributes, from config or otherwise.

        """
        if self.section in config.conf:
            for k in config.conf[self.section]:
                setattr(self, k, config.conf[self.section][k])

    @abc.abstractmethod
    def find_granules(self, start=datetime.datetime.min,
                                  end=datetime.datetime.max,
                    include_last_before=False):
        """Loop through all granules for indicated period.

        This is a generator that will loop through all granules from
        `start` to `end`, inclusive.

        See also: `find_granules_sorted`

        Arguments:

            start (datetime.datetime): Start
                Starting datetime.  When omitted, start at complete
                beginning of dataset.

            end (datetime.datetime): End
                End datetime.  When omitted, continue to end of dataset.
                Last granule will start before this datetime, but contents
                may continue beyond it.

            include_last_before (bool): Be inclusive
                When True, also return the last granule /before/ start, so
                that a reader is sure to include all data in the covered
                period.  When False, the first granule yielded is the
                first granule starting after start.
                 
        Yields:
            pathlib.Path objects for all files in dataset.  Sorting is not
            guaranteed; if you need guaranteed sorting, use
            `find_granules_sorted`.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def find_granules_sorted(self, start=None, end=None):
        """Yield all granules sorted by starting time then ending time.

        For details, see `find_granules`.
        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def find_most_recent_granule_before(self, instant,
            **locator_args):
        """Find granule covering instant

        Find granule started most recently before `instant`.

        Arguments:

            instant (datetime.datetime): Time to search for
                datetime for which a granule is sought
                beginning of dataset.

            **locator_args:
                Any other keyword arguments that the particular dataset
                needs.  Commonly, `satname` is needed.
                 
        Returns:
            pathlib.Path object for sought granule.

        """
        ...

    # Cannot use functools.lru_cache because it cannot handle mutable
    # results or arguments
    @utils.cache.mutable_cache(maxsize=10)
    def read_period(self, start=None,
                          end=None,
                          onerror="skip",
                          fields="all",
                          pseudo_fields=None,
                          sorted=True,
                          locator_args=None,
                          reader_args=None,
                          limits=None,
                          simple_filters=(),
                          orbit_filters=None,
                          enforce_no_duplicates=True):
        """Read all granules between start and end, in bulk.

        Arguments:

            start (datetime.datetime): Start
                Starting datetime.  When omitted, start at complete
                beginning of dataset.

            end (datetime.datetime): End
                End datetime.  When omitted, continue to end of dataset.
                Last granule will start before this datetime, but contents
                may continue beyond it.  Can also be a datetime.timedelta,
                which will then be interpreted as a length after start.
 
            onerror (str): What to do on errors
                When reading many files, some files may have problems.  If
                onerror is set to "skip", files with errors are skipped
                and reading continues with the next file.  This is the
                default behaviour.  If onerror is set to "raise", the
                method will reraise the original exception as soon as a
                problem occurs.

            fields (Iterable[str] or str): What fields to return
                What fields to read from dataset.  Either a collection
                of strings corresponding to fields to read, or the str
                "all" (default), which means read all fields.

            pseudo_fields (Mapping[str, function]): See documentation for
                self.read.

            sorted (bool): Should the granules be read in sorted order?
                Defaults to true.  NB: does not currently guarantee that
                the actual results are sorted; this is up to the
                individual reading routines!

            locator_args (dict): Extra keyword arguments passed on to
                granule finding routines.

            reader_args (dict): Extra keyword arguments to be passed on to
                reading routine.  Since these differ per type, those are
                probably documented in the class docstring.

            limits (dict): Limitations to apply to each granule.  For the
                exact format, see `:func:typhon.math.array.limit_ndarray`.

            simple_filters (container/iterable): collection of functions to be
                applied for filtering.  Must take ndarray input, must give
                ndarray output.  For more advanced filtering, pass
                instances of implementations of filters.OrbitFilter to
                orbit_filters.
    
            orbit_filters (Sequence[filters.OrbitFilter]): more advanced
                filters to apply for each orbit or after all orbits.
                Those are instances of filters.OrbitFilter and thus have
                state, which can be useful for e.g. overlap checking.  For
                each of those, the system will call .reset() before the
                first orbit is read, .filter(...) after each orbit, and
                .finalise() at the end.

        Returns:
            
            Masked array containing all data in period.  Invalid data may
            or may not be masked, depending on the behaviour of the
            reading routine implemented by the subclass.
        """

        if locator_args is None:
            locator_args = {}

        if reader_args is None:
            reader_args = {}

        if limits is None:
            limits = {}

        if orbit_filters is None:
            orbit_filters = self.default_orbit_filters

        start = start or self.start_date
        end = end or self.end_date
        if isinstance(end, datetime.timedelta):
            end = start + end

        finder = self.find_granules_sorted if sorted else self.find_granules
        logging.info("Reading {self.name:s} {locator_args!s} "
                     "for period {start:%Y-%m-%d %H:%M:%S} "
                     " – {end:%Y-%m-%d %H:%M:%S}".format(**vars()))

        dobar = sorted and progressbar and sys.stdout.isatty()
        if dobar:
            bar = progressbar.ProgressBar(max_value=1,
                    widgets=[progressbar.Bar("=", "[", "]"), " ",
                             progressbar.Percentage(),
                             ' (', progressbar.ETA(), ') '])
            bar.start()
            bar.update(0)
        else:
            logging.info("Psst!  If you install the progressbar2 package, "
                "you will get a fancy progressbar!")
        anygood = False
        arr = None
        N = 0
        time = self.time_field
        for of in orbit_filters:
            of.reset()
        # NB: https://stackoverflow.com/a/37585628/974555
        extra_filter_args = collections.ChainMap(*(of.args_to_reader
            for of in orbit_filters))

        overlap_filters = [
            f for f in orbit_filters if isinstance(f, filters.OverlapFilter)]
        late = overlap_filters[0].late if len(overlap_filters) > 0 else True
        if len(overlap_filters) > 1:
            raise ValueError("Found {:d} overlap filters: {!s}".format(
                len(overlap_filters), overlap_filters))
        for (g_start, gran) in finder(start, end, return_time=True, 
                                      include_last_before=True,
                                      **locator_args):
            try:
                (cont, extra) = self.read(str(gran), fields=fields,
                    pseudo_fields=pseudo_fields, **reader_args,
                    **extra_filter_args)
                for of in orbit_filters:
                    cont = of.filter(cont, **extra)
                # FIXME: handle cases where very few scanlines are left
                # after filtering.  We may find errors downstream if there
                # are very few scanlines left.
                if (not late and
                    cont[time].size > 0):

                    if (enforce_no_duplicates and sorted and
                            arr is not None and
                            N>0 and (cont[time][0] <= latest)):
                        if isinstance(latest, xarray.DataArray):
                            latest = latest.values.astype("M8[ms]")
                        if isinstance(cont[time][0], xarray.DataArray):
                            conttime = cont[time][0].values.astype("M8[ms]")
                        else:
                            conttime = cont[time][0]
                        raise InvalidDataError(
                            "Reading routine for {!s} returned data starting "
                            "{:%Y-%m-%d %H:%M:%S}, which precedes last entry "
                            "for preceding granule at {:%Y-%m-%d %H:%M:%S}. "
                            "As data are supposed to be sorted in time, this "
                            "probably means duplicate removal is not working "
                            "as it should, or you are using an implementation "
                            "that deliberately leaves duplicates in, and you "
                            "should be passing enforce_no_duplicates=False to "
                            "read_period.".format(gran,
                                conttime.astype(datetime.datetime),
                                latest.astype(datetime.datetime)))

                    # NB: when datasets erroneously contain duplicate coordinates
                    # (I'm looking at you, FIDUCEO/FCDR_HIRS#159!), this
                    # comparison will cause a failure in xarray.  In this case,
                    # compare values instead.
                    arrrr = cont[time]
                    if isinstance(arrrr, xarray.DataArray):
                        arrrr = arrrr.values
                    if not (arrrr[1:] >= arrrr[:-1]).all():
                        raise InvalidDataError("Reader for {!s} returned data "
                            "with unsorted time.  This must be fixed.".format(
                                f))

                cont = self._apply_limits_and_filters(cont, limits, simple_filters)

            except (DataFileError, filters.FilterError) as exc:
                if onerror == "skip": # fields that reader relies upon
                    logging.error("Can not read file {}: {}".format(
                        gran, exc.args[0]))
                    continue
                else:
                    raise
            else:
                (arr, N, latest) = self._add_gran_to_data(arr, cont, N, start,
                                     end, g_start)
#                contents.append(
#                    cont[(cont[time]<=end)&(cont[time]>=start)])
                if N > 0:
                    anygood = True
            if dobar:
                bar.update(max((g_start-start) / (end-start),0))
        if dobar:
            bar.update(1)
            bar.finish()
        if anygood:
            # NB: filter finalise before or after dataset finalise?
            #
            arr = self._finalise_arr(arr, N)
            for of in orbit_filters:
                arr = of.finalise(arr)

            if "flags" in self.related:
                arr = self.flag(arr)
                    
            return arr
        else:
            raise DataFileError("Can not find any valid data!")

    def _apply_limits_and_filters(self, cont, limits, simple_filters):
        if isinstance(cont, xarray.Dataset):
            if len(limits)>0:
                raise NotImplementedError( 
                    "limits not implemented on xarray datasets")
            oldsize = cont[self.time_field].size
            for f in simple_filters:
                cont = f(cont)
            logging.debug("Filters reduced number from "
                "{:d} to {:d}".format(oldsize, cont[self.time_field].size))
            return cont
        oldsize = cont.size
        cont = tpmath.array.limit_ndarray(cont, limits)
        for f in simple_filters:
            cont = f(cont)
        if cont.size < oldsize:
            logging.debug("Applying limitations, reducing "
                "{:d} to {:d}".format(oldsize, cont.size))
        return cont

    def _add_gran_to_data(self, arr, cont, N, start, end, g_start):
        # first ensure we only add the segment we actually want
        if isinstance(cont, xarray.Dataset):
            tm = cont[self.time_field].values.astype("M8[ms]")
            cont = cont.loc[dict.fromkeys(
                utils.get_time_coordinates(cont)&cont.dims.keys(),
                slice(start, end))]
        else:
            cont = cont[(cont[self.time_field]<end)&(cont[self.time_field]>=start)]
        if isinstance(cont, xarray.Dataset):
            # xarray reads lazily, but we want to merge all granules into
            # a single object at the end, so have to load it into memory
            # anyway; rather do it now than all at once later, so that we
            # can keep track of how fast things are going
            if cont[self.time_field].size > 0: # avoid https://github.com/pydata/xarray/issues/1329
                cont.load()
            if arr is None:
                arr = [cont]
                N = cont[self.time_field].size
            else:
                arr.append(cont)
                N += cont[self.time_field].size
            for i in range(len(arr)):
                if arr[-i][self.time_field].size > 0:
                    latest = arr[-1][self.time_field][-1]
                    break
            else:
                latest = numpy.datetime64(0, "ms")
        else:
            if arr is None:
                arr = cont
                N = cont.size
            else:
                if (N+cont.size) > arr.size: # need to allocate more
                    frac_done = max((g_start-start) / (end-start),0)
                    # suppose all future files have on average the
                    # same size?  Until we have reached 10%, simply
                    # double every time.  After that, extrapolate more
                    # cleverly. 
                    newsize = (int((N+cont[self.time_field].size)//frac_done)
                                if frac_done > 0.1
                                else (N+cont[self.time_field].size) * 2)
                    arr = self._ensure_large_enough(arr, cont, N, newsize,
                        frac_done)
                self._add_cont_to_arr(arr, N, cont)
                N += cont[self.time_field].size
            if arr[self.time_field].size > 0:
                latest = arr[self.time_field][N-1]
            else:
                latest = numpy.datetime64(0, "ms")
        return arr, N, latest

    def _ensure_large_enough(self, arr, cont, N, newsize, frac_done):
        """Allocate new space while adding gran to data
        
        Helper for _add_gran_to_data, part of the read_period family of
        helpers.  Does NOT add cont to arr!"""
        
        if isinstance(cont, xarray.Dataset):
            raise NotImplementedError("Not used for xarray datasets. "
                "But see version history at "
                "https://arts.mi.uni-hamburg.de/trac/rt/browser/typhon/trunk/typhon/datasets/dataset.py?rev=10396#L462 "
                "if there is a wish to reimplemented!")
        else:
            if newsize * arr.itemsize > self.maxsize:
                raise MemoryError("This dataset is too large "
                    "for typhons little mind.  Continuing might "
                    "ultimately need {:,.0f} MiB of RAM.  This exceeds my "
                    "maximum (self.maxsize) of {:,.0f} MiB. "
                    "Sorry! ".format(
                        newsize*arr.itemsize/MiB,
                        self.maxsize/MiB))
            logging.debug(
                "New size ({:d} items, {:,.0f} MiB) would exceed allocated "
                "size ({:d} items, {:,.0f} MiB).  I'm {:.3%} "
                "through.  Allocating new: {:d} items, {:,.0f} "
                "MiB.  New size: {:d} items, {:,.0f} "
                "MiB.".format(N+cont.size,
                    (cont.nbytes+arr.nbytes)/MiB,
                    arr.size, arr.nbytes/MiB, frac_done,
                    newsize-arr.size, (newsize-arr.size)*arr.itemsize/MiB,
                    newsize, newsize*arr.itemsize/MiB))
            mod = (numpy.ma if hasattr(arr, "mask") else numpy)
            arr = mod.concatenate(
                (arr, mod.zeros(dtype=arr.dtype, shape=newsize-arr.size)))
        return arr

    def _add_cont_to_arr(self, arr, N, cont):
        """Changes arr in-situ, does not return"""
        if isinstance(cont, xarray.Dataset):
            # we should already know it's large enough
            # for arr[self.time_field] I start at N
            # for the other time coordinates at the relative "speed" they
            # are behind N
            # but this is not guaranteed to be regular so I would need to
            # keep trac of each individually, or inspect it on-the-fly
            # this approximation may be good enough for pre-allocation
            # (which is approximate anyway), when actually storing we need
            # to do a better job… for each time coordinate, check when it
            # “dies”
            raise NotImplementedError("This is not used for xarrays. "
                "But see comment in source-code for some thoughts.")
        else:
            arr[N:(N+cont.size)] = cont
            #arr = self._finalise_arr(arr, N)

    def _finalise_arr(self, arr, N):
        if isinstance(arr, list):
            logging.debug("Concatenating {N:d} DataArrays...".format(N=N))
            if self.concat_coor is None:
                return utils.concat_each_time_coordinate(*arr)
            else:
                return xarray.concat(arr, dim=self.concat_coor)
            logging.debug("Done!")
        else:
            return self._correct_overallocation(arr, N)

    @staticmethod
    def _correct_overallocation(arr, N):
        if isinstance(arr, xarray.Dataset):
            raise RuntimeError("We shouldn't be here.  Ever.")
        logging.debug("Correcting overallocation ({:d}->{:d})".format(
            arr.size, N))
        return arr[:N]

    @abc.abstractmethod
    def _read(self, f, fields="all"):
        """Read granule in file, low-level.

        To be implemented by subclass.  Do not call this method; call
        :func:`Dataset.read` instead.

        For documentation, see :func:`Dataset.read`.
        """

        raise NotImplementedError()

    # Cannot use functools.lru_cache because it cannot handle mutable
    # results or arguments
    @utils.cache.mutable_cache(maxsize=10)
    def read(self, f=None, fields="all", pseudo_fields=None, **kwargs):
        """Read granule in file and do some other fixes

        Shall return an ndarray with at least the fields lat, lon, time.

        Arguments:

            f (str): String containing path to file to read.

            fields (Iterable[str] or str): What fields to return.
                See :func:`Dataset.read_period` for details.

            pseudo_fields (Mapping[str, function]): Additional fields to
                calculate on-the-fly after every read.  That may be more
                attractive from a memory point of view.  In this mapping,
                the keys will be names added to the dtype of the returned
                ndarray (at the top level).  The values are functions.
                Each function must take a 

            Any further keyword arguments are passed on to the particular
            reading routine.  For details, please refer to the docstring
            for the class.

        Returns:

            Masked array containing data in file with selected fields.
        """
        if isinstance(f, pathlib.PurePath):
            f = str(f)
        if pseudo_fields is None:
            pseudo_fields = collections.OrderedDict()
        # expand pseudo_fields with items from my_pseudo_fields not
        # already there, and also process dependent fields
        if fields != "all":
            fields = list(fields)
            for (k, (deps, v, cond)) in self.my_pseudo_fields.items():
                if k in fields:
                    for d in deps:
                        if (d not in fields and
                                all(kwargs.get(condfn) in condval for (condfn,
                                condval) in cond.items())):
                            fields.append(d)
            for mandatory in self.mandatory_fields:
                if mandatory not in fields:
                    fields.append(mandatory)
        for (k, (deps, v, cond)) in self.my_pseudo_fields.items():
            if (fields=="all" or 
                    k in fields and k not in pseudo_fields) and (
                    all(kwargs.get(condfn) in condval
                            for (condfn, condval) in cond.items())):
                pseudo_fields[k] = v
        logging.debug("Reading {:s}".format(f))
        # should not pass pseudo_fields on to reader, it will get confused
        fields = (fields if fields == "all" else
                  [f for f in fields if not f in pseudo_fields])
        (M, extra) = (self._read(f, fields=fields, **kwargs)
                if f is not None
                else self._read(fields=fields, **kwargs))
        M = self._add_pseudo_fields(M, pseudo_fields)
        return (M, extra)

    def _add_pseudo_fields(self, M, pseudo_fields):
        D = collections.OrderedDict()
        for (k, fnc) in pseudo_fields.items():
            try:
                D[k] = fnc(M, D)
            except TypeError as exc:
                if "positional argument" in exc.args[0]:
                    D[k] = fnc(M)
                else:
                    raise
        if D != {}:
            if self.read_returns == "ndarray":
                mod = (numpy.ma if hasattr(M, "mask") else numpy)
                newM = mod.zeros(shape=M.shape,
                    dtype=M.dtype.descr + [(k, v.dtype, v.shape[1:]) for (k,v) in D.items()])
                for k in M.dtype.names:
                    newM[k] = M[k]
            elif self.read_returns == "xarray":
                newM = M
            else:
                raise ValueError("read_returns must be ndarray or xarray")

            for (k, v) in D.items():
                newM[k] = v
                
            M = newM
        return M

    def __str__(self):
        return "<{self.__class__.__name__:s}:{self.name:s}>".format(self=self)

#   Leave disk-memoisation out of typhon until dependency on joblib has
#   been decided.
#
#    @tools.mark_for_disk_cache(
#        process=dict(
#            my_data=lambda x: x.view(dtype="i1")))
    def combine(self, my_data, other_obj, other_data=None, other_args=None, trans=None,
                timetol=numpy.timedelta64(1, 's'), time_name="time"):
        """Combine with data from other dataset.

        Combine a set of measurements from this dataset with another
        dataset, where each individual measurement correspond to exactly
        one from the other one, as identified by time/lat/lon, orbitid, or
        measurument id, or other characteristics.  The object attribute
        unique_fields determines how those are found.

        The other dataset may contain flags, DMPs, or different
        information altogether.

        Arguments:
        
            my_data (ndarray): Data for self.
                A (masked) array with a dtype such as returned from
                `self.read <Dataset.read>`.

            other_obj (Dataset): Dataset to match
                Object from a Dataset subclass from which to find matching
                data.

            other_data (ndarray): Data for other.  Optional.
                Optionally, pass data for other object.  If not provided
                or None, this will be read using other_obj.

            other_args (dict): Keyword arguments passed to
                other_obj.read_period.  May need to contain things like
                {"locator_args": {"satname": "noaa18"}}

            trans (collections.OrderedDict): Dictionary of what field in `my_data`
                corresponds to what field in `other_data`.  Optional; by
                default, merges self.unique_fields and
                other_obj.unique_fields, and assumes names between the two
                are identical.  Order is relevant for optimal recursive
                bisection search for matches, which is to be implemented.
    
            timetol (timedelta64): For datetime types, `isclose` does not
                work (https://github.com/numpy/numpy/issues/5610).  User
                must pass an explicit tolerance, defaulting to 1 second.

            time_name (str): Name for my time field.  Defaults to "time".
                Used to determine the period to read from the other
                dataset.

        Returns:

            Masked ndarray of same size as `my_data` and same `dtype` as
            returned by `other_obj.read`.

        TODO: Allow user to pass already-read data from other dataset.
        """

        if trans is None:
            flds = list(self.unique_fields & other_obj.unique_fields)
            trans = collections.OrderedDict(zip(flds, flds))

        if other_args is None:
            other_args = {}

        if self.read_returns == "ndarray":
            if other_data is None:
                first = my_data[time_name].min().astype(datetime.datetime)
                last = my_data[time_name].max().astype(datetime.datetime)

        elif self.read_returns == "xarray":
            if other_data is None:
                first = my_data[time_name].min().values.astype("M8[ms]"
                    ).astype(datetime.datetime)
                last = my_data[time_name].max().values.astype("M8[ms]"
                    ).astype(datetime.datetime)
        else:
            raise ValueError("read_returns must be ndarray or xarray")

        if other_data is None:
            other_data = other_obj.read_period(first, last,
                            **other_args)
        (my_prim, other_prim) = next(iter(trans.items()))

        if self.read_returns == "ndarray":
            # my_data needs to be sorted
            try:
                my_data._fill_value
            except AttributeError:
                my_data.sort(order=my_prim, kind="heapsort")
            else:
                # see 
                # https://github.com/numpy/numpy/issues/8069
                my_data.sort(order=my_prim, kind="heapsort",
                                    fill_value=my_data._fill_value)

        elif self.read_returns == "xarray":

            # ensure sorted along time dimension
            if not len(my_data[my_prim].dims) == len(other_data[other_prim].dims) == 1:
                raise ValueError(
                    "Must use 1-D index for finding combinations, "
                    "{:s} has {:d} dimensions: {!s}, "
                    "{:s} has {:d} dimensions: {!s}.".format(
                        my_prim, len(my_data[my_prim].dims),
                        my_data[my_prim].dims,
                        other_prim, len(other_data[other_prim].dims),
                        other_data[other_prim].dims))
            my_dim = my_data[my_prim].dims[0]
            other_dim = other_data[other_prim].dims[0]
            my_data = my_data.loc[
                {my_data[my_prim].dims[0]:
                 numpy.argsort(my_data[my_prim])}]

        # through interpolation, find times in `other` closest to times in
        # myself; hopefully that means the difference is zero
        if issubclass(my_data[my_prim].dtype.type, numpy.datetime64):
            x = other_data[other_prim].astype("<M8[ms]").astype("f8")
            xx = my_data[my_prim].astype("<M8[ms]").astype("f8")
        else:
            x = other_data[other_prim]
            xx = my_data[my_prim]

        if self.read_returns == "ndarray":
            ii = numpy.interp(xx, x,
                numpy.arange(other_data[other_prim].shape[0])).round().astype(
                    numpy.uint64)
        else:
            if not other_data[other_prim].dtype.kind=="M":
                raise ValueError("When finding combinations based "
                    "on xarray datasets, I can only do so based on "
                    "time coordinates, but {:s} has dtype {!s}.".format(
                        other_prim, other_data[other_prim].dtype))
            for (k, v) in other_data.data_vars.items():
                if v.dtype.kind == "M":
                    raise ValueError("Found time data variable "
                        "that is not a coordinate, I'm confused, "
                        "should I interpolate it?  Assign as a coordinate "
                        "to be sure!")
            ii = dict.fromkeys(utils.get_time_dimensions(other_data))
            for k in ii.keys():
                x = other_data[k].astype("<M8[ms]").astype("f8")
                ii[k] = numpy.interp(xx, x,
                    numpy.arange(x.shape[0])).round().astype(numpy.int64)

#        if self.read_returns == "ndarray":
#            other_combi = other_data[ii]
#            # now go through fields to see that there is an actual match
#        elif self.read_returns == "xarray":
        other_combi = other_data[ii]

        if self.read_returns == "ndarray":
            near = numpy.all([(numpy.isclose(my_data[f_my], other_combi[f_oth])
                    if issubclass(my_data[f_my].dtype.type,
                                  numpy.inexact) else
                    abs(my_data[f_my] - other_combi[f_oth]) < timetol
                    if issubclass(my_data[f_my].dtype.type,
                                  numpy.datetime64) else
                    my_data[f_my] == other_combi[f_oth])
                        for (f_my, f_oth) in trans.items()], 0)
        elif self.read_returns == "xarray":
            near = numpy.all([
                (numpy.isclose(my_data[f_my].values,
                               other_combi[f_oth].values)
                    if issubclass(my_data[f_my].dtype.type,
                                  numpy.inexact) else
                    abs(my_data[f_my].values -
                        other_combi[f_oth].values) < timetol
                    if issubclass(my_data[f_my].dtype.type,
                                  numpy.datetime64) else
                    my_data[f_my].values == other_combi[f_oth].values)
                        for (f_my, f_oth) in trans.items()], 0)

        if not near.any():
            raise ValueError("Did not find any secondaries!  Are times off?")

        if not near.all():
            logging.warn("Only {:d}/{:d} ({:%}) of secondaries found".format(
                near.sum(), near.size, near.sum()/near.size))

        if self.read_returns == "ndarray":
            try:
                other_combi.mask[~near] = numpy.ones_like(other_combi.mask[~near])
            except AttributeError: # not a MaskedArray yet
                other_combi = numpy.ma.masked_where(~near, other_combi)
        elif self.read_returns == "xarray":
            vars_with_dim = [k for k in other_combi.data_vars.keys()
                             if other_dim in other_combi[k].dims]
            for v in vars_with_dim:
                other_combi[v][{"time": ~near}] = numpy.nan
            
        return other_combi

    def get_additional_field(self, M, fld):
        """Get additional field.

        Get field from other dataset, original objects, or otherwise.
        To be implemented by subclass implementations.

        Exact fields depend on subclass.

        Arguments:

            M (ndarray): ndarray with existing data
                A (masked) array with a dtype such as returned from
                `self.read <Dataset.read>`.

            fld (str): Additional field to read from original data

        Returns:
            
            ndarray with fields of M + fld.
        """
        raise NotImplementedError("Must be implemented by child-class")

    def as_xarray_dataset(self):
        raise NotImplementedError("Dataset {!s} has not implemented "
            "conversion to xarray".format(self))

class SingleFileDataset(Dataset):
    """Represents a dataset where all measurements are in one file.
    """

    start_date = datetime.datetime.min
    end_date = datetime.datetime.max
    srcfile = None

    # docstring in parent
    def find_granules(self, start=datetime.datetime.min,
                            end=datetime.datetime.max,
                            include_last_before=False):
        if start < self.end_date and end > self.start_date:
            yield self.srcfile

    # docstring in parent
    def find_granules_sorted(self, start=datetime.datetime.min,
                                   end=datetime.datetime.max,
                                   include_last_before=False):
        yield from self.find_granules(start, end)

    def get_times_for_granule(self, gran=None):
        return (self.start_date, self.end_date)

    def read(self, f=None, fields="all"):
        return super().read(f or self.srcfile, fields)

    def find_most_recent_granule_before(self, instant,
            **locator_args):
        if instant > self.start_date:
            yield from self.find_granules(**locator_args)
        else:
            raise GranuleLocatorError("Instant out of range: "
                "{:%Y-%m-%d %H:%M:%S}".format(instant))

class MultiFileDataset(Dataset):
    """Represents a dataset where measurements are spread over multiple
    files.

    If filenames contain timestamps, this information is used to determine
    the time for a granule or measurement.  If filenames do not contain
    timestamps, this information is obtained from the file contents.

    Attributes:

        basedir (pathlib.Path or str):
            Describes the directory under which all granules are located.
            Can be either a string or a pathlib.Path object.

        subdir (pathlib.Path or str):
            Describes the directory within basedir where granules are
            located.  May contain string formatting directives where
            particular fields are replaces, such as `year`, `month`, and
            `day`.  For example: `subdir = '{year}/{month}'`.  Sorting
            cannot be more narrow than by day.

        re (str):
            Regular expression that should match valid granule files within
            `basedir` / `subdir`.  Should use symbolic group names to capture
            relevant information when possible, such as starting time, orbit
            number, etc.  For time identification, relevant fields are
            contained in MultiFileDataset.date_info, where each field also
            exists in a version with "_end" appended.
            MultiFileDataset.refields contains all recognised fields.

            If any _end fields are found, the ending time is equal to the
            beginning time with any _end fields replaced.  If no _end
            fields are found, the `granule_duration` attribute is used to
            determine the ending time, or the file is read to get the
            ending time (hopefully the header is enough).

        granule_cache_file (pathlib.Path or str):
            If set, use this file to cache information related to
            granules.  This is used to cache granule times if those are
            not directly inferred from the filename.  Otherwise, this is
            not used.  The full path to this file shall be `basedir` /
            `granule_cache_file`.

        granule_duration (datetime.timedelta):
            If the filename contains starting times but no ending times,
            granule_duration is used to determine the ending time.  This
            should be a datetime.timedelta object.
    """
    basedir = None
    subdir = ""
    re = None
    _re = None # compiled version, do not touch
    granule_cache_file = None
    _granule_start_times = None
    granule_duration = None

    datefields = "year month day hour minute second".split()
    # likely extended later.  Note that "tod" is also a datefield,
    # interpreted as seconds since 00
    refields = ["".join(x)
        for x in itertools.product(datefields, ("", "_end"))]
    valid_field_values = {}

    # When set to a string, interpreted as a path to database of granules
    # with corresponding first lines.  This may be relevant for cases like
    # the NOAA and MetOp satelline sensors, where subsequent granules
    # contain some repetitions of lines from previous granules.
    granules_firstline_file = None
    granules_firstline_db = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._str2path("basedir", "subdir", "granule_cache_file")
        self._open_granule_file()
        if self.re is not None:
            self._re = re.compile(self.re)

    def _str2path(self, *args):
        for attr in args:
            if (getattr(self, attr) is not None and
                    not isinstance(getattr(self, attr), pathlib.PurePath)):
                setattr(self, attr, pathlib.Path(getattr(self, attr)))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_granule_start_times"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_granule_file()

    def find_dir_for_time(self, dt):
        """Find the directory containing granules/measurements at (date)time

        For a given datetime object, find the directory that contains
        granules/measurument files for this particular time.

        Arguments:
            
            dt (datetime.datetime): Timestamp for inquiry.
                In reality, any object with `year`, `month`, and `day`
                attributes works.

        Returns:

            pathlib.Path object pointing to to relevant directory
        """
        return pathlib.Path(str(self.basedir / self.subdir).format(
            year=dt.year, month=dt.month, day=dt.day,
            doy=dt.timetuple().tm_yday))

    def get_mandatory_fields(self):
        fm = string.Formatter()
        fields = {f[1] for f in fm.parse(str(self.basedir / self.subdir))}
        if fields == {None}:
            fields = set()
        return fields - set(self.datefields)

    def verify_mandatory_fields(self, extra):
        mandatory = self.get_mandatory_fields()
        found = set(extra.keys())
        missing = set()
        if not mandatory <= found:
            missing = mandatory - found
            # I'm deleting elements so I must loop over a copy
            for elem in missing.copy():
                try:
                    extra[elem] = getattr(self, elem)
                except AttributeError:
                    pass
                else:
                    missing.remove(elem)
        if missing:
            raise GranuleLocatorError("Missing fields needed to search for "
                "{self.name:s} files.  Need: {mandatory:s}. "
                "Found: {found:s}.  Missing: {missing:s}".format(
                    self=self,
                    mandatory=", ".join(mandatory),
                    found=", ".join(found) or "(none)",
                    missing=", ".join(missing)))
        for field in found:
            if field in self.valid_field_values:
                if not extra[field] in self.valid_field_values[field]:
                    raise GranuleLocatorError("Encountered invalid {field!s} "
                        "when searching for {self.name!s} files. "
                        "Found: {value!s}. Valid: "
                        "{valid_values!s}".format(
                            field=field, self=self, extra=extra,
                            value=extra[field],
                            valid_values=self.valid_field_values[field]))

    def get_subdir_resolution(self):
        """Return the resolution for the subdir precision.

        Returns "year", "month", "day", or None (if there is no subdir).

        Based on parsing of self.subdir attribute.
        """
        fm = string.Formatter()
        fields = {f[1] for f in fm.parse(str(self.subdir))}
        if "day" in fields:
            return "day"
        if "month" in fields:
            if "doy" in fields:
                raise ValueError("Format string has both month and doy")
            return "month"
        if "year" in fields:
            if "doy" in fields:
                return "day"
            return "year"

    def get_path_format_variables(self):
        """What extra format variables are needed in find_granules?

        Depending on the dataset, `find_granules` needs zero or more extra
        formatting valiables.  For example, TOVS instruments require the
        satellite.  Required formatting arguments are determined by
        self.basedir and self.subdir.
        """
        return {x[1] for x in string.Formatter().parse(
                    str(self.basedir / self.subdir))} - {"year", "month",
                                                         "day"}

    def iterate_subdirs(self, d_start, d_end, **extra):
        """Iterate through all subdirs in dataset.

        Note that this does not check for existance of those directories.

        Yields a 2-element tuple where the first contains information on
        year(/month/day), and the second is the path.

        Arguments:

            d_start (datetime.date): Starting date.
            d_end (datetime.date): Ending date
            **extra: Any extra keyword arguments.  This will be passed on to
                format self.basedir / self.subdir, in case the standard fields
                like year, month, etc. do not provide enough information.

        Yields:

            pathlib.Path objects for each directory in the dataset
            containing files between `d_start` and `d_end`.
        """

        self.verify_mandatory_fields(extra)
        # depending on resolution, iterate by year, month, or day.
        # Resolution is determined by provided fields in self.subdir.
        d = d_start
        res = self.get_subdir_resolution()

        pst = str(self.basedir / self.subdir)
        if res == "year":
            year = d.year
            while datetime.date(year, 1, 1) <= d_end:
                yield (dict(year=year),
                    pathlib.Path(pst.format(year=year, **extra)))
                year += 1
        elif res == "month":
            year = d.year
            month = d.month
            while datetime.date(year, month, 1) <= d_end:
                yield (dict(year=year, month=month),
                    pathlib.Path(pst.format(year=year, month=month,
                                            **extra)))
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
        elif res == "day":
            #while d < d_end:
            if any(x[1] == "doy" for x in string.Formatter().parse(pst)):
                while d <= d_end:
                    doy = d.timetuple().tm_yday
                    yield (dict(year=d.year, doy=doy),
                        pathlib.Path(pst.format(year=d.year, doy=doy,
                                                **extra)))
                    d += datetime.timedelta(days=1)
            else:
                while d <= d_end:
                    yield (dict(year=d.year, month=d.month, day=d.day),
                        pathlib.Path(pst.format(
                            year=d.year, month=d.month, day=d.day,
                            **extra)))
                    d = d + datetime.timedelta(days=1)
        else:
            yield ({}, pathlib.Path(pst.format(**extra)))
          
    def find_granules(self, dt_start=None, dt_end=None, 
            include_last_before=False, **extra):
        """Yield all granules/measurementfiles in period

        Accepts extra keyword arguments.  Meaning depends on actual
        dataset.  Could be something like a satellite name in the case of
        sensors occurring on multiple platforms, like HIRS.  To see what
        keyword arguments are accepted or possibly needed for a particular
        dataset, call self.get_path_format_variables()

        If keyword argument `return_time` is present and True, yield
        tuples of (start_time, path) rather than just `path`.

        The results are usually sorted by start time, but this is not
        guaranteed and depends on the filesystem.  If you need sorted
        granules, please use find_granules_sorted.

        Arguments:

            d_start (datetime.date): Starting date.
            d_end (datetime.date): Ending date
            include_last_before (bool): Include last granule starting
                before.
            **extra: Any extra keyword arguments.  This will be passed on to
                format self.basedir / self.subdir, in case the standard fields
                like year, month, etc. do not provide enough information.

        Yields:

            pathlib.Path objects for each datafile in the dataset between
            `dt_start` and `dt_end`.
        """

        if dt_start is None:
            dt_start = self.start_date

        if dt_end is None:
            dt_end = self.end_date

        return_time = extra.pop("return_time", False)

        self.verify_mandatory_fields(extra)

        d_start = (dt_start.date()
                if isinstance(dt_start, datetime.datetime) 
                else dt_start)
        d_end = (dt_end.date()
                if isinstance(dt_end, datetime.datetime) 
                else dt_end)
        found_any_dirs = False
        found_any_grans = False
        logging.debug(("Searching for {!s} granules between {!s} and {!s} "
                      ).format(self.name, dt_start, dt_end))
        before = None
        if include_last_before and dt_start > self.start_date:
            try:
                before = self.find_most_recent_granule_before(dt_start, 
                        return_time=return_time, **extra)
                yield before
            except GranuleLocatorError: # no problem
                pass

        for (timeinfo, subdir) in self.iterate_subdirs(d_start, d_end,
                                                       **extra):
            if subdir.exists() and subdir.is_dir():
                logging.debug("Searching directory {!s}".format(subdir))
                found_any_dirs = True
                for child in subdir.iterdir():
                    if before is not None and child == before[1]: # already yielded it as "last before"
                        continue
                    m = self._re.fullmatch(child.name)
                    if m is not None:
                        fit = True
                        for k in m.groupdict().keys() & extra.keys():
                            if m[k] != extra[k]:
                                fit = False # not the right file
                                break
                        if not fit:
                            continue
                        found_any_grans = True
                        try:
                            (g_start, g_end) = self.get_times_for_granule(child,
                                **timeinfo)
                        except InvalidFileError as e:
                            logging.error(
                                "Skipping {!s}.  Problem: {}".format(
                                    child, e.args[0]))
                            continue
                        if g_end >= dt_start and g_start <= dt_end:
                            if return_time:
                                yield (g_start, child)
                            else:
                                yield child
        if not found_any_dirs:
            logging.warning("Found no directories.  Make sure {self.name:s} has "
                  "coverage in {dt_start:%Y-%m-%d %H:%M:%S} – "
                  "{dt_end:%Y-%m-%d %H:%M:%S} and that you spelt any "
                  "additional information correctly: ".format(self=self,
                  dt_start=dt_start, dt_end=dt_end) + str(extra) + " Satellite "
                  "names and other fields are case-sensitive!")
        elif not found_any_grans:
            logging.warning("Directories searched appear to contain no matching "
                  "files.  Make sure basedir, subdir, and regexp are "
                  "correct and you did not misspell any extra "
                  "information: " + str(extra))
            
    def find_granules_sorted(self, dt_start=None, dt_end=None, 
                include_last_before=False, **extra):
        """Yield all granules, sorted by times.

        For documentation, see :func:`~Dataset.find_granules`.
        """

        allgran = list(self.find_granules(dt_start, dt_end, 
                       include_last_before, **extra))

        # I've been through all granules at least once, so all should be
        # cached now; no need for additional hints when granule timeinfo
        # obtainable only with hints from subdir, which is not included in
        # the re-matching method
        if extra.get("return_time", False):
            yield from sorted(allgran)
        else:
            yield from sorted(allgran, key=self.get_times_for_granule)

    def find_most_recent_granule_before(self, instant, **locator_args):
        # docstring in parent class
        return_time = locator_args.pop("return_time", False)
        res = self.get_subdir_resolution()
        if res == "day":
            d0 = datetime.datetime(instant.year, instant.month, instant.day,
                                   0, 0, 0)
            d1 = d0 + datetime.timedelta(days=2) # closed interval
            d0 = d0 - datetime.timedelta(days=1) # granule may start yesterday
        elif res == "month":
            d0 = datetime.datetime(instant.year, instant.month, 1,
                                   0, 0, 0)
            d1 = d0 + datetime.timedelta(days=31*2)
            d0 = d0 - datetime.timedelta(days=31)
        elif res == "year":
            d0 = datetime.datetime(instant.year-1, 1, 1,
                                   0, 0, 0)
            d1 = datetime.datetime(instant.year+1, 12, 31, 23, 59, 59)
        else:
            d0 = d1 = None # search entire dataset

        first = True
        for (time_st, gran) in self.find_granules_sorted(d0, d1,
                        return_time=True, **locator_args):
            if time_st > instant:
                if not first:
                    if return_time:
                        return (lasttime, lastgran)
                    else:
                        return lastgran
                break
            lasttime = time_st
            lastgran = gran
            first = False
        raise GranuleLocatorError("Can not find any granule "
            "before {:%Y-%m-%d %H:%M:%S}".format(instant))

    @staticmethod
    def _getyear(gd, s, alt):
        """Extract year info from group-dict

        Taking a group dict and a string, get an int for the year.
        The group dict should come from re.fullmatch().groupdict().  The
        second argument is typically "year" or "year_end".  If this is a
        4-digit string, it is taken as a year.  If it is a 2-digit string,
        it is taken as a 2-digit year, which is taken as 19xx for <= 68,
        and 20xx for >= 69, according to POSIX and ISO C standards.
        If there is no match, return alt.

        Arguments:
            gd (dict): Group-dict such as returned by re module
            s (str): String to match for
            int (str): Value returned in case there is no match

        Returns:
            int: year
        """

        if s in gd:
            i = int(gd[s])
            if len(gd[s]) == 2:
                return 1900 + i if i> 68 else 2000 + i
            elif len(gd[s]) == 4:
                return i
            else:
                raise ValueError("Found {:d}-digit string for the year. "
                    "Expected 2 or 4 digits.  Giving up.".format(len(gd[s])))
        else:
            return alt

    def get_info_for_granule(self, p):
        """Return dict (re.fullmatch) for granule, based on re

        Arguments:
            
            p (pathlib.Path): path to granule

        Returns:
            
            dict: dictionary with info, such as returned by
            :func:`re.fullmatch`.
        """

        if not isinstance(p, pathlib.Path):
            p = pathlib.Path(p)
        m = self._re.fullmatch(p.name)
        return m.groupdict()

    def get_times_for_granule(self, p, **kwargs):
        """For granule stored in `path`, get start and end times.

        May take hints for year, month, day, hour, minute, second, and
        their endings, according to self.datefields

        Arguments:

            p (pathlib.Path): path to granule
            **kwargs: Any more info that may be needed

        Returns:
            (datetime, datetime): Start and end time for granule

        """
        if not isinstance(p, pathlib.PurePath):
            p = pathlib.PurePath(p)
        if str(p) in self._granule_start_times.keys():
            (start, end) = self._granule_start_times[str(p)]
        else:
            gd = self.get_info_for_granule(p)
            if (any(f in gd.keys() for f in self.datefields) and
                (any(f in gd.keys() for f in {x+"_end" for x in self.datefields})
                        or self.granule_duration is not None)):
                st_date = [int(gd.get(p, kwargs.get(p, 0))) for p in self.datefields]
                td = datetime.timedelta()
                if st_date[1] == st_date[2] == 0:
                    if "doy" in gd:
                        td += datetime.timedelta(days=int(gd["doy"])-1)
                    # month and day can't be 0...
                    st_date[1] = st_date[1] or 1
                    st_date[2] = st_date[2] or 1
                # maybe it's a two-year notation
                st_date[0] = self._getyear(gd, "year", kwargs.get("year", 0))

                try:
                    start = datetime.datetime(*st_date)
                except ValueError as e:
                    raise InvalidFileError("File {!s} has invalid "
                        "starting datetime, giving up".format(p)) from v
                if "tod" in gd and start.time() == datetime.time(0):
                    td += datetime.timedelta(seconds=int(gd["tod"]))
                start += td
                if any(k.endswith("_end") for k in gd.keys()):
                    # FIXME: Does this go well at the end of
                    # year/month/day boundary?  Should really makes sure
                    # that end is the first time occurance after
                    # start_time fulfilling the provided information.
                    end_date = st_date.copy()
                    end_date[0] = self._getyear(gd, "year_end", kwargs.get("year_end", st_date[0]))
                    end_date[1:] = [int(gd.get(p+"_end",
                                               kwargs.get(p+"_end", sd_x)))
                                   for (p, sd_x) in zip(self.datefields[1:],
                                                        st_date[1:])]
                    try:
                        end = datetime.datetime(*end_date)
                    except ValueError as v:
                        raise InvalidFileError("File {!s} has invalid "
                            "ending datetime, giving up".format(p)) from v
                    if end_date < st_date: # must have crossed date boundary
                        end += datetime.timedelta(days=1)
                elif self.granule_duration is not None:
                    end = start + self.granule_duration
                else:
                    raise RuntimeError("This code should never execute")
            else:
                # implementation depends on dataset
                (start, end) = self.get_time_from_granule_contents(str(p))
                self._granule_start_times[str(p)] = (start, end)
        return (start, end)

    # not an abstract method because subclasses need to implement it /if
    # and only if starting/ending times cannot be determined from the filename
    def get_time_from_granule_contents(self, p):
        """Get datetime objects for beginning and end of granule

        If it returns None, then use same as start time.

        Arguments:

            p (pathlib.Path): Path to file

        Returns:

            (datetime, datetime): 2-tuple for start and end times
        """
        raise ValueError(
            ("To determine starting and end-times for a {0} dataset, "
             "I need to read the file.  However, {0} has not implemented the "
             "get_time_from_granule_contents method.".format(
                type(self).__name__)))

    def _open_granule_file(self):
        if self.granule_cache_file is not None:
            p = str(self.basedir / self.granule_cache_file)
            try:
                self._granule_start_times = shelve.open(p, protocol=4)
            except OSError:
                logging.error(("Unable to open granule file {} RW.  "
                               "Opening copy instead.").format(p))
                tf = tempfile.NamedTemporaryFile()
                shutil.copyfile(p, tf.name)
                #self._granule_start_times = shelve.open(p, flag='r')
                self._granule_start_times = shelve.open(tf.name)
        else:
            self._granule_start_times = {}

    def _filter_firstline(self, f, M):
        # by default, do nothing
        raise NotImplementedError("Dataset {self:s} {self.name:s} needs "
            "database for storing a table with the first new scanline per "
            "granule, i.e. the first line not contained in the previous. "
            "Please define `granules_firstline` in source-code or "
            "configuration file.")

class SingleMeasurementPerFileDataset(MultiFileDataset):
    """Represents datasets where each file contains one measurement.

    An example of this would be ACE-FTS, or some radio-occultation datasets.

    Attributes:

        filename_fields (Mapping[str, dtype])

            dict with {name, dtype} for fields that should be copied from
            the filename (as obtained with self.re) into the header
    """

    granule_duration = datetime.timedelta(0)
    filename_fields = {}


    @abc.abstractmethod
    def read_single(self, p, fields="all"):
        """Read a single measurement from a single file.

        Shall take one argument (a path object) and return a tuple with
        (header, measurement).  The header shall contain information like
        latitude, longitude, time.

        Arguments:

            p (pathlib.Path): path to file
            
            fields (Iterable[str] or str): What fields to return.
                See :func:`Dataset.read_period` for details.
        """
        raise NotImplementedError()

    _head_dtype = {}
    def _read(self, p, fields="all"):
        """Reads a single measurement converted to ndarray

        Arguments:

            p (pathlib.Path): path to file

            fields (Iterable[str] or str): What fields to return.
                See :func:`Dataset.read_period` for details.
        """

        (head, body) = self.read_single(p, fields=fields)

        dt = [(s+body.shape if len(s)==2 else (s[0], s[1], s[2]+body.shape))
                for s in body.dtype.descr]
        dt.extend([("lat", "f8"), ("lon", "f8"), ("time", "M8[s]")])
        dt.extend([(s, self._head_dtype[s])
                for s in (head.keys() & self._head_dtype.keys())
                if s not in {"lat", "lon", "time"}])
        if self.filename_fields:
            info = self.get_info_for_granule(p)
            dt.extend(self.filename_fields.items())
        D = numpy.ma.empty(1, dt)

        for nm in body.dtype.names:
            D[nm] = body[nm]

        for nm in {"lat", "lon", "time"}:
            D[nm] = head[nm]

        for nm in head.keys() & D.dtype.names:
            if nm not in {"lat", "lon", "time"}:
                D[nm] = head[nm]

        if self.filename_fields:
            for nm in self.filename_fields.keys():
                D[nm] = info[nm]

        return (D, {})

class NetCDFDataset:
    """Mixin for any dataset where the contents are in NetCDF.

    This may provide a good default for any NetCDF-based dataset.  The
    reading routine will take the most commonly occurring dimension as the
    ndarray axes, and the rest within structured multidimensional dtype.

    USE WITH CARE!  PROVISIONAL API!

    This one should use xarray, of course!  See
    https://arts.mi.uni-hamburg.de/trac/rt/ticket/145
    """

    read_returns = "ndarray" # can be set to xarray

    def _get_dtype_from_vars(self, alldims, allvars, fields, prim):
        """Get dtype from alldims, allvars
        """
        return numpy.dtype([
            (k,
             "f4" if (getattr(v, "Scale", 1)!=1) else v.dtype,
             tuple(s for (i, s) in enumerate(v.shape)
                 if v.dimensions[i] != alldims[prim].name))
             for (k, v) in allvars.items()
             if ((alldims[prim].name in v.dimensions) if fields=="all"
                  else (k in fields))])

    def _read(self, f, fields="all",
              pseudo_fields=None,
              prim=None):
        if self.read_returns == "ndarray":
            return (self._read_ndarray(f, fields, pseudo_fields, prim), {})
        elif self.read_returns == "xarray":
            ds = xarray.open_dataset(f)
            if fields != "all":
                ds = ds[fields]
            if pseudo_fields is not None:
                warnings.warn("Ignoring pseudo-fields", UserWarning)
            # if there is any problem loading the file, I want to know
            # that now, not whenever I happen to read the data
            ds.load()
            return (ds, {})

    def _read_ndarray(self, f, fields="all", pseudo_fields=None,
                      prim=None):
        if pseudo_fields is None:
            pseudo_fields = {}
        with netCDF4.Dataset(f, 'r') as ds:

            # generic conversion of NetCDF to ndarray; consider the most
            # common dimension, make this one the shape of the ndarray,
            # copy over all variables that share this dimension, and
            # ignore the rest, unless variables are passed explicitly

            # first do the “pseudo fields” so I know the dtype before I
            # create the structured dtype
            extra = {nm: f(ds) for (nm, f) in pseudo_fields.items()}

            #
            # if contained in one layer of groups, flatten those first

            if ds.groups != {}: # NB: empty OrderedDict == {}
                alldims = collections.OrderedDict(
                    itertools.chain.from_iterable(
                        ds[x].dimensions.items() for x in ds.groups.keys()))
                allvars = collections.OrderedDict(
                    itertools.chain.from_iterable(
                        ds[x].variables.items() for x in ds.groups.keys()))
            else:
                alldims = ds.dimensions
                allvars = ds.variables
            # count most frequent dimensions
            cnt = collections.Counter(
                itertools.chain.from_iterable(
                    allvars[var].dimensions for var in allvars.keys()))
            prim = prim or cnt.most_common(1)[0][0]
            n = alldims[prim].size
            M = numpy.zeros(shape=(n,),
                dtype=self._get_dtype_from_vars(alldims, allvars,
                                                fields, prim).descr +
                      [(nm, v.dtype, v.shape[1:]) for (nm, v) in
                                  extra.items()])
            M = numpy.ma.masked_array(M)

            for v in M.dtype.names:
                try:
                    M[v][...] = extra[v][...]
                except KeyError:
                    # should be direct instead
                    pass
                else:
                    continue
                try:
                    M[v][...] = allvars[v][...] * getattr(allvars[v], "Scale", 1)
                    try:
                        M[v].mask[M[v]==allvars[v]._FillValue] = True
                    except AttributeError:
                        # no fill value or not a masked array...?
                        pass
                except TypeError:
                    pass # probably not a numeric type

        return M

class HomemadeDataset(NetCDFDataset, MultiFileDataset):
    """For any dataset created by typhon or its dependencies.

    Currently supports only saving to npz, through the save_npz method.
    Eventually, should also support other file formats, in particular
    NetCDF.
    """

    stored_name = None
    write_basedir = write_subdir = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._str2path("write_basedir", "write_subdir", "stored_name")

    def find_granule_for_time(self, **kwargs):
        """Find granule for specific time.

        May or may not exist.

        Arguments (kw only) are passed on to format directories stored in
        self.basedir / self.subdir / self.stored_name, along with
        self.__dict__.  One special keyword, "write", can be used when
        writing data, as some datasets in "production mode" have different
        files for writing than reading.

        Returns path to granule.
        """

        if kwargs.get("mode") == "write":
            d = (getattr(self, "write_basedir", self.basedir) /
                 getattr(self, "write_subdir", self.subdir) /
                 self.stored_name)
        else:
            d = self.basedir / self.subdir / self.stored_name
        # next line inspired by http://stackoverflow.com/a/32196401/974555
        # which was posted by Alek Kowalczyk on 2015-08-25 and licensed
        # under CC-BY-SA 3.0.  I believe my code is sufficiently far from
        # the original to not be a derivative work and that it can legally
        # be MIT-licensed.
        subsdict = {attr: getattr(self, attr) 
                    for attr in dir(self)}
        subsdict.update(**kwargs)
        nm = pathlib.Path(str(d).format(**subsdict))
        return nm

    def _read(self, f, fields="all"):
        if f.endswith("npz") or f.endswith("npy"):
            if fields != "all":
                warnings.warn("Ignoring fields≠'all'!", UserWarning)
            return (numpy.load(f)["arr_0"], {})
        elif f.endswith("nc"):
            try:
                return super()._read(f, fields=fields)
            except ValueError as v:
                if v.args[0] == ("zero-size array to reduction operation "
                                 "minimum which has no identity"):
                    # A bug in xarray will cause a ValueError if trying to
                    # decode the times in a NetCDF file with length 0.
                    # See https://github.com/pydata/xarray/issues/1329 .
                    # Although I can't presently solve this (not decoding
                    # would be undesirable due to inconsistencies), I can
                    # at least replace it by an exception that means "bad
                    # data".
                    raise InvalidDataError("It looks like the NetCDF file "
                        "is empty or at least has a datetime variable with "
                        "length 0.  Due to "
                        "https://github.com/pydata/xarray/issues/1329 "
                        "I can't currently handle that.  Original message "
                        "was: " + v.args[0]) from v
                else:
                    raise

#    def find_granules(self, start, end):
#        raise StopIteration()

#    @abc.abstractmethod
#    def quicksave(self, f):
#        """Quick save to file
#
#        :param str f: File to save to
#        """

    def save_npz(self, path, M):
        """Save to compressed npz

        Arguments:

            path (pathlib.Path): Path to store to

            M (ndarray): Contents of what to store.
        """
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez_compressed(str(path), M)

class MultiSatelliteDataset(metaclass=utils.metaclass.AbstractDocStringInheritor):
    satellites = set()
    @property
    def valid_field_values(self):
        return {"satname": self.satellites}

class HyperSpectral(Dataset, em.FwmuMixin):
    """Superclass for any hyperspectral instrument
    """
    freqfile = None

# Not yet transferred from pyatmlab to typhon, not clean enough:
#
# - ProfileDataset
# - StationaryDataset
# - HyperSpectral

class DatasetDeque:
    """A deque-like object sliding through a dataset.

    For a particular dataset, keep data corresponding to time period in
    memory.  When reading new data, discard a corresponding time period of
    data in the past.  This should be useful to process longer periods of
    data that do not fit into memory, to calculate sliding window
    statistics, or to perform processing where values nearby in time are
    required.
    """

    dsobj = window = init_time = center_time = data = None
    def __init__(self, ds, window, init_time, *args, **kwargs):
        """Initialise a DatasetDeque

        Arguments:

            ds [Dataset]: Dataset that this belongs to, for example,
                HIRS().  This dataset must have an implementation of
                .as_xarray_dataset().

            window [timedelta]: Duration that should be kept in memory.
                For example, datetime.timedelta(hours=48).

            init_time [datetime]: Instant around which initial window
                shall be centred.

            Remaining arguments passed to read_period.
        """
        self.dsobj = ds
        self.window = window
        self.init_time = init_time
        self.center_time = init_time

        self.reset(newtime=self.init_time, *args, **kwargs)

    def reset(self, newtime=None, *args, **kwargs):
        """Reset to initial conditions

        Sets data to indicated time or self.init_time ± self.window/2

        Arguments:

            newtime [datetime.datetime]: optional, time at which to
            center window.  If not given, reset to initial time.

            Remaining arguments passed to read_period.

        """
        self.edges = (self.init_time-self.window/2,
                      self.init_time+self.window/2)
        self.center_time = newtime or self.init_time
        M = self.dsobj.read_period(
            *self.edges, *args, **kwargs)

        self.data = self.dsobj.as_xarray_dataset(M)
            
    def move(self, period, *args, **kwargs):
        """Read period on the right, discard period on the left

        This moves the window to the right by `period` by reading `period`
        new data, appending this on the right, and discarding an equally
        long period on the left.

        Arguments:

            period [timedelta]: Duration by which to shift.
                Must be positive.

            Remaining arguments passed to read_period.
        """

        if period > self.window:
            raise ValueError("Shifting period ({!s}) exceeds window "
                "length ({!s})!".format(period, self.window))
        self.center_time += period
        ok = False
        try:
            Mnew = self.dsobj.read_period(
                self.edges[1],
                self.edges[1]+period,
                *args, **kwargs)
        except: # syntactically required
            raise
        else:
            datanew = self.dsobj.as_xarray_dataset(Mnew)
            ok = True
        finally:
            self.edges = (self.edges[0] + period,
                          self.edges[1] + period)

        # need to convert to ms or it will become int and indexing will
        # fail, see https://github.com/numpy/numpy/issues/8546
        # and https://github.com/pydata/xarray/issues/1240
        #
        # If the very first iteration fails to read the first period and
        # is off by X hours, EVERYTHING will be off by X hours if I simply
        # reset it based on the first time in self.data!  Therefore the
        # 'min' function.
#        newst = self.data["time"][0].values.astype("M8[ms]") + numpy.timedelta64(period)
        newst = min(numpy.datetime64(self.edges[0]), self.data["time"][0].values.astype("M8[ms]") + numpy.timedelta64(period))
        # BUG: need workaround for https://github.com/pydata/xarray/issues/1297
        all_encodings = {k: v.encoding for (k, v) in self.data.data_vars.items()}
        self.data = xarray.concat(
            (self.data.sel(time=slice(newst, None)),
             datanew),
            dim="time")
        for (k, v) in self.data.data_vars.items():
            v.encoding = all_encodings[k]

    def resize(self, window):
        """Resize window
        """

        self.window = window
        self.edges = (self.center_time-window/2, self.center_time+window/2)
        self.data = self.dsobj.as_xarray_dataset(
            self.dsobj.read_period(*self.edges))

    def __repr__(self):
        return "<{:s} {:s} centred at {:%Y-%m-%d %H:%M}>".format(
            self.__class__.__name__, repr(self.dsobj),
            self.center_time)
