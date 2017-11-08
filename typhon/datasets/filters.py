"""Collection of classes related to filtering

"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import sys
import abc
import dbm
import logging
import tempfile
import pathlib
import shutil
import datetime

import numpy
try:
    import progressbar
except ImportError:
    progressbar = None

from . import dataset

class FilterError(Exception):
    """For any errors related to filtering.
    """

class OutlierFilter(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def filter_outliers(self, C):
        ...

class MEDMAD(OutlierFilter):
    """Outlier filter based on Median Absolute Deviation

    """

    def __init__(self, cutoff):
        self.cutoff = cutoff
    
    def filter_outliers(self, C):
        cutoff = self.cutoff
        if C.ndim == 3:
            med = numpy.ma.median(
                C.reshape(C.shape[0]*C.shape[1], C.shape[2]),
                0)
            mad = numpy.ma.median(
                abs(C - med).reshape(C.shape[0]*C.shape[1], C.shape[2]),
                0)
        elif C.ndim < 3:
            med = numpy.ma.median(C.reshape((-1,)))
            mad = numpy.ma.median(abs(C - med).reshape((-1,)))
        else:
            raise ValueError("Cannot filter outliers on "
                "input with {ndim:d}>3 dimensions".format(ndim=C.ndim))
        fracdev = ((C - med)/mad)
        return abs(fracdev) > cutoff

class OrbitFilter(metaclass=abc.ABCMeta):
    """Generic, abstract class for any kind of filtering.

    Implementations of this class are intended to be used between and
    after each orbit file is read by Dataset.read_period.  One important
    implementation for this class are the various OverlapFilter
    implementations, that make sure overlaps are removed.  This can either
    be done after each orbit or when all orbits have been read.

    Different filters may need different meta-information that needs to be
    provided by specific Dataset._read implementations.  For example, the
    FirstlineDBFilter needs a header name (or the entire header from which
    to extract this).  Each filterer can state what keyword arguments need
    to be passed on to the reading routine using the args_to_reader
    attribute.
    """

    args_to_reader = {}

    @abc.abstractmethod
    def reset(self):
        ...

    @abc.abstractmethod
    def filter(self, scanlines, **extra):
        ...

    @abc.abstractmethod
    def finalise(self, arr):
        ...

class TimeMaskFilter(OrbitFilter):
    """Throw out bad (masked) times.
    """

    def __init__(self, ds):
        self.ds = ds

    def reset(self):
        pass

    def filter(self, scanlines, **extra):
        # when time is masked, we should REALLY despair.  We want
        # to have sequential scanlines.  Throw them out already!
        if scanlines["time"].mask.any():
            logging.warning("Throwing out {:d} scanlines because "
                "their times are flagged and not sequential".format(
                    scanlines["time"].mask.sum()))
            good = ~scanlines["time"].mask
            return scanlines[good] 
        else:
            return scanlines

    def finalise(self, arr):
        return arr

class HIRSTimeSequenceDuplicateFilter(OrbitFilter):
    """Force scanlines to be in the proper sequence without duplicates

    This is for handling time sequence and duplicate issues within a
    single orbit file.  The two have to be in one filter because the
    application of the latter relies on the application of the first.
    I suppose one could conceivably want to remove time sequence problems
    but keep duplicate scanlines.
    """

    def reset(self):
        pass

    def filter(self, scanlines, **extra):
        goodorder = scanlines["hrs_scnlin"][1:] > scanlines["hrs_scnlin"][:-1]
        if not goodorder.all():
            logging.warning("{!s} has {:d} scanlines are out of "
                "order, resorting".format(path, (~goodorder).sum()))
            neworder = numpy.argsort(scanlines["hrs_scnlin"].data)
            scanlines = scanlines[neworder]        

        # if there still are any now, it can only be due to duplicates
        goodorder = scanlines["hrs_scnlin"][1:] > scanlines["hrs_scnlin"][:-1]
        if not goodorder.all():
            logging.warning("{!s} has {:d} duplicate "
                "scanlines (judging from scanline number), removing".format(path, (~goodorder).sum()))
            (_, ii) = numpy.unique(scanlines["hrs_scnlin"],
                                   return_index=True)
            scanlines = scanlines[ii]


        # still time sequence issues?
        goodtime = numpy.argsort(scanlines["time"]) == numpy.arange(scanlines.size)
        if not goodtime.all():
            logging.warning("{!s} (still) has time sequence issues. "
                "Dropping {:d} scanlines to be on the safe side. "
                "This is probably overconservative.".format(path,
                (~goodtime).sum()))
            scanlines = scanlines[goodtime]

        # in some cases, like 1985-11-30T17:19:45.056 on NOAA-9,
        # there are scanlines with different scanline numbers but
        # the same time!
        (_, ii) = numpy.unique(scanlines["time"], return_index=True)
        if ii.size < scanlines["time"].size:
            logging.warning("Oops!  There are scanlines with different "
                "scanline numbers but the same time!  Removing {:d} "
                "more lines.  I hope that's it!".format(
                    scanlines["time"].size-ii.size))
            scanlines = scanlines[ii]
            cc = cc[ii, :, :]

        return scanlines

    def finalise(self, arr):
        return arr

class OverlapFilter(OrbitFilter):
    """Implementations to feed into firstline filtering

    This is used in tovs HIRS reading routine.
    """

class FirstlineDBFilter(OverlapFilter):
    def __init__(self, ds, granules_firstline_file):
        self.ds = ds
        self.granules_firstline_file = granules_firstline_file

    def reset(self):
        pass

    _tmpdir = None
    _firstline_db = None
    def filter(self, scanlines, header):
        """Filter out any scanlines that existed in the previous granule.

        Only works on datasets implementing get_dataname from the header.
        """
        dataname = self.ds.get_dataname(header, robust=True)
        if self._firstline_db is None:
            try:
                self._firstline_db = dbm.open(
                    str(self.granules_firstline_file), "r")
            except dbm.error as e: # presumably a lock
                tmpdir = tempfile.TemporaryDirectory()
                self._tmpdir = tmpdir # should be deleted only when object is
                tmp_gfl = str(pathlib.Path(tmpdir.name,
                    self.granules_firstline_file.name))
                logging.warning("Cannot read GFL DB at {!s}: {!s}, "
                    "presumably in use, copying to {!s}".format(
                        self.granules_firstline_file, e.args, tmp_gfl))
                shutil.copyfile(str(self.granules_firstline_file),
                    tmp_gfl)
                self.granules_firstline_file = tmp_gfl
                self._firstline_db = dbm.open(tmp_gfl)
        try:
            firstline = int(self._firstline_db[dataname])
        except KeyError as e:
            raise FilterError("Unable to filter firstline: {:s}".format(
                e.args[0])) from e
        if firstline > scanlines.shape[0]:
            logging.warning("Full granule {:s} appears contained in previous one. "
                "Refusing to return any lines.".format(dataname))
            return scanlines[0:0]
        return scanlines[scanlines["hrs_scnlin"] > firstline]    

    def update_firstline_db(self, satname=None, start_date=None, end_date=None,
            overwrite=False):
        """Create / update the firstline database

        Create or update the database describing for each granule what the
        first scanline is that doesn't occur in the preceding granule.

        If a granule is entirely contained within the previous one,
        firstline is set to L+1 where L is the number of lines.
        """
        prev_head = prev_line = None
        satname = satname or self.ds.satname
        start_date = start_date or self.ds.start_date
        end_date = end_date or self.ds.end_date
        if end_date > datetime.datetime.now():
            end_date = datetime.datetime.now()
        logging.info("Updating firstline-db {:s} for "
            "{:%Y-%m-%d}--{:%Y-%m-%d}".format(satname, start_date, end_date))
        count_updated = count_all = 0
        with dbm.open(str(self.granules_firstline_file), "c") as gfd:
            try:
                bar = progressbar.ProgressBar(max_value=1,
                    widgets=[progressbar.Bar("=", "[", "]"), " ",
                        progressbar.Percentage(), ' (',
                        progressbar.AdaptiveETA(), " -> ",
                        progressbar.AbsoluteETA(), ') '])
            except AttributeError:
                dobar = False
                bar = None
                logging.info("If you had the "
                    "progressbar2 module, you would have gotten a "
                    "nice progressbar.")
            else:
                dobar = sys.stdout.isatty()
                if dobar:
                    bar.start()
                    bar.update(0)
            for (g_start, gran) in self.ds.find_granules_sorted(start_date, end_date,
                            return_time=True, satname=satname):
                try:
                    (cur_line, extra) = self.ds.read(gran,
                        apply_scale_factors=False, calibrate=False,
                        apply_flags=False)
                    cur_head = extra["header"]
                    cur_time = self.ds._get_time(cur_line)
                except (dataset.InvalidFileError,
                        dataset.InvalidDataError) as exc:
                    logging.error("Could not read {!s}: {!s}".format(gran, exc))
                    continue
                lab = self.ds.get_dataname(cur_head, robust=True)
                if lab in gfd and not overwrite:
                    logging.debug("Already present: {:s}".format(lab))
                elif prev_line is not None:
                    # what if prev_line is None?  We don't want to define any
                    # value for the very first granule we process, as we might
                    # be starting to process in the middle...
                    if cur_time.max() > prev_time.max():
                        # Bugfix 2017-01-16: do not get confused between
                        # the index and the hrs_scnlin field.  So far, I'm using
                        # the index to set firstline but the hrs_scnlin
                        # field to apply it.
                        #first = (cur_time > prev_time[-1]).nonzero()[0][0]
                        # Bugfix 2017-08-21: instead of taking the last
                        # time from the previous granule, take the
                        # maximum; this allows for time sequence errors.
                        # See #139
                        first = cur_line["hrs_scnlin"][cur_time > prev_time.max()].min()
                        logging.debug("{:s}: {:d}".format(lab, first))
                    else:
                        first = cur_line["hrs_scnlin"].max()+1
                        logging.info("{:s}: Fully contained in {:s}!".format(
                            lab, self.ds.get_dataname(prev_head, robust=True)))
                    gfd[lab] = str(first)
                    count_updated += 1
                prev_line = cur_line.copy()
                prev_head = cur_head.copy()
                prev_time = cur_time.copy()
                if dobar:
                    bar.update((g_start-start_date)/(end_date-start_date))
                count_all += 1
            if dobar:
                bar.update(1)
                bar.finish()
            logging.info("Updated {:d}/{:d} granules".format(count_updated, count_all))

    def finalise(self, arr):
        return arr

class NullLineFilter(OverlapFilter):
    """Do not filter firstlines at all
    """

    def reset(self):
        pass

    def filter(self, path, header, scanlines):
        return scanlines

    def finalise(self, arr):
        return arr

class HIRSBestLineFilter(OverlapFilter):
    """Choose best between overlaps.

    Currently works only for HIRS.
    """
    def __init__(self, ds):
        self.ds = ds

    def reset(self):
        pass

    def filter(self, path, header, scanlines):
        """Choose best lines in overlap between last/current/next granule
        """

        # self.ds.read should be using caching already, so no need to keep
        # track of what I've already read here.  Except that caching only
        # works if the arguments are identical, which they aren't.
        # Consider applying caching on a lower level?  But then I need to
        # store more…
        prevnext = [
            self.ds.read(
                self.ds.find_most_recent_granule_before(
                    scanlines["time"][idx].astype(datetime.datetime) +
                        datetime.timedelta(minutes=Δmin)),
                fields=["hrs_qualind", "hrs_scnlin", "time"],
                return_header=False,
                apply_scale_factors=False, calibrate=False, apply_flags=False,
                filter_firstline=False, apply_filter=False, max_flagged=1.0)
                        for (idx, Δmin) in [(0, -1), (-1, 1)]]

        #
        raise NotImplementedError("Not implemented yet beyond this point")


    def finalise(self, arr):
        return arr


