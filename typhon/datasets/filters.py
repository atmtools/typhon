"""Collection of classes related to filtering

"""

# Any commits made to this module between 2015-05-01 and 2019-02-01
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
import warnings

import numpy
try:
    import progressbar
except ImportError:
    progressbar = None

from . import dataset

logger = logging.getLogger(__name__)

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

    def __init__(self, cutoff, fallback_min_std=0.1):
        self.cutoff = cutoff
        self.fallback_min_std = 0.1
    
    def filter_outliers(self, C):
        cutoff = self.cutoff
        if C.ndim == 3:
            med = numpy.ma.median(
                C.reshape(C.shape[0]*C.shape[1], C.shape[2]),
                0)
            mad = numpy.ma.median(
                abs(C - med).reshape(C.shape[0]*C.shape[1], C.shape[2]),
                0)
            if (mad==0).any():
                # use fallback
                med[mad==0] = C[..., mad==0].reshape(C.shape[0]*C.shape[1], (mad==0).sum()).mean(0)
                mad[mad==0] = numpy.c_[
                    C[..., mad==0].reshape(C.shape[0]*C.shape[1], (mad==0).sum()).std(0),
                    numpy.tile(self.fallback_min_std, (mad==0).sum())].max(1)
        elif C.ndim < 3:
            med = numpy.ma.median(C.reshape((-1,)))
            mad = numpy.ma.median(abs(C - med).reshape((-1,)))
            if mad==0:
                med = C.mean()
                mad = max(C.std(), self.fallback_min_std)
        else:
            raise ValueError("Cannot filter outliers on "
                "input with ndim={ndim:d}>3 dimensions".format(ndim=C.ndim))
        fracdev = ((C - med)/mad)
        return abs(fracdev) > cutoff

class OrbitFilter:
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

    Moments in filtering that this class takes is responsible for:

    - Before the first orbit is read: .reset()
    - After each orbit is read. .filter(...)
    - After the last orbit is read: .finalise(...)

    You can also use this class for validation, i.e. to check that data
    are correct and if they aren't, raise an exception, if they are,
    return data as-is.
    """

    args_to_reader = {}

    def reset(self):
        pass

    def filter(self, scanlines, **extra):
        return scanlines

    def finalise(self, arr):
        return arr
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

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
            logger.warning("Throwing out {:d} scanlines because "
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
            logger.warning("{:d} scanlines are out of "
                "order, resorting".format(
                (~goodorder).sum()))
            neworder = numpy.argsort(scanlines["hrs_scnlin"].data)
            scanlines = scanlines[neworder]        

        # if there still are any now, it can only be due to duplicates
        goodorder = scanlines["hrs_scnlin"][1:] > scanlines["hrs_scnlin"][:-1]
        if not goodorder.all():
            logger.warning("{:d} duplicate "
                "scanlines (judging from scanline number), removing".format((~goodorder).sum()))
            (_, ii) = numpy.unique(scanlines["hrs_scnlin"],
                                   return_index=True)
            scanlines = scanlines[ii]


        # still time sequence issues?
        goodtime = numpy.argsort(scanlines["time"]) == numpy.arange(scanlines.size)
        if not goodtime.all():
            logger.warning("Still has time sequence issues! "
                "Dropping {:d} scanlines to be on the safe side. "
                "This is probably overconservative.".format((~goodtime).sum()))
            scanlines = scanlines[goodtime]

        # in some cases, like 1985-11-30T17:19:45.056 on NOAA-9,
        # there are scanlines with different scanline numbers but
        # the same time!
        (_, ii) = numpy.unique(scanlines["time"], return_index=True)
        if ii.size < scanlines["time"].size:
            logger.warning("Oops!  There are scanlines with different "
                "scanline numbers but the same time!  Removing {:d} "
                "more lines.  I hope that's it!".format(
                    scanlines["time"].size-ii.size))
            scanlines = scanlines[ii]

        return scanlines

    def finalise(self, arr):
        return arr

class HIRSFlagger(OrbitFilter):
    """Apply HIRS flags and raise error in case of failure
    """

    def __init__(self, ds, max_flagged=0.5):
        self.ds = ds
        self.max_flagged = max_flagged

    def reset(self):
        pass

    def filter(self, scanlines, **extra):
        scanlines = self.ds.get_mask_from_flags(extra["header"],
            scanlines, max_flagged=self.max_flagged)

        return scanlines

    def finalise(self, arr):
        return arr

class HIRSCalibCountFilter(OrbitFilter):
    """Apply masking based on calibration count filter for HIRS
    """

    def __init__(self, ds, filter_calibcounts:OutlierFilter):
        self.ds = ds
        self.filter_calibcounts = filter_calibcounts

    def reset(self):
        pass

    def filter(self, scanlines, **extra):

        for v in self.ds.views:
            x = scanlines[self.ds.scantype_fieldname] == getattr(self.ds,
                    "typ_{:s}".format(v))
            if not x.any():
                raise dataset.InvalidDataError("Out of {:d} scanlines, "
                    "found no {:s} views, cannot calibrate!".format(
                        scanlines.shape[0], v))
            scanlines.mask["counts"][x, 8:, :] = self.ds.filter_calibcounts.filter_outliers(
                scanlines["counts"][x, 8:, :])


        cc = scanlines["calcof_sorted"]
#        scanlines = self.ds.apply_calibcount_filter(scanlines)
        if cc.ndim == 4:
            calibzero = (cc[:, :, 1, :]==0).all(2)
        if cc.ndim == 3:
            calibzero = (cc==0).all(2)
        if "bt" in scanlines.dtype.names:
            scanlines["bt"].mask[...] |= calibzero[:, numpy.newaxis, :19]
        if "radiance" in scanlines.dtype.names:
            scanlines["radiance"].mask[...] |= calibzero[:, numpy.newaxis, :20]
        try:
            # if one is masked, so should the other…
            scanlines["radiance"].mask[:, :, :19] |= scanlines["bt"].mask
        except ValueError: # xarray raises ValueError, not KeyError
            # not a problem if either of those fields does not exists
            pass

        return scanlines

    def finalise(self, arr):
        return arr

class HIRSPRTTempFilter(OrbitFilter):
    """Apply masking to PRT counts when they are outliers
    """

    def __init__(self, ds, filter_prttemps:OutlierFilter):
        self.ds = ds
        self.filter_prttemps = filter_prttemps

    def reset(self):
        pass

    def filter(self, scanlines, **extra):
        for fld in [nm for nm in scanlines.dtype.names if
                    nm.startswith("temp_")]:
            scanlines[fld].mask |= self.ds.filter_prttemps.filter_outliers(
                                    scanlines[fld])
        return scanlines

    def finalise(self, arr):
        return arr

class OverlapFilter(OrbitFilter):
    """Implementations to feed into firstline filtering

    This is used in tovs HIRS reading routine.
    """
    
    # the 'late' attribute should be set to True if all duplicate checking
    # is done late; this will suppress after-orbit verification of absense
    # of duplicates.
    late = False

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
                logger.warning("Cannot read GFL DB at {!s}: {!s}, "
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
            logger.warning("Full granule {:s} appears contained in previous one. "
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
        logger.info("Updating firstline-db {:s} for "
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
                logger.info("If you had the "
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
                        apply_scale_factors=False, calibrate=False)
                    cur_head = extra["header"]
                    cur_time = self.ds._get_time(cur_line)
                except (dataset.InvalidFileError,
                        dataset.InvalidDataError) as exc:
                    logger.error("Could not read {!s}: {!s}".format(gran, exc))
                    continue
                lab = self.ds.get_dataname(cur_head, robust=True)
                if lab in gfd and not overwrite:
                    logger.debug("Already present: {:s}".format(lab))
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
                        logger.debug("{:s}: {:d}".format(lab, first))
                    else:
                        first = cur_line["hrs_scnlin"].max()+1
                        logger.info("{:s}: Fully contained in {:s}!".format(
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
            logger.info("Updated {:d}/{:d} granules".format(count_updated, count_all))

    def finalise(self, arr):
        return arr

class NullLineFilter(OverlapFilter):
    """Do not filter firstlines at all
    """

    def reset(self):
        pass

    def filter(self, scanlines, header):
        return scanlines

    def finalise(self, arr):
        return arr

class HIRSBestLineFilter(OverlapFilter):
    """Choose best between overlaps.

    Currently works only for HIRS.
    """

    # those are expected to be different between orbits, others are not
    knowndiff = {'hrs_scnlin', 'hrs_qualind', 'hrs_linqualflgs',
                 'hrs_chqualflg'}
    warn_overlap = False # set to True if you want warnings when overlaps inconsistent

    late = True
    def __init__(self, ds, warn_overlap=False):
        self.ds = ds
        self.warn_overlap = warn_overlap

    orbits = []
    def reset(self):
        self.orbits.clear()

    def filter(self, scanlines, header):
        self.orbits.append(
            (self.ds.get_dataname(header, robust=True),
             scanlines["time"].min(),
             scanlines["time"].max(),
             scanlines.size))
        return scanlines

    def select_winner(self, rep):
        """Select "winning" scanline among 2 or more

        Between 2 or more identical scanlines, select the one that fits
        best.  That probably means least flags etc.

        Takes a ndarray with dtype containing at least the HIRS flag
        fields.  Dimension should be (n,) where n>1.

        Returns the index of the best choice.
        """
        
        # relevant fields:
        #
        # - hrs_qualind
        # - hrs_linqualflgs
        # - hrs_chqualflg
        # - hrs_mnfrqual
        #
        # But this differs between HIRS/2/3/4...
        # need some neutral way of "scoring" how bad it is

        scores = self.ds.flagscore(rep)
        return numpy.argmin(scores)

    def finalise(self, arr, verify_overlap_consistency=False):
        """For all sets of duplicates, select optimal one

        Arguments:

            arr [ndarray], dtype with structured ndarray according to what
            the HIRS reading routine produces

            verify_overlap_consistency [bool].  Verify that overlapping
            scanlines have consistent information.  Defaults to False
            because this can be very slow.

        Returns:

            same array but sorted and with duplicates removed; for each
            pair of duplicates, best alternative is chosen
        """
        # Find all pairs of duplicates (usually in sets of 2) and the
        # with corresponding indices and multiplicity (count).  
        arrsrt = arr[numpy.argsort(arr["time"])]
        _, ii, cnt = numpy.unique(arrsrt["time"],
            return_index=True, return_counts=True)
        logger.debug("Selecting optimal scanlines for "
            f"{ii.size:d} overlapping pairs")
        # (mult_ii, mult_cnt) take the same values as if I were to do:
        # for (mult_ii, mult_cnt) in zip(ii[cnt>1], cnt[cnt>1]), but I
        # want to keep the indices so I can /write/ to ii
        multcnt_i_all = (cnt>1).nonzero()[0]
        for multcnt_i in multcnt_i_all:
            mult_ii = ii[multcnt_i]
            mult_cnt = cnt[multcnt_i]
            rep = arrsrt[mult_ii:mult_ii+mult_cnt]
            if self.warn_overlap:
                fields_notclose = {nm for nm in rep.dtype.names
                    if not
                    (rep[nm][0]==rep[nm]
                     if rep[nm].dtype.kind[0] in "MmS"
                     else numpy.isclose(rep[nm][0, ...], rep[nm])
                    ).all()} - self.knowndiff
                if len(fields_notclose) > 0:
                    warnings.warn(
                        "Overlapping or duplicate scanlines "
                        "have inconsistent values for ".format(
                            rep[0]["time"].astype(datetime.datetime))
                        + ", ".join(list(fields_notclose)),
                            UserWarning)
            # ii normally contains the index of the first of a sequence of
            # duplicates; select_winner returns the index of the optimal
            # choice within the set 'rep' of repeated scanlines; the sum
            # will thus be the index of our scanline of choice
            ii[multcnt_i] += self.select_winner(rep)
        return arrsrt[ii]
