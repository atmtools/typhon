"""Datasets for TOVS/ATOVS

This module imports typhon.physics.units and therefore has a soft
dependency on the pint units library.  Import this module only if you can
accept such a dependency.
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import sys
import io
import tempfile
import subprocess
import datetime
import logging
import gzip
import shutil
import abc
import pathlib
import warnings
import xarray
import numpy
try:
    import progressbar
except ImportError:
    progressbar = None
 
try:
    import coda
except ImportError:
    print("Unable to import coda, won't read IASI EPS L1C. "
        "If you need to read IASI EPS L1C, please obtain CODA from "
        "http://stcorp.nl/coda/ and install.  Good luck.",
        file=sys.stderr)
    
from . import dataset
from ..utils import (metaclass, safe_eval)
from .. import utils
 
# from .. import physics
from .. import math as tpmath
from ..physics.units import ureg
from ..physics.units import radiance_units as rad_u
from ..physics.units import em
from .. import config

from . import filters

from . import _tovs_defs

class Radiometer(metaclass=metaclass.AbstractDocStringInheritor):
    srf_dir = ""
    srf_backend_response = ""
    srf_backend_f = ""

class ATOVS:
    """Functionality in common with all ATOVS.

    Designed as mixin.
    """
    @staticmethod
    def _get_time(scanlines, prefix):
        return (scanlines[prefix + "scnlinyr"].astype("M8[Y]") - 1970 +
                (scanlines[prefix + "scnlindy"]-1).astype("m8[D]") +
                 scanlines[prefix + "scnlintime"].astype("m8[ms]"))

class HIRS(dataset.MultiSatelliteDataset, Radiometer, dataset.MultiFileDataset):
    """High-resolution Infra-Red Sounder.

    This class can read HIRS l1b as published in the NOAA CLASS archive.

    Reading routines as for any datasets (see documentation for Dataset,
    MultiFileDataset, and others).

    Specifically for HIRS: when reading a single file (i.e. h.read(path)),
    takes keyword arguments:

        radiance_units.  Defaults to "si", by which I annoyingly mean
        W/(m²·sr·Hz).  Set to "classic" if you want mW/(m²·sr·cm^{-1}),
        which is the unit more commonly used for HIRS and which it is
        calibrated against.

        apply_scale_factors.  If true (defaults true), apply scale factors
        as documented in KLM / POD guides.  This is required when
        calibrate is True.

        calibrate.  If true (defaults true), apply calibration.  When
        false, will not return any brightness temperatures or radiances,
        just counts.  Note that this relates to the native NOAA
        calibration, not to any new calibration such as developed for
        FIDUCEO.

        max_flagged.  Float between 0 and 1.  If a larger proportion than
        this number is flagged, raise an exception (FIXME DOC) and throw
        away the entire granule.

    Note that this class only reads in the standard HIRS data with its
    standard calibration.  Innovative calibrations including uncertainties
    are implemented in HIRSFCDR.

    To use this class, you need to define in your typhonrc the following
    settings in the section 'hirs':

        basedir

        subdir

        re
        
            (TODO: migrate to definition?)

        format_definition_file

            only for FIXME

    Work in progress.

    TODO/FIXME:

    - What is the correct way to use the odd bit parity?  Information in
      NOAA KLM User's Guide pages 3-31 and 8-154, but I'm not sure how to
      apply it.
    - If datasets like MHS or AVHRR are added some common code could
      probably move to a class between HIRS and MultiFileDataset, or to a
      mixin such as ATOVS.
    - Better handling of duplicates between subsequent granules.
      Currently it takes all lines from the older granule and none from
      the newer, but this should be decided on a case-by-case basis (Jon
      Mittaz, personal communication).
    """

    name = section = "hirs"
    format_definition_file = ""
    n_channels = 20
    n_calibchannels = 19
    n_minorframes = 64
    n_perline = 56
    count_start = 2
    count_end = 22
    granules_firstline_file = pathlib.Path("")
    re = r"(L?\d*\.)?NSS.HIR[XS].(?P<satcode>.{2})\.D(?P<year>\d{2})(?P<doy>\d{3})\.S(?P<hour>\d{2})(?P<minute>\d{2})\.E(?P<hour_end>\d{2})(?P<minute_end>\d{2})\.B(?P<B>\d{7})\.(?P<station>.{2})\.gz"
    satname = None

    temperature_fields = None

    # For convenience, define scan type codes.  Stores in hrs_scntyp.
    typ_Earth = 0
    typ_space = 1
    #typ_ict = 2 # internal cold calibration target; only used on HIRS/2
    typ_iwt = 3 # internal warm calibration target

    _fact_shapes = {"hrs_h_fwcnttmp": (4, 5)}
    _data_vars_props = None

    max_valid_time_ptp = numpy.timedelta64(3, 'h')
    filter_calibcounts = filter_prttemps = filters.MEDMAD(10)
    flag_fields = set()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mandatory_fields |= {"hrs_scnlin", "calcof_sorted"}
        self.mandatory_fields |= self.flag_fields
        self.granules_firstline_file = pathlib.Path(self.granules_firstline_file)
        if not self.granules_firstline_file.is_absolute():
            self.granules_firstline_file = self.basedir.joinpath(
                self.granules_firstline_file)
        self.default_orbit_filters = (
            filters.FirstlineDBFilter(self, self.granules_firstline_file),
            filters.TimeMaskFilter(self),
            filters.HIRSTimeSequenceDuplicateFilter(),
            filters.HIRSFlagger(self, max_flagged=0.5),
            filters.HIRSCalibCountFilter(self, self.filter_calibcounts),
            filters.HIRSPRTTempFilter(self, self.filter_prttemps),
            )
        if self.satname is not None:
            (self.start_date, self.end_date) = _tovs_defs.HIRS_periods[self.satname]
        # set here so inheriting classes can extend
        self.temperature_fields = {"iwt", "ict", "fwh", "scanmirror", "primtlscp",
            "sectlscp", "baseplate", "elec", "patch_full", "scanmotor", "fwm",
            "ch"}
        self._data_vars_props = _tovs_defs.HIRS_data_vars_props[self.version].copy()

    # docstring in class and parent
    def _read(self, path, fields="all",
                    apply_scale_factors=True, 
                    apply_calibration=True,
                    radiance_units="si"):
        if path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(str(path), 'rb') as f:
            self.seekhead(f)
            (header_dtype, line_dtype) = self.get_dtypes(f)
            header_bytes = f.read(header_dtype.itemsize)
            header = numpy.frombuffer(header_bytes, header_dtype)
            n_lines = header["hrs_h_scnlin"][0]
            scanlines_bytes = f.read()
            try:
                scanlines = numpy.frombuffer(scanlines_bytes, line_dtype)
            except ValueError as v:
                raise dataset.InvalidFileError("Can not read "
                    "whole number of records.  Expected {:d} scanlines, "
                    "but found {:d} lines with a remainder of {:d} "
                    "bytes.  File appears truncated.".format(
                        n_lines, *divmod(len(scanlines_bytes),
                                         line_dtype.itemsize))) from v
        if scanlines.shape[0] != n_lines:
            raise dataset.InvalidFileError(
                "Problem reading {!s}.  Header promises {:d} scanlines, but I found only {:d} — "
                "corrupted file? ".format(path, n_lines, scanlines.shape[0]))
        if n_lines < 2:
            raise dataset.InvalidFileError(
                "Problem reading {!s}.  File contains only {:d} scanlines. "
                "My reading routine cannot currently handle that.".format(
                    path, n_lines))
        n_lines = scanlines.shape[0]

        if apply_scale_factors:
            (header, scanlines) = self._apply_scale_factors(header, scanlines)
        if apply_calibration:
            if not apply_scale_factors:
                raise ValueError("Can't calibrate if not also applying"
                                 " scale factors!")
            (lat, lon) = self.get_pos(scanlines)
            if not (1 < lat.ptp() <= 180) or not (1 < lon.ptp() <= 360):
                raise dataset.InvalidDataError(
                    "Range of latitude {:.1f}°, longitude {:.1f}, suspect!".format(
                        lat.ptp(), lon.ptp()))
            other = self.get_other(scanlines)

            cc = self.get_cc(scanlines)
            cc = cc[:, numpy.argsort(self.channel_order), ...]
            elem = scanlines["hrs_elem"].reshape(n_lines,
                        self.n_minorframes, self.n_wordperframe)
            # x & ~(1<<12)   ==   x - 1<<12     ==    x - 4096    if this
            # bit is set
            counts = elem[:, :self.n_perline, self.count_start:self.count_end]
            counts = counts - self.counts_offset
            counts = counts[:, :, numpy.argsort(self.channel_order)]
            rad_wn = self.calibrate(cc, counts)

            # Convert radiance to BT
            (wn, c1, c2) = self.get_wn_c1_c2(header)

            # convert wn to SI units
            wn = wn * (1 / ureg.cm)
            wn = wn.to(1 / ureg.m)
            bt = self.rad2bt(rad_wn[:, :, :self.n_calibchannels], wn, c1, c2)

            # extract more info from TIP
            temp = self.get_temp(header, elem,
                scanlines["hrs_anwrd"]
                    if "hrs_anwrd" in scanlines.dtype.names
                    else None)

            # Copy over all fields... should be able to use
            # numpy.lib.recfunctions.append_fields but incredibly slow!
            scanlines_new = numpy.ma.empty(shape=scanlines.shape,
                dtype=(scanlines.dtype.descr +
                    [("radiance", "f4", (self.n_perline, self.n_channels,)),
                     ("counts", "i2", (self.n_perline, self.n_channels,)),
                     ("bt", "f4", (self.n_perline, self.n_calibchannels,)),
                     ("time", "M8[ms]"),
                     ("lat", "f8", (self.n_perline,)),
                     ("lon", "f8", (self.n_perline,)),
                     ("calcof_sorted", "f8", cc.shape[1:])] +
                    [("temp_"+k, "f4", temp[k].squeeze().shape[1:])
                        for k in temp.keys()] +
                    other.dtype.descr))
            for f in scanlines.dtype.names:
                scanlines_new[f] = scanlines[f]
            for f in temp.keys():
                scanlines_new["temp_" + f] = temp[f].squeeze()
            for f in other.dtype.names:
                scanlines_new[f] = other[f]
            if radiance_units == "si":
                scanlines_new["radiance"] = rad_wn.to(rad_u["si"],
                    "radiance")
            elif radiance_units == "classic":
                scanlines_new["radiance"] = rad_wn.to(rad_u["ir"],
                    "radiance")
            else:
                raise ValueError("Invalid value for radiance_units. "
                    "Expected 'si' or 'classic'.  Got "
                    "{:s}".format(radiance_units))
            scanlines_new["counts"] = counts
            scanlines_new["bt"] = bt
            scanlines_new["lat"] = lat
            scanlines_new["lon"] = lon
            time = self._get_time(scanlines)
            if time.ptp() > self.max_valid_time_ptp:
                raise dataset.InvalidDataError("Time span appears to be "
                    "{!s}.  That can't be right!".format(
                        time.ptp().astype(datetime.timedelta)))
            scanlines_new["time"] = self._get_time(scanlines)
            scanlines_new["calcof_sorted"] = cc
            scanlines = scanlines_new

            header_new = numpy.empty(shape=header.shape,
                dtype=(header.dtype.descr +
                    [("dataname", "<U42")]))
            for f in header.dtype.names:
                header_new[f] = header[f]
            try:
                header_new["dataname"] = self.get_dataname(header)
            except ValueError as exc:
                warnings.warn("Could not read dataname from header: " +
                    exc.args[0])
                header_new["dataname"] = pathlib.Path(path).stem
            header = header_new
        else: # i.e. not calibrating
            # FIXME: this violates DRY a bit, as those lines also occur
            # in the If:-block, but hard to do incrementally.  However, I
            # just rely on times being available too much.
            scanlines_new = numpy.empty(shape=scanlines.shape,
                dtype=(scanlines.dtype.descr +
                    [("time", "M8[ms]")]))
            scanlines_new["time"] = self._get_time(scanlines)
            for f in scanlines.dtype.names:
                scanlines_new[f] = scanlines[f]
            scanlines = scanlines_new

        if fields != "all":
            # I'd like to use catch_warnings, but this triggers
            # http://bugs.python.org/issue29672 thus flooding my screen
            # for any *other* warning wherever this is called in a loop,
            # which it is.  Commenting out again :(
#            with warnings.catch_warnings():
                # selecting multiple fields from a structured masked array
                # leads to a FutureWarning, see
                # https://github.com/numpy/numpy/issues/8383 .
                # I believe this is a false alert.
#                warnings.filterwarnings("ignore", category=FutureWarning)
            scanlines = scanlines[fields]

        # TODO:
        # - Add other meta-information from TIP
        extra = {"header": header}
        return (scanlines, extra)
       
    def _add_pseudo_fields(self, M, pseudo_fields):
        if isinstance(M, tuple):
            return (M[0], super()._add_pseudo_fields(M[1], pseudo_fields))
        else:
            return super()._add_pseudo_fields(M, pseudo_fields)

    def check_parity(self, counts):
        """Verify parity for counts
        
        NOAA KLM Users Guide – April 2014 Revision, Section 3.2.2.4,
        Page 3-31, Table 3.2.2.4-1:

        > Minor Word Parity Check is the last bit of each minor Frame
        > or data element and is inserted to make the total number of
        > “ones” in that data element odd. This permits checking for
        > loss of data integrity between transmission from the instrument
        > and reconstruction on the ground.

        """
        raise NotImplementedError("Parity checking not implemented yet.")

    def rad2bt(self, rad_wn, wn, c1, c2):
        """Apply the standard radiance-to-BT conversion from NOAA KLM User's Guide.

        Applies the standard radiance-to-BT conversion as documented by
        the NOAA KLM User's Guide.  This is based on a linearisation of a
        radiance-to-BT mapping for the entire channel.  A more accurate
        method is available in typhon.physics.em.SRF.channel_radiance2bt,
        which requires explicit consideration of the SRF.  Such
        consideration is implicit here.  That means that this method
        is only valid assuming the nominal SRF!

        This method relies on values reported in the header of each
        granule.  See NOAA KLM User's Guide, Table 8.3.1.5.2.1-1., page
        8-108.  Please convert to SI units first.

        NOAA KLM User's Guide, Section 7.2.

        :param rad_wn: Spectral radiance per wanenumber
            [W·sr^{-1}·m^{-2}·{m^{-1}}^{-1}]
        :param wn: Central wavenumber [m^{-1}].
            Note that unprefixed SI units are used.
        :param c1: c1 as contained in hrs_h_tempradcnv
        :param c2: c2 as contained in hrs_h_tempradcnv
        """

        # if possible, ensure it's in base — not needed if I already rely
        # on pint beyond
#        try:
#            rad_wn = rad_wn.to(ureg. W / (ureg.m**2 * ureg.sr * (1/ureg.m)))
#        except AttributeError:
#            pass
        #rad_f = em.specrad_wavenumber2frequency(rad_wn)
        rad_f = rad_wn.to(rad_u["si"], "radiance")
        # standard inverse Planck function
        T_uncorr = em.specrad_frequency_to_planck_bt(rad_f,
            wn.to(ureg.Hz, "sp"))#em.wavenumber2frequency(wn))

        # fails with FloatingPointError…
        # see https://github.com/numpy/numpy/issues/4895
        #T_corr = (T_uncorr - ureg.Quantity(c1, ureg.K))/c2
        T_corr = ureg.Quantity(numpy.ma.MaskedArray(
                (T_uncorr.m.data - c1)/c2,
                mask=T_uncorr.mask),
            ureg.K)

        return T_corr

    def id2no(self, satid):
        """Translate satellite id to satellite number.

        Sources:
        - POD guide, Table 2.0.4-3.
        - KLM User's Guide, Table 8.3.1.5.2.1-1.
        - KLM User's Guide, Table 8.3.1.5.2.2-1.

        WARNING: Does not support NOAA-13 or TIROS-N!
        """

        return _tovs_defs.HIRS_ids[self.version][satid]

    def id2name(self, satid):
        """Translate satellite id to satellite name.

        See also id2no.

        WARNING: Does not support NOAA-13 or TIROS-N!
        """
        
        return _tovs_defs.HIRS_names[self.version][satid]

    # translation from HIRS.l1b format documentation to dtypes
    _trans_tovs2dtype = {"C": "|S",
                         "I1": ">i1",
                         "I2": ">i2",
                         "I4": ">i4"}
    _cmd = ("pdftotext", "-f", "{first}", "-l", "{last}", "-layout",
            "{pdf}", "{txt}")
    @classmethod
    def get_definition_from_PDF(cls, path_to_pdf):
        """Get HIRS definition from NWPSAF PDF.

        This method needs the external program pdftotext.  Put the result
        in header_dtype manually, but there are some corrections (see
        comments in source code in _tovs_defs).

        :param str path_to_pdf: Path to document
            NWPSAF-MF-UD-003_Formats.pdf
        :returns: (head_dtype, head_format, line_dtype, line_format)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfile = tmpdir + "/def"
            subprocess.check_call([a.format(
                first=cls.pdf_definition_pages[0],
                last=cls.pdf_definition_pages[1], pdf=path_to_pdf,
                txt=tmpfile) for a in cls._cmd])
#            head_fmt.seek(0, io.SEEK_END)
#            line_fmt.seek(0, io.SEEK_END)
            head_dtype = []
            line_dtype = []
            with open(tmpfile, encoding="utf-8") as tf:
                for line in tf:
                    if not line.strip().startswith("hrs"):
                        continue
                    (name, type, ws, nw, *descr) = line.strip().split()
                    dtp = head_dtype if name.startswith("hrs_h") else line_dtype
                    dtp.append(
                        (name,
                         cls._trans_tovs2dtype[type] + 
                                (ws if type=="C" else ""),
                         tools.safe_eval(nw)))
        return (head_dtype, line_dtype)

    def _apply_scale_factors(self, header, scanlines):
        #new_head_dtype = self.header_dtype.descr.copy()
        new_head_dtype = header.dtype.descr.copy()
        new_line_dtype = scanlines.dtype.descr.copy()
        for (i, dt) in enumerate(header.dtype.descr):
            if dt[0] in _tovs_defs.HIRS_scale_factors[self.version]:
                new_head_dtype[i] = (dt[0], ">f8") + dt[2:]
        for (i, dt) in enumerate(scanlines.dtype.descr):
            if dt[0] in _tovs_defs.HIRS_scale_factors[self.version]:
                new_line_dtype[i] = (dt[0], ">f8") + dt[2:]
        new_head = numpy.empty(shape=header.shape, dtype=new_head_dtype)
        new_line = numpy.empty(shape=scanlines.shape, dtype=new_line_dtype)
        for (targ, src) in [(new_head, header), (new_line, scanlines)]:
            for f in targ.dtype.names:
                # NB: I can't simply say targ[f] = src[f] / 10**0, because
                # this will turn it into a float and refuse to cast it
                # into an int dtype
                if f in _tovs_defs.HIRS_scale_factors[self.version]:
                    # FIXME: does this work for many scanlines?
                    targ[f] = src[f] / numpy.power(
                            _tovs_defs.HIRS_scale_bases[self.version],
                            _tovs_defs.HIRS_scale_factors[self.version][f])
                else:
                    targ[f] = src[f]
        return (new_head, new_line)

    def get_iwt(self, header, elem):
        """Get temperature of internal warm target
        """
        (iwt_fact, iwt_counts) = self._get_iwt_info(header, elem)
        return self._convert_temp(iwt_fact, iwt_counts)
    
    #@staticmethod
    def _convert_temp(self, fact, counts):
        """Convert counts to temperatures based on factors.

        Relevant to IWT, ICT, filter wheel, telescope, etc.

        Conversion is based on polynomial expression

        a_0 + a_1 * c_0 + a_2 * c_1^2 + ...

        Source related to HIRS/2 and HIRS/2I, but should be the same for
        HIRS/3 and HIRS/4.  Would be good to confirm this.

        Also flag outliers.

        Source: NOAA Polar Satellite Calibration: A System Description.
            NOAA Technical Report, NESDIS 77

        TODO: Verify outcome according to Sensor Temperature Ranges
            HIRS/3: KLM, Table 3.2.1.2.1-1.
            HIRS/4: KLM, Table 3.2.2.2.1-1.

        """

        # FIXME: Should be able to merge those conditions into a single
        # expression with some clever use of Ellipsis
        N = fact.shape[-1]
        if counts.ndim == 3:
            tmp = (counts[:, :, :, numpy.newaxis].astype("double") **
                    numpy.arange(1, N)[numpy.newaxis, numpy.newaxis, numpy.newaxis, :])
            M = (fact[:, 0:1] +
                        (fact[:, numpy.newaxis, 1:] * tmp).sum(3))
        elif counts.ndim == 2:
            tmp = (counts[..., numpy.newaxis].astype("double") **
                   numpy.arange(1, N).reshape((1,)*counts.ndim + (N-1,))) 
            M = fact[0:1] + (fact[numpy.newaxis, numpy.newaxis, 1:] * tmp).sum(-1)
        elif counts.ndim == 1:
            fact = fact.squeeze()
            M = (fact[0] + 
                    (fact[numpy.newaxis, 1:] * (counts[:, numpy.newaxis].astype("double")
                        ** numpy.arange(1, N)[numpy.newaxis, :])).sum(1))
        else:
            raise NotImplementedError("ndim = {:d}".format(counts.ndim))

        M = numpy.ma.asarray(M)
        return M

    @abc.abstractmethod
    def get_wn_c1_c2(self, header):
        """Read central wavenumber, c1, and c2 from header

        Given a header such as returned by self.read, return central
        wavenumber and the coefficients c1 and c2.
        """
        ...

    @abc.abstractmethod
    def seekhead(self, f):
        """Seek open file to header position.

        For some files, CLASS prepends 512 bytes to the file.  This method
        shall make sure the file pointer is in the correct position to
        start reading.
        """
        ...

    @abc.abstractmethod
    def calibrate(self, cc, counts):
        ...
            
    @abc.abstractmethod
    def get_mask_from_flags(self, header, lines, max_flagged=0.5):
        """Set mask in lines, based on header and lines info

        Given header and lines such as returned by self.read, determine
        flags and set them on lines as appropriate.  Returns lines as a
        masked array.
        """
        ...

    @abc.abstractmethod
    def get_cc(self, scanlines):
        """Extract calibration coefficients from scanlines.

        """
        ...

    @abc.abstractmethod
    def get_dtypes(self, fp):
        """Get dtypes for file.

        Needs an open file object.  Used internally by reading routine.
        """
        ...

    @abc.abstractmethod
    def get_pos(scanlines):
        """Get lat-lon from scanlines.

        """
        ...

    @abc.abstractmethod
    def _get_time(scanlines):
        """Read time from scanlines.

        Shall return an ndarray with M8[s] dtype.
        """
        ...

    @abc.abstractmethod
    def get_other(scanlines):
        """Get other information from scanlines.

        Exact content depends on implementation.  Can be things like
        scantype, solar zenith angles, etc.
        """
        ...

    def get_temp(self, header, elem, anwrd):
        """Get temperatures from header, element, anwrd

        Used internally.  FIXME DOC.
        """
        # note: subclasses should still expand this
        N = elem.shape[0]
        return dict(
            iwt = self._convert_temp(*self._get_iwt_info(header, elem)),
            ict = self._convert_temp(*self._get_ict_info(header, elem)),
            fwh = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_fwcnttmp"),
                    elem[:, 60, 2:22].reshape(N, 4, 5)),
            scanmirror = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_scmircnttmp"),
                    elem[:, 62, 2]),
            primtlscp = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_pttcnttmp"),
                    elem[:, 62, 3]),
            sectlscp = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_sttcnttmp"),
                    elem[:, 62, 4]),
            baseplate = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_bpcnttmp"),
                    elem[:, 62, 5]),
            elec = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_electcnttmp"),
                    elem[:, 62, 6]),
            patch_full = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_patchfcnttmp"),
                    elem[:, 62, 7]),
            scanmotor = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_scmotcnttmp"),
                    elem[:, 62, 8]),
            fwm = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_fwmcnttmp"),
                    elem[:, 62, 9]),
            ch = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_chsgcnttmp"),
                    elem[:, 62, 10]),
        )

    def _reshape_fact(self, name, fact, robust=False):
        if name in self._fact_shapes:
            try:
                return fact.reshape(self._fact_shapes[name])
            except ValueError:
                if robust:
                    return fact
                else:
                    raise
        else:
            return fact


    def _get_temp_factor(self, head, name):
        satname = self.id2name(head["hrs_h_satid"][0])
        # TIROS-N and NOAA-11 have same code...
        if satname == "NOAA11" and ((numpy.ascontiguousarray(
                head["hrs_h_startdatadatetime"]).view(">u2")[0] & 0xfe00)
                >> 9) < 1985:
            satname = "TIROSN"
        fact = _tovs_defs.HIRS_count_to_temp[satname][name[6:]]
        return self._reshape_fact(name, fact, robust=True)

    def _get_iwt_info(self, head, elem):
        iwt_counts = elem[:, 58, self.count_start:self.count_end].reshape(
            (elem.shape[0], 4, 5))
        iwt_fact = self._get_temp_factor(head, "hrs_h_iwtcnttmp")
        return (iwt_fact, iwt_counts)

    def _get_ict_info(self, head, elem):
        ict_counts = elem[:, 59, self.count_start:self.count_end]
        ict_counts = ict_counts.reshape(elem.shape[0], 4, 5)
        ict_fact = self._get_temp_factor(head, "hrs_h_ictcnttmp")
        return (ict_fact, ict_counts)

    @abc.abstractmethod
    def get_dataname(self, header, robust=False):
        """Extract dataname from header.
        """

    def calc_time_since_last_calib(self, M):
        """Calculate time since last calibration.

        Calculate time (in seconds) since the last calibration cycle.

        Arguments:

            M [ndarray]

                ndarray of the same type as returned by self.read.  Must
                be contiguous or results will be wrong.

        Returns:

            ndarray, seconds since last calibration cycle
        """

        # explicit loop may be somewhat slower than broadcasting, but
        # broadcasting is O(N²) in memory — not acceptable!
        ix_iwt = (M[self.scantype_fieldname]==self.typ_iwt).nonzero()[0]
        time_iwt = M["time"][M[self.scantype_fieldname]==self.typ_iwt]
        tsc = numpy.ma.masked_all(shape=M.shape, dtype="m8[ms]")
        for i in range(ix_iwt.shape[0]):
            lst = ix_iwt[i]
            try:
                nxt = ix_iwt[i+1]
            except IndexError:
                nxt = time_iwt.shape[0]
            tsc[lst:nxt] = M["time"][lst:nxt] - M["time"][lst]
            
        return tsc.astype("m8[ms]").astype("f4")/1000

    def count_lines_since_last_calib(self, M):
        """Count scanlines since last calibration.

        Count scanlines since the last calibration cycle.

        Arguments:

            M [ndarray]

                ndarray of the same type as returned by self.read.  Must
                be contiguous or results will be wrong.

        Returns:

            ndarray (uint16), scanlines since last calibration cycle
        """

        # explicit loop may be somewhat slower than broadcasting, but
        # broadcasting is O(N²) in memory — not acceptable!
        ix_iwt = (M[self.scantype_fieldname]==self.typ_iwt).nonzero()[0]
        time_iwt = M["time"][M[self.scantype_fieldname]==self.typ_iwt]
        lsc = numpy.ma.masked_all(shape=M.shape, dtype="u2")
        for i in range(ix_iwt.shape[0]):
            lst = ix_iwt[i]
            try:
                nxt = ix_iwt[i+1]
            except IndexError:
                nxt = time_iwt.shape[0]
            lsc[lst:nxt] = numpy.arange(0, nxt-lst)
            
        return lsc

    def as_xarray_dataset(self, M, skip_dimensions=(),
                          rename_dimensions={}):
        """Convert structured ndarray to xarray dataset

        From an object with a dtype such as may be returned by self.read,
        return an xarray dataset.

        This method is in flux and its API is currently not stable.  There
        needs to be a proper system of defining the variable names etc.

        See tickets 145, 148, 149.

        NB: xarray does not support masked arrays (see
        https://github.com/pydata/xarray/issues/1194).  Therefore, I
        convert each integer type to float/double.  To prevent a loss of
        precision, I double the memory allocation for each, as int8
        always fits in float16, int16 always fits in float32, and int32
        always fits in float64.  Note that assuming the encoding is set
        correctly in data_vars_props (such as in _tovs_defs) it will still
        be stored as the appropriate integer type.

        Arguments:

            M [ndarray]

                ndarray of same type as returned by self.read.

            skip_dimensions [Container[str]]

                dimensions that shall not be included.  For example,
                normal HIRS data has a scanpos dimension, HIRS-HIRS
                collocations do not; to convert collocaitons, pass
                skip_dimensions=["scanpos"].

            rename_dimensions [Mapping[str,str]]

                dimensions that shall be renamed.  For example, for
                collocations you may want to rename "time" to
                "scanline" or to "collocation".
        """

        p = self._data_vars_props
        add_to_all = {"source": "copied from NOAA native L1B format"}
        origvar = "original_name_l1b"
        data_vars = {
            p[v][0]:
               ([rename_dimensions.get(d,d) for d in p[v][1] if d not in skip_dimensions],
                (M[v].data
                 if isinstance(M, numpy.ma.MaskedArray)
                 else M[v]).astype( 
                    M[v].dtype
                    if M[v].dtype.kind[0] in "MOSUVmcfb" or "bit" in p[v][0]
                    else "f{:d}".format(M[v].dtype.itemsize*2)),
                {**p[v][2],
                 **add_to_all,
                 **{origvar: v}})
            for v in p.keys() & set(M.dtype.names)}

        coords = dict(
            time = (("time",), M["time"]),
            scanline = (("time",), numpy.arange(M.shape[0])),
            scanpos = (("scanpos",), numpy.arange(1, self.n_perline+1)),
            channel = (("channel",), numpy.arange(1, self.n_channels+1)),
            calibrated_channel = (("calibrated_channel",), numpy.arange(1, self.n_calibchannels+1)),
                )

        if "lon" in M.dtype.names:
            coords["lon"] = (("time", "scanpos"), M["lon"])
        if "lat" in M.dtype.names:
            coords["lat"] = (("time", "scanpos"), M["lat"])

        coords = {rename_dimensions.get(k, k):
                ([rename_dimensions.get(d, d) for d in v[0] if d not in skip_dimensions],
                 v[1],
                 {**(p[k][2] if k in p else {}),
                  **(add_to_all if k in M.dtype.names else {}),
                  **({origvar: k} if k in M.dtype.names else {})})
                for (k, v) in coords.items() if k not in skip_dimensions}

        ds = xarray.Dataset(
            {k:v for (k,v) in data_vars.items() if not k in coords.keys()},
            coords,
            {"title": "HIRS L1C"})

        for v in p.keys() & set(M.dtype.names):
            ds[p[v][0]].encoding = p[v][3]
            if ("bit" not in p[v][0] and
                ds[p[v][0]].dtype.kind not in "Mm" and
                hasattr(M[v], "mask")):
                ds[p[v][0]].values[M[v].mask] = numpy.nan #(
#                numpy.nan if M[v].dtype.kind.startswith('f')
#                else p[v][3]["_FillValue"])
        
        return ds

    def flagscore(self, M):
        """Calculate "flag score"; higher is worse.

        To aid the selection of the "best" scanline between subsequenc
        scanlines, calculate a "flag score".
        """
        score = numpy.zeros(shape=M.shape, dtype="f4")

        qif = _tovs_defs.QualIndFlagsHIRS[self.version]
        mff = _tovs_defs.MinorFrameFlagsHIRS[self.version]

        qi = M["hrs_qualind"]
        mf = M["hrs_mnfrqual"]

        for flag in qif:
            score += ((qi & flag)!=0)
        # add extra penalty for donotuse
        score += 10 * ((qi & qif.qidonotuse)!=0)

        for flag in mff:
            score += ((mf & flag)!=0).mean(1)

        return score

class HIRSPOD(HIRS):
    """Read early HIRS such as documented in POD guide.

    Methods are mostly as for HIRS class.
    """
    n_wordperframe = 22
    counts_offset = 0

    typ_ict = 2 # internal cold calibration target; only used on HIRS/2
    views = ("ict", "iwt", "space", "Earth")
    scantype_fieldname = "scantype"

    # “distance” between space and IWCT views
    dist_space_iwct = 2

    # HIRS/2 has LZA only for the edge of the scan.  Linear interpolation
    # is not good enough; scale with a single reference array for other
    # positions.  Reference array taken from
    # NSS.HIRX.M2.D06325.S1200.E1340.B0046667.SV
    ref_lza = (
        numpy.array([ 59.19,  56.66,  54.21,  51.82,  49.48,  47.19,  44.93,  42.7 ,
        40.5 ,  38.33,  36.17,  34.03,  31.91,  29.8 ,  27.7 ,  25.61,
        23.53,  21.46,  19.4 ,  17.34,  15.29,  13.24,  11.2 ,   9.16,
         7.12,   5.09,   3.05,   1.02,   1.01,   3.05,   5.08,   7.12,
         9.15,  11.19,  13.24,  15.28,  17.34,  19.39,  21.46,  23.53,
        25.6 ,  27.69,  29.79,  31.9 ,  34.02,  36.16,  38.32,  40.49,
        42.69,  44.92,  47.18,  49.47,  51.81,  54.2 ,  56.65,  59.18]))

    flag_fields = {"hrs_qualind"}
    # docstring in parent
    def seekhead(self, f):
        f.seek(0, io.SEEK_SET)

    def calibrate(self, cc, counts):
        """Apply the standard calibration from NOAA POD Guide

        Returns radiance in SI units (W m^-2 sr^-1 Hz^-1).

        POD Guide, section 4.5
        """

        # Equation 4.5-1
        # should normally have no effect as channels should be linear,
        # according to POD Guide, page 4-26
        # order is 0th, 1st, 2nd order term
        nc = cc[:, numpy.newaxis, :, 2, :]
        counts = nc[..., 0] + nc[..., 1] * counts + nc[..., 2] * counts**2

        # Equation 4.5.1-1
        # Use auto-coefficient.  There's also manual coefficient.
        # order is 2nd, 1st, 0th order term
        ac = cc[:, numpy.newaxis, :, 1, :]
        rad = ac[..., 2] + ac[..., 1] * counts + ac[..., 0] * counts**2

        if not (cc[:, :, 0, :]==0).all():
            logging.warning("Found non-zero values for manual coefficient! "
                "Usually those are zero but when they aren't, I don't know "
                "which ones to use.  Use with care. ")

        # This is apparently calibrated in units of mW/m2-sr-cm-1.
        rad = ureg.Quantity(rad, rad_u["ir"])

        return rad

    # docstring in parent class
    def get_wn_c1_c2(self, header):
        h =  _tovs_defs.HIRS_coeffs[self.version][self.id2no(header["hrs_h_satid"][0])]
        return numpy.vstack([h[i] for i in range(1, 20)]).T

    # docstring in parent class   
    def get_mask_from_flags(self, header, lines, max_flagged=0.5):
        # for flag bits, see POD User's Guide, page 4-4 and 4-5.

        # quality indicators
        qi = lines["hrs_qualind"]
        qidict = dict(
            qifatal =    qi & (1<<31),
            qitimeseqerr =  qi & (1<<30),
            qidatagap =     qi & (1<<29),
            qidwell =       qi & (1<<28),
            qidatafill =    qi & (1<<27),
            qidacs =        qi & (1<<26),
            # 25-24: scan type
            qimirrorlocked =qi & (1<<23),
            qimirrorpos =   qi & (1<<22),
            qimirrorrepos = qi & (1<<21),
            qifiltersync =  qi & (1<<20),
            qiscanpattern = qi & (1<<19),
            qicalibration = qi & (1<<18),
            qinoearth     = qi & (1<<17),
            qiearthloc_delta   = qi & (1<<16),
            qibitsync     = qi & (1<<15),
            qisyncerror   = qi & (1<<14),
            qiframesync   = qi & (1<<13),
            qiflywheel    = qi & (1<<12),
            qibitslip     = qi & (1<<11),
            qitipparity   = qi & (1<<10),
            qiauxbiterror = qi & (1<<9),
            # 8: spare
            # 0-7: counters
        )

        # FIXME: minor frame quality, POD User's Guide, page 4-10
        bad_bt = (lines["hrs_qualind"] & 0xcffffe00) != 0
        earthcounts = lines["hrs_qualind"] & 0x03000000 == 0
        calibcounts = ~earthcounts
        # treat earth and calib counts separately; a stuck mirror is not a
        # problem in calibration mode, and some mirror repositioning etc.
        # may even be on purpose
        bad_earthcounts = earthcounts & ((lines["hrs_qualind"] & 0xccfbfe00) != 0)
        bad_calibcounts = calibcounts & ((lines["hrs_qualind"] & 0xccdbfe00) != 0)
        # different for non-earth-views

        if "bt" in lines.dtype.names:
            lines["bt"].mask[bad_bt, :, :] = True
        if "radiance" in lines.dtype.names:
            lines["radiance"].mask[bad_bt, :, :] = True
        if "counts" in lines.dtype.names:
            lines["counts"].mask[bad_earthcounts|bad_calibcounts, :, :] = True

            if lines["counts"].mask.sum() > lines["counts"].size*max_flagged:
                raise dataset.InvalidDataError(
                    "Excessive amount of flagged data ({:.2%}). ".format(
                        lines["counts"].mask.sum()/lines["counts"].size) +
                    ', '.join("{:s} ({:.2%})".format(k[2:], (v!=0).sum()/v.size) for (k, v)
                        in qidict.items() if (v!=0).sum()/v.size > 0.01))

        return lines

    # unfinished
#     def process_elem(self):
#         out_of_sync =        (tiptop &  (1<< 6)) >> 6 == 0
#         element_number =     (tiptop & ((1<<13) - (1<<7)))  >>  7
#         ch1_period_monitor = (tiptop & ((1<<19) - (1<<13))) >> 13
#         el_cal_level =       (tiptop & ((1<<24) - (1<<19))) >> 19
#         encoder_position =   (tiptop & ((1<<32) - (1<<24))) >> 24
#         
    # docstring in parent
    def get_cc(self, scanlines):
        cc = scanlines["hrs_calcof"].reshape(scanlines.shape[0], 3,
                self.n_channels, 3)
        cc = numpy.swapaxes(cc, 2, 1)
        return cc
        
    # docstring in parent
    def get_temp(self, header, elem, anwrd):
        N = elem.shape[0]
        D = super().get_temp(header, elem, anwrd)
        # FIXME: need to add temperatures from minor frame 62 here.  See
        # NOAA POD GUIDE, chapter 4, page 4-8 (PDF page 8)
        #
        # FIXME: Also minor frame 61: patch_exp, fsr, ...
#            patch_exp = self._convert_temp(
#                    self._get_temp_factor(header, "hrs_h_patchexpcnttmp").reshape(1, 6),
#                    elem[:, 61, 2:7].reshape(N, 1, 5)),
#            fsr = self._convert_temp(
#                    self._get_temp_factor(header, "hrs_h_fsradcnttmp").reshape(1, 6),
#                    elem[:, 61, 7:12].reshape(N, 1, 5)))
        D.update(dict(
            patch_exp = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_patchexpcnttmp").reshape(1, 5),
                    elem[:, 61, 2:7].reshape(N, 1, 5)),
            fsr = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_fsradcnttmp").reshape(1, 5),
                    elem[:, 61, 7:12].reshape(N, 1, 5))))
        return D

    # docstring in parent
    @staticmethod
    def _get_time(scanlines):
        # NOAA POD User's Guide, page 4-4
        # year is "contained in first 7 bits of first 2 bytes"
        # But despite having 7 bits it only uses 2 digits and resets from
        # 99 to 0 after 2000
        year02 = ((numpy.ascontiguousarray(
            scanlines["hrs_scnlintime"]).view(">u2").reshape(
                -1, 3)[:, 0] & 0xfe00) >> 9)
        year = numpy.where(year02<70, year02+2000, year02+1900) 
        # doy is "right-justified in first two bytes"
        doy = (numpy.ascontiguousarray(
            scanlines["hrs_scnlintime"]).view(">u2").reshape(
                -1, 3)[:, 0] & 0x01ff)
        # "27-bit millisecond UTC time of day is right-justified in last
        # four bytes"
        # Make sure we interpret those four bytes as big-endian!
        time_ms = ((
            numpy.ascontiguousarray(
                numpy.ascontiguousarray(
                    scanlines["hrs_scnlintime"]
                ).view("uint16").reshape(-1, 3)[:, 1:]
            ).view(">u4")) & 0xffffffe0)
        return (year.astype("M8[Y]") - 1970 +
                (doy-1).astype("m8[D]") +
                 time_ms.astype("m8[ms]").squeeze())

    # docstring in parent
    @staticmethod
    def get_pos(scanlines):
        # See POD User's Guide, page 4-7
        lat = scanlines["hrs_pos"][:, ::2] / 128
        lon = scanlines["hrs_pos"][:, 1::2] / 128
        return (lat, lon)

    # docstring in parent
    def get_other(self, scanlines):
        # See POD User's Guide, page 4-7
        # not actually available for HIRS/2
        # Use reference from HIRS/4 (MetOp-A) along with single lza value
        # given for HIRS/2 to “scale up” full array
        M = numpy.empty(shape=scanlines.shape,
            dtype=[
                ("scantype", "i1"),
                ("lza_approx", "f4", self.n_perline),
                ])
        M["scantype"] = (scanlines["hrs_qualind"] & 0x03000000)>>24
        #M["lza_approx"] = ((scanlines["hrs_satloc"][:, 1]/128)
        M["lza_approx"] = ((scanlines["hrs_lza"][:]/128)
            / self.ref_lza[0])[:, numpy.newaxis] * self.ref_lza[numpy.newaxis, :]
        return M

    def get_dtypes(self, f):
        """Get dtypes for header and lines

        Takes as argument fp to open granule file.

        Before 1995, a record was 4256 bytes.
        After 1995, it became 4253 bytes.
        This change appears undocumented but can be find in the AAPP
        source code at AAPP/src/tools/bin/hirs2_class_to_aapp.F
        """
        pos = f.tell()
        # check starting year
        self.seekhead(f)
        f.seek(2, io.SEEK_SET)
        year = ((numpy.frombuffer(f.peek(2), "<u2", count=1)[0] 
                            & 0xfe) >> 1)
        year += (2000 if year < 70 else 1900)
        if year < 1995:
            hd =  _tovs_defs.HIRS_header_dtypes[2][4256]
            ln =  _tovs_defs.HIRS_line_dtypes[2][4256]
        else:
            hd = _tovs_defs.HIRS_header_dtypes[2][4253]
            ln = _tovs_defs.HIRS_line_dtypes[2][4253]
        f.seek(pos, io.SEEK_SET)
        return (hd, ln)

    # docstring in parent
    def get_dataname(self, header, robust=False):
        # See POD User's Guide, page 2-6; this is in EBCDIC
        dataname = header["hrs_h_dataname"][0].decode("EBCDIC-CP-BE")
        if len(dataname) == 0:
            msg = ("Dataname empty for {satname!s}.  "
                "I have noticed this problem for: TIROS-N (all data), "
                "NOAA-6 (before 1985-10-31 08:27), NOAA-7 (all data), "
                "NOAA-8 (all data), NOAA-9 (before 1985-10-31 13:30). "
                "Those data appear to have a different header.  If you "
                "happen to know documentation for ancient headers, please "
                "e-mail g.holl@reading.ac.uk!".format(satname=self.satname))
            if robust:
                dataname = (
                    str(header["hrs_h_startdatadatetime"][0]) +
                    str(header["hrs_h_enddatadatetime"][0]) +
                    str(header["hrs_h_satid"][0])).replace("\\", "").replace("'", "")
                warnings.warn(msg + " Trying ad-hoc dataname for now.")
            else:
                raise ValueError(msg)
        return dataname

# docstring in parent
class HIRS2(HIRSPOD):
    """Sole implementation of HIRSPOD
    """
    satellites = {"tirosn": {"TIROSN", "TN", "tn", "tirosn", "NOAA05", "NOAA5", "noaa05", "noaa5", "N05", "n05", "N5", "n5"},
                  "noaa06": {"NOAA06", "NOAA6", "noaa06", "noaa6", "N06", "n06", "N6", "n6"},
                  "noaa07": {"NOAA07", "NOAA7", "noaa07", "noaa7", "N07", "n07", "N7", "n7"},
                  "noaa08": {"NOAA08", "NOAA8", "noaa08", "noaa8", "N08", "n08", "N8", "n8"},
                  "noaa09": {"NOAA09", "NOAA9", "noaa09", "noaa9", "N09", "n09", "N9", "n9"},
                  "noaa10": {"NOAA10", "noaa10", "N10", "n10"},
                  "noaa11": {"NOAA11", "noaa11", "N11", "n11"},
                  "noaa12": {"NOAA12", "noaa12", "N12", "n12"},
                  "noaa13": {"NOAA13", "noaa13", "N13", "n13"},
                  "noaa14": {"NOAA14", "noaa14", "N14", "n14"}}
    version = 2

    channel_order = numpy.asarray(_tovs_defs.HIRS_channel_order[2])

    start_date = datetime.datetime(1978, 10, 29)
    end_date = datetime.datetime(2006, 10, 10)
    
# docstring in parent
class HIRS2I(HIRS2):
    # identical fileformat, as far as I can tell.
    # Differ only in channel positions.
    satellites = {"noaa11", "noaa14"}

# docstring in parent
class HIRSKLM(ATOVS, HIRS):
    counts_offset = 4096
    n_wordperframe = 24
    views = ("iwt", "space", "Earth")
    scantype_fieldname = "hrs_scntyp"

    # “distance” between space and IWCT views
    dist_space_iwct = 1

    flag_fields = {"hrs_qualind", "hrs_linqualflgs", "hrs_mnfrqual",
                   "hrs_elem", "hrs_chqualflg"}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature_fields |= {"an_rd", "an_baseplate", "an_el",
            "an_pch", "an_scnm", "an_fwm", "patch_exp", "fsr"}

    # docstring in parent
    def seekhead(self, f):
        f.seek(0, io.SEEK_SET)
        if f.peek(3)[:3] in {b"NSS", b"CMS", b"DSS", b"UKM"}:
            f.seek(0, io.SEEK_SET)
        else: # assuming additional header
            f.seek(512, io.SEEK_SET)
            if not f.peek(3)[:3] in {b"NSS", b"CMS", b"DSS", b"UKM"}:
                raise dataset.InvalidFileError(
                    "Could not find header in {:s}".format(f.name))

    def calibrate(self, cc, counts):
        """Apply the standard calibration from NOAA KLM User's Guide.

        NOAA KLM User's Guide, section 7.2, equation (7.2-3), page 7-12,
        PDF page 286:

        r = a₀ + a₁C + a₂²C

        where C are counts and a₀, a₁, a₂ contained in hrs_calcof as
        documented in the NOAA KLM User's Guide:
        - Section 8.3.1.5.3.1, Table 8.3.1.5.3.1-1. and
        - Section 8.3.1.5.3.2, Table 8.3.1.5.3.2-1.,
        """
        rad = (cc[:, numpy.newaxis, :, 2]
             + cc[:, numpy.newaxis, :, 1] * counts 
             + cc[:, numpy.newaxis, :, 0] * counts**2)
        # This is apparently calibrated in units of mW/m2-sr-cm-1.
        # Convert to SI units.
        rad = ureg.Quantity(rad, rad_u["ir"])
#            ureg.mW / (ureg.m**2 * ureg.sr * (1/ureg.cm)))
#        # Convert to SI base units (not needed anymore with pint)
#        rad = rad.to(ureg.W / (ureg.m**2 * ureg.sr * (1/ureg.m)),
#            "radiance")
        return rad

    # docstring in parent
    def get_wn_c1_c2(self, header):
        return header["hrs_h_tempradcnv"].reshape(self.n_calibchannels, 3).T

    # docstring in parent
    def get_mask_from_flags(self, header, lines, max_flagged=0.5):
        # These four entries are contained in each data frame and consider
        # the quality of the entire frame.  See Table 8.3.1.5.3.1-1. and
        # Table 8.3.1.5.3.2-1., 
        # However, it is too drastic to reject everything, because some
        # flags mean "some channel uncalibrated", for example.  This does
        # not affect counts.
        #
        # In practice, the usefulness of these indicators depends a lot on
        # the satellite.  For example, for NOAA-15, a lot of useful data
        # is flagged and a lot of outliers are unflagged.

        # FIXME!  Those have changed between HIRS/3 and HIRS/4 — FIXME!
        # FIXME!  How are they in MetOp-A — FIXME!
        #
        # On second thought — these flags have so many false negatives and
        # false positives, that it is difficult to use them in practice.

        # FIXME: This should be rewritten to use
        # _tovs_defs.QualIndFlagsHIRSKLM and its children

        # quality indicators
        qi = lines["hrs_qualind"]
        qidonotuse =    qi & (1<<31)
        qitimeseqerr =  qi & (1<<30)
        qidatagap =     qi & (1<<29)
        qinofullcalib = qi & (1<<28)
        qinoearthloc =  qi & (1<<27)
        qifirstgood =   qi & (1<<26)
        qistatchng =    qi & (1<<25)

        lq = lines["hrs_linqualflgs"]
        # time problems
        tmbadcanfix =   lq & (1<<23)
        tmbadnofix =    lq & (1<<22)
        tmnotcnstnt =   lq & (1<<21)
        tmrpt =         lq & (1<<20)

        # calibration anomalies
        cabadtime =     lq & (1<<15)
        cafewer =       lq & (1<<14)
        cabadprt =      lq & (1<<13)
        camargprt =     lq & (1<<12)
        cachmiss =      lq & (1<<11)
        cainstmode =    lq & (1<<10)
        camoon =        lq & (1<<9)

        # channel-specific in subclass

        # earth location problems
        elbadtime =     lq & (1<<7)
        elquestime =    lq & (1<<6)
        elmargreason =  lq & (1<<5)
        elunreason =    lq & (1<<4)

        # minor frame
        mf = lines["hrs_mnfrqual"]
        mfsusptime =    mf & (1<<7)
        mfhasfill =     mf & (1<<6)
        mfhastipdwell = mf & (1<<5)
        mfsusppacsqc =  mf & (1<<4)
        mfmirlock =     mf & (1<<3)
        mfmirposerr =   mf & (1<<2)
        mfmirmoved =    mf & (1<<1)
        # last bit is parity, but I can't seem to figure out how to use
        # it.  It doesn't seem to work at all, so I'll ignore it.

        
        # which ones imply bad BT?
        # which ones imply bad counts?

        #badline = (lines["hrs_qualind"] | lines["hrs_linqualflgs"]) != 0
        #badchan = lines["hrs_chqualflg"] != 0
        # Does this sometimes mask too much?
        #badmnrframe = lines["hrs_mnfrqual"] != 0
        # NOAA KLM User's Guide, page 8-154: Table 8.3.1.5.3.1-1.
        # consider flag for “valid”
        elem = lines["hrs_elem"].reshape(lines.shape[0], 64, 24)
        cnt_flags = elem[:, :, 22]
        mfvalid = ((cnt_flags & 0x8000) >> 15) == 1
        #badmnrframe |= (~valid)
        # When I assume that HIRS's "odd parity bit" is really an "even
        # parity bit", I get bad parity for ~0.2% of cases.  If I assume
        # the documentation is correct, I get bad parity for 99.8% of
        # cases.  The parity bit is the second bit (i.e. 0x4000).
        badparity = (((cnt_flags & 0x8000) >> 15) ^
                     ((cnt_flags & 0x4000) >> 14)) == 1
        #return (badline, badchannel, badmnrframe)

    
        for fld in set(lines.dtype.names) - {"lat", "lon", "time"}:
            # only for the most serious offences
            # ...but normally still leave lat/lon/time intact...

            lines[fld].mask |= qidonotuse.reshape(([lines.shape[0]] +
                    [1]*(lines[fld].ndim-1)))!=0

        # If time is no longer sequential we still mask it.   Caller might
        # want to get rid of the lines altogether.  Remask
        # qidonotuse-marked times if they do not occur after previous and
        # before next.  Note that this ONLY takes care of bad times
        # already marked as qidonotuse; other problematic times need to be
        # taken care of elsewhere and are out of the scope of this function
        delta_t = numpy.diff(lines["time"])
        # take care of "too early":
        # we always accept the time for the very first measurement, so we
        # call .nonzero() on qidonotuse[1:] and compensate for the
        # off-by-one-error thus introduced.  
        #lines["time"].mask[qidonotuse.nonzero()[0][
        lines["time"].mask[(qidonotuse[1:].nonzero()[0]+1)[
            (numpy.sign(delta_t[(qidonotuse!=0)[1:]].astype(numpy.int64)) != 1)]] = True
        # take care of "late outliers" (alway accept the very last
        # measurement)
        lines["time"].mask[(qidonotuse[:-1].nonzero()[0])[
            (numpy.sign(delta_t[(qidonotuse!=0)[:-1]].astype(numpy.int64)) != 1)]] = True

        for fld in ("counts", "bt"):
            # Where a channel is bad, mask the entire scanline
            # NB: BT has only 19 channels
            #lines[fld].mask |= badchan[:, numpy.newaxis, :lines[fld].shape[2]]

            # Where a minor frame is bad or parity fails, mask all channels
            #lines[fld].mask |= badmnrframe[:, :56, numpy.newaxis]
            #lines[fld].mask |= badparity[:, :56, numpy.newaxis]

            # Where an entire line is bad, mask all channels at entire
            # scanline
            #lines[fld].mask |= badline[:, numpy.newaxis, numpy.newaxis]
            lines[fld].mask |= camoon[:, numpy.newaxis, numpy.newaxis]!=0

            if header["hrs_h_instid"][0] in {306, 307}:
                # MetOp seems to always have "mirror position error"!  I
                # can't afford to reject data.
                bm = 0xfa
            else:
                # mirror moved, position error, or locked
                bm = 0xfe
            lines[fld].mask |= (mf & bm)[:, :self.n_perline, numpy.newaxis]!=0
            

        # Some lines are marked as space view or black body view
        for v in ("bt", "radiance"):
            lines[v].mask |= (lines["hrs_scntyp"] != self.typ_Earth)[:, numpy.newaxis, numpy.newaxis]

            # Where radiances are negative, mask individual values as masked
            lines[v][:, :, :19].mask |= (lines["radiance"][:, :, :19] <= 0)

            # Where counts==0, mask individual values
            # WARNING: counts==0 is within the valid range for some channels!
            lines[v][:, :, :19].mask |= (elem[:, :self.n_perline, 2:21]==0)

        if lines["counts"].mask.sum() > lines["counts"].size*max_flagged:
            raise dataset.InvalidDataError(
                "Excessive amount of flagged data ({:.2%}). "
                "Moon ({:.2%}), mirror position error ({:.2%}), "
                "mirror moved ({:.2%}).".format(
                    lines["counts"].mask.sum()/lines["counts"].size,
                    (camoon!=0).sum()/camoon.size,
                    (mfmirposerr[:, :self.n_perline]!=0).sum()/
                     mfmirposerr[:, :self.n_perline].size,
                    (mfmirmoved[:, :self.n_perline]!=0).sum()/
                     mfmirmoved[:, :self.n_perline].size))

        return lines

    # docstring in parent
    def get_cc(self, scanlines):
        cc = scanlines["hrs_calcof"].reshape(scanlines.shape[0], self.n_channels, 
                scanlines.dtype["hrs_calcof"].shape[0]//self.n_channels)
        return cc

    # docstring in parent
    def get_temp(self, header, elem, anwrd):
        N = elem.shape[0]
        D = super().get_temp(header, elem, anwrd)
        D.update(dict(
            an_rd = self._convert_temp_analog(
                    header[0]["hrs_h_rdtemp"],
                    anwrd[:, 0]), # ok
            an_baseplate = self._convert_temp_analog(
                    header[0]["hrs_h_bptemp"],
                    anwrd[:, 1]), # bad
            an_el = self._convert_temp_analog(
                    header[0]["hrs_h_eltemp"],
                    anwrd[:, 2]), # OK
            an_pch = self._convert_temp_analog(
                    header[0]["hrs_h_pchtemp"],
                    anwrd[:, 3]), # OK
            an_scnm = self._convert_temp_analog(
                    header[0]["hrs_h_scnmtemp"],
                    anwrd[:, 5]), # bad
            an_fwm = self._convert_temp_analog(
                    header[0]["hrs_h_fwmtemp"],
                    anwrd[:, 6])), # bad
            patch_exp = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_patchexpcnttmp").reshape(1, 6),
                    elem[:, 61, 2:7].reshape(N, 1, 5)),
            fsr = self._convert_temp(
                    self._get_temp_factor(header, "hrs_h_fsradcnttmp").reshape(1, 6),
                    elem[:, 61, 7:12].reshape(N, 1, 5)))
        return D

    def _convert_temp_analog(self, F, C):
        V = C.astype("float64")*0.02
        return (F * V[:, numpy.newaxis]**numpy.arange(F.shape[0])[numpy.newaxis, :]).sum(1)

    @staticmethod
    def read_cpids(path):
        """Read calibration parameters input data sets (CPIDS)

        Should contain a CPIDS file for HIRS, such as NK.cpids.HIRS.
        Read telemetry conversiot data from a Calibration Parameters Input
        Data Sets (CPIDS) source file, such as available at NOAA.  Some
        were sent by Dejiang Han <dejiang.han@noaa.gov> to Gerrit Holl
        <g.holl@reading.ac.uk> on 2016-02-17.
        """

        D = {}
        with path.open(mode="rb") as fp:
            fp.readline()
            analogcc = numpy.genfromtxt(fp, max_rows=16, dtype="f4")
            fp.readline()
            fp.readline()
            digatcc = numpy.genfromtxt(fp, max_rows=11, dtype="f4")
            fp.readline()
            fp.readline()
            digalc1 = numpy.genfromtxt(fp, max_rows=1, dtype="f4")
            digalc2 = numpy.genfromtxt(fp, max_rows=1, dtype="f4")
            fp.readline()
            fp.readline()
            # filter wheel housing
            D["fwcnttemp"] = numpy.genfromtxt(fp, max_rows=4, dtype="f4")
            fp.readline()
            fp.readline()
            D["ictcnttmp"] = numpy.genfromtxt(fp, max_rows=4, dtype="f4")
            fp.readline()
            fp.readline()
            D["iwtcnttmp"] = numpy.genfromtxt(fp, max_rows=5, dtype="f4")
            fp.readline()
            fp.readline()
            D["sttcnttmp"] = numpy.genfromtxt(fp, max_rows=1, dtype="f4")


        D.update(zip(
            "an_rdtemp an_bptemp an_eltemp an_pchtemp an_fhcc "
            "an_scnmtemp an_fwmtemp an_p5v an_p10v an_p75v an_m75v "
            "an_p15v an_m15v an_fwmcur an_scmcur "
            "an_pchcpow".split(), analogcc))

        D.update(zip(
             "tttcnttmp patchexpcnttmp fsradcnttmp scmircnttmp "
             "pttcnttmp bpcnttmp electcnttmp patchfcnttmp scmotcnttmp "
             "fwmcnttmp".split(), digatcc))

        D.update(zip(
            "fwthc ecdac pcp smccc fmccc p15vdccc m15vdccc p7.5vdccc "
            "m7.5vdccc p10vdccc".split(), digalc1))

        D["p5vdccc"] = digalc2.squeeze()

        return D

    # docstring in parent
    @staticmethod
    def get_pos(scanlines):
        lat = scanlines["hrs_pos"][:, ::2]
        lon = scanlines["hrs_pos"][:, 1::2]
        return (lat, lon)

    @staticmethod
    def _get_time(scanlines):
        # according to http://stackoverflow.com/a/26807879/974555
        # I need to pass the class now?
        return super(HIRSKLM, HIRSKLM)._get_time(scanlines, prefix="hrs_")

    # docstring in parent
    def get_other(self, scanlines):
        M = numpy.empty(shape=(scanlines.shape[0],),
            dtype=[
                ("sol_za", "f4", self.n_perline),
                ("sat_za", "f4", self.n_perline),
                ("loc_aa", "f4", self.n_perline)])
        M["sol_za"] = scanlines["hrs_ang"][:, ::3]
        M["sat_za"] = scanlines["hrs_ang"][:, 1::3]
        M["loc_aa"] = scanlines["hrs_ang"][:, 2::3]
        return M

    # docstring in parent
    def get_dtypes(self, f):
        return (self.header_dtype, self.line_dtype)

    # docstring in parent
    def get_dataname(self, header, robust=False):
        return header["hrs_h_dataname"][0].decode("US-ASCII")

    def flagscore(self, M):
        score = super().flagscore(M)

        lqf = _tovs_defs.LinQualFlagsHIRS[self.version]
        chf = _tovs_defs.ChQualFlagsHIRS[self.version]

        lq = M["hrs_linqualflgs"]
        ch = M["hrs_chqualflg"]

        for flag in lqf:
            score += ((lq & flag)!=0)

        for flag in chf:
            score += ((ch & flag)!=0).mean(1)

        return score

# docstring in parent
class HIRS3(HIRSKLM):
    pdf_definition_pages = (26, 37)
    version = 3

    satellites = {"noaa15": {"NOAA15", "noaa15", "N15", "n15"},
                  "noaa16": {"NOAA16", "noaa16", "N16", "n16"},
                  "noaa17": {"NOAA17", "noaa17", "N17", "n17"}}

    header_dtype = _tovs_defs.HIRS_header_dtypes[3]
    line_dtype = _tovs_defs.HIRS_line_dtypes[3]

    channel_order = numpy.asarray(_tovs_defs.HIRS_channel_order[3])

    start_date = datetime.datetime(1999, 1, 1)
    end_date = datetime.datetime(2016, 12, 31) # update as appropriate

    def get_mask_from_flags(self, header, lines, max_flagged=0.5):
        lines = super().get_mask_from_flags(header, lines,
            max_flagged=max_flagged)

        # channel quality indicators
        cq = lines["hrs_chqualflg"][:, numpy.argsort(self.channel_order)]
        cqbadbb =       cq & (1<<5)
        cqbadsv =       cq & (1<<4)
        cqbadprt =      cq & (1<<3)
        cqmargbb =      cq & (1<<2)
        cqmargsv =      cq & (1<<1)
        cqmargprt =     cq & 1

        return lines

# docstring in parent
class HIRS4(HIRSKLM):
    satellites = {"noaa18": {"NOAA18", "noaa18", "N18", "n18"},
                  "noaa19": {"NOAA19", "noaa19", "N19", "n19"},
                  "metopa": {"METOPA", "metopa", "MA", "ma"},
                  "metopb": {"METOPB", "metopb", "MB", "mb"}}
    pdf_definition_pages = (38, 54)
    version = 4

    header_dtype = _tovs_defs.HIRS_header_dtypes[4]
    line_dtype = _tovs_defs.HIRS_line_dtypes[4]
    
    channel_order = numpy.asarray(_tovs_defs.HIRS_channel_order[4])
    
    _fact_shapes = {
        "hrs_h_ictcnttmp": (4, 6),
        "hrs_h_fwcnttmp": (4, 6)}

    start_date = datetime.datetime(2005, 6, 5)
    end_date = datetime.datetime(2016, 12, 31) # update as appropriate

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature_fields.add("terttlscp")

    def _get_iwt_info(self, head, elem):
        iwt_counts = numpy.concatenate(
            (elem[:, 58, self.count_start:self.count_end],
             elem[:, 59, 12:17]), 1).reshape((elem.shape[0], 5, 5))
        iwt_fact = self._get_temp_factor(head, "hrs_h_iwtcnttmp").reshape(5, 6)
        iwt_counts = iwt_counts.astype("int64")
        return (iwt_fact, iwt_counts)

    def _get_ict_info(self, head, elem):
        ict_counts = elem[:, 59, 2:7]
        ict_fact = self._get_temp_factor(head, "hrs_h_ictcnttmp")[0, :6]
        return (ict_fact, ict_counts)

    def _get_temp_factor(self, head, name):
        return self._reshape_fact(name, head[name])

    def get_temp(self, header, elem, anwrd):
        """Extract temperatures
        """
        D = super().get_temp(header, elem, anwrd)
        # new in HIRS/4
        D["terttlscp"] = self._convert_temp(
            header["hrs_h_tttcnttmp"],
            elem[:, 59, 17:22].reshape(elem.shape[0], 1, 5))
        return D

    def get_mask_from_flags(self, header, lines, max_flagged=0.5):
        lines = super().get_mask_from_flags(header, lines, max_flagged=max_flagged)

        # channel quality indicators
        cq = lines["hrs_chqualflg"][:, numpy.argsort(self.channel_order)]
        cqfailed=       cq & (1<<5)
        cqanom  =       cq & (1<<4)
        cqslopehcf =    cq & (1<<3)
        cqbbnedc =      cq & (1<<2)
        cqspnedc =      cq & (1<<1)
        cqnotappl =     cq & 1

        #badchan = (cqnotappl!=0) | (cqspnedc!=0) | (cqbbnedc!=0)
        badchan = (cq & 0x07) != 0
        
        for v in ("counts", "bt", "radiance"):
            lines[v].mask |= badchan[:, numpy.newaxis, :lines[v].shape[2]]

        return lines

class IASIEPS(dataset.MultiFileDataset, dataset.HyperSpectral):
    """Read IASI from EUMETSAT EPS L1C

    This class depends on the Common Data Access Toolbox (CODA).

    http://stcorp.nl/coda/

    Reading data requires that in TYPHONRC, the variables 'tmpdir' and
    'tmpdirb' are set in [main].
    """

    name = section = "iasi"
    start_date = datetime.datetime(2007, 5,  29, 5, 8, 56)
    end_date = datetime.datetime(2015, 11, 17, 16, 38, 59)
    granule_duration = datetime.timedelta(seconds=6200)
    _dtype = numpy.dtype([
        ("time", "M8[ms]"),
        ("lat", "f4", (4,)),
        ("lon", "f4", (4,)),
        ("satellite_zenith_angle", "f4", (4,)),
        ("satellite_azimuth_angle", "f4", (4,)),
        ("solar_zenith_angle", "f4", (4,)),
        ("solar_azimuth_angle", "f4", (4,)),
        ("spectral_radiance", "f4", (4, 8700))])

    # Minimum temporary space for unpacking
    # Warning: race conditions needs to be addressed.
    # As a workaround, choose very large minspace.
    minspace = 1e10

    @staticmethod
    def __obtain_from_mdr(c, field):
        fieldall = numpy.concatenate([getattr(x.MDR, field)[:, :, :,
            numpy.newaxis] for x in c.MDR if hasattr(x, 'MDR')], 3)
        fieldall = numpy.transpose(fieldall, [3, 0, 1, 2])
        return fieldall

    def _read(self, path, fields="all"):
        tmpdira = config.conf["main"]["tmpdir"]
        tmpdirb = config.conf["main"]["tmpdirb"]
        tmpdir = (tmpdira 
            if shutil.disk_usage(tmpdira).free > self.minspace
            else tmpdirb)
            
        # FIXME: this could use typhon.utils.decompress
        with tempfile.NamedTemporaryFile(mode="wb", dir=tmpdir, delete=True) as tmpfile:
            with gzip.open(str(path), "rb") as gzfile:
                logging.debug("Decompressing {!s}".format(path))
                gzcont = gzfile.read()
                logging.debug("Writing decompressed file to {!s}".format(tmpfile.name))
                tmpfile.write(gzcont)
                del gzcont

            # All the hard work is in coda
            logging.debug("Reading {!s}".format(tmpfile.name))
            cfp = coda.open(tmpfile.name)
            c = coda.fetch(cfp)
            logging.debug("Sorting info...")
            n_scanlines = c.MPHR.TOTAL_MDR
            start = datetime.datetime(*coda.time_double_to_parts_utc(c.MPHR.SENSING_START))
            has_mdr = numpy.array([hasattr(m, 'MDR') for m in c.MDR],
                        dtype=numpy.bool)
            bad = numpy.array([
                (m.MDR.DEGRADED_PROC_MDR|m.MDR.DEGRADED_INST_MDR)
                        if hasattr(m, 'MDR') else True
                        for m in c.MDR],
                            dtype=numpy.bool)
            dlt = numpy.concatenate(
                [m.MDR.OnboardUTC[:, numpy.newaxis]
                    for m in c.MDR
                    if hasattr(m, 'MDR')], 1) - c.MPHR.SENSING_START
            M = numpy.ma.zeros(
                dtype=self._dtype,
                shape=(n_scanlines, 30))
            M["time"][has_mdr] = numpy.datetime64(start, "ms") + numpy.array(dlt*1e3, "m8[ms]").T
            specall = self.__obtain_from_mdr(c, "GS1cSpect").astype("f8")
            # apply scale factors
            first = c.MDR[0].MDR.IDefNsfirst1b
            last = c.MDR[0].MDR.IDefNslast1b
            for (slc_st, slc_fi, fact) in zip(
                    filter(None, c.GIADR_ScaleFactors.IDefScaleSondNsfirst),
                    c.GIADR_ScaleFactors.IDefScaleSondNslast,
                    c.GIADR_ScaleFactors.IDefScaleSondScaleFactor):
                # Documented intervals are closed [a, b]; Python uses
                # half-open [a, b).
                specall[..., (slc_st-first):(slc_fi-first+1)] *= pow(10.0, -fact)
            M["spectral_radiance"][has_mdr] = specall
            locall = self.__obtain_from_mdr(c, "GGeoSondLoc")
            M["lon"][has_mdr] = locall[:, :, :, 0]
            M["lat"][has_mdr] = locall[:, :, :, 1]
            satangall = self.__obtain_from_mdr(c, "GGeoSondAnglesMETOP")
            M["satellite_zenith_angle"][has_mdr] = satangall[:, :, :, 0]
            M["satellite_azimuth_angle"][has_mdr] = satangall[:, :, :, 1]
            solangall = self.__obtain_from_mdr(c, "GGeoSondAnglesSUN")
            M["solar_zenith_angle"][has_mdr] = solangall[:, :, :, 0]
            M["solar_azimuth_angle"][has_mdr] = solangall[:, :, :, 1]
            for fld in M.dtype.names:
                M.mask[fld][~has_mdr, ...] = True
                M.mask[fld][bad, ...] = True
            m = c.MDR[0].MDR
            wavenumber = (m.IDefSpectDWn1b * numpy.arange(m.IDefNsfirst1b, m.IDefNslast1b+0.1) * (1/ureg.metre))
            if self.wavenumber is None:
                self.wavenumber = wavenumber
            elif abs(self.wavenumber - wavenumber).max() > (0.05 * 1/(ureg.centimetre)):
                raise ValueError("Inconsistent wavenumbers")
            return (M, {})

class IASISub(dataset.HomemadeDataset, dataset.HyperSpectral):
    name = section = "iasisub"
    subdir = "{month}"
    stored_name = "IASI_1C_selection_{year}_{month}_{day}.npz"
    re = r"IASI_1C_selection_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<day>\d{1,2}).npz"
    start_date = datetime.datetime(2011, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2011, 12, 31, 23, 59, 59)

    
    def _read(self, *args, **kwargs):
        if self.frequency is None:
            self.frequency = numpy.loadtxt(self.freqfile)
        return super()._read(*args, **kwargs)

    def get_times_for_granule(self, p, **kwargs):
        gd = self.get_info_for_granule(p)
        (year, month, day) = (int(gd[m]) for m in "year month day".split())
        # FIXME: this isn't accurate, it usually starts slightly later...
        start = datetime.datetime(year, month, day, 0, 0, 0)
        # FIXME: this isn't accurate, there may be some in the next day...
        end = datetime.datetime(year, month, day, 23, 59, 59)
        return (start, end)

class TOVSCollocatedDataset:
    """Mixin for any TOVS collocated dataset.  Different because of scanlines.

    Should be mixed in before Dataset
    """

    colloc_dim = "UNDEFINED"

    def combine(self, M, other_obj, *args, col_field,
            col_field_slice=slice(None), trans, col_dim_name=None, **kwargs):
        MM = super().combine(M, other_obj, *args, trans=trans, **kwargs)
        if self.read_returns == "ndarray":
            # do something about entire scanlines being returned
            scnlin_names = [f[0] for f in MM.dtype.descr 
                if len(f)>2 and f[2][0]==other_obj.n_perline]
            # strip out scanline dimension
            new_dtp = [(f[0], f[1], tuple(i for i in f[2] if i!=other_obj.n_perline))
                        if f[0] in scnlin_names else f for f in MM.dtype.descr]
            idx_all = numpy.arange(M.size)
            MM_new = numpy.ma.zeros(dtype=new_dtp, shape=MM.shape)
            for fld in MM.dtype.names:
                if fld in scnlin_names:
                    # see http://stackoverflow.com/a/23435869/974555
                    # Ashwini Chaudhary, 2014-05-02
                    # NB: I believe this does not count as a derivative
                    # work and that the CC-BY-SA 3.0 licensing conditions
                    # do not apply here
                    MM_new[fld][...] = MM[fld][idx_all, 
                        M[col_field][col_field_slice], ...]
                else:
                    MM_new[fld][...] = MM[fld][...]
            return MM_new
        elif self.read_returns == "xarray":
            scnlin_names = [k for (k, v) in MM.data_vars.items()
                if col_dim_name in v.dims]
            (my_var, other_var) = next(iter(trans.items()))
            my_dim = M[my_var].dims[0]
            other_dim = MM[other_var].dims[0]
            # repeat the above-linked SO feat but for xarray…  does not
            # appear to work directly, need to pass through .values and
            # carefully translate the right order of dimensions to the
            # correct indices
            
            idx_all = numpy.arange(M.dims[self.colloc_dim])
            for varname in scnlin_names:
                MM[varname] = (
                    [self.colloc_dim if MM[varname].coords[d].dtype.kind=="M" else d
                     for d in MM[varname].dims if d != col_dim_name],
                         MM[varname].values[
                            tuple(M[col_field].values if d == col_dim_name
                            else idx_all if MM[varname].coords[d].dtype.kind == "M"
                            else slice(None) for d in MM[varname].dims)
                         ])
            # how parent class returned it, there are now several
            # equally-sized dimensions all related to time: time,
            # scanline_earth, calibration_cycle, etc.  Make sure they all
            # get the dimension matchup_count or whatever I use for the
            # collocations.

            # NB:  I want to drop the dimensions, but keep the
            # coordinates using the matchup_number dimension.  A simple
            # rename would rename the coordinates, such that matchup_number ends
            # up having the contents of whatever coordinate came last, and
            # the rest vanishes; need to reassign right coordinates after
            # renaming.  As of xarray 0.10, I need to manually drop them
            # first or xarray will complain that "the new name
            # 'matchup_count' conflicts"
            timedims = utils.get_time_dimensions(MM)
            MM = MM.drop(timedims).rename(dict.fromkeys(timedims, self.colloc_dim)
                ).assign_coords(
                    **{d: (self.colloc_dim, MM.coords[d].values)
                       for d in utils.get_time_dimensions(MM)},
                    **{self.colloc_dim: M[self.colloc_dim].values})
            return MM
        else:
            raise ValueError("read_returns must be xarray or ndarray")


class HIASI(TOVSCollocatedDataset, dataset.NetCDFDataset, dataset.MultiFileDataset, dataset.HyperSpectral):
    """"HIRS-IASI collocations
    """
    name = section = "hiasi"
    subdir = "{year:04d}/{month:02d}"
    re = (r"W_XX-EUMETSAT-Darmstadt,SATCAL\+COLLOC\+LEOLEOIR,"
          r"opa\+HIRS\+M02\+IASI_C_EUMS_(?P<year>\d{4})(?P<month>\d{2})"
          r"(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})_"
          r"(?P<year_end>\d{4})(?P<month_end>\d{2})(?P<day_end>\d{2})"
          r"(?P<hour_end>\d{2})(?P<minute_end>\d{2})(?P<second_end>\d{2})\.nc")
    start_date = datetime.datetime(2013, 1, 1)
    end_date = datetime.datetime(2014, 1, 1)
    colloc_dim = "line"
    concat_coor = "line"
    
    def _read(self, f, fields="all"):
        (M, extra) = super()._read(f, fields)
        if self.read_returns == "ndarray":
            # functionality in numpy.lib.recfunctions.append_fields is too
            # slow!
            MM = numpy.zeros(shape=M.shape,
                             dtype=M.dtype.descr + [("time", "M8[s]")])
            MM["time"] = M["mon_time"].astype("M8[s]")
            for f in M.dtype.names:
                MM[f][...] = M[f][...]
            return (MM[numpy.argsort(MM["time"])], extra)
        elif self.read_returns == "xarray":
            M["time"] = ("line", M["ref_time"])
            M = utils.undo_xarray_floatification(M)
            # make units understandable by UADA / pint
            M["ref_radiance"].attrs["units"] = \
                    M["ref_radiance"].attrs["units"] \
                                     .replace("m2", "m**2") \
                                     .replace("^-1", "**(-1)")
            # make sure data are sorted in time
            M = M.loc[{"line": numpy.argsort(M["time"])}]
            return (M, extra)
        else:
                raise ValueError(f"Invalid read_returns: {self.read_returns:s}")

    def combine(self, M, other_obj, *args, **kwargs):
        MM = super().combine(M, other_obj, *args, col_field="mon_column",
                             col_dim_name="scanpos", **kwargs)
        # do something about entire scanlines being returned
        return MM

class HIRSHIRS(TOVSCollocatedDataset, dataset.NetCDFDataset, dataset.MultiFileDataset):
    """HIRS-HIRS collocations from Brockmann Consult

    A.k.a. MMD05
    """
    name = section = "hirshirs"
    subdir = "hirs_{prim:s}_{sec:s}"
    re = (r"mmd05_hirs-(?P<prim>.{2,3})_hirs-(?P<sec>.{2,3})_"
          r"(?P<year>\d{4})-(?P<doy>\d{3})_(?P<year_end>\d{4})-"
          r"(?P<doy_end>\d{3})\.nc")
    start_date = HIRS2.start_date
    end_date = HIRS4.end_date

    colloc_dim = "matchup_count"
    concat_coor = "matchup_count"

    def _read(self, f, fields="all"):
        (M, extra) = super()._read(f, fields)

        if self.read_returns == "ndarray":
            timefields = [x for x in M.dtype.descr
                          if x[0].endswith("acquisition_time")]
            if len(timefields) != 2:
                raise dataset.InvalidFileError("Expected 2 "
                    "fields for time, found {:d}".format(len(timefields)))

            MM = numpy.ma.zeros(shape=M.shape,
                             dtype=M.dtype.descr +
                             [("alltime", "M8[s]", timefields[0][2]
                                if len(timefields[0])>2 else ()),
                              ("time", "M8[s]")])

            # Always use primary time so that time is always increasing
    #        MM["alltime"] = (
    #            numpy.ma.masked_less_equal(M[timefields[0][0]], 0)*.5 +
    #            numpy.ma.masked_less_equal(M[timefields[1][0]], 0)*.5
    #                        ).astype("M8[s]")
            MM["alltime"] = (
                numpy.ma.masked_less_equal(M[timefields[0][0]], 0)
                            ).astype("M8[s]")
            MM["time"] = MM["alltime"][:, int(MM["alltime"].shape[1]//2+1),
                                          int(MM["alltime"].shape[2]//2+1)]
            for fld in M.dtype.fields:
                MM[fld][...] = M[fld][...]
            # it appears there may still have been overlaps in the data that
            # went into the matchups... try to fix that here
            # BUG!  (time1, time2) is not unique!  Time is per SCANLINE,
            # so unique would be (time1, x1, time2, x2)!
            ii = numpy.argsort(M[timefields[0][0]][:, 3, 3])
            Msrt = M[ii]
            (_, iiu) = numpy.unique(
                numpy.c_[
                    Msrt[timefields[0][0]][:, 3, 3],
                    Msrt[timefields[1][0]][:, 3, 3]
                        ].data.view("i4,i4"),
                return_index=True)
            if iiu.size < MM.size:
                logging.warning("There were duplicates in {!s}!  Removing " 
                "{:.2%}".format(f, 1-iiu.size/MM.size))
            return (MM[ii][iiu], extra)
        elif self.read_returns == "xarray":
            # in some files, the dtype for x and y is turned into float
            # by xarray because it has a _FillValue attribute
            M = utils.undo_xarray_floatification(M)
            # acquisition_time has attribute "unit" instead of "units";
            # use regular "time" instead
            timefields = [k for k in M.data_vars.keys()
                          if k.endswith("time")
                     and not k.endswith("acquisition_time")]
            
            # estimate corrected time per pixel depending on location in
            # scan position; this prevents false detection of
            # duplicates and unsorted data.
            for p in (x[5:-3] for x in M.dims.keys() if x.endswith("nx")):
                M["hirs-{:s}_time".format(p)] += (M["hirs-{:s}_x".format(p)]*100).astype("m8[ms]")

            if len(timefields) != 2:
                raise dataset.InvalidFileError("Expected 2 "
                    "fields for time, found {:d}".format(len(timefields)))

            # need to manually flag bad times as this wasn't done earlier
            # in the processing
            for fld in timefields:
                M[fld].values[(M[fld].values.astype("M8[ms]")
                    > datetime.datetime.now()) |
                              (M[fld].values.astype("M8[ms]")
                    < datetime.datetime(1975, 1, 1))] = numpy.datetime64("NaT")

            tm1 = M[timefields[0]][:, 3, 3]
            tm2 = M[timefields[1]][:, 3, 3]

            # NB: tm1+tm2 is invalid so calculate average like this
            newtime = tm1 + (tm2-tm1)/2
            M["time"] = ((self.colloc_dim,), newtime)

            # BUG!  (time1, time2) is not unique!  Time is per SCANLINE,
            # so unique would be (time1, x1, time2, x2)!
            ii = numpy.argsort(M[timefields[0]][:, 3, 3])
            Ms = M.loc[
                {self.colloc_dim: ii,
                 **{k: 3 for k in M.dims.keys()
                    if k != self.colloc_dim}}][timefields]
            Msn = numpy.c_[Ms[timefields[0]].values, Ms[timefields[1]].values]

            # see http://stackoverflow.com/a/16973510/974555
            # Posted by Jaime, 2013-06-06.
            # I believe this does not count as a derivative work and that
            # specific CC-BY-SA 3.0 conditions do not apply here.
            Msnu = numpy.ascontiguousarray(Msn).view(
                numpy.dtype((numpy.void, Msn.dtype.itemsize * 2)))

            (_, iiu) = numpy.unique(Msnu, return_index=True)
            if iiu.size < M.dims[self.colloc_dim]:
                logging.warning("There were duplicates in {!s}!  Removing "
                    "{:.2%}".format(f, 1-iiu.size/M.dims[self.colloc_dim]))

            M = M[{self.colloc_dim: iiu}]
            M = M[{self.colloc_dim: numpy.argsort(M["time"])}]

            return (M, extra)
            
        else:
            raise ValueError("read_returns should be xarray or ndarray, "
                "found {!s}".format(self.read_returns))

    def combine(self, M, other_obj, *args, col_field, **kwargs):
        return super().combine(M, other_obj, *args, col_field=col_field,
            col_field_slice=(slice(None), 3, 3), **kwargs)

class MHSL1C(ATOVS, dataset.NetCDFDataset, dataset.MultiFileDataset):
    name = section = "mhs_l1c"
    subdir = "{satname:s}_mhs_{year:04d}/{month:02d}/{day:02d}"
    re = (r"(\d*\.)?NSS.MHSX.(?P<satcode>.{2})\.D(?P<year>\d{2})"
          r"(?P<doy>\d{3})\.S(?P<hour>\d{2})(?P<minute>\d{2})\.E"
          r"(?P<hour_end>\d{2})(?P<minute_end>\d{2})\.B(?P<B>\d{7})\.(?P<station>.{2})\.h5")
    start_date = datetime.datetime(1999, 1, 1)
    end_date = datetime.datetime(2016, 4, 1)

    def _get_dtype_from_vars(self, alldims, allvars, fields, prim):
        # MHS L1C has phony dims with repeated names between groups, so
        # need to determine by value (size) rather than name
        return numpy.dtype([
            (k,
             "f4" if (getattr(v, "Scale", 1)!=1) else v.dtype,
             tuple(s for (i, s) in enumerate(v.shape)
                 if v.shape[i] != alldims[prim].size))
             for (k, v) in allvars.items()
             if ((alldims[prim].size in v.shape) if fields=="all"
                  else (k in fields))])

    def _read(self, f, fields="all"):
        try:
            (M, extra) = super()._read(f, fields)
        except ValueError as e:
            # some MHS L1C files have multiple 'phony_0' dimensions, even
            # though they should have different dimensions with the same
            # size.  This crashes my clever NetCDF reader.  Until I have a
            # bugfix, make a workaround.
            raise dataset.InvalidFileError("Encountered ValueError"
            "probably due to phony files...") from e
        # functionality in numpy.lib.recfunctions.append_fields is too
        # slow!
        MM = numpy.zeros(shape=M.shape,
                         dtype=M.dtype.descr + [("time", "M8[s]")])
        MM["time"] = self._get_time(M, prefix="")
        #MM["time"] = M["mon_time"].astype("M8[s]")
        for f in M.dtype.names:
            MM[f][...] = M[f][...]
        return (MM, extra)

def which_hirs(satname):
    """Given a satellite, return right HIRS object
    """
    for h in {HIRS2, HIRS3, HIRS4}:
        for (k, v) in h.satellites.items():
            if satname in {k}|v:
                return h(satname=satname)
    else:
        raise ValueError("Unknown HIRS satellite: {:s}".format(satname))

def norm_tovs_name(satname):
    """Given a TOVS satellite name, return normalised name

    """

    for h in {HIRS2, HIRS3, HIRS4}:
        for (k, v) in h.satellites.items():
            if satname in {k}|v:
                return k
    raise ValueError(f"Unknown TOVS satellite: {satname:s}")
