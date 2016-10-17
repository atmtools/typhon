"""Relevant definitions for TOVS
"""

import datetime

import numpy
import collections
from typhon.constants import K

from numpy import (float32, array)

# Sources:
#
# For HIRS/2:
#
# - NOAA Polar Orbiter Data User's Guide, POD Guide,
# http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/podug/index.htm
# http://www1.ncdc.noaa.gov/pub/data/satellite/publications/podguides/TIROS-N%20thru%20N-14/pdf/
#
# Header:
#   - Before 1992-09-08: Appendix K
#   - Between 1992-09-08 and 1994-11-15: Appendix L
#   - After 1994-11-15: Chapter 2, Table 2.0.4-1
#
# Body:
#   - Chapter 4
#
# For HIRS/3 and HIRS/4:
#
# - NOAA KLM User's Guide,
# http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/klm/index.htm
# http://www1.ncdc.noaa.gov/pub/data/satellite/publications/podguides/N-15%20thru%20N-19/pdf/0.0%20NOAA%20KLM%20Users%20Guide.pdf
# -   and NWFSAF guide
# https://nwpsaf.eu/deliverables/aapp/NWPSAF-MF-UD-003_Formats.pdf
#
# HIRS/3, headers: Table 8.3.1.5.2.1-1., page 8-98 – 8-115
# HIRS/3, data: Table 8.3.1.5.3.1-1., page 8-142 — 8-169
# HIRS/4, headers: Table 8.3.1.5.2.2-1., page 8-115 – 8-142
# HIRS/4, data: Table 8.3.1.5.3.2-1., page 8-169 – 8-187

# I don't know a practical way of extracting scale factors automatically, as the
# NWPSAF document only lists them in comments/text, and I don't know how
# to automatically map the NWPsaf document to the KLM User's Guide.
# Values are obtained from NOAA KLM User's Guide, April 2014 revision.
#
# Note that scale factors are mostly defined in powers of 10, so a scale factor
# of 1 still means a multiplication factor of 10.  Some scale factors are
# defined as factors of 2.
#
# Scale factor should either be scalar, or match the size of one line of
# data.

HIRS_scale_factors = {}

_tmpsclfct = (6., 9., 14., 17., 21., 25.)
_tmpsclfct4 = numpy.tile(_tmpsclfct, 4)
_tmpsclfct5 = numpy.tile(_tmpsclfct, 5)

HIRS_scale_factors[3] = dict(
    hrs_h_calinf = 6,
    hrs_h_tempradcnv = numpy.concatenate((numpy.tile(6, 12*3), numpy.tile((5, 6, 6), 7))),
    hrs_h_iwtcnttmp = _tmpsclfct5,
    hrs_h_ictcnttmp = _tmpsclfct4,
    hrs_h_tttcnttmp = _tmpsclfct,
    hrs_h_fwcnttmp = _tmpsclfct4,
    hrs_h_patchexpcnttmp = _tmpsclfct,
    hrs_h_fsradcnttmp = _tmpsclfct,
    hrs_h_scmircnttmp = _tmpsclfct,
    hrs_h_pttcnttmp = _tmpsclfct,
    hrs_h_sttcnttmp = _tmpsclfct,
    hrs_h_bpcnttmp = _tmpsclfct,
    hrs_h_electcnttmp = _tmpsclfct,
    hrs_h_patchfcnttmp = _tmpsclfct,
    hrs_h_scmotcnttmp = _tmpsclfct,
    hrs_h_fwmcnttmp = _tmpsclfct,
    hrs_h_chsgcnttmp = _tmpsclfct,
    hrs_h_20solfiltirrad = 6,
    hrs_h_20equifiltwidth = 6,
    hrs_h_nadloctol = 1,
    hrs_h_rollerr = 3,
    hrs_h_pitcherr = 3,
    hrs_h_yawerr = 3,
    hrs_h_smaxis = 5,
    hrs_h_eccen = 8,
    hrs_h_incli = 5,
    hrs_h_argper = 5,
    hrs_h_rascnod = 5,
    hrs_h_manom = 5,
    hrs_h_xpos = 5,
    hrs_h_ypos = 5,
    hrs_h_zpos = 5,
    hrs_h_xvel = 8,
    hrs_h_yvel = 8,
    hrs_h_zvel = 8,
    hrs_h_earthsun = 6,
    hrs_h_rdtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_bptemp = (2, 2, 3, 3, 3, 5),
    hrs_h_eltemp = (2, 2, 3, 3, 3, 5),
    hrs_h_pchtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_fhcc = (2, 2, 3, 3, 3, 5),
    hrs_h_scnmtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_fwmtemp = (2, 2, 3, 3, 3, 5),
    hrs_h_p5v = (2, 2, 3, 3, 3, 5),
    hrs_h_p10v = (2, 2, 3, 3, 3, 5),
    hrs_h_p75v = (2, 2, 3, 3, 3, 5),
    hrs_h_m75v = (2, 2, 3, 3, 3, 5),
    hrs_h_p15v = (2, 2, 3, 3, 3, 5),
    hrs_h_m15v = (2, 2, 3, 3, 3, 5),
    hrs_h_fwmcur = (2, 2, 3, 3, 3, 5),
    # NOTE: In KLM User's Guide on page 8-114, Scan Motor Coefficient 2 is
    # missing.  Assuming this to be the same as coefficient 2 for all the
    # surrounding headers.
    hrs_h_scmcur = (2, 2, 3, 3, 3, 5),
    hrs_h_pchcpow = (2, 2, 3, 3, 3, 5),
    hrs_calcof = numpy.tile((12, 9, 6), 20),
    hrs_scalcof = numpy.tile((12, 9, 6), 20),
    hrs_rollang = 3,
    hrs_pitchang = 3,
    hrs_yawang = 3,
    hrs_scalti = 1,
    hrs_ang = 2,
    hrs_pos = 4)

HIRS_scale_factors[4] = HIRS_scale_factors[3].copy()

HIRS_scale_factors[2] = dict(
    # NB: normalisation coefficients have reversed order

    hrs_calcof = numpy.concatenate(
        (numpy.tile(numpy.array([44, 30, 22]), 40),
         numpy.tile(numpy.array([22, 30, 44]), 20)))
    )

HIRS_scale_bases = {}

HIRS_scale_bases[2] = 2
HIRS_scale_bases[3] = 10
HIRS_scale_bases[4] = 10

HIRS_header_dtypes = {}
HIRS_line_dtypes = {}

# Obtained using get_definition_from_PDF.  Please note correction!
HIRS_header_dtypes[3] = numpy.dtype([('hrs_h_siteid', '|S3', 1),
      ('hrs_h_blank', '|S1', 1),
      ('hrs_h_l1bversnb', '>i2', 1),
      ('hrs_h_l1bversyr', '>i2', 1),
      ('hrs_h_l1bversdy', '>i2', 1),
      ('hrs_h_reclg', '>i2', 1),
      ('hrs_h_blksz', '>i2', 1),
      ('hrs_h_hdrcnt', '>i2', 1),
      ('hrs_h_filler0', '>i2', 3),
      ('hrs_h_dataname', '|S42', 1),
      ('hrs_h_prblkid', '|S8', 1),
      ('hrs_h_satid', '>i2', 1),
      ('hrs_h_instid', '>i2', 1),
      ('hrs_h_datatyp', '>i2', 1),
      ('hrs_h_tipsrc', '>i2', 1),
      ('hrs_h_startdatajd', '>i4', 1),
      ('hrs_h_startdatayr', '>i2', 1),
      ('hrs_h_startdatady', '>i2', 1),
      ('hrs_h_startdatatime', '>i4', 1),
      ('hrs_h_enddatajd', '>i4', 1),
      ('hrs_h_enddatayr', '>i2', 1),
      ('hrs_h_enddatady', '>i2', 1),
      ('hrs_h_enddatatime', '>i4', 1),
      ('hrs_h_cpidsyr', '>i2', 1),
      ('hrs_h_cpidsdy', '>i2', 1),
      ('hrs_h_filler1', '>i2', 4),
      ('hrs_h_inststat1', '>i4', 1),
      ('hrs_h_filler2', '>i2', 1),
      ('hrs_h_statchrecnb', '>i2', 1),
      ('hrs_h_inststat2', '>i4', 1),
      ('hrs_h_scnlin', '>i2', 1),
      ('hrs_h_callocsclin', '>i2', 1),
      ('hrs_h_misscnlin', '>i2', 1),
      ('hrs_h_datagaps', '>i2', 1),
      ('hrs_h_okdatafr', '>i2', 1),
      ('hrs_h_pacsparityerr', '>i2', 1),
      ('hrs_h_auxsyncerrsum', '>i2', 1),
      ('hrs_h_timeseqerr', '>i2', 1),
      ('hrs_h_timeseqerrcode', '>i2', 1),
      ('hrs_h_socclockupind', '>i2', 1),
      ('hrs_h_locerrind', '>i2', 1),
      ('hrs_h_locerrcode', '>i2', 1),
      ('hrs_h_pacsstatfield', '>i2', 1),
      ('hrs_h_pacsdatasrc', '>i2', 1),
      ('hrs_h_filler3', '>i4', 1),
      ('hrs_h_spare1', '|S8', 1),
      ('hrs_h_spare2', '|S8', 1),
      ('hrs_h_filler4', '>i2', 5),
      ('hrs_h_autocalind', '>i2', 1),
      ('hrs_h_solarcalyr', '>i2', 1),
      ('hrs_h_solarcaldy', '>i2', 1),
      ('hrs_h_calinf', '>i4', 80),
      ('hrs_h_filler5', '>i4', 2),
      ('hrs_h_tempradcnv', '>i4', 57),
      ('hrs_h_20solfiltirrad', '>i2', 1),
      ('hrs_h_20equifiltwidth', '>i2', 1),
      # CORRECTION! NWPSAF guide says there is 1 field
      # here, but in reality it is 2 (see NOAA KLM User's
      # Guide, page 8-110, PDF page 421)
      ('hrs_h_filler6', '>i4', 2),
      ('hrs_h_modelid', '|S8', 1),
      ('hrs_h_nadloctol', '>i2', 1),
      ('hrs_h_locbit', '>i2', 1),
      ('hrs_h_filler7', '>i2', 1),
      ('hrs_h_rollerr', '>i2', 1),
      ('hrs_h_pitcherr', '>i2', 1),
      ('hrs_h_yawerr', '>i2', 1),
      ('hrs_h_epoyr', '>i2', 1),
      ('hrs_h_epody', '>i2', 1),
      ('hrs_h_epotime', '>i4', 1),
      ('hrs_h_smaxis', '>i4', 1),
      ('hrs_h_eccen', '>i4', 1),
      ('hrs_h_incli', '>i4', 1),
      ('hrs_h_argper', '>i4', 1),
      ('hrs_h_rascnod', '>i4', 1),
      ('hrs_h_manom', '>i4', 1),
      ('hrs_h_xpos', '>i4', 1),
      ('hrs_h_ypos', '>i4', 1),
      ('hrs_h_zpos', '>i4', 1),
      ('hrs_h_xvel', '>i4', 1),
      ('hrs_h_yvel', '>i4', 1),
      ('hrs_h_zvel', '>i4', 1),
      ('hrs_h_earthsun', '>i4', 1),
      ('hrs_h_filler8', '>i4', 4),
      ('hrs_h_rdtemp', '>i2', 6),
      ('hrs_h_bptemp', '>i2', 6),
      ('hrs_h_eltemp', '>i2', 6),
      ('hrs_h_pchtemp', '>i2', 6),
      ('hrs_h_fhcc', '>i2', 6),
      ('hrs_h_scnmtemp', '>i2', 6),
      ('hrs_h_fwmtemp', '>i2', 6),
      ('hrs_h_p5v', '>i2', 6),
      ('hrs_h_p10v', '>i2', 6),
      ('hrs_h_p75v', '>i2', 6),
      ('hrs_h_m75v', '>i2', 6),
      ('hrs_h_p15v', '>i2', 6),
      ('hrs_h_m15v', '>i2', 6),
      ('hrs_h_fwmcur', '>i2', 6),
      ('hrs_h_scmcur', '>i2', 6),
      ('hrs_h_pchcpow', '>i2', 6),
      # CORRECTION: Due to the earlier error, there's 889
      # left, not 890, for the total itemsize must remain
      # 4608
      ('hrs_h_filler9', '>i4', 890)])

HIRS_line_dtypes[3] = numpy.dtype([('hrs_scnlin', '>i2', 1),
      ('hrs_scnlinyr', '>i2', 1),
      ('hrs_scnlindy', '>i2', 1),
      ('hrs_clockdrift', '>i2', 1),
      ('hrs_scnlintime', '>i4', 1),
      ('hrs_scnlinf', '>i2', 1),
      ('hrs_mjfrcnt', '>i2', 1),
      ('hrs_scnpos', '>i2', 1),
      ('hrs_scntyp', '>i2', 1),
      ('hrs_filler1', '>i4', 2),
      ('hrs_qualind', '>i4', 1),
      ('hrs_linqualflgs', '>i4', 1),
      ('hrs_chqualflg', '>i2', 20),
      ('hrs_mnfrqual', '>i1', 64),
      ('hrs_filler2', '>i4', 4),
      ('hrs_calcof', '>i4', 60),
      ('hrs_scalcof', '>i4', 60),
      ('hrs_filler3', '>i4', 3),
      ('hrs_navstat', '>i4', 1),
      ('hrs_attangtime', '>i4', 1),
      ('hrs_rollang', '>i2', 1),
      ('hrs_pitchang', '>i2', 1),
      ('hrs_yawang', '>i2', 1),
      ('hrs_scalti', '>i2', 1),
      ('hrs_ang', '>i2', 168),
      ('hrs_pos', '>i4', 112),
      ('hrs_filler4', '>i4', 2),
      ('hrs_elem', '>i2', 1536),
      ('hrs_filler5', '>i4', 3),
      ('hrs_digbinvwbf', '>i2', 1),
      ('hrs_digitbwrd', '>i2', 1),
      ('hrs_aninvwbf', '>i4', 1),
      ('hrs_anwrd', '>i1', 16),
      ('hrs_filler6', '>i4', 11)])

HIRS_header_dtypes[4] = numpy.dtype([('hrs_h_siteid', '|S3', 1),
      ('hrs_h_blank', '|S1', 1),
      ('hrs_h_l1bversnb', '>i2', 1),
      ('hrs_h_l1bversyr', '>i2', 1),
      ('hrs_h_l1bversdy', '>i2', 1),
      ('hrs_h_reclg', '>i2', 1),
      ('hrs_h_blksz', '>i2', 1),
      ('hrs_h_hdrcnt', '>i2', 1),
      ('hrs_h_filler0', '>i2', 3),
      ('hrs_h_dataname', '|S42', 1),
      ('hrs_h_prblkid', '|S8', 1),
      ('hrs_h_satid', '>i2', 1),
      ('hrs_h_instid', '>i2', 1),
      ('hrs_h_datatyp', '>i2', 1),
      ('hrs_h_tipsrc', '>i2', 1),
      ('hrs_h_startdatajd', '>i4', 1),
      ('hrs_h_startdatayr', '>i2', 1),
      ('hrs_h_startdatady', '>i2', 1),
      ('hrs_h_startdatatime', '>i4', 1),
      ('hrs_h_enddatajd', '>i4', 1),
      ('hrs_h_enddatayr', '>i2', 1),
      ('hrs_h_enddatady', '>i2', 1),
      ('hrs_h_enddatatime', '>i4', 1),
      ('hrs_h_cpidsyr', '>i2', 1),
      ('hrs_h_cpidsdy', '>i2', 1),
      # CORRECTION! NWPSAF guide says there are 4 fields
      # here, but in reality there is 1 (see NOAA KLM
      # Users Guide – April 2014 Revision, page 8-117, PDF
      # page 428)
      ('hrs_h_fov1offset', '>i2', 1),
      ('hrs_h_instrtype', '|S6', 1),
      ('hrs_h_inststat1', '>i4', 1),
      ('hrs_h_filler1', '>i2', 1),
      ('hrs_h_statchrecnb', '>i2', 1),
      ('hrs_h_inststat2', '>i4', 1),
      ('hrs_h_scnlin', '>i2', 1),
      ('hrs_h_callocsclin', '>i2', 1),
      ('hrs_h_misscnlin', '>i2', 1),
      ('hrs_h_datagaps', '>i2', 1),
      ('hrs_h_okdatafr', '>i2', 1),
      ('hrs_h_pacsparityerr', '>i2', 1),
      ('hrs_h_auxsyncerrsum', '>i2', 1),
      ('hrs_h_timeseqerr', '>i2', 1),
      ('hrs_h_timeseqerrcode', '>i2', 1),
      ('hrs_h_socclockupind', '>i2', 1),
      ('hrs_h_locerrind', '>i2', 1),
      ('hrs_h_locerrcode', '>i2', 1),
      ('hrs_h_pacsstatfield', '>i2', 1),
      ('hrs_h_pacsdatasrc', '>i2', 1),
      ('hrs_h_filler2', '>i4', 1),
      ('hrs_h_spare1', '|S8', 1),
      ('hrs_h_spare2', '|S8', 1),
      ('hrs_h_filler3', '>i2', 5),
      ('hrs_h_autocalind', '>i2', 1),
      ('hrs_h_solarcalyr', '>i2', 1),
      ('hrs_h_solarcaldy', '>i2', 1),
      ('hrs_h_calinf', '>i4', 80),
      # CORRECTION! NWPSAF calls this hrs_h_filler5, which
      # already occurs a few lines down.
      ('hrs_h_filler4', '>i4', 2),
      ('hrs_h_tempradcnv', '>i4', 57),
      ('hrs_h_20solfiltirrad', '>i2', 1),
      ('hrs_h_20equifiltwidth', '>i2', 1),
      # CORRECTION! NWPSAF guide says there is 1 such
      # field, in reality there are 2.  See NOAA KLM
      # User's Guide, April 2014 Revision, Page 8-124 /
      # PDF Page 435
      ('hrs_h_filler5', '>i4', 2),
      ('hrs_h_modelid', '|S8', 1),
      ('hrs_h_nadloctol', '>i2', 1),
      ('hrs_h_locbit', '>i2', 1),
      ('hrs_h_filler6', '>i2', 1),
      ('hrs_h_rollerr', '>i2', 1),
      ('hrs_h_pitcherr', '>i2', 1),
      ('hrs_h_yawerr', '>i2', 1),
      ('hrs_h_epoyr', '>i2', 1),
      ('hrs_h_epody', '>i2', 1),
      ('hrs_h_epotime', '>i4', 1),
      ('hrs_h_smaxis', '>i4', 1),
      ('hrs_h_eccen', '>i4', 1),
      ('hrs_h_incli', '>i4', 1),
      ('hrs_h_argper', '>i4', 1),
      ('hrs_h_rascnod', '>i4', 1),
      ('hrs_h_manom', '>i4', 1),
      ('hrs_h_xpos', '>i4', 1),
      ('hrs_h_ypos', '>i4', 1),
      ('hrs_h_zpos', '>i4', 1),
      ('hrs_h_xvel', '>i4', 1),
      ('hrs_h_yvel', '>i4', 1),
      ('hrs_h_zvel', '>i4', 1),
      ('hrs_h_earthsun', '>i4', 1),
      ('hrs_h_filler7', '>i4', 4),
      ('hrs_h_rdtemp', '>i4', 6),
      ('hrs_h_bptemp', '>i4', 6),
      ('hrs_h_eltemp', '>i4', 6),
      ('hrs_h_pchtemp', '>i4', 6),
      ('hrs_h_fhcc', '>i4', 6),
      ('hrs_h_scnmtemp', '>i4', 6),
      ('hrs_h_fwmtemp', '>i4', 6),
      ('hrs_h_p5v', '>i4', 6),
      ('hrs_h_p10v', '>i4', 6),
      ('hrs_h_p75v', '>i4', 6),
      ('hrs_h_m75v', '>i4', 6),
      ('hrs_h_p15v', '>i4', 6),
      ('hrs_h_m15v', '>i4', 6),
      ('hrs_h_fwmcur', '>i4', 6),
      ('hrs_h_scmcur', '>i4', 6),
      ('hrs_h_pchcpow', '>i4', 6),
      ('hrs_h_iwtcnttmp', '>i4', 30),
      ('hrs_h_ictcnttmp', '>i4', 24),
      ('hrs_h_tttcnttmp', '>i4', 6),
      ('hrs_h_fwcnttmp', '>i4', 24),
      ('hrs_h_patchexpcnttmp', '>i4', 6),
      ('hrs_h_fsradcnttmp', '>i4', 6),
      ('hrs_h_scmircnttmp', '>i4', 6),
      ('hrs_h_pttcnttmp', '>i4', 6),
      ('hrs_h_sttcnttmp', '>i4', 6),
      ('hrs_h_bpcnttmp', '>i4', 6),
      ('hrs_h_electcnttmp', '>i4', 6),
      ('hrs_h_patchfcnttmp', '>i4', 6),
      ('hrs_h_scmotcnttmp', '>i4', 6),
      ('hrs_h_fwmcnttmp', '>i4', 6),
      ('hrs_h_chsgcnttmp', '>i4', 6),
      ('hrs_h_conversions', '>i4', 11),
      ('hrs_h_moonscnlin', '>i2', 1),
      ('hrs_h_moonthresh', '>i2', 1),
      ('hrs_h_avspcounts', '>i4', 20),
      ('hrs_h_startmanyr', '>i2', 1),
      ('hrs_h_startmandy', '>i2', 1),
      ('hrs_h_startmantime', '>i4', 1),
      ('hrs_h_endmanyr', '>i2', 1),
      ('hrs_h_endmandy', '>i2', 1),
      ('hrs_h_endmantime', '>i4', 1),
      ('hrs_h_deltav', '>i4', 3),
      ('hrs_h_mass', '>i4', 2),
      ('hrs_h_filler8', '>i2', 1302)])

HIRS_line_dtypes[4] = numpy.dtype([('hrs_scnlin', '>i2', 1),
      ('hrs_scnlinyr', '>i2', 1),
      ('hrs_scnlindy', '>i2', 1),
      ('hrs_clockdrift', '>i2', 1),
      ('hrs_scnlintime', '>i4', 1),
      ('hrs_scnlinf', '>i2', 1),
      ('hrs_mjfrcnt', '>i2', 1),
      ('hrs_scnpos', '>i2', 1),
      ('hrs_scntyp', '>i2', 1),
      ('hrs_filler1', '>i4', 2),
      ('hrs_qualind', '>i4', 1),
      ('hrs_linqualflgs', '>i4', 1),
      ('hrs_chqualflg', '>i2', 20),
      # Correction: Not |S1 but >i1
      #('hrs_mnfrqual', '|S1', 64),
      ('hrs_mnfrqual', '>i1', 64),
      ('hrs_filler2', '>i4', 4),
      ('hrs_calcof', '>i4', 60),
      ('hrs_scalcof', '>i4', 60),
      ('hrs_yawsteering', '>i2', 3),
      ('hrs_totattcorr', '>i2', 3),
      ('hrs_navstat', '>i4', 1),
      ('hrs_attangtime', '>i4', 1),
      ('hrs_rollang', '>i2', 1),
      ('hrs_pitchang', '>i2', 1),
      ('hrs_yawang', '>i2', 1),
      ('hrs_scalti', '>i2', 1),
      ('hrs_ang', '>i2', 168),
      ('hrs_pos', '>i4', 112),
      ('hrs_moonang', '>i2', 1),
      # CORRECTION: NWPSAF formatting guide calls this
      # filler4.  Should be filler3.
      ('hrs_filler3', '>i2', 3),
      ('hrs_elem', '>i2', 1536),
      # CORRECTION: NWPSAF formatting guide calls this
      # filler5.  Should be filler4.
      ('hrs_filler4', '>i4', 3),
      ('hrs_digitbupdatefg', '>i2', 1),
      ('hrs_digitbwrd', '>i2', 1),
      ('hrs_analogupdatefg', '>i4', 1),
      # CORRECTION: |S1 does not make sense, read as >i1 instead
      ('hrs_anwrd', '>i1', 16),
      ('hrs_filler5', '>i4', 11)])

HIRS_channel_order = {}
HIRS_channel_order[2] = [1, 17, 2, 3, 13, 4, 18, 11, 19, 7, 8, 20, 10, 14,
                         6, 5, 15, 12, 16, 9]
HIRS_channel_order[3] = [1, 17, 2, 3, 13, 4, 18, 11, 19, 7, 8, 20, 10, 14,
                         6, 5, 15, 12, 16, 9]
HIRS_channel_order[4] = HIRS_channel_order[3].copy()

# obtained manually from POD User's Guide
#
# Note that for HIRS/2, the file format changes.  Three formats are
# documented:
#
# - Before 1992-09-08
# - Between 1992-09-08 and 1994-11-15
# - Since 1994-11-15
#
# There are additional changes within these periods; for example, see POD
# Guide Appendices H and L.  See also the comment on sources above.
#
# There is also an undocumented change between 1994-12-31 and 1995-01-01.
# Prior to 1995-01-01, the record size was 4256.  Starting 1995-01-01, the
# record size was 4259.  This applies to all HIRS/2.  See also AAPP source
# code, AAPP/src/tools/bin/hirs2_class_to_aapp.F 

# Source: POD User's Guide, Table 2.0.4-1.
HIRS_header_dtypes[2] = {x: numpy.dtype([
      ("hrs_h_satid", ">i1", 1),
      ('hrs_h_datatyp', '>i1', 1),
      ('hrs_h_startdatadatetime', '|S6', 1), # read as bytes for now
      ('hrs_h_scnlin', '>i2', 1),
      ('hrs_h_enddatadatetime', '|S6', 1), # read as bytes for now  
      ('hrs_h_pbid', '|S7', 1),
      ('hrs_h_autocalind', '>i1', 1),
      ('hrs_h_datagaps', '>i2', 1),
      ('hrs_h_dacsqual', '>u2', 3),
      ('hrs_h_calid', '>u1', 2),
      ('hrs_h_dacsstat', '>i1', 1),
      ('hrs_h_attcorr', '>i1', 1),
      ('hrs_h_nadloctol', '>i1', 1),
      ('hrs_h_filler0', '>i1', 1),
      ('hrs_h_startdatayr', '>i2', 1),
      ('hrs_h_dataname', '|S42', 1), # EBCDIC!
      ('hrs_h_filler1', '>i1', x-82)])
        for x in (4253, 4256)}

# Source: POD User's Guide, Section 4-1
HIRS_line_dtypes[2] = {x: numpy.dtype([('hrs_scnlin', '>i2', 1),
      ('hrs_scnlintime', '|S6', 1), # read as bytes for now
      ('hrs_qualind', '>i4', 1),
      ('hrs_earthlocdelta', '>i4', 1),
      ('hrs_calcof', '>i4', 60*3),
      ('hrs_satloc', '>i2', 2),
      ('hrs_pos', '>i2', 112),
      ('hrs_elem', '>i2', 1408),
      ('hrs_mnfrqual', '>i1', 64),
      ('hrs_filler0', '>i1', x-3844),
    ])
        for x in (4253, 4256)}

# For HIRS/3, conversion of counts to brightness temperatures for Digital
# A Telemetry is not included in the files, but partially included in the
# KLM User's Guide.  However, this includes only coefficients for the
# Internal Warm Target (IWT) Platinum Resistance Thermometer (PRT) and the
# secondary telescope.

HIRS_count_to_temp = {}

for sat in {"TIROSN", "NOAA06", "NOAA07", "NOAA08", "NOAA09", "NOAA10", "NOAA11", "NOAA12", "NOAA14", "NOAA15", "NOAA16", "NOAA17"}:
    HIRS_count_to_temp[sat] = {}




##########
#
# NOAA-15
#
##########

# Table D.1-2.

HIRS_count_to_temp["NOAA15"]["iwtcnttmp"] = numpy.array([
    [301.42859, 6.5398670E-03, 8.9808960E-08, 4.7877130E-11, 1.3453590E-15],
    [301.44106, 6.5306330E-03, 8.7115040E-08, 4.7387900E-11, 1.4460280E-15],
    [301.43252, 6.5332780E-03, 8.2485710E-08, 4.7301670E-11, 1.6099050E-15],
    [301.39868, 6.5244370E-03, 8.0380230E-08, 4.7093000E-11, 1.6976440E-15]])

# Table D.1-16

HIRS_count_to_temp["NOAA15"]["sttcnttmp"] = numpy.array([
    260.29119, 1.693469E-02, -2.413170E-06, 4.019185E-10, 1.175655E-14])

# From CPIDS.  Read with read_cpids static method in HIRS3 class.

HIRS_count_to_temp["NOAA15"].update(
{'an_bptemp': array([  3.70377594e+02,  -3.21026192e+01,  -5.87326813e+00,
         6.28739882e+00,  -1.59110904e+00,   1.30866602e-01], dtype=float32),
 'an_eltemp': array([  2.60250488e+02,   1.47315903e+01,  -2.95523500e+00,
         9.69779789e-01,  -1.68153003e-01,   1.41547797e-02], dtype=float32),
 'an_fhcc': array([ 0. ,  0.1,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_fwmcur': array([ 0. ,  0.1,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_fwmtemp': array([  2.60262512e+02,   1.45856800e+01,  -2.77048612e+00,
         8.75885010e-01,  -1.47869006e-01,   1.25750499e-02], dtype=float32),
 'an_m15v': array([ 0., -4.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_m75v': array([ 0., -2.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_p10v': array([ 0. ,  2.5,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_p15v': array([ 0.,  4.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_p5v': array([ 0.        ,  1.33299994,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_p75v': array([ 0.,  2.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_pchcpow': array([ 0.        ,  0.        ,  0.00353787,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_pchtemp': array([  8.91846008e+01,   2.72437401e+01,   3.57790589e+00,
        -6.08588517e-01,   1.96566701e-01,  -1.35194296e-02], dtype=float32),
 'an_rdtemp': array([  1.48442703e+02,   2.30715199e+01,   1.26176703e+00,
         2.44658798e-01,  -2.82212403e-02,   3.67786898e-03], dtype=float32),
 'an_scmcur': array([ 0.        ,  0.40000001,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_scnmtemp': array([  3.62926605e+02,  -1.27381601e+01,  -2.45753994e+01,
         1.46874399e+01,  -3.36056089e+00,   2.71880895e-01], dtype=float32),
 'bpcnttmp': array([  2.60152802e+02,   1.80009194e-02,  -4.33221521e-06,
         1.68535097e-09,  -3.45452806e-13,   3.51498589e-17], dtype=float32),
 'ecdac': 128.0,
 'electcnttmp': array([  2.60212402e+02,   1.77217592e-02,  -3.88480112e-06,
         1.41277201e-09,  -2.76174103e-13,   2.88728088e-17], dtype=float32),
 'fmccc': 0.000122047,
 'fsradcnttmp': array([  2.17415100e+02,  -2.01619808e-02,   9.72466296e-07,
        -3.61915012e-11,   2.48054594e-15,  -6.65384085e-19], dtype=float32),
 'fwcnttmp': array([[  3.01380798e+02,   6.56869682e-03,   8.46994226e-08,
          3.52145500e-11,   1.40276798e-15,   6.76365786e-19],
       [  3.01428406e+02,   6.56327698e-03,   8.66939374e-08,
          3.59567411e-11,   1.19012904e-15,   6.00155982e-19],
       [  3.01398712e+02,   6.56972593e-03,   8.74942998e-08,
          3.62973610e-11,   1.15142302e-15,   5.87443819e-19],
       [  3.01429901e+02,   6.56631822e-03,   8.76398190e-08,
          3.65978393e-11,   1.18029001e-15,   5.73580323e-19]], dtype=float32),
 'fwmcnttmp': array([  2.60237213e+02,   1.76968705e-02,  -3.89413617e-06,
         1.42785705e-09,  -2.81459806e-13,   2.94360890e-17], dtype=float32),
 'fwthc': 0.000122047,
 'ictcnttmp': array([[  2.71597900e+02,   6.15457585e-03,   7.42702824e-08,
          4.62459203e-11,   1.39593200e-15,   1.44500202e-18],
       [  2.71544708e+02,   6.15644921e-03,   7.42007273e-08,
          4.64201108e-11,   1.36701504e-15,   1.43162796e-18],
       [  2.71588196e+02,   6.15480589e-03,   7.44043973e-08,
          4.62681803e-11,   1.39534596e-15,   1.43814097e-18],
       [  2.71556610e+02,   6.15352299e-03,   7.34121102e-08,
          4.62263491e-11,   1.40123803e-15,   1.43649798e-18]], dtype=float32),
 'iwtcnttmp': array([[  3.01413513e+02,   6.57448499e-03,   9.35651769e-08,
          3.83247288e-11,   1.02939596e-15,   5.15902725e-19],
       [  3.01425201e+02,   6.57052314e-03,   9.14469567e-08,
          3.63838404e-11,   1.08132904e-15,   5.94124078e-19],
       [  3.01416504e+02,   6.57331385e-03,   8.69622383e-08,
          3.62765408e-11,   1.23284597e-15,   5.94438510e-19],
       [  3.01382385e+02,   6.56480715e-03,   8.50495780e-08,
          3.59759514e-11,   1.30361801e-15,   5.99368506e-19],
#       [ -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45,
#         -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45]
        ], dtype=float32),
 'm15vdccc': 0.00488187,
 'm7.5vdccc': 0.0024409399,
 'p10vdccc': 0.0030511699,
 'p15vdccc': 0.00488187,
 'p5vdccc': array(0.0016272900393232703, dtype=float32),
 'p7.5vdccc': 0.0024409399,
 'patchexpcnttmp': array([  1.17874496e+02,   7.54731707e-03,   1.55108907e-07,
         2.97717406e-12,  -3.03142901e-16,   6.03511015e-21], dtype=float32),
 'patchfcnttmp': array([  1.77005096e+02,  -2.67255604e-02,   1.57804095e-06,
        -8.83819684e-11,   4.43753107e-15,  -9.92087275e-20], dtype=float32),
 'pcp': 5.2717999e-09,
 'pttcnttmp': array([  2.60183594e+02,   1.77792292e-02,  -3.99376586e-06,
         1.48313695e-09,  -2.94619391e-13,   3.05483491e-17], dtype=float32),
 'scmircnttmp': array([  2.60212494e+02,   1.77891403e-02,  -4.05322498e-06,
         1.53413604e-09,  -3.11199010e-13,   3.23719085e-17], dtype=float32),
 'scmotcnttmp': array([  2.60174805e+02,   1.80763397e-02,  -4.64382219e-06,
         1.98685890e-09,  -4.52424610e-13,   4.75908704e-17], dtype=float32),
 'smccc': 0.00048818701,
 'sttcnttmp': array([  2.60218597e+02,   1.76680405e-02,  -3.80478809e-06,
         1.35795697e-09,  -2.59891907e-13,   2.71867897e-17], dtype=float32),
 'tttcnttmp': array([ -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45,
        -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45], dtype=float32)})


###########
#
# NOAA-16
#
###########




# Table D.2-2.

HIRS_count_to_temp["NOAA16"]["iwtcnttmp"] = numpy.array([
    [301.45076, 6.530210E-03, 8.326151E-08, 4.724724E-11, 1.565263E-15],
    [301.39565, 6.527550E-03, 8.417738E-08, 4.727738E-11, 1.460746E-15],
    [301.40733, 6.528222E-03, 8.314237E-08, 4.721744E-11, 1.543985E-15],
    [301.40280, 6.525508E-03, 8.269671E-08, 4.707211E-11, 1.549894E-15]])

# Table D.2-5

HIRS_count_to_temp["NOAA16"]["sttcnttmp"] = numpy.array([
    260.42546, 1.659977E-02, -2.118035E-06, 3.040075E-10, 2.251628E-14])

HIRS_count_to_temp["NOAA16"].update({'an_bptemp': array([  3.76275208e+02,  -4.53256111e+01,   3.57388902e+00,
          3.15575099e+00,  -1.11919606e+00,   1.05117500e-01], dtype=float32),
 'an_eltemp': array([  2.60264008e+02,   1.44002800e+01,  -2.47772288e+00,
          7.13218272e-01,  -1.10677399e-01,   9.59365070e-03], dtype=float32),
 'an_fhcc': array([ 0. ,  0.1,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_fwmcur': array([ 0. ,  0.1,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_fwmtemp': array([  2.60263611e+02,   1.44389801e+01,  -2.52789688e+00,
          7.40266502e-01,  -1.17179900e-01,   1.01552904e-02], dtype=float32),
 'an_m15v': array([ 0., -4.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_m75v': array([ 0., -2.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_p10v': array([ 0. ,  2.5,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_p15v': array([ 0.,  4.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_p5v': array([ 0.        ,  1.33299994,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_p75v': array([ 0.,  2.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_pchcpow': array([ 0.        ,  0.        ,  0.00323232,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_pchtemp': array([  8.90127869e+01,   2.80284100e+01,   2.28540802e+00,
          2.18928307e-01,  -4.56520915e-03,   3.04379012e-03], dtype=float32),
 'an_rdtemp': array([  1.48821304e+02,   2.32149792e+01,   1.10821795e+00,
          3.61145407e-01,  -5.90152703e-02,   6.37528300e-03], dtype=float32),
 'an_scmcur': array([ 0.        ,  0.40000001,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_scnmtemp': array([  3.76967987e+02,  -4.77349586e+01,   6.32570124e+00,
          1.76793396e+00,  -7.99267113e-01,   7.76391104e-02], dtype=float32),
 'bpcnttmp': array([  2.60173615e+02,   1.80298798e-02,  -4.40669919e-06,
          1.74705395e-09,  -3.66525414e-13,   3.75688791e-17], dtype=float32),
 'ecdac': 126.9,
 'electcnttmp': array([  2.60201202e+02,   1.79508403e-02,  -4.34550793e-06,
          1.72305603e-09,  -3.62058501e-13,   3.72600297e-17], dtype=float32),
 'fmccc': 0.00012205,
 'fsradcnttmp': array([  2.17282700e+02,  -1.97234601e-02,   1.07960796e-06,
         -1.91398702e-10,  -3.21655701e-15,   7.78428572e-18], dtype=float32),
 'fwcnttemp': array([[  3.01411987e+02,   6.55641407e-03,   8.40527719e-08,
           3.51422190e-11,   1.40518001e-15,   6.02617620e-19],
        [  3.01394592e+02,   6.55640988e-03,   8.75943371e-08,
           3.59484005e-11,   1.14841405e-15,   6.01105585e-19],
        [  3.01366699e+02,   6.56030979e-03,   8.70886581e-08,
           3.62333497e-11,   1.14657101e-15,   5.86742421e-19],
        [  3.01416412e+02,   6.56819018e-03,   8.64483809e-08,
           3.63988908e-11,   1.22419999e-15,   5.80611307e-19]], dtype=float32),
 'fwmcnttmp': array([  2.60190704e+02,   1.79553293e-02,  -4.34574122e-06,
          1.72283898e-09,  -3.62677797e-13,   3.73873990e-17], dtype=float32),
 'fwthc': 0.00012205,
 'ictcnttmp': array([[  2.71599396e+02,   6.15675421e-03,   7.47755280e-08,
           4.73783096e-11,   1.26539099e-15,   1.37553601e-18],
        [  2.71528015e+02,   6.15314208e-03,   7.43782209e-08,
           4.73621906e-11,   1.25532803e-15,   1.37013101e-18],
        [  2.71559387e+02,   6.15429506e-03,   7.52066569e-08,
           4.72736399e-11,   1.31265405e-15,   1.37755599e-18],
        [  2.71542511e+02,   6.14904519e-03,   7.45702380e-08,
           4.70955913e-11,   1.32394098e-15,   1.38037998e-18]], dtype=float32),
 'iwtcnttmp': array([[  3.01435394e+02,   6.57160580e-03,   8.79144935e-08,
           3.58572096e-11,   1.17321496e-15,   6.13516603e-19],
        [  3.01379486e+02,   6.56759785e-03,   8.87803182e-08,
           3.62602691e-11,   1.07273705e-15,   5.93432606e-19],
        [  3.01391296e+02,   6.56732218e-03,   8.76279884e-08,
           3.64611084e-11,   1.16576096e-15,   5.79377102e-19],
        [  3.01416412e+02,   6.56819018e-03,   8.64483809e-08,
           3.63988908e-11,   1.22419999e-15,   5.80611307e-19],
#        [ -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45,
#          -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45]
          ], dtype=float32),
 'm15vdccc': 0.0048819999,
 'm7.5vdccc': 0.002441,
 'p10vdccc': 0.0030513001,
 'p15vdccc': 0.0048819999,
 'p5vdccc': array(0.0016273000510409474, dtype=float32),
 'p7.5vdccc': 0.002441,
 'patchexpcnttmp': array([  1.17874496e+02,   7.54731707e-03,   1.55108907e-07,
          2.97717406e-12,  -3.03142901e-16,  -6.03511015e-21], dtype=float32),
 'patchfcnttmp': array([  1.77002899e+02,  -2.67265700e-02,   1.58030196e-06,
         -8.75226627e-11,   4.28541793e-15,  -1.54504104e-19], dtype=float32),
 'pcp': 4.8149e-09,
 'pttcnttmp': array([  2.60193695e+02,   1.78072993e-02,  -4.06154913e-06,
          1.52738999e-09,  -3.06705209e-13,   3.17159907e-17], dtype=float32),
 'scmircnttmp': array([  2.60251587e+02,   1.77359991e-02,  -3.99525015e-06,
          1.50482704e-09,  -3.04373687e-13,   3.17892491e-17], dtype=float32),
 'scmotcnttmp': array([  2.60221497e+02,   1.77455600e-02,  -3.98245902e-06,
          1.48361001e-09,  -2.96058290e-13,   3.07978003e-17], dtype=float32),
 'smccc': 0.00048819999,
 'sttcnttmp': array([  2.60141907e+02,   1.83010492e-02,  -4.86018689e-06,
          2.03571604e-09,  -4.44539398e-13,   4.50989987e-17], dtype=float32),
 'tttcnttmp': array([ -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45,
         -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45], dtype=float32)})

# Table D.3-11

HIRS_count_to_temp["NOAA17"]["iwtcnttmp"] = numpy.array([
    [301.41859, 0.006539867, 8.909E-08, 4.78771E-11, 1.34536E-15],
    [301.43106, 0.006530633, 8.7115E-08, 4.73879E-11, 1.44603E-15],
    [301.42252, 0.006533278, 8.24857E-08, 4.73017E-11, 1.60991E-15],
    [301.38868, 0.006524437, 8.03802E-08, 4.7093E-11, 1.69764E-15]])

# Table D.3-12

HIRS_count_to_temp["NOAA17"]["sttcnttmp"] = numpy.array([
    260.29119, 0.01693469, -2.41317E-06, 4.01919E-10, 1.17566E-14])

HIRS_count_to_temp["NOAA17"].update({'an_bptemp': array([  3.92104492e+02,  -7.96604385e+01,   3.41925392e+01,
         -9.81301785e+00,   1.50782895e+00,  -9.88210589e-02], dtype=float32),
 'an_eltemp': array([  2.60244995e+02,   1.44812202e+01,  -2.60142994e+00,
          7.78941095e-01,  -1.25195503e-01,   1.07011301e-02], dtype=float32),
 'an_fhcc': array([ 0. ,  0.1,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_fwmcur': array([ 0. ,  0.1,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_fwmtemp': array([  2.60191315e+02,   1.45072002e+01,  -2.64123297e+00,
          8.01428199e-01,  -1.30442798e-01,   1.11418096e-02], dtype=float32),
 'an_m15v': array([ 0., -4.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_m75v': array([ 0., -2.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_p10v': array([ 0. ,  2.5,  0. ,  0. ,  0. ,  0. ], dtype=float32),
 'an_p15v': array([ 0.,  4.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_p5v': array([ 0.        ,  1.33299994,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_p75v': array([ 0.,  2.,  0.,  0.,  0.,  0.], dtype=float32),
 'an_pchcpow': array([ 0.        ,  0.        ,  0.00283849,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_pchtemp': array([  8.93951111e+01,   2.80723591e+01,   2.50468206e+00,
          7.13794008e-02,   3.12695988e-02,   8.50563229e-05], dtype=float32),
 'an_rdtemp': array([  1.49585098e+02,   2.34919109e+01,   6.93066120e-01,
          5.70401907e-01,  -1.04450203e-01,   9.96694714e-03], dtype=float32),
 'an_scmcur': array([ 0.        ,  0.40000001,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32),
 'an_scnmtemp': array([  3.91348297e+02,  -7.83955231e+01,   3.34030609e+01,
         -9.57179546e+00,   1.47124505e+00,  -9.63906273e-02], dtype=float32),
 'bpcnttmp': array([  2.60227997e+02,   1.76241901e-02,  -3.76684193e-06,
          1.34240696e-09,  -2.56063400e-13,   2.67096305e-17], dtype=float32),
 'ecdac': 126.9,
 'electcnttmp': array([  2.60253113e+02,   1.75747294e-02,  -3.71142301e-06,
          1.31308697e-09,  -2.49063113e-13,   2.60866295e-17], dtype=float32),
 'fmccc': 0.00012217541,
 'fsradcnttmp': array([  2.18759201e+02,  -2.02267095e-02,   9.72406156e-07,
         -3.22270093e-11,   2.66176800e-15,  -8.53018374e-19], dtype=float32),
 'fwcnttemp': array([[  3.01412201e+02,   6.56620087e-03,   8.79867983e-08,
           3.67505887e-11,   1.08780196e-15,   5.55655423e-19],
        [  3.01380005e+02,   6.56472798e-03,   8.76000925e-08,
           3.64091986e-11,   1.13992095e-15,   5.78498584e-19],
        [  3.01346588e+02,   6.55445317e-03,   8.79269919e-08,
           3.67157416e-11,   1.02790201e-15,   5.49458289e-19],
        [  3.01396606e+02,   6.55583292e-03,   8.79913600e-08,
           3.67267397e-11,   1.06731498e-15,   5.52815195e-19]], dtype=float32),
 'fwmcnttmp': array([  2.60160187e+02,   1.77523401e-02,  -3.96557198e-06,
          1.47228696e-09,  -2.93374293e-13,   3.06104704e-17], dtype=float32),
 'fwthc': 0.00012217541,
 'ictcnttmp': array([[  2.71561188e+02,   6.15443802e-03,   7.57553522e-08,
           4.77436597e-11,   1.19608495e-15,   1.34354304e-18],
        [  2.71530609e+02,   6.15363382e-03,   7.48070477e-08,
           4.83639309e-11,   1.23037602e-15,   1.29974704e-18],
        [  2.71500000e+02,   6.15507923e-03,   7.59992034e-08,
           4.81990316e-11,   1.21275403e-15,   1.32013300e-18],
        [  2.71488312e+02,   6.15241705e-03,   7.58234791e-08,
           4.91532093e-11,   1.16315496e-15,   1.24066204e-18]], dtype=float32),
 'iwtcnttmp': array([[  3.01373688e+02,   6.56326907e-03,   8.83572326e-08,
           3.71399092e-11,   1.05792202e-15,   5.34739696e-19],
        [  3.01369385e+02,   6.55610021e-03,   8.73095800e-08,
           3.68215909e-11,   1.09520100e-15,   5.49939191e-19],
        [  3.01360504e+02,   6.56493707e-03,   8.91571190e-08,
           3.74102901e-11,   9.69859713e-16,   5.09190878e-19],
        [  3.01449097e+02,   6.56154612e-03,   8.75372805e-08,
           3.60005914e-11,   1.10067495e-15,   6.01643718e-19],
#        [ -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45,
#          -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45]
        ], dtype=float32),
 'm15vdccc': 0.004887017,
 'm7.5vdccc': 0.0024435089,
 'p10vdccc': 0.003054386,
 'p15vdccc': 0.004887017,
 'p5vdccc': array(0.0016285990132018924, dtype=float32),
 'p7.5vdccc': 0.0024435089,
 'patchexpcnttmp': array([  1.18034798e+02,   7.56970514e-03,   1.50482705e-07,
          2.04621606e-12,   1.30451505e-16,   6.34576001e-20], dtype=float32),
 'patchfcnttmp': array([  1.77567795e+02,  -2.68405695e-02,   1.58019895e-06,
         -8.70805927e-11,   4.39441201e-15,  -7.01649169e-20], dtype=float32),
 'pcp': 4.2369721e-09,
 'pttcnttmp': array([  2.60243713e+02,   1.77103393e-02,  -3.93904111e-06,
          1.46475998e-09,  -2.92797091e-13,   3.06347200e-17], dtype=float32),
 'scmircnttmp': array([  2.60165710e+02,   1.75241493e-02,  -3.62645892e-06,
          1.26677502e-09,  -2.38263402e-13,   2.52010003e-17], dtype=float32),
 'scmotcnttmp': array([  2.60213806e+02,   1.76519994e-02,  -3.78480399e-06,
          1.34942002e-09,  -2.58299296e-13,   2.70292995e-17], dtype=float32),
 'smccc': 0.00048870168,
 'sttcnttmp': array([  2.60188599e+02,   1.77363493e-02,  -3.96122914e-06,
          1.47536205e-09,  -2.95038191e-13,   3.08171795e-17], dtype=float32),
 'tttcnttmp': array([ -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45,
         -1.40129846e-45,  -1.40129846e-45,  -1.40129846e-45], dtype=float32)}
)

# Remaining information for NOAA-15 onward is based on CPIDS information
# sent by Dejiang Han <dejiang.han@noaa.gov> to Gerrit Holl
# <g.holl@reading.ac.uk> on 2016-02-17 and 2016-02-25,,
# contained in document
# “Pre-K_Telemetry_Parameters.pdf”, sent by Deijiang Han
# <dejiang.han@noaa.gov> to Gerrit Holl <g.holl@reading.ac.uk> on
# 2016-02-25, and stored by Gerrit in the git repo '2015_FIDUCEO' at
# bitbucket, at WP4/HIRS/pre_launch_data/Pre-K_Telemetry_Parameters.pdf

# For HIRS/2, the POD guide does not appear to include any coefficients.
# Information is scarce.

# Source: Levin Gary, J Nelson, Frank W Porto, Data Extraction and
# calibration of TIROS-N/NOAA radiometers, NOAA Technical Memorandum NESS
# 107, Appendix B http://docs.lib.noaa.gov/rescue/TIROS/QC8795U4no107.pdf
#
# "This document contains Appendix B for TIROS-N, NOAA-9 and NOAA-10.
# Appendix B for other spacecraft will be issued separately."
#
# AAPP also contains coefficients in
# data/calibration/coef/hirs/calcoef.dat

# PDF page 77
#HIRS_count_to_temp["TIROSN"]["iwtcnttmp"] = "FIXME"

# NOAA F/9
# AAPP/src/calibration/libhirsc1/calcoef.dat : 913
# or OSO-EM/POES-0270
# or NESS 107, PDF page 92
HIRS_count_to_temp["NOAA09"]["iwtcnttmp"] = numpy.array([
 [28.2238+K,6.52238e-03,8.62819e-08,4.81437e-11,1.16950e-15],
 [28.2066+K,6.51928e-03,8.59682e-08,4.81011e-11,1.17422e-15],
 [28.2159+K,6.52446e-03,8.61933e-08,4.81459e-11,1.17357e-15],
 [28.2138+K,6.51965e-03,8.58931e-08,4.81048e-11,1.18056e-15]])

HIRS_count_to_temp["NOAA09"].update(
    {"ictcnttmp": numpy.array([
    [-1.65255+K, 6.04056E-03, 7.35066E-08, 7.55589E-11, 1.43631E-15],
    [-1.71422+K, 6.04355E-03, 7.36043E-08, 7.56091E-11, 1.43456E-15],
    [-1.68446+K, 6.03979E-03, 7.33514E-08, 7.33514E-11, 1.43901E-15],
    [-1.65907+K, 6.03885E-03, 7.34900E-08, 7.55388E-11, 1.42696E-15]]),
     "fwcnttmp": numpy.array([
    [28.1949+K, 6.51264E-03, 8.59918E-08, 4.80551E-11, 1.16901E-15],
    [28.2017+K, 6.52156E-03, 8.60778E-08, 4.81226E-11, 1.16877E-15],
    [28.1953+K, 6.51171E-03, 8.59595E-08, 4.80356E-11, 1.16953E-15],
    [28.2065+K, 6.52561E-03, 8.61091E-08, 4.81359E-11, 1.17885E-15]]),
     "patchexpcnttmp": numpy.array([ # NB: manually drawn circle final 6
    [-155.197+K, 7.55076E-03, 1.53359E-07, 2.98578E-12, -6.15827E-16]]),
    # TOTHIER
     "fsradcnttpm": numpy.array([
    [-55.2015+K, -0.0202043, 9.87098E-07, -4.85923E-11, 2.23676E-15]]),
     "scmircnttmp": numpy.array([
    [-12.9307+K, 0.0168306, -2.25548E-06, 3.32533E-10, 2.14454E-14]]),
     "pttcnttmp": numpy.array([
    [-12.8906+K, 0.0168187, -2.25216E-06, 3.31743E-10, 2.14941E-14]]),
     "sttcnttmp": numpy.array([
    [-12.9634+K, 0.0168226, -2.24673E-06, 3.29283E-10, 2.18498E-14]]),
     "bpcnttmp": numpy.array([
    [-12.9488+K, 0.0168261, -2.24877E-06, 3.29881E-10, 2.17861E-14]]),
     "electcnttmp": numpy.array([
    [-12.9606+K, 0.0168081, -2.23523E-06, 3.24923E-10, 2.23598E-14]]),
     "patchfcnttmp": numpy.array([
    [-95.7222+K, -0.0268139, 1.58297E-06, -8.87254E-11, 4.30667E-15]]),
     "scmotcnttmp": numpy.array([
    [-12.9457+K, 0.0168203, -2.23845E-06, 3.25463E-10, 2.22358E-14]]),
     "fwmcnttmp": numpy.array([
    [-12.9501+K, 0.0168513, -2.27076E-06, 3.36984E-10, 2.08690E-14]]),
     "chsgcnttmp": numpy.array([
    [13.8538+K, -0.0103199, 2.90704E-07, -3.00507E-10, 2.49086E-14]])})


# HIRS/2, NOAA G/10
# AAPP/src/calibration/libhirsc1/calcoef.dat : 1021
# or OSO-EM/POES-0263
# or NESS 107, PDF page 104
#
# NB: OSO-EM/POES-0263 has more detail than the other sources

HIRS_count_to_temp["NOAA10"]["iwtcnttmp"] = numpy.array([
 [28.2235+K,6.52161e-03,8.63442e-08,4.81141e-11,1.16671e-15],
 [28.2189+K,6.51610e-03,8.59659e-08,4.80415e-11,1.16682e-15],
 [28.2152+K,6.52074e-03,8.61310e-08,4.80831e-11,1.16844e-15],
 [28.2215+K,6.51970e-03,8.61605e-08,4.80733e-11,1.16825e-15]])
   
HIRS_count_to_temp["NOAA10"].update(
    {"ictcnttmp": numpy.array([
    [-1.164962+K, 6.03689E-03, 7.34912E-08, 7.54314E-11, 1.42586E-15],
    [-1.70384+K, 6.04024E-03, 7.37735E-08, 7.54898E-11, 1.43042E-15],
    [-1.70852+K, 6.03912E-03, 7.36121E-08, 7.54685E-11, 1.42635E-15],
    [-1.64355+K, 6.03858E-03, 7.34838E-08, 7.54628E-11, 1.42588E-15]]),
     "fwcnttmp": numpy.array([
    [28.2397+K, 6.52436E-03, 8.62374E-08, 4.81100E-11, 1.16956E-15],
    [28.2304+K, 6.51641E-03, 8.59091E-08, 4.80366E-11, 1.16251E-15],
    [28.2255+K, 6.51977E-03, 8.60432E-08, 4.80685E-11, 1.16687E-15],
    [28.2421+K, 6.51274E-03, 8.57481E-08, 4.80110E-11, 1.16202E-15]]),
     "patchexpcnttmp": numpy.array([
    [-155.4416+K, 7.536789E-03, 1.524822E-07, 1.4873E-12, -2.565222E-16]]),
     "fsradcnttpm": numpy.array([
    [-55.77669+K, -0.02017624, 9.817677E-07, -4.836361E-11, 2.599754E-15]]),
     "scmircnttmp": numpy.array([
    [-12.9608+K, 0.0168004, -2.23610E-06, 3.25812E-10, 1.21004E-14]]),
     "pttcnttmp": numpy.array([
    [-12.9762+K, 0.0168154, -2.23704E-06, 3.25789E-10, 2.22007E-14]]),
     "sttcnttmp": numpy.array([
    [-12.9445+K, 0.0167932, -2.22566E-06, 3.21847E-10, 2.25101E-14]]),
     "bpcnttmp": numpy.array([
    [-12.8863+K, 0.0167926, -2.23476E-06, 3.25616E-10, 2.21272E-14]]),
     "electcnttmp": numpy.array([
    [-12.9509+K, 0.0168066, -2.23946E-06, 3.26607E-10, 2.20469E-14]]),
     "patchfcnttmp": numpy.array([
    [-96.28774+K, -0.02670914, 1.580789E-06, -9.235212E-11, 4.581839E-15]]),
     "scmotcnttmp": numpy.array([
    [-12.9595+K, 0.0168041, -2.22971E-06, 3.22726E-10, 2.24640E-14]]),
     "fwmcnttmp": numpy.array([
    [-12.9371+K, 0.0168011, -2.22929E-06, 3.22858E-10, 2.25543E-14]]),
     "chsgcnttmp": numpy.array([
    [13.9371+K, -0.01029897, 3.599862E-07, -2.832785E-10, 1.102146E-14]])})


# HIRS/2I, FM - 1I, NOAA - H/11
#
# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal11.asc
# or AAPP/src/calibration/libhirsc1/calcoef.dat : 1129
# or OSO-PO/POES-0342
#
# NB: Some sources have coefficients to convert to °C, others to K!
# NOTE: OSO-PO/POES-0342 has offsets 0.01K smaller

HIRS_count_to_temp["NOAA11"]["iwtcnttmp"] = numpy.array([
 [28.2221+K,6.52057e-03,9.05197e-08,4.73741e-11,8.29062e-16],
 [28.2256+K,6.52283e-03,9.13565e-08,4.73871e-11,7.86019e-16],
 [28.2625+K,6.51819e-03,9.18444e-08,4.75139e-11,7.30508e-16],
 [28.2242+K,6.51875e-03,9.04524e-08,4.72894e-11,8.06020e-16]])
   
HIRS_count_to_temp["NOAA11"].update(
    {"ictcnttmp": numpy.array([
    [-1.66149+K, 6.05183E-03, 8.36527E-08, 7.39875E-11, 6.36552E-16],
    [-1.64027+K, 6.06415E-03, 6.71813E-08, 7.32452E-11, 2.09389E-16],
    [-1.62225+K, 6.05344E-03, 6.50493E-08, 7.36357E-11, 2.31484E-15],
    [-1.58883+K, 6.05475E-03, 6.51235E-08, 7.36992E-11, 2.31215E-15]]),
     "fwcnttmp": numpy.array([
    [28.2881+K, 6.52783E-03, 8.47513E-08, 4.67773E-11, 1.41152E-15],
    [28.2261+K, 6.53212E-03, 8.32388E-08, 4.72510E-11, 1.55037E-15],
    [28.1887+K, 6.52847E-03, 8.34361E-08, 4.76895E-11, 1.60687E-15],
    [28.2607+K, 6.51994E-03, 8.34024E-08, 4.69065E-11, 1.45467E-15]]),
     "patchexpcnttmp": numpy.array([
    [-155.6846+K, 7.54243E-03, 1.50306E-07, 1.64668E-12, 3.91776E-16]]),
     "fsradcnttpm": numpy.array([
    [-56.3904+K, -2.01207E-02, 9.86605E-07, -5.07092E-11, 2.75734E-15]]),
     "scmircnttmp": numpy.array([
    [-12.9637+K, 1.69314E-2, -2.40346E-06, 3.9885E-10, 1.23468E-14]]),
     "pttcnttmp": numpy.array([
    [-12.8350+K, 1.68966E-2, -2.38757E-06, 3.84036E-10, 1.27566E-14]]),
     "sttcnttmp": numpy.array([
    [-12.8867+K, 1.69249E-2, -2.40909E-06, 4.00712E-10, 1.22148E-14]]),
     "bpcnttmp": numpy.array([
    [-12.8543+K, 1.68916E-2, -2.38309E-06, 3.94308E-10, 1.27083E-14]]),
     "electcnttmp": numpy.array([
    [-12.9442+K, 1.69049E-2, -2.38823E-06, 3.94308E-10, 1.27083E-14]]),
     "patchfcnttmp": numpy.array([
    [-96.2308+K, -2.68462E-2, 1.56695E-06, 7.72502E-11, 7.54464E-15]]),
     "scmotcnttmp": numpy.array([
    [-12.9429+K, 1.69301E-2, -2.40365E-06, 3.97902E-10, 1.24430E-14]]),
     "fwmcnttmp": numpy.array([
    [-12.7006+K, 1.64457E-2, -1.95255E-06, 2.37550E-10, 3.17240E-14]]),
     "chsgcnttmp": numpy.array([
    [12.4889+K, -9.06432e-3, 5.06557E-07, -5.30145E-10, 2.65042-14]])})


# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal12.asc
# or AAPP/src/calibration/libhirsc1/calcoef.dat : 1237
# or OSO-PO/POES-0271
#
# NB: Some sources have coefficients to convert to °C, others to K!
# NOTE: OSO-PO/POES-0443 has offsets 0.01K smaller
#
# HIRS/2, NG, NOAA - D/12

# NOTE: OSO-PO/POES-0443 has offsets 0.01K smaller!
HIRS_count_to_temp["NOAA12"]["iwtcnttmp"] = numpy.array([
    [28.2330+K, 6.52454e-03,8.63834e-08,4.81705e-11,1.17918e-15],
    [28.2349+K, 6.51937e-03,8.61601e-08,4.81257e-11,1.17221e-15],
    [28.2492+K, 6.51150e-03,8.58417e-08,4.80590e-11,1.17105e-15],
    [28.2377+K, 6.52702e-03,8.63606e-08,4.81834e-11,1.17766e-15]])
    
HIRS_count_to_temp["NOAA12"].update(
    {"ictcnttmp": numpy.array([
    [-1.70036+K, 6.04583E-03, 7.36419E-08, 7.56354E-11, 1.44062E-15],
    [-1.65101+K, 6.03811E-03, 7.36193E-08, 7.55440E-11, 1.43640E-15],
    [-1.66014+K, 6.04015E-03, 7.34807E-08, 7.55699E-11, 1.43751E-15],
    [-1.65136+K, 6.03752E-03, 7.33257E-08, 7.55369E-11, 1.43161-15]]),
     "fwcnttmp": numpy.array([
    [28.2221+K, 6.51373E-03, 8.59968E-08, 4.80660E-11, 1.17528E-15],
    [28.2112+K, 6.53128E-03, 8.65921E-08, 4.82197E-11, 1.17912E-15],
    [28.2249+K, 6.52089E-03, 8.60196E-08, 4.81186E-11, 1.16907E-15],
    [28.2249+K, 6.51519E-03, 8.58945E-08, 4.80783E-11, 1.16956E-15]]),
     "patchexpcnttmp": numpy.array([
    [-155.700+K, 7.53317E-03, 1.51966E-07, 3.11118E-12, 8.02310E-18]]),
     "fsradcnttpm": numpy.array([
    [-56.1307+K, -0.020177, 9.91298E-07, -4.71603E-11, 1.92522E-15]]),
     "scmircnttmp": numpy.array([
    [-12.8971+K, 0.0168158, -2.23437E-06, 3.23752E-10, 2.24851E-14]]),
     "pttcnttmp": numpy.array([
    [-12.9639+K, 0.016810, -2.23795E-06, 3.25980E-10, 2.22135E-14]]),
     "sttcnttmp": numpy.array([
    [-12.9134+K, 0.0168019, -2.22561E-06, 3.21094E-10, 2.27971E-14]]),
     "bpcnttmp": numpy.array([
    [-12.9655+K, 0.016840, -2.25183E-06, 3.30385E-10, 2.17492E-14]]),
     "electcnttmp": numpy.array([
    [-12.9659+K, 0.0168519, -2.27114E-06, 3.37073E-10, 2.08839E-14]]),
     "patchfcnttmp": numpy.array([
    [-96.3341+K, -0.0267854, 1.58865E-06, -8.89173E-11, 4.19066E-15]]),
     "scmotcnttmp": numpy.array([
    [-12.9534+K, 0.016803, -2.23045E-06, 3.23111E-10, 2.25228E-14]]),
     "fwmcnttmp": numpy.array([
    [-12.9575+K, 0.0168222, -2.24124E-06, 3.26210E-10, 2.22081E-14]]),
     "chsgcnttmp": numpy.array([
    [13.7213+K, -0.010311, 2.89418E-07, -3.00307E-10, 2.56626-14]])})

# http://www.sat.dundee.ac.uk/noaa14.html
# or AAPP/src/calibration/libhirsc1/calcoef.dat : 1345
# or OSO-PO/POES-0443 (FM - 3I; HIRS/2I) Table 4.1-1 "Digital A Telemetry
# Conversion (Fourth-Order Polynomial), contained in
# “Pre-K_Telemetry_Parameters.pdf”
#
# NB: Some sources have coefficients to convert to °C, others to K!
 
# HIRS/2I, FM - 3I, NOAA - J/14,     

HIRS_count_to_temp["NOAA14"].update(
    {"iwtcnttmp": # NOTE: OSO-PO/POES-0443 has offsets 0.01K smaller!
    numpy.array([
    [28.23795+K, 6.52106e-03,8.27531e-08,4.65675e-11,1.45893E-15],
    [28.22735+K, 6.53105e-03,8.36999e-08,4.66502e-11,1.44768E-15],
    [28.24698+K, 6.52318e-03,8.26203e-08,4.65039e-11,1.51146E-15],
    [28.21675+K, 6.52655e-03,8.36699e-08,4.67894e-11,1.41383E-15]]),
     "ictcnttmp": numpy.array([
    [-1.639719+K, 6.058305e-03, 7.30042e-08, 7.190141E-11, 1.532254E-15],
    [-1.676158+K, 6.062627e-03, 7.194456e-08, 7.19288E-11, 1.569662E-15],
    [-1.664393+K, 6.060146E-03, 7.218341E-08, 7.198368E-11, 1.542339E-15],
    [-1.650414+K, 6.055869E-03, 7.271501E-08, 7.208245E-11, 1.495386E-15]]),
     "fwcnttmp": numpy.array([
    [28.25454+K, 6.517852E-03, 8.383772E-08, 4.662422E-11, 1.390598E-15],
    [28.22579+K, 6.582672E-03, 8.484221E-08, 4.675707E-11, 1.387996E-15],
    [28.24453+K, 6.521369E-03, 8.331517E-08, 4.648439E-11, 1.432773E-15],
    [28.19867+K, 6.524292E-03, 8.346979E-08, 4.643709E-11, 1.446272E-15]]),
     "patchexpcnttmp": numpy.array([
    [-155.2574+K, 7.541707E-03, 1.439548E-07, 3.092415E-12, 9.992008E-16]]),
     "fsradcnttpm": numpy.array([
    [-56.16665+K, -2.009704E-02, 9.841351E-07, -4.649764E-11, 1.512245E-15]]),
     "scmircnttmp": numpy.array([
    [-12.77747+K, 1.65525E-02, -2.113233E-06, 3.081038E-10, 2.166063E-14]]),
     "pttcnttmp": numpy.array([
    [-13.73046+K, 1.655457E-02, -2.125267E-06, 3.13368E-10, 2.082416E-14]]),
     "sttcnttmp": numpy.array([
    [-12.71208+K, 1.65439E-02, -2.109315E-06, 3.052881E-10, 2.203706E-14]]),
     "bpcnttmp": numpy.array([
    [-13.78767+K, 1.656044E-02, -2.106339E-06, 3.029825E-10, 2.238227E-14]]),
     "electcnttmp": numpy.array([
    [-12.70893+K, 1.654305E-02, -2.115256E-06, 3.09039E-10, 2.149756E-14]]),
     "patchfcnttmp": numpy.array([
    [-95.78484+K, -2.67893E-02, 1.563783E-06, -8.730941E-11, 5.548513E-15]]),
     "scmotcnttmp": numpy.array([
    [-12.69014+K, 1.648452E-02, -2.055571E-06, 2.87316E-10, 2.403221E-14]]),
     "fwmcnttmp": numpy.array([
    [-12.74053+K, 1.652814E-02, -2.106452E-06, 3.071005E-10, 2.150916E-14]]),
     "chsgcnttmp": numpy.array([
    [13.67123+K, -1.033193E-02, 3.098585E-07, -2.888921E-10, 2.03276E-14]])})


# Warning: Did not find any source for NOAA8, NOAA7, NOAA6, TIROS-N.
for sat in "NOAA08 NOAA07 NOAA06 TIROSN".split():
    for fld in HIRS_count_to_temp["NOAA09"].keys():
        HIRS_count_to_temp[sat][fld] = (
            HIRS_count_to_temp["NOAA09"][fld])

# Fill what's missing with dummies
dummy = numpy.ma.array([numpy.ma.masked])
for sat in {"TIROSN", "NOAA06", "NOAA07", "NOAA08", "NOAA09", "NOAA10", "NOAA11",
            "NOAA12", "NOAA14", "NOAA15", "NOAA16", "NOAA17"}:
    # first one is filter wheel housing
    for field in {"fwcnttmp", "patchexpcnttmp", "fsradcnttmp",
                  "scmircnttmp", "pttcnttmp", "sttcnttmp", "bpcnttmp",
                  "electcnttmp", "patchfcnttmp", "scmotcnttmp",
                  "fwmcnttmp", "chsgcnttmp", "iwtcnttmp"}:
        if not field in HIRS_count_to_temp[sat]:
            # When it's (n, 6) for HIRS/4, it's (n, 5) for HIRS/3 and
            # HIRS/2
            HIRS_count_to_temp[sat][field] = numpy.tile(dummy,
                    (HIRS_header_dtypes[4]["hrs_h_"+field].shape[0]//6, 5))
    if not "ictcnttmp" in HIRS_count_to_temp[sat]:
        # but this one is (4, 5) on HIRS/2 and HIRS/3
        HIRS_count_to_temp[sat]["ictcnttmp"] = numpy.tile(dummy, (4, 5))

# For HIRS/2, central wavenumbers and coefficients for BT conversion are
# not included in the headers.  Include them here.  Taken from Nick
# Bearsons HIRStoHDF code at
# https://svn.ssec.wisc.edu/repos/HIRStoHDF/trunk/src/HTH_HIRS2_inc.py
#
# FIXME: verify agreement with NOAA POD Guide

HIRS_coeffs = {
    2: {
        14:
       { 1 : ( 668.90 ,  0.002  , 0.99998 ),
         2 : ( 679.36 ,  -0.000 , 0.99997 ),
         3 : ( 689.63 ,  0.011  , 0.99994 ),
         4 : ( 703.56 ,  0.001  , 0.99994 ),
         5 : ( 714.50 ,  -0.014 , 0.99997 ),
         6 : ( 732.28 ,  0.026  , 0.99989 ),
         7 : ( 749.64 ,  0.019  , 0.99991 ),
         8 : ( 898.67 ,  0.067  , 0.99977 ),
         9 : ( 1028.31,  0.050  , 0.99980 ),
        10 : ( 796.04 ,  0.021  , 0.99990 ),
        11 : ( 1360.95,  0.073  , 0.99971 ),
        12 : ( 1481.00,  0.284  , 0.99931 ),
        13 : ( 2191.32,  0.021  , 0.99996 ),
        14 : ( 2207.36,  0.020  , 0.99997 ),
        15 : ( 2236.39,  0.024  , 0.99998 ),
        16 : ( 2268.12,  0.018  , 0.99996 ),
        17 : ( 2420.24,  0.026  , 0.99992 ),
        18 : ( 2512.21,  0.042  , 0.99993 ),
        19 : ( 2647.91,  0.313  , 0.99946 )},

        13:
       { 1 :  ( 668.81 , -0.077,  1.00019 ),
         2  : ( 679.59 , 0.020 ,  0.99992 ),
         3  : ( 690.18 , 0.016 ,  0.99993 ),
         4  : ( 703.02 , 0.018 ,  0.99991 ),
         5  : ( 715.96 , 0.040 ,  0.99986 ),
         6  : ( 732.98 , 0.028 ,  0.99987 ),
         7  : ( 749.34 , -0.034,  1.00000 ),
         8  : ( 902.39 , 0.544 ,  0.99916 ),
         9  : ( 1028.77, 0.062 ,  0.99979 ),
         10 : ( 792.82 , -0.005,  0.99994 ),
         11 : ( 1359.95, 0.090 ,  0.99972 ),
         12 : ( 1479.90, 0.292 ,  0.99931 ),
         13 : ( 2189.06, 0.022 ,  0.99997 ),
         14 : ( 2212.55, 0.021 ,  0.99997 ),
         15 : ( 2231.68, 0.029 ,  0.99993 ),
         16 : ( 2267.04, 0.022 ,  0.99999 ),
         17 : ( 2418.31, 0.025 ,  0.99992 ),
         18 : ( 2516.80, 0.058 ,  0.99970 ),
         19 : ( 2653.33, 0.264 ,  0.99927 )},

        12:
       { 1  : ( 667.58 ,    0.007 ,   0.99996),
         2  : ( 680.18 ,    0.007 ,   0.99995),
         3  : ( 690.01 ,    0.019 ,   0.99989),
         4  : ( 704.22 ,    0.026 ,   0.99988),
         5  : ( 716.32 ,    0.021 ,   0.99990),
         6  : ( 732.81 ,    0.140 ,   0.99964),
         7  : ( 751.92 ,    0.058 ,   0.99982),
         8  : ( 900.45 ,    0.358 ,   0.99940),
         9  : ( 1026.66,    0.181 ,   0.99985),                       
         10 : ( 1223.44,    0.377 ,   0.99975),
         11 : ( 1368.68,    0.175 ,   0.99992),
         12 : ( 1478.59,    0.265 ,   0.99863),
         13 : ( 2190.37,    0.078 ,   1.00042),
         14 : ( 2210.51,    0.017 ,   0.99995),
         15 : ( 2236.62,    -0.023,   0.99950),
         16 : ( 2267.62,    0.021 ,   0.99995),
         17 : ( 2361.64,    0.022 ,   0.99997),
         18 : ( 2514.68,    0.058 ,   0.99992),
         19 : ( 2653.48,    0.344 ,   0.99950)},

        11:
       { 1  : ( 668.99 ,    0.007 ,   0.99996),
         2  : ( 678.89 ,    0.010 ,   0.99994),
         3  : ( 689.70 ,    0.007 ,   0.99992),
         4  : ( 703.25 ,    -0.003,   0.99995),
         5  : ( 716.83 ,    0.014 ,   0.99991),
         6  : ( 732.11 ,    0.019 ,   0.99991),
         7  : ( 749.48 ,    0.032 ,   0.99988),
         8  : ( 900.51 ,    0.077 ,   0.99988),                       
         9  : ( 1031.19,    0.068 ,   0.99975),
         10 : ( 795.69 ,    -0.001,   0.99994),
         11 : ( 1361.10,    0.074 ,   0.99972),
         12 : ( 1479.86,    0.288 ,   0.99994),
         13 : ( 2189.94,    0.022 ,   0.99994),
         14 : ( 2209.66,    0.018 ,   0.99995),
         15 : ( 2239.26,    0.020 ,   0.99995),
         16 : ( 2267.80,    0.015 ,   0.99993),
         17 : ( 2416.32,    0.024 ,   0.99991),
         18 : ( 2511.83,    0.045 ,   0.99990),
         19 : ( 2664.07,    0.325 ,   0.99949)},
         
         10:   
       { 1  : ( 667.70 ,    0.033 ,   0.99989),
         2  : ( 680.23 ,    0.018 ,   0.99992),
         3  : ( 691.15 ,    -0.006,   0.99994),
         4  : ( 704.33 ,    -0.002,   0.99994),
         5  : ( 716.30 ,    -0.064,   1.00007),
         6  : ( 733.13 ,    0.065 ,   0.99980),
         7  : ( 750.72 ,    0.073 ,   0.99979),
         8  : ( 899.50 ,    0.218 ,   0.99957),
         9  : ( 1029.01,    0.195 ,   0.99987),                       
         10 : ( 1224.07,    0.327 ,   0.99965),
         11 : ( 1363.32,    0.046 ,   0.99963),
         12 : ( 1489.42,    0.645 ,   1.00064),
         13 : ( 2191.38,    0.072 ,   1.00036),
         14 : ( 2208.74,    0.079 ,   1.00045),
         15 : ( 2237.49,    -0.026,   0.99947),
         16 : ( 2269.09,    0.041 ,   1.00019),
         17 : ( 2360.00,    0.040 ,   1.00019),
         18 : ( 2514.58,    0.098 ,   1.00025),
         19 : ( 2665.38,    0.462 ,   1.00067) },
         
        9:
       { 1  : ( 667.67 ,    0.034 ,   0.99989),
         2  : ( 679.84 ,    0.024 ,   0.99991),
         3  : ( 691.46 ,    0.092 ,   0.99975),
         4  : ( 703.37 ,    0.002 ,   0.99993),
         5  : ( 717.16 ,    0.013 ,   0.99991),
         6  : ( 732.64 ,    -0.023,   0.99997),
         7  : ( 749.48 ,    -0.006,   0.99995),
         8  : ( 898.53 ,    0.126 ,   0.99969),                       
         9  : ( 1031.61,    0.187 ,   0.99987),
         10 : ( 1224.74,    0.569 ,   1.00010),
         11 : ( 1365.12,    0.033 ,   0.99961),
         12 : ( 1483.24,    0.353 ,   0.99911),
         13 : ( 2189.97,    -0.001,   0.99980),
         14 : ( 2209.18,    0.007 ,   0.99984),
         15 : ( 2243.14,    0.027 ,   1.00003),
         16 : ( 2276.46,    0.099 ,   1.00038),
         17 : ( 2359.05,    0.004 ,   0.99977),
         18 : ( 2518.14,    0.084 ,   1.00012),
         19 : ( 2667.80,    0.448 ,   1.00040)},
         
        
        8:
       { 1  : ( 667.41 ,    0.099 ,   0.99971),
         2  : ( 679.45 ,    0.147 ,   0.99962),
         3  : ( 690.90 ,    0.143 ,   0.99964),
         4  : ( 702.97 ,    0.010 ,   0.99991),
         5  : ( 717.56 ,    -0.001,   0.99994),
         6  : ( 732.97 ,    0.193 ,   0.99955),
         7  : ( 747.90 ,    -0.104,   1.00013),
         8  : ( 901.08 ,    0.429 ,   0.99931),
         9  : ( 1027.11,    0.140 ,   0.99984),                       
         10 : ( 1224.05,    0.450 ,   0.99988),
         11 : ( 1366.17,    0.108 ,   0.99978),
         12 : ( 1486.92,    0.530 ,   1.00008),
         13 : ( 2189.28,    0.051 ,   1.00022),
         14 : ( 2211.71,    0.063 ,   1.00029),
         15 : ( 2238.06,    0.015 ,   0.99992),
         16 : ( 2271.43,    0.029 ,   1.00004),
         17 : ( 2357.11,    0.018 ,   0.99993),
         18 : ( 2515.53,    0.080 ,   1.00007),
         19 : ( 2661.85,    0.489 ,   1.00061)},
        
        7:
       { 1  : ( 667.92 ,    -0.010,   1.00001),
         2  : ( 679.21 ,    0.100 ,   0.99973),
         3  : ( 691.56 ,    -0.018,   0.99997),
         4  : ( 704.63 ,    0.026 ,   0.99989),
         5  : ( 717.05 ,    -0.009,   0.99995),
         6  : ( 733.20 ,    -0.081,   1.00008),
         7  : ( 749.20 ,    -0.054,   1.00003),                       
         8  : ( 898.94 ,    0.332 ,   0.99942),
         9  : ( 1027.38,    0.205 ,   0.99987),
         10 : ( 1224.89,    0.469 ,   0.99994),
         11 : ( 1363.85,    0.114 ,   0.99983),
         12 : ( 1489.06,    0.573 ,   1.00028),
         13 : ( 2183.05,    0.047 ,   1.00013),
         14 : ( 2208.28,    0.060 ,   1.00028),
         15 : ( 2239.84,    0.021 ,   0.99993),
         16 : ( 2271.33,    0.032 ,   1.00008),
         17 : ( 2357.55,    0.032 ,   1.00005),
         18 : ( 2512.83,    0.026 ,   0.99968),
         19 : ( 2663.79,    0.637 ,   1.00171)},
         
        6:
       { 1  : ( 668.02 ,    0.025 ,   0.99992),
         2  : ( 679.94 ,    0.151 ,   0.99900),
         3  : ( 690.44 ,    0.115 ,   0.99970),
         4  : ( 704.69 ,    0.041 ,   0.99984),
         5  : ( 717.43 ,    -0.035,   1.00000),
         6  : ( 732.47 ,    0.066 ,   0.99980),
         7  : ( 748.48 ,    -0.101,   1.00012),
         8  : ( 900.64 ,    0.185 ,   0.99961),
         9  : ( 1029.48,    0.268 ,   0.99990),                       
         10 : ( 1217.77,    -0.205,   0.99877),
         11 : ( 1368.05,    0.073 ,   0.99966),
         12 : ( 1485.76,    0.597 ,   1.00026),
         13 : ( 2190.60,    0.022 ,   1.00000),
         14 : ( 2210.09,    -0.001,   0.99978),
         15 : ( 2237.76,    0.029 ,   0.99999),
         16 : ( 2269.43,    0.015 ,   0.99991),
         17 : ( 2360.42,    0.011 ,   0.99984),
         18 : ( 2514.97,    0.051 ,   0.99985),
         19 : ( 2654.58,    0.482 ,   1.00042)}
     } }

HIRS_ids = {
    2: {
        4: 7,
        6: 8,
        7: 9,
        8: 10,
        1: 11, # Warning: identical to TIROS-N
        5: 12,
        2: 6, # Warning: identical to NOAA-13
        3: 14
    },
    3: {
        2: 16,
        4: 15,
        6: 17},
    4: {
        7: 18,
        8: 19,
        11: "A", # MetOp-A
        12: "B", # MetOp-B
        13: "C"}
}

HIRS_names = {
    2: {
        4: "NOAA07",
        6: "NOAA08",
        7: "NOAA09",
        8: "NOAA10",
        1: "NOAA11",
        5: "NOAA12",
        #2: "NOAA13", # Does not exist
        2: "NOAA06",
        3: "NOAA14"
    },
    # NOAA KLM User's Guide, page 8-100
    3: {
        2: "NOAA16",
        4: "NOAA15",
        6: "NOAA17"},
    # NOAA KLM User's Guide, page 8-116
    4: {
        7: "NOAA18",
        8: "NOAA19",
        11: "METOPA", # MetOp-A
        12: "METOPB", # MetOp-B
        13: "METOPC"} # Does not exist
}

# from
# http://www.nsof.class.noaa.gov/release/data_available/tovs_atovs/index.htm#hirs2
HIRS_periods = dict(
    tirosn =    (datetime.datetime(1978, 10, 21),
                 datetime.datetime(1981, 2, 27)),
    noaa06 =     (datetime.datetime(1979, 6, 30),
                 datetime.datetime(1986, 11, 17)),
    noaa07 =     (datetime.datetime(1981, 6, 24),
                 datetime.datetime(1985, 2, 18)),
    noaa08 =     (datetime.datetime(1983, 4, 25),
                 datetime.datetime(1985, 10, 14)),
    noaa09 =     (datetime.datetime(1984, 12, 13),
                 datetime.datetime(1988, 11, 7)),
    noaa10 =    (datetime.datetime(1986, 11, 25),
                 datetime.datetime(1991, 9, 16)),
    noaa11 =    (datetime.datetime(1988, 9, 24),
                 datetime.datetime(2000, 4, 26)),
    noaa12 =    (datetime.datetime(1991, 5, 14),
                 datetime.datetime(1998, 12, 14)),
    noaa14 =    (datetime.datetime(1994, 12, 30),
                 datetime.datetime(2006, 10, 10)),
    noaa15 =    (datetime.datetime(1998, 10, 26),
                 datetime.datetime.now()),
    noaa16 =    (datetime.datetime(2001, 2, 26),
                 datetime.datetime(2014, 6, 5)),
    noaa17 =    (datetime.datetime(2002, 8, 24),
                 datetime.datetime(2013, 4, 9)),
    noaa18 =    (datetime.datetime(2005, 6, 5),
                 datetime.datetime.now()),
    noaa19 =    (datetime.datetime(2009, 4, 21),
                 datetime.datetime.now()),
    metopa =    (datetime.datetime(2007, 5, 21),
                 datetime.datetime.now()),
    metopb =    (datetime.datetime(2013, 1, 15),
                 datetime.datetime.now()))


# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal11.asc

# http://noaasis.noaa.gov/NOAASIS/pubs/CAL/cal12.asc

# http://www.sat.dundee.ac.uk/noaa14.html

# NOAA POLAR SATELLITE CALIBRATION: A SYSTEM DESCRIPTION,
#   NOAA Technical Report NESDIS 77
# http://docs.lib.noaa.gov/rescue/TIROS/QC8795U47no77.pdf

# DATA EXTRACTION AND CALIBRATION OF TIROS-N/NOAA RADIOMETER
#   NOAA Technical Memorandum NESS 107
# http://docs.lib.noaa.gov/rescue/TIROS/QC8795U4no107.pdf
