"""Datasets for models
"""

import datetime

import numpy

from . import dataset

class ERAInterim(dataset.NetCDFDataset, dataset.MultiFileDataset):
    # example path:
    # /badc/ecmwf-era-interim/data/gg/as/2015/01/01/ggas201501011200.nc
    name = "era_interim"
    subdir = "data/{AA:s}/{B:s}{C:s}/{year:04d}/{month:02d}/{day:02d}"
    re = (r"(?P<AA>[a-z]{2})(?P<B>[a-z])(?P<C>[a-z])"
          r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
          r"(?P<hour>\d{2})(?P<minute>\d{2})\.nc")
    start_date = datetime.datetime(1979, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2016, 1, 1, 0, 0, 0)
    granule_duration = datetime.timedelta(hours=6)
    def _read(self, f, *args, **kwargs):
        # may need to convert, but I currently need gg/as which on CEMS is
        # already in converted format, so will assume NetCDF
        M = super()._read(f, *args,
                          pseudo_fields={"time": self._get_time_from_ds},
                          **kwargs)
        return M

    @staticmethod
    def _get_time_from_ds(ds):
        epoch = datetime.datetime.strptime(ds["t"].time_origin,
                                           "%d-%b-%Y:%H:%M:%S")
        return numpy.datetime64(epoch) + ds["t"][:].astype("timedelta64[D]")
