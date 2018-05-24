# -*- coding: utf-8 -*-

"""This module contains functions for python's datetime/timedelta objects
"""

from datetime import datetime, timedelta
from numbers import Number

import netCDF4
import numpy as np
import pandas as pd

__all__ = [
    "date2num",
    "num2date",
    "set_time_resolution",
    "to_datetime",
    "to_timedelta",
]


def set_time_resolution(datetime_obj, resolution):
    """Set the resolution of a python datetime object.

    Args:
        datetime_obj: A python datetime object.
        resolution: A string indicating the required resolution.

    Returns:
        A datetime object truncated to *resolution*.

    Examples:

    .. code-block:: python

        from typhon.utils.time import set_time_resolution, to_datetime

        dt = to_datetime("2017-12-04 12:00:00")
        # datetime.datetime(2017, 12, 4, 12, 0)

        new_dt = set_time_resolution(dt, "day")
        # datetime.datetime(2017, 12, 4, 0, 0)

        new_dt = set_time_resolution(dt, "month")
        # datetime.datetime(2017, 12, 1, 0, 0)
    """
    if resolution == "year":
        return set_time_resolution(datetime_obj, "day").replace(month=1, day=1)
    elif resolution == "month":
        return set_time_resolution(datetime_obj, "day").replace(day=1)
    elif resolution == "day":
        return datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    elif resolution == "hour":
        return datetime_obj.replace(minute=0, second=0, microsecond=0)
    elif resolution == "minute":
        return datetime_obj.replace(second=0, microsecond=0)
    elif resolution == "second":
        return datetime_obj.replace(microsecond=0)
    elif resolution == "millisecond":
        return datetime_obj.replace(
            microsecond=int(datetime_obj.microsecond / 1000) * 1000
        )
    else:
        raise ValueError("Cannot set resolution to '%s'!" % resolution)


def to_datetime(obj):
    """Convert an object to a python datetime object.

    Args:
        obj: Can be a string with time information, a numpy.datetime64 or a
            pandas.Timestamp object.

    Returns:
        A python datetime object.

    Examples:

    .. code-block:: python

        dt = to_datetime("2017-12-04 12:00:00")
        # dt is datetime.datetime(2017, 12, 4, 12, 0)
    """

    if isinstance(obj, datetime):
        return obj
    else:
        return pd.to_datetime(obj).to_pydatetime()


def to_timedelta(obj, numbers_as=None):
    """Convert an object to a python datetime object.

    Args:
        obj: Can be a string with time information, a number, a
            numpy.timedelta64 or a pandas.Timedelta object.
        numbers_as: A string that indicates how numbers should be
            interpreted. Allowed values are *weeks*, *days*, *hours*,
            *minutes*, *seconds*, *milliseconds* and *microseconds.

    Returns:
        A python datetime object.

    Examples:

    .. code-block:: python

        # A timedelta object with 200 seconds
        t = to_timedelta("200 seconds")

        # A timedelta object with 24 days
        t = to_timedelta(24, numbers_as="days")
    """

    if numbers_as is None:
        numbers_as = "seconds"

    if isinstance(obj, timedelta):
        return obj
    elif isinstance(obj, Number):
        return timedelta(**{numbers_as: int(obj)})
    else:
        return pd.to_timedelta(obj).to_pytimedelta()


unit_mapper = {
    "nanoseconds": "ns",
    "microseconds": "us",
    "milliseconds": "ms",
    "seconds": "s",
    "hours": "h",
    "minutes": "m",
    "days": "D",
}


class InvalidUnitString(Exception):
    def __init__(self, *args, **kwargs):
        super(InvalidUnitString, self).__init__(*args, **kwargs)


def date2num(dates, units, calendar=None):
    """Convert an array of integer into datetime objects.

    This function optimizes the date2num function of python-netCDF4 if the
    standard calendar is used.

    Args:
        dates: Either an array of numpy.datetime64 objects (if standard
            gregorian calendar is used), otherwise an array of python
            datetime objects.
        units: A string with the format "{unit} since {epoch}",
            e.g. "seconds since 1970-01-01T00:00:00".
        calendar: (optional) Standard is gregorian. If others are used,
            netCDF4.num2date will be called.

    Returns:
        An array of integers.
    """
    if calendar is None:
        calendar = "gregorian"
    else:
        calendar = calendar.lower()

    if calendar != "gregorian":
        return netCDF4.date2num(dates, units, calendar)

    try:
        unit, epoch = units.split(" since ")
    except ValueError:
        raise InvalidUnitString("Could not convert to numeric values!")

    converted_data = \
        dates.astype("M8[%s]" % unit_mapper[unit]).astype("int")

    # numpy.datetime64 cannot read certain time formats while pandas can.
    epoch = pd.Timestamp(epoch).to_datetime64()

    if epoch != np.datetime64("1970-01-01"):
        converted_data -= np.datetime64("1970-01-01") - epoch
    return converted_data


def num2date(times, units, calendar=None):
    """Convert an array of integers into datetime objects.

    This function optimizes the num2date function of python-netCDF4 if the
    standard calendar is used.

    Args:
        times: An array of integers representing timestamps.
        units: A string with the format "{unit} since {epoch}",
            e.g. "seconds since 1970-01-01T00:00:00".
        calendar: (optional) Standard is gregorian. If others are used,
            netCDF4.num2date will be called.

    Returns:
        Either an array of numpy.datetime64 objects (if standard gregorian
        calendar is used), otherwise an array of python datetime objects.
    """
    try:
        unit, epoch = units.split(" since ")
    except ValueError:
        raise InvalidUnitString("Could not convert to datetimes!")

    if calendar is None:
        calendar = "gregorian"
    else:
        calendar = calendar.lower()

    if calendar != "gregorian":
        return netCDF4.num2date(times, units, calendar).astype(
            "M8[%s]" % unit_mapper[unit])

    # Numpy uses the epoch 1970-01-01 natively.
    converted_data = times.astype("M8[%s]" % unit_mapper[unit])

    # numpy.datetime64 cannot read certain time formats while pandas can.
    epoch = pd.Timestamp(epoch).to_datetime64()

    # Maybe there is another epoch used?
    if epoch != np.datetime64("1970-01-01"):
        converted_data -= np.datetime64("1970-01-01") - epoch
    return converted_data

