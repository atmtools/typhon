# -*- coding: utf-8 -*-

"""This module contains functions for python's datetime/timedelta objects
"""
import functools
import time

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
    "Timer",
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


class Timer:
    """Provide a simple time profiling utility.

    Timer class adapted from blog entry [0].

    [0] https://www.huyng.com/posts/python-performance-analysis

    Parameters:
        verbose: Print results after stopping the timer.
        info (str): Allows to add additional information to output.
            The given string is printed before the measured time.
            If `None`, default information is added depending on the use case.
        timefmt (str): Format string to control the output of the measured time.
            The names 'minutes' and 'seconds' can be used.

    Examples:
        Timer in with statement:

        >>> import time
        >>> with Timer():
        ...     time.sleep(1)
        elapsed time: 0m 1.001s

        Timer as object:

        >>> import time
        >>> t = Timer().start()
        >>> time.sleep(1)
        >>> t.stop()
        elapsed time: 0m 1.001s

        As function decorator:

        >>> @Timer()
        ... def own_function(s):
        ...     import time
        ...     time.sleep(s)
        >>> own_function(1)
        own_function: 0m 1.001s

    """
    def __init__(self, info=None, timefmt='{minutes:d}m {seconds:.3f}s',
                 verbose=True):
        self.verbose = verbose
        self.timefmt = timefmt
        self.info = info

        # Define variables to store start and end time, assigned during runtime.
        self.starttime = None
        self.endtime = None

    def __call__(self, func):
        """Allows to use a Timer object as a decorator."""
        # When no additional information is given, add the function name is
        # Timer is used as decorator.
        if self.info is None:
            self.info = func.__name__

        @functools.wraps(func)  # Preserve the original signature and docstring.
        def wrapper(*args, **kwargs):
            with self:
                # Call the original function in a Timer context.
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start timer."""
        self.starttime = time.time()
        return self

    def stop(self):
        """Stop timer."""
        self.endtime = time.time()
        secs = self.endtime - self.starttime

        # Build a string containing the measured time for output.
        timestr = self.timefmt.format(
            minutes=int(secs // 60),
            seconds=secs % 60,
        )

        # If no additional information is specified add default information
        # to make the output more readable.
        if self.info is None:
            self.info = 'elapsed time'

        if self.verbose:
            # Connect additional information and measured time for output.
            print('{info}: {time}'.format(info=self.info, time=timestr))