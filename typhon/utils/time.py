# -*- coding: utf-8 -*-

"""This module contains functions for python's datetime/timedelta objects
"""

from datetime import datetime, timedelta
from numbers import Number

import pandas as pd


__all__ = [
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
            obj: Can be a string with time information, a numpy.timedelta64 or
                a pandas.Timedelta object.
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
