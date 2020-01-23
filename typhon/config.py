#!/usr/bin/env python3

"""Handle specific configuration settings, such as data locations.

Configuration is handled with a configuration file with a
:mod:`configparser` syntax.

The location of the configuration file is determined by the environment
variable TYPHONRC.  If this is not set, it will use ~/.typhonrc.
"""

import pathlib
import os
from configparser import (ConfigParser, ExtendedInterpolation)


__all__ = [
    'conf',
]


conf = ConfigParser(interpolation=ExtendedInterpolation())
conf.optionxform = str
p = pathlib.Path(os.getenv("TYPHONRC", "~/.typhonrc")).expanduser()
conf.read(str(p))
