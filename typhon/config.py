#!/usr/bin/python

"""Handle specific configuration settings, such as data locations.

Configuration is handled with a configuration file with a
:mod:`configparser` syntax.

The location of the configuration file is determined by the environment
variable TYPHONRC.  If this is not set, it will use ~/.typhonrc.
"""

import pathlib
import os
import configparser


class _Configurator(object):
    def __init__(self):
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        p = pathlib.Path(os.getenv("TYPHONRC", "~/.typhonrc")).expanduser()
        config.read(str(p))
        self.config = config

    def __call__(self, arg, section='main'):
        return self.config.get(section, arg)
config = _Configurator()
conf = config.config


def get_config(arg, section='main'):
    """Get value for configuration variable.

    Arguments:

        arg [str]: Name of configuration variable

    Returns:

        Value for configuration variable
    """
    confer = _Configurator()

    return confer(arg, section=section)
