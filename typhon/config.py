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
    config = None

    def init(self):
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        p = pathlib.Path(os.getenv("TYPHONRC", "~/.typhonrc")).expanduser()
        config.read(str(p))
        self.config = config
    
    def __call__(self, arg):
        if self.config is None:
            self.init()
        return self.config.get("main", arg)
config = _Configurator()
config.init()
conf = config.config

def get_config(arg):
    """Get value for configuration variable.

    Arguments:

        arg [str]: Name of configuration variable

    Returns:

        Value for configuration variable
    """
    confer = _Configurator()

    return confer(arg)
