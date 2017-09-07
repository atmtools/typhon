# -*- coding: utf-8 -*-

"""Handling of environment variables.

The environment variables are represented as package-wide constants.

===================== ======================================
``ARTS_BUILD_PATH``   ARTS ``build/`` directory.
``ARTS_DATA_PATH``    Additional search path for data files.
``ARTS_INCLUDE_PATH`` Search path for include files.
===================== ======================================

The environment variables can be set in the in the configuration file
(:mod:`typhon.config`) under the section ``environment``::

    [environment]
    ARTS_BUILD_PATH: /path/to/arts/build/

Note:
    If the environment variable is also set explicitly,
    the value set in the configuration file is ignored.

"""
import os

import typhon.config


# Environment variables that are recognized and handled by ARTS.
ENVIRONMENT = [
    'ARTS_BUILD_PATH',
    'ARTS_DATA_PATH',
    'ARTS_INCLUDE_PATH',
]

for key in ENVIRONMENT:
    # Check if the given key is set as process environment variable.
    if key in os.environ:
        # If so, set the same-named constant and continue with the next key.
        globals()[key] = os.environ.get(key)
        continue

    # If the key was not found in the process environment,
    # try to get the value from the TYPHONRC configuration file.
    try:
        value = typhon.config.get_config(key, section='environment')
    except:
        value = None
    finally:
        # Set the variable name in the global naming space.
        globals()[key] = value

__all__ = ENVIRONMENT
