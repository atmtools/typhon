# -*- coding: utf-8 -*-


from .version import __version__

try:
    __ATRASU_SETUP__
except:
    __ATRASU_SETUP__ = False

if not __ATRASU_SETUP__:
	from . import spectra
	from . import setup_atmosphere
	from . import const
	from . import nonltecalc
	from . import mathmatics
	from . import rtc


