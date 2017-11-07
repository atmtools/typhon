#!/usr/bin/env python

"""Contains metaclasses for typhon development and others

This module contains metaclasses that are used within typhon, but that may
also be useful for others.
"""

# Any commits made to this module between 2015-05-01 and 2017-03-01
# by Gerrit Holl are developed for the EC project “Fidelity and
# Uncertainty in Climate Data Records from Earth Observations (FIDUCEO)”.
# Grant agreement: 638822
# 
# All those contributions are dual-licensed under the MIT license for use
# in typhon, and the GNU General Public License version 3.

import abc


# Use a metaclass to inherit docstrings
#
# This code, or an earlier version, was posted by user 'unubtu' on Stack
# Overflow at 2011-11-11 at http://stackoverflow.com/a/8101118/974555
# and subsequently edited.  It is licensed under CC-BY-SA 3.0.  This
# notice may not be removed.  An earlier version is authored by
# Paul McGuire <ptmcg@austin.rr.com> and was posted at
# https://groups.google.com/forum/#!msg/comp.lang.python/HkB1uhDcvdk/lWzWtPy09yYJ


class DocStringInheritor(type):
    """Automatically inherit docstrings.

    If overriding a method for a class with a DocStringInheritor
    metaclass, the subclass can omit the docstring.  The docstring will be
    inherited from the superclass.

    This is inspired by the default behaviour in MatlabⓇ.  The
    implementation is inspired by Paul Mcguire, see
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    """
    def __new__(mcs, name, bases, clsdict):
        if not('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (
                    mro_cls for base in bases for mro_cls in base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    clsdict['__doc__'] = doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls
                                for base in bases
                                for mro_cls in base.mro()
                                if hasattr(mro_cls, attr)):
                    doc = getattr(getattr(mro_cls, attr), '__doc__')
                    if doc:
                        attribute.__doc__ = doc
                        # added by Gerrit
                        attribute.__doc__ += ("\n\nDocstring inherited "
                                              "from {:s}".format(
                                                  mro_cls.__name__))
                        break
        # GH: replaced `type` by super() to not break multiple inheritance
        return super().__new__(mcs, name, bases, clsdict)


class AbstractDocStringInheritor(DocStringInheritor, abc.ABCMeta):
    """Automatically inherit docstrings for abstract classes.

    For details, see DocStringInheritor.
    """
    pass
